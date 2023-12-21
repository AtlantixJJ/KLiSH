"""KMeans initialized SVM loss training with cosine similarity-based weight pruning."""
import sys
sys.path.insert(0, ".")
import argparse, os
import numpy as np
import torch
from torchvision import utils as vutils
from tqdm import tqdm

from lib.visualizer import segviz_torch, VideoRecorder, plot_dict
from lib.misc import set_cuda_devices
from models.helper import build_generator
from lib.misc import DictRecorder
from lib.op import *
from lib.cluster import *


EPS = 1e-9


def loss_func_batched_l2svc(feat, w, svm_label, coef=10.0):
  """Calculate the L2 SVC loss with multiple gradient accumulation (to save GPU memory).
  Args:
    feat: (num_sample, num_feature) tensor, assumed to be on GPU.
    w: (num_class, num_feature) tensor, assumed to be on the same device as feat.
    svm_label: (num_sample, num_class) tensor, the SVM label for each sample (-1 or 1).
    coef: The loss weight of the slack term.
  Return:
    The value of loss function per class.
  """
  acc_svm_loss, N = 0, 8
  B = feat.shape[0] // N
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    svm_loss = torch.square((1 - svm_label[st:ed] * seg).clamp(min=0)).mean(0) * coef / N
    acc_svm_loss += svm_loss.detach()
    svm_loss.mean().backward() # accumulate grad
  return acc_svm_loss


def loss_func_batched_l1svc(feat, w, svm_label, coef=10.0):
  """Calculate the L1 SVC loss with multiple gradient accumulation (to save GPU memory).
  Args:
    feat: (num_sample, num_feature) tensor, assumed to be on GPU.
    w: (num_class, num_feature) tensor, assumed to be on the same device as feat.
    svm_label: (num_sample, num_class) tensor, the SVM label for each sample (-1 or 1).
    coef: The loss weight of the slack term.
  Return:
    The value of loss function per class.
  """
  acc_svm_loss, N = 0, 8
  B = feat.shape[0] // N
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    svm_loss = (1 - svm_label[st:ed] * seg).clamp_(min=0).mean(0) * coef / N
    acc_svm_loss += svm_loss.detach()
    svm_loss.mean().backward() # accumulate grad
  return acc_svm_loss


def train_svm_l2reg_l2svc(mfeat, w, optim, msvm_label, coef=10.0, ret_class=False):
  """Train L2 SVM with multiple GPUs.
  Args:
    mfeat: A list of (num_sample, num_feature) tensors, assumed to be on different GPUs.
    w: (num_class, num_feature) tensor.
    optim: The optimizer.
    msvm_label: A list of (num_sample, num_class) tensors, assumed to be on different GPUs and aligned with mfeat.
    coef: The loss weight of the slack term.
  Return:
    The value of loss function per class.
  """
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(loss_func_batched_l2svc,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"), msvm_label[d_id], coef))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    svm_loss += threads[d_id].res.cuda() / len(mfeat)
  reg_loss = 0.5 * torch.square(w).sum(1)
  reg_loss.mean().backward()
  if ret_class:
    return reg_loss + svm_loss
  return (reg_loss + svm_loss).mean()


def train_svm_l2reg_l1svc(mfeat, w, optim, msvm_label, coef=10.0, ret_class=False):
  """Train L2 SVM with multiple GPUs.
  Args:
    mfeat: A list of (num_sample, num_feature) tensors, assumed to be on different GPUs.
    w: (num_class, num_feature) tensor.
    optim: The optimizer.
    msvm_label: A list of (num_sample, num_class) tensors, assumed to be on different GPUs and aligned with mfeat.
    coef: The loss weight of the slack term.
  Return:
    The value of loss function per class.
  """
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(loss_func_batched_l1svc,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"), msvm_label[d_id], coef))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    svm_loss += threads[d_id].res.cuda() / len(mfeat)
  reg_loss = 0.5 * torch.square(w).sum(1)
  reg_loss.mean().backward()
  if ret_class:
    return reg_loss + svm_loss
  return (reg_loss + svm_loss).mean()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--name", default="svm_alter")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq")
  parser.add_argument("--layer-idx", default="auto", type=str)
  parser.add_argument("--N", default=256, type=int)
  # training
  parser.add_argument("--latent-type", default="trunc-wp", type=str)
  parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
  parser.add_argument("--svm-iter", default=100, type=int)
  parser.add_argument("--svm-type", default="l1", type=str)
  parser.add_argument("--resample-interval", default=200, type=int)
  parser.add_argument("--seed", default=1997, type=int)
  args = parser.parse_args()
  n_gpu = set_cuda_devices(args.gpu_id)
  devices = list(range(n_gpu))

  B = args.N // n_gpu # The number of samples put in each GPU
  S, minsize, N_viz = 256, 64, min(B, 16)
  name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_{args.N}_{args.svm_type}svm_{args.seed}"
  svm_func = train_svm_l2reg_l1svc if args.svm_type == "l1" else \
    train_svm_l2reg_l2svc

  K = 100
  seed = 1991
  truncation = 0.5

  kmeans_wpath = f"{args.expr}/kmeans/{args.G_name}_l{args.latent_type}_i{args.layer_idx}_K{K}_N256_S256_{seed}_arccos.pth"
  prefix = f"{args.expr}/{args.name}/{name}"
  os.system(f"mkdir {args.expr}/{args.name}")

  print(f"=> Preparing {args.N} samples in {S} resolution ...")
  Gs = [build_generator(args.G_name, randomize_noise=True,
          truncation_psi=truncation).net.to(f"cuda:{d}") for d in devices]
  if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
  with torch.no_grad():
    wps = sample_latent(Gs[0], args.N, args.latent_type)
    wps = [wps[B*i:B*(i+1)].to(f"cuda:{i}") for i in range(n_gpu)]
  mimage, mfeat = multigpu_sample_layer_feature(
    Gs=Gs, S=S, N=None, wps=wps,
    layer_idx=args.layer_idx, latent_type=args.latent_type)
  B, H, W, C = mfeat[0].shape
  for i in range(n_gpu):
    mfeat[i] = mfeat[i].view(-1, C)
  print(f"=> GPU Feature: {mfeat[0].shape} ({B}, {H}, {W}, {C})")

  w = torch.load(kmeans_wpath).cuda().requires_grad_(True)
  mlabel, msvm_label = multigpu_get_svm_label(mfeat, w)
  print(f"=> Init from {kmeans_wpath}: {w.shape[0]} classes.")

  viz_feat = mfeat[0].view(B, H, W, C)[:N_viz].detach().view(-1, C)
  mimage = [bu(x[:N_viz], (H, W)) for x in mimage]
  labels = torch.matmul(viz_feat, w.permute(1, 0)).argmax(1)
  pred_viz = segviz_torch(labels.view(N_viz, H, W).cpu())
  disp = []
  for i in range(mimage[0].shape[0]):
    disp.extend([mimage[0][i], pred_viz[i]])
  vutils.save_image(torch.stack(disp), f"{prefix}_kmeans.png",
    nrow=8, padding=2, pad_value=0)
  del viz_feat, pred_viz, disp

  optim = torch.optim.LBFGS([w])
  record = DictRecorder()
  video_recorder = VideoRecorder(prefix, N_viz)
  data_count, count = args.N, 0
  tree, cur_label = {}, w.shape[0]
  idx2label = [i for i in range(w.shape[0])]
  for i in tqdm(range(200)):
    def closure():
      return svm_func(mfeat, w, optim, msvm_label)

    for j in range(args.svm_iter):
      w_prev = w.clone().detach()
      total_loss = optim.step(closure)
      w_diff = float((w - w_prev).abs().max())
      record.add("Total Loss", min(total_loss, 1))
      record.add("Max L2 Norm of W for Class", w.norm(p=2, dim=1).max())
      record.add("K", w.shape[0])
      record.add("Data count", data_count)
      record.add("W/difference", min(w_diff, 0.1))
      if w_diff < 1e-4:
        break
    
    total_loss_c = svm_func(mfeat, w, optim, msvm_label, ret_class=True)
    p = int(total_loss_c.argmax())
    record.add("Converged Total Loss", total_loss_c.mean())
    record.add("Converged Max Class Loss", total_loss_c[p])

    # forge test label
    for j in range(len(mlabel)):
      delta = (msvm_label[j][:, p:p+1] > 0).float() * 2 # (L, 1)
      msvm_label[j][:, :p] += delta
      msvm_label[j][:, p+1:] += delta
      assert msvm_label[j].max() < 1.1

    test_w = w.clone().detach().requires_grad_(True)
    test_optim = torch.optim.LBFGS([test_w])
    def test_closure():
      return svm_func(mfeat, test_w, test_optim, msvm_label)
    for j in range(args.svm_iter):
      w_prev = test_w.clone().detach()
      test_optim.step(test_closure)
      w_diff = float((test_w - w_prev).abs().max())
      if w_diff < 1e-4:
        break
    new_loss_c = svm_func(
      mfeat, test_w, test_optim, msvm_label, ret_class=True)
    loss_dec = total_loss_c + total_loss_c[p] - new_loss_c
    loss_dec[p] = 0
    q = int(loss_dec.argmax())
    record.add("Converged Max Loss Decrease", loss_dec[q])
    # save merge tree
    tree[count] = {
      "W": torch2numpy(w),
      "Total Objective": torch2numpy(total_loss_c.mean()),
      "Merging Metric": torch2numpy(loss_dec[q]),
      "Deleting Metric": torch2numpy(total_loss_c[p])}
    # visualize
    video_recorder(None, w, None, mimage[0], mfeat[0],
      mlabel[0].view(B, H, W), None, None, None, q, p)

    tree[count]["merge"] = [idx2label[q], idx2label[p], cur_label]
    cur_label += 1
    torch.save(tree, f"{prefix}_tree.pth")
    idx2label[q] = count # p -> q, q -> cur_label
    del idx2label[p]
    count += 1

    # merge weights
    new_w = w.clone().detach().requires_grad_(False)
    new_w[q].copy_(test_w[q])
    new_w = delete_index(new_w, p, 0)
    new_w = new_w.clone().detach().requires_grad_(True)
    new_optim = torch.optim.LBFGS([new_w])
    del optim, w
    optim, w = new_optim, new_w
    torch.cuda.empty_cache()
    # merge label
    for j in range(len(mlabel)):
      # recover original label
      msvm_label[j].fill_(-1).scatter_(1, mlabel[j].unsqueeze(1), 1)
      # merge
      delta = (msvm_label[j][:, p] > 0).float() * 2
      msvm_label[j][:, q] += delta
      t = msvm_label[j].cpu().detach()
      t = delete_index(t, p, 1)
      msvm_label[j] = None
      torch.cuda.empty_cache()
      msvm_label[j] = t.to(f"cuda:{j}")
      mlabel[j] = msvm_label[j].argmax(1)
    plot_dict(record, f"{prefix}_record.png")
    if w.shape[0] <= 1:
      break

  torch.save(tree, f"{prefix}_tree.pth")
  plot_dict(record, f"{prefix}_record.png")
  video_recorder.clean_sync()