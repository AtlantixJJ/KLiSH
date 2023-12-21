"""KMeans initialized SVM loss training with cosine similarity-based weight pruning."""
import sys
sys.path.insert(0, ".")
import argparse, os, math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm

from lib.visualizer import segviz_torch, VideoRecorder, VideoWriter, plot_dict
from lib.misc import set_cuda_devices, imwrite
from lib.op import *
from models.helper import build_generator
from lib.misc import DictRecorder
from lib.cluster import *


EPS = 1e-9


def calc_metric_worker(feat, svm_label, w):
  """The worker for calculating merging metrics in a thread.
  Args:
    feat: (L, C)
    w: (K, C)
  """
  dm, dm_c, N = 0, 0, 8
  B = feat.shape[0] // N
  with torch.no_grad():
    for i in range(N):
      st, ed = B * i, B * (i + 1)
      seg = torch.matmul(feat[st:ed], w.permute(1, 0))
      # P(p's margin | p)
      ld_area = ((seg >= -1) & (seg <= 1))
      p_area = (svm_label[st:ed] > 0)
      pnum = p_area.sum(0).float()
      dm += (p_area & ld_area).sum(0).float()
      dm_c += pnum # (K, 1)
      # distance to margin
      #dm += torch.square((1 - svm_label[st:ed] * seg).clamp(min=0)).sum(0)
      #dm_c += pnum
  return dm, dm_c


def calculate_metric(mfeat, msvm_label, w):
  """Calculate merging metrics with multiple threads.
  """
  dm, dm_c, threads = 0, 0, []
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(calc_metric_worker,
      mfeat[d_id], msvm_label[d_id],
      w.clone().to(f"cuda:{d_id}")))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    dm += threads[d_id].res[0].cuda()
    dm_c += threads[d_id].res[1].cuda()
  dm /= dm_c.clamp(min=EPS)
  return dm


def loss_func_batched_l1svc(feat, w, svm_label):
  acc_svm_loss, N = 0, 8
  B = feat.shape[0] // N
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    svm_loss = 100 * (1 - svm_label[st:ed] * seg).clamp_(min=0).mean() / N
    acc_svm_loss += svm_loss.detach()
    svm_loss.backward() # accumulate grad
  return acc_svm_loss


def train_svm_l2reg_l1svc(mfeat, w, optim, msvm_label, mlabel=None):
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(loss_func_batched_l1svc,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"), msvm_label[d_id]))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    svm_loss += threads[d_id].res.cuda() / len(mfeat)
  reg_loss = 0.5 * torch.square(w).sum(1).mean()
  reg_loss.backward()
  total_loss = reg_loss + svm_loss
  return total_loss


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--name", default="svm_lowdensity")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq")
  parser.add_argument("--layer-idx", default="auto", type=str)
  parser.add_argument("--N", default=256, type=int)
  # training
  parser.add_argument("--latent-type", default="trunc-wp", type=str)
  parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
  parser.add_argument("--svm-iter", default=100, type=int)
  parser.add_argument("--resample-interval", default=200, type=int)
  parser.add_argument("--seed", default=1997, type=int)
  args = parser.parse_args()
  n_gpu = set_cuda_devices(args.gpu_id)
  devices = list(range(n_gpu))

  B = args.N // n_gpu
  S, minsize, N_viz = 256, 64, min(B, 16)
  name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_{args.N}_{args.seed}"

  if "ada_cat" in args.name or "ada_dog" in args.name or "ada_wild" in args.name:
    K = 50
  else:
    K = 100
  seed = 1993
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
      return train_svm_l2reg_l1svc(mfeat, w, optim, msvm_label, mlabel)
    
    for j in range(args.svm_iter):
      w_prev = w.clone().detach()
      total_loss = optim.step(closure)
      w_diff = float((w - w_prev).abs().max())
      record.add("Max W L2 Norm", w.norm(p=2, dim=1).max())
      record.add("Total Loss", min(total_loss, 10))
      record.add("K", w.shape[0])
      record.add("Data count", data_count)
      record.add("W/difference", min(w_diff, 0.1))
      if w_diff < 1e-4:
        break
      
    dm = calculate_metric(mfeat, msvm_label, w)
    p = int(dm.argmax())
    record.add("Max_p P(p's margin | p)", dm[p])
    record.add("1/M Sum_p P(p's margin | p)", dm.mean())
    # forge test label
    for j in range(len(mlabel)):
      delta = (msvm_label[j][:, p:p+1] > 0).float() * 2 # (L, 1)
      msvm_label[j][:, :p] += delta
      msvm_label[j][:, p+1:] += delta

    test_w = w.clone().detach().requires_grad_(True)
    test_optim = torch.optim.LBFGS([test_w])
    def test_closure():
      return train_svm_l2reg_l1svc(mfeat, test_w, test_optim, msvm_label, mlabel)
    for j in range(args.svm_iter):
      w_prev = test_w.clone().detach()
      total_loss = test_optim.step(test_closure)
      w_diff = float((test_w - w_prev).abs().max())
      if w_diff < 1e-4:
        break

    mm = calculate_metric(mfeat, msvm_label, test_w)
    Pdec = dm + dm[p] - mm
    Pdec[p] = 0
    q = int(Pdec.argmax())
    record.add("Max_q Pm(p) + Pm(q) - Pm(p+q)", Pdec[q])
    #record.add("Max_q Em(p) + Em(q) - Em(p+q)", Pdec[q])
    # save merge tree
    tree[count] = {
      "W": torch2numpy(w),
      "Total Objective": torch2numpy(dm.mean()),
      "Merging Metric": torch2numpy(Pdec[q]),
      "Deleting Metric": torch2numpy(dm[p])}
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