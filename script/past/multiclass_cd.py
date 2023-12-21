"""KMeans initialized SVM loss training with cosine similarity-based weight pruning."""
import sys
sys.path.insert(0, ".")
import argparse, os, math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm

from lib.visualizer import segviz_torch, SimpleVideoRecorder, plot_dict
from lib.misc import set_cuda_devices
from lib.op import *
from models.helper import build_generator
from lib.misc import DictRecorder
from lib.cluster import *


EPS = 1e-9


def calc_metric_worker(feat, bl, w):
  """The worker for calculating merging metrics in a thread.
  Args:
    feat: (L, C)
    w: (K, C)
  """
  N, K = 8, w.shape[0]
  B = feat.shape[0] // N
  pdist = torch.zeros(K, K).to(w)
  with torch.no_grad():
    for i in range(N):
      st, ed = B * i, B * (i + 1)
      seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (B, K)
      for k in range(w.shape[0]):
        diff = seg[:, k:k+1] - seg
        pdist[k] += (diff * bl[st:ed, k:k+1]).clamp(max=1).sum(dim=0)
  return pdist


def calculate_metric(mfeat, bl, class_size, w):
  """Calculate merging metrics with multiple threads.
  """
  pdist, threads = 0, []
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(calc_metric_worker,
      mfeat[d_id], bl[d_id], w.clone().to(f"cuda:{d_id}")))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    pdist += threads[d_id].res.cpu()
  dist = pdist / class_size.clamp(min=1).unsqueeze(1)
  return dist + dist.permute(1, 0)


def mc_cd_raw(feat, w, bl, class_size, coef):
  """Multi-Class Class-wise Distance: raw form
  """
  N = 16
  B = feat.shape[0] // N
  t = class_size.to(w).clamp(min=1)
  acc_loss = 0
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (B, K)
    xi_p, xi_n = 0, 0
    for k in range(w.shape[0]):
      diff = seg[:, k:k+1] - seg
      #xi_p = xi_p + ((1 - diff) * bl[st:ed, k:k+1]).clamp(min=0).sum(dim=0) / t[k]
      xi_n = xi_n + ((1 + diff) * bl[st:ed]).clamp(min=0).sum(dim=0) / t
    #loss = (xi_p + xi_n).sum() / N * coef
    loss = xi_n.sum() / N * coef
    loss.backward()
    acc_loss += loss.detach()
  return acc_loss


def mc_cd(feat, w, label, bl, class_size, loss_type, coef):
  N = 8
  B = feat.shape[0] // N
  t = class_size.to(w).clamp(min=1)
  acc_loss = 0
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1)) # (L, 1)
    margin = (1 - bl[st:ed] + seg - s_true).clamp(min=0)
    if loss_type == "l2":
      margin = torch.square(margin)
    class_loss = torch.matmul(margin.permute(1, 0), bl[st:ed])
    svm_loss = (class_loss / t.unsqueeze(0)).sum() * coef
    #svm_loss = class_loss.sum() * coef
    svm_loss.backward()
    acc_loss += svm_loss.detach()
  return acc_loss


def train_closure(mfeat, w, optim, mlabel, mbin_label,
                  class_size, loss_type, coef):
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(mc_cd,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"),
      mlabel[d_id], mbin_label[d_id], class_size,
      loss_type, coef))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    svm_loss += threads[d_id].res.cpu()
  reg_loss = torch.square(w).sum() * 0.5
  reg_loss.backward()
  total_loss = reg_loss.cpu() + svm_loss
  return total_loss


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster",
    help="The directory of experiments.")
  parser.add_argument("--name", default="mc_cd",
    help="The name of the experiment.")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq",
    help="The name of generator, should be in models/pretrained/pytorch folder.")
  parser.add_argument("--layer-idx", default="auto", type=str,
    help="The layer indice used for collecting features, use ``auto'' for using the default layer selection process.")
  parser.add_argument("--N", default=256, type=int,
    help="The number of image samples.")
  parser.add_argument("--S", default=256, type=int,
    help="The image resolution.")
  parser.add_argument("--use-bias", default=0, type=int,
    choices=[0, 1],
    help="Whether to use bias.")
  # training
  parser.add_argument("--latent-type", default="trunc-wp", type=str,
    choices=["trunc-wp", "wp", "mix-wp"],
    help="The latent type of StyleGANs.")
  parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
  parser.add_argument("--svm-iter", default=100, type=int,
    help="Maximum SVM training iterations.")
  parser.add_argument("--N-viz", default=16, type=int,
    help="The number of visualizing images.")
  parser.add_argument("--resample-interval", default=200, type=int,
    help="How often to resample the data. Set to 200 for no resampling.")
  parser.add_argument("--seed", default=1997, type=int)
  args = parser.parse_args()
  n_gpu = set_cuda_devices(args.gpu_id)
  devices = list(range(n_gpu))

  coef = 1 / args.N
  B = args.N // n_gpu
  S, N_viz = 256, min(B, 16)
  name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_{args.N}_{args.seed}"

  #if "ada_cat" in args.name or "ada_dog" in args.name or "ada_wild" in args.name:
  K, seed, truncation = 100, 1993, 0.5

  kmeans_wpath = f"{args.expr}/kmeans/{args.G_name}_l{args.latent_type}_i{args.layer_idx}_K{K}_N256_S256_{seed}_arccos.pth"
  prefix = f"{args.expr}/{args.name}/{name}"
  if not os.path.exists(f"{args.expr}/{args.name}"):
    os.makedirs(f"{args.expr}/{args.name}")

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
  
  mlabel = [torch.matmul(feat, w.to(feat).permute(1, 0)).argmax(1)
    for feat in mfeat]
  mbin_label = [torch.zeros(y.shape[0], w.shape[0]).to(y.device).\
    scatter_(1, y.unsqueeze(1), 1) for y in mlabel]
  class_size = sum([y.sum(0).cpu() for y in mbin_label])
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
  video_recorder = SimpleVideoRecorder(prefix=prefix, N_viz=N_viz)
  count, tree, cur_label = 0, {}, w.shape[0]
  idx2label = [i for i in range(w.shape[0])]
  for i in tqdm(range(200)):
    def closure():
      return train_closure(mfeat, w, optim, mlabel, mbin_label,
                            class_size, "l1", coef)

    for j in range(args.svm_iter):
      w_prev = w.clone().detach()
      total_loss = optim.step(closure)
      w_diff = float((w - w_prev).abs().max())
      record.add("Max W L2 Norm", w.norm(p=2, dim=1).max())
      record.add("Total Loss", min(total_loss, 10))
      record.add("K", w.shape[0])
      record.add("W/difference", min(w_diff, 0.1))
      if w_diff < 1e-4:
        break
    
    # calculate metric
    dist = calculate_metric(mfeat, mbin_label, class_size, w)
    min_lld, (q, p) = upper_triangle_minmax(dist, "min")
    record.add("Minimum Linear Distance", min_lld)

    video_recorder(w, mimage[-1], mfeat[-1],
      mlabel[-1].view(B, H, W), q, p)
    # save merge tree
    tree[count] = {
      "W": torch2numpy(w),
      "Merging Metric": torch2numpy(min_lld),
      "Deleting Metric": torch2numpy(min_lld)}
    tree[count]["merge"] = [idx2label[q], idx2label[p], cur_label]
    cur_label += 1
    torch.save(tree, f"{prefix}_tree.pth")
    idx2label[q] = count # p -> q, q -> cur_label
    del idx2label[p]
    count += 1

    # merge weights
    new_w = w.clone().detach().requires_grad_(False)
    r = class_size[p] / (class_size[p] + class_size[q] + EPS)
    new_w[q] = new_w[p] * r + new_w[q] * (1 - r)
    new_w = delete_index(new_w, p, 0)
    new_w = new_w.clone().detach().requires_grad_(True)
    new_optim = torch.optim.LBFGS([new_w])
    del optim, w
    optim, w = new_optim, new_w
    torch.cuda.empty_cache()
    for j in range(len(mlabel)):
      mbin_label[j][:, q] += mbin_label[j][:, p]
      t = mbin_label[j].cpu().detach()
      t = delete_index(t, p, 1)
      mbin_label[j] = None
      torch.cuda.empty_cache()
      mbin_label[j] = t.to(f"cuda:{j}")
      mlabel[j] = mbin_label[j].argmax(1)
    class_size = sum([y.sum(0).cpu() for y in mbin_label])
    plot_dict(record, f"{prefix}_record.png")
    if w.shape[0] <= 1:
      break

  torch.save(tree, f"{prefix}_tree.pth")
  plot_dict(record, f"{prefix}_record.png")
  video_recorder.clean_sync()