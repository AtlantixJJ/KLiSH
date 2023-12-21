"""KMeans initialized SVM loss training with cosine similarity-based weight pruning."""
import sys
sys.path.insert(0, ".")
import argparse, os, math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm

from lib.visualizer import segviz_torch, VideoRecorder, plot_dict
from lib.misc import set_cuda_devices
from models.helper import build_generator
from lib.misc import DictRecorder
from lib.op import *
from lib.cluster import *

EPS = 1e-8


def calc_metric_worker(feat, svm_label, w):
  """The worker for calculating merging metrics in a thread.
  Args:
    feat: (L, C)
    w: (K, C)
  """
  N, K = 8, w.shape[0]
  B = feat.shape[0] // N
  pdist, ndist = torch.zeros(K, K).to(w), torch.zeros(K, K).to(w)
  pnum, nnum = torch.zeros(K, K).to(w), torch.zeros(K, K).to(w)
  with torch.no_grad():
    for i in range(N):
      st, ed = B * i, B * (i + 1)
      seg = torch.matmul(feat[st:ed], w.permute(1, 0))
      bl = (svm_label[st:ed] > 0).float()
      num = bl.sum(0)
      for k in range(w.shape[0]):
        mask_seg = seg[:, k:k+1] * bl
        ndist[k] += mask_seg.clamp(min=-1).sum(dim=0)
        nnum[k] += num
        pdist[k] += mask_seg[:, k].clamp(max=1).sum(dim=0)
        pnum[k] += num[k]
  return pdist, ndist, pnum, nnum


def calculate_metric(mfeat, msvm_label, w):
  """Calculate merging metrics with multiple threads.
  """
  pdist, ndist, dpnum, dnnum, threads = 0, 0, 0, 0, []
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(calc_metric_worker,
      mfeat[d_id], msvm_label[d_id],
      w.clone().to(f"cuda:{d_id}")))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    pdist += threads[d_id].res[0].cpu()
    ndist += threads[d_id].res[1].cpu()
    dpnum += threads[d_id].res[2].cpu()
    dnnum += threads[d_id].res[3].cpu()
  return pdist / dpnum.clamp(min=1), ndist / dnnum.clamp(min=1)


def ovr_svc(feat, w, svm_label, loss_type, coef):
  acc_svm_loss, N = 0, 8
  B = feat.shape[0] // N
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    margin = (1 - svm_label[st:ed] * seg).clamp(min=0)
    if loss_type == "l2":
      margin = torch.square(margin)
    svm_loss = margin.sum() * coef
    svm_loss.backward()
    acc_svm_loss += svm_loss.detach()
  return acc_svm_loss


def svm_closure(mfeat, w, optim, msvm_label, loss_type, coef):
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(ovr_svc,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"),
      msvm_label[d_id], loss_type, coef))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    svm_loss += threads[d_id].res.cpu()
  reg_loss = torch.square(w).sum() * 0.5
  reg_loss.backward()
  total_loss = reg_loss.cpu() + svm_loss
  return total_loss


def nondiag_minmax(M, func):
  t = M.diag()
  for i in range(M.shape[0]):
    M[i, i] = torch.inf if func == min else -torch.inf
  ind = int(M.view(-1).argmin()) if func == min \
          else int(M.view(-1).argmax())
  for i in range(M.shape[0]):
    M[i, i] = t[i]
  x, y = ind // M.shape[0], ind % M.shape[0]
  return M[x, y], (x, y)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--name", default="ovr_cd")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq")
  parser.add_argument("--layer-idx", default="auto", type=str)
  parser.add_argument("--N", default=256, type=int)
  # training
  parser.add_argument("--latent-type", default="trunc-wp", type=str)
  parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
  parser.add_argument("--svm-iter", default=100, type=int)
  parser.add_argument("--svm-type", default="l2", type=str)
  parser.add_argument("--resample-interval", default=200, type=int)
  parser.add_argument("--seed", default=1997, type=int)
  args = parser.parse_args()
  n_gpu = set_cuda_devices(args.gpu_id)
  devices = list(range(n_gpu))

  coef = 1 / args.N / 1000
  B = args.N // n_gpu
  S, minsize, N_viz = 256, 64, min(B, 16)
  name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_{args.N}_{args.svm_type}svm_{args.seed}"

  K = 100
  seed = 1993
  truncation = 0.5

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
      return svm_closure(mfeat, w, optim, msvm_label, args.svm_type, coef)
    for j in range(args.svm_iter):
      w_prev = w.clone().detach()
      total_loss = optim.step(closure)
      w_diff = float((w - w_prev).abs().max())
      record.add("Max W L2 Norm", w.norm(p=2, dim=1).max())
      record.add("Total Loss", min(total_loss, 100))
      record.add("K", w.shape[0])
      record.add("Data count", data_count)
      record.add("W/difference", min(w_diff, 0.1))
      if w_diff < 1e-4:
        break

    pdist, ndist = calculate_metric(mfeat, msvm_label, w)
    dist = (pdist - ndist) / 2
    #w_norm = w.norm(p=2, dim=1).cpu()
    #lld = dist / w_norm.unsqueeze(1).clamp(min=EPS)
    for i in range(dist.shape[0]):
      dist[i, i] = torch.inf
    indice = dist.view(-1).argsort()
    topn_indice = indice[:100]
    
    min_lld, (q, p) = nondiag_minmax(lld, min)
    record.add("Minimum Linear Distance", min_lld)

    # save merge tree
    tree[count] = {
      "W": torch2numpy(w),
      "Merging Metric": torch2numpy(min_lld),
      "Deleting Metric": torch2numpy(min_lld)}
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
    w = delete_index(w.requires_grad_(False), p, 0)
    w = w.clone().detach().requires_grad_(True)
    optim = torch.optim.LBFGS([w])
    torch.cuda.empty_cache()
    for j in range(len(mlabel)):
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