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
  pdist, ndist = torch.zeros(K, K).to(w), torch.zeros(K, K).to(w)
  pnum, nnum = torch.zeros(K, K).to(w), torch.zeros(K, K).to(w)
  with torch.no_grad():
    for i in range(N):
      st, ed = B * i, B * (i + 1)
      seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (B, K)
      num = bl[st:ed].sum(0)
      for k in range(w.shape[0]):
        diff = seg[:, k:k+1] - seg
        pdist[k] += (diff * bl[st:ed, k:k+1]).clamp(max=1).sum(dim=0)
        pnum[k] += num[k]
        ndist[k] += (diff * bl[st:ed]).clamp(min=-1).sum(dim=0)
        nnum[k] += num
  return pdist, ndist, pnum, nnum


def calculate_metric(mfeat, bl, w):
  """Calculate merging metrics with multiple threads.
  """
  pdist, ndist, threads = 0, 0, []
  dpnum, dnnum = 0, 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(calc_metric_worker,
      mfeat[d_id], bl[d_id], w.clone().to(f"cuda:{d_id}")))
    threads[-1].start()
  for d_id in range(len(mfeat)):
    threads[d_id].join()
    pdist += threads[d_id].res[0].cpu()
    ndist += threads[d_id].res[1].cpu()
    dpnum += threads[d_id].res[2].cpu()
    dnnum += threads[d_id].res[3].cpu()
  return pdist / dpnum.clamp(min=1), ndist / dnnum.clamp(min=1)


def cs_svc(feat, w, label, delta, loss_type, coef):
  N = 8
  B = feat.shape[0] // N
  acc_loss = 0
  for i in range(N):
    st, ed = B * i, B * (i + 1)
    seg = torch.matmul(feat[st:ed], w.permute(1, 0)) # (L, K)
    s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1)) # (L, 1)
    seg = seg + delta[st:ed]
    max_values = seg.max(dim=1, keepdim=True).values
    margin = max_values - s_true
    if loss_type == "l2":
      margin = torch.square(margin)
    svm_loss = margin.sum() * coef
    svm_loss.backward()
    acc_loss += svm_loss.detach()
  return acc_loss


def svm_closure(mfeat, w, optim, mlabel, mdelta, loss_type, coef):
  optim.zero_grad()
  threads, svm_loss = [], 0
  for d_id in range(len(mfeat)):
    threads.append(GeneralThread(cs_svc,
      mfeat[d_id], w.clone().to(f"cuda:{d_id}"),
      mlabel[d_id], mdelta[d_id], loss_type, coef))
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
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--name", default="cs_lowdistd")
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

  coef = 1 / args.N / 1000
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
  mlabel = [torch.matmul(feat, w.to(feat).permute(1, 0)).argmax(1)
    for feat in mfeat]
  mdelta = [torch.ones(y.shape[0], w.shape[0]).to(y.device).\
    scatter_(1, y.unsqueeze(1), 0) for y in mlabel]
  pnum = sum([y.sum(0).cpu() for y in mdelta])
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
      return svm_closure(mfeat, w, optim, mlabel, mdelta, "l2", coef)

    for j in range(args.svm_iter):
      w_prev = w.clone().detach()
      total_loss = optim.step(closure)
      w_diff = float((w - w_prev).abs().max())
      record.add("Max W L2 Norm", w.norm(p=2, dim=1).max())
      record.add("Total Loss", total_loss)
      record.add("K", w.shape[0])
      record.add("W/difference", min(w_diff, 0.1))
      if w_diff < 1e-4:
        break
    
    # reuse delta for binary label to save GPU memory
    for j in range(len(mlabel)):
      mdelta[j].fill_(0).scatter_(1, mlabel[j].unsqueeze(1), 1)
    # calculate metric
    pos_dist, neg_dist = calculate_metric(mfeat, mdelta, w)
    lld = (pos_dist - neg_dist) / 2
    #diff_w = (w.unsqueeze(1) - w.unsqueeze(0)).cpu()
    #diff_w_norm = torch.sqrt(torch.square(diff_w).sum(2) + EPS)
    #lld = dist / diff_w_norm
    min_lld, (q, p) = upper_triangle_minmax(lld, "min")
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
    r = pnum[p] / (pnum[p] + pnum[q] + EPS)
    new_w[q] = new_w[p] * r + new_w[q] * (1 - r)
    new_w = delete_index(new_w, p, 0)
    new_w = new_w.clone().detach().requires_grad_(True)
    new_optim = torch.optim.LBFGS([new_w])
    del optim, w
    optim, w = new_optim, new_w
    torch.cuda.empty_cache()
    for j in range(len(mlabel)):
      mdelta[j][:, q] += mdelta[j][:, p]
      t = mdelta[j].cpu().detach()
      t = delete_index(t, p, 1)
      mdelta[j] = None
      torch.cuda.empty_cache()
      mdelta[j] = t.to(f"cuda:{j}")
      mlabel[j] = mdelta[j].argmax(1)
      # reset mdelta to its original usage
      mdelta[j].fill_(1).scatter_(1, mlabel[j].unsqueeze(1), 0)
    pnum = sum([y.sum(0).cpu() for y in mdelta])
    plot_dict(record, f"{prefix}_record.png")
    if w.shape[0] <= 1:
      break

  torch.save(tree, f"{prefix}_tree.pth")
  plot_dict(record, f"{prefix}_record.png")
  video_recorder.clean_sync()