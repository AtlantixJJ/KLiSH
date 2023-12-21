"""Visualize binary mask."""
import sys
sys.path.insert(0, ".")
import argparse, os, tqdm, math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from lib.visualizer import segviz_numpy
from lib.misc import set_cuda_devices
from lib.op import image2torch, pairwise_dist_minibatch, bu
from models.helper import build_generator
from sklearn.cluster import DBSCAN


class DBSCANThread(threading.Thread):
  def __init__(self, save_dir, H, W, image, dist, eps, ratio):
    super().__init__()
    self.H, self.W = H, W
    self.image = image
    self.save_dir = save_dir
    self.dist = dist
    self.eps = eps
    self.ratio = ratio
  
  def run(self):
    N, H, W = self.image.shape[0], self.H, self.W
    prefix = f"{self.save_dir}/{N}_{eps:.2f}_{ratio:.5f}"
    alg = DBSCAN(
      eps=self.eps,
      min_samples=int(self.dist.shape[0] * self.ratio),
      metric='precomputed',
      n_jobs=-1)
    label = alg.fit_predict(self.dist).reshape(-1, H, W)
    vizs = np.stack([segviz_numpy(l) for l in label]) # (N, H, W, 3)
    vizs = image2torch(vizs)
    disp = []
    for i in range(N):
      disp += [self.image[i], vizs[i]]
    vutils.save_image(torch.stack(disp),
      f"{prefix}_viz.png",
      nrow=4, padding=2, pad_value=0.5)


def run(save_dir, H, W, image, dist, eps, ratio):
  N = image.shape[0]
  prefix = f"{save_dir}/{N}_{eps:.2f}_{ratio:.5f}"
  alg = DBSCAN(
    eps=eps,
    min_samples=int(dist.shape[0] * ratio),
    metric='precomputed',
    n_jobs=-1)
  label = alg.fit_predict(dist).reshape(-1, H, W)
  vizs = np.stack([segviz_numpy(l) for l in label]) # (N, H, W, 3)
  vizs = image2torch(vizs)
  disp = []
  for i in range(N):
    disp += [image[i], vizs[i]]
  vutils.save_image(torch.stack(disp),
    f"{prefix}_viz.png",
    nrow=4, padding=2, pad_value=0.5)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--name", default="dbscan")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq")
  parser.add_argument("--layer-idx", default=9, type=int)
  # training
  parser.add_argument("--gpu-id", default="0", type=str)
  parser.add_argument("--N", default=4, type=int)
  # optimization arguments
  parser.add_argument("--seed", default=2021, type=int)
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  os.system(f"mkdir {args.expr}/{args.name}")

  if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

  print(f"=> Preparing unsupervised data")
  G = build_generator(args.G_name).net
  z = torch.randn(args.N, 512)
  feat, image = [], []
  with torch.no_grad():
    for i in tqdm.tqdm(range(args.N)):
      z_ = z[i:i+1].cuda()
      wp = G.mapping(z_).unsqueeze(1).repeat(1, G.num_layers, 1)
      image_, feature_ = G.synthesis(wp, generate_feature=True)
      if len(feat) == 0:
        # use double throughout the computation
        _, C, H, W = feature_[args.layer_idx].shape
        feat = torch.zeros(args.N, H, W, C).cuda()
      feat[i].copy_(feature_[args.layer_idx][0].permute(1, 2, 0))
      image.append(image_)
    image = torch.cat(bu(image, feat.size(2)))
    image = (1 + image.clamp(-1, 1)).cpu() / 2
    feat /= feat.norm(p=2, dim=3, keepdim=True) # force arccos

  del G
  X = feat.view(-1, C)
  dist = pairwise_dist_minibatch(X, X, 'arccos').numpy()
  print("=> Distance matrix calculation done.")
  del X

  with ThreadPoolExecutor(max_workers=10) as executor:
    for eps in np.linspace(0.5, 0.7, 10):
      for ratio in np.linspace(0.01, 0.02, 10):
        #threads.append(DBSCANThread(f"{args.expr}/{args.name}/", H, W, image, dist, eps, ratio))
        executor.submit(run, f"{args.expr}/{args.name}/", H, W, image, dist, eps, ratio)
    