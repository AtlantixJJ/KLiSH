"""Visualize binary mask."""
import sys
sys.path.insert(0, ".")
import argparse, os, tqdm, math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from lib.visualizer import segviz_numpy
from lib.misc import set_cuda_devices
from lib.op import image2torch, pairwise_dist_minibatch, sample_layer_feature, sample_latent
from models.helper import build_generator
from sklearn.cluster import DBSCAN


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
  parser.add_argument("--name", default="sc")
  # architecture
  parser.add_argument("--G-name", default="stylegan2_ffhq")
  parser.add_argument("--layer-idx", default="auto", type=int)
  parser.add_argument("--latent-type", default="trunc-wp", type=str,
    choices=["trunc-wp", "wp", "mix-wp"],
    help="The latent type of StyleGANs.")
  # training
  parser.add_argument("--gpu-id", default="0", type=str)
  parser.add_argument("--N", default=4, type=int)
  # optimization arguments
  parser.add_argument("--seed", default=1993, type=int)
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  os.system(f"mkdir {args.expr}/{args.name}")

  if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

  print(f"=> Preparing unsupervised data")
  G = build_generator(args.G_name).net

  with torch.no_grad():
    wps = sample_latent(G, args.N, args.latent_type)
  feat = sample_layer_feature(
    G, args.N, wps, args.layer_idx, args.latent_type)

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
    