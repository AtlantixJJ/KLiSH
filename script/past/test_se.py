"""Test semantic extractors.
"""
import sys, argparse
sys.path.insert(0, ".")
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger

from models.helper import *
from lib.callback import *
from lib.evaluate import evaluate_SE, write_results


def get_features(synthesis, wp, resolution, P=None, is_large_mem=False):
  images, features, labels = [], [], []
  with torch.no_grad():
    for i in range(wp.shape[0]):
      image, feature = synthesis(wp[i:i+1], generate_feature=True)
      if P:
        labels.append(P(image, size=resolution).long())
      if is_large_mem:
        feature = [f.cpu() for f in feature]
      features.append(feature)
      images.append(image)
  features = [torch.cat([feats[i] for feats in features])
    for i in range(len(features[0]))]
  images = bu(torch.cat(images), resolution)
  images = ((images.clamp(-1, 1) + 1) / 2).cpu()
  if P:
    labels = torch.cat(labels)
    N, H, W = labels.shape
    svm_labels = -torch.ones(N, P.n_class, H, W).to(wp)
    svm_labels.scatter_(1, labels.unsqueeze(1), 1)
    return images, features, labels, svm_labels
  return images, features


def main(args):
  from predictors.face_segmenter import FaceSegmenter
  from predictors.scene_segmenter import SceneSegmenter

  G = build_generator(args.G_name)
  is_face = "celebahq" in args.G_name or "ffhq" in args.G_name
  resolution = 512 if is_face else 256
  if is_face:
    P = FaceSegmenter()
  else:
    P = SceneSegmenter() if args.full_label \
      else SceneSegmenter(model_name=args.G_name)
  print(f"=> Segmenter has {P.n_class} classes")

  SE = load_semantic_extractor(args.SE)
  SE.cuda().train()

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  np.random.seed(args.seed)

  name = args.SE.split("/")[-2].replace("expr", "results/")
  fpath = f"results/semantics/{name}_els{args.latent_strategy}.txt"
  num = 10000
  mIoU, c_ious = evaluate_SE(SE, G.net, P,
    resolution, num, args.latent_strategy)
  write_results(fpath, mIoU, c_ious)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Architecture setting
  parser.add_argument('--latent-strategy', type=str, default='trunc-wp',
    choices=['notrunc-mixwp', 'trunc-wp', 'notrunc-wp'],
    help='notrunc-mixwp: mixed W+ without truncation. trunc-wp: W+ with truncation. notrunc-wp: W+ without truncation.')
  parser.add_argument('--G-name', type=str, default='stylegan2_ffhq',
    help='The model type of generator')
  parser.add_argument('--SE', type=str, default='expr/semantics/stylegan2_ffhq_LSE_lnormal_lstrunc-wp_lwnone_lr0.001/stylegan2_ffhq_LSE.pth',
    help='The model type of semantic extractor')
  parser.add_argument('--full-label', type=int, default=0,
    help='Default: 0, use selected label. 1: use full label.')
  parser.add_argument('--gpu-id', type=str, default='0',
    help='GPUs to use.')
  parser.add_argument('--seed', type=int, default=1113,
    help='The random seed.')
  args = parser.parse_args()
  from lib.misc import set_cuda_devices
  set_cuda_devices(args.gpu_id)
  main(args)