"""Summarize the evaluation of all the semantic extractors."""
import argparse, glob
import numpy as np
from collections import OrderedDict

from lib.misc import *
from lib.op import torch2numpy
from predictors.face_segmenter import CELEBA_CATEGORY15
from lib.evaluate import read_results


def formal_name(name):
  if type(name) is list:
    return [formal_name(n) for n in name]
  finds = ["stylegan", "pggan", "bedroom", "Church", "celebahq", "ffhq"]
  subs = ["StyleGAN", "PGGAN", "LSUN-Bedroom", "LSUN-Church", "CelebAHQ", "FFHQ"]
  for find, sub in zip(finds, subs):
    name = name.replace(find, sub)
  return name


def str_table_single_std(dic, output_std=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  strs = [col_names]
  for row_name in row_names:
    if len(dic[row_name]) == 0:
      continue
    s = [row_name]
    for col_name in col_names:
      if len(dic[row_name][col_name]) == 0:
        continue
      mean = dic[row_name][col_name]["mean"]
      std = dic[row_name][col_name]["std"]
      if output_std:
        item_str = f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"
      else:
        item_str = f"{mean * 100:.1f}"
      s.append(item_str)
    strs.append(s)
  return strs


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/predictions_face", help="")
  args = parser.parse_args()

  name = "derive"
  face_labels = CELEBA_CATEGORY15[1:]
  Gs = [
    "stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq",
    "stylegan2_bedroom", "stylegan2_church",
    "stylegan_bedroom", "stylegan_church", 
    "pggan_bedroom", "pggan_church"]

  dic, gdic = OrderedDict(), OrderedDict()
  mdic, gmdic = OrderedDict(), OrderedDict()
  for model_name in ["human", "deeplabv3", "init"]:
    dic[model_name], gdic[model_name] = {}, {}
    mdic[model_name], gmdic[model_name] = {}, {}
    for n_sample in ["1", "10"]:
      gdic[model_name][n_sample] = {}
      gmdic[model_name][n_sample] = {}
      mdic[model_name][n_sample] = {}
      obs = []
      c_obs = [[] for _ in face_labels]
      for rind in "01234":
        ds_show = f"$N={n_sample}, R={rind}$"
        if model_name == "init":
          ds_name = f"fs{n_sample}-us{rind}"
        else:
          ds_name = f"r{rind}_n{n_sample}"
        dic[model_name][ds_show] = {}
        try:
          fpath = glob.glob(f"{args.dir}/{model_name}/*{ds_name}_*.txt")[0]
          mIoU, c_iou = read_results(fpath)
        except:
          continue
        for i in range(len(c_iou)):
          dic[model_name][ds_show][face_labels[i]] = c_iou[i]
          c_obs[i].append(c_iou[i])
        gdic[model_name][n_sample][rind] = mIoU
        obs.append(mIoU)
      if len(obs) > 3:
        obs, c_obs = np.array(obs), np.array(c_obs)
        mean, std = obs.mean(), obs.std(ddof=1)
        c_mean, c_std = c_obs.mean(1), c_obs.std(1, ddof=1)
        gmdic[model_name][n_sample] = {"mean": mean, "std": std}
        for i in range(c_obs.shape[0]):
          mdic[model_name][n_sample][face_labels[i]] = {
            "mean": c_mean[i], "std": c_std[i]}

  model_name = "unsupervised"
  dic[model_name], gdic[model_name] = {}, {}
  for name in ["us0_c28"]:
    mIoU, c_iou = read_results(f"{args.dir}/unsupervised/{name}.txt")
    gdic[model_name][name] = {"unsupervised": mIoU}
    dic[model_name][name] = {}
    for i in range(len(c_iou)):
      dic[model_name][name][face_labels[i]] = c_iou[i]

  for k, v in dic.items():
    try:
      strs = str_table_single(dic[k])
      with open(f"results/tex/{k}_IS_class.tex", "w") as f:
        f.write(str_latex_table(strs))
      strs = str_table_single_std(mdic[k], output_std=False)
      with open(f"results/tex/{k}_IS_class_std.tex", "w") as f:
        f.write(str_latex_table(strs))
    except:
      continue
    
  key_sorted = list(gdic.keys())
  key_sorted.sort()
  gdic = {k : gdic[k] for k in key_sorted}
  print(gdic)
  strs = str_table_multiple(gdic)
  with open(f"results/tex/IS.tex", "w") as f:
    f.write(str_latex_table(strs))
  strs = str_table_single_std(gmdic)
  with open(f"results/tex/IS_std.tex", "w") as f:
    f.write(str_latex_table(strs))