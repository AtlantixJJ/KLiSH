"""Summarize the evaluation of all the semantic extractors."""
import sys, argparse, glob
sys.path.insert(0, ".")
import numpy as np
from collections import OrderedDict

from lib.misc import *
from predictors.face_segmenter import CELEBA_CATEGORY15
from lib.evaluate import read_results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="results/semantics", help="")
  args = parser.parse_args()

  name = "derive"
  label_dic = read_selected_labels()
  face_labels = CELEBA_CATEGORY15[1:]
  for G in ["stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq"]:
    label_dic[G] = face_labels

  Gs = [
    "stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq",
    "stylegan2_bedroom", "stylegan2_church",
    "stylegan_bedroom", "stylegan_church", 
    "pggan_bedroom", "pggan_church"]
  show_names = ["StyleGAN2-FFHQ", "StyleGAN-CelebAHQ", "PGGAN-CelebAHQ",
    "StyleGAN2-Bedroom", "StyleGAN2-Church",
    "StyleGAN-Bedroom", "StyleGAN-Church", 
    "PGGAN-Bedroom", "PGGAN-Church"]

  dic, gdic = OrderedDict(), OrderedDict()
  mdic, gmdic = OrderedDict(), OrderedDict()
  for G in Gs: 
    G_show = show_names[Gs.index(G)]
    dic[G_show], gdic[G_show] = {}, {}
    for model_name in ["SVM", "LSE", "NSE-1", "NSE-2"]:
      dic[G_show][model_name] = {}
      if model_name == "SVM":
        ds_name = f"{G}_LSE_lsvm-1"
      else:
        ds_name = f"{G}_{model_name}_lnormal"

      try:
        fpath = glob.glob(f"{args.dir}/*{ds_name}*.txt")[0]
        mIoU, c_iou = read_results(fpath)
      except:
        continue
      
      label_names = label_dic[G]
      for i in range(len(c_iou)):
        dic[G_show][model_name][label_names[i]] = c_iou[i]
      gdic[G_show][model_name] = mIoU

  # Add delta between SVM and LSE
  for G in gdic.keys():
    #ratio = (gdic[G]["SVM"] - gdic[G]["LSE"]) / gdic[G]["LSE"]

    d = {} # re-index the columns
    d["SVM"] = gdic[G]["SVM"]
    #d["LSE"] = gdic[G]["LSE"]
    #d["$\\Delta\\%$"] = ratio

    gdic[G] = d

  for k, v in dic.items():
    try:
      strs = str_table_single(dic[k])
      with open(f"results/tex/{k}_SE_class.tex", "w") as f:
        f.write(str_latex_table(strs))
    except:
      continue
    
  key_sorted = list(gdic.keys())
  key_sorted.sort()
  gdic = {k : gdic[k] for k in key_sorted}
  print(gdic)
  strs = str_table_single(gdic)
  with open(f"results/tex/SE.tex", "w") as f:
    f.write(str_latex_table(strs))