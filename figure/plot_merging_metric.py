"""Create qualitative figures used in the paper."""
import sys
sys.path.insert(0, ".")
import argparse, glob
import numpy as np
import torch
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
import matplotlib.pyplot as plt


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-dir",
    default="expr/cluster/svm_lowdensity")
  parser.add_argument("--out-dir",
    default="results/plot")
  args = parser.parse_args()

  files = glob.glob(f"{args.in_dir}/*tree.pth")
  files.sort()

  for fp in files:
    name = fp[fp.rfind("/")+1:-9] # exclude tree.pth
    tree = torch.load(fp, map_location="cpu")
    x, mm, dm, l = [], [], [], []
    for k, v in tree.items():
      x.append(k)
      dm.append(v["Deleting Metric"])
      mm.append(v["Merging Metric"])
      l.append(v["Total Objective"])
    plt.figure(figsize=(20, 6))
    ax = plt.subplot(1, 3, 1)
    ax.plot(x, dm)
    ax.set_xlabel("$t$")
    if "lowdensity" in args.in_dir:
      #ax.set_title("$Max_k Pm_k(\\textbf{W}; \\mathcal{C}^{t})$")
      ax.set_title("Max_k Pm_k$")
    else:
      ax.set_title("Max_p E(p's dist2 to margin | p)")
    ax = plt.subplot(1, 3, 2)
    ax.plot(x, mm)
    ax.set_xlabel("$t$")
    if "lowdensity" in args.in_dir:
      #ax.set_title("$Max_{k \\neq p^t} Pm_{p^t} + Pm_{k} - Pm_1(\\overline{\\textbf{W}}_k; \\overline{\\mathcal{A}}_k)$")
      ax.set_title("Max_q Pm_p + Pm_q - Pm_{p+q}")
    else:
      ax.set_title("Max_q Em(p) + Em(q) - Em(p+q)")
    ax = plt.subplot(1, 3, 3)
    ax.plot(x, l)
    if "lowdensity" in args.in_dir:
      #ax.set_title("\\mathcal{L}_K")
      ax.set_title("1/K sum_k Pm_k")
    else:
      ax.set_title("Max_q Em(p) + Em(q) - Em(p+q)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/merge-metric_{name}.png")
    plt.close()