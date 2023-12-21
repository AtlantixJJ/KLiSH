"""Divide original kmeans file's euclidean weight with 2."""
import glob
import torch

new_dir = "expr/cluster/kmeans"

new_fpaths = glob.glob(f"{new_dir}/*.pth")
new_fpaths.sort()

for new_fpath in new_fpaths:
    fname = new_fpath[new_fpath.rfind("/") + 1 :]
    new_file = torch.load(new_fpath)
    for k in [200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]:
        new_file["euclidean"][k]["weight"] /= 2
        new_file["euclidean"][k]["bias"] /= 2
    torch.save(new_file, new_fpath)
