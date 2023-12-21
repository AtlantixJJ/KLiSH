import torch, glob

old_dir = "expr/cluster/kmeans_backup"
new_dir = "expr/cluster/kmeans"

new_fpaths = glob.glob(f"{new_dir}/*.pth")
new_fpaths.sort()

for new_fpath in new_fpaths:
    fname = new_fpath[new_fpath.rfind("/")+1:]
    old_fpath = f"{old_dir}/{fname}"
    new_file = torch.load(new_fpath)
    old_file = torch.load(old_fpath)
    for k in [200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]:
        for key in new_file:
            new_file[key][k] = old_file[key][k]
    torch.save(new_file, new_fpath)