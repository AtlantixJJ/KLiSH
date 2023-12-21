"""Draw the pyramid figure of unsupervised segmentation data."""
import sys, torch, argparse, os

sys.path.insert(0, ".")
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from lib.op import bu
from lib.dataset import ImageSegmentationDataset
from lib.visualizer import segviz_torch
from predictors.helper import P_from_name


def worker():
    """Worker."""
    DATA_DIR = f"data/generated/{args.G_name}_s1113"

    if args.G_name == "stylegan2_ffhq":
        indice = [5, 8, 9]
    elif args.G_name == "ada_metface":
        indice = [3, 4, 14]
    else:
        indice = [5, 8, 10]
    label_set = [f for f in os.listdir(DATA_DIR) if "label_" in f][0]
    ds = ImageSegmentationDataset(DATA_DIR, use_split="test", label_folder=label_set)
    dl = DataLoader(ds, batch_size=1)

    images, pred_vizs = [], []

    for i, (x, y) in enumerate(dl):
        if i > max(indice):
            break
        if i not in indice:
            continue
        images.append((x + 1) / 2)
        pred_vizs.append(segviz_torch(y))
    pred_vizs, images = torch.cat(pred_vizs), torch.cat(images)
    large_disp, small_disp = [], []
    S = images.shape[3]

    # make 2 - 4 pyramid
    N_large = len(indice) // 3
    for i in range(N_large):
        large_disp.extend([images[i], pred_vizs[i]])
    for i in range(N_large, len(indice)):
        small_disp.extend([images[i], pred_vizs[i]])
    large_disp = torch.stack(large_disp)
    D = 10
    large_disp = vutils.make_grid(large_disp, nrow=N_large * 2, padding=D, pad_value=1)[
        :, D:-D, D:-D
    ]
    H, W = images.shape[2:]
    nW = (S - D) // 2
    nH = int(H * (nW / W))
    small_disp = bu(torch.stack(small_disp), (nH, nW))
    small_disp = vutils.make_grid(small_disp, nrow=N_large * 4, padding=D, pad_value=1)[
        :, D:-D, D:-D
    ]
    ones = torch.ones((3, D, large_disp.shape[2]))
    vutils.save_image(
        torch.cat([large_disp, ones, small_disp], 1),
        f"results/plot/{args.G_name}_usd_pyramid.png",
    )

    # make two parallel
    disp = []
    for i in range(1, 3):
        disp.extend([images[i], pred_vizs[i]])
    disp = torch.stack(disp)
    vutils.save_image(
        disp,
        f"results/plot/{args.G_name}_usd_parallel.png",
        nrow=N_large * 4,
        padding=D,
        pad_value=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--G-name", default="all")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    if args.G_name == "all":
        for G_name in [
            "stylegan2_ffhq",
            "stylegan2_car",
            "ada_wild",
            "ada_cat",
            "ada_metface",
            "ada_dog",
        ]:
            args.G_name = G_name
            worker()
    else:
        worker()
