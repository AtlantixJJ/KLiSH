"""Generate plot of Unsupervised Fine-grained Segmentation."""
import torch, argparse, sys
import torchvision.utils as vutils
import numpy as np

sys.path.insert(0, ".")
from lib.op import bu
from lib.dataset import SimpleDataset
from lib.visualizer import segviz_torch
from predictors.helper import P_from_name


def worker():
    """Worker."""
    UNSUP_DIR = f"expr/image_seg/{args.G_name}_s1113/"
    if args.G_name == "stylegan2_ffhq":
        # indice = [6, 0, 7] # [0, 4, 5, 6, 8, 10, 11, 14, 15]
        pyramid_indice = [0, 4, 6, 8, 10, 14]
        parallel_indice = [0, 4]
        DATA_DIR = "data/CelebAMask-HQ"
        net = P_from_name(f"{UNSUP_DIR}/klish_c26/deeplabv3+_c26.pth")
        ds = SimpleDataset(f"{DATA_DIR}/image", (args.size, args.size))
    elif args.G_name == "stylegan2_car":
        # indice = [0, 4, 5]
        pyramid_indice = [13, 10, 5, 9, 14, 15]
        parallel_indice = [2, 3]
        DATA_DIR = "data/VOC2010"
        net = P_from_name(f"{UNSUP_DIR}/klish_c12/deeplabv3+_c12.pth")
        ds = SimpleDataset(f"{DATA_DIR}/demo_image", (args.size, args.size))
    N = len(pyramid_indice)

    rng = np.random.RandomState(1)
    orig_labels = np.arange(0, net.n_class)
    new_labels = np.arange(0, net.n_class)
    if args.G_name == "stylegan2_ffhq":
        rng.shuffle(new_labels)

    images, pred_vizs = [], []
    for idx in pyramid_indice:
        x = ds[idx].unsqueeze(0)
        if args.G_name == "stylegan2_car":
            x[:, :, :64] = 0
            x[:, :, -64:] = 0
        pred = net(x * 2 - 1).argmax(1)
        new_pred = torch.zeros_like(pred)
        for orig, new in zip(orig_labels, new_labels):
            new_pred[pred == orig] = new
        pred = new_pred
        if args.G_name == "stylegan2_car":
            x = x[:, :, 64:-64]
            pred = pred[:, 64:-64]
        images.append(x)
        pred_vizs.append(segviz_torch(pred))
    pred_vizs, images = torch.cat(pred_vizs), torch.cat(images)
    large_disp, small_disp = [], []
    S = images.shape[3]

    # make 2 - 4 pyramid
    N_large = len(pyramid_indice) // 3
    for i in range(N_large):
        large_disp.extend([images[i], pred_vizs[i]])
    for i in range(N_large, len(pyramid_indice)):
        small_disp.extend([images[i], pred_vizs[i]])
    large_disp = torch.stack(large_disp)
    D = 20
    large_disp = vutils.make_grid(large_disp, nrow=N_large * 2, padding=D, pad_value=1)
    large_disp = large_disp[:, D:-D, D:-D]
    H, W = images.shape[2:]
    nW = (S - D) // 2
    nH = int(H * (nW / W))
    small_disp = bu(torch.stack(small_disp), (nH, nW))
    small_disp = vutils.make_grid(small_disp, nrow=N_large * 4, padding=D, pad_value=1)
    small_disp = small_disp[:, D:-D, D:-D]
    ones = torch.ones((3, D, large_disp.shape[2]))
    vutils.save_image(
        torch.cat([large_disp, ones, small_disp], 1),
        f"results/plot/{args.G_name}_ufs_pyramid_{N}.png",
    )

    # make two parallel
    disp = []
    for i in parallel_indice:
        disp.extend([images[i], pred_vizs[i]])
    disp = vutils.make_grid(
        torch.stack(disp), nrow=N_large * 4, padding=D, pad_value=1
    )[:, D:-D, D:-D]
    vutils.save_image(disp, f"results/plot/{args.G_name}_ufs_parallel_{N}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    if args.G_name == "all":
        for G_name in [
            "stylegan2_ffhq",
            "stylegan2_car",
        ]:
            args.G_name = G_name
            worker()
    else:
        worker()
