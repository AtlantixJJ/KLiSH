"""Test image segmentation networks.
"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from lib.visualizer import segviz_torch
from lib.misc import set_cuda_devices
from predictors.helper import P_from_name
from lib.evaluate import evaluate_dataset, write_results
from lib.dataset import dataloader_from_name


def label_perm_func(P, w):
    """Generate new prediction function for permuted labels."""

    def func(x):
        seg = P(x)
        label = seg.argmax(1)
        bin_label = torch.zeros_like(seg).scatter_(1, label.unsqueeze(1), 1)
        perm_label = F.conv2d(bin_label, w)  # .argmax(1)
        return perm_label

    return func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/predictions_face",
        help="The experiment directory.",
    )
    parser.add_argument("--dataset", type=str, default="CelebAHQ-Mask")
    parser.add_argument("--calc", type=int, default=0)
    parser.add_argument("--gpu-id", type=str, default="-1")
    parser.add_argument(
        "--model",
        type=str,
        default="expr/image_seg/CelebAMask-HQ/c15/deeplabv3+_c15.pth",
    )
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    device = "cpu" if args.gpu_id == "-1" else "cuda:0"
    torch.set_grad_enabled(False)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    net = P_from_name(args.model).to(device)
    dl = dataloader_from_name(args.dataset, use_split="test")
    N_PERM, K, S = 200, dl.dataset.n_class, dl.dataset.image_size
    model_name = args.model.split("/")[-2]
    if "klish" in args.model:  # need label permutation
        with open("figure/klish_selected_clusters.json", "r", encoding="ascii") as f:
            klish_selector = json.load(f)
        seed = klish_selector["seed"]["stylegan2_ffhq"]
        n_cluster = klish_selector["n_cluster"]["stylegan2_ffhq"]
        fname = f"stylegan2_ffhq_klish_{seed}_bias_permweights.pth"
        perm_w = torch.load(f"expr/eval_clustering/{fname}")["bias"][n_cluster]
        perm_w = perm_w.unsqueeze(-1).unsqueeze(-1).to(device)
        P = label_perm_func(net, perm_w)
    elif "ahc" in args.model:
        with open("figure/klish_selected_clusters.json", "r", encoding="ascii") as f:
            klish_selector = json.load(f)
        n_cluster = klish_selector["n_cluster"]["stylegan2_ffhq"]
        tree = torch.load(
            f"expr/eval_clustering/stylegan2_ffhq_ahc_1990_nobias_permweights.pth"
        )["nobias"]
        perm_w = tree[n_cluster].unsqueeze(-1).unsqueeze(-1).to(device)
        P = label_perm_func(net, perm_w)
    else:
        P = net

    N_DISP = 8
    disp = []
    rng = np.random.RandomState(1)
    orig_labels = np.arange(0, net.n_class)
    new_labels = np.arange(0, net.n_class)
    rng.shuffle(new_labels)
    for i, (x, y) in enumerate(dl.dataset):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        orig_pred, perm_pred = [f(x.to(device)).argmax(1).cpu() for f in [net, P]]
        new_pred = torch.zeros_like(orig_pred)
        for orig, new in zip(orig_labels, new_labels):
            new_pred[orig_pred == orig] = new
        orig_pred = new_pred
        vizs = [segviz_torch(label) for label in [orig_pred, perm_pred, y]]
        disp.extend([(x.clamp(-1, 1) + 1) / 2] + vizs)
        if i > N_DISP:
            break
    vutils.save_image(
        torch.cat(disp),
        f"{args.out_dir}/{model_name}.png",
        nrow=4,
        padding=20,
        pad_value=255,
    )

    if args.calc:
        write_results(f"{args.out_dir}/{model_name}.txt", *evaluate_dataset(dl, P, K))
