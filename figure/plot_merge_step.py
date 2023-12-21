"""Visualize a single merging step."""
import sys

sys.path.insert(0, ".")
import argparse, json, matplotlib, glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

matplotlib.style.use("seaborn-poster")
matplotlib.style.use("ggplot")
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.visualizer import (
    segviz_torch,
    heatmap_torch,
    draw_box,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
)
from lib.cluster import LinearClassifier
from lib.misc import set_cuda_devices, formal_name
from lib.op import bu, torch2image, cat_mut_iou
from models.helper import build_generator, sample_layer_feature


def visualize(N, H, W, C, image, feat, models):
    """Visualize."""
    with torch.no_grad():
        image = bu(image, (H, W))
        seg1, seg2 = [model(feat.view(-1, C)) for model in models]
        label1, label2 = [seg.argmax(1).view(N, H, W).cpu() for seg in [seg1, seg2]]

    new_label2 = label2.clone()
    IoU = cat_mut_iou(label1.view(N, -1), label2.view(N, -1))[0]

    for i in range(IoU.shape[0]):
        max_ind = IoU[i].argmax()
        if IoU[i, max_ind] > 0.9:
            new_label2[label2 == max_ind] = i

    pred1_viz = segviz_torch(label1)  # (N, 3, H, W)
    pred2_viz = segviz_torch(new_label2)  # (N, 3, H, W)
    cls_img = []
    D = 10
    for i in range(image.shape[0]):
        cls_img.append(
            vutils.make_grid(
                torch.stack([image[i], pred1_viz[i], pred2_viz[i]]),
                nrow=1,
                padding=D,
                pad_value=1,
            )[:, D:-D, D:-D]
        )

    # 1, 21
    # ffhq_indice = [11, 2, 5, 8, 9, 10, 12, 16, 17, 18, 19, 23, 26, 29, 30, 34]
    ffhq_indice = [28, 1, 46, 3, 5, 6, 9, 12, 14, 15, 17, 19, 20, 21, 24, 27, 30]
    smaps_viz = []
    for i in range(image.shape[0]):
        smap = seg1.view(N, H, W, -1)[i].permute(2, 0, 1)
        smap = smap[ffhq_indice].clone()
        heatmap = heatmap_torch(smap.clamp(-1, 1)).cpu()
        img = vutils.make_grid(heatmap, nrow=4, padding=D, pad_value=1).unsqueeze(0)
        img = draw_box(img, (0, 0, H, W), D, np.array([96, 253, 62]) / 255.0)
        idx, idy = 1, 2  # hardcoded
        img = draw_box(
            img,
            (idx * (H + D), idy * (W + D), H, W),
            D,
            np.array([254, 244, 77]) / 255.0,
        )
        smaps_viz.append(img[0])
    return cls_img, smaps_viz


def main():
    """Entrance."""
    with open("results/tex/cluster_bestindice.json", "r", encoding="ascii") as f:
        best_seed_dic = json.load(f)
        G_name = formal_name(args.G_name)
        best_seed_dic = best_seed_dic[G_name]

    cuts = [54, 53]
    klish_seed = best_seed_dic["KLiSH (bias)"]
    klish_wfile = torch.load(
        f"{args.expr}/klish/{args.G_name}_iauto_b1_heuristic_ovrsvc-l2_{klish_seed}_tree.pth"
    )
    klish_slse = LinearClassifier.load_as_lc(klish_wfile, None, cuts)["bias"]
    klish_slse = [klish_slse[k] for k in cuts]

    cls_img, smaps_viz = visualize(N, H, W, C, image, feat, klish_slse)
    for i, x in enumerate(cls_img):
        vutils.save_image(x, f"{prefix}_cls{i}.png")
    for i, x in enumerate(smaps_viz):
        img = torch2image(x, data_range="[0,1]")
        ax = plt.subplot()
        ax.axis("off")
        ax.grid("off")
        ax.imshow(img)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colors1 = POSITIVE_COLOR(np.linspace(0.0, 1.0, 128))
        colors2 = NEGATIVE_COLOR(np.linspace(1.0, 0.0, 128))
        # combine them and build a new colormap
        colors = np.vstack((colors2, colors1))
        cmap = LinearSegmentedColormap.from_list("my", colors)
        norm = plt.Normalize(0, 1)
        plt.colorbar(cm.ScalarMappable(norm, cmap), cax=cax)
        plt.tight_layout()
        plt.savefig(f"{prefix}_smap{i}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--out-dir", default="results/plot")
    # architecture
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--layer-idx", default="auto", type=str)
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--N", default=4, type=int)
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)

    G = build_generator(args.G_name).net
    if "ffhq" in args.G_name or "car" in args.G_name:
        wps = []
        # do not replicate other figure
        for i in range(30, 30 + args.N):
            wps.append(torch.load(f"data/{args.G_name}_fewshot/latent/wp_{i:02d}.npy"))
        wps = torch.cat(wps).cuda()
        image, feat = sample_layer_feature(
            G, args.N, wps=wps, layer_idx=args.layer_idx, latent_type="trunc-wp"
        )
    else:
        image, feat = sample_layer_feature(
            G, args.N, layer_idx=args.layer_idx, latent_type="trunc-wp"
        )
    image = bu(image, feat.shape[2])
    N, H, W, C = feat.shape
    feat = feat.reshape(-1, C)
    del G
    prefix = f"{args.out_dir}/{args.G_name}"
    main()
