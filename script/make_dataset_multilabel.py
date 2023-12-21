"""Generate a GAN dataset from KLiSH clustered weights.
"""
import os
import argparse
import json
from models.helper import (
    build_generator,
    load_semantic_extractor,
    generate_image,
    auto_layer_selection,
    sample_latent,
)
from models.semantic_extractor import SimpleLSE
from lib.op import bu, torch2numpy
from lib.misc import set_cuda_devices, imwrite
from lib.visualizer import segviz_numpy
from tqdm import tqdm
from torchvision import utils as vutils
import torch.nn.functional as F
import torch
import numpy as np


def main():
    """Entrance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="expr")
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--layer-idx", default="auto", type=str)
    parser.add_argument("--N", default=50000, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--latent-type", default="trunc-wp", type=str)
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--seed", default=1113, type=int)
    parser.add_argument(
        "--generate-fewshot",
        default=0,
        type=int,
        help="Whether to generate fewshot training data.",
    )
    parser.add_argument(
        "--generate-klish",
        default=0,
        type=int,
        help="Whether to generate KLiSH training data.",
    )
    parser.add_argument(
        "--generate-ahc",
        default=0,
        type=int,
        help="Whether to generate AHC training data.",
    )
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)
    torch.set_grad_enabled(False)

    S = args.resolution
    if args.generate_fewshot:
        anns = ["human"] if args.G_name == "stylegan2_car" else ["deeplabv3"]
    else:
        anns = []
    LSEs = {}
    for ann_type in anns:
        model_dir = f"expr/fewshot_{ann_type}/{args.G_name}_LSE_fewshot/"
        dirs = os.listdir(model_dir)
        dirs.sort()
        for d in dirs:
            if "." in d:
                continue
            f = f"{model_dir}/{d}/model.pth"
            try:
                LSEs[f"{ann_type}_{d}"] = load_semantic_extractor(f).cuda()
            except:
                print(f"!> {f} not found!")

    G = build_generator(args.G_name).net.cuda()
    features = generate_image(G, sample_latent(G, 1), generate_feature=True)[1]
    shapes = [f.shape for f in features]
    layer_indice = auto_layer_selection(shapes)

    with open("figure/klish_selected_clusters.json", "r", encoding="ascii") as f:
        cluster_selector = json.load(f)
    selected_clusters = cluster_selector["n_cluster"][args.G_name]
    if args.generate_klish:
        klish_file_format = "iauto_b1_heuristic_ovrsvc-l2_{seed}_tree.pth"
        seed = cluster_selector["seed"][args.G_name]
        klish_file_name = klish_file_format.format(seed=seed)
        tree = torch.load(f"expr/cluster/klish/{args.G_name}_{klish_file_name}")
        dic = tree[selected_clusters]
        mtd_name = "klish"
    else:
        ahc_fname = f"{args.G_name}_ltrunc-wp_iauto_N32_S64_arccos_1990"
        tree = torch.load(f"expr/cluster/ahc/{ahc_fname}.pth")
        dic = tree[selected_clusters]
        mtd_name = "ahc"
    w = dic["weight"].cuda()
    if "bias" in dic:
        b = dic["bias"].cuda()
    else:
        b = None
    n_class = w.shape[0]
    assert n_class == selected_clusters
    slse = SimpleLSE(w, layer_indice, S, b)
    slse.n_class = selected_clusters
    LSEs[mtd_name] = slse

    name = f"{args.G_name}_s{args.seed}"
    prefix = f"data/generated/{name}"

    if not os.path.exists(f"{prefix}/wps.npy"):
        os.system(f"mkdir {prefix}")
        wps = [
            sample_latent(G, 100, args.latent_type).cpu()
            for _ in tqdm(range(args.N // 100))
        ]
        wps = torch.cat(wps)
        torch.save(wps, f"{prefix}/wps.npy")
    else:
        wps = torch.load(f"{prefix}/wps.npy")

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if not os.path.exists(f"{prefix}/image"):
        os.makedirs(f"{prefix}/image")
    map_labels = {}
    for k, net in LSEs.items():
        n_class = net.n_class
        if args.G_name == "stylegan2_car":
            print("=> Removing background label of car")
            y = []
            for i in tqdm(range(100)):
                res = generate_image(G, wps[i : i + 1].cuda(), generate_feature=True)
                feat = res[1]
                if isinstance(net, SimpleLSE):
                    y_ = net(feat, S).argmax(1)
                else:
                    y_ = net(feat, S)[-1].argmax(1)
                y.append(y_)
            y = torch.cat(y)
            remove_labels = torch.cat([y[:, :64], y[:, -64:]]).reshape(-1)
            normal_labels = y[:, 64:-64].reshape(-1)
            remove_bins = torch.bincount(remove_labels, minlength=n_class)
            normal_bins = torch.bincount(normal_labels, minlength=n_class)
            ratio = remove_bins / (remove_bins + normal_bins).clamp(min=1)
            keep_label = torch.nonzero(ratio <= 0.5).squeeze(1)
            remove_label = torch.nonzero(ratio > 0.5).squeeze(1)
            new_n = keep_label.shape[0]
            map_label = torch.zeros((new_n + 1, n_class))
            map_label[0, remove_label] = 1
            for i, idx in enumerate(keep_label):
                map_label[i + 1, idx] = 1
            map_labels[k] = map_label.view(-1, n_class, 1, 1).cuda()
            print(f"=> Reduce {k} from {net.n_class} to {new_n}")
            net.n_class = new_n + 1
        label_dir = f"{prefix}/label_{k}_c{net.n_class}"
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        labelviz_dir = f"{prefix}/labelviz_{k}_c{net.n_class}"
        if not os.path.exists(labelviz_dir):
            os.makedirs(labelviz_dir)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    for i in tqdm(range(args.N)):
        image, feat = generate_image(G, wps[i : i + 1].cuda(), generate_feature=True)

        image = bu((image.clamp(-1, 1) + 1) / 2, S)
        if "car" in args.G_name:
            image = image[:, 64:-64]
        if not os.path.exists(f"{prefix}/image/{i:07d}.jpg"):
            vutils.save_image(image, f"{prefix}/image/{i:07d}.jpg")
        for j in range(len(feat)):
            feat[j] = feat[j].float()
        for k, net in LSEs.items():
            if isinstance(net, SimpleLSE):
                label = net(feat, S).argmax(1)
            else:
                label = net(feat, S)[-1].argmax(1)
            n_class = net.n_class
            if "car" in args.G_name:
                w = map_labels[k]
                new_label = torch.zeros(1, w.shape[1], S, S).to(label.device)
                new_label.scatter_(1, label.unsqueeze(1), 1)
                new_label = F.conv2d(new_label, w).argmax(1)
                label = new_label[:, 64:-64]
            disp_label = label.repeat(3, 1, 1)
            disp_label = torch2numpy(disp_label.permute(1, 2, 0))
            label_dir = f"{prefix}/label_{k}_c{n_class}"
            imwrite(f"{label_dir}/{i:07d}.png", disp_label)
            if i < 100:
                labelviz_dir = f"{prefix}/labelviz_{k}_c{n_class}"
                disp_label_viz = segviz_numpy(disp_label[..., 0])
                imwrite(f"{labelviz_dir}/{i:07d}.png", disp_label_viz)


if __name__ == "__main__":
    main()
