"""Clustering by Agglomerative Hierachical Clustering."""
# pylint: disable=invalid-name,line-too-long
import os
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from models import helper
from lib.cluster import LinearClassifier
from lib.op import copy_tensor
from lib.misc import imwrite, set_cuda_devices
from lib.visualizer import visualize_segmentation


def get_seg(tree, n_cluster):
    """
    Args:
        tree: (n_samples - 1, 2), tree[i] denotes that in iterations i, tree[i][0] and tree[i][1] are merged to form the node n_samples + i.
        n_cluster: The target n_clusters.
    Returns:
        A list of points belonging to each cluster.
    """
    n_sample = tree.shape[0] + 1
    dic = {}
    for i in range(n_sample - n_cluster):
        l, r = tree[i]
        l_list = dic[l] if l in dic else [l]
        r_list = dic[r] if r in dic else [r]
        # dict stores left right child info
        dic[i + n_sample] = l_list + r_list
        if l in dic:
            del dic[l]
        if r in dic:
            del dic[r]
    labels = torch.zeros(n_sample).long()

    # make sure existing labels are colorized in the same way
    color_table = {}
    for i in range(n_sample - 2, n_sample - n_cluster - 1, -1):
        p = i + n_sample
        l, r = tree[i]
        if p not in color_table:
            color_table[p] = 1
        color_table[l], color_table[r] = color_table[p], n_sample - i
        len_l = len(dic[l]) if l in dic else 0
        len_r = len(dic[r]) if r in dic else 0
        if len_l < len_r:
            color_table[l], color_table[r] = color_table[r], color_table[l]

    keys = list(dic.keys())
    for k in keys:
        labels[dic[k]] = color_table[k]
    return labels


def seg2weight(feat, label):
    """Initialize the weight from segmentation (clustering results).
    Args:
        feat: (N, C)
        label: (N,) M class
    Returns:
        W: (M, C) normalized to unit vector.
        compact_label: Make the label consecutive.
    """
    indice = label.unique()
    compact_label = torch.zeros(*label.shape).long()
    w = []
    for new_idx, old_idx in enumerate(indice):
        mask = label == old_idx
        compact_label[mask] = new_idx
        w.append(feat[mask].mean(0))
    return torch.stack(w), compact_label


def main():
    """Entrance."""
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--name", default="ahc")
    # architecture
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--layer-idx", default="auto", type=str)
    parser.add_argument(
        "--dist", default="arccos", type=str, choices=["euclidean", "arccos"]
    )
    parser.add_argument(
        "--n-samples", default=32, type=int, help="The number of image samples."
    )
    parser.add_argument(
        "--resolution", default=64, type=int, help="The image resolution."
    )
    parser.add_argument(
        "--n-viz", default=16, type=int, help="The number of visualizing images."
    )
    parser.add_argument(
        "--latent-type",
        default="trunc-wp",
        type=str,
        choices=["trunc-wp", "wp", "mix-wp"],
        help="The latent type of StyleGANs.",
    )
    parser.add_argument(
        "--skip-existing", default=1, type=int, help="Whether to skip existing files."
    )
    parser.add_argument("--gpu-id", default="0", type=str, help="GPU device.")
    parser.add_argument("--seed", default=1990, type=int)
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    device_ids = list(range(n_gpu))
    agg_prefix = f"{args.expr}/{args.name}/{args.G_name}_l{args.latent_type}_i{args.layer_idx}_N{args.n_samples}_S{args.resolution}_{args.dist}_{args.seed}"
    if not os.path.exists(f"{args.expr}/{args.name}"):
        os.makedirs(f"{args.expr}/{args.name}")
    if os.path.exists(f"{agg_prefix}.pth") and args.skip_existing:
        print(f"!> {agg_prefix} exists, skip.")
        return

    print(f"=> Preparing {args.n_samples} samples in {args.resolution} resolution...")
    image, feat = helper.sample_generator_feature(
        args.G_name,
        latent_type=args.latent_type,
        layer_idx=args.layer_idx,
        randomize_noise=True,
        n_samples=args.n_samples,
        size=args.resolution,
        seed=args.seed,
        device_ids=device_ids,
        cpu=True,
    )
    image, feat = image[0], feat[0]
    N, H, W, C = feat.shape
    res = {}
    start_time = datetime.today()
    if args.dist == "arccos":
        with torch.no_grad():
            feat_norm = feat.norm(p=2, dim=3, keepdim=True)
            feat /= feat_norm
        print("=> AHC(average) on arccos metric.")
        alg = AgglomerativeClustering(
            100, compute_full_tree=True, linkage="average", affinity="cosine"
        )
        X = feat.view(-1, C).numpy()
        gt_label = alg.fit_predict(X)
        with torch.no_grad():
            feat *= feat_norm
    else:
        print("=> AHC(ward) on euclidean metric.")
        alg = AgglomerativeClustering(
            100, compute_full_tree=True, linkage="ward", affinity="euclidean"
        )
        X = feat.view(-1, C).numpy()
        gt_label = alg.fit_predict(X)
    del X
    feat = feat.cuda()
    gt_label = torch.from_numpy(gt_label).view(-1, H, W)
    viz = visualize_segmentation(image, gt_label)
    imwrite(f"{agg_prefix}_{args.dist}_gt.png", viz)
    for k in tqdm(range(2, 101)):
        orig_label = get_seg(alg.children_, k)
        w, label = seg2weight(feat.view(-1, C), orig_label)
        w_norm = w.norm(p=2, dim=1, keepdim=True)
        if args.dist == "euclidean":
            b = -(w * w).sum(1) / 2
            b = (b - b.mean()) / w_norm.squeeze()
        w /= w_norm
        w = copy_tensor(w.cuda(), True)
        b = copy_tensor(b.cuda(), True) if args.dist == "euclidean" else None
        label = label.view(-1, H, W).cuda()
        model = LinearClassifier(w, b)
        optim = torch.optim.Adam(model.parameters())

        def _closure():  # on a whole dataset
            optim.zero_grad()
            ce_loss = F.cross_entropy(model(feat).permute(0, 3, 1, 2), label)
            reg_loss = 1e-3 * (model.weight**2).mean()
            loss = ce_loss + reg_loss
            loss.backward()
            return loss

        for epoch_idx in tqdm(range(1000)):
            w = model.weight
            w_prev = copy_tensor(w)
            optim.step(_closure)
            w_diff = float((w - w_prev).abs().max())
            if w_diff < 1e-4:
                break

        res[k] = model.clone().detach().to("cpu").state_dict()
        pred = model(feat).argmax(-1)
        if k % 10 == 0:
            viz = visualize_segmentation(image, pred)
            imwrite(f"{agg_prefix}_{k:03d}.png", viz)
    torch.save(res, f"{agg_prefix}.pth")
    seconds = (datetime.today() - start_time).total_seconds()
    with open(f"{agg_prefix}.txt", "w", encoding="ascii") as f:
        f.write(str(seconds))


if __name__ == "__main__":
    main()
