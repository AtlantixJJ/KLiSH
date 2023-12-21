"""Compute various clustering metrics of clustering algorithms.
"""
# pylint: disable=invalid-name,consider-using-f-string
import json
import os
import argparse
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import lib
from lib.cluster import KASP, load_as_slse
from lib.metric import (
    rand_index,
    adjusted_rand_index,
    fowlkes_mallows_index,
    adjusted_mutual_info,
    homogeneity_score,
)
from lib.op import bu
from lib.visualizer import segviz_torch
from models import helper
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter


def eval_kasp(
    algs,
    feature,
    dt_bl,
    gt_bl,
    cmat,
    vizs,
    idx,
):
    """Evaluate KASP clustering."""
    dt_init = algs.kmeans_model(feature).argmax(1).view(-1)
    dt_bl.fill_(0)
    dt_bl.scatter_(1, dt_init.view(-1).unsqueeze(1), 1)
    for num_class in cmat.keys():
        pmat = algs.merge_matrix[num_class]
        final_dt_bl = torch.matmul(dt_bl, pmat.cuda()).clamp(max=1)
        dt = final_dt_bl.argmax(1)
        cm = torch.matmul(gt_bl, final_dt_bl)
        cmat[num_class][idx].copy_(cm)
        if idx == 0:  # visualization
            vizs.append(dt.detach().cpu())


def eval_slse(
    algs,
    feature,
    dt_bl,
    gt_bl,
    cmat,
    vizs,
    idx,
):
    """Evaluate SLSE model."""
    for num_class in cmat.keys():
        seg = algs[num_class](feature)
        dt = seg.argmax(1).view(-1)
        dt_bl.fill_(0).scatter_(1, dt.unsqueeze(1), 1)
        cm = torch.matmul(gt_bl, dt_bl[:, :num_class])
        cmat[num_class][idx].copy_(cm)
        if idx == 0:  # visualization
            vizs.append(dt.detach().cpu())


def find_label_permutation(mat):
    """
    Args:
      mat: (N, K1, K2) contingency matrix (float64). K1 < K2.
    Returns:
      w : (K1 + X, K2) permutation matrix, X is unmatched extra classes.
    """
    _, gt_num, dt_num = mat.shape
    gts = mat.sum(2)  # (N, K1)
    dts = mat.sum(1)  # (N, K2)
    w = torch.zeros(gt_num, dt_num).to(mat)  # (K1, K2)
    indice = list(range(dt_num))
    while len(indice) > 0:
        N = len(indice)
        ext_w = w.unsqueeze(-1).repeat(1, 1, gt_num * N)
        for i in range(gt_num):
            for j, real_j in enumerate(indice):
                ext_w[i, real_j, i * N + j] = 1
        ext_isct = torch.einsum("ijk,jkl->ijl", mat, ext_w)  # (N, K1, M)
        ext_union = torch.einsum("ik,jkl->ijl", dts, ext_w)
        ext_union.add_(gts.unsqueeze(2))
        ext_union.sub_(ext_isct)  # (N, K1, M)
        ext_count = (ext_union > 0).sum(0).clamp(min=1)  # (K1, M)
        ext_union.clamp_(min=1)
        ext_isct.div_(ext_union)
        ext_IoU = ext_isct.sum(0) / ext_count  # (K1, M)
        ext_mIoU = ext_IoU.mean(0)  # (M,)
        ind = int(ext_mIoU.argmax())
        gt_idx, dt_idx = ind // N, ind % N
        w[gt_idx, indice[dt_idx]] = 1
        del indice[dt_idx]
    return w


def main():
    """Calculate ARI of a clustering algorithm."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", default="expr/cluster", help="The directory of experiments."
    )
    parser.add_argument(
        "--out-dir",
        default="expr/eval_clustering",
        help="The directory of output results.",
    )
    parser.add_argument(
        "--in-dir", default="", help="Specify a custom input model directory."
    )
    parser.add_argument(
        "--G-name",
        default="stylegan2_ffhq",
        help="The name of generator, should be in models/pretrained/pytorch folder.",
    )
    parser.add_argument(
        "--eval-name",
        default="klish",
        choices=["kmeans", "klish", "kasp", "ahc"],
        help="The name of algorithm to be evaluated.",
    )
    parser.add_argument(
        "--layer-idx",
        default="auto",
        type=str,
        help="The layer indice used for collecting features, use ``auto'' for using the default layer selection process.",
    )
    parser.add_argument(
        "--n-samples", default=10000, type=int, help="The number of samples."
    )
    parser.add_argument(
        "--latent-type",
        default="trunc-wp",
        type=str,
        choices=["trunc-wp", "wp", "mix-wp"],
        help="The latent type of StyleGANs.",
    )
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument(
        "--skip-existing",
        default=1,
        type=int,
        help="Whether to skip existing result file.",
    )
    parser.add_argument("--train-seed", default=1990, type=int)
    parser.add_argument(
        "--eval-seed", default=2022, type=int, help="The seed for this evaluation."
    )
    args = parser.parse_args()
    lib.misc.set_cuda_devices(args.gpu_id)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    save_prefix = f"{args.out_dir}/{args.G_name}_{args.eval_name}_{args.layer_idx}_{args.train_seed}"
    if os.path.exists(f"{save_prefix}_bias_permweights.pth") and args.skip_existing:
        print(f"=> {save_prefix} exists, skip.")
        return

    generator = helper.build_generator(
        args.G_name, randomize_noise=True, truncation_psi=0.5
    ).net.cuda()
    features = helper.generate_image(
        generator, helper.sample_latent(generator, 1), generate_feature=True
    )[1]
    shapes = [f.shape for f in features]
    if args.layer_idx == "auto":
        layers = helper.auto_layer_selection(shapes)
    else:
        layers = [int(l) for l in args.layer_idx.split(",")]
    is_face = "celebahq" in args.G_name or "ffhq" in args.G_name
    resolution = 512 if is_face else 256
    segmenter = FaceSegmenter() if is_face else SceneSegmenter(model_name=args.G_name)
    print(f"=> Segmenter: {segmenter.n_class} classes: {segmenter.labels}")
    if len(args.in_dir) > 0:
        slse = load_as_slse(
            args.eval_name,
            layers,
            resolution,
            args.in_dir,
            args.G_name,
            args.train_seed,
            args.layer_idx,
            True,
        )
    else:
        slse = load_as_slse(
            args.eval_name, layers, resolution, args.expr, args.G_name, args.train_seed, args.layer_idx
        )
    if "kasp" in args.eval_name:
        cluster_numbers = {key: list(range(2, 101)) for key in slse}
    else:
        cluster_numbers = {key: list(slse[key].keys()) for key in slse}
    max_cluster = max(list(cluster_numbers.values())[0])
    gt_bl = torch.zeros(segmenter.n_class, resolution**2).cuda()
    dt_bl = torch.zeros(resolution**2, max_cluster).cuda()
    cmat = {
        key: {
            k: torch.zeros(args.n_samples, segmenter.n_class, k).long()
            for k in cluster_numbers[key]
        }
        for key in slse
    }
    vizs = {key: [] for key in slse}
    if args.eval_seed > 0:
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    with torch.no_grad():
        wps = helper.sample_latent(generator, args.n_samples, args.latent_type)
        for i in tqdm(range(args.n_samples)):
            image, feature = helper.generate_image(generator, wps[i : i + 1], generate_feature=True)
            gt = segmenter(image, resolution).long().view(-1)
            gt_bl.fill_(0).scatter_(0, gt.unsqueeze(0), 1)
            for bias, algs in slse.items():
                func_args = [algs, feature, dt_bl, gt_bl, cmat[bias], vizs[bias], i]
                if isinstance(algs, KASP):
                    eval_kasp(*func_args)
                else:
                    eval_slse(*func_args)
                if i == 0:  # visualization
                    vizs[bias].append(gt.detach().cpu())
                    viz_image = bu((image.clamp(-1, 1) + 1) / 2, 256).cpu()
        del dt_bl, gt_bl, gt, image, feature
        torch.cuda.empty_cache()
        shape = [1, 1, resolution, resolution]
        mIoU_table = {key: {} for key in cmat}
        perm_vizs = {key: [] for key in cmat}
        perm_weights = {key: {} for key in cmat}
        for bias, algs in slse.items():
            for idx, num_class in enumerate(tqdm(cluster_numbers[bias])):
                cm = cmat[bias][num_class].float().cuda()  # (N, K1, K2)
                perm_w = find_label_permutation(cm)  # (K1, K2)
                perm_weights[bias][num_class] = perm_w.clone().detach().cpu()
                gts, dts = cm.sum(2), cm.sum(1)  # (N, K1), (N, K2)
                new_dts = (dts.unsqueeze(1) * perm_w.unsqueeze(0)).sum(-1)  # (N, K1)
                isct = (cm * perm_w.unsqueeze(0)).sum(-1)  # (N, K1)
                union = new_dts + gts - isct  # (N, K1)
                count = (union > 0).sum(0).clamp(min=1)  # (K1,)
                union.clamp_(min=1)
                IoU = (isct / union).sum(0) / count  # (K1,)
                mIoU_table[bias][num_class] = IoU.mean().cpu()

                # convert original label to permuted label
                y = vizs[bias][idx].unsqueeze(1)
                y_bl = torch.zeros(y.shape[0], num_class).scatter_(1, y, 1).cuda()
                y = torch.matmul(y_bl, perm_w.float().permute(1, 0)).argmax(1).cpu()
                perm_vizs[bias].append(y.detach())

                del gts, dts, new_dts, isct, union, count, IoU, y_bl
                torch.cuda.empty_cache()
            disp = [bu(segviz_torch(y.view(*shape)), 256) for y in vizs[bias]]
            vutils.save_image(
                torch.cat(disp + [viz_image]),
                f"{save_prefix}_{bias}_origviz.png",
                nrow=10,
            )
            perm_vizs[bias].append(vizs[bias][-1])
            disp = [bu(segviz_torch(y.view(*shape)), 256) for y in perm_vizs[bias]]
            vutils.save_image(
                torch.cat(disp + [viz_image]),
                f"{save_prefix}_{bias}_permviz.png",
                nrow=10,
            )
            torch.save(perm_weights, f"{save_prefix}_{bias}_permweights.pth")

    res = {}
    index_names = [
        "Adjusted Mutual Information",
        "Rand Index",
        "Adjusted Rand Index",
        "Fowlkes Mallows Index",
        "Homogeneity",
        "Completeness",
        "V Measure Score",
        "mIoU",
    ]
    for bias, cms in cmat.items():
        res[bias] = {}
        xs = []
        metrics = {n: [] for n in index_names}
        for num_class in tqdm(cluster_numbers[bias]):
            if num_class not in cms:
                continue
            cm = cms[num_class].double().sum(0)
            mIoU = mIoU_table[bias][num_class]
            cm = cmat[bias][num_class]
            n_samples = cm.shape[0] * cm[0].sum()
            contingency = cm.sum(0).numpy()
            ri = rand_index(n_samples, contingency)
            ari = adjusted_rand_index(n_samples, contingency)
            fmi = fowlkes_mallows_index(n_samples, contingency)
            hs, cs, vms = homogeneity_score(contingency)
            contingency = (contingency / 1000).astype("int64")
            n_samples = contingency.sum()
            ami = adjusted_mutual_info(n_samples, contingency)
            scores = [ami, ri, ari, fmi, hs, cs, vms, mIoU]
            for n, v in zip(index_names, scores):
                metrics[n].append(float(v))
            xs.append(int(num_class))
        res[bias]["Clusters"] = xs
        res[bias].update(metrics)

    for bias_usage, result in res.items():
        with open(f"{save_prefix}_{bias_usage}.json", "w", encoding="ascii") as fp:
            json.dump(result, fp, indent=4)


if __name__ == "__main__":
    main()
