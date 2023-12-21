"""Compute the ARI of KLiSH and K-means.
"""
# pylint: disable=invalid-name
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from predictors.scene_segmenter import SceneSegmenter
from predictors.face_segmenter import FaceSegmenter
from models.helper import (
    build_generator,
    auto_layer_selection,
    sample_latent,
    generate_image,
)
from lib.misc import set_cuda_devices
from lib.cluster import KASP, load_as_slse
from lib.visualizer import segviz_torch
from lib.op import bu


def label_perm_func(P, w):
    """Generate a permutation function."""

    def func(x):
        seg = P(x)
        label = seg.argmax(1)
        bin_label = torch.zeros_like(seg).scatter_(1, label.unsqueeze(1), 1)
        perm_label = F.conv2d(bin_label, w)  # .argmax(1)
        return perm_label

    return func


def find_label_permutation(mat):
    """
    Args:
      mat: (K1, K2) contingency matrix. K1 < K2.
    Returns:
      w : (K1 + X, K2) permutation matrix, X is unmatched extra classes.
    """
    gts, dts = mat.sum(1), mat.sum(0)
    w = torch.zeros_like(mat)
    indice = list(range(w.shape[1]))
    while len(indice) > 0:
        N = len(indice)
        ext_w = w.unsqueeze(-1).repeat(1, 1, mat.shape[0] * N)
        for i in range(mat.shape[0]):
            for j, real_j in enumerate(indice):
                ext_w[i, real_j, i * N + j] = 1
        ext_isct_sum = (mat.unsqueeze(-1) * ext_w).sum(1)  # (K1, M)
        ext_union_sum = (dts.view(1, -1, 1) * ext_w).sum(1)  # (K1, M)
        ext_union_sum = ext_union_sum - ext_isct_sum + gts.view(-1, 1)
        ext_valid = (ext_union_sum > 1).float()
        ext_IoU = (ext_valid * ext_isct_sum) / ext_union_sum.clamp(min=1)

        ind = int(ext_IoU.mean(0).cpu().argmax())
        gt_idx, dt_idx = ind // N, ind % N
        w[gt_idx, indice[dt_idx]] = 1
        del indice[dt_idx]
    return w.float().cpu()


def eval_kasp(feature, merge_record, gt_bl, cmat, key, i, resolution):
    """Helper function"""
    kmeans_label = merge_record.kmeans_model(feature).argmax(1).view(-1)
    orig_dtbl = torch.zeros(resolution**2, merge_record.K).cuda()
    orig_dtbl.scatter_(1, kmeans_label.unsqueeze(1), 1)
    for k in range(2, 101):
        pmat = merge_record.merge_matrix[k]
        dt_bl = torch.matmul(orig_dtbl, pmat.cuda())
        cm = torch.matmul(gt_bl, dt_bl)
        cmat[key][k].add_(cm.cpu().long())


def main():
    """Calculate ARI of a clustering algorithm."""
    save_name = f"{args.G_name}_{args.eval_name}_{args.train_seed}"
    if os.path.exists(f"{save_name}_nobias.pth") and os.path.exists(
        f"{save_name}_bias.pth"
    ):
        print(f"=> {save_name} exists, skip.")
        exit(0)

    generator = build_generator(
        args.G_name, randomize_noise=True, truncation_psi=0.5
    ).net.cuda()
    features = generate_image(
        generator, sample_latent(generator, 1), generate_feature=True
    )[1]
    shapes = [f.shape for f in features]
    layers = auto_layer_selection(shapes)
    is_face = "celebahq" in args.G_name or "ffhq" in args.G_name
    resolution = 512 if is_face else 256
    segmenter = FaceSegmenter() if is_face else SceneSegmenter(model_name=args.G_name)
    print(f"=> Segmenter: {segmenter.n_class} classes: {segmenter.labels}")
    slse = load_as_slse(
        args.eval_name, layers, resolution, args.expr, args.G_name, args.train_seed
    )
    gt_bl = torch.zeros(segmenter.n_class, resolution**2).to("cuda:0")
    dt_bl = torch.zeros(resolution**2, 100).to("cuda:0")
    if args.eval_name == "kmeans":
        cluster_numbers = list(range(10, 101, 10))
    else:
        cluster_numbers = list(range(2, 101))
    cmat = {
        key: {k: torch.zeros(segmenter.n_class, k).long() for k in cluster_numbers}
        for key in slse
    }
    if args.eval_seed > 0:
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)
    with torch.no_grad():
        wps = sample_latent(generator, args.n_samples, args.latent_type)
        for i in tqdm(range(args.n_samples)):
            image, feature = generate_image(
                generator, wps[i : i + 1], generate_feature=True
            )
            gt = segmenter(image, resolution).long().view(-1)
            gt_bl.fill_(0).scatter_(0, gt.unsqueeze(0), 1)
            for key, merge_record in slse.items():
                if isinstance(merge_record, KASP):
                    eval_kasp(feature, merge_record, gt_bl, cmat, key, i, resolution)
                    continue
                for k in cluster_numbers:
                    dt = merge_record[k](feature).argmax(1).view(-1)
                    dt_bl.fill_(0).scatter_(1, dt.unsqueeze(1), 1)
                    cm = torch.matmul(gt_bl, dt_bl[:, :k])
                    cmat[key][k].add_(cm.cpu().long())

        res = {}
        wp = sample_latent(generator, 1)
        image, feature = generate_image(generator, wp, generate_feature=True)
        gt = segmenter(image, resolution).long().view(-1)
        for key in cmat:
            res[key] = {}
            viz_perms, viz_origs = [], []
            alg = slse[key]
            for num_class, cm in tqdm(cmat[key].items()):
                perm_w = find_label_permutation(cm.cuda().float())
                res[key][num_class] = perm_w.cpu()
                if isinstance(alg, KASP):
                    orig = alg.kmeans_model(feature)
                    orig_dtbl = torch.zeros_like(orig)
                    orig_dtbl.scatter_(1, orig.argmax(1).unsqueeze(1), 1)
                    pmat = alg.merge_matrix[num_class]
                    dt = F.conv2d(
                        orig_dtbl, pmat.permute(1, 0).cuda().unsqueeze(-1).unsqueeze(-1)
                    ).argmax(1)
                    pmat = torch.matmul(perm_w, pmat.permute(1, 0))
                    pmat = pmat.unsqueeze(-1).unsqueeze(-1).to(orig)
                    perm_dt = F.conv2d(orig_dtbl, pmat).argmax(1)
                else:
                    dt_seg = alg[num_class](feature)
                    dt = dt_seg.argmax(1)
                    dt_bl = torch.zeros_like(dt_seg)
                    dt_bl.scatter_(1, dt.unsqueeze(1), 1)
                    pmat = perm_w.unsqueeze(-1).unsqueeze(-1).to(dt_seg)
                    perm_dt = F.conv2d(dt_bl, pmat).argmax(1)
                perm_dt_map = perm_dt.view(1, 1, resolution, resolution).cpu()
                orig_dt_map = dt.view(1, 1, resolution, resolution).cpu()
                viz_origs.append(bu(segviz_torch(orig_dt_map), 256))
                viz_perms.append(bu(segviz_torch(perm_dt_map), 256))

            gt_map = gt.view(1, 1, resolution, resolution).cpu()
            viz_origs.append(bu((image.clamp(-1, 1) + 1) / 2, 256).cpu())
            viz_perms.append(bu(segviz_torch(gt_map), 256))
            vutils.save_image(
                torch.cat(viz_origs),
                f"expr/label_perm/{save_name}_{key}_origviz.png",
                nrow=10,
            )
            vutils.save_image(
                torch.cat(viz_perms),
                f"expr/label_perm/{save_name}_{key}_permviz.png",
                nrow=10,
            )

    for bias_usage, result in res.items():
        torch.save(result, f"expr/label_perm/{save_name}_{bias_usage}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", default="expr/cluster", help="The directory of experiments."
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
        "--n-samples", default=100, type=int, help="The number of samples."
    )
    parser.add_argument(
        "--latent-type",
        default="trunc-wp",
        type=str,
        choices=["trunc-wp", "wp", "mix-wp"],
        help="The latent type of StyleGANs.",
    )
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--train-seed", default=1990, type=int)
    parser.add_argument(
        "--eval-seed", default=2021, type=int, help="The seed for this evaluation."
    )
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    devices = list(range(n_gpu))

    if not os.path.exists("expr/label_perm"):
        os.makedirs("expr/label_perm")

    main()
