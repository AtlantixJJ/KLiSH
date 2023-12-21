"""Show the result of clustering.
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


def eval_kasp(
    algs,
    feature,
    dt_bl,
    cluster_numbers,
    vizs,
    idx,
):
    """Evaluate KASP clustering."""
    dt_init = algs.kmeans_model(feature).argmax(1).view(-1)
    dt_bl.fill_(0)
    dt_bl.scatter_(1, dt_init.view(-1).unsqueeze(1), 1)
    for num_class in cluster_numbers:
        pmat = algs.merge_matrix[num_class]
        final_dt_bl = torch.matmul(dt_bl, pmat.cuda()).clamp(max=1)
        dt = final_dt_bl.argmax(1)
        vizs.append(dt.detach().cpu())


def eval_slse(
    algs,
    feature,
    dt_bl,
    cluster_numbers,
    vizs,
    idx,
):
    """Evaluate SLSE model."""
    for num_class in cluster_numbers:
        seg = algs[num_class](feature)
        dt = seg.argmax(1).view(-1)
        dt_bl.fill_(0).scatter_(1, dt.unsqueeze(1), 1)
        vizs.append(dt.detach().cpu())


def main():
    """Calculate ARI of a clustering algorithm."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", default="expr/cluster", help="The directory of experiments."
    )
    parser.add_argument(
        "--out-dir",
        default="expr/show_cluster",
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
        "--n-samples", default=2, type=int, help="The number of samples."
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
        "--eval-seed", default=2022, type=int, help="The seed for this evaluation."
    )
    args = parser.parse_args()
    n_gpu = lib.misc.set_cuda_devices(args.gpu_id)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    save_prefix = f"{args.out_dir}/{args.G_name}_{args.eval_name}_{args.train_seed}"
    if os.path.exists(f"{save_prefix}_nobias.json") and os.path.exists(
        f"{save_prefix}_bias.json"
    ):
        print(f"=> {save_prefix} exists.")
        return

    generator = helper.build_generator(
        args.G_name, randomize_noise=True, truncation_psi=0.5
    ).net.cuda()
    wps = helper.sample_latent(generator, 1)
    features = helper.generate_image(generator, wps, generate_feature=True)[1]
    shapes = [f.shape for f in features]
    layers = helper.auto_layer_selection(shapes)
    is_face = "celebahq" in args.G_name or "ffhq" in args.G_name
    resolution = 512 if is_face else 256
    if len(args.in_dir) > 0:
        slse = load_as_slse(
            args.eval_name,
            layers,
            resolution,
            args.in_dir,
            args.G_name,
            args.train_seed,
            custom_path=True,
        )
    else:
        slse = load_as_slse(
            args.eval_name, layers, resolution, args.expr, args.G_name, args.train_seed
        )
    if "kasp" in args.eval_name:
        cluster_numbers = {key: list(range(2, 101)) for key in slse}
    else:
        cluster_numbers = {key: list(slse[key].keys()) for key in slse}
    shape = [-1, resolution, resolution]
    max_cluster = max(list(cluster_numbers.values())[0])
    dt_bl = torch.zeros(resolution**2, max_cluster).cuda()
    if args.eval_seed > 0:
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)
    with torch.no_grad():
        wps = helper.sample_latent(generator, args.n_samples, args.latent_type)
        for i in tqdm(range(args.n_samples)):
            image, feature = helper.generate_image(generator, wps[i : i + 1], generate_feature=True)
            viz_image = bu((image.clamp(-1, 1) + 1) / 2, 256).cpu()
            for bias, algs in slse.items():
                vizs = []
                func_args = [algs, feature, dt_bl, cluster_numbers[bias], vizs, i]
                if isinstance(algs, KASP):
                    eval_kasp(*func_args)
                else:
                    eval_slse(*func_args)
                disp = [bu(segviz_torch(y.view(*shape)), 256) for y in vizs]
                vutils.save_image(
                    torch.cat(disp + [viz_image]),
                    f"{save_prefix}_{bias}_{i:02d}_origviz.png",
                    nrow=10,
                )


if __name__ == "__main__":
    main()
