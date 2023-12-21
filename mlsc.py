"""Maximum Linear Separability Clustering."""
# pylint: disable=invalid-name,line-too-long
import argparse
import os
import glob
import torch
from datetime import datetime

from lib.cluster import MLSClustering
from lib.misc import set_cuda_devices
from models import helper


def main():
    """Run maximum linear separability clustering."""
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument(
        "--expr", default="expr/cluster", help="The directory of experiments."
    )
    parser.add_argument("--name", default="klish", help="The name of the experiment.")
    # architecture
    parser.add_argument(
        "--G-name",
        default="stylegan2_ffhq",
        help="The name of generator, should be in " "models/pretrained/pytorch folder.",
    )
    parser.add_argument(
        "--layer-idx",
        default="auto",
        type=str,
        help="The layer indice used for collecting features, use"
        " ``auto'' for using the default layer selection process.",
    )
    parser.add_argument(
        "--k-init", default=100, type=int, help="The initial K-means clusters."
    )
    parser.add_argument(
        "--metric",
        default="heuristic",
        type=str,
        help="The merging metric.",
    )
    parser.add_argument(
        "--objective",
        default="ovrsvc-l2",
        type=str,
        help="The objective of training. The first term is one "
        "of ovrsvc, mcsvc, mcmld. The second term is one of"
        "l1, l2.",
    )
    parser.add_argument(
        "--svm-coef", default=5000.0, type=float, help="The coefficient of SVM."
    )
    parser.add_argument(
        "--l1-coef", default=0.0, type=float, help="The coefficient of SVM."
    )
    parser.add_argument(
        "--l2-coef", default=1.0, type=float, help="The coefficient of SVM."
    )
    parser.add_argument(
        "--use-bias",
        default=1,
        type=int,
        help="Whether to use bias.",
    )
    parser.add_argument(
        "--n-samples", default=256, type=int, help="The number of image samples."
    )
    parser.add_argument("--class-idx", default=-1, type=int, help="The index of class.")
    parser.add_argument(
        "--class-number", default=-1, type=int, help="Total number of classes."
    )
    parser.add_argument(
        "--resolution", default=256, type=int, help="The image resolution."
    )
    # training
    parser.add_argument(
        "--latent-type",
        default="trunc-wp",
        type=str,
        choices=["trunc-wp", "wp", "mix-wp"],
        help="The latent type of StyleGANs.",
    )
    parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument(
        "--max-iter",
        default=20,
        type=int,
        help="Maximum SVM training L-BFGS iterations.",
    )
    parser.add_argument(
        "--n-viz", default=16, type=int, help="The number of visualizing images."
    )
    parser.add_argument(
        "--skip-existing", default=1, type=int, help="Whether to skip existing files."
    )
    parser.add_argument("--seed", default=1990, type=int)
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    devices = list(range(n_gpu))

    if not os.path.exists(f"{args.expr}/{args.name}"):
        os.makedirs(f"{args.expr}/{args.name}")

    prefix = f"{args.expr}/{args.name}/{args.G_name}_i{args.layer_idx}_b{args.use_bias}_{args.metric}_{args.objective}_{args.seed}"
    if os.path.exists(f"{prefix}_tree.pth") and args.skip_existing:
        print(f"!> Skip {prefix}")
        exit()

    kmeans_prefix = f"{args.expr}/kmeans/{args.G_name}_l{args.latent_type}_i{args.layer_idx}*{args.seed}"

    print(f"=> Preparing {args.n_samples} samples in {args.resolution} resolution...")
    if args.class_idx > -1 and args.class_number > -1:
        print(f"=> Conditioning on class {args.class_idx} of {args.class_number}")
        c = torch.zeros(args.n_samples // n_gpu, args.class_number)
        c[:, args.class_idx] = 1
        cs = [c.clone().detach() for d in range(n_gpu)]
        prefix = f"{prefix}_c{args.class_idx}"
        kmeans_prefix += f"_c{args.class_idx}"
    else:
        print("=> No class conditioning.")
        cs = None
    mimage, mfeat = helper.sample_generator_feature(
        args.G_name,
        latent_type=args.latent_type,
        layer_idx=args.layer_idx,
        n_samples=args.n_samples,
        cs=cs,
        size=args.resolution,
        device_ids=devices,
        seed=args.seed + 10,  # different from K-means' data
    )
    B, H, W, C = mfeat[0].shape
    for i in range(n_gpu):
        mfeat[i] = mfeat[i].view(-1, C)
    print(f"=> GPU Feature: {mfeat[0].shape} ({B}, {H}, {W}, {C})")

    kmeans_file = glob.glob(f"{kmeans_prefix}.pth")[0]
    print(f"=> Initialize from K-means results: {kmeans_file}")
    kmeans_res = torch.load(kmeans_file)

    start_time = datetime.today()
    if args.use_bias == 1:
        mx2 = [torch.square(x.norm(p=2, dim=1)) for x in mfeat]
        w_init = kmeans_res["euclidean"][args.k_init]["weight"].cuda()
    else:
        mx2 = None
        w_init = kmeans_res["arccos"][args.k_init]["weight"].cuda()
    alg = MLSClustering(
        w_init=w_init,
        use_bias=(args.use_bias == 1),
        metric=args.metric,
        objective=args.objective,
        n_viz=args.n_viz,
        image_shape=[H, W],
        max_iter=args.max_iter,
        svm_coef=args.svm_coef,
        l1_coef=args.l1_coef,
        l2_coef=args.l2_coef,
        save_prefix=prefix,
    )
    alg.fit(mx=mfeat, mx2=mx2, mimage=mimage)
    seconds = (datetime.today() - start_time).total_seconds()
    with open(f"{prefix}.txt", "w", encoding="ascii") as f:
        f.write(str(seconds))


if __name__ == "__main__":
    main()
