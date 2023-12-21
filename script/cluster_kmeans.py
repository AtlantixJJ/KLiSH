"""KMeans clustering the features of generators."""
# pylint: disable=invalid-name,line-too-long
import os
import argparse
import torch
from datetime import datetime

from models import helper
from lib.cluster import MultiGPUKMeansPP
from lib.op import bu
from lib.misc import set_cuda_devices, imwrite
from lib.visualizer import visualize_segmentation, plot_dict


def main():
    """K-means++ clustering."""
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--name", default="kmeans")
    # architecture
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--layer-idx", default="auto", type=str)
    parser.add_argument(
        "--n-samples", default=256, type=int, help="The number of image samples."
    )
    parser.add_argument(
        "--resolution", default=256, type=int, help="The image resolution."
    )
    parser.add_argument("--ALL-K", default="100,40,30,20", type=str)
    parser.add_argument("--dist", default="euclidean,arccos", type=str)
    # training
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
    parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--skip-existing", default=1, type=int)
    parser.add_argument("--seed", default=1990, type=int)
    parser.add_argument("--class-idx", default=-1, type=int, help="The index of class.")
    parser.add_argument(
        "--class-number", default=-1, type=int, help="Total number of classes."
    )
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    args.gpu_id = list(range(n_gpu))
    torch.set_grad_enabled(False)
    if not os.path.exists(f"{args.expr}/{args.name}"):
        os.makedirs(f"{args.expr}/{args.name}")

    name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_N{args.n_samples}_S{args.resolution}_{args.seed}"
    kmeans_prefix = f"{args.expr}/{args.name}/{name}"
    if os.path.exists(f"{kmeans_prefix}.pth") and args.skip_existing:
        print(f"=> Skip because {kmeans_prefix}.pth exists.")
        return

    print(f"=> Preparing {args.n_samples} samples in {args.resolution} resolution...")
    ALL_K = [int(x) for x in args.ALL_K.split(",")]
    B = args.n_samples // len(args.gpu_id)
    if args.class_idx > -1 and args.class_number > -1:
        print(f"=> Conditioning on class {args.class_idx} of {args.class_number}")
        c = torch.zeros(args.n_samples // n_gpu, args.class_number)
        c[:, args.class_idx] = 1
        cs = [c.clone().detach() for d in range(n_gpu)]
        kmeans_prefix = f"{kmeans_prefix}_c{args.class_idx}"
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
        device_ids=args.gpu_id,
        seed=args.seed,
    )
    B, H, W, C = mfeat[0].shape
    N_viz = min(args.n_viz, B)
    mfeat_norm2 = []
    for i in range(n_gpu):
        norm = mfeat[i].norm(p=2, dim=3, keepdim=True)
        mfeat_norm2.append(torch.square(norm.view(-1)))  # (L, 1)
        mfeat[i] = mfeat[i].view(-1, C)
    viz_image = bu(mimage[0][:N_viz], (H, W))
    print(f"=> GPU Feature: {mfeat[0].shape} ({B}, {H}, {W}, {C})")

    res = {"euclidean": {}, "arccos": {}}
    for dist in args.dist.split(","):
        if dist == "arccos":
            for i in range(n_gpu):
                mfeat[i] /= torch.sqrt(mfeat_norm2[i].unsqueeze(1)).clamp(min=1e-5)
        for k in ALL_K:
            start_time = datetime.today()
            alg = MultiGPUKMeansPP(k, dist, seed=args.seed)
            alg.fit(mfeat, mfeat_norm2, verbose=True)
            seconds = (datetime.today() - start_time).total_seconds()
            time_fpath = f"{kmeans_prefix}_{dist}_{k:03d}.txt"
            with open(time_fpath, "w", encoding="ascii") as f:
                f.write(str(seconds))
            res[dist][k] = alg.param
            viz_feat = mfeat[0].view(B, H, W, C)[:N_viz].view(-1, C).cpu()
            label = alg.predict(viz_feat).view(N_viz, H, W)
            viz = visualize_segmentation(viz_image, label)
            imwrite(f"{kmeans_prefix}_{dist}_{k:03d}.png", viz)
            plot_dict(alg.record, f"{kmeans_prefix}_{dist}_{k:03d}_record.png")
            del viz_feat, label, viz, alg
            torch.cuda.empty_cache()
    torch.save(res, f"{kmeans_prefix}.pth")
    # key1: euclidean or arccos; key2: The cluster number;
    # key3: W, bias


if __name__ == "__main__":
    main()
