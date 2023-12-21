"""Calculate the class feature centroids.
"""
import os
import argparse
import torch

from tqdm import tqdm
from predictors.face_segmenter import FaceSegmenter
from predictors.scene_segmenter import SceneSegmenter
from lib.misc import set_cuda_devices
from lib.evaluate import evaluate_SE, write_results
from lib.op import bu
from models import helper
from models.semantic_extractor import SimpleLSE


def main():
    """Entrace."""
    parser = argparse.ArgumentParser()
    # Architecture setting
    parser.add_argument(
        "--latent-strategy",
        type=str,
        default="trunc-wp",
        choices=["notrunc-mixwp", "trunc-wp", "notrunc-wp"],
        help="notrunc-mixwp: mixed W+ without truncation. trunc-wp: W+ with truncation. notrunc-wp: W+ without truncation.",
    )
    parser.add_argument(
        "--G-name", type=str, default="stylegan2_ffhq", help="The model type of generator"
    )
    parser.add_argument(
        "--expr",
        type=str,
        default="expr/class_centroids",
        help="The experiment directory.",
    )
    parser.add_argument(
        "--gpu-id", type=str, default="0", help="GPUs to use. (default: %(default)s)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Total number of samples."
    )
    # evaluation settings
    parser.add_argument(
        "--eval", type=int, default=1, help="Whether to evaluate after training."
    )
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)
    torch.set_grad_enabled(False)

    DIR = f"{args.expr}/{args.G_name}_LSE_ls{args.latent_strategy}_lwnone"
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    G = helper.build_generator(args.G_name)
    is_face = "celebahq" in args.G_name or "ffhq" in args.G_name
    if is_face:
        P = FaceSegmenter()
    else:
        P = SceneSegmenter(model_name=args.G_name)
    print(f"=> Segmenter has {P.n_class} classes")

    features = G(G.easy_sample(1))["feature"]
    dims = [s.shape[1] for s in features]
    layers = list(range(len(dims)))
    SE = helper.build_semantic_extractor(
        lw_type="none",
        model_name="LSE",
        n_class=P.n_class,
        dims=dims,
        layers=layers,
        use_bias=True,
    )

    wps = helper.sample_latent(G.net, args.n_samples, args.latent_strategy)
    resolution = 512 if is_face else 256
    by = torch.zeros(P.n_class, resolution**2).to(wps).double()
    acc_count = torch.zeros(P.n_class).to(wps.device).double()
    centroids = []
    for wp in tqdm(wps, total=wps.shape[0]):
        image, features = helper.generate_image(G.net, wp.unsqueeze(0), generate_feature=True)
        y = P(bu(image, resolution).clamp(-1, 1))[0].view(-1)  # (H, W)
        by.fill_(0).scatter_(0, y.unsqueeze(0), 1)  # (M, HW)
        count = by.sum(1)
        alpha = (count / (acc_count + count).clamp(min=1)).unsqueeze(1)
        acc_count += count
        by /= count.unsqueeze(1).clamp(min=1)
        for i, layer_idx in enumerate(layers):
            feat = features[layer_idx]
            v = bu(feat, resolution)[0].view(-1, y.shape[0]).double()  # (C, HW)
            c = torch.matmul(by, v.T)  # (M, C)
            if len(centroids) <= i:
                centroids.append(torch.zeros_like(c))
            centroids[i] = centroids[i] * (1 - alpha) + c * alpha

    SE.requires_grad_(False)
    for i, layer in enumerate(SE.extractor):
        w = centroids[i]  # (M, C)
        b = -0.5 * torch.square(w).sum(1)
        layer.weight[..., 0, 0].copy_(w)
        layer.bias.copy_(b)
    helper.save_semantic_extractor(SE, f"{DIR}/{args.G_name}_LSE.pth")

    if args.eval == 1:
        if not os.path.exists("results/class_centroids/"):
            os.makedirs("results/class_centroids/")
        res_dir = DIR.replace(args.expr, "results/class_centroids/")
        mIoU, c_ious = evaluate_SE(
            SE, G.net, P, resolution, args.n_samples, args.latent_strategy
        )
        write_results(f"{res_dir}_els{args.latent_strategy}.txt", mIoU, c_ious)


if __name__ == "__main__":
    main()
