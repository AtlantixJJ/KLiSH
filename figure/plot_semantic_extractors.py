"""Plot the segmentation results from semantic extractors.
"""
import argparse
import torch
from torchvision import utils as vutils
from tqdm import tqdm

from models.helper import (
    build_generator,
    load_semantic_extractor,
    generate_image,
    sample_latent,
)
from predictors.helper import build_predictor
from models.semantic_extractor import SimpleLSE
from lib.op import bu
from lib.misc import set_cuda_devices
from lib.visualizer import segviz_torch


def main(G_name):
    """Entrance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="expr")
    parser.add_argument("--name", default="klish")
    parser.add_argument("--layer-idx", default="auto", type=str)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--latent-type", default="trunc-wp", type=str)
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--seed", default=1113, type=int)
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)
    torch.set_grad_enabled(False)

    S = args.resolution
    LSEs = {}
    lse_fpath = (
        "expr/semantics/{G_name}_LSE_lnormal_lstrunc-wp_lwnone_lr0.001/{G_name}_LSE.pth"
    )
    lse_fpath = lse_fpath.format(G_name=G_name)
    LSEs["LSE"] = load_semantic_extractor(lse_fpath).cuda()
    centroid_fpath = (
        "expr/class_centroids/{G_name}_LSE_lstrunc-wp_lwnone/{G_name}_LSE.pth"
    )
    centroid_fpath = centroid_fpath.format(G_name=G_name)
    LSEs["Class Centroid"] = load_semantic_extractor(centroid_fpath).cuda()

    G = build_generator(G_name).net.cuda()
    P = build_predictor("face_seg").cuda()

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    wps = sample_latent(G, args.N, args.latent_type).cpu()

    disp = []
    for i in tqdm(range(args.N)):
        image, feat = generate_image(G, wps[i : i + 1].cuda(), generate_feature=True)
        image = bu(image.clamp(-1, 1), S)
        gt_label_viz = segviz_torch(P(image))
        image = (image + 1) / 2
        disp.extend([image.cpu(), gt_label_viz.cpu()])
        for _, net in LSEs.items():
            if isinstance(net, SimpleLSE):
                label = net(feat, S).argmax(1)
            else:
                label = net(feat, S)[-1].argmax(1)
            disp_label_viz = segviz_torch(label)
            disp.append(disp_label_viz.cpu())
    vutils.save_image(
        torch.cat(disp),
        f"results/plot/semantic_extraction_{G_name}.png",
        nrow=4,
        padding=20,
        pad_value=255,
    )


if __name__ == "__main__":
    for G_name in ["stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq"]:
        main(G_name)
