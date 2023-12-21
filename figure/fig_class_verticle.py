"""Draw the KLiSH results figure (verticle) in the paper."""
# pylint: disable=wrong-import-position,wrong-import-order,multiple-imports,invalid-name,line-too-long
import argparse
import json
from torchvision import utils as vutils
import torch
from functools import partial
from models import helper
from lib.op import bu, cat_mut_iou
from lib.misc import set_cuda_devices
from lib.visualizer import segviz_torch
from lib.cluster import KASP, LinearClassifier


def consistent_label(labels):
    """Make the label consistent."""
    M, N, H, W = labels.shape
    new_labels = labels.clone()
    color_idx = labels[0].max() + 1
    for i in range(1, labels.shape[0]):
        IoU = cat_mut_iou(
            labels[i].view(N, -1),
            labels[i - 1].view(N, -1))[0]
        for j in range(IoU.shape[0]):
            max_ind = IoU[j].argmax()
            mask = labels[i] == j
            if IoU[j, max_ind] > 0.7:
                new_idx = int(new_labels[i - 1][mask].mode().values)
            else:
                new_idx = color_idx
                color_idx += 1
            new_labels[i][mask] = new_idx
    return new_labels


def visualize(image, feat, algs):
    """Draw the figure.
    Args:
        algs: The collection of algorithms. The first sery of the first
              algorithm is used as the reference for label colorization.
    """
    N, H, W, C = feat.shape
    feat = feat.view(-1, C)
    image = bu(image, (H, W))
    res = []
    for alg_sery in algs:
        alg_res = []
        for alg in alg_sery:
            if isinstance(alg, LinearClassifier):
                alg_res.append(alg(feat).argmax(1).cpu())
            else:
                alg_res.append(alg(feat).cpu())
        res.append(alg_res)

    ref_label = res[0][0].clone().detach()
    vizs = []
    for sery_res in res:
        temp_label = torch.stack([ref_label] + sery_res).view(-1, N, H, W)
        c_label = consistent_label(temp_label)[1:].view(-1, H, W)
        vizs.append(segviz_torch(c_label).view(-1, N, 3, H, W))

    cls_img = []
    D = 10
    ones = torch.ones((1, 3, image.shape[2], D))
    for i in range(image.shape[0]):
        disp = [image[i:i+1], ones, ones]
        for idx, viz in enumerate(vizs):
            for j in range(viz.shape[0]):
                disp.extend([viz[j:j+1, i], ones])
            disp.append(ones)
            if idx + 1 == len(vizs):
                disp = disp[:-2]
        cls_img.append(torch.cat(disp, 3)[0])
    return cls_img


def main():
    """Main entrance."""
    train_seed = 1990
    cuts = {
        "stylegan2_ffhq": [50, 30],
        "stylegan2_car":  [50, 20],
    }[args.G_name]

    ahc_file = torch.load(f"{args.expr}/ahc/{args.G_name}_ltrunc-wp_iauto_N16_S64_arccos_{train_seed}.pth")
    ahc_slse = LinearClassifier.load_as_lc(None, ahc_file, cuts)
    ahc_slse = [ahc_slse["nobias"][k] for k in cuts]

    klish_wfile = torch.load(f"{args.expr}/iou/{args.G_name}_iauto_b0_iou_ovrsvc-l2_{train_seed}_tree.pth")
    if args.G_name == "stylegan2_car":
        klish_slse = LinearClassifier.load_as_lc(None, klish_wfile, [50, 34])
        klish_slse = [klish_slse["nobias"][k] for k in [50, 34]]
    else:
        klish_slse = LinearClassifier.load_as_lc(None, klish_wfile, cuts)
        klish_slse = [klish_slse["nobias"][k] for k in cuts]

    kmeans_file = torch.load(f"{args.expr}/kmeans/{args.G_name}_ltrunc-wp_iauto_N256_S256_{train_seed}.pth")
    kmeans_all_slse = LinearClassifier.load_as_lc(
        kmeans_file["euclidean"], kmeans_file["arccos"],
        [200, 100] + cuts)
    kmeans_slse = [kmeans_all_slse["nobias"][k] for k in [100] + cuts]

    sc_file = f"{args.expr}/sc/{args.G_name}_ltrunc-wp_iauto_N256_S256_{train_seed}.json"
    with open(sc_file, "r", encoding="ascii") as f:
        sc_dic = json.load(f)
    sc_alg = KASP(kmeans_all_slse["nobias"][200])
    sc_alg.restore(sc_dic["arccos"])
    sc_funcs = [partial(sc_alg.predict, n_cluster=k) for k in cuts]

    cls_img = visualize(image, feat,
                        [kmeans_slse, ahc_slse, sc_funcs, klish_slse])

    for i, img in enumerate(cls_img):
        if "stylegan2_car" == args.G_name:
            img = img[:, 32:-32]
        vutils.save_image(
            img, f"results/plot/{args.G_name}_mergeclass{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--out-dir", default="results/plot")
    # architecture
    parser.add_argument("--G-name", default="all")
    parser.add_argument("--w-path",
                        default="expr/cluster/klish", type=str)
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--N", default=4, type=int)
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    torch.set_grad_enabled(False)

    if args.G_name == "all":
        G_names = [
            "stylegan2_ffhq", "stylegan2_car"]
    else:
        G_names = [args.G_name]

    for G_name in G_names:
        args.G_name = G_name
        G = helper.build_generator(args.G_name).net.cuda()
        if "ffhq" in args.G_name or "car" in args.G_name:
            wps = []
            # do not replicate other figure
            for i in range(25, 25 + args.N):
                wps.append(torch.load(
                    f"data/{args.G_name}_fewshot/latent/wp_{i:02d}.npy"))
            wps = torch.cat(wps).cuda()
            image, feat = helper.sample_layer_feature(
                                    G, args.N, wps=wps)
        else:
            image, feat = helper.sample_layer_feature(
                                    G, args.N)
        image = bu(image, feat.shape[2])
        main()
        del G, feat, image
        torch.cuda.empty_cache()
