"""Draw the KLiSH results figure in the paper."""
# pylint: disable=wrong-import-position,wrong-import-order,multiple-imports,invalid-name,line-too-long
from torchvision import utils as vutils
import argparse, torch, json
import numpy as np
from functools import partial
from scipy.optimize import linear_sum_assignment

from models import helper
from lib.op import bu, cat_mut_iou
from lib.misc import formal_name, set_cuda_devices
from lib.visualizer import segviz_torch
from lib.cluster import KASP, LinearClassifier


def consistent_label(labels):
    """Make the label consistent."""
    new_labels = labels.clone()
    N = labels.shape[1]
    for i in range(1, labels.shape[0]):
        IoU = cat_mut_iou(labels[0].view(N, -1), labels[i].view(N, -1))[0]
        old_indice, new_indice = linear_sum_assignment(IoU.cpu().numpy(), True)
        assert IoU.shape[0] >= IoU.shape[1]  # make sure the #clusters is decreasing
        for old_idx, new_idx in zip(old_indice, new_indice):
            new_labels[i][labels[i] == new_idx] = old_idx
    return new_labels


def get_figure_images(feat, algs):
    """Get the figure images and consistent colorization.
    The reference label is the first clustering of the first algorithm.

    Args:
        feat: The raw feature block
        algs: The collection of algorithms. The first sery of the first
              algorithm is used as the reference for label colorization.
    """
    N, H, W, C = feat.shape
    feat = feat.view(-1, C)
    with torch.no_grad():
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
    # shuffle the label to make the figure look better
    rng = np.random.RandomState(1)
    orig_labels = ref_label.unique()
    indice = orig_labels.cpu().numpy()
    rng.shuffle(indice)
    new_ref_label = torch.zeros_like(ref_label)
    for orig, new in zip(orig_labels, indice):
        mask = ref_label == orig
        new_ref_label[mask] = new
    ref_label = new_ref_label

    vizs = []
    for idx, sery_res in enumerate(res):
        temp_label = [ref_label] + sery_res if idx > 0 else sery_res
        temp_label = torch.stack([ref_label] + sery_res).view(-1, N, H, W).cuda()
        c_label = consistent_label(temp_label)[1:].view(-1, H, W)
        vizs.append(segviz_torch(c_label.cpu()).view(-1, N, 3, H, W))
    return vizs


def make_grid(image, vizs, crop_verticle=0, padding_size=10):
    """Make a image grid from the visualized images."""
    cls_img = []
    for i in range(image.shape[0]):
        img = image[i : i + 1]
        if crop_verticle > 0:
            img = img[:, :, crop_verticle:-crop_verticle]
            new_W = int(img.shape[3] ** 2 / img.shape[2])
            img = bu(img, (img.shape[3], new_W))
        ones = torch.ones((1, 3, padding_size, img.shape[3]))
        disp = [img, ones, ones, ones]
        for idx, viz in enumerate(vizs):
            for j in range(viz.shape[0]):
                img = viz[j : j + 1, i]
                if crop_verticle > 0:
                    img = img[:, :, crop_verticle:-crop_verticle]
                    new_W = int(img.shape[3] ** 2 / img.shape[2])
                    img = bu(img, (img.shape[3], new_W))
                disp.extend([img, ones])
            disp.append(ones)
            if idx + 1 == len(vizs):
                disp = disp[:-2]
        cls_img.append(torch.cat(disp, 2)[0])  # (3, LH, W)
    return cls_img


def worker(image, feat, args):
    """Worker for a single feature block."""
    with open("results/tex/cluster_bestindice.json", "r", encoding="ascii") as f:
        best_seed_dic = json.load(f)
        G_name = formal_name(args.G_name)
        best_seed_dic = best_seed_dic[G_name]
    
    coarse_cuts = [30, 20]
    fine_cuts = {
        "stylegan2_ffhq": [30, 20, 26],
        "stylegan2_car": [30, 20, 14],
        "stylegan_celebahq": [30, 20, 30],
        "pggan_celebahq": [30, 20, 26],

        #"ada_cat": [30, 20, 7],
        #"ada_dog": [30, 20, 9],
        #"ada_wild": [30, 20, 22],
        #"ada_metface": [30, 20, 18],
        #"stylegan2_bedroom": [30, 20, 18],
        #"stylegan2_church": [30, 20, 11],
        #"stylegan_bedroom": [30, 20, 16],
        #"stylegan_church": [30, 20, 15],

        "ada_cat": [30, 7],
        "ada_dog": [30, 9],
        "ada_wild": [30, 22],
        "ada_metface": [30, 18],
        "stylegan2_bedroom": [30, 18],
        "stylegan2_church": [30, 11],
        "stylegan_bedroom": [30, 16],
        "stylegan_church": [30, 15],
    }[args.G_name]

    kmeans_cuts = [100, 30, 20]

    ahc_seed = best_seed_dic["AHC (arccos)"]
    ahc_file = torch.load(
        f"{args.expr}/ahc/{args.G_name}_ltrunc-wp_iauto_N32_S64_arccos_{ahc_seed}.pth"
    )
    ahc_slse = LinearClassifier.load_as_lc(None, ahc_file, fine_cuts)["nobias"]
    ahc_slse = [ahc_slse[k] for k in fine_cuts]

    klish_seed = best_seed_dic["KLiSH (bias)"]
    klish_wfile = torch.load(
        f"{args.expr}/klish/{args.G_name}_iauto_b1_heuristic_ovrsvc-l2_{klish_seed}_tree.pth"
    )
    klish_slse = LinearClassifier.load_as_lc(klish_wfile, None, fine_cuts)["bias"]
    klish_slse = [klish_slse[k] for k in fine_cuts]

    kmeans_seed = best_seed_dic["K-means (euclidean)"]
    kmeans_file = torch.load(
        f"{args.expr}/kmeans/{args.G_name}_ltrunc-wp_iauto_N256_S256_{kmeans_seed}.pth"
    )
    kmeans_all_slse = LinearClassifier.load_as_lc(
        kmeans_file["euclidean"], None, kmeans_cuts
    )["bias"]
    kmeans_slse = [kmeans_all_slse[k] for k in kmeans_cuts]

    kasp_seed = best_seed_dic["KASP (euclidean)"]
    kasp_file = (
        f"{args.expr}/kasp/{args.G_name}_ltrunc-wp_iauto_N256_S256_{kasp_seed}.json"
    )
    with open(kasp_file, "r", encoding="ascii") as f:
        kasp_dic = json.load(f)
    kasp_alg = KASP(kmeans_all_slse[100])
    kasp_alg.restore(kasp_dic["euclidean"])
    kasp_funcs = [partial(kasp_alg.predict, n_cluster=k) for k in coarse_cuts]
    print("=> Loading complete.")

    vizs = get_figure_images(feat, [kmeans_slse, kasp_funcs, ahc_slse, klish_slse])
    crop_verticle = image.shape[3] // 8 if args.G_name == "stylegan2_car" else 0
    cls_img = make_grid(image, vizs, crop_verticle)
    prefix = f"results/plot/{args.G_name}_mergeclass"
    for i, img in enumerate(cls_img):
        vutils.save_image(img, f"{prefix}{i}.png")
        print(f"=> Saved to {prefix}{i}.png")


def main():
    """Entrace."""
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--out-dir", default="results/plot")
    # architecture
    parser.add_argument("--G-name", default="stylegan2_ffhq")
    parser.add_argument("--w-path", default="expr/cluster/klish", type=str)
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--N", default=4, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--seed", default=2022, type=int)
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)
    torch.set_grad_enabled(False)

    G = helper.build_generator(args.G_name).net.cuda()
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if "ffhq" in args.G_name or "car" in args.G_name:
        wps = []
        # start from 25 so that it do not replicate other figure
        for i in range(25, 25 + args.N):
            wps.append(torch.load(f"data/{args.G_name}_fewshot/latent/wp_{i:02d}.npy"))
        wps = torch.cat(wps).cuda()
    else:
        wps = helper.sample_latent(G, args.N)
    image, feat = helper.sample_layer_feature(G, args.N, args.resolution, wps=wps)
    image = bu(image, feat.shape[2])
    worker(image, feat, args)
    del G, feat, image
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
