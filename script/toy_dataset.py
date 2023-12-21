"""Experiment on toy datasets."""
# pylint: disable=invalid-name
import os
import sys

sys.path.insert(0, ".")
import argparse
from sklearn import svm
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from lib.cluster import MultiGPUKMeansPP, MLSClustering, KASP
from lib.visualizer import high_contrast_arr
from lib.misc import set_cuda_devices

matplotlib.style.use("seaborn-poster")
matplotlib.style.use("ggplot")


def generate_gaussian(sizes, mus, sigmas):
    """Generate a mixture of guassians dataset."""
    n_total = sum(sizes)
    res = torch.zeros(n_total, 2)
    cls = torch.zeros(
        n_total,
    )
    count = 0
    for idx, n in enumerate(sizes):
        c, s = mus[idx], sigmas[idx]
        points = torch.matmul(torch.randn(n, 2), s) + c.unsqueeze(0)
        res[count : count + n] = points
        cls[count : count + n] = idx
        count += n
    return res, count


def plot_result(alg, xs, xy, n_class, RESOLUTION, x_min, x_max, y_min, y_max):
    """Show the results of clustering."""
    pred_y = alg.predict(xs, n_class)
    c = high_contrast_arr[pred_y.numpy()] / 255.0
    smap = alg.predict(xy.view(-1, 2), n_class, True)
    smap = smap.view(RESOLUTION, RESOLUTION, -1).max(2).values
    plt.contour(xy[..., 0], xy[..., 1], smap, levels=50, alpha=0.2)
    plt.scatter(xs[:, 0], xs[:, 1], s=2, c=c)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])


def main():
    """Entrance."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", default="expr/toy", help="The directory of experiments."
    )
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument(
        "--eval-seed", default=2021, type=int, help="The seed for this evaluation."
    )
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)

    if not os.path.exists(args.expr):
        os.makedirs(args.expr)

    CLASS_SIZE = [800, 400, 800, 400, 400, 1000]
    CLASS_CENTER = torch.Tensor(
        [[-7, -7], [-4, -4], [-2, 0], [2, 5], [3, 6], [3, -1]]
    ).float()
    CLASS_SIGMA = torch.Tensor(
        [
            [[2, 0], [0, 0.6]],
            [[0.1, 0.3], [0.3, 2]],
            [[0.5, 0], [0, 1]],
            [[2, -0.3], [-0.3, 0.2]],
            [[1, 0], [0, 1]],
            [[1, 0.5], [0.5, 1]],
        ]
    ).float()
    n_class = 3
    xs, ys = generate_gaussian(CLASS_SIZE, CLASS_CENTER, CLASS_SIGMA)
    xs_np = xs.numpy()
    x_min, y_min = xs.min(0).values
    x_max, y_max = xs.max(0).values
    x_min, y_min = x_min - 0.5, y_min - 0.5
    x_max, y_max = x_max + 0.5, y_max + 0.5
    RESOLUTION = 1024
    indice = torch.arange(0, RESOLUTION)
    xy = torch.stack(torch.meshgrid(indice, indice), -1).float()
    xy[..., 0] = xy[..., 0] / RESOLUTION * (x_max - x_min) + x_min
    xy[..., 1] = xy[..., 1] / RESOLUTION * (y_max - y_min) + y_min

    ns = [n_class + x for x in [0, 1, 2, 4, 8, 16, 32, 64, 128]]
    for n_cluster in ns:
        torch.manual_seed(n_cluster)
        torch.cuda.manual_seed(n_cluster)

        kmeans_alg = KMeans(n_clusters=n_cluster)
        pred_y = kmeans_alg.fit_predict(xs_np)
        centroids = kmeans_alg.cluster_centers_
        c = high_contrast_arr[pred_y] / 255.0
        plt.scatter(xs[:, 0], xs[:, 1], s=2, c=c)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker="x")
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.savefig(f"{args.expr}/kmeans_{n_cluster:03d}_viz.png")
        plt.close()

        if n_cluster == n_class:
            print("=> AHC")
            ahc_alg = AgglomerativeClustering(n_clusters=n_cluster)
            pred_y = ahc_alg.fit_predict(xs_np)
            c = high_contrast_arr[pred_y] / 255.0
            plt.scatter(xs[:, 0], xs[:, 1], s=2, c=c)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.savefig(f"{args.expr}/ahc_{n_cluster:03d}_viz.png")
            plt.close()

            print("=> Spectral Clustering")
            sc_alg = SpectralClustering(n_clusters=n_cluster)
            pred_y = sc_alg.fit_predict(xs_np)
            c = high_contrast_arr[pred_y] / 255.0
            plt.scatter(xs[:, 0], xs[:, 1], s=2, c=c)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.savefig(f"{args.expr}/sc_{n_cluster:03d}_viz.png")
            plt.close()

            continue  # groundtruth

        w_init = torch.from_numpy(centroids).float().cuda()
        name = f"heuristic_{n_cluster:03d}"
        mldc_alg1 = MLSClustering(
            w_init,
            use_bias=True,
            metric="heuristic",
            objective="ovrsvc-l2",
            svm_coef=5000,
            max_iter=20,
            save_prefix=f"{args.expr}/{name}",
        )
        mldc_alg1.fit([xs.cuda()])
        plot_result(mldc_alg1, xs, xy, n_class, RESOLUTION, x_min, x_max, y_min, y_max)
        plt.savefig(f"{args.expr}/{name}_viz.png")
        plt.close()

        """
        name = f"mcsvc_{n_cluster:03d}"
        mldc_alg1 = MLSClustering(
            w_init, b_init,
            metric="mld", objective="mcsvc-l2",
            max_iter=1000, svm_coef=1000,
            save_prefix=f"{args.expr}/{name}")
        mldc_alg1.fit([xs.cuda()])
        plot_result(mldc_alg1)
        
        name = f"mcmld_{n_cluster:03d}"
        mldc_alg2 = MLSClustering(
            w_init, b_init,
            metric="mld", objective="mcmld-l2",
            max_iter=1000, svm_coef=1000,
            save_prefix=f"{args.expr}/{name}")
        mldc_alg2.fit([xs.cuda()])
        plot_result(mldc_alg2)
        """


if __name__ == "__main__":
    main()
