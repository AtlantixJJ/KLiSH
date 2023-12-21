"""Experiment on toy datasets."""
# pylint: disable=invalid-name
import os
import sys

sys.path.insert(0, ".")
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.svm import LinearSVC
from lib.cluster import MultiGPUKMeansPP, MLSClustering, KASP
from lib.misc import set_cuda_devices
from PIL import ImageColor

beautiful_colors = ["#fe4a49", "#2ab7ca", "#fed766", "#e6e6ea", "#f4f4f8"]
color_arr = np.array([ImageColor.getrgb(c) for c in beautiful_colors])


def generate_datasets(sizes, mus, sigmas, is_gaussians):
    """Generate a mixture of guassians dataset."""
    n_total = sum(sizes)
    res = torch.zeros(n_total, 2)
    cls = torch.zeros(
        n_total,
    )
    count = 0
    for idx, n in enumerate(sizes):
        c, s = mus[idx], sigmas[idx]
        if is_gaussians[idx]:
            noise = torch.randn(n, 2)
        else:
            noise = torch.rand(n, 2) * 4 - 2
        points = torch.matmul(noise, s) + c.unsqueeze(0)
        res[count : count + n] = points
        cls[count : count + n] = idx
        count += n
    return res


def coef_line(w, b, x_var=True):
    """Calculate the coefficient of y = kx + b.
    Args:
        w: 2-dim vector.
        b: scalar.
        w[0] x + x[1] y + b = 0
    """
    if x_var:
        return float(-w[0] / w[1]), float(-b / w[1])
    return float(-w[1] / w[0]), float(-b / w[0])


def points_from_line(w, b, x_min, x_max, y_min, y_max, R):
    """Draw points from line equation.
    Args:
        w, b: The equation is w[0] x + w[1] y + b = 0.

    """
    if w[0] > w[1]:  # y-variable
        lin_y = np.linspace(y_min, y_max, R)
        k, b = coef_line(w, b, False)
        return np.stack([lin_y * k + b, lin_y], 1)
    # x-variable
    lin_x = np.linspace(x_min, x_max, R)
    k, b = coef_line(w, b, True)
    return np.stack([lin_x, lin_x * k + b], 1)


def select_region(points, sec_x, w, b, c1, c2):
    """
    Args:
        points: (N, 2)
        sec_x: scalar
        w: (M, 2)
        b: (M, )
        c1, c2: scalar, the clusters of the boundary
    """
    mask = points[:, 0] < sec_x
    score = np.matmul(w, points.T) + b[:, None]  # (M, N)
    score_target = score[[c1, c2]].max(0).values  # (N,) the maximum of target classes
    compl_indice = [i for i in range(w.shape[0]) if i not in [c1, c2]]
    score_other = score[compl_indice].max(0).values  # (N,) the maximum of other classes
    suc = (score_target - score_other) > 0
    if suc[mask].sum() > suc[~mask].sum():
        return points[mask]
    return points[~mask]


def plot_result_2d(w, b, R, x_min, x_max, y_min, y_max, line_color="brown"):
    """Show the results of clustering."""
    k1, b1 = coef_line(w[1] - w[0], b[1] - b[0])
    k2, b2 = coef_line(w[2] - w[1], b[2] - b[1])
    sec_x = (b2 - b1) / (k1 - k2)
    for c1 in range(w.shape[0]):
        for c2 in range(c1 + 1, w.shape[0]):
            dw, db = w[c1] - w[c2], b[c1] - b[c2]
            points = points_from_line(dw, db, x_min, x_max, y_min, y_max, R)
            points = select_region(points, sec_x, w, b, c1, c2)
            plt.plot(points[:, 0], points[:, 1], c=line_color)


def create_stdplot(x_min, x_max, y_min, y_max):
    """Create a standard plot."""
    plt.figure(
        figsize=(10, 10),
        facecolor=color_arr[-1] / 255.0,
        edgecolor="black",
        tight_layout=True,
    )
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.axis("off")


def unify_labeling(pred_y, pred_ref):
    """Given two clustering results, unify their label assignments."""
    new_label = np.zeros_like(pred_y)
    arr = torch.from_numpy(pred_y)
    for i in range(3):
        my_label = int(arr[pred_ref == i].mode().values)
        new_label[pred_y == my_label] = i
    return new_label


if __name__ == "__main__":
    """Entrance."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", default="expr/toy_demo", help="The directory of experiments."
    )
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument(
        "--eval-seed", default=2021, type=int, help="The seed for this evaluation."
    )
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)

    if not os.path.exists(args.expr):
        os.makedirs(args.expr)

    RESOLUTION = 4096
    CLASS_SIZE = np.array([2, 2, 2, 1, 1, 2, 1]) * 200
    CLASS_CENTER = torch.Tensor(
        [[-6, -5], [-4, -3.5], [-2, 0], [2, 5], [3, 6], [3, -1], [4, -3]]
    ).float()
    CLASS_SIGMA = torch.Tensor(
        [
            [[0.6, -0.5], [0, 1.5]],
            [[0, -0.6], [0.6, 2]],
            [[0.5, 0], [0, 1]],
            [[2, -0.3], [-0.3, 0.2]],
            [[1, 0], [0, 1]],
            [[1, 0.5], [0.5, 1]],
            [[1, -0.5], [-0.5, 2]],
        ]
    ).float()
    IS_GAUSSIAN = [False, False, True, True, True, True, True]

    xs = generate_datasets(CLASS_SIZE, CLASS_CENTER, CLASS_SIGMA, IS_GAUSSIAN)
    ys = np.zeros(xs.shape[:1], dtype="int32")
    cum = CLASS_SIZE.cumsum()
    ys[cum[2] : cum[4]] = 1
    ys[cum[4] : cum[6]] = 2
    xs_np = xs.numpy()
    x_min, y_min = xs.min(0).values
    x_max, y_max = xs.max(0).values
    x_min, y_min = x_min - 0.1, y_min - 0.1
    x_max, y_max = x_max + 0.1, y_max + 0.1
    if y_max - y_min > x_max - x_min:
        c = (x_max + x_min) / 2
        d = (y_max - y_min) / 2
        x_max, x_min = c + d, c - d
    else:
        c = (y_max + y_min) / 2
        d = (x_max - x_min) / 2
        y_max, y_min = c + d, c - d
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)

    kmeans_alg = KMeans(n_clusters=3)
    kmeans_y = kmeans_alg.fit_predict(xs_np)
    centroids = kmeans_alg.cluster_centers_
    c = color_arr[kmeans_y] / 255.0
    create_stdplot(x_min, x_max, y_min, y_max)
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, marker="x", c="black")
    w = torch.from_numpy(centroids)
    b = -0.5 * (w * w).sum(1)

    plot_result_2d(w, b, RESOLUTION, x_min, x_max, y_min, y_max)
    plt.savefig(f"{args.expr}/kmeans.png")
    plt.close()

    svm_alg = LinearSVC(
        penalty="l2",
        loss="hinge",
        dual=True,
        tol=1e-9,
        C=10,
        multi_class="crammer_singer",
        max_iter=1e6,
    )
    svm_alg.fit(xs, ys)
    ys_reorder = unify_labeling(ys, kmeans_y)
    c = color_arr[ys_reorder] / 255.0
    create_stdplot(x_min, x_max, y_min, y_max)
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    w = torch.from_numpy(svm_alg.coef_)
    b = torch.from_numpy(svm_alg.intercept_)
    plot_result_2d(w, b, RESOLUTION, x_min, x_max, y_min, y_max, "darkgrey")
    plt.savefig(f"{args.expr}/svm_viz.png")
    plt.close()

    ahc_alg = AgglomerativeClustering(n_clusters=3)
    pred_y = ahc_alg.fit_predict(xs_np)
    pred_y = unify_labeling(pred_y, kmeans_y)
    c = color_arr[pred_y] / 255.0
    create_stdplot(x_min, x_max, y_min, y_max)
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    plt.savefig(f"{args.expr}/ahc.png")
    plt.close()

    sc_alg = SpectralClustering(n_clusters=3)
    pred_y = sc_alg.fit_predict(xs_np)
    pred_y = unify_labeling(pred_y, kmeans_y)
    c = color_arr[pred_y] / 255.0
    create_stdplot(x_min, x_max, y_min, y_max)
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    plt.savefig(f"{args.expr}/sc.png")
    plt.close()

    kmeans_alg = KMeans(n_clusters=4)
    pred_y = kmeans_alg.fit_predict(xs_np)
    centroids = kmeans_alg.cluster_centers_
    c = color_arr[pred_y] / 255.0
    create_stdplot(x_min, x_max, y_min, y_max)
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker="x", c="grey")
    plt.savefig(f"{args.expr}/kmeans_init.png")
    plt.close()

    w_init = torch.from_numpy(centroids).float().cuda()
    name = "heuristic_3"
    klish_alg = MLSClustering(
        w_init,
        use_bias=True,
        metric="heuristic",
        objective="ovrsvc-l2",
        svm_coef=100,
        max_iter=1000,
        save_prefix=f"{args.expr}/{name}",
    )
    klish_alg.fit([xs.cuda()])
    create_stdplot(x_min, x_max, y_min, y_max)
    pred_y = klish_alg.predict(xs, 3).numpy()
    pred_y = unify_labeling(pred_y, kmeans_y)
    c = color_arr[pred_y] / 255.0
    plt.scatter(xs[:, 0], xs[:, 1], s=5, c=c)
    dic = klish_alg.merge_record[3]
    plot_result_2d(dic["weight"], dic["bias"], RESOLUTION, x_min, x_max, y_min, y_max)
    plt.savefig(f"{args.expr}/{name}_viz.png")
    plt.close()
