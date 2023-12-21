"""Visualize clustering evaluation result."""
# pylint: disable=wrong-import-position,wrong-import-order,multiple-imports,invalid-name
import json, argparse, glob, sys

sys.path.insert(0, ".")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use("seaborn-poster")
matplotlib.style.use("ggplot")

from lib.misc import formal_name, str_table_single, str_latex_table, str_csv_table


def simplify_name(n):
    n = n.replace("8,10,12,14,16", "F")
    n = n.replace("9,11,13,15,17", "S")
    n = n.replace("8,9,10,11,12,13,14,15,16,17", "A")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/eval_clustering_ablation")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    MIN_CLUSTERS, MAX_CLUSTERS = 20, 40
    x_slices = {
        "StyleGAN2-FFHQ": [20, 30, 40],
        "StyleGAN-CelebAHQ": [20, 30, 40],
        "PGGAN-CelebAHQ": [20, 30, 40],
    }
    dist_dic = {
        "KASP": ["bias"],
        "K-means": ["bias"],
        "AHC": ["bias", "nobias"],
        "KLiSH": ["bias"],
    }
    index_names = [
        "Adjusted Mutual Information",
        "Rand Index",
        "Adjusted Rand Index",
        "Fowlkes Mallows Index",
        "Homogeneity",
        "Completeness",
        "V Measure Score",
        "mIoU",
    ]
    in_shorts = ["AMI", "RI", "ARI", "FMI", "H", "C", "VMS", "mIoU"]
    table_indices = ["ARI", "AMI", "mIoU"]
    files = glob.glob(f"{args.expr}/*.json")
    fig_data = {}
    for fp in files:
        fname = fp[fp.rfind("/") + 1 : fp.rfind(".")]
        res = fname.split("_")
        if len(res) == 5:
            g, ds, mtd, seed, bias_usage = res
            layers = "9,11,13,15,17"  # for 1024x1024 generators
        else:
            g, ds, mtd, layers, seed, bias_usage = res
        g_name = formal_name(f"{g}-{ds}")
        mtd_name = {
            "klish": "KLiSH",
            "kmeans": "K-means",
            "ahc": "AHC",
            "kasp": "KASP",
        }[mtd]
        mtd_name = f"{mtd_name}_{layers}"
        if g_name not in fig_data:
            fig_data[g_name] = {}
        if mtd_name not in fig_data[g_name]:
            fig_data[g_name][mtd_name] = {}
        if bias_usage not in fig_data[g_name][mtd_name]:
            fig_data[g_name][mtd_name][bias_usage] = {}
        with open(fp, "r", encoding="ascii") as f:
            res_dic = json.load(f)

        legends = []
        for key, val in res_dic.items():
            arr = np.array(val)
            if key not in fig_data[g_name][mtd_name][bias_usage]:
                fig_data[g_name][mtd_name][bias_usage][key] = []
            fig_data[g_name][mtd_name][bias_usage][key].append(arr)
        if "seed" not in fig_data[g_name][mtd_name][bias_usage]:
            fig_data[g_name][mtd_name][bias_usage]["seed"] = [seed]
        else:
            fig_data[g_name][mtd_name][bias_usage]["seed"].append(seed)

    best_ind = {}
    data_dic = {}
    for g_name, g_dic in fig_data.items():
        data_dic[g_name] = {}
        best_ind[g_name] = {}
        for index_name, in_short in zip(index_names, in_shorts):
            for x_slice in x_slices[g_name]:
                data_dic[g_name][f"{in_short}@{x_slice}"] = {}
            plt.figure(figsize=(10, 5))
            ax = plt.subplot(1, 1, 1)
            legends = []
            # for mtd_name in ["K-means", "KASP", "AHC", "KLiSH"]:
            mtd_names = list(g_dic.keys())
            mtd_names.sort()
            for mtd_name in mtd_names:
                mtd_dic = g_dic[mtd_name]
                for bias_usage, bias_dic in mtd_dic.items():
                    if bias_usage not in ["bias", "nobias"]:
                        continue
                    legends.append(simplify_name(mtd_name))

                    arrs = bias_dic[index_name]
                    if arrs[0].shape[0] == 0:
                        print(
                            f"!> {g_name}/{mtd_name}/{bias_usage}/{index_name} missing data!"
                        )
                        break
                    xs = bias_dic["Clusters"]
                    masks = [(MAX_CLUSTERS >= x) & (x >= MIN_CLUSTERS) for x in xs]
                    new_arrs = [a[m] for a, m in zip(arrs, masks)]
                    lengths = np.array([a.shape[0] for a in new_arrs])
                    valid = lengths == lengths[0]
                    valid_arrs = [a for i, a in enumerate(new_arrs) if valid[i]]
                    y = np.stack(valid_arrs)
                    xs = [x[m] for i, (x, m) in enumerate(zip(xs, masks))
                        if valid[i]]
                    assert np.abs(sum(xs) - xs[0] * len(xs)).sum() < 1e-5
                    x = xs[0]
                    indice = x.argsort()
                    x = x[indice]
                    y = y[:, indice]
                    y_min = y.min(0)
                    y_mean = y.mean(0)
                    y_max = y.max(0)
                    ax.plot(x, y_mean, "-o", linewidth=1, markersize=5)
                    ax.fill_between(x, y_min, y_max, alpha=0.2)

                    # make table
                    if in_short not in table_indices:
                        continue
                    
                    if bias_usage not in dist_dic[mtd_name.split("_")[0]]:
                        continue
                    y_std = y.std(axis=0, ddof=1)
                    y_mean = y.mean(axis=0)
                    if legends[-1] not in best_ind[g_name]:
                        # print(mtd_name, g_name, bias_dic["seed"], y.sum(axis=1))
                        max_ind = int(y.max(1).argmax())
                        best_ind[g_name][legends[-1]] = {
                            "idx": max_ind,
                            "seed": bias_dic["seed"][max_ind],
                        }
                    # should be sorted by AMI
                    max_ind = best_ind[g_name][legends[-1]]["idx"]
                    for x_slice in x_slices[g_name]:
                        ind = x.searchsorted(x_slice)
                        data_dic[g_name][f"{in_short}@{x_slice}"][legends[-1]] = float(
                            y[max_ind, ind]
                        )
                        # {"mean": float(y_mean[ind]), "std": float(y_std[ind])}
            plt.legend(legends, fontsize=18, bbox_to_anchor=(1, 0.7))
            plt.title(index_name)
            plt.tight_layout()
            plt.savefig(f"{args.out_dir}/plot/layers_{in_short}_{g_name}.png")
            plt.close()
        tex_fpath = f"{args.out_dir}/tex/layers_{g_name}.tex"
        with open(tex_fpath, "w", encoding="ascii") as f:
            dic = data_dic[g_name]
            keys = list(dic.keys())
            keys.sort(key=lambda x: int(x.split("@")[1]))
            dic = {k: dic[k] for k in keys}
            f.write(str_latex_table(str_table_single(dic)))
    tex_fpath = f"{args.out_dir}/tex/layers_cluster_bestindice.json"
    best_ind = {
        k1: {k2: v2["seed"] for k2, v2 in v1.items()} for k1, v1 in best_ind.items()
    }
    with open(tex_fpath, "w", encoding="ascii") as f:
        json.dump(best_ind, f, indent=2)
