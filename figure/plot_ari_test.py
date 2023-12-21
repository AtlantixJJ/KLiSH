"""Visualize the ARI result."""
# pylint: disable=wrong-import-position,wrong-import-order,multiple-imports,invalid-name
import json, argparse, glob, sys
sys.path.insert(0, ".")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')
from lib.misc import formal_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr")
    parser.add_argument("--name", default="test_ARI")
    parser.add_argument("--min", default=20, type=int)
    parser.add_argument("--max", default=50, type=int)
    parser.add_argument("--out-dir", default="results/plot")
    args = parser.parse_args()
    MIN_CLUSTERS = 10
    files = glob.glob(f"{args.expr}/{args.name}/*.json")
    fig_data = {}
    for fp in files:
        fname = fp[fp.rfind("/")+1:fp.rfind(".")]
        g, ds, mtd, seed, bias_usage = fname.split("_")
        g_name = formal_name(f"{g}-{ds}")
        mtd_name = {
            "klish": "KLiSH",
            "kmeans": "K-means",
            "ahc": "AHC",
            "sc": "KASP"}[mtd]
        if g_name not in fig_data:
            fig_data[g_name] = {}
        if mtd_name not in fig_data[g_name]:
            fig_data[g_name][mtd_name] = {"bias": [], "nobias": []}
        with open(fp, "r", encoding="ascii") as f:
            res_dic = json.load(f)

        legends = []
        if "x" not in fig_data[g_name][mtd_name] and len(res_dic["x"]) > 0:
            fig_data[g_name][mtd_name]["x"] = np.array(res_dic["x"])
        arr = np.array(res_dic["y"])
        fig_data[g_name][mtd_name][bias_usage].append(arr)

    # compare bias under each method
    for g_name, g_dic in fig_data.items():
        for mtd_name, mtd_dic in g_dic.items():
            x = mtd_dic["x"]
            legends = []
            plt.figure(figsize=(9, 5))
            ax = plt.subplot(1, 1, 1)
            for bias_usage, arrs in mtd_dic.items():
                if bias_usage == "x":
                    continue
                if mtd_name == "KLiSH":
                    bias_suffix = {"bias": " (bias)", "nobias": ""}[bias_usage]
                else:
                    bias_suffix = {"bias": " (euclidean)",
                                 "nobias": " (arccos)"}[bias_usage]
                legends.append(f"{mtd_name}{bias_suffix}")

                if arrs[0].shape[0] == 0:
                    print(f"!> {g_name} {mtd_name} {bias_usage} missing data!")
                    break
                mask = (x >= args.min) & (x <= args.max)
                y = np.stack(arrs)
                y_min = y.min(0)[mask]
                y_mean = y.mean(0)[mask]
                y_max = y.max(0)[mask]
                ax.plot(x[mask], y_mean, "-o", linewidth=1, markersize=5)
                ax.fill_between(x[mask], y_min, y_max, alpha=0.2)
            plt.legend(legends, fontsize=18)
            plt.tight_layout()
            plt.savefig(f"{args.out_dir}/{args.name}_{g_name}_{mtd_name}_bias.png")
            plt.close()
            
    # compare each method under bias
    bias_usages = ["bias", "nobias"]
    for bias_usage in bias_usages:
        for g_name, g_dic in fig_data.items():
            plt.figure(figsize=(9, 5))
            ax = plt.subplot(1, 1, 1)
            legends = []
            for mtd_name, mtd_dic in g_dic.items():
                x = mtd_dic["x"]

                if mtd_name == "KLiSH":
                    bias_suffix = {"bias": " (bias)", "nobias": ""}[bias_usage]
                else:
                    bias_suffix = {"bias": " (euclidean)",
                                 "nobias": " (arccos)"}[bias_usage]
                legends.append(f"{mtd_name}{bias_suffix}")

                arrs = mtd_dic[bias_usage]
                if arrs[0].shape[0] == 0:
                    print(f"!> {g_name} {mtd_name} {bias_usage} missing data!")
                    break
                mask = (x >= args.min) & (x <= args.max)
                y = np.stack(arrs)
                y_min = y.min(0)[mask]
                y_mean = y.mean(0)[mask]
                y_max = y.max(0)[mask]
                ax.plot(x[mask], y_mean, "-o", linewidth=1, markersize=5)
                ax.fill_between(x[mask], y_min, y_max, alpha=0.2)
            plt.legend(legends, fontsize=18)
            plt.tight_layout()
            plt.savefig(f"{args.out_dir}/{args.name}_{g_name}_{bias_usage}.png")
            plt.close()

"""
plt.figure(figsize=(9, 5))
plt.plot(
    res["kmeans"]["x"],
    res["kmeans"]["y"],
    "rX--",
    linewidth=1)
x = np.array(res["klish"]["x"])
y = np.array(res["klish"]["y"])
mask = x >= 20
x = x[mask]
y = y[mask]
plt.plot(x, y, "b-o", linewidth=1, markersize=5)
plt.xlabel("#clusters", fontsize=18)
plt.ylabel("ARI", fontsize=18)
plt.legend(["K-means", "KLiSH"], fontsize=18)
plt.tight_layout()
plt.savefig(f"{args.out_dir}/ARI_{fname}.png")
plt.close()
"""
