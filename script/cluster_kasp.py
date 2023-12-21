"""KASP clustering the features of generators, initialized from K-means results."""
# pylint: disable=invalid-name,consider-using-f-string
import os
import json
import argparse
import torch
import glob
from sklearn.cluster import SpectralClustering
from tqdm import tqdm


def main():
    """KASP clustering entrance."""
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument("--expr", default="expr/cluster")
    parser.add_argument("--name", default="kmeans")
    args = parser.parse_args()

    if not os.path.exists(f"{args.expr}/kasp"):
        os.makedirs(f"{args.expr}/kasp")

    kmeans_files = glob.glob(f"{args.expr}/{args.name}/*.pth")
    kmeans_files.sort()
    for fp in kmeans_files:
        kmeans_res = torch.load(fp)
        name = fp[fp.rfind("/") + 1 : -4]
        kasp_prefix = f"{args.expr}/kasp/{name}"

        print(f"=> Running KASP on {name}")
        dic = {}
        for dist in ["euclidean", "arccos"]:
            if dist not in kmeans_res:
                continue
            k_init = 100
            w_np = kmeans_res[dist][k_init]["weight"].cpu().numpy()
            dic[dist] = {}
            for new_k in tqdm(range(2, 101)):
                if new_k == k_init:
                    dic[dist][new_k] = list(range(100))
                    continue
                alg = SpectralClustering(new_k,
                    affinity="nearest_neighbors",
                    n_neighbors=3,
                    assign_labels="discretize"
                )
                label_perm = alg.fit_predict(w_np)
                dic[dist][new_k] = label_perm.tolist()
        json.dump(dic, open(f"{kasp_prefix}.json", "w", encoding="ascii"))


if __name__ == "__main__":
    main()
