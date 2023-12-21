import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, optimal_leaf_ordering


def get_primitives(tree, idx):
    """Get the primitive cluster index of a tree node."""
    if "l-p" in tree[idx] and "r-p" in tree[idx]:
        return tree[idx]["l-p"] + tree[idx]["r-p"]
    return [idx]


def trace_index(tree, idx):
    """Trace the index"""
    stack = [idx]
    trace = []
    while len(stack) > 0:
        idx = stack.pop(0)
        trace.append(idx)
        for i, node in enumerate(tree):
            if "l" in node and "r" in node and idx in [node["l"], node["r"]]:
                stack.append(i)
    return trace


def convert_to_scipy(merge_record):
    """Convert the merge record to scipy format
    Returns:
        tree: A dictionary in custom format.
        Z: (n_clusters, 4). Z[:, 0], Z[:, 1] is the index to be merged.
            Z[:, 2] is the value. Z[:, 3] is the size.
    """
    n_clusters = len(list(merge_record.keys()))
    max_clusters = max(list(merge_record.keys()))
    tree = [{} for i in range(max_clusters)]
    Z = np.zeros((n_clusters, 4))
    for i, (_, step_dic) in enumerate(merge_record.items()):
        val = step_dic["Merge Metric"]
        left_index, right_index, root_index = step_dic["Merge Cluster Index"]
        node = {"l": left_index, "r": right_index, "val": val}
        node["l-p"] = get_primitives(tree, left_index)
        node["r-p"] = get_primitives(tree, right_index)
        assert len(tree) == root_index
        tree.append(node)
        Z[i, 0] = left_index
        Z[i, 1] = right_index
        Z[i, 2] = val
        Z[i, 3] = len(node["l-p"]) + len(node["r-p"])
    return tree, Z


if __name__ == "__main__":
    fpath = "expr/cluster/mld-100/stylegan2_ffhq_iauto_b1_mld_mcmld-l2_1991_tree.pth"
    merge_record = torch.load(fpath)
    tree, Z = convert_to_scipy(merge_record)
    Z[:, 2] = Z[:, 2].max() * 1.1 - Z[:, 2]

    plt.figure(figsize=(30, 30))
    N = 10
    labels = Z[:N, :2].reshape(-1)
    unique_labels = np.unique(labels)
    new_labels = np.arange(unique_labels.shape[0])
    label_map = {old: new for old, new in zip(unique_labels, new_labels)}
    new_Z = np.zeros((N, 4))
    new_Z[:] = Z[:N]
    for i in range(N):
        for j in range(2):
            new_Z[i, j] = label_map[Z[i, j]]
    dendrogram(new_Z, leaf_font_size=18)
    plt.savefig(f"tree_analysis_{N}.png")
    plt.close()

    plt.figure(figsize=(30, 30))
    Z[:, 2] = np.arange(Z.shape[0])
    dendrogram(Z, leaf_font_size=18)
    plt.savefig("tree_analysis.png")
    plt.close()
