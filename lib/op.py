"""Pytorch or Numpy operation functions.
"""
# pylint: disable=too-many-arguments,invalid-name
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from lib.misc import GeneralThread

EPS = 1e-6
MIN_FREQ = 1e-3


def cat_mut_iou(label_1: Tensor, label_2: Tensor):
    """Calculate category IoU between two label.
    Args:
      label1, label2: (N, HW)
    Returns:
      IoU, count
    """
    n1, n2 = label_1.max() + 1, label_2.max() + 1
    res = torch.zeros(n1, n2).to(label_1.device)
    count = torch.zeros(n1, n2).to(label_1.device)
    for y_1 in label_1.unique():
        mask_1 = label_1 == y_1
        for y_2 in label_2.unique():
            mask_2 = label_2 == y_2
            isct = (mask_1 & mask_2).sum(1)  # (N,)
            union = (mask_1 | mask_2).sum(1)
            valid = union > 0
            count[y_1, y_2] = valid.sum()
            res[y_1, y_2] = (isct / union.float().clamp(min=EPS)).sum()
    return res / count.clamp(min=1), count


def pairwise_dist(mat_1, mat_2, dist="arccos"):
    """
    Args:
      mat_1: (N, C)
      mat_2: (M, C)
      dist: The distance metric type, euclidean or arccos. ip denotes inner product.
    Returns:
      A distance matrix of shape (N, M) of double dtype
    """
    if dist == "euclidean":
        dist = torch.zeros(mat_1.shape[0], mat_2.shape[0]).to(mat_1)
        n_batch = 4
        assert mat_1.shape[0] % n_batch == 0
        batch_size = mat_1.shape[0] // n_batch
        for i in range(n_batch):
            for j in range(mat_2.shape[0]):  # usually B is smaller
                indice = slice(i * batch_size, (i + 1) * batch_size)
                prod = torch.square(mat_1[indice] - mat_2[j : j + 1])
                dist[indice, j].copy_(prod.sum(1))
        return torch.sqrt(dist)
    if dist == "arccos":  # make sure A are unit vectors
        inner_product = torch.matmul(mat_1, mat_2.permute(1, 0))
        return inner_product.clamp_(-1 + EPS, 1 - EPS).acos_()
    if dist == "ip":
        return torch.exp(-torch.matmul(mat_1, mat_2.permute(1, 0)) / 500)
    return 0


def upper_triangle_minmax(mat, func="min"):
    """Take minimum or maximum on the upper triangle of a matrix."""
    val = mat[0, 1]
    ind = (0, 1)
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            if func == "min":
                if mat[i, j] < val:
                    val = mat[i, j]
                    ind = (i, j)
            elif func == "max":
                if mat[i, j] > val:
                    val = mat[i, j]
                    ind = (i, j)
    return val, ind


def multigpu_map(func, margs, reduce="sum"):
    """Map a function on a multigpu list of arguments.
    Args:
        func: The function must return a list of Tensor.
        margs: A list of either multigpu tensor list or non-list elements.
        reduce: sum, append.
    """
    threads = []
    n_device = len(margs[0])
    for d_id in range(n_device):
        args = [arg[d_id] if isinstance(arg, list) else arg for arg in margs]
        threads.append(GeneralThread(func, *args))
        threads[-1].start()
    res = None
    for thr in threads:
        thr.join()
        if res is None:
            if reduce == "sum":
                res = [0] * len(thr.res)
            elif reduce == "append":
                res = [[] for _ in thr.res]
        for i, r in enumerate(thr.res):
            if reduce == "sum":
                res[i] = res[i] + r.cpu().clone().detach()
            elif reduce == "append":
                res[i].append(r)
    return res


def pairwise_dist_minibatch(mat_1, mat_2, dist):
    """Calculate pairwise distance in minibath.
    Warning: gradient is not kept.
    Args:
        mat_1, mat_2: torch tensor.
    Returns:
        A numpy array.
    """
    res = np.zeros((mat_1.shape[0], mat_2.shape[0]))
    n_batch = int(1024**2 * 8 / mat_1.shape[1])
    for i in range(mat_1.shape[0] // n_batch + 1):
        abg, aed = i * n_batch, min(mat_1.shape[0], (i + 1) * n_batch)
        if abg >= mat_1.shape[0]:
            break
        submat_1 = mat_1[abg:aed]
        for j in range(mat_2.shape[0] // n_batch + 1):
            bbg, bed = j * n_batch, min(mat_2.shape[0], (j + 1) * n_batch)
            if bbg >= mat_2.shape[0]:
                break
            res[abg:aed, bbg:bed] = torch2numpy(
                pairwise_dist(submat_1, mat_2[bbg:bed].to(mat_1), dist)
            )
    return res


def lerp(a, b, x, y, i):
    """
    Args:
      input from [a, b], output to [x, y], current position i
    """
    return (i - a) / (b - a) * (y - x) + x


def int2onehot(x, n):
    """Convert an integer label to onehot label."""
    z = torch.zeros(x.shape[0], n, *x.shape[1:]).to(x.device)
    return z.scatter_(1, x.unsqueeze(1), 1)


def bu(img, size, align_corners=True):
    """Bilinear interpolation with Pytorch.

    Args:
      img : a list of tensors or a tensor.
    """
    if isinstance(img, list):
        return [
            F.interpolate(i, size=size, mode="bilinear", align_corners=align_corners)
            for i in img
        ]
    return F.interpolate(img, size=size, mode="bilinear", align_corners=align_corners)


def tocpu(x):
    """Convert to CPU Tensor."""
    return x.clone().detach().cpu()


def delete_index(x, index, dim):
    """Delete an element in a Tensor."""
    if dim == 0:
        return torch.cat([x[:index], x[index + 1 :]], 0)
    if dim == 1:
        return torch.cat([x[:, :index], x[:, index + 1 :]], 1)
    return x


def delete_index_reorder(x, index, dim):
    """Delete an element in a Tensor by moving subsequent parts over previous ones."""
    if index >= x.shape[dim] - 1:
        if dim == 0:
            x[-1] = 0
        elif dim == 1:
            x[:, -1] = 0

    for i in range(index, x.shape[dim] - 1):
        if dim == 0:
            x[i].copy_(x[i + 1])
        elif dim == 1:
            x[:, i].copy_(x[:, i + 1])
    return x


def modify_optim(optim_src, optim_tar, p):
    """Modify optimizer state by deleting p-th element."""
    if isinstance(optim_src, torch.optim.Adam):
        _modify_optim_adam(optim_src, optim_tar, p)
    elif isinstance(optim_src, torch.optim.LBFGS):
        pass
        # do not modify LBFGS as the hessian cannot be modified
        # otherwise the training will be unstable
        # _modify_optim_lbfgs(optim_src, optim_tar, p)


def _modify_optim_lbfgs(optim_src, optim_tar, p):
    """Modify LBFGS state by deleting p-th element."""

    def _del_flat_param(x):
        offset = 0
        v = []
        for param in src_params:
            x_p = x[offset : offset + param.numel()].view_as(param)
            x_p_d = delete_index(x_p, p, 0).view(-1)
            v.append(x_p_d)
            offset += param.numel()
        return torch.cat(v).detach()

    def _process_list(vals):
        if isinstance(vals[0], Tensor) and vals[0].numel() > 1:
            return [_del_flat_param(x) for x in vals]
        return vals

    src_params = optim_src.param_groups[0]["params"]
    for g1, g2 in zip(optim_tar.param_groups, optim_src.param_groups):
        for p1, p2 in zip(g1["params"], g2["params"]):
            state_dict = optim_src.state[p2]
            for key, val in state_dict.items():
                if isinstance(val, list):
                    optim_tar.state[p1][key] = _process_list(val)
                elif isinstance(val, Tensor) and val.numel() > 1:
                    optim_tar.state[p1][key] = _del_flat_param(val)
                else:
                    optim_tar.state[p1][key] = val


def _modify_optim_adam(optim_src, optim_tar, p):
    """Modify Adam state by deleting p-th element."""
    for g1, g2 in zip(optim_tar.param_groups, optim_src.param_groups):
        for p1, p2 in zip(g1["params"], g2["params"]):
            optim_tar.state[p1]["step"] = optim_src.state[p2]["step"]
            optim_tar.state[p1]["exp_avg"] = delete_index(
                optim_src.state[p2]["exp_avg"], p, 0
            )
            optim_tar.state[p1]["exp_avg_sq"] = delete_index(
                optim_src.state[p2]["exp_avg_sq"], p, 0
            )


def copy_tensor(x, grad=False):
    """Copy a tensor."""
    return x.clone().detach().requires_grad_(grad)


def torch2numpy(x):
    """Convert a Tensor to a numpy array."""
    if isinstance(x, float):
        return x
    return x.detach().cpu().numpy()


def torch2image(x, data_range="[-1,1]"):
    """Convert torch tensor in [-1, 1] scale to be numpy array format
    image in (N, H, W, C) in [0, 255] scale.
    """
    if data_range == "[-1,1]":
        x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).cpu().numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)
    elif len(x.shape) == 3:
        x = x.transpose(1, 2, 0)  # (C, H, W)
    return x.astype("uint8")


def image2torch(x):
    """Process [0, 255] (N, H, W, C) numpy array format
    image into [0, 1] scale (N, C, H, W) torch tensor.
    """
    y = torch.from_numpy(x).float() / 255.0
    if len(x.shape) == 3 and x.shape[2] == 3:
        return y.permute(2, 0, 1).unsqueeze(0)
    if len(x.shape) == 4:
        return y.permute(0, 3, 1, 2)
    return 0
