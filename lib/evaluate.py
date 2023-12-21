"""Evaluation related utilities.
"""
from models.semantic_extractor import SELearner
from torchmetrics.functional import jaccard_index
import numpy as np
from tqdm import tqdm
import torch


def write_results(res_path, mIoU, c_ious):
    """Write results to a txt file. Paired with read_results."""
    with open(res_path, "w") as f:
        c_ious = [float(i) for i in c_ious]
        s = [str(c) for c in c_ious]
        f.write(str(float(mIoU)) + "\n")
        f.write(" ".join(s))


def read_results(res_path):
    """Read results from a txt file. Paired with write_results."""
    with open(res_path, "r") as f:
        mIoU = float(f.readline().strip())
        c_iou = [float(i) for i in f.readline().strip().split(" ")]
    x = np.array(c_iou)
    mIoU_ = float(x[x > -0.1].mean())
    if abs(mIoU_ - mIoU) > 1e-3:
        print(f"!> {mIoU_} does not match original {mIoU}")
        return mIoU_, c_iou
    return mIoU, c_iou


def aggregate_res(res):
    """Aggregate results."""
    # r[0] is pixelacc, r[1] is IoU
    ic_iou = torch.stack([r[1] for r in res])
    c_iou = torch.zeros(ic_iou.shape[1])
    for c in range(ic_iou.shape[1]):
        val = ic_iou[:, c]
        val = val[val > -0.1]
        c_iou[c] = -1 if val.shape[0] == 0 else val.mean()
    mIoU = c_iou[c_iou > -0.1].mean()
    return mIoU, c_iou


def aggregate_iou(IoU):
    """Aggregate instance-level classwise IoU to global classwise IoU and mIoU.
    Args:
      IoU : torch.Tensor of shape (M, N_CLASS). Absent score is -1.
    """
    n_class = IoU.shape[1]
    cIoU = torch.zeros(n_class)
    for y in range(n_class):
        val = IoU[:, y]
        val = val[val > -0.1]
        cIoU[y] = -1 if val.shape[0] == 0 else val.mean()
    mIoU = cIoU[cIoU > -0.1].mean()
    return mIoU, cIoU


def evaluate_SE(SE, G, P, resolution, num, ls="trunc-wp"):
    """Evaluate extractor mIoU on learner."""
    learner = SELearner(SE, G, P, resolution=resolution, latent_strategy=ls).cuda()
    res = []
    for i in tqdm(range(num)):
        with torch.no_grad():
            seg, label = learner(torch.randn(1, 512).cuda())

        dt = seg[-1].argmax(1)
        gt = label
        IoU = jaccard_index(
            dt,
            gt,
            num_classes=SE.n_class,
            ignore_index=0,
            absent_score=-1,
            reduction="none",
        )
        pixelacc = (dt == gt).sum() / float(dt.shape.numel())
        res.append([pixelacc, IoU])
    mIoU, c_ious = aggregate_res(res)
    return mIoU, c_ious


def evaluate_dataset(dl, P, n_class):
    """Evaluate predictor mIoU on a dataloader."""
    res = []
    for x, y in tqdm(dl):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            y_pred = P(x).argmax(1)
        IoU = jaccard_index(
            y_pred,
            y,
            num_classes=n_class,
            ignore_index=0,
            absent_score=-1,
            reduction="none",
        )
        pixelacc = (y_pred == y).sum() / float(y_pred.shape.numel())
        res.append([pixelacc, IoU])
    mIoU, c_ious = aggregate_res(res)
    return mIoU, c_ious
