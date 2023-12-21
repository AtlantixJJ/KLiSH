"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import op


def mask_cross_entropy_loss(mask, x, y):
    """Accumulate cross-entropy loss only on masked region."""
    ce = F.cross_entropy(x, y, reduction="none")
    return (mask * ce).sum() / mask.sum()


def mask_focal_loss(mask, x, y):
    """Accumulate focal loss only on masked region."""
    ce = FocalLoss()(x, y, reduction="none")
    return (mask * ce).sum() / mask.sum()


class BinarySVMLoss(nn.Module):
    """The binary L2 SVM loss.
    Args:
      coef: The coefficient of SVM loss. Default is 3000.
    """

    def __init__(self, coef=5000.0):
        super().__init__()
        self.coef = coef

    def forward(self, inputs, targets):
        """Calculate loss."""
        svm_label = -torch.ones_like(inputs)
        svm_label.scatter_(1, targets.unsqueeze(1), 1)
        margin = (1 - svm_label * inputs).clamp(min=0)
        return torch.square(margin).mean() * self.coef


class FocalLoss(nn.Module):
    """Focal loss."""

    def __init__(self, binary=False, alpha=1.0, gamma=5, T=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma  # 指数
        self.T = T
        if binary:
            self.func = F.binary_cross_entropy_with_logits
        else:
            self.func = F.cross_entropy

    def forward(self, inputs, targets, reduction="mean"):
        """Calculate focal loss."""
        loss = self.func(inputs * self.T, targets, reduction="none")
        pt = torch.exp(-loss)
        focal_loss = self.alpha * (1 - pt).pow(self.gamma) * loss
        if reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss


def segloss(segs, label, loss_fn):
    """The final version of loss."""
    segloss = []
    size = label.size(2)
    for seg in segs:
        seg = op.bu(seg, size) if seg.size(2) != size else seg
        segloss.append(loss_fn(seg, label))
    return segloss


def segloss_bce(segs, label, loss_fn_layer, loss_fn_final):
    """Use BCE for each layer. It is slow and CPU intensive."""
    N = len(segs)
    segloss = []
    onehot = op.int2onehot(label, segs[0].shape[1])
    # BCE loss
    for i in range(N):
        seg = segs[i]
        segloss.append(
            loss_fn_layer(
                seg if seg.size(2) == label.size(2) else op.bu(seg, label.size(2)),
                onehot,
            )
        )
    # CE loss
    final = segs[-1]
    segloss.append(
        loss_fn_final(
            final if final.size(2) == label.size(2) else op.bu(final, label.size(2)),
            label,
        )
    )
    return segloss
