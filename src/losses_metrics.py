import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Dice multiclasse (Milletari et al., 2016), média sobre as classes presentes.
    Pixels Void são removidos antes do cálculo."""

    def __init__(self, n_classes, ignore_index, eps=1.0):
        super().__init__()
        self.n, self.ignore, self.eps = n_classes, ignore_index, eps

    def forward(self, logits, target):
        valid = (target != self.ignore)
        prob = logits.float().softmax(1) * valid.unsqueeze(1)
        tgt = F.one_hot(target.clamp(max=self.n - 1), self.n).permute(0, 3, 1, 2).float()
        tgt = tgt * valid.unsqueeze(1)
        dims = (0, 2, 3)
        inter = (prob * tgt).sum(dims)
        denom = prob.sum(dims) + tgt.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        present = tgt.sum(dims) > 0
        return 1 - dice[present].mean()


def build_loss(name, n_classes, ignore_index, weights=None):
    """name: 'ce' | 'wce' (CE ponderada) | 'ce_dice'"""
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index,
                             weight=weights if name == "wce" else None)
    if name in ("ce", "wce"):
        return ce
    if name == "ce_dice":
        dice = SoftDiceLoss(n_classes, ignore_index)
        return lambda logits, y: ce(logits, y) + dice(logits, y)
    raise ValueError(name)


class ConfusionMatrix:
    def __init__(self, n, ignore_index):
        self.n, self.ignore = n, ignore_index
        self.m = np.zeros((n, n), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1).flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        k = t != self.ignore
        self.m += np.bincount(self.n * t[k] + pred[k],
                              minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        tp = np.diag(self.m)
        denom = self.m.sum(0) + self.m.sum(1) - tp
        return tp / np.maximum(denom, 1)

    def summary(self):
        iou = self.iou()
        return {"mIoU": float(iou.mean()),
                "pixel_acc": float(np.diag(self.m).sum() / max(self.m.sum(), 1)),
                "iou_per_class": iou.tolist()}
