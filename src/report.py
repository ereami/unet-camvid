"""Tabelas e figuras para o relatório, a partir de results.csv e dos logs."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import CLASSES, IGNORE_INDEX, MEAN, STD

# paleta oficial do CamVid (11 classes) + Void em preto
PALETTE = np.array([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
                    [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
                    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]], np.uint8)


def summary_table(out_dir):
    """mIoU de teste: média ± desvio-padrão sobre seeds, por variante."""
    df = pd.read_csv(os.path.join(out_dir, "results.csv"))
    g = df.groupby("name")
    t = pd.DataFrame({
        "n_seeds": g.size(),
        "params_M": g["params_M"].first().round(2),
        "test_mIoU": g["test_mIoU"].mean().round(3).astype(str) + " ± "
                     + g["test_mIoU"].std().fillna(0).round(3).astype(str),
        "pixel_acc": g["test_pixel_acc"].mean().round(3),
        "train_min": g["train_min"].mean().round(1),
    })
    return t


def per_class_table(out_dir):
    df = pd.read_csv(os.path.join(out_dir, "results.csv"))
    return df.groupby("name")[[f"iou_{c}" for c in CLASSES]].mean().round(3)


def plot_curves(out_dir, variants=None, seed=0, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for log in sorted(glob.glob(f"{out_dir}/runs/*_s{seed}/log.csv")):
        name = os.path.basename(os.path.dirname(log)).rsplit("_s", 1)[0]
        if variants and name not in variants:
            continue
        d = pd.read_csv(log)
        ax[0].plot(d.epoch + 1, d.train_loss, label=name, lw=1.2)
        v = d.dropna(subset=["val_mIoU"])
        ax[1].plot(v.epoch + 1, v.val_mIoU, label=name, lw=1.2)
    ax[0].set(xlabel="época", ylabel="loss de treino")
    ax[1].set(xlabel="época", ylabel="mIoU validação")
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_examples(model, dataset, idx, device, path=None):
    """Imagem | anotação | predição, para os índices escolhidos."""
    model.eval()
    fig, ax = plt.subplots(len(idx), 3, figsize=(9, 2.3 * len(idx)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(idx):
        x, y = dataset[i]
        pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        img = np.clip(x.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
        for c, (im, t) in enumerate([(img, "imagem"), (PALETTE[y.numpy()], "anotação"),
                                     (PALETTE[pred], "predição")]):
            ax[r, c].imshow(im); ax[r, c].set_axis_off()
            if r == 0:
                ax[r, c].set_title(t, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
