"""CamVid (divisão do SegNet): 367 treino / 101 validação / 233 teste, 360x480.

Rótulos 0..10 = classes; 11 = Void (ignorado na perda e nas métricas).
Todo o dataset é carregado na RAM (~700 imagens pequenas) para eliminar
o gargalo de disco no Colab.
"""
import glob
import os
import subprocess

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
NUM_CLASSES = len(CLASSES)   # 11
IGNORE_INDEX = 11            # Void

# Estatísticas do ImageNet são só uma normalização razoável; não há pré-treino.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REPO = "https://github.com/alexgkendall/SegNet-Tutorial.git"


def download_camvid(dest="/content/data"):
    """Baixa apenas a pasta CamVid do repositório SegNet-Tutorial (sparse checkout)."""
    root = os.path.join(dest, "SegNet-Tutorial", "CamVid")
    if os.path.isdir(os.path.join(root, "train")):
        return root
    os.makedirs(dest, exist_ok=True)
    cwd = os.path.join(dest, "SegNet-Tutorial")
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--sparse", REPO, cwd], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "CamVid"], cwd=cwd, check=True)
    return root


def _load_split(root, split):
    imgs = sorted(glob.glob(os.path.join(root, split, "*.png")))
    x = np.stack([np.array(Image.open(f).convert("RGB")) for f in imgs])
    y = np.stack([np.array(Image.open(f.replace(f"/{split}/", f"/{split}annot/")))
                  for f in imgs])
    return x, y.astype(np.int64)


def class_weights(y, mode="median_freq"):
    """Pesos por classe via median frequency balancing (Eigen & Fergus, 2015)."""
    counts = np.bincount(y.ravel(), minlength=IGNORE_INDEX + 1)[:NUM_CLASSES]
    freq = counts / counts.sum()
    w = np.median(freq) / freq
    return torch.tensor(w, dtype=torch.float32)


class CamVid(Dataset):
    """crop: tamanho do recorte aleatório no treino (None = imagem inteira).
    augment: liga flip horizontal + brilho/contraste. O recorte aleatório é
    parte do protocolo de treino (custo/época) e fica ativo nas duas condições,
    para que a ablação de augmentation não seja confundida com tamanho de entrada.
    """

    def __init__(self, root, split="train", crop=None, augment=False):
        self.x, self.y = _load_split(root, split)
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 255.0
        lab = self.y[i]
        if self.crop:
            h, w = lab.shape
            c = self.crop
            top = np.random.randint(0, h - c + 1)
            left = np.random.randint(0, w - c + 1)
            img = img[top:top + c, left:left + c]
            lab = lab[top:top + c, left:left + c]
        if self.augment:
            if np.random.rand() < 0.5:                     # flip horizontal
                img, lab = img[:, ::-1], lab[:, ::-1]
            alpha = np.random.uniform(0.8, 1.2)            # contraste
            beta = np.random.uniform(-0.1, 0.1)            # brilho
            img = np.clip(alpha * img + beta, 0, 1)
        img = (img - MEAN) / STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        lab = torch.from_numpy(np.ascontiguousarray(lab))
        return img, lab
