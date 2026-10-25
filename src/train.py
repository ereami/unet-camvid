"""Uma função de treino parametrizada para todas as variantes do estudo.

Protocolo:
  * seleção do melhor checkpoint pelo mIoU de VALIDAÇÃO;
  * o TESTE é avaliado uma única vez, com o melhor checkpoint, ao final;
  * log por época em CSV (curvas), resultado final anexado a results.csv;
  * checkpoint salvo a cada época -> retomada automática se o Colab cair.
"""
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, IGNORE_INDEX, NUM_CLASSES, CamVid, class_weights
from .losses_metrics import ConfusionMatrix, build_loss
from .model import UNet, count_params


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 0
    base: int = 32            # largura da U-Net
    skip: bool = True         # skip connections
    loss: str = "ce"          # 'ce' | 'wce' | 'ce_dice'
    augment: bool = True      # flip + brilho/contraste
    crop: int = 256           # recorte aleatório no treino
    epochs: int = 120
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 2
    num_workers: int = 2


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(wid):
    np.random.seed(torch.initial_seed() % 2 ** 32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            cm.update(model(x), y)
    return cm.summary()


def run(cfg: Config, data_root: str, out_dir: str, datasets=None):
    """datasets: tupla (train, val, test) já carregada, para reaproveitar a RAM
    entre execuções. O dataset de treino recebe crop/augment de cfg."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(out_dir, "runs", f"{cfg.name}_s{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path, best_path = f"{run_dir}/last.pt", f"{run_dir}/best.pt"
    log_path = f"{run_dir}/log.csv"
    results_path = os.path.join(out_dir, "results.csv")

    if os.path.exists(f"{run_dir}/final.json"):
        print(f"[{cfg.name} s{cfg.seed}] já concluído, pulando.")
        return json.load(open(f"{run_dir}/final.json"))

    set_seed(cfg.seed)
    if datasets is None:
        train_ds = CamVid(data_root, "train")
        val_ds, test_ds = CamVid(data_root, "val"), CamVid(data_root, "test")
    else:
        train_ds, val_ds, test_ds = datasets
    train_ds.crop, train_ds.augment = cfg.crop, cfg.augment

    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          worker_init_fn=_worker_init, generator=g,
                          persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, 4, num_workers=cfg.num_workers)
    test_dl = DataLoader(test_ds, 4, num_workers=cfg.num_workers)

    model = UNet(n_classes=NUM_CLASSES, base=cfg.base, skip=cfg.skip).to(device)
    weights = class_weights(train_ds.y).to(device)
    criterion = build_loss(cfg.loss, NUM_CLASSES, IGNORE_INDEX, weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start, best_miou, log = 0, -1.0, []
    if os.path.exists(ckpt_path):                       # retomada
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best_miou = ck["epoch"] + 1, ck["best_miou"]
        log = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) else []
        print(f"Retomando de época {start}")

    print(f"[{cfg.name} s{cfg.seed}] {count_params(model)/1e6:.2f}M parâmetros, {device}")
    t0 = time.time()
    for epoch in range(start, cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        row = {"epoch": epoch, "train_loss": tot / n, "lr": sched.get_last_lr()[0],
               "elapsed_min": (time.time() - t0) / 60}
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            val = evaluate(model, val_dl, device)
            row.update(val_mIoU=val["mIoU"], val_pixel_acc=val["pixel_acc"])
            if val["mIoU"] > best_miou:
                best_miou = val["mIoU"]
                torch.save(model.state_dict(), best_path)
            print(f"ep {epoch+1:3d} loss {row['train_loss']:.3f} "
                  f"val mIoU {val['mIoU']:.3f} ({row['elapsed_min']:.1f} min)")
        log.append(row)
        pd.DataFrame(log).to_csv(log_path, index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_miou": best_miou}, ckpt_path)

    # Teste: uma única vez, com o melhor checkpoint de validação
    model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, test_dl, device)
    final = {**asdict(cfg), "params_M": count_params(model) / 1e6,
             "best_val_mIoU": best_miou, "test_mIoU": test["mIoU"],
             "test_pixel_acc": test["pixel_acc"],
             "train_min": log[-1]["elapsed_min"] if log else None,
             **{f"iou_{c}": v for c, v in zip(CLASSES, test["iou_per_class"])}}
    json.dump(final, open(f"{run_dir}/final.json", "w"), indent=2)
    pd.DataFrame([final]).to_csv(results_path, mode="a", index=False,
                                 header=not os.path.exists(results_path))
    print(f"TESTE mIoU {test['mIoU']:.3f} | pixel acc {test['pixel_acc']:.3f}")
    return final
