"""Desenho experimental: cada variante muda UM fator em relação ao baseline.

  Q1 (arquitetura)       baseline  vs  no_skip
  Q2 (desbalanceamento)  baseline (CE) vs  wce  vs  ce_dice
  Q3 (regularização)     baseline  vs  no_aug
  Q4 (capacidade, opc.)  base16  vs  baseline (32)  vs  base64
"""
from .train import Config

VARIANTS = {
    "baseline": {},
    "no_skip":  {"skip": False},
    "wce":      {"loss": "wce"},
    "ce_dice":  {"loss": "ce_dice"},
    "no_aug":   {"augment": False},
    "base16":   {"base": 16},
    "base64":   {"base": 64},
}

SEEDS = [0, 1, 2]


def configs(variants=None, seeds=SEEDS, **overrides):
    for name in variants or VARIANTS:
        for s in seeds:
            yield Config(name=name, seed=s, **{**VARIANTS[name], **overrides})
