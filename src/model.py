"""U-Net (Ronneberger et al., 2015) parametrizável, treinada do zero.

base:  número de filtros do primeiro nível (16 / 32 / 64) -> ablação de capacidade
skip:  liga/desliga as skip connections -> ablação arquitetural
       (skip=False vira um encoder-decoder puro, no estilo SegNet sem índices)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, cin, cout, skip=True):
        super().__init__()
        self.skip = skip
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin // 2 + (cout if skip else 0), cout)

    def forward(self, x, skip_feat):
        x = self.up(x)
        if self.skip:
            x = torch.cat([skip_feat, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=11, base=32, skip=True):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]          # ex.: 32, 64, 128, 256, 512
        self.inc = DoubleConv(in_ch, c[0])
        self.downs = nn.ModuleList(DoubleConv(c[i], c[i + 1]) for i in range(4))
        self.ups = nn.ModuleList(Up(c[i + 1], c[i], skip) for i in reversed(range(4)))
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        # 4 poolings exigem lados múltiplos de 16 (360 -> 368)
        ph, pw = (-h) % 16, (-w) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(F.max_pool2d(feats[-1], 2)))
        x = feats.pop()
        for up in self.ups:
            x = up(x, feats.pop())
        return self.head(x)[..., :h, :w]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
