"""Своя модель долей: свёрточный стем по лог-мелу + трансформер по времени с RoPE.

Идея архитектуры — из статьи Beat This! (Foscarin et al., ISMIR 2024): свёртки, потом
трансформер, два выхода (доля, сильная доля), устойчивый к сдвигу лосс. Код написан с нуля.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, base=10000.0):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    half = D // 2
    freqs = torch.exp(-math.log(base) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
        self.pdrop = drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.pdrop if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.n2(x)))


class ConvStem(nn.Module):
    """(B, T, 128) → (B, T, d). Свёртки по (частота, время), пулинг только по частоте."""

    def __init__(self, d, n_mels=128):
        super().__init__()
        # первая свёртка сразу сжимает частоту в 4 раза (128 → 32), дальше пулинг ×2 трижды → 4
        layers = [nn.Conv2d(1, 32, (4, 3), stride=(4, 1), padding=(0, 1)), nn.BatchNorm2d(32), nn.GELU()]
        ch = [32, 64, 128, 128]
        for i in range(3):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(ch[i + 1]),
                nn.GELU(),
                nn.MaxPool2d((2, 1)),
            ]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(ch[-1] * (n_mels // 32), d)
        self.inorm = nn.BatchNorm1d(n_mels)

    def forward(self, x):
        x = self.inorm(x.transpose(1, 2)).unsqueeze(1)  # (B,1,F,T)
        x = self.net(x)  # (B,C,F/16,T)
        B, C, Fq, T = x.shape
        return self.proj(x.reshape(B, C * Fq, T).transpose(1, 2))


class BeatModel(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.stem = ConvStem(d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)  # логиты: доля, сильная доля

    def forward(self, mel):
        x = self.stem(mel.float())
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))  # (B, T, 2)


def shift_tolerant_bce(logits, target, tol=3, pos_weight=None):
    """Лосс, терпимый к сдвигу разметки на ±tol кадров (±60 мс при 50 fps).

    Положительный кадр штрафуется по максимуму предсказания в окне ±tol;
    отрицательные кадры в этом окне вокруг разметки не штрафуются вовсе.
    logits, target: (B, T, C), target ∈ {0,1}; на кадрах padding target = -1.
    """
    valid = target >= 0
    t = target.clamp(min=0)
    k = 2 * tol + 1
    pooled = F.max_pool1d(logits.transpose(1, 2), k, 1, tol).transpose(1, 2)
    near = F.max_pool1d(t.transpose(1, 2), k, 1, tol).transpose(1, 2)
    pos = t > 0.5
    neg = (near < 0.5) & valid
    pw = pos_weight if pos_weight is not None else torch.ones(logits.shape[-1], device=logits.device)
    lp = F.binary_cross_entropy_with_logits(pooled, torch.ones_like(pooled), reduction="none") * pw
    ln = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    n = valid.sum().clamp(min=1)
    return (lp[pos & valid].sum() + ln[neg].sum()) / n
