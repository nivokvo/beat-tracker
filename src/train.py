"""Обучение BeatModel с нуля.

python src/train.py --data data/synth/train data/real/train --val data/synth/val --out runs/r1 --steps 30000
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import score_one  # noqa: E402
from features import FPS, dequantize, times_to_frames  # noqa: E402
from infer import activations, beats_from_act  # noqa: E402
from model import BeatModel, shift_tolerant_bce  # noqa: E402
from postprocess import BarHMM  # noqa: E402

CROP = 1500
STRETCH_P = float(__import__("os").environ.get("STRETCH_P", "0"))  # доля кусков с растяжкой по темпу


def stretch(mel, y, r):
    """Мел (T,F) и цели (T,2) с темпом ×1/r: новая длина round(T/r); положительные кадры переносятся, −1 (маска) сохраняется."""
    T = len(mel)
    T2 = max(1, int(round(T / r)))
    pos = np.arange(T2) * (T - 1) / max(T2 - 1, 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None]
    mel2 = (mel[i0] * (1 - w) + mel[i1] * w).astype(np.float32)
    y2 = np.zeros((T2, y.shape[1]), np.float32)
    for c in range(y.shape[1]):
        if np.all(y[:, c] < 0):
            y2[:, c] = -1
            continue
        idx = np.flatnonzero(y[:, c] > 0.5)
        y2[np.clip(np.round(idx * (T2 - 1) / max(T - 1, 1)).astype(int), 0, T2 - 1), c] = 1
        y2[np.round(np.flatnonzero(y[:, c] < 0) * (T2 - 1) / max(T - 1, 1)).astype(int), c] = -1  # паддинг/маска
    return mel2, y2


def targets(n, beats, downbeats):
    y = np.zeros((n, 2), np.float32)
    y[times_to_frames(beats, n), 0] = 1
    y[times_to_frames(downbeats, n), 1] = 1
    return y


class Crops(Dataset):
    """Случайные 30-секундные куски. Папки пересканируются — новые песни из рендера подхватываются на лету."""

    def __init__(self, dirs, weights=None, length=10**7, aug=True, cache_size=120):
        self.dirs = list(dirs)
        w = weights or [1.0] * len(self.dirs)
        self.w_all = np.array(w[: len(self.dirs)], dtype=float)
        self.length = length
        self.aug = aug
        self.cache = {}
        self.cache_size = cache_size
        self.calls = 0
        self._rescan()

    def _rescan(self):
        self.groups = [sorted(Path(d).glob("*.npz")) for d in self.dirs]
        keep = [i for i, g in enumerate(self.groups) if g]
        self.groups = [self.groups[i] for i in keep]
        w = self.w_all[keep]
        self.w = w / w.sum()

    def __len__(self):
        return self.length

    def _load(self, f):
        if f not in self.cache:
            d = np.load(f)
            mel = d["mel"]  # в кэше держим uint8, чтобы не съесть RAM
            y = targets(len(mel), d["beats"], d["downbeats"])
            if "mask_db" in d.files and bool(d["mask_db"]):
                y[:, 1] = -1  # сильные доли неизвестны (сетку правили по BPM автора) — не учим
            self.cache[f] = (mel, y)
            if len(self.cache) > self.cache_size:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[f]

    def __getitem__(self, i):
        self.calls += 1
        if self.calls % 500 == 0:
            self._rescan()
        rng = random.Random()
        g = self.groups[np.random.choice(len(self.groups), p=self.w)]
        mel, y = self._load(g[rng.randrange(len(g))])
        # растяжка по темпу (как в Beat This!): кусок длиной CROP·r кадров сжимается/растягивается в CROP,
        # доли пересчитываются. Темп всех источников (Slakh, живые барабаны, гитара, псевдоразметка) гуляет ±25 %.
        r = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))) if self.aug and rng.random() < STRETCH_P else 1.0
        src = int(round(CROP * r))
        n = len(mel)
        if n > src:
            # чаще берём начало песни — важно для «сетки с первой доли»
            s = 0 if rng.random() < 0.15 else rng.randrange(n - src)
            mel, y = mel[s: s + src], y[s: s + src]
        mel = dequantize(mel).copy()
        y = y.copy()
        if r != 1.0:
            mel, y = stretch(mel, y, r)
            mel, y = mel[:CROP], y[:CROP]  # короткий трек при r < 1 мог стать длиннее куска
        if self.aug:
            for _ in range(2):  # маски по частоте
                w = rng.randrange(0, 16)
                f0 = rng.randrange(0, 128 - w)
                mel[:, f0: f0 + w] = 0
            if rng.random() < 0.5:  # сдвиг по высоте ±2 мел-бина
                mel = np.roll(mel, rng.randint(-2, 2), axis=1)
            mel *= rng.uniform(0.8, 1.2)
        if len(mel) < CROP:
            pad = CROP - len(mel)
            mel = np.pad(mel, ((0, pad), (0, 0)))
            y = np.pad(y, ((0, pad), (0, 0)), constant_values=-1)
        return torch.from_numpy(mel), torch.from_numpy(y)


def validate(model, device, val_dir, refs_dir, n=40):
    files = sorted(Path(val_dir).glob("*.npz"))[:n]
    hmm = BarHMM()
    scores = []
    model.eval()
    import tempfile
    for f in files:
        d = np.load(f)
        act = activations(model, device, dequantize(d["mel"]))
        est = beats_from_act(act, hmm)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(est, tf)
        scores.append(score_one(Path(refs_dir) / (f.stem + ".json"), tf.name))
        Path(tf.name).unlink()
    model.train()
    keys = scores[0].keys()
    return {k: float(np.mean([s[k] for s in scores])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_refs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--resume")
    ap.add_argument("--init", help="только веса, без оптимизатора и шага")
    ap.add_argument("--compile", action="store_true", help="torch.compile (triton-windows): шаг ~15 % быстрее, памяти меньше")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val_refs = a.val_refs or str(Path(a.val).parent / (Path(a.val).name + "_refs"))
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = dict(d=a.d, layers=a.layers, heads=8, drop=0.1)
    model = BeatModel(**cfg).to(device)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    step0 = 0
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model"])
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]

    def lr_at(s):
        warm = 1000
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / (a.steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=step0 - 1 if step0 else -1)
    fwd = torch.compile(model) if a.compile else model  # validate() и сохранение работают с исходной model
    dl = DataLoader(Crops(a.data, a.weights), batch_size=a.bs, num_workers=a.workers,
                    persistent_workers=True, pin_memory=True)
    pw = torch.tensor([1.0, 2.0], device=device)
    best = -1
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    run = 0.0
    it = iter(dl)
    for step in range(step0 + 1, a.steps + 1):
        mel, y = next(it)
        mel, y = mel.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(mel)
        loss = shift_tolerant_bce(logits.float(), y, tol=3, pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > step0 + 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, device, a.val, val_refs)
            v["step"] = step
            v["loss"] = run
            log.write(json.dumps(v) + "\n")
            log.flush()
            print("VAL", {k: round(x, 4) for k, x in v.items()}, flush=True)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}
            torch.save(ck, out / "last.pt")
            score = v["beat_F"] + v["downbeat_F"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val": v}, out / "best.pt")


if __name__ == "__main__":
    main()
