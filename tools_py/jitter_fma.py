"""«Плавание» сетки на настоящем роке FMA — мерка того, что слышно на прослушивании с метрономом.

python tools_py/jitter_fma.py [--n 300]
Для каждой настройки постобработки: доля долей, длина которых отличается от медианы соседних (±4 доли) больше чем на 6 %
(Кристина слышит такие места как «замедляется / ускоряется»), и число смен сильной доли.
Активации — из кэша псевдоразметки (runs/acts_fma/<модель>/).
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CFGS = [  # (имя, λ, TWO_PASS)
    ("lam30", 30, None), ("lam20", 20, None), ("lam10", 10, None),
    ("lam30+tp0.2", 30, "0.2,1,0.9"), ("lam30+tp0.15", 30, "0.15,1,0.9"), ("lam30+tp0.1x2", 30, "0.1,2,0.9"), ("lam20+tp0.15", 20, "0.15,1,0.9"), ("lam10+tp0.15", 10, "0.15,1,0.9"), ("lam10+tp0.1x2", 10, "0.1,2,0.9"),
]


def wobble(t):
    ibi = np.diff(t)
    if len(ibi) < 12:
        return np.nan
    med = np.array([np.median(ibi[max(0, i - 4): i + 5]) for i in range(len(ibi))])
    return float(np.mean(np.abs(ibi / med - 1) > 0.06))


def one(args):
    path, lam, tp = args
    if tp:
        os.environ["TWO_PASS"] = tp
    else:
        os.environ.pop("TWO_PASS", None)
    from infer import second_pass
    from postprocess import BarHMM, trim_silence
    a = np.load(path)["act"].astype(np.float32)
    ev = BarHMM(lam=lam).decode(a[:, 0], a[:, 1])
    if tp:
        s, w, mp = map(float, tp.split(","))
        ev = second_pass(ev, a[:, 0], a[:, 1], s, w, mp)
    ev = trim_silence(ev, a[:, 0])
    t = np.array([e[0] for e in ev])
    return wobble(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="r7_s10000")
    a = ap.parse_args()
    files = sorted((ROOT / "runs/acts_fma" / a.model).glob("*.npz"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), min(a.n, len(files)), replace=False)]
    only = os.environ.get("JIT_ONLY")
    with Pool(int(os.environ.get("JIT_WORKERS", "6"))) as p:
        for name, lam, tp in CFGS:
            if only and name not in only.split(";"):
                continue
            w = np.array(p.map(one, [(str(f), lam, tp) for f in files]))
            w = w[~np.isnan(w)]
            print(f"{a.model} {name}: wobble mean {w.mean():.4f} median {np.median(w):.4f} tracks>5% {np.mean(w > 0.05):.3f}", flush=True)


if __name__ == "__main__":
    main()
