"""Псевдоразметка рока/попа из FMA (CC BY/CC0) двумя учителями с фильтром согласия.

python data_prep/pseudo_fma.py runs/r7/s10000.pt runs/r6/s14000.pt [--workers 8]
1) Активации обоих учителей на GPU → кэш runs/acts_fma/<учитель>/<id>.npz.
2) Каждый учитель декодируется своей HMM; трек берём, только если сетки согласны
   (доли F ≥ 0,9 и сильные доли F ≥ 0,8 при ±70 мс) и темп одинаков. Разметка — из среднего активаций.
BPM автора у FMA нет, поэтому согласие двух разных моделей — единственная защита от ошибок «вдвое».
Выход: data/real/fma_pseudo/<id>.npz (mel, beats, downbeats, nums) — тот же формат, что у pseudo3.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import mir_eval
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import dequantize  # noqa: E402
from postprocess import BarHMM, trim_silence  # noqa: E402

MEL = Path("D:/ml-qildo-data/raw/fma/mel")
import os  # noqa: E402
OUT = ROOT / os.environ.get("PSEUDO_FMA_OUT", "data/real/fma_pseudo")
CACHE = ROOT / "runs/acts_fma"


def cache_acts(weights):
    from infer import activations, load_model
    for w in weights:
        tag = "_".join(Path(w).with_suffix("").parts[-2:])
        d = CACHE / tag
        d.mkdir(parents=True, exist_ok=True)
        m, dev = load_model(w)
        for f in sorted(MEL.glob("*.npz")):
            o = d / f.name
            if not o.exists():
                np.savez(o, act=activations(m, dev, dequantize(np.load(f)["mel"])).astype(np.float16))
        yield tag


def decode(act):
    from infer import select_meter  # выбор размера по контрасту сильных долей (иначе HMM почти всегда ставит 4)
    ev = BarHMM().decode(act[:, 0], act[:, 1])
    ev = trim_silence(select_meter(ev, act[:, 0], act[:, 1], act), act[:, 0])
    return np.array([t for t, _ in ev]), np.array([b for _, b in ev])


def judge_one(args):
    sid, tags = args
    out = OUT / f"{sid}.npz"
    if out.exists():
        return "exists"
    acts = [np.load(CACHE / t / f"{sid}.npz")["act"].astype(np.float32) for t in tags]
    grids = [decode(a) for a in acts]
    (t1, b1), (t2, b2) = grids[0], grids[1]
    if len(t1) < 32 or len(t2) < 32:
        return "short"
    fb = mir_eval.beat.f_measure(t1, t2, 0.07)
    fd = mir_eval.beat.f_measure(t1[b1 == 1], t2[b2 == 1], 0.07)
    tempo_ratio = np.median(np.diff(t1)) / np.median(np.diff(t2))
    if fb < 0.9 or abs(tempo_ratio - 1) > 0.03:
        return "disagree_beats"
    if fd < 0.8:
        return "disagree_downbeats"
    t, b = decode(np.mean(acts, axis=0))
    OUT.mkdir(parents=True, exist_ok=True)
    mel = np.load(MEL / f"{sid}.npz")["mel"]
    np.savez_compressed(out, mel=mel, beats=t, downbeats=t[b == 1], nums=b)
    return "kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs=2)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tags = list(cache_acts(a.weights))
    print("acts cached", tags, flush=True)
    ids = sorted(f.stem for f in (CACHE / tags[0]).glob("*.npz") if (CACHE / tags[1] / f.name).exists())
    stats = {}
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(judge_one, [(s, tags) for s in ids], chunksize=4)):
            stats[r] = stats.get(r, 0) + 1
            if i % 100 == 0:
                print(i, len(ids), stats, flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
