"""Псевдоразметка v2 (noisy student): учитель = среднее активаций нескольких версий модели,
исправление октавы темпа по BPM автора, сильные доли маскируются, если сетку правили.

python data_prep/pseudo_label2.py runs/r3/best.pt runs/r3/s12000.pt
Источники: ccMixter (data/raw/ccmixter) и incompetech (data/raw/kml), CC BY.
Пропускаем: тестовые треки, три трека сравнения с Qildo.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FPS, dequantize  # noqa: E402
from infer import activations, load_model  # noqa: E402
from postprocess import BarHMM, trim_silence  # noqa: E402

import os  # noqa: E402

OUT = ROOT / os.environ.get("PSEUDO_OUT", "data/real/pseudo2")  # v2: PSEUDO_OUT=data/real/pseudo3
QILDO_TITLES = {"Funky Chunk", "Local Forecast - Elevator", "Sneaky Snitch"}


def sources():
    ccm_test = {l.strip() for l in open(ROOT / "data/real/ccm_test_ids.txt") if l.strip()}
    for mf in (ROOT / "data/raw/ccmixter").glob("meta*.jsonl"):
        for l in open(mf, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in ccm_test:
                yield "ccm_" + r["id"], ROOT / "data/raw/ccmixter/mel" / f"{r['id']}.npz", r.get("bpm")
    for mf in (ROOT / "data/raw/kml").glob("meta_*.jsonl"):
        for l in open(mf, encoding="utf-8"):
            r = json.loads(l)
            if not r["test"] and r["title"] not in QILDO_TITLES:
                yield "kml_" + r["id"], ROOT / "data/raw/kml/mel" / f"{r['id']}.npz", r.get("bpm")


def main(weights):
    models = [load_model(w) for w in weights]
    hmm = BarHMM()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"kept": 0, "octave_fixed": 0, "rejected": 0, "no_bpm": 0}
    seen = set()
    for sid, path, bpm_tag in sources():
        if sid in seen or not path.exists() or (OUT / f"{sid}.npz").exists():
            continue
        seen.add(sid)
        try:
            bpm_tag = float(bpm_tag)
        except (TypeError, ValueError):
            stats["no_bpm"] += 1
            continue
        mel = np.load(path)["mel"]
        act = np.mean([activations(m, d, dequantize(mel)) for m, d in models], axis=0)
        ev = trim_silence(hmm.decode(act[:, 0], act[:, 1]), act[:, 0])
        if len(ev) < 32:
            stats["rejected"] += 1
            continue
        t = np.array([e[0] for e in ev])
        b = np.array([e[1] for e in ev])
        ratio = (60 / np.median(np.diff(t))) / bpm_tag
        mask_db = False
        if abs(ratio - 1) < 0.04:
            pass
        elif abs(ratio - 2) / 2 < 0.04:  # мы вдвое быстрее автора: берём каждую вторую долю (фаза с большей активацией)
            fr = np.clip((t * FPS).astype(int), 0, len(act) - 1)
            ph = int(np.argmax([act[fr[k::2], 0].mean() for k in (0, 1)]))
            t = t[ph::2]
            b = np.ones(len(t), int)
            mask_db = True
            stats["octave_fixed"] += 1
        elif abs(ratio - 0.5) / 0.5 < 0.04:  # мы вдвое медленнее: вставляем середины
            t = np.sort(np.concatenate([t, (t[:-1] + t[1:]) / 2]))
            b = np.ones(len(t), int)
            mask_db = True
            stats["octave_fixed"] += 1
        else:
            stats["rejected"] += 1
            continue
        fr = np.clip((t * FPS).astype(int), 0, len(act) - 1)
        if act[fr, 0].mean() < 0.3:
            stats["rejected"] += 1
            continue
        np.savez_compressed(OUT / f"{sid}.npz", mel=mel, beats=t, downbeats=t[b == 1] if not mask_db else t[:0],
                            nums=b, mask_db=mask_db)
        stats["kept"] += 1
        if sum(stats.values()) % 100 == 0:
            print(stats, flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
