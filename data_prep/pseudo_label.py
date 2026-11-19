"""Псевдоразметка реальных CC BY песен с ccMixter своей моделью + фильтр по BPM автора.

python data_prep/pseudo_label.py runs/r2/best.pt
Берём трек, только если медианный темп сетки совпадает с bpm из метаданных автора (±4%)
и активация на долях уверенная. Пишем в data/real/ccm_pseudo/<id>.npz (mel, beats, downbeats).
Тестовые треки (data/real/ccm_test_ids.txt) пропускаем.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FPS, dequantize  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402

MEL = ROOT / "data/raw/ccmixter/mel"
OUT = ROOT / "data/real/ccm_pseudo"


def main(weights):
    model, dev = load_model(weights)
    test_ids = set()
    tf = ROOT / "data/real/ccm_test_ids.txt"
    if tf.exists():
        test_ids = {l.strip() for l in tf.open() if l.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    kept = rej = 0
    for line in open(ROOT / "data/raw/ccmixter/meta.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in test_ids or not (MEL / f"{r['id']}.npz").exists():
            continue
        try:
            bpm_tag = float(r["bpm"])
        except (TypeError, ValueError):
            rej += 1
            continue
        mel = np.load(MEL / f"{r['id']}.npz")["mel"]
        act = activations(model, dev, dequantize(mel))
        grid = beats_from_act(act)
        if len(grid) < 32:
            rej += 1
            continue
        t = np.array([g["time"] for g in grid])
        b = np.array([g["beat"] for g in grid])
        bpm = 60 / np.median(np.diff(t))
        ratio = bpm / bpm_tag
        ok_tempo = any(abs(ratio - k) / k < 0.04 for k in (1.0, 2.0, 0.5))
        conf = float(act[np.clip((t * FPS).astype(int), 0, len(act) - 1), 0].mean())
        if not ok_tempo or conf < 0.35:
            rej += 1
            continue
        np.savez_compressed(OUT / f"{r['id']}.npz", mel=mel, beats=t, downbeats=t[b == 1], nums=b)
        kept += 1
    print(json.dumps({"kept": kept, "rejected": rej}))


if __name__ == "__main__":
    main(sys.argv[1])
