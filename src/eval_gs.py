"""Все метрики на GuitarSet, игрок 05 (не участвовал в обучении).

python src/eval_gs.py weights/beat_model.pt [ещё веса ...]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import score_one  # noqa: E402
from features import dequantize  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402

KEYS = ["beat_F", "downbeat_F", "CMLt", "AMLt", "missed_start", "tempo_x2_frac", "downbeat_wrong_frac"]


def main(paths):
    files = sorted((ROOT / "data/real/gs_test").glob("*.npz"))
    only = ROOT / "preds/librosa_gs_test"  # librosa считали на 30 записях — сравниваем на тех же
    if "--same-as-librosa" in paths:
        paths = [p for p in paths if p != "--same-as-librosa"]
        keep = {f.stem for f in only.glob("*.json")}
        files = [f for f in files if f.stem in keep]
    for p in paths:
        m, dev = load_model(p)
        ss = []
        for f in files:
            est = beats_from_act(activations(m, dev, dequantize(np.load(f)["mel"])))
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(est, tf)
            ss.append(score_one(ROOT / "data/real/gs_test_refs" / (f.stem + ".json"), tf.name))
            Path(tf.name).unlink()
        print(p, len(ss), {k: round(float(np.mean([s[k] for s in ss])), 3) for k in KEYS}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
