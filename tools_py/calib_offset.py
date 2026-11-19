"""Калибровка сдвига сетки на ОТЛОЖЕННЫХ (val) данных: пики сети систематически опаздывают на ~20 мс.

python tools_py/calib_offset.py runs/r7/s8000.pt [ещё веса через +]
Для каждого val-набора: медиана ошибки (оценка − ближайшая опорная доля, |e|<70 мс) и F по долям
при сдвигах −40…+10 мс. Тестовые наборы не используются.
"""
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import dequantize  # noqa: E402
from infer import activations, load_model  # noqa: E402
from postprocess import BarHMM, trim_silence  # noqa: E402

VAL = [("slakh_val", "data/slakh/val", "data/slakh/val_refs", 50), ("groove_val", "data/groove/val", "data/groove/val_refs", 46),
       ("synth_val", "data/synth/val", "data/synth/val_refs", 40)]


def main(spec):
    ws = spec.split("+")
    models = [load_model(w) for w in ws]
    h = BarHMM()
    allerr, rows = [], {}
    for name, d, refd, n in VAL:
        files = sorted((ROOT / d).glob("*.npz"))[:n]
        errs, pairs = [], []
        for f in files:
            mel = dequantize(np.load(f)["mel"])
            a = np.mean([activations(m, dev, mel) for m, dev in models], axis=0)
            est = np.array([t for t, _ in trim_silence(h.decode(a[:, 0], a[:, 1]), a[:, 0])])
            ref = np.array([r["time"] for r in json.loads((ROOT / refd / (f.stem + ".json")).read_text())])
            if len(est) < 4 or len(ref) < 4:
                continue
            j = np.clip(np.searchsorted(ref, est), 1, len(ref) - 1)
            near = np.where(np.abs(ref[j] - est) < np.abs(ref[j - 1] - est), ref[j], ref[j - 1])
            e = est - near
            errs.extend(e[np.abs(e) < 0.07])
            pairs.append((ref, est))
        errs = np.array(errs)
        allerr.extend(errs)
        fs = {off: np.mean([mir_eval.beat.f_measure(r, e + off, 0.07) for r, e in pairs]) for off in np.arange(-0.04, 0.0101, 0.005)}
        best = max(fs, key=fs.get)
        rows[name] = (np.median(errs) * 1000, fs[0.0] if 0.0 in fs else fs[min(fs, key=abs)], best * 1000, fs[best])
        print(f"{name:11s} median err {np.median(errs) * 1000:+.1f} ms | F@0 {rows[name][1]:.4f}  best offset {best * 1000:+.0f} ms F {fs[best]:.4f}", flush=True)
    print(f"ALL median err {np.median(allerr) * 1000:+.1f} ms -> recommended BEAT_OFFSET {-np.median(allerr):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
