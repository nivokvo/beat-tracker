"""Сравнение сетки Qildo (с YouTube-версии) с нашими моделями на той же песне (mp3 incompetech, мел из data/raw/kml/mel).

python tools_py/qildo_compare2.py qildo_cmp/day2/qildo_analysis.json runs/r4/s10000.pt runs/r6/s14000.pt
Сдвиг между версиями ищется по максимуму совпадения долей (±0,25 с). Эталона нет — смотрим:
темп против BPM автора, такты не той длины, где сетка начинается и кончается, согласие с Qildo.
"""
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools_py"))
from features import FPS, dequantize  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402
from qildo_compare import bar_stats, qildo_grid  # noqa: E402


def offset(qt, ot):
    best = (0.0, -1)
    for d in np.arange(-0.25, 0.2501, 0.02):  # версии отличаются на сотые доли секунды; шире — риск сдвига на целую долю
        f = mir_eval.beat.f_measure(qt + d, ot, 0.07)
        if f > best[1]:
            best = (round(float(d), 2), f)
    return best[0]


def summary(grid, dur, author):
    t = np.array([g["time"] for g in grid])
    bpm = 60 / np.median(np.diff(t)) if len(t) > 2 else 0
    bs = bar_stats(grid)
    return {"bpm": round(float(bpm), 1), "vs_author": round(float(bpm / author), 2) if author else None,
            "first": round(float(t[0]), 2) if len(t) else None, "last": round(float(t[-1]), 2) if len(t) else None,
            "dur": round(dur, 1), "n": len(t), "bars": bs.get("bars"), "irregular_bars": bs.get("irregular_bars"),
            "bar_len_hist": bs.get("bar_len_hist")}


def main(qpath, *weights):
    q_all = json.loads(Path(qpath).read_text(encoding="utf-8"))
    models = {Path(w).parent.name + "_" + Path(w).stem: load_model(w) for w in weights}
    out_dir = Path(qpath).parent
    report = {}
    for sid, q in q_all.items():
        mel = dequantize(np.load(ROOT / "data/raw/kml/mel" / f"{sid}.npz")["mel"])
        dur = len(mel) / FPS
        qg = qildo_grid(q)
        r = {"qildo": summary(qg, dur, q["author_bpm"])}
        qt, qd = np.array(q["beats"]), np.array(q["downbeats"])
        for name, (m, dev) in models.items():
            ours = beats_from_act(activations(m, dev, mel))
            (out_dir / f"{sid}_{name}.json").write_text(json.dumps(ours))
            ot = np.array([g["time"] for g in ours])
            od = np.array([g["time"] for g in ours if g["beat"] == 1])
            d = offset(qt, ot)
            s = summary(ours, dur, q["author_bpm"])
            s.update(yt_offset=d, agree_beats_F=round(mir_eval.beat.f_measure(qt + d, ot, 0.07), 3),
                     agree_downbeats_F=round(mir_eval.beat.f_measure(qd + d, od, 0.07), 3))
            r[name] = s
        report[q["title"]] = r
        print(q["title"], json.dumps(r, ensure_ascii=False), flush=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
