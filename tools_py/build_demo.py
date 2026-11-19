"""Данные для демо-страницы: по каждому треку из qildo_cmp — сетки (наша и Qildo), длины тактов,
30-секундный фрагмент (mp3 base64) там, где у сетки Qildo больше всего сбитых тактов, плюс таблица kml30.

python tools_py/build_demo.py demo/template.html demo/qildo_beats_demo.html [runs/kml30_<name>.json]
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TRACKS = [("funky_chunk", "Funky Chunk"), ("sneaky_snitch", "Sneaky Snitch"), ("local_forecast", "Local Forecast — Elevator")]
SR = 44100
WIN = 30.0
NOTES = {
    "local_forecast": "Свинг. Новая модель здесь считает восьмые: темп вдвое выше (166 вместо 81), каждый «такт» — половина настоящего. "
                      "Это известная слабость (см. отчёт). Такты при этом ровные, а у текущей сетки 34 такта другой длины.",
}


def bars(grid):
    """[(начало такта, число долей)]"""
    out, start, n = [], None, 0
    for g in grid:
        if g["beat"] == 1:
            if start is not None:
                out.append((round(start, 3), n))
            start, n = g["time"], 1
        elif start is not None:
            n += 1
    return out


def main_len(bs):
    if not bs:
        return 4
    v, c = np.unique([b[1] for b in bs], return_counts=True)
    return int(v[np.argmax(c)])


def window(qb, dur):
    """Окно WIN с наибольшим числом сбитых тактов у Qildo (если сбоев нет — с 20-й секунды)."""
    m = main_len(qb)
    bad = np.array([t for t, n in qb if n != m])
    best, best_a = 0, min(20.0, max(0.0, dur - WIN))
    for a in np.arange(0, max(dur - WIN, 0) + 1e-9, 1.0):
        k = int(((bad >= a + 2) & (bad < a + WIN - 4)).sum()) if len(bad) else 0
        if k > best:
            best, best_a = k, float(a)
    return best_a


def mp3_b64(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="MP3")
    return base64.b64encode(buf.getvalue()).decode()


def main(tpl, out, kml=None):
    data = {"tracks": []}
    for d, title in TRACKS:
        p = ROOT / "qildo_cmp" / d
        ours = json.loads((p / "ours.json").read_text())
        qg = json.loads((p / "qildo_grid.json").read_text())
        y, sr = sf.read(p / "mix.wav", dtype="float32")
        y = y.mean(1) if y.ndim == 2 else y
        dur = len(y) / sr
        ob, qb = bars(ours), bars(qg)
        a = window(qb, dur)
        seg = y[int(a * sr): int((a + WIN) * sr)]
        seg = seg / (np.abs(seg).max() + 1e-9) * 0.8
        ot = np.array([g["time"] for g in ours])
        qa = json.loads((p / "qildo_analysis.json").read_text())
        data["tracks"].append({
            "id": d, "title": title, "dur": round(dur, 2), "win": [a, a + WIN],
            "bpm_ours": round(float(60 / np.median(np.diff(ot))), 1), "bpm_qildo": qa["bpm"],
            "ours": [[round(g["time"], 3), g["beat"]] for g in ours],
            "qildo": [[round(g["time"], 3), g["beat"]] for g in qg],
            "bars_ours": ob, "bars_qildo": qb, "main_ours": main_len(ob), "main_qildo": main_len(qb),
            "audio": mp3_b64(seg), "note": NOTES.get(d, ""),
        })
    if kml:
        data["kml30"] = json.loads(Path(kml).read_text(encoding="utf-8"))
    html = Path(tpl).read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    Path(out).write_text(html, encoding="utf-8")
    print("ok", out, round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
