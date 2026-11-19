"""Сравнение сетки Qildo с нашей на одной и той же записи (сумма их дорожек).

python tools_py/qildo_compare.py qildo_cmp/<track> runs/r2/best.pt [start_s]
Пишет: mix.wav, qildo_grid.json, ours.json, listen_A/B.mp3 (вслепую, ключ в blind_key.json), отчёт в stdout.
"""
import glob
import json
import random
import sys
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools_py"))
from click_overlay import click  # noqa: E402
from infer import load_model, run_file  # noqa: E402

SR = 44100


def qildo_grid(q):
    db = set(q["downbeats"])
    out, k = [], 0
    for t in q["beats"]:
        k = 1 if t in db else k + 1
        out.append({"time": t, "beat": k})
    return out


def bar_stats(grid):
    """Длины тактов в долях: сколько разных и как часто такт «скачет»."""
    nums = [g["beat"] for g in grid]
    lens, cur = [], None
    for n in nums:
        if n == 1:
            if cur is not None:
                lens.append(cur)
            cur = 1
        elif cur is not None:
            cur += 1
    lens = np.array(lens)
    if len(lens) == 0:
        return {"bars": 0}
    vals, cnt = np.unique(lens, return_counts=True)
    main = vals[np.argmax(cnt)]
    return {"bars": int(len(lens)), "bar_len_hist": {int(v): int(c) for v, c in zip(vals, cnt)},
            "irregular_bars": int(np.sum(lens != main))}


def excerpt(y, grid, a, b):
    seg = y[int(a * SR): int(b * SR)] * 0.55
    c = np.zeros_like(seg)
    hi, lo = click(1500, 0.05), click(800, 0.04)
    for ev in grid:
        s = int((ev["time"] - a) * SR)
        if 0 <= s < len(seg):
            k = hi * 0.9 if ev["beat"] == 1 else lo * 0.35
            e = min(len(c), s + len(k))
            c[s:e] += k[: e - s]
    return np.clip(seg + c, -1, 1)


def main(d, weights, start=None):
    d = Path(d)
    if not (d / "mix.wav").exists():
        ys = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(str(d / "*.flac")))]
        n = min(len(y) for y in ys)
        mix = sum(y[:n] for y in ys)
        sf.write(d / "mix.wav", mix / np.abs(mix).max() * 0.9, SR, subtype="PCM_16")
    model, dev = load_model(weights)
    ours = run_file(model, dev, str(d / "mix.wav"))
    (d / "ours.json").write_text(json.dumps(ours))
    q = json.loads((d / "qildo_analysis.json").read_text())
    qg = qildo_grid(q)
    (d / "qildo_grid.json").write_text(json.dumps(qg))
    ot = np.array([x["time"] for x in ours]); od = np.array([x["time"] for x in ours if x["beat"] == 1])
    qt = np.array(q["beats"]); qd = np.array(q["downbeats"])
    rep = {
        "agree_beats_F": round(mir_eval.beat.f_measure(qt, ot, 0.07), 3),
        "agree_downbeats_F": round(mir_eval.beat.f_measure(qd, od, 0.07), 3),
        "bpm_ours": round(60 / np.median(np.diff(ot)), 1), "bpm_qildo": q["bpm"],
        "first_beat_ours": ot[0] if len(ot) else None, "first_beat_qildo": qt[0],
        "bars_ours": bar_stats(ours), "bars_qildo": bar_stats(qg),
    }
    y, _ = sf.read(d / "mix.wav", dtype="float32")
    y = y.mean(1) if y.ndim == 2 else y
    a = float(start) if start is not None else 20.0
    names = ["qildo", "ours"]
    random.Random(hash(str(d)) & 0xFFFF).shuffle(names)
    key = {"A": names[0], "B": names[1]}
    (d / "blind_key.json").write_text(json.dumps(key))
    for lab, name in key.items():
        sf.write(d / f"listen_{lab}.mp3", excerpt(y, qg if name == "qildo" else ours, a, a + 30), SR, format="MP3")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])
