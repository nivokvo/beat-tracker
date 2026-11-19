"""Живые записи с ручной разметкой → мел + эталон в формате Qildo (для судейских срезов, не для обучения).

python data_prep/prep_real_eval.py
- KRAISLER (CC BY 4.0): скрипка + фортепиано, rubato; доли и сильные доли вручную. Берём mix_studio.
- Schubert Winterreise (CC BY 3.0): голос + фортепиано; вручную размечены начала тактов. Доли внутри такта
  делим поровну по размеру из MusicXML (x/4, x/2 — числитель; 6/8, 9/8, 12/8 — числитель/3). Сильные доли точные,
  слабые — приближение, поэтому для Winterreise смотреть в первую очередь на downbeat_F.
- Candombe (CC BY 4.0): живые барабаны, доли и сильные доли вручную.
Выход: data/real_eval/<набор>/<id>.npz (mel) и data/real_eval/<набор>_refs/<id>.json.
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_audio, logmel, quantize  # noqa: E402

RAW = Path("D:/ml-qildo-data/raw")
OUT = ROOT / "data/real_eval"


def save(name, sid, audio, grid):
    d, r = OUT / name, OUT / f"{name}_refs"
    d.mkdir(parents=True, exist_ok=True)
    r.mkdir(parents=True, exist_ok=True)
    if not (d / f"{sid}.npz").exists():
        np.savez_compressed(d / f"{sid}.npz", mel=quantize(logmel(load_audio(str(audio)))))
    (r / f"{sid}.json").write_text(json.dumps([{"time": round(float(t), 4), "beat": int(b)} for t, b in grid]))


def number(times, is_down):
    out, k = [], 0
    for t, db in zip(times, is_down):
        k = 1 if db else (k + 1 if k else 0)
        out.append((t, k))
    first = next((i for i, (_, b) in enumerate(out) if b == 1), None)
    if first:  # доли до первой сильной — затакт: нумеруем назад по размеру первого полного такта
        nxt = next((i for i in range(first + 1, len(out)) if out[i][1] == 1), len(out))
        meter = max(2, nxt - first)
        for i in range(first - 1, -1, -1):
            out[i] = (out[i][0], (out[i + 1][1] - 2) % meter + 1)
    return out


def kraisler():
    base = RAW / "kraisler/KRAISLER"
    for f in sorted((base / "annotation_csv").glob("*_beats.csv")):
        sid = f.stem.split("_")[0]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        grid = number([float(r["beat_time"]) for r in rows], [r["beat_type"] == "db" for r in rows])
        save("kraisler", sid, base / "performance_wav" / f"{sid}_mix_studio.wav", grid)


def candombe():
    base = RAW / "candombe"
    for f in sorted((base / "candombe_annotations/with_bar_number").glob("*.csv")):
        rows = [ln.strip().split(",") for ln in open(f, encoding="utf-8") if ln.strip()]
        grid = [(float(t), int(bb.split(".")[1])) for t, bb in rows]
        save("candombe", f.stem, base / "candombe_audio" / f"{f.stem}.flac", grid)


def beats_per_measure(xml_path):
    """Число долей в каждом такте по MusicXML (номер такта → доли)."""
    root = ET.parse(xml_path).getroot()
    part = root.find("part")
    cur, out = 4, {}
    for i, m in enumerate(part.findall("measure"), start=1):
        t = m.find("attributes/time")
        if t is not None and t.find("beats") is not None:
            num, den = int(re.sub(r"\D.*", "", t.find("beats").text)), int(t.find("beat-type").text)
            cur = num // 3 if den == 8 and num % 3 == 0 and num > 3 else num
        out[i] = cur
    return out


def winterreise():
    base = RAW / "winterreise"
    for f in sorted((base / "02_Annotations/ann_audio_measure").glob("*.csv")):
        audio = base / "01_RawData/audio_wav" / f"{f.stem}.wav"
        if not audio.exists():
            continue  # в открытом архиве только две записи из девяти
        song = "_".join(f.stem.split("_")[:2])
        bpm_of = beats_per_measure(base / "01_RawData/score_musicxml" / f"{song}.xml")
        rows = list(csv.DictReader(open(f, encoding="utf-8"), delimiter=";"))
        st = [float(r["start"]) for r in rows]
        ms = [int(float(r["measure"])) for r in rows]
        grid = []
        for i in range(len(st) - 1):
            n = bpm_of.get(ms[i], 4)
            for k in range(n):
                grid.append((st[i] + k * (st[i + 1] - st[i]) / n, k + 1))
        save("winterreise", f.stem, audio, grid)


if __name__ == "__main__":
    for fn in (kraisler, candombe, winterreise):
        fn()
        print(fn.__name__, len(list((OUT / fn.__name__).glob("*.npz"))), flush=True)
