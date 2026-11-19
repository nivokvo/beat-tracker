"""Проверка на 30 отложенных треках Маклауда (в обучение не шли): темп сетки против BPM автора.

python src/eval_kml30.py runs/r3/best.pt [ещё веса для ансамбля ...] --name r3
python src/eval_kml30.py --librosa --name librosa
Пишет runs/kml30_<name>.json и печатает: доля треков с темпом автора (±4 %), ×2, ×½, прочее,
плюс время до первой доли. Сетку каждого трека сохраняет в preds/kml30_<name>/.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FPS, load_audio, logmel  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402


def tempo_class(bpm, ref, tol=0.04):
    for name, k in (("ok", 1), ("x2", 2), ("x0.5", 0.5), ("x1.5", 1.5), ("x0.67", 2 / 3)):
        if abs(bpm / (ref * k) - 1) < tol:
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="*")
    ap.add_argument("--name", required=True)
    ap.add_argument("--librosa", action="store_true")
    ap.add_argument("--tta", type=float, nargs="+", default=[1.0], help="растяжки мела, напр. 0.95 1 1.05")
    ap.add_argument("--set", default="kml30", choices=["kml30", "judge"], help="judge = 150 треков судьи v2")
    a = ap.parse_args()
    lst, aud = ("test_list.json", "test_audio") if a.set == "kml30" else ("judge_list.json", "judge_audio")
    tests = json.loads((ROOT / "data/raw/kml" / lst).read_text(encoding="utf-8"))
    models = [load_model(w) for w in a.weights]
    out_dir = ROOT / "preds" / f"{a.set}_{a.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in tests:
        sid = hashlib.md5(p["filename"].encode()).hexdigest()[:12]
        mp3 = ROOT / "data/raw/kml" / aud / f"{sid}.mp3"
        if not mp3.exists():
            print("нет файла", p["title"])
            continue
        t0 = time.time()
        y = load_audio(str(mp3))
        if a.librosa:
            import librosa
            from features import SR
            _, bt = librosa.beat.beat_track(y=y, sr=SR, units="time")
            ev = [{"time": float(t), "beat": i % 4 + 1} for i, t in enumerate(bt)]
        else:
            mel = logmel(y)
            acts = []
            for r in a.tta:  # растяжка мела по времени в r раз, активации возвращаем на исходную шкалу
                T = len(mel)
                if r == 1.0:
                    mr = mel
                else:
                    src = np.arange(int(T / r)) * r
                    mr = np.stack([np.interp(src, np.arange(T), mel[:, k].astype(np.float32)) for k in range(mel.shape[1])], 1)
                ar = np.mean([activations(m, d, mr) for m, d in models], axis=0)
                if r != 1.0:
                    ar = np.stack([np.interp(np.arange(T), np.arange(len(ar)) * r, ar[:, k]) for k in range(ar.shape[1])], 1)
                acts.append(ar)
            act = np.mean(acts, axis=0)
            ev = beats_from_act(act)
        dt = time.time() - t0
        t = np.array([e["time"] for e in ev])
        bpm = 60 / np.median(np.diff(t)) if len(t) > 2 else 0
        ref = float(p["bpm"])
        cls = tempo_class(bpm, ref) if bpm else "other"
        (out_dir / f"{sid}.json").write_text(json.dumps(ev))
        rows.append({"title": p["title"], "id": sid, "ref_bpm": ref, "bpm": round(float(bpm), 1), "class": cls,
                     "first_beat": round(float(t[0]), 2) if len(t) else None, "n": len(ev), "sec": round(dt, 1)})
        print(f"{p['title'][:28]:28s} автор {ref:5.0f}  мы {bpm:6.1f}  {cls}", flush=True)
    summ = {c: sum(r["class"] == c for r in rows) for c in ("ok", "x2", "x0.5", "x1.5", "x0.67", "other")}
    summ["n"] = len(rows)
    print(summ)
    (ROOT / "runs" / f"{a.set}_{a.name}.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
