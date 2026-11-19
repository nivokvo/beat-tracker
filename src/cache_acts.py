"""Активации модели для всех треков с известным BPM → runs/acts_<name>/<sid>.npz (для выбора уровня темпа).

python src/cache_acts.py --name r3 runs/r3/best.pt
Источники: incompetech (обучение, BPM автора), ccMixter (BPM из тегов), 30 тестовых mp3 Маклауда (только проверка).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import dequantize, load_audio, logmel  # noqa: E402
from infer import activations, load_model  # noqa: E402

QILDO_TITLES = {"Funky Chunk", "Local Forecast - Elevator", "Sneaky Snitch"}


def fnum(v):
    try:
        v = float(v)
        return v if 30 < v < 300 else None
    except (TypeError, ValueError):
        return None


def sources():
    """(sid, split, bpm, loader)"""
    tests = json.loads((ROOT / "data/raw/kml/test_list.json").read_text(encoding="utf-8"))
    for p in tests:
        sid = hashlib.md5(p["filename"].encode()).hexdigest()[:12]
        mp3 = ROOT / "data/raw/kml/test_audio" / f"{sid}.mp3"
        if mp3.exists():
            yield "kmltest_" + sid, "test", fnum(p["bpm"]), lambda m=mp3: logmel(load_audio(str(m)))
    for mf in (ROOT / "data/raw/kml").glob("meta_*.jsonl"):
        for l in open(mf, encoding="utf-8"):
            r = json.loads(l)
            f = ROOT / "data/raw/kml/mel" / f"{r['id']}.npz"
            if not r["test"] and r["title"] not in QILDO_TITLES and f.exists() and fnum(r.get("bpm")):
                yield "kml_" + r["id"], "train", fnum(r["bpm"]), lambda f=f: dequantize(np.load(f)["mel"])
    ccm_test = {l.strip() for l in open(ROOT / "data/real/ccm_test_ids.txt") if l.strip()}
    for mf in (ROOT / "data/raw/ccmixter").glob("meta*.jsonl"):
        for l in open(mf, encoding="utf-8"):
            r = json.loads(l)
            f = ROOT / "data/raw/ccmixter/mel" / f"{r['id']}.npz"
            if r["id"] not in ccm_test and f.exists() and fnum(r.get("bpm")):
                yield "ccm_" + r["id"], "train_ccm", fnum(r["bpm"]), lambda f=f: dequantize(np.load(f)["mel"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    models = [load_model(w) for w in a.weights]
    out = ROOT / "runs" / f"acts_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    index, seen = [], set()
    for sid, split, bpm, load in sources():
        if sid in seen:
            continue
        seen.add(sid)
        index.append({"sid": sid, "split": split, "bpm": bpm})
        f = out / f"{sid}.npz"
        if f.exists():
            continue
        try:
            act = np.mean([activations(m, d, load()) for m, d in models], axis=0)
        except Exception as e:  # битый файл — пропускаем
            print("fail", sid, type(e).__name__, flush=True)
            index.pop()
            continue
        np.savez_compressed(f, act=act.astype(np.float16))
    (out / "index.json").write_text(json.dumps(index))
    print("done", len(index), {s: sum(r["split"] == s for r in index) for s in ("test", "train", "train_ccm")}, flush=True)


if __name__ == "__main__":
    main()
