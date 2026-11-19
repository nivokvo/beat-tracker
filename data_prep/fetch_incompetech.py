"""Каталог Кевина Маклауда (incompetech.com, CC BY 4.0) → мел для обучения + 30 отложенных треков для теста.

python data_prep/fetch_incompetech.py --part 0 --parts 2
Тест: 30 треков, по жанрам и темпам, 1,5–6 мин; их mp3 остаются в data/raw/kml/test_audio/.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_audio, logmel, quantize  # noqa: E402

KML = ROOT / "data/raw/kml"
BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"


def secs(s):
    h, m, x = (int(v) for v in s.split(":"))
    return h * 3600 + m * 60 + x


def pick_test(pieces, n=30):
    """Детерминированно: по корзинам (жанр × темп) берём по одному треку, пока не наберём n."""
    ok = [p for p in pieces if p.get("bpm") and p["bpm"].replace(".", "").isdigit() and 90 <= secs(p["length"]) <= 360]
    key = lambda p: hashlib.md5(p["filename"].encode()).hexdigest()  # noqa: E731
    buckets = {}
    for p in sorted(ok, key=key):
        b = (p["genre"], min(int(float(p["bpm"]) // 40), 4))
        buckets.setdefault(b, []).append(p)
    out = []
    for b in sorted(buckets, key=lambda b: -len(buckets[b])):
        if len(out) < n:
            out.append(buckets[b][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    pieces = json.loads((KML / "pieces.json").read_text(encoding="utf-8"))
    test = pick_test(pieces)
    test_names = {p["filename"] for p in test}
    (KML / "test_list.json").write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding="utf-8")
    for d in ["mel", "test_audio", "tmp"]:
        (KML / d).mkdir(parents=True, exist_ok=True)
    meta = open(KML / f"meta_{a.part}.jsonl", "a", encoding="utf-8")
    done = {f.stem for f in (KML / "mel").glob("*.npz")}
    jl = KML / "judge_list.json"  # судья v2 (fetch_kml_judge.py) — никогда в обучение
    judge_names = {p["filename"] for p in json.loads(jl.read_text(encoding="utf-8"))} if jl.exists() else set()
    for i, p in enumerate(pieces):
        if i % a.parts != a.part or p["filename"] in judge_names:
            continue
        sid = hashlib.md5(p["filename"].encode()).hexdigest()[:12]
        is_test = p["filename"] in test_names
        if sid in done and not is_test:
            continue
        dst = (KML / "test_audio" if is_test else KML / "tmp") / f"{sid}.mp3"
        try:
            if not dst.exists():
                url = BASE + urllib.parse.quote(p["filename"])
                part = dst.with_suffix(".part")  # оборванная загрузка не должна выглядеть готовым mp3
                subprocess.run(["curl", "-sfL", "--max-time", "900", "-o", str(part), url], check=True)
                part.replace(dst)
            if not is_test:
                y = load_audio(str(dst))
                np.savez_compressed(KML / "mel" / f"{sid}.npz", mel=quantize(logmel(y)))
                dst.unlink()
        except Exception as e:
            print("fail", p["filename"], type(e).__name__, flush=True)
            dst.unlink(missing_ok=True)
            continue
        meta.write(json.dumps({"id": sid, "title": p["title"], "filename": p["filename"], "bpm": p["bpm"],
                               "genre": p["genre"], "length": p["length"], "test": is_test,
                               "license": "CC BY 4.0", "source": "https://incompetech.com"}, ensure_ascii=False) + "\n")
        meta.flush()
        time.sleep(0.3)
    print("done", flush=True)


if __name__ == "__main__":
    main()
