"""Судья v2: 150 треков Маклауда, которых нет ни в обучении, ни в тесте kml30 → data/raw/kml/judge_audio/.

python data_prep/fetch_kml_judge.py [--n 150]
Берём из каталога (pieces.json) только треки, которые ни разу не скачивались для обучения
(нет в meta_*.jsonl), с BPM автора и длиной 1,5–6 мин; по корзинам жанр × темп, детерминированно.
Список — data/raw/kml/judge_list.json. Эти треки нельзя пускать в обучение (fetch_incompetech их пропускает).
"""
import argparse
import hashlib
import json
import subprocess
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KML = ROOT / "data/raw/kml"
BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"


def secs(s):
    h, m, x = (int(v) for v in s.split(":"))
    return h * 3600 + m * 60 + x


def pick(n):
    pieces = json.loads((KML / "pieces.json").read_text(encoding="utf-8"))
    used = set()
    for f in KML.glob("meta_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                used.add(json.loads(line)["filename"])
    used |= {p["filename"] for p in json.loads((KML / "test_list.json").read_text(encoding="utf-8"))}
    ok = [p for p in pieces if p["filename"] not in used and p.get("bpm")
          and p["bpm"].replace(".", "").isdigit() and 60 <= float(p["bpm"]) <= 220
          and 90 <= secs(p["length"]) <= 360]
    key = lambda p: hashlib.md5(("judge" + p["filename"]).encode()).hexdigest()  # noqa: E731
    buckets = {}
    for p in sorted(ok, key=key):
        buckets.setdefault((p["genre"], min(int(float(p["bpm"]) // 40), 4)), []).append(p)
    out = []
    while len(out) < n and any(buckets.values()):
        for b in sorted(buckets, key=lambda b: -len(buckets[b])):
            if buckets[b] and len(out) < n:
                out.append(buckets[b].pop(0))
    return out, len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    lst = KML / "judge_list.json"
    if lst.exists():
        judge = json.loads(lst.read_text(encoding="utf-8"))
    else:
        judge, pool = pick(a.n)
        lst.write_text(json.dumps(judge, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"выбрано {len(judge)} из {pool} свободных", flush=True)
    (KML / "judge_audio").mkdir(exist_ok=True)
    for p in judge:
        sid = hashlib.md5(p["filename"].encode()).hexdigest()[:12]
        dst = KML / "judge_audio" / f"{sid}.mp3"
        if dst.exists() and dst.stat().st_size > 100_000:
            continue
        part = dst.with_suffix(".part")
        r = subprocess.run(["curl", "-sfL", "--max-time", "600", "-o", str(part), BASE + urllib.parse.quote(p["filename"])])
        if r.returncode == 0 and not dst.exists():
            part.replace(dst)
        else:
            part.unlink(missing_ok=True)
            print("fail", p["title"], flush=True)
    print("done", len(list((KML / "judge_audio").glob("*.mp3"))), flush=True)


if __name__ == "__main__":
    main()
