"""Скачать только 30 тестовых треков Маклауда (data/raw/kml/test_list.json) в data/raw/kml/test_audio/.

python data_prep/fetch_kml_test.py
Нужен для сравнения с Qildo: друзья заливают те же треки, наша модель считает по этим mp3.
"""
import hashlib
import json
import subprocess
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KML = ROOT / "data/raw/kml"
BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"

for p in json.loads((KML / "test_list.json").read_text(encoding="utf-8")):
    sid = hashlib.md5(p["filename"].encode()).hexdigest()[:12]
    dst = KML / "test_audio" / f"{sid}.mp3"
    if dst.exists() and dst.stat().st_size > 100_000:
        continue
    part = dst.with_suffix(".part")
    r = subprocess.run(["curl", "-sfL", "--max-time", "600", "-o", str(part), BASE + urllib.parse.quote(p["filename"])])
    if r.returncode == 0 and not dst.exists():
        part.replace(dst)
        print("ok", p["title"], flush=True)
    else:
        part.unlink(missing_ok=True)
        print("fail" if r.returncode else "skip", p["title"], flush=True)
print("done", flush=True)
