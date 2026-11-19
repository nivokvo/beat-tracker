"""Треки ccMixter только с лицензией CC BY (Attribution 3.0/4.0) → метаданные + mp3 → лог-мел.

python data_prep/fetch_ccmixter.py --n 1500 --out data/raw/ccmixter
Для каждого трека в meta.jsonl: id, название, автор, ссылка на страницу и лицензию, bpm (если указал автор), теги.
mp3 удаляется после подсчёта мела, кроме треков из тестового списка (--keep_audio_ids).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import load_audio, logmel, quantize  # noqa: E402

API = "http://ccmixter.org/api/query?f=json&lic=by&sort=rank&limit={limit}&offset={offset}"
UA = {"User-Agent": "beat-research-client/0.1", "Referer": "https://ccmixter.org/"}
SKIP_TAGS = {"a_cappella", "acappella", "spoken_word", "sample", "loops"}


def get(url, timeout=60):
    # urllib падает на сверхдлинном заголовке ответа ccMixter, поэтому через curl
    import subprocess
    r = subprocess.run(["curl", "-sfL", "--max-time", str(timeout), "-A", UA["User-Agent"],
                        "-e", UA["Referer"], url], capture_output=True, check=True)
    return r.stdout


def licence_ok(item):
    u = item.get("license_url", "")
    return "/licenses/by/" in u  # только чистый BY: без NC, ND, SA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--start_offset", type=int, default=0)
    ap.add_argument("--meta_name", default="meta.jsonl")
    ap.add_argument("--out", default="data/raw/ccmixter")
    ap.add_argument("--keep_audio_ids", default="", help="файл со списком id, чьи mp3 оставить")
    ap.add_argument("--keep_first", type=int, default=80, help="mp3 первых N треков оставить (кандидаты в тест)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "mel").mkdir(parents=True, exist_ok=True)
    keep = set()
    if a.keep_audio_ids and Path(a.keep_audio_ids).exists():
        keep = {l.strip() for l in open(a.keep_audio_ids) if l.strip()}
    meta_path = out / a.meta_name
    seen = set()
    if meta_path.exists():
        seen = {json.loads(l)["id"] for l in open(meta_path, encoding="utf-8")}
    metaf = open(meta_path, "a", encoding="utf-8")
    offset, got = a.start_offset, len(seen)
    while got < a.n:
        items = json.loads(get(API.format(limit=10, offset=offset)).decode("utf-8", "replace"))  # API дублирует JSON в заголовке X-JSON, большие страницы ломают HTTP-клиенты
        if not items:
            break
        offset += len(items)
        for it in items:
            uid = str(it["upload_id"])
            if uid in seen or not licence_ok(it):
                continue
            tags = set((it.get("upload_tags") or "").strip(",").split(","))
            if tags & SKIP_TAGS:
                continue
            mp3s = [f for f in it.get("files", []) if f.get("file_format_info", {}).get("default-ext") == "mp3"]
            if not mp3s:
                continue
            f = mp3s[0]
            dst = out / "audio" / f"{uid}.mp3"
            try:
                if not dst.exists():
                    dst.write_bytes(get(f["download_url"], timeout=120))
                y = load_audio(str(dst))
                if len(y) < 22050 * 30:
                    dst.unlink(missing_ok=True)
                    continue
                np.savez_compressed(out / "mel" / f"{uid}.npz", mel=quantize(logmel(y)))
            except Exception as e:
                print("fail", uid, type(e).__name__, flush=True)
                dst.unlink(missing_ok=True)
                continue
            if uid not in keep and got >= a.keep_first:
                dst.unlink(missing_ok=True)
            rec = {
                "id": uid, "name": it.get("upload_name"), "artist": it.get("user_name"),
                "page": it.get("file_page_url"), "license": it.get("license_name"),
                "license_url": it.get("license_url"), "bpm": (it.get("upload_extra") or {}).get("bpm"),
                "tags": sorted(t for t in tags if t), "dur_s": round(len(y) / 22050, 1),
            }
            metaf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            metaf.flush()
            seen.add(uid)
            got += 1
            if got % 50 == 0:
                print("got", got, flush=True)
            time.sleep(0.5)  # не долбим сайт
    print("done", got)


if __name__ == "__main__":
    main()
