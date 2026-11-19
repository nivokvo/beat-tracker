"""Рок и поп из Free Music Archive только под CC BY / CC0 / Public Domain Mark → лог-мел (для псевдоразметки).

python data_prep/fetch_fma.py --out D:/ml-qildo-data/raw/fma/mel [--limit N]
fma_full.zip весит 943 ГБ, поэтому читаем его как удалённый файл: zipfile сам ищет оглавление,
а каждое чтение превращается в HTTP Range-запрос. Скачиваются только выбранные mp3 (~5 МБ каждый).
Лицензия каждого трека — из fma_metadata/tracks.csv; список с лицензиями пишется в <out>/../selected.csv (для DATA.md).
"""
import argparse
import ast
import io
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_audio, logmel, quantize  # noqa: E402

URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
META = Path("D:/ml-qildo-data/raw/fma/fma_metadata")
ROCK, POP = 12, 10


class HttpFile(io.RawIOBase):
    """Файл по HTTP с seek: каждое read — Range-запрос (с повторами)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:  # noqa: BLE001
                print("retry", type(e).__name__, flush=True)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("range failed")
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def select():
    t = pd.read_csv(META / "tracks.csv", header=[0, 1], index_col=0, low_memory=False)
    lic = t[("track", "license")].fillna("")
    bad = lic.str.contains(r"non-?commercial|share.?alike|no.?deriv|music sharing|common law", case=False, regex=True)
    good = (lic.str.contains("attribution", case=False) | lic.str.contains(r"cc0|public domain", case=False)) & ~bad
    gall = t[("track", "genres_all")].fillna("[]").map(ast.literal_eval)
    rp = gall.map(lambda g: ROCK in g or POP in g)
    dur = t[("track", "duration")]
    sel = t[good & rp & (dur >= 60) & (dur <= 600)]
    return pd.DataFrame({"title": sel[("track", "title")], "artist": sel[("artist", "name")],
                         "license": sel[("track", "license")], "genre_top": sel[("track", "genre_top")],
                         "duration": sel[("track", "duration")]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/ml-qildo-data/raw/fma/mel")
    ap.add_argument("--limit", type=int, default=10**9)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = select()
    sel.to_csv(out.parent / "selected.csv")
    print("selected", len(sel), "hours", round(sel["duration"].sum() / 3600, 1), flush=True)
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), 1 << 20))
    names = {int(Path(n).stem): n for n in zf.namelist() if n.endswith(".mp3")}
    print("zip entries", len(names), flush=True)
    stats = {}
    for i, tid in enumerate(sel.index[: a.limit]):
        dst = out / f"{tid:06d}.npz"
        if dst.exists() or tid not in names:
            stats["skip"] = stats.get("skip", 0) + 1
            continue
        try:
            data = zf.read(names[tid])
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(data)
            y = load_audio(tf.name)
            Path(tf.name).unlink()
            np.savez_compressed(dst, mel=quantize(logmel(y)))
            stats["ok"] = stats.get("ok", 0) + 1
        except Exception as e:  # noqa: BLE001
            stats["err:" + type(e).__name__] = stats.get("err:" + type(e).__name__, 0) + 1
        if i % 25 == 0:
            print(i, len(sel), stats, flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
