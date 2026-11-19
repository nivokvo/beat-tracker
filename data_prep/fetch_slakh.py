"""Slakh2100-redux (CC BY 4.0, Lakh MIDI, отрендеренный профессиональными сэмпл-инструментами) → лог-мел + доли из MIDI.

python data_prep/fetch_slakh.py --out data/slakh [--workers 6] [--max_tracks N]
Архив tar.gz на 104 ГБ на диск не ложится, поэтому читаем его потоком: несколько соединений качают
куски по Range подряд, tarfile распаковывает поток на лету. Из каждого трека берём только mix.flac и
all_src.mid, считаем мел и сразу выбрасываем звук. Стемы проходят мимо.
Фильтры те же, что у нашего рендера Lakh (render_lakh.py): размер, квантованность MIDI, минимум долей.
"""
import argparse
import io
import json
import queue
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_prep"))
from features import SR, logmel, quantize  # noqa: E402
from render_lakh import ALLOWED_TS, beat_numbers, quantized_fraction  # noqa: E402

URL = "https://zenodo.org/api/records/4599666/files/slakh2100_flac_redux.tar.gz/content"
SIZE = 104_300_000_000  # уточняется по Content-Range первого ответа


class RangeStream(io.RawIOBase):
    """Читает URL последовательными кусками, качая до `ahead` кусков параллельно."""

    def __init__(self, url, chunk=32 << 20, workers=6, ahead=12, start=0):
        self.url, self.chunk = url, chunk
        self.total = self._size()
        self.next_fetch = start
        self.pos = start
        self.buf = b""
        self.done = {}
        self.lock = threading.Condition()
        self.sem = threading.Semaphore(ahead)
        self.jobs = queue.Queue()
        self.threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def _size(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Range"].split("/")[-1])

    def _feed(self):
        off = self.next_fetch
        while off < self.total:
            self.sem.acquire()
            self.jobs.put(off)
            off += self.chunk
        for _ in self.threads:
            self.jobs.put(None)

    def _get(self, off):
        end = min(off + self.chunk, self.total) - 1
        for attempt in range(20):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - off + 1:
                    return data
            except Exception as e:  # noqa: BLE001
                print("retry", off, type(e).__name__, flush=True)
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"chunk {off} failed")

    def _worker(self):
        while True:
            off = self.jobs.get()
            if off is None:
                return
            data = self._get(off)
            with self.lock:
                self.done[off] = data
                self.lock.notify_all()

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.pos >= self.total:
                return 0
            with self.lock:
                while self.pos not in self.done:
                    self.lock.wait()
                self.buf = self.done.pop(self.pos)
            self.sem.release()
            self.pos += len(self.buf)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def mix_to_mel(mix_bytes):
    """mix.flac → квантованный мел (uint8) и длительность; держать в памяти звук не нужно."""
    y, sr = sf.read(io.BytesIO(mix_bytes), dtype="float32")
    if y.ndim == 2:
        y = y.mean(1)
    y = soxr.resample(y, sr, SR).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return quantize(logmel(y)), len(y) / SR


def process(track, mel, last, midi_bytes, out):
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    ts = {(t.numerator, t.denominator) for t in pm.time_signature_changes}
    if not ts or not ts <= ALLOWED_TS:
        return "ts"
    beats, downbeats = pm.get_beats(), pm.get_downbeats()
    if quantized_fraction(pm, beats) < 0.85:
        return "unquantized"
    notes = [n for i in pm.instruments for n in i.notes]
    if not notes:
        return "empty"
    first_note = min(n.start for n in notes)
    last_note = min(max(n.end for n in notes), last)
    nums_all = beat_numbers(beats, downbeats)
    m = (beats >= first_note - 0.05) & (beats <= last_note)
    beats, nums = beats[m], nums_all[m]
    downbeats = beats[nums == 1]
    if len(beats) < 16 or len(downbeats) < 4:
        return "few_beats"
    split = "val" if "/validation/" in track else ("test" if "/test/" in track else "train")
    name = track.rstrip("/").split("/")[-1]
    (out / split).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / split / f"{name}.npz", mel=mel, beats=beats, downbeats=downbeats, nums=nums)
    rp = out / f"{split}_refs"
    rp.mkdir(parents=True, exist_ok=True)
    (rp / f"{name}.json").write_text(json.dumps([{"time": round(float(t), 4), "beat": int(n)} for t, n in zip(beats, nums)]))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/slakh")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_tracks", type=int, default=10**9)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    stream = RangeStream(URL, workers=a.workers)
    print("size", stream.total, flush=True)
    # Архив перемешан: mix.flac и all_src.mid одного трека лежат далеко друг от друга.
    # Мел микса сразу кладём на диск (_mel/), MIDI держим в памяти (килобайты), трек собираем, когда есть обе части.
    meld = out / "_mel"
    meld.mkdir(parents=True, exist_ok=True)
    midis, stats, t0 = {}, {}, time.time()
    with tarfile.open(fileobj=io.BufferedReader(stream, 8 << 20), mode="r|gz") as tar:
        for mem in tar:
            name = mem.name
            if "/omitted/" in name or not mem.isfile() or not (name.endswith("/mix.flac") or name.endswith("/all_src.mid")):
                continue
            track = name.rsplit("/", 1)[0]
            tn = track.split("/")[-1]
            if any((out / s / f"{tn}.npz").exists() for s in ("train", "val", "test")):
                continue
            mp = meld / f"{tn}.npz"
            try:
                if name.endswith(".flac"):
                    mel, last = mix_to_mel(tar.extractfile(mem).read())
                    np.savez(mp, mel=mel, last=last)
                else:
                    midis[track] = tar.extractfile(mem).read()
            except Exception as e:  # noqa: BLE001
                stats["err:" + type(e).__name__] = stats.get("err:" + type(e).__name__, 0) + 1
                continue
            if track in midis and mp.exists():
                try:
                    with np.load(mp) as d:  # закрыть файл до удаления (Windows)
                        mel, last = d["mel"], float(d["last"])
                    r = process(track, mel, last, midis.pop(track), out)
                    mp.unlink()
                except Exception as e:  # noqa: BLE001
                    r = "err:" + type(e).__name__
                stats[r] = stats.get(r, 0) + 1
                n = sum(stats.values())
                if n % 20 == 0:
                    gb = stream.pos / 1e9
                    print(f"{n} tracks {stats} {gb:.1f}/{stream.total / 1e9:.1f} GB {gb * 1e3 / (time.time() - t0):.1f} MB/s", flush=True)
                if stats.get("ok", 0) >= a.max_tracks:
                    break
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
