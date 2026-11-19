"""Живые барабаны: Groove MIDI Dataset (Magenta, CC BY 4.0) + партии Lakh MIDI (CC BY) без барабанов,
подогнанные по времени под сетку долей барабанщика.

python data_prep/render_groove.py --gmd D:/ml-qildo-data/raw/groove/groove --out data/groove --k 4 --workers 14

Барабанщики GMD играли под клик, поэтому доли — это темп-карта их MIDI (tempo = bpm из info.csv),
а звук — их реальная игра со всей человеческой микроритмикой, свингом и шаффлом.
Партнёр из Lakh: берём K-ю сильную долю Lakh-трека и кусочно-линейно отображаем его доли на доли
барабанщика (pm.adjust_times), так что гармония и бас идут в такт с живыми барабанами.
Для свинговых/шаффловых грувов партия Lakh тоже свингуется. Часть примеров — одни барабаны.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_prep"))
from features import SR, logmel, quantize  # noqa: E402
from render_lakh import FLUIDSYNTH, SOUNDFONTS, audio_augment, beat_numbers, quantized_fraction, swing  # noqa: E402

LMD = ROOT / "data/raw/lmd/x/lmd_full"
SWINGY = ("jazz", "swing", "shuffle", "blues", "funk")  # в funk GMD много шаффла и свинговых шестнадцатых


def lakh_partners(n, seed):
    """MIDI Lakh, которые уже прошли фильтры нашего рендера (train-сплит)."""
    srcs = sorted({p.stem.rsplit("_", 1)[0] for p in (ROOT / "data/synth/train_meta").glob("*.json")})
    random.Random(seed).shuffle(srcs)
    return [str(LMD / s[0] / f"{s}.mid") for s in srcs[:n]]


def gmd_beats(pm, dur):
    beats = pm.get_beats()
    db = pm.get_downbeats()
    m = beats < dur - 0.05
    return beats[m], db[db < dur - 0.05]


def fit_partner(lpm, g_beats, g_db, rng, do_swing):
    """Отобразить доли Lakh-партнёра на доли барабанщика. None, если не подходит."""
    lpm.instruments = [i for i in lpm.instruments if not i.is_drum and i.notes]
    if len(lpm.instruments) < 2:
        return None
    lts = {(t.numerator, t.denominator) for t in lpm.time_signature_changes}
    if lts and lts != {(4, 4)}:
        return None
    lb, ldb = lpm.get_beats(), lpm.get_downbeats()
    if len(ldb) < 8 or quantized_fraction(lpm, lb) < 0.85:
        return None
    if do_swing:
        swing(lpm, float(rng.uniform(0.6, 0.67)))
        lb = lpm.get_beats()
    # первая сильная доля барабанщика ↔ случайная сильная доля Lakh из первой половины трека
    g0 = int(np.searchsorted(g_beats, g_db[0] - 1e-3))
    j = int(np.searchsorted(lb, ldb[int(rng.integers(0, max(1, len(ldb) // 2)))] - 1e-3))
    n = min(len(g_beats) - g0, len(lb) - j)
    if n < 16:
        return None
    src = lb[j:j + n]
    dst = g_beats[g0:g0 + n]
    # до и после куска: срезаем ноты, остальное отображается по долям
    for ins in lpm.instruments:
        ins.notes = [x for x in ins.notes if src[0] <= x.start < src[-1]]
        for x in ins.notes:
            x.end = min(x.end, src[-1])
        ins.control_changes = [c for c in ins.control_changes if src[0] <= c.time < src[-1]]
        ins.pitch_bends = [b for b in ins.pitch_bends if src[0] <= b.time < src[-1]]
    lpm.adjust_times(src, dst)
    if not any(i.notes for i in lpm.instruments):
        return None
    return lpm


def render_midi(pm, sf_name):
    with tempfile.TemporaryDirectory() as td:
        mid, wav = os.path.join(td, "a.mid"), os.path.join(td, "a.wav")
        pm.write(mid)
        subprocess.run([str(FLUIDSYNTH), "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5", "-r", str(SR), "-F", wav,
                        str(SOUNDFONTS[sf_name]), mid], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
        y, sr = sf.read(wav, dtype="float32")
    return y.mean(1) if y.ndim == 2 else y


def render_one(job):
    row, gmd_dir, out, k, partners, seed = job
    try:
        name = f"{Path(row['midi_filename']).stem}_{row['drummer']}_{k}"
        split = {"train": "train", "validation": "val", "test": "test"}[row["split"]]
        npz_path = Path(out) / split / f"{name}.npz"
        if npz_path.exists():
            return "exists"
        rng = np.random.default_rng(seed)
        drums, sr = sf.read(Path(gmd_dir) / row["audio_filename"], dtype="float32")
        if drums.ndim == 2:
            drums = drums.mean(1)
        drums = soxr.resample(drums, sr, SR).astype(np.float32)
        dur = len(drums) / SR
        gpm = pretty_midi.PrettyMIDI(str(Path(gmd_dir) / row["midi_filename"]))
        g_beats, g_db = gmd_beats(gpm, dur)
        if len(g_beats) < 16 or len(g_db) < 4:
            return "few_beats"
        info = {"style": row["style"], "bpm": row["bpm"], "drummer": row["drummer"]}
        y = drums / (np.max(np.abs(drums)) + 1e-9)
        if rng.random() < 0.8 and row["time_signature"] == "4-4":
            do_swing = any(s in row["style"].lower() for s in SWINGY) and rng.random() < 0.7
            lpm = None
            for partner in partners:  # не каждый трек Lakh подходит (размер, квантованность) — пробуем несколько
                try:
                    lpm = fit_partner(pretty_midi.PrettyMIDI(partner), g_beats, g_db, rng, do_swing)
                except Exception:  # noqa: BLE001
                    lpm = None
                if lpm is not None:
                    break
            if lpm is not None:
                sf_name = list(SOUNDFONTS)[int(rng.integers(len(SOUNDFONTS)))]
                mus = render_midi(lpm, sf_name)[: len(y)]
                mus = np.pad(mus, (0, len(y) - len(mus)))
                mus = mus / (np.max(np.abs(mus)) + 1e-9)
                g = float(rng.uniform(0.25, 1.0))  # громкость барабанов относительно музыки
                y = g * y + mus
                info.update(partner=Path(partner).stem, soundfont=sf_name, drum_gain=round(g, 2), swing=do_swing)
        y = y / (np.max(np.abs(y)) + 1e-9) * float(rng.uniform(0.3, 0.95))
        y = audio_augment(y, rng)
        nums = beat_numbers(g_beats, g_db)
        np_mel = quantize(logmel(y))
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, mel=np_mel, beats=g_beats, downbeats=g_beats[nums == 1], nums=nums)
        rp = Path(out) / f"{split}_refs" / f"{name}.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps([{"time": round(float(t), 4), "beat": int(n)} for t, n in zip(g_beats, nums)]))
        (Path(out) / f"{split}_meta").mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{split}_meta" / f"{name}.json").write_text(json.dumps(info))
        if int(hashlib.md5(name.encode()).hexdigest()[:2], 16) < 6:
            (Path(out) / "listen").mkdir(parents=True, exist_ok=True)
            sf.write(Path(out) / "listen" / f"{name}.wav", y, SR, subtype="PCM_16")
        return "ok_mix" if "partner" in info else "ok_drums"
    except Exception as e:  # noqa: BLE001
        if os.environ.get("RENDER_DEBUG"):
            traceback.print_exc()
        return "err:" + type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmd", required=True, help="папка groove/ с info.csv")
    ap.add_argument("--out", default="data/groove")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--min_dur", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=10**9)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(Path(a.gmd) / "info.csv", encoding="utf-8"))
            if r["beat_type"] == "beat" and r["audio_filename"] and float(r["duration"]) >= a.min_dur
            and r["time_signature"] in ("4-4", "3-4", "6-8")][: a.limit]
    partners = lakh_partners(len(rows) * a.k * 6, 0)
    jobs = []
    for i, r in enumerate(rows):
        kk = 1 if r["split"] != "train" else a.k  # val/test — по одному варианту
        for k in range(kk):
            seed = int(hashlib.md5(f"{r['midi_filename']}{k}".encode()).hexdigest()[:8], 16)
            j = (i * a.k + k) * 6
            jobs.append((r, a.gmd, a.out, k, [partners[(j + q) % len(partners)] for q in range(6)], seed))
    print("performances", len(rows), "jobs", len(jobs), flush=True)
    stats = {}
    with Pool(a.workers, maxtasksperchild=30) as pool:
        for i, r in enumerate(pool.imap_unordered(render_one, jobs, chunksize=2)):
            stats[r] = stats.get(r, 0) + 1
            if i % 100 == 0:
                print(i, len(jobs), stats, flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
