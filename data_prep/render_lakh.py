"""Рендер Lakh MIDI (CC BY 4.0) в аудио со своими аугментациями → лог-мел + точные доли из MIDI.

python data_prep/render_lakh.py --midi data/raw/lmd/lmd_full --out data/synth --n 6000 --workers 14
Каждая песня → <out>/<split>/<md5>_<k>.npz (mel float16 (T,128), beats, downbeats)
и <out>/<split>_refs/<md5>_<k>.json в формате Qildo для evaluate.py.
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import SR, logmel, quantize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLUIDSYNTH = next((ROOT / "tools" / "fluidsynth").rglob("fluidsynth.exe"), None)
_SF_ALL = {
    "fluidr3": ROOT / "data/raw/sf_fluidr3/FluidR3_GM.sf2",
    "musescore": ROOT / "data/raw/sf_musescore/MuseScore_General.sf2",
}
# только полностью скачанные (пока MuseScore качается, им не пользуемся)
SOUNDFONTS = {k: v for k, v in _SF_ALL.items() if v.exists() and not v.with_suffix(".part").exists()}
if os.environ.get("SOUNDFONTS"):
    SOUNDFONTS = {k: v for k, v in SOUNDFONTS.items() if k in os.environ["SOUNDFONTS"].split(",")}
ALLOWED_TS = {(2, 4), (3, 4), (4, 4), (6, 8), (12, 8)}
MAX_DUR = 240.0


def split_of(md5):
    h = hashlib.md5(md5.encode()).hexdigest()
    return "val" if int(h[:4], 16) % 20 == 0 else "train"


def quantized_fraction(pm, beats):
    """Доля начал нот, лежащих на сетке 1/4 доли (±25 мс) — признак того, что MIDI выровнен по своей темп-карте."""
    on = np.concatenate([[n.start for n in i.notes] for i in pm.instruments if i.notes] or [[]])
    if len(on) < 50 or len(beats) < 8:
        return 0.0
    on = on[on < beats[-1]]
    idx = np.clip(np.searchsorted(beats, on) - 1, 0, len(beats) - 2)
    ibi = beats[idx + 1] - beats[idx]
    frac = (on - beats[idx]) / ibi
    # сетка шестнадцатых или триольная (свинг, шаффл, 6/8): берём ближайшую
    err = np.min([np.abs(frac * k - np.round(frac * k)) * ibi / k for k in (4, 3, 6)], axis=0)
    return float(np.mean(err < 0.025))


def beat_numbers(beats, downbeats):
    nums = np.zeros(len(beats), dtype=int)
    db = set(np.round(downbeats, 4))
    k = 0
    started = False
    for i, b in enumerate(beats):
        if round(b, 4) in db:
            k = 1
            started = True
        elif started:
            k += 1
        nums[i] = k
    # доли до первой сильной: нумеруем назад от неё
    first = np.argmax(nums > 0) if started else len(beats)
    if started and first > 0:
        meter = int(np.max(nums)) if np.max(nums) > 1 else 4
        meter = max(2, min(meter, 12))
        for i in range(first - 1, -1, -1):
            nums[i] = (nums[i + 1] - 2) % meter + 1
    return nums


SWING_P = float(os.environ.get("SWING_P", "0"))
RUBATO_P = float(os.environ.get("RUBATO_P", "0"))
# ONLY_METER=3: рендерим только 3/4 и 6/8 (у модели почти нет вальсов — Lakh в основном 4/4)
ONLY_METER = os.environ.get("ONLY_METER")


def rubato(pm, rng, strong_until=0.0):
    """Свободный темп: длина каждой доли умножается на плавную случайную кривую (сумма медленных синусов
    с периодом 3–16 долей, ±10–25 %) плюс редкие ферматы в конце такта. Ноты и темп-карта сдвигаются
    вместе (adjust_times), поэтому pm.get_beats() после этого — точные доли «живого» исполнения.
    До strong_until (вступление одним инструментом) колебания вдвое сильнее."""
    beats = pm.get_beats()
    if len(beats) < 8:
        return {}
    ibi = np.diff(beats)
    k = np.arange(len(ibi))
    amp = float(rng.uniform(0.10, 0.25))
    curve = np.zeros(len(ibi))
    for _ in range(3):
        per = float(rng.uniform(3, 16))
        curve += np.sin(2 * np.pi * k / per + rng.uniform(0, 2 * np.pi)) / 3
    curve = curve * amp + rng.normal(0, 0.03, len(ibi))  # дрожание отдельных долей
    if strong_until > 0:  # после свободного вступления группа держит темп почти ровно
        curve[beats[:-1] >= strong_until] *= 0.3
    downs = set(np.round(pm.get_downbeats(), 4))
    for i in range(len(ibi) - 1):  # фермата: последняя доля такта тянется в 1,4–2 раза
        if round(beats[i + 1], 4) in downs and rng.random() < 0.06:
            curve[i] += float(np.log(rng.uniform(1.4, 2.0)))
    new_ibi = ibi * np.exp(np.clip(curve, -0.7, 0.9))
    new = np.concatenate([[beats[0]], beats[0] + np.cumsum(new_ibi)])
    end = pm.get_end_time()
    orig = np.concatenate([[0.0], beats, [max(end, beats[-1]) + 1.0]])
    tgt = np.concatenate([[0.0], new, [new[-1] + max(end, beats[-1]) + 1.0 - beats[-1]]])
    if beats[0] <= 0:
        orig, tgt = orig[1:], tgt[1:]
    pm.adjust_times(orig, tgt)
    return {"rubato_amp": round(amp, 3), "rubato_intro": round(float(strong_until), 1)}


def swingable(pm):
    """Простой размер и ровные восьмые: на слабой восьмой (0,5 доли) нот больше, чем на триольных позициях."""
    ts = {(t.numerator, t.denominator) for t in pm.time_signature_changes}
    if not ts <= {(2, 4), (3, 4), (4, 4)}:
        return False
    beats = pm.get_beats()
    on = np.concatenate([[n.start for n in i.notes] for i in pm.instruments if i.notes] or [[]])
    on = on[on < beats[-1]] if len(beats) > 8 else on[:0]
    if len(on) < 50:
        return False
    idx = np.clip(np.searchsorted(beats, on) - 1, 0, len(beats) - 2)
    frac = (on - beats[idx]) / (beats[idx + 1] - beats[idx])
    straight = np.sum(np.abs(frac - 0.5) < 0.04)
    trip = np.sum(np.abs(frac - 2 / 3) < 0.04) + np.sum(np.abs(frac - 1 / 3) < 0.04)
    return straight >= 20 and straight > 2 * trip


def swing(pm, s):
    beats = pm.get_beats()
    end = pm.get_end_time()
    mids = beats[:-1] + 0.5 * np.diff(beats)
    orig = np.sort(np.concatenate([[0.0], beats, mids, [end + 1.0]]))
    new = np.sort(np.concatenate([[0.0], beats, beats[:-1] + s * np.diff(beats), [end + 1.0]]))
    orig, keep = np.unique(orig, return_index=True)
    pm.adjust_times(orig, new[keep])


def augment(pm, rng):
    info = {}
    # темп ×0.8…1.25
    scale = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
    end = pm.get_end_time()
    pm.adjust_times([0, end], [0, end / scale])
    info["tempo_scale"] = scale
    # свинг: вторая восьмая каждой доли сдвигается на s ∈ [0,56; 0,68] доли (кусочно-линейно внутри доли).
    # Границы долей не двигаются, поэтому разметка остаётся точной.
    if rng.random() < SWING_P and swingable(pm):
        s = float(rng.uniform(0.56, 0.68))
        swing(pm, s)
        info["swing"] = round(s, 3)
    # смена тембров внутри семейства
    if rng.random() < 0.4:
        for ins in pm.instruments:
            if not ins.is_drum:
                ins.program = (ins.program // 8) * 8 + int(rng.integers(0, 8))
        info["programs"] = "shuffled"
    # без барабанов вообще
    if rng.random() < 0.15:
        pm.instruments = [i for i in pm.instruments if not i.is_drum]
        info["drums"] = "removed"
    # «вступление одним инструментом»: первые 2–8 тактов играет один мелодический инструмент
    elif rng.random() < (0.6 if RUBATO_P >= 1 else 0.35):
        db = pm.get_downbeats()
        n_bars = int(rng.integers(4, 17)) if RUBATO_P >= 1 else int(rng.integers(2, 9))
        if len(db) > n_bars + 2:
            cut = db[n_bars]
            melodic = [i for i in pm.instruments if not i.is_drum and i.notes]
            if melodic:
                keep = melodic[int(rng.integers(len(melodic)))]
                if RUBATO_P >= 1 and rng.random() < 0.5:
                    keep.program = int(rng.integers(0, 8))  # чаще фортепиано, как в балладах
                for ins in pm.instruments:
                    if ins is not keep:
                        ins.notes = [n for n in ins.notes if n.start >= cut]
                info["solo_intro_bars"] = n_bars
                info["_cut"] = float(cut)
    # свободный темп (rubato): после всех правок нот, чтобы сдвинуть всё вместе
    if rng.random() < RUBATO_P:
        info.update(rubato(pm, rng, strong_until=info.get("_cut", 0.0)))
    info.pop("_cut", None)
    return info


def audio_augment(y, rng):
    """Свои аугментации тембра поверх саундфонтов: EQ, перегруз, реверб, сжатие полосы."""
    from scipy.signal import butter, fftconvolve, sosfilt
    if rng.random() < 0.5:  # наклон спектра / полки
        f = float(np.exp(rng.uniform(np.log(150), np.log(6000))))
        kind = "low" if rng.random() < 0.5 else "high"
        sos = butter(2, f, btype=kind, fs=SR, output="sos")
        g = float(rng.uniform(0.2, 0.8))
        y = (1 - g) * y + g * sosfilt(sos, y).astype(np.float32) * 2
    if rng.random() < 0.2:  # перегруз
        k = float(rng.uniform(2, 15))
        y = np.tanh(k * y) / np.tanh(k)
    if rng.random() < 0.4:  # реверб: экспоненциально затухающий шум
        t60 = float(rng.uniform(0.2, 2.0))
        n = int(t60 * SR)
        ir = rng.normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)
        ir[0] = 1.0 / float(rng.uniform(0.05, 0.5))
        wet = fftconvolve(y, ir)[: len(y)]
        y = y + float(rng.uniform(0.1, 0.5)) * wet / (np.max(np.abs(wet)) + 1e-9) * np.max(np.abs(y))
    if rng.random() < 0.15:  # «телефон»/дешёвый mp3: полоса 200–4000 Гц
        sos = butter(4, [200, 4000], btype="band", fs=SR, output="sos")
        y = sosfilt(sos, y)
    y = np.asarray(y, dtype=np.float32)
    return y / (np.max(np.abs(y)) + 1e-9) * float(rng.uniform(0.3, 0.95))


def render_one(job):
    path, out, k, seed = job
    try:
        md5 = Path(path).stem
        split = split_of(md5)
        name = f"{md5}_{k}"
        npz_path = Path(out) / split / f"{name}.npz"
        if npz_path.exists():
            return "exists"
        rng = np.random.default_rng(seed)
        pm = pretty_midi.PrettyMIDI(str(path))
        ts = {(t.numerator, t.denominator) for t in pm.time_signature_changes}
        if not ts or not ts <= ALLOWED_TS:
            return "ts"
        if ONLY_METER and not ts <= {(3, 4), (6, 8)}:
            return "meter"
        dur = pm.get_end_time()
        if dur < 30 or dur > 900:
            return "dur"
        if len([i for i in pm.instruments if i.notes]) < 3:
            return "few_instr"
        if quantized_fraction(pm, pm.get_beats()) < 0.85:
            return "unquantized"
        if SWING_P >= 1 and not swingable(pm):  # режим «только свинг»: ровные треки не рендерим
            return "no_swing"
        info = augment(pm, rng)
        beats = pm.get_beats()
        downbeats = pm.get_downbeats()
        # обрезка до MAX_DUR
        end = min(pm.get_end_time(), MAX_DUR)
        for ins in pm.instruments:  # не рендерим то, что потом отрежем
            ins.notes = [n for n in ins.notes if n.start < end]
            ins.control_changes = [c for c in ins.control_changes if c.time < end]
            ins.pitch_bends = [b for b in ins.pitch_bends if b.time < end]
        if not any(i.notes for i in pm.instruments):
            return "empty"
        sf_name = list(SOUNDFONTS)[int(rng.integers(len(SOUNDFONTS)))]
        with tempfile.TemporaryDirectory() as td:
            mid = os.path.join(td, "a.mid")
            wav = os.path.join(td, "a.wav")
            pm.write(mid)
            subprocess.run(
                [str(FLUIDSYNTH), "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5", "-r", str(SR), "-F", wav, str(SOUNDFONTS[sf_name]), mid],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            )
            y, sr = sf.read(wav, dtype="float32")
        assert sr == SR
        if y.ndim == 2:
            y = y.mean(1)
        y = y[: int(end * SR)]
        # первая нота — начало «песни»: тишину в начале оставляем как есть (так и в реальных треках)
        peak = np.max(np.abs(y)) + 1e-9
        y = y / peak * float(rng.uniform(0.3, 0.95))
        if rng.random() < 0.3:
            y = y + rng.normal(0, float(rng.uniform(1e-4, 3e-3)), size=y.shape).astype(np.float32)
        y = audio_augment(y, rng)
        mel = logmel(y)
        last = len(y) / SR
        # доли только там, где играет музыка: от первой ноты до последней
        first_note = min(n.start for i in pm.instruments for n in i.notes)
        last_note = min(max(n.end for i in pm.instruments for n in i.notes), last)
        nums_all = beat_numbers(beats, downbeats)
        m = (beats >= first_note - 0.05) & (beats <= last_note)
        beats, nums = beats[m], nums_all[m]
        downbeats = beats[nums == 1]
        if len(beats) < 16 or len(downbeats) < 4:
            return "few_beats"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, mel=quantize(mel), beats=beats, downbeats=downbeats, nums=nums)
        ref = [{"time": round(float(t), 4), "beat": int(n)} for t, n in zip(beats, nums)]
        rp = Path(out) / f"{split}_refs" / f"{name}.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(ref))
        info.update(soundfont=sf_name, src=md5, dur=round(last, 1))
        (Path(out) / f"{split}_meta").mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{split}_meta" / f"{name}.json").write_text(json.dumps(info))
        # немного wav для прослушивания
        if int(hashlib.md5(md5.encode()).hexdigest()[4:6], 16) < 4 or os.environ.get("RENDER_SAVE_WAV"):
            (Path(out) / "listen").mkdir(parents=True, exist_ok=True)
            sf.write(Path(out) / "listen" / f"{name}.wav", y, SR, subtype="PCM_16")
        return "ok"
    except Exception as e:  # битые MIDI в Lakh не редкость
        if os.environ.get("RENDER_DEBUG"):
            traceback.print_exc()
        return "err:" + type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000, help="сколько MIDI попробовать")
    ap.add_argument("--k", type=int, default=1, help="рендеров на один MIDI")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.midi).rglob("*.mid"))
    random.Random(a.seed).shuffle(files)
    files = files[: a.n]
    jobs = [(str(f), a.out, k, int(hashlib.md5(f"{f.stem}{k}{a.seed}".encode()).hexdigest()[:8], 16))
            for f in files for k in range(a.k)]
    stats = {}
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(render_one, jobs, chunksize=4)):
            key = r.split(":")[0] if r.startswith("err") else r
            stats[key] = stats.get(key, 0) + 1
            if i % 200 == 0:
                print(i, len(jobs), stats, flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
