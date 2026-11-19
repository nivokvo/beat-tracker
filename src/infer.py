"""Инференс: аудио → активации модели (окнами с перекрытием) → HMM → [{"time","beat"}]."""
import os

import numpy as np
import torch

from features import FPS, load_audio, logmel
from model import BeatModel
from postprocess import METERS, BarHMM, trim_silence

METER_MARGIN = float(os.environ.get("METER_MARGIN", "0.03"))  # запас контраста для смены размера
PAD_S = 0.0  # с тишины перед записью (опыт: PAD_S=1; в v2 — по итогам срезов)
BEAT_OFFSET = 0.0  # с; калибруется на отложенных данных (tools_py/calib_offset.py)
WIN = 1500  # 30 с
HOP = 1000


def second_pass(ev, beat, down, sigma=0.2, tempo_w=1.0, meter_p=0.9):
    """Второй проход под конкретную песню: темп-априор с центром в темпе первого прохода (вместо общих 110 BPM)
    и априор размера, найденного в первом проходе. Песня держит один темп и размер — перескоки «вдвое»
    и смена размера посреди песни становятся дорогими."""
    t = np.array([e[0] for e in ev])
    if len(t) < 16:
        return ev
    bpm = 60.0 / np.median(np.diff(t))
    nums = np.array([e[1] for e in ev])
    starts = np.flatnonzero(nums == 1)
    lens = np.diff(starts)
    meter = int(np.bincount(lens).argmax()) if len(lens) else 4
    meters = METERS
    mp = {m: (meter_p if m == meter else (1 - meter_p) / max(1, len(meters) - 1)) for m in meters} if meter in meters else None
    hmm = BarHMM(tempo_center=float(np.clip(bpm, 55, 215)), tempo_sigma=sigma, tempo_w=tempo_w, meter_prior=mp)
    return hmm.decode(beat, down)


def meter_contrast(act, frames, m):
    """Насколько сильнее активация сильной доли на каждой m-й доле (лучшая фаза), чем на остальных."""
    d = np.array([act[max(0, i - 2): i + 3, 1].max() for i in frames])
    best = -9.0
    for ph in range(m):
        on, off = d[ph::m], np.delete(d, np.arange(ph, len(d), m))
        if len(on) > 3 and len(off) > 3:
            best = max(best, float(on.mean() - off.mean()))
    return best


def select_meter(ev, beat, down, act):
    """HMM почти всегда выбирает такт на 4: когда активация сильной доли умеренная (~0,2–0,4), путь с меньшим
    числом сильных долей дешевле. Поэтому размер выбираем отдельно — по контрасту активации сильной доли
    на каждой 3-й и каждой 4-й доле — и при несовпадении декодируем заново с этим размером."""
    if len(ev) < 16:
        return ev
    frames = np.clip(np.round(np.array([e[0] for e in ev]) * FPS).astype(int), 0, len(act) - 1)
    c = {m: meter_contrast(act, frames, m) for m in METERS}
    m = max(c, key=c.get)
    cur = max(e[1] for e in ev)
    # менять размер только при явном перевесе: при почти равных контрастах решение случайно
    if m == cur or cur not in c or c[m] < c[cur] + METER_MARGIN:
        return ev
    return BarHMM(meters=(m,)).decode(beat, down)


def beat_alternation(act, ev):
    """min/max средней активации доли на чётных и нечётных долях. ≈1 — доли ровные; заметно меньше —
    каждая вторая «доля» слабее, то есть сетка считает восьмые (свинг, шаффл) — темп вдвое."""
    fr = np.clip(np.round(np.array([e[0] for e in ev]) * FPS).astype(int), 0, len(act) - 1)
    b = np.array([act[max(0, i - 2): i + 3, 0].max() for i in fr])
    e, o = b[0::2].mean(), b[1::2].mean()
    return float(min(e, o) / max(e, o, 1e-9))


def halve_if_eighths(ev, beat, down, act, thr):
    if len(ev) < 32:
        return ev
    bpm = 60.0 / np.median(np.diff([e[0] for e in ev]))
    if bpm / 2 < 55 or beat_alternation(act, ev) >= thr:
        return ev
    ev2 = BarHMM(max_bpm=bpm * 0.7, tempo_center=bpm / 2).decode(beat, down)
    return select_meter(ev2, beat, down, act)


def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = BeatModel(**ck.get("cfg", {}))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), device


@torch.no_grad()
def activations(model, device, mel):
    """Активации (T, 2). Перед записью добавляются PAD_S секунд тишины (нулевой лог-мел), потом срезаются:
    без контекста сеть не видит удар в самом первом кадре (песня, начатая «с раза»)."""
    pad = int(round(float(os.environ.get("PAD_S", PAD_S)) * FPS))
    if pad:
        mel = np.concatenate([np.zeros((pad, np.asarray(mel).shape[1]), np.float32), np.asarray(mel, np.float32)])
        return _activations(model, device, mel)[pad:]
    return _activations(model, device, mel)


@torch.no_grad()
def _activations(model, device, mel):
    T = len(mel)
    acc = np.zeros((T, 2))
    w = np.zeros(T)
    win_fn = np.hanning(WIN + 2)[1:-1] + 1e-3
    starts = list(range(0, max(T - WIN, 0) + 1, HOP))
    if not starts or starts[-1] + WIN < T:
        starts.append(max(T - WIN, 0))
    x = torch.from_numpy(np.asarray(mel, dtype=np.float32)).to(device)
    for s in starts:
        seg = x[s: s + WIN][None]
        p = torch.sigmoid(model(seg))[0].float().cpu().numpy()
        n = len(p)
        acc[s: s + n] += p * win_fn[:n, None]
        w[s: s + n] += win_fn[:n]
    return acc / np.maximum(w, 1e-9)[:, None]


def beats_from_act(act, hmm=None, select_tempo=False):  # выбор уровня темпа не дал прироста с r4 (REPORT.md)
    beat, down = act[:, 0], act[:, 1]
    if hmm is None and select_tempo:
        from tempo_select import choose  # выбор уровня темпа (weights/tempo_select.json), иначе обычная HMM
        ev = choose(act)
    else:
        ev = (hmm or BarHMM()).decode(beat, down)
        if hmm is None and os.environ.get("METER_SELECT", "1") == "1":
            ev = select_meter(ev, beat, down, act)
        if hmm is None and os.environ.get("HALVE_THR"):
            ev = halve_if_eighths(ev, beat, down, act, float(os.environ["HALVE_THR"]))
        if hmm is None and os.environ.get("TWO_PASS"):
            ev = second_pass(ev, beat, down, *map(float, os.environ["TWO_PASS"].split(",")))
    ev = trim_silence(ev, beat)
    off = float(os.environ.get("BEAT_OFFSET", str(BEAT_OFFSET)))  # пики сети опаздывают на ~20 мс (лосс с допуском ±3 кадра)
    return [{"time": round(max(0.0, float(t) + off), 4), "beat": int(b)} for t, b in ev]


def sound_span(mel, thr=0.3):
    """Где в записи вообще есть звук: первый и последний кадр с энергией выше порога (лог-мел; тишина = 0)."""
    loud = np.flatnonzero(np.asarray(mel, np.float32).max(1) > thr)
    if len(loud) == 0 or loud[-1] - loud[0] < FPS // 2:  # меньше полсекунды звучания
        return None, None
    return loud[0] / FPS - 0.1, loud[-1] / FPS + 0.1


def run_file(model, device, wav_path, hmm=None):
    """model — одна модель или список [(model, device), ...] (ансамбль: среднее активаций)."""
    mel = logmel(load_audio(wav_path))
    models = model if isinstance(model, list) else [(model, device)]
    act = np.mean([activations(m, d, mel) for m, d in models], axis=0)
    a, z = sound_span(mel)
    if a is None:  # тишина или меньше полсекунды звука — долей нет
        return []
    out = []
    for e in beats_from_act(act, hmm):
        if a <= e["time"] <= z:
            out.append(e)
    return out
