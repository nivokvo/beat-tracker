"""Своя постобработка: HMM «указатель в такте» (идея Krebs/Böck), Viterbi на numpy.

Состояние = (размер m, интервал доли i кадров, позиция p в такте, 0..m*i-1).
Позиция растёт на 1 каждый кадр. На границе доли интервал может смениться на соседний
(штраф exp(-λ·|ln(j/i)|)) — темп плавно «дышит», но вдвое скакнуть не может, потому что
переход ×2 стоит очень дорого. Фаза такта меняется только вместе с тактом — сильная доля
не «переползает».
"""
import os

import numpy as np

FPS = 50.0
LAM = float(os.environ.get("HMM_LAM", "30"))  # для опытов; в рабочей версии 30
MIN_BPM = float(os.environ.get("HMM_MINBPM", "55"))  # SMC-разбор (2026): 55 BPM ломает медленные песни
METERS = tuple(int(x) for x in os.environ.get("HMM_METERS", "3,4").split(","))  # варианты долей в такте
OBS_LAMBDA = float(os.environ.get("HMM_OBS", "64"))
DOWN_SCALE = float(os.environ.get("HMM_DOWN", "1"))  # усиление сигнала сильной доли: сеть недооценивает её (обычно ~0,3 при доле ~0,9)  # во сколько раз «не-доля» шире доли в модели наблюдений


class BarHMM:
    def __init__(self, min_bpm=MIN_BPM, max_bpm=215, meters=METERS, n_tempi=60, lam=LAM, obs_lambda=OBS_LAMBDA,
                 meter_prior=None, tempo_w=1.0, tempo_center=110.0, tempo_sigma=0.5):
        lo, hi = 60 * FPS / max_bpm, 60 * FPS / min_bpm
        iv = np.unique(np.round(np.exp(np.linspace(np.log(lo), np.log(hi), n_tempi))).astype(int))
        self.iv = iv
        self.meters = meters
        self.obs_lambda = obs_lambda
        # индексы состояний
        first, size, info = [], [], []
        off = 0
        for m in meters:
            for i in iv:
                first.append(off)
                size.append(m * i)
                info.append((m, i))
                off += m * i
        self.first = np.array(first)
        self.size = np.array(size)
        self.info = info
        self.S = off
        pos = np.concatenate([np.arange(s) for s in self.size])
        ivs = np.concatenate([np.full(s, i) for (m, i), s in zip(info, self.size)])
        self.pos_in_beat = pos % ivs
        self.beat_idx = pos // ivs
        self.ivs = ivs
        # переходы интервалов на границе доли
        li = np.log(iv)
        self.T_iv = -lam * np.abs(li[:, None] - li[None, :])  # log-prob (до нормировки)
        self.T_iv -= np.logaddexp.reduce(self.T_iv, axis=1, keepdims=True)
        # априорный темп: штраф на каждую долю за уход от ~110 BPM (против ошибок «вдвое»)
        bpm = 60 * FPS / iv
        self.T_iv = self.T_iv - tempo_w * 0.5 * (np.log(bpm / tempo_center) / tempo_sigma) ** 2
        mp = meter_prior or {m: 1.0 / len(meters) for m in meters}
        self.log_meter = {m: np.log(mp[m]) for m in meters}

    def observation(self, beat, down):
        """beat, down: (T,) вероятности. → (T, 3) логи для классов [не-доля, доля, сильная]."""
        eps = 1e-6
        beat = np.clip(beat, eps, 1 - eps)
        down = np.clip(down, eps, 1 - eps)
        down = np.minimum(down * DOWN_SCALE, beat * 0.999)
        on_beat = np.clip(beat - down, eps, 1)
        non = (1 - beat) / (self.obs_lambda - 1)
        return np.log(np.stack([non, on_beat, down], 1))

    def state_class(self):
        width = np.maximum(1, self.ivs // self.obs_lambda)
        on = self.pos_in_beat < width
        cls = np.zeros(self.S, dtype=np.int64)
        cls[on & (self.beat_idx > 0)] = 1
        cls[on & (self.beat_idx == 0)] = 2
        return cls

    def _prep(self):
        if hasattr(self, "_starts"):
            return
        self._per_m = {}
        for m in self.meters:
            gs = [g for g, (mm, _) in enumerate(self.info) if mm == m]
            ends = np.stack([self.first[g] + np.arange(m) * self.info[g][1] + self.info[g][1] - 1 for g in gs])
            starts = np.stack([self.first[g] + np.arange(m) * self.info[g][1] for g in gs])
            ends_prev = np.roll(ends, 1, axis=1)  # для доли k' предыдущая — k'-1
            gsel = np.array(gs)
            T_iv = self.T_iv  # интервалы в группах идут в том же порядке, что self.iv
            self._per_m[m] = (ends_prev, starts, T_iv)
        self._starts = np.concatenate([v[1].ravel() for v in self._per_m.values()])
        self._is_start = np.zeros(self.S, bool)
        self._is_start[self._starts] = True

    def decode(self, beat, down):
        """Viterbi. Возвращает список (время, номер доли в такте)."""
        self._prep()
        T = len(beat)
        obs = self.observation(beat, down)
        cls = self.state_class()
        delta = np.full(self.S, -np.log(self.S))
        for g, (m, i) in enumerate(self.info):
            delta[self.first[g]: self.first[g] + self.size[g]] += self.log_meter[m]
        delta = delta + obs[0, cls]
        n_start = len(self._starts)
        bp = np.empty((T, n_start), dtype=np.int32)  # предки только для начал долей
        shift_src = np.arange(self.S) - 1
        not_start = ~self._is_start
        for t in range(1, T):
            new = np.empty(self.S)
            new[not_start] = delta[shift_src[not_start]]
            col = 0
            for m, (ends_prev, starts, T_iv) in self._per_m.items():
                E = delta[ends_prev]  # (G, m)
                sc = E[:, None, :] + T_iv[:, :, None]  # (G, G', m)
                arg = sc.argmax(0)  # (G', m)
                best = np.take_along_axis(sc, arg[None], 0)[0]
                new[starts.ravel()] = best.ravel()
                src = np.take_along_axis(ends_prev, arg, 0)  # (G', m)
                k = starts.size
                bp[t, col: col + k] = src.ravel()
                col += k
            delta = new + obs[t, cls]
        start_pos = np.full(self.S, -1, dtype=np.int64)
        start_pos[self._starts] = np.arange(n_start)
        path = np.empty(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        self.last_score = float(delta.max()) / T  # лог-правдоподобие лучшего пути на кадр
        for t in range(T - 1, 0, -1):
            s = path[t]
            path[t - 1] = bp[t, start_pos[s]] if start_pos[s] >= 0 else s - 1
        pib = self.pos_in_beat[path]
        bidx = self.beat_idx[path]
        return [(t / FPS, int(bidx[t]) + 1) for t in range(T) if pib[t] == 0]


def trim_silence(events, beat, thr=0.05):
    """Отбросить доли до первой и после последней заметной активации (тишина в начале/конце)."""
    act = np.where(beat > thr)[0]
    if len(act) == 0:
        return []
    a, z = act[0] / FPS - 0.1, act[-1] / FPS + 0.1
    return [(t, b) for t, b in events if a <= t <= z]
