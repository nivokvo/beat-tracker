"""Наложить метроном на песню по сетке, чтобы проверить разметку на слух.

python tools_py/click_overlay.py song.mp3 grid.json out.wav
Сильная доля — высокий щелчок (1500 Гц), остальные — низкий (900 Гц). Стерео: песня слева+справа, щелчки в центре.
"""
import json
import sys

import librosa
import numpy as np
import soundfile as sf

SR = 44100


def click(freq, dur=0.04):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 80)


def main(song, grid, out):
    y, _ = librosa.load(song, sr=SR, mono=True)
    y = 0.6 * y / (np.max(np.abs(y)) + 1e-9)
    c = np.zeros_like(y)
    hi, lo = click(1500), click(900)
    for ev in json.load(open(grid, encoding="utf-8")):
        s = int(ev["time"] * SR)
        k = hi if ev["beat"] == 1 else lo * 0.7
        e = min(len(c), s + len(k))
        if 0 <= s < len(c):
            c[s:e] += k[: e - s] * 0.5
    sf.write(out, np.clip(y + c, -1, 1), SR, subtype="PCM_16")


if __name__ == "__main__":
    main(*sys.argv[1:4])
