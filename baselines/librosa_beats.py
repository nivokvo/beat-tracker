"""Бейзлайн: librosa.beat.beat_track + сильная доля по максимуму низкочастотного акцента (такт 4/4).

python baselines/librosa_beats.py song.wav out.json
"""
import json
import sys

import librosa
import numpy as np


def run(path, meter=4):
    y, sr = librosa.load(path, sr=22050, mono=True)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(frames) == 0:
        return []
    # низкочастотный акцент (бочка/бас) в каждой доле
    S = np.abs(librosa.stft(y, hop_length=hop, n_fft=2048))
    low = S[: int(150 / (sr / 2048))].sum(0)
    acc = low[np.clip(frames, 0, len(low) - 1)]
    best = max(range(meter), key=lambda p: acc[p::meter].mean() if len(acc[p::meter]) else 0)
    nums = [((i - best) % meter) + 1 for i in range(len(times))]
    return [{"time": round(float(t), 4), "beat": int(n)} for t, n in zip(times, nums)]


if __name__ == "__main__":
    out = run(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f)
