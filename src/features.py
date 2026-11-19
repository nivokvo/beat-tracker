"""Общие признаки для обучения и инференса: лог-мел, 50 кадров/с."""
import numpy as np
import librosa

SR = 22050
HOP = 441  # 50 fps
N_FFT = 1024
N_MELS = 128
FPS = SR / HOP


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def logmel(y):
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=30, fmax=11000, power=1.0
    )
    return np.log1p(1000.0 * m).T.astype(np.float16)  # (T, 128)


def times_to_frames(times, n_frames):
    f = np.round(np.asarray(times) * FPS).astype(int)
    return f[(f >= 0) & (f < n_frames)]


MEL_Q = 24.0  # хранение на диске: uint8 = round(mel * 24), покрывает лог-мел до 10.6 (реально до ~9)


def quantize(mel):
    return np.clip(np.round(np.asarray(mel, np.float32) * MEL_Q), 0, 255).astype(np.uint8)


def dequantize(mel):
    mel = np.asarray(mel)
    return mel.astype(np.float32) / MEL_Q if mel.dtype == np.uint8 else mel.astype(np.float32)
