"""GuitarSet (CC BY 4.0, https://zenodo.org/records/3371780) → npz + refs.

Игрок 05 — тест (внешний судья), остальные — обучение.
python data_prep/prep_guitarset.py
"""
import json
import sys
from pathlib import Path

import jams
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import load_audio, logmel, quantize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/guitarset/gx"
OUT = ROOT / "data/real"


def main():
    for jf in sorted((SRC / "annotation").glob("*.jams")):
        name = jf.stem
        split = "gs_test" if name.startswith("05_") else "gs_train"
        wav = SRC / "audio" / f"{name}_mic.wav"
        j = jams.load(str(jf))
        a = j.search(namespace="beat_position")[0]
        t = np.array([o.time for o in a.data])
        n = np.array([o.value["position"] for o in a.data])
        dur = j.file_metadata.duration
        m = t <= dur
        t, n = t[m], n[m]
        mel = logmel(load_audio(str(wav)))
        (OUT / split).mkdir(parents=True, exist_ok=True)
        (OUT / f"{split}_refs").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / split / f"{name}.npz", mel=quantize(mel), beats=t, downbeats=t[n == 1], nums=n)
        (OUT / f"{split}_refs" / f"{name}.json").write_text(
            json.dumps([{"time": round(float(a), 4), "beat": int(b)} for a, b in zip(t, n)]))
        (OUT / f"{split}_wavs.txt").open("a").write(str(wav) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
