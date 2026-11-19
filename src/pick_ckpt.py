"""Выбор версии модели по внешним судьям: GuitarSet (игрок 05) + совпадение долей с Qildo.

python src/pick_ckpt.py runs/r3/s*.pt runs/r2/s12000.pt
"""
import json
import sys
import tempfile
from pathlib import Path

import mir_eval
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import score_one  # noqa: E402
from features import dequantize, load_audio, logmel  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402

GS = sorted(Path("data/real/gs_test").glob("*.npz"))
QILDO = ["funky_chunk", "local_forecast", "sneaky_snitch"]
QMEL = {t: logmel(load_audio(f"qildo_cmp/{t}/mix.wav")) for t in QILDO}


def main(cks):
    for ck in cks:
        m, dev = load_model(ck)
        ss = []
        for f in GS:
            est = beats_from_act(activations(m, dev, dequantize(np.load(f)["mel"])))
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(est, tf)
            ss.append(score_one(Path("data/real/gs_test_refs") / (f.stem + ".json"), tf.name))
            Path(tf.name).unlink()
        gs = {k: round(float(np.mean([s[k] for s in ss])), 3) for k in ["beat_F", "downbeat_F", "AMLt", "tempo_x2_frac"]}
        q = []
        for t in QILDO:
            est = beats_from_act(activations(m, dev, QMEL[t]))
            et = np.array([e["time"] for e in est])
            qa = json.loads(Path(f"qildo_cmp/{t}/qildo_analysis.json").read_text())
            q.append(f"{t[:6]} {mir_eval.beat.f_measure(np.array(qa['beats']), et, 0.07):.2f}")
        print(ck, torch.load(ck, weights_only=False).get("step"), "GS", gs, "| Qildo beats", ", ".join(q), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
