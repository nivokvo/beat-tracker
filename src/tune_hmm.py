"""Подбор параметров HMM на отложенных данных (не на тесте).

python src/tune_hmm.py runs/r2/best.pt
Активации считаются один раз, дальше перебор (lam, obs_lambda) по сетке.
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import score_one  # noqa: E402
from features import dequantize  # noqa: E402
from infer import activations, load_model  # noqa: E402
from postprocess import BarHMM, trim_silence  # noqa: E402

SETS = [("data/synth/val", "data/synth/val_refs", 40), ("data/real/gs_train", "data/real/gs_train_refs", 40)]


def main(weights):
    model, dev = load_model(weights)
    acts = []
    for d, refs, n in SETS:
        files = sorted(Path(d).glob("*.npz"))
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.choice(len(files), min(n, len(files)), replace=False)]
        for f in files:
            acts.append((d, Path(refs) / (f.stem + ".json"), activations(model, dev, dequantize(np.load(f)["mel"]))))
    print("acts", len(acts), flush=True)
    grid = list(itertools.product([0.0, 0.5, 1.0, 2.0, 4.0], [32, 64]))
    out = []
    for tw, ol in grid:
        lam = 30
        hmm = BarHMM(lam=lam, obs_lambda=ol, tempo_w=tw)
        per = {}
        for d, ref, act in acts:
            ev = trim_silence(hmm.decode(act[:, 0], act[:, 1]), act[:, 0])
            est = [{"time": round(float(t), 4), "beat": int(b)} for t, b in ev]
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(est, tf)
            s = score_one(ref, tf.name)
            Path(tf.name).unlink()
            per.setdefault(d, []).append(s)
        row = {"tempo_w": tw, "obs_lambda": ol}
        for d, ss in per.items():
            k = Path(d).name
            for m in ["beat_F", "downbeat_F", "CMLt", "AMLt", "tempo_x2_frac"]:
                row[f"{k}:{m}"] = round(float(np.mean([s[m] for s in ss])), 3)
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(sys.argv[2] if len(sys.argv) > 2 else "runs/hmm_tune.jsonl").write_text("\n".join(json.dumps(r) for r in out))


if __name__ == "__main__":
    main(sys.argv[1])
