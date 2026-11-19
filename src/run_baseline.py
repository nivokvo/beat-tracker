"""Прогнать бейзлайн librosa по списку wav и сохранить предсказания.

python src/run_baseline.py data/real/gs_test_wavs.txt preds/librosa_gs_test
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from librosa_beats import run  # noqa: E402


def one(args):
    wav, out = args
    name = Path(wav).stem.replace("_mic", "")
    (Path(out) / f"{name}.json").write_text(json.dumps(run(wav)))


if __name__ == "__main__":
    wavs = [w.strip() for w in open(sys.argv[1]) if w.strip()]
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    with Pool(8) as p:
        p.map(one, [(w, sys.argv[2]) for w in wavs])
