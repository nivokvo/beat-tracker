"""Qildo, задача 4: доли и сильные доли.

python beats.py song.wav out.json
→ [{"time": 0.512, "beat": 1}, ...]   (beat — номер доли в такте, 1 = сильная доля)

Работает на GPU, если он есть, иначе на CPU. Сеть не нужна.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from infer import load_model, run_file  # noqa: E402

WEIGHTS = ROOT / "weights" / "beat_model.pt"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    model, device = load_model(WEIGHTS, os.environ.get("BEATS_DEVICE"))  # BEATS_DEVICE=cpu — принудительно CPU
    out = run_file(model, device, sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
