"""Метрики Qildo по срезам: стиль GuitarSet (живая гитара соло — заодно «вступление одним инструментом»),
медленные (<90 BPM), Slakh test (рок/поп из MIDI). Те же функции, что в evaluate.score_one.

python src/eval_slices.py runs/r4/s10000.pt runs/r6/s15000.pt [...]
Пишет runs/slices_<имя>.json и печатает таблицу.
"""
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import score_one  # noqa: E402
from features import dequantize  # noqa: E402
from infer import activations, beats_from_act, load_model  # noqa: E402

KEYS = ["beat_F", "downbeat_F", "CMLt", "AMLt", "missed_start", "tempo_x2_frac", "downbeat_wrong_frac"]
SHORT = ["bF", "dbF", "CMLt", "AMLt", "miss0", "x2", "dbWrong"]
GS_STYLE = {"BN": "gs:bossa", "Funk": "gs:funk", "Jazz": "gs:jazz/swing", "Rock": "gs:rock", "SS": "gs:songwriter"}


def ref_bpm(ref):
    t = np.array([b["time"] for b in json.loads(Path(ref).read_text())])
    return 60.0 / np.median(np.diff(t)) if len(t) > 2 else 0.0


def items():
    """(npz, ref, [срезы])"""
    out = []
    for f in sorted((ROOT / "data/real/gs_test").glob("*.npz")):
        ref = ROOT / "data/real/gs_test_refs" / (f.stem + ".json")
        m = re.match(r"\d+_([A-Za-z]+)\d-(\d+)-", f.stem)
        tags = ["gs:all", GS_STYLE.get(m.group(1), "gs:other"), "gs:" + ("solo" if f.stem.endswith("solo") else "comp")]
        if int(m.group(2)) < 90:
            tags.append("slow<90")
        out.append((f, ref, tags))
    for f in sorted((ROOT / "data/slakh/test").glob("*.npz")):
        ref = ROOT / "data/slakh/test_refs" / (f.stem + ".json")
        tags = ["slakh:all"] + (["slow<90"] if ref_bpm(ref) < 90 else [])
        out.append((f, ref, tags))
    # свободный темп: val-сплит рендера RUBATO_P=1 (40 штук), отдельно треки со свободным вступлением
    for f in sorted((ROOT / "data/synth_rubato/val").glob("*.npz"))[:40]:
        ref = ROOT / "data/synth_rubato/val_refs" / (f.stem + ".json")
        meta = json.loads((ROOT / "data/synth_rubato/val_meta" / (f.stem + ".json")).read_text())
        out.append((f, ref, ["rubato:all"] + (["rubato:intro"] if meta.get("rubato_intro") else [])))
    # живые записи с ручной разметкой (data_prep/prep_real_eval.py): rubato и живые барабаны
    for name in ("kraisler", "winterreise", "candombe"):
        for f in sorted((ROOT / "data/real_eval" / name).glob("*.npz")):
            out.append((f, ROOT / "data/real_eval" / f"{name}_refs" / (f.stem + ".json"), [f"real:{name}"]))
    # живые барабаны Groove MIDI (test-сплит, барабанщики не из обучения) с партиями Lakh или соло
    for f in sorted((ROOT / "data/groove/test").glob("*.npz")):
        ref = ROOT / "data/groove/test_refs" / (f.stem + ".json")
        meta = json.loads((ROOT / "data/groove/test_meta" / (f.stem + ".json")).read_text())
        tags = ["groove:" + ("mix" if "partner" in meta else "drums")]
        if any(s in meta["style"] for s in ("jazz", "shuffle", "blues", "funk")):
            tags.append("groove:swingy")
        if meta["style"].startswith(("rock", "pop", "punk")):
            tags.append("groove:rock/pop")
        out.append((f, ref, tags))
    return out


def run(path, its):
    only = os.environ.get("ONLY")  # напр. ONLY=real: — только срезы с этим префиксом
    if only:
        its = [x for x in its if any(t.startswith(tuple(only.split(','))) for t in x[2])]
    # "a.pt+b.pt" — ансамбль: среднее активаций. Активации кэшируются (runs/acts_slices/<модель>/),
    # так что перебор ансамблей и настроек HMM дальше идёт без сети.
    loaded = {}

    def act_of(w, f):
        tag = "_".join(Path(w).with_suffix("").parts[-2:]) + ("_pad" + os.environ["PAD_S"] if float(os.environ.get("PAD_S", 0)) else "")
        c = ROOT / "runs/acts_slices" / tag / f"{f.parent.parent.name}_{f.parent.name}__{f.stem}.npy"
        if c.exists():
            return np.load(c).astype(np.float32)
        if w not in loaded:
            loaded[w] = load_model(w)
        m, dev = loaded[w]
        a = activations(m, dev, dequantize(np.load(f)["mel"]))
        c.parent.mkdir(parents=True, exist_ok=True)
        np.save(c, a.astype(np.float16))
        return a

    rows = []
    for f, ref, tags in its:
        est = beats_from_act(np.mean([act_of(w, f) for w in path.split("+")], axis=0))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(est, tf)
        s = score_one(ref, tf.name)
        Path(tf.name).unlink()
        rows.append((tags, s))
    agg = defaultdict(list)
    for tags, s in rows:
        for t in tags:
            agg[t].append(s)
    return {t: {"n": len(v), **{k: round(float(np.mean([s[k] for s in v])), 3) for k in KEYS}} for t, v in sorted(agg.items())}


def main(paths):
    its = items()
    res = {}
    for p in paths:
        name = "+".join("_".join(Path(w).with_suffix("").parts[-2:]) for w in p.split("+"))
        name += os.environ.get("NAME_SUFFIX", "")
        res[name] = run(p, its)
        (ROOT / "runs" / f"slices_{name}.json").write_text(json.dumps(res[name], indent=1))
    tags = sorted({t for r in res.values() for t in r})
    print("| срез | n | модель | " + " | ".join(SHORT) + " |")
    print("|---|---|---|" + "---|" * len(SHORT))
    for t in tags:
        for name, r in res.items():
            if t in r:
                print(f"| {t} | {r[t]['n']} | {name} | " + " | ".join(f"{r[t][k]:.3f}" for k in KEYS) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
