"""Подбор темпового априора и гибкости темпа HMM (tempo_center, tempo_sigma, lam) на НЕ тестовых данных.

python src/tune_prior.py runs/r4/s10000.pt --name r4
Наборы для подбора: GuitarSet игроки 00–04 (gs_train, есть джаз и босса), Slakh val, песни incompetech с BPM автора
(обучающая часть kml, не судья и не тест). Треки режутся до 60 с — ошибки уровня темпа видны и так.
Активации кэшируются в runs/acts_tune_<name>/.
"""
import argparse
import itertools
import json
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import score_one  # noqa: E402
from features import dequantize  # noqa: E402
from postprocess import BarHMM, trim_silence  # noqa: E402

CROP = 3000  # 60 с


def tempo_class(bpm, ref, tol=0.04):
    for name, k in (("ok", 1), ("x2", 2), ("x0.5", 0.5)):
        if abs(bpm / (ref * k) - 1) < tol:
            return name
    return "other"


def build_cache(weights, cache):
    from infer import activations, load_model
    model, dev = load_model(weights)
    rng = np.random.default_rng(0)
    jobs = []
    gs = sorted((ROOT / "data/real/gs_train").glob("*.npz"))
    jobs += [("gs", f, ROOT / "data/real/gs_train_refs" / (f.stem + ".json")) for f in gs[::2]]  # 150 из 300
    sv = sorted((ROOT / "data/slakh/val").glob("*.npz"))
    jobs += [("slakh", sv[i], ROOT / "data/slakh/val_refs" / (sv[i].stem + ".json"))
             for i in rng.choice(len(sv), 80, replace=False)]
    meta = {}
    for mf in (ROOT / "data/raw/kml").glob("meta_*.jsonl"):
        for line in open(mf, encoding="utf-8"):
            r = json.loads(line)
            meta[r["id"]] = r
    km = sorted((ROOT / "data/raw/kml/mel").glob("*.npz"))
    km = [f for f in km if f.stem in meta and 50 < float(meta[f.stem]["bpm"] or 0) < 250]
    jobs += [("kml", km[i], float(meta[km[i].stem]["bpm"])) for i in rng.choice(len(km), 200, replace=False)]
    cache.mkdir(parents=True, exist_ok=True)
    for kind, f, ref in jobs:
        out = cache / f"{kind}__{f.stem}.npz"
        if out.exists():
            continue
        d = np.load(f)
        mel = dequantize(d["mel"])
        if kind == "kml":  # середина песни, вступления у Маклауда часто без ритма
            s = max(0, len(mel) // 2 - CROP // 2)
            mel = mel[s: s + CROP]
        else:
            mel = mel[:CROP]
        act = activations(model, dev, mel)
        np.savez(out, act=act.astype(np.float32), ref=str(ref))


def eval_cfg(args):
    cache, cfg = args
    center, sigma, lam = cfg
    hmm = BarHMM(lam=lam, tempo_center=center, tempo_sigma=sigma)
    res = {"gs": [], "slakh": [], "kml": []}
    for f in sorted(Path(cache).glob("*.npz")):
        kind = f.stem.split("__")[0]
        d = np.load(f)
        act, ref = d["act"], str(d["ref"])
        ev = trim_silence(hmm.decode(act[:, 0], act[:, 1]), act[:, 0])
        if kind == "kml":
            t = np.array([e[0] for e in ev])
            bpm = 60 / np.median(np.diff(t)) if len(t) > 2 else 0
            res["kml"].append(tempo_class(bpm, float(ref)) if bpm else "other")
            continue
        est = [{"time": float(t), "beat": int(b)} for t, b in ev]
        ref_all = json.loads(Path(ref).read_text())
        ref_cut = [r for r in ref_all if r["time"] < CROP / 50]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tr, \
                tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as te:
            json.dump(ref_cut, tr)
            json.dump(est, te)
        s = score_one(tr.name, te.name)
        Path(tr.name).unlink()
        Path(te.name).unlink()
        res[kind].append((f.stem, s))
    out = {"center": center, "sigma": sigma, "lam": lam}
    for k in ("gs", "slakh"):
        for m in ("beat_F", "downbeat_F", "tempo_x2_frac"):
            out[f"{k}_{m}"] = round(float(np.mean([s[m] for _, s in res[k]])), 4)
    jz = [s for n, s in res["gs"] if "Jazz" in n or "BN" in n]
    out["gs_jazzbn_beat_F"] = round(float(np.mean([s["beat_F"] for s in jz])), 4)
    out["kml_ok"] = sum(c == "ok" for c in res["kml"])
    out["kml_x2"] = sum(c == "x2" for c in res["kml"])
    out["kml_x05"] = sum(c == "x0.5" for c in res["kml"])
    out["kml_n"] = len(res["kml"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cache = ROOT / "runs" / f"acts_tune_{a.name}"
    build_cache(a.weights, cache)
    grid = list(itertools.product([100, 110, 120, 135], [0.35, 0.5, 0.8, 2.0], [30]))
    grid += [(110, 0.5, 15), (110, 0.5, 60)]
    rows = []
    with Pool(a.workers) as p:
        for r in p.imap_unordered(eval_cfg, [(str(cache), c) for c in grid]):
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (ROOT / "runs" / f"tune_prior_{a.name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
