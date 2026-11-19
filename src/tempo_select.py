"""Выбор уровня темпа (метрического уровня) поверх HMM.

Главная ошибка на реальных песнях — сетка на «соседнем» уровне: вдвое быстрее/медленнее, ×1,5 (6/8 против 3/4).
HMM даёт темп T. Строим кандидатов T·{1/2, 2/3, 3/4, 1, 4/3, 3/2, 2}, для каждого — HMM в узком окне темпа,
считаем признаки сетки (активация на долях, между долями, на третях, правдоподобие, темп) и выбираем
кандидата линейной softmax-моделью. Модель учится на BPM авторов (incompetech, ccMixter) — это настоящая
разметка темпа, не наша. Проверка — 30 отложенных треков Маклауда.

python src/tempo_select.py train runs/acts_r3       # обучить → weights/tempo_select.json, отчёт на тесте
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from postprocess import FPS, BarHMM, trim_silence  # noqa: E402

RATIOS = (0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0)
BPM_LO, BPM_HI = 40.0, 240.0
WEIGHTS = ROOT / "weights" / "tempo_select.json"
_HMM_CACHE = {}


def hmm_for(c):
    key = round(c, 1)
    if key not in _HMM_CACHE:
        _HMM_CACHE[key] = BarHMM(min_bpm=c / 1.12, max_bpm=c * 1.12, n_tempi=12, tempo_w=0.0)
    return _HMM_CACHE[key]


def grid_bpm(ev):
    t = np.array([e[0] for e in ev])
    return 60 / np.median(np.diff(t)) if len(t) > 3 else 0.0


def sample(x, times):
    idx = np.clip(np.round(np.asarray(times) * FPS).astype(int), 0, len(x) - 1)
    # максимум в окне ±1 кадр — доли на сетке 20 мс
    return np.maximum.reduce([x[np.clip(idx + k, 0, len(x) - 1)] for k in (-1, 0, 1)])


def features(act, ev, c, score):
    b, d = act[:, 0].astype(np.float64), act[:, 1].astype(np.float64)
    t = np.array([e[0] for e in ev])
    nb = np.array([e[1] for e in ev])
    if len(t) < 4:
        return None
    mid = (t[:-1] + t[1:]) / 2
    th1 = t[:-1] + (t[1:] - t[:-1]) / 3
    th2 = t[:-1] + 2 * (t[1:] - t[:-1]) / 3
    on = sample(b, t)
    lt = np.log2(c / 100.0)
    down_on = sample(d, t[nb == 1]) if (nb == 1).any() else np.zeros(1)
    m3 = float(np.mean([e[1] for e in ev if e[1] > 0]) if ev else 0)
    return np.array([
        on.mean(), (on > 0.3).mean(), sample(b, mid).mean(), (sample(b, mid) > 0.3).mean(),
        np.maximum(sample(b, th1), sample(b, th2)).mean(), down_on.mean(), score,
        lt, lt * lt, float(max(nb) == 3) if len(nb) else 0.0, 1.0,
    ])


def candidates(act, base_ev=None):
    """[(bpm кандидата, события, признаки)]"""
    if base_ev is None:
        base_ev = BarHMM().decode(act[:, 0], act[:, 1])
    T = grid_bpm(base_ev)
    out = []
    for r in RATIOS:
        c = T * r
        if not (BPM_LO <= c <= BPM_HI):
            continue
        h = hmm_for(c)
        ev = h.decode(act[:, 0], act[:, 1])
        f = features(act, ev, c, h.last_score)
        if f is not None:
            out.append((grid_bpm(ev), ev, f))
    return out


def load_selector():
    if WEIGHTS.exists():
        w = json.loads(WEIGHTS.read_text())
        return np.array(w["w"]), np.array(w["mu"]), np.array(w["sd"])
    return None


def choose(act, selector=None):
    """→ события лучшего кандидата (или обычной HMM, если селектора нет)."""
    selector = selector if selector is not None else load_selector()
    base = BarHMM().decode(act[:, 0], act[:, 1])
    if selector is None:
        return base
    w, mu, sd = selector
    cand = candidates(act, base)
    if not cand:
        return base
    F = (np.stack([c[2] for c in cand]) - mu) / sd
    return cand[int(np.argmax(F @ w))][1]


# ---------- обучение ----------

def match(bpm, ref, tol=0.04):
    return abs(bpm / ref - 1) < tol


def _one(args):
    acts_dir, r = args
    cache = Path(acts_dir) / "cand" / f"{r['sid']}.npz"
    if cache.exists():
        z = np.load(cache)
        bpm, F, base_bpm = z["bpm"], z["F"], float(z["base_bpm"])
    else:
        act = np.load(Path(acts_dir) / f"{r['sid']}.npz")["act"].astype(np.float32)
        base = BarHMM().decode(act[:, 0], act[:, 1])
        cand = candidates(act, base)
        if not cand:
            return None
        bpm, F, base_bpm = np.array([c[0] for c in cand]), np.stack([c[2] for c in cand]), grid_bpm(base)
        np.savez(cache, bpm=bpm, F=F, base_bpm=base_bpm)
    return {"sid": r["sid"], "split": r["split"], "ref": r["bpm"], "bpm": list(bpm), "F": F,
            "y": np.array([match(b, r["bpm"]) for b in bpm]), "base_ok": match(base_bpm, r["bpm"])}


def build(acts_dir, splits, procs=4):
    from multiprocessing import Pool
    idx = [r for r in json.loads((Path(acts_dir) / "index.json").read_text()) if r["split"] in splits]
    (Path(acts_dir) / "cand").mkdir(exist_ok=True)
    with Pool(procs) as pool:
        res = pool.map(_one, [(acts_dir, r) for r in idx], chunksize=4)
    return [g for g in res if g is not None]


def fit(groups, mu, sd, l2=1e-2, steps=3000, lr=0.1):
    use = [g for g in groups if g["y"].any()]
    w = np.zeros(groups[0]["F"].shape[1])
    for _ in range(steps):
        grad = l2 * w
        for g in use:
            X = (g["F"] - mu) / sd
            s = X @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            tgt = g["y"] / g["y"].sum()
            grad += X.T @ (p - tgt) / len(use)
        w -= lr * grad
    return w


def accuracy(groups, w, mu, sd):
    ok = 0
    for g in groups:
        k = int(np.argmax(((g["F"] - mu) / sd) @ w))
        ok += bool(g["y"][k])
    return ok / max(len(groups), 1)


def main():
    acts = sys.argv[2]
    tr = build(acts, {"train", "train_ccm"})
    te = build(acts, {"test"})
    allF = np.concatenate([g["F"] for g in tr])
    mu, sd = allF.mean(0), allF.std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0  # смещение
    w = fit(tr, mu, sd)
    for name, gs in (("обучение", tr), ("тест 30", te)):
        print(f"{name}: {len(gs)} треков | обычная HMM {np.mean([g['base_ok'] for g in gs]):.2f} | "
              f"достижимо {np.mean([g['y'].any() for g in gs]):.2f} | с выбором уровня {accuracy(gs, w, mu, sd):.2f}")
    for g in te:
        k = int(np.argmax(((g["F"] - mu) / sd) @ w))
        print(f"  {g['sid'][:20]:20s} автор {g['ref']:5.0f}  выбрали {g['bpm'][k]:6.1f}  {'ok' if g['y'][k] else '--'}")
    WEIGHTS.write_text(json.dumps({"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "ratios": RATIOS}))
    print("сохранено", WEIGHTS)


if __name__ == "__main__":
    main()
