"""Эксперимент: признаки и модель для выбора уровня темпа. Отбор — 5-кратная кросс-валидация по трекам
на обучающих данных (тест 30 не трогаем до финала).

python src/tempo_exp.py runs/acts_r3
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tempo_select import FPS, match  # noqa: E402


def acf_feats(act, bpm):
    """Периодичность активаций на периоде доли и такта кандидата."""
    out = []
    P = 60 * FPS / bpm
    for ch in (0, 1):
        x = act[:, ch].astype(np.float64)
        x = x - x.mean()
        n = len(x)
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac /= ac[0] + 1e-9

        def at(lag):
            lag = int(round(lag))
            return ac[max(1, lag - 1): lag + 2].max() if 1 <= lag < n - 2 else 0.0

        if ch == 0:
            out += [at(P), at(2 * P), at(P / 2), at(P * 1.5), at(P / 3 * 2)]
        else:
            out += [at(3 * P), at(4 * P), at(2 * P), at(P)]
    return out


def load(acts_dir):
    acts_dir = Path(acts_dir)
    idx = json.loads((acts_dir / "index.json").read_text())
    G = []
    for r in idx:
        z = np.load(acts_dir / "cand" / f"{r['sid']}.npz")
        act = np.load(acts_dir / f"{r['sid']}.npz")["act"].astype(np.float32)
        bpm, F = z["bpm"], z["F"]
        extra = np.array([acf_feats(act, b) for b in bpm])
        X = np.concatenate([F[:, :-1], extra], 1)  # без столбца-смещения
        rel = X - X.max(0, keepdims=True)  # относительно лучшего кандидата трека
        G.append({"sid": r["sid"], "split": r["split"], "X": np.concatenate([X, rel], 1),
                  "y": np.array([match(b, r["bpm"]) for b in bpm]), "base_ok": match(float(z["base_bpm"]), r["bpm"])})
    return G


def cv(G, make, k=5, seed=0):
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, k, len(G))
    hits = np.zeros(len(G), bool)
    for f in range(k):
        tr = [g for g, ff in zip(G, fold) if ff != f]
        X = np.concatenate([g["X"] for g in tr])
        y = np.concatenate([g["y"] for g in tr])
        m = make().fit(X, y)
        for i, (g, ff) in enumerate(zip(G, fold)):
            if ff == f:
                hits[i] = g["y"][int(np.argmax(m.predict_proba(g["X"])[:, 1]))]
    return hits


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    G = load(sys.argv[1])
    tr = [g for g in G if g["split"] != "test"]
    kml = np.array([g["split"] == "train" for g in tr])
    print(f"треков {len(tr)} (kml {kml.sum()}), обычная HMM: всё {np.mean([g['base_ok'] for g in tr]):.3f}, "
          f"kml {np.mean([g['base_ok'] for g in tr if g['split'] == 'train']):.3f}")
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "gbt_small": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05),
        "gbt": lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05),
    }
    for name, mk in models.items():
        h = np.mean([cv(tr, mk, seed=s) for s in range(3)], 0)
        print(f"{name:10s} CV: всё {h.mean():.3f}, kml {h[kml].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
