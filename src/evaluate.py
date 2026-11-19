"""Оценка в духе Qildo: mir_eval.beat по долям и сильным долям + свои метрики.

Использование:
    python src/evaluate.py refs/ preds/        # refs/<id>.json и preds/<id>.json
Формат обоих файлов: [{"time": 0.512, "beat": 1}, ...] (beat == 1 — сильная доля).
"""
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

WIN = 0.07  # окно Qildo ±70 мс


def load(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    t = np.array([x["time"] for x in items], dtype=float)
    b = np.array([x["beat"] for x in items], dtype=int)
    o = np.argsort(t)
    return t[o], b[o]


def missed_at_start(ref, est, win=WIN):
    """Сколько опорных долей до первой совпавшей доли оценки (пропуск начала сетки)."""
    if len(est) == 0:
        return len(ref)
    for i, r in enumerate(ref):
        if np.min(np.abs(est - r)) <= win:
            return i
    return len(ref)


def tempo_octave_errors(ref, est, win_beats=8):
    """Доля окон по 8 опорных долей, где локальный темп оценки ≈ ×2 или ×0.5 от опорного."""
    if len(ref) < win_beats + 1 or len(est) < 3:
        return 0.0, 0
    bad = 0
    total = 0
    for s in range(0, len(ref) - win_beats, win_beats // 2):
        a, z = ref[s], ref[s + win_beats]
        ref_ibi = np.median(np.diff(ref[s : s + win_beats + 1]))
        e = est[(est >= a) & (est <= z)]
        if len(e) < 2:
            continue
        est_ibi = np.median(np.diff(e))
        r = est_ibi / ref_ibi
        total += 1
        if abs(np.log2(r) - 1) < 0.2 or abs(np.log2(r) + 1) < 0.2:
            bad += 1
    return (bad / total if total else 0.0), bad


def downbeat_shifts(ref_t, ref_b, est_t, est_b, win=WIN):
    """Сдвиги сильной доли: совпавшая доля, у которой номер в такте ≠ опорному.

    Возвращает (доля долей с неверным номером, число смен фазы такта по ходу песни).
    """
    wrong = 0
    matched = 0
    phases = []
    for t, b in zip(est_t, est_b):
        j = np.argmin(np.abs(ref_t - t))
        if abs(ref_t[j] - t) > win:
            continue
        matched += 1
        if b != ref_b[j]:
            wrong += 1
        phases.append((b - ref_b[j]) % 12)
    changes = int(np.sum(np.diff(phases) != 0)) if len(phases) > 1 else 0
    return (wrong / matched if matched else 0.0), changes


def score_one(ref_path, est_path):
    rt, rb = load(ref_path)
    et, eb = load(est_path)
    # mir_eval по умолчанию выкидывает доли первых 5 секунд; Qildo важен старт — не выкидываем
    r = mir_eval.beat.trim_beats(rt, min_beat_time=0.0)
    e = mir_eval.beat.trim_beats(et, min_beat_time=0.0)
    rd, ed = rt[rb == 1], et[eb == 1]
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(r, e)
    oct_frac, _ = tempo_octave_errors(rt, et)
    db_wrong, db_changes = downbeat_shifts(rt, rb, et, eb)
    return {
        "beat_F": mir_eval.beat.f_measure(r, e, f_measure_threshold=WIN),
        "downbeat_F": mir_eval.beat.f_measure(rd, ed, f_measure_threshold=WIN),
        "CMLt": cmlt,
        "AMLt": amlt,
        "missed_start": missed_at_start(rt, et),
        "tempo_x2_frac": oct_frac,
        "downbeat_wrong_frac": db_wrong,
        "downbeat_phase_changes": db_changes,
    }


def score_dir(ref_dir, est_dir, verbose=False):
    rows = {}
    for rp in sorted(Path(ref_dir).glob("*.json")):
        ep = Path(est_dir) / rp.name
        if not ep.exists():
            continue
        rows[rp.stem] = score_one(rp, ep)
        if verbose:
            print(rp.stem, {k: round(v, 3) for k, v in rows[rp.stem].items()})
    if not rows:
        raise SystemExit("нет пар ref/pred")
    keys = next(iter(rows.values())).keys()
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
    mean["n"] = len(rows)
    mean["songs_with_x2"] = int(sum(r["tempo_x2_frac"] > 0 for r in rows.values()))
    mean["songs_missed_start"] = int(sum(r["missed_start"] > 0 for r in rows.values()))
    return mean, rows


if __name__ == "__main__":
    mean, _ = score_dir(sys.argv[1], sys.argv[2], verbose="-v" in sys.argv)
    print(json.dumps({k: round(v, 4) for k, v in mean.items()}, ensure_ascii=False, indent=1))
