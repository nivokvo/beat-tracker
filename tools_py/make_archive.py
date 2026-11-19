"""Собрать архив решения для отправки: код, веса, README, отчёт, данные/лицензии.

python tools_py/make_archive.py dist/qildo_task4.zip [--weights weights_cand/r9_s8000.pt] [--report REPORT_v2.md]
Данные (аудио, мел) в архив не входят — их скачивают скрипты из data_prep/ (см. README и DATA.md).
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["beats.py", "README.md", "REPORT.md", "DATA.md", "requirements.txt", "requirements-train.txt",
         "weights/beat_model.pt", "weights/tempo_select.json"]
DIRS = {"src": "*.py", "data_prep": "*.py", "baselines": "*.py"}
TOOLS = ["click_overlay.py", "qildo_compare.py", "qildo_compare2.py", "jitter_fma.py", "calib_offset.py", "snapshot_watch.py", "build_demo.py", "make_archive.py"]


def main(out, weights=None, report=None):
    """weights/report — подставить другие файлы под именами weights/beat_model.pt и REPORT.md."""
    subst = {}
    if weights:
        subst["weights/beat_model.pt"] = ROOT / weights
    if report:
        subst["REPORT.md"] = ROOT / report
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [f for f in FILES if (ROOT / f).exists()]
    for d, pat in DIRS.items():
        names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted((ROOT / d).glob(pat))]
    names += [f"tools_py/{t}" for t in TOOLS if (ROOT / "tools_py" / t).exists()]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(subst.get(n, ROOT / n), f"qildo_task4/{n}")
    print(out, round(out.stat().st_size / 1e6, 1), "MB,", len(names), "файлов")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--weights")
    ap.add_argument("--report")
    a = ap.parse_args()
    main(a.out, a.weights, a.report)
