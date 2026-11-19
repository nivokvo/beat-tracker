"""Сохранять снимок модели после каждой проверки во время обучения: runs/<run>/s<step>.pt (только веса).

python tools_py/snapshot_watch.py runs/r4
Читает log.jsonl; когда появляется новый шаг, берёт last.pt и сохраняет веса без оптимизатора.
"""
import json
import sys
import time
from pathlib import Path

import torch

run = Path(sys.argv[1])
done = {p.stem for p in run.glob("s*.pt")}
while True:
    try:
        lines = (run / "log.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for l in lines:
        step = json.loads(l)["step"]
        name = f"s{step}"
        if name in done:
            continue
        time.sleep(20)  # дать дописать last.pt
        try:
            ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            if ck.get("step") != step:  # прошлые шаги уже не восстановить, будущий ещё не записан
                if ck.get("step", 0) > step:
                    done.add(name)
                continue
            torch.save({k: v for k, v in ck.items() if k in ("model", "cfg", "step")}, run / f"{name}.pt")
            done.add(name)
            print("saved", name, flush=True)
        except Exception as e:  # файл пишется — попробуем в следующий круг
            print("retry", name, type(e).__name__, flush=True)
    if (run / "DONE").exists():
        break
    time.sleep(30)
