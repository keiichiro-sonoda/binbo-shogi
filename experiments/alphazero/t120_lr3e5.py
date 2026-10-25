#!/usr/bin/env python3
"""Task 120: learning rate 1e-5 -> 3e-5 (32M screening).

Signs that the students are update-limited: best epochs sit at the end (Task 109 best
22/24, Task 106 12/12), doubling the epochs gave +43 (Task 109), and the feature
transformer weights are tiny (Task 109: int16 |w| 99.9th percentile 9 of 127, max 59;
Task 106 at 12 epochs: 5 / 22). Adam lr 1e-5 is small next to other NNUE trainers
(docs/research_survey_2026-10-07.md §3.3). Task 061's sweep (2M-scale data, 18 epochs)
had lr 3e-5 at +240.8 vs 1e-5 at +188.5 (1e-5 was adopted); the data is now 10x larger.

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: --lr 3e-5 (instead of 1e-5). Everything else = Task 106:
    `caches/shared_32m`, search labels `work/task104/rescore_search`, Task 106 mask,
    098 NNUE_ARGS + --augment-mirror, 12 epochs. Control = `task106_student_filter`
    (standard-frame games reused).
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, 100,000 nodes, seeds 20272001-08, 800 games per anchor), pooled.
  * decision: CI low > 0 -> LR3E5_HELPS (next: 64M at 3e-5) / CI high < 0 ->
    LR3E5_HURTS / otherwise LR3E5_NEUTRAL.
  * gates: 0cp GREEN, no illegal move / engine failure.
  * descriptive: val loss and best epoch vs control; feature-transformer weight size.

Interruptible (scripts/job.sh pause|resume task120).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402

TAG = "task120"
CONTROL = "task106_student_filter"
LR = "3e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_lr3e5"
    eval_name = f"{TAG}_student_lr3e5"
    eval_dir = data / "eval_dirs" / eval_name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK,
                 data / "eval_dirs" / CONTROL / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T120_PREFLIGHT_OK head={head} control={CONTROL} lr={LR}", flush=True)
    failures = []

    if not (model_dir / "history.json").exists():
        print("=== T120 STAGE train (32M, Task 106 recipe, Task 106 mask, lr 3e-5) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, "--augment-mirror",
                  "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN,
                  "--row-mask", data / BASE_MASK, "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                  "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                  "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{TAG}_yane.log", cwd=repo, env=env)
    gate = logs / f"{TAG}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    ctrl_hist = json.loads((data / "models" / CONTROL / "history.json").read_text(encoding="utf-8"))
    print(f"T120_VAL best_val_loss={hist['best_val_loss']:.1f} control={ctrl_hist['best_val_loss']:.1f}", flush=True)
    print(f"T120_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T120 STAGE standard frame (control, then candidate) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    pooled, per, bad_o = std_frame.delta(cand, ctrl, "(lr3e-5 - task106)")
    failures += bad_c + bad_k + bad_o
    for name, d in per.items():
        print(f"T120_FRAME {name} delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T120_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "LR3E5_HELPS" if lo > 0 else "LR3E5_HURTS" if hi < 0
               else "LR3E5_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 120, "git": {"head": head, "pushed": False}, "lr": LR,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T120_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
