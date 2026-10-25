#!/usr/bin/env python3
"""Task 125: factorizer (king-independent P features) at 32M.

HalfKP gives every (king square, piece-square) pair its own row, so "a silver on 5e"
is learnt 81 times, each copy only from positions with the king on that square. A
factorizer adds a king-independent table V[p] during training (effective weight
W[k, p] + V[p]); V learns from every position with p and W only the king-specific
remainder. At export V is folded into every king's row (`trainer/nnue.fold_ft_factor`),
so the deployed network, its speed and the YaneuraOu file are unchanged. Stockfish:
removing the factorizers from the start cost about -24 (16B positions; the gain is in
the early part of training); tanuki-: HalfKP^ at 256x2-32-32 / 8B positions -44.4
(docs/research_survey_2026-10-07.md §3.4). Our 64M students are saturated on the
training-recipe axis (Task 124), which points at the data; sharing across king squares
is the cheap way to get more out of the same data.

Checks before pre-registration: the folded non-factorized model reproduces the
factorized forward exactly (max |diff| 0.0 with and without fake quant, tied); a
factorized CPU run exports and passes the 0cp gate (32/32 exact, GREEN x2); the
non-factorized path is unchanged (same train loss as before the change).

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: --factorize. Everything else = the best 32M student
    `task122_student_32m_lr6e-5` (control; standard-frame games reused):
    `caches/shared_32m`, search labels `work/task104/rescore_search`, Task 106 mask,
    098 NNUE_ARGS with lr 6e-5, --augment-mirror, 12 epochs.
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, 100,000 nodes, seeds 20272001-08, 800 games per anchor), pooled.
  * decision: CI low > 0 -> FACTORIZER_HELPS (next: 64M) / CI high < 0 ->
    FACTORIZER_HURTS / otherwise FACTORIZER_NEUTRAL.
  * gates: 0cp GREEN, no illegal move / engine failure.
  * descriptive: val loss and best epoch vs control, epoch time.

Interruptible (scripts/job.sh pause|resume task125).
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

TAG = "task125"
CONTROL = "task122_student_32m_lr6e-5"
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_factor"
    eval_name = f"{TAG}_student_factor"
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
    print(f"T125_PREFLIGHT_OK head={head} control={CONTROL} lr={LR}", flush=True)
    failures = []

    if not (model_dir / "history.json").exists():
        print("=== T125 STAGE train (32M, Task 106 recipe, Task 106 mask, lr 6e-5, factorizer) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, "--factorize", "--augment-mirror",
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
    print(f"T125_VAL best_val_loss={hist['best_val_loss']:.1f} control={ctrl_hist['best_val_loss']:.1f}", flush=True)
    print(f"T125_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T125 STAGE standard frame (control, then candidate) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    pooled, per, bad_o = std_frame.delta(cand, ctrl, "(factorizer - lr6e-5)")
    failures += bad_c + bad_k + bad_o
    for name, d in per.items():
        print(f"T125_FRAME {name} delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T125_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "FACTORIZER_HELPS" if lo > 0 else "FACTORIZER_HURTS" if hi < 0
               else "FACTORIZER_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 125, "git": {"head": head, "pushed": False}, "lr": LR, "factorize": True,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T125_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
