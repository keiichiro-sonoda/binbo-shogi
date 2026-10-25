#!/usr/bin/env python3
"""Task 131: win-rate-space power loss (|sigmoid(pred) - sigmoid(label/600)|^2.5), 32M.

docs/research_survey_2026-10-12.md proposal 2. Stockfish's nnue-pytorch trains on
|p - q|^k in win-rate space (k about 2.5; Sopel: k=2.6 +10 over MSE) and the shogi
trainers of tanuki- follow it. Our loss is Huber (delta 600) on cp clamped to +-2000.
Caveat known before the run: Task 111's descriptive arm A (win-rate MSE, a 4-epoch
continuation of the then adopted eval) was -22.6 [-62.0, +16.7] against the cp loss, so
win-rate space alone may cost. Only the Stockfish-style loss is measured here (the user
chose one arm); a loss would close the loss-function direction.

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable vs the control `task128_student_qsrelc` (Task 128 arm C; standard-frame
    games on the Task 127 yardstick reused): the loss. `--objective wdl --wdl-lambda 1.0
    --wdl-power 2.5` (target sigmoid(label / 600), prediction sigmoid(logit); the
    exported cp is unchanged: logit * 600). Same cache `caches/shared_32m_qs119`, labels
    `work/task128/labels`, no row mask, 098 NNUE_ARGS with lr 6e-5, --factorize,
    --augment-mirror, 12 epochs. Scale 600 is kept (no scaling change).
  * yardstick = Task 127 (tanuki- @20,000 / AobaNNUE @10,000, 100,000 nodes, seeds
    20272001-08, 800 games per anchor), pooled.
  * decision (power - control): CI low > 0 -> POWER_LOSS_HELPS (next: 64M vs the
    adopted eval) / CI high < 0 -> POWER_LOSS_HURTS / otherwise POWER_LOSS_NEUTRAL.
  * gates: 0cp GREEN; no illegal move / engine failure; openings identical.
  * descriptive: best epoch, val loss (win-rate units, not comparable with the control).

Interruptible (scripts/job.sh pause|resume task131).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402

TAG = "task131"
CONTROL = "task128_student_qsrelc"
CACHE, LABELS = "caches/shared_32m_qs119", "work/task128/labels"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
LOSS_ARGS = ["--objective", "wdl", "--wdl-lambda", "1.0", "--wdl-power", "2.5"]


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    name = f"{TAG}_student_power25"
    model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / "eval_dirs" / CONTROL / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T131_PREFLIGHT_OK head={head} control={CONTROL} anchors={std_frame.ANCHORS} loss={LOSS_ARGS}", flush=True)
    failures = []

    if not (model_dir / "history.json").exists():
        print("=== T131 STAGE train (32M, win-rate power loss 2.5) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, *LOSS_ARGS,
                  "--factorize", "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN, "--checkpoint-minutes", "10",
                  "--resume", *stop_args],
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
    print(f"T131_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.5f}", flush=True)

    print("=== T131 STAGE standard frame ===", flush=True)
    ctrl, bad = std_frame.measure(CONTROL, env)
    failures += bad
    cand, bad = std_frame.measure(name, env)
    failures += bad
    pooled, per, bad = std_frame.delta(cand, ctrl, "(power2.5 - task128 C)")
    failures += bad
    print(f"T131_FRAME pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
          + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "POWER_LOSS_HELPS" if lo > 0 else "POWER_LOSS_HURTS" if hi < 0
               else "POWER_LOSS_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 131, "git": {"head": head, "pushed": False}, "loss": LOSS_ARGS,
                               "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T131_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
