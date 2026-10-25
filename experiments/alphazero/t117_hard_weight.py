#!/usr/bin/env python3
"""Task 117: weight the hard positions up instead of dropping them (32M screening).

Task 115 dropped the rows where the two teachers disagree (|gikou@d8 - search| >= 600
cp, 7.8% of the Task 106-kept rows) and lost 35 Elo; Task 116 dropped the same number
of random rows and was neutral (-11.1 [-35.0, +12.9]), and Task 115 was worse than
random (-24.2 [-47.6, -0.7]). The disagreement rows are therefore worth learning. Here:
keep every Task 106 row and weight each row's loss by its teacher gap instead.

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: per-row loss weight (train_nnue --row-weight, Task 117).
    w = 1 + min(gap, 1200) / 600 with gap = |clip(gikou, +-2000) - clip(search, +-2000)|
    (`work/task115/mask/train_teacher_gap.npy`), normalised to mean 1 over the trained
    rows (range 0.74-2.21, median 0.91; rows with gap >= 600 carry 13.7% of the loss
    weight instead of 7.8%). Mask = Task 106 mask, labels = search labels, recipe
    098 NNUE_ARGS + --augment-mirror, 12 epochs, `caches/shared_32m`.
    Control = `task106_student_filter` (standard-frame games reused).
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, both evals at 100,000 nodes, seeds 20272001-08, 800 games per anchor), pooled.
  * decision: CI low > 0 -> HARD_WEIGHT_HELPS (next: confirm at 64M on fresh seeds) /
    CI high < 0 -> HARD_WEIGHT_HURTS / otherwise HARD_WEIGHT_NEUTRAL.
  * gates: 0cp GREEN, no illegal move / engine failure, weight file finite and
    aligned (31,950,000 rows).
  * descriptive: val loss (unweighted, same val rows) vs control; per-anchor deltas.

Interruptible (scripts/job.sh pause|resume task117).
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

TAG = "task117"
CONTROL = "task106_student_filter"
GAP = "work/task115/mask/train_teacher_gap.npy"
GAP_SCALE, GAP_CAP = 600, 1200
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def build_weight(data, work):
    out = work / "weight"
    if (out / "train_weight.npy").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    gap = np.load(data / GAP)
    base = np.load(data / BASE_MASK)
    if not gap.shape[0] == base.shape[0] == LIMIT_TRAIN:
        raise RuntimeError(f"rows differ: gap {gap.shape} mask {base.shape}")
    w = (1.0 + np.minimum(gap, GAP_CAP).astype(np.float32) / GAP_SCALE).astype(np.float32)
    w /= float(w[base].mean())
    np.save(out / "train_weight.npy", w)
    tw = w[base]
    stats = {"rows": int(w.size), "trained_rows": int(base.sum()), "finite": bool(np.isfinite(w).all()),
             "min": float(tw.min()), "median": float(np.median(tw)), "max": float(tw.max()),
             "hard_row_share": float((gap[base] >= GAP_SCALE).mean()),
             "hard_weight_share": float(tw[gap[base] >= GAP_SCALE].sum() / tw.sum())}
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("T117_WEIGHT " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_hardweight"
    eval_name = f"{TAG}_student_hardweight"
    eval_dir = data / "eval_dirs" / eval_name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK, data / GAP,
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
    print(f"T117_PREFLIGHT_OK head={head} control={CONTROL} gap_scale={GAP_SCALE} gap_cap={GAP_CAP}", flush=True)
    failures = []

    print("=== T117 STAGE weight (teacher gap on Task 106-kept rows) ===", flush=True)
    mask = build_weight(data, work)
    if not mask["finite"] or mask["rows"] != LIMIT_TRAIN:
        failures.append(f"weight file bad: {mask}")

    if not (model_dir / "history.json").exists():
        print("=== T117 STAGE train (32M, Task 106 recipe, Task 106 mask, teacher-gap weights) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS, "--augment-mirror",
                  "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN,
                  "--row-mask", data / BASE_MASK,
                  "--row-weight", work / "weight/train_weight.npy", "--checkpoint-minutes", "10", "--resume", *stop_args],
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
    print(f"T117_VAL best_val_loss={hist['best_val_loss']:.1f} control={ctrl_hist['best_val_loss']:.1f}", flush=True)
    print(f"T117_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T117 STAGE standard frame (control, then candidate) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    pooled, per, bad_o = std_frame.delta(cand, ctrl, "(hardweight - task106)")
    failures += bad_c + bad_k + bad_o
    for name, d in per.items():
        print(f"T117_FRAME {name} delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T117_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "HARD_WEIGHT_HELPS" if lo > 0 else "HARD_WEIGHT_HURTS" if hi < 0
               else "HARD_WEIGHT_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 117, "git": {"head": head, "pushed": False}, "weight": mask,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T117_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
