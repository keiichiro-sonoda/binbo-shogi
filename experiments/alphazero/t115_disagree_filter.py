#!/usr/bin/env python3
"""Task 115: skip positions where the two teachers disagree (32M screening).

Every position carries two teacher values: the gikou@d8 eval of the hcpe (the
champion's teacher, `caches/shared_32m/train_evals.npy`) and the WS-T + gumbel MCTS
search value the current students learn (`work/task104/rescore_search`). They agree
closely overall (corr 0.952, median |d| 239 cp), but on a tail they differ by a game's
worth. A large gap means at least one teacher is wrong there or the position is not
statically readable; training on that label can be noise. Here: on the rows Task 106
keeps, clip both to +-2000 (train_nnue's --clamp-cp) and skip the row when
|gikou - search| >= 600 cp. Calibration on every 20th kept row (labels compared to
each other only, no model / game result): 7.7% of kept rows reach 600.

Task 114 (qsearch-widened filter, -30.5 on the anchors) is the caution: dropping hard
positions can hurt. The threshold is therefore set to drop well under 10%.

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: the row mask. Screening scale = Task 106 exactly: `caches/shared_32m`,
    search labels `work/task104/rescore_search`, recipe 098 NNUE_ARGS + --augment-mirror,
    12 epochs. Control = `task106_student_filter` (its standard-frame games from Task 114
    are reused). Candidate mask = Task 106 mask AND |clip(gikou) - clip(search)| < 600.
  * judged per docs/evaluation_protocol.md: standard anchor frame (tanuki- @10,000,
    AobaNNUE @5,000, both evals at 100,000 nodes, seeds 20272001-08, 800 games per
    anchor per eval), pooled (candidate - control) over the 1,600 shared openings.
    No direct match.
  * gates: 0cp GREEN, no illegal move / engine failure, extra skip fraction within
    [4%, 12%] of the Task 106-kept rows.
  * decision: CI low > 0 -> DISAGREE_FILTER_HELPS (next: the same mask at 64M) /
    CI high < 0 -> DISAGREE_FILTER_HURTS / otherwise DISAGREE_FILTER_NEUTRAL.
  * descriptive: val loss vs control (val rows are not filtered, so it is comparable).

Interruptible (scripts/job.sh pause|resume task115).
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

TAG = "task115"
CONTROL = "task106_student_filter"
GAP_MIN = 600
CLAMP_CP = 2000
SKIP_RANGE = (0.04, 0.12)
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def build_mask(data, work):
    out = work / "mask"
    if (out / "train_keep.npy").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(data / BASE_MASK)
    g = np.load(data / CACHE / "train_evals.npy", mmap_mode="r")
    s = np.load(data / LABELS / "train_dlshogi_evals.npy", mmap_mode="r")
    if not g.shape[0] == s.shape[0] == base.shape[0] == LIMIT_TRAIN:
        raise RuntimeError(f"row counts differ: gikou {g.shape} search {s.shape} mask {base.shape}")
    gap = np.empty(base.shape[0], dtype=np.int16)
    for lo in range(0, base.shape[0], 1_000_000):
        hi = lo + 1_000_000
        gc = np.clip(np.asarray(g[lo:hi], dtype=np.int32), -CLAMP_CP, CLAMP_CP)
        sc = np.clip(np.asarray(s[lo:hi], dtype=np.int32), -CLAMP_CP, CLAMP_CP)
        gap[lo:hi] = np.abs(gc - sc)
    keep = base & (gap < GAP_MIN)
    np.save(out / "train_keep.npy", keep)
    np.save(out / "train_teacher_gap.npy", gap)
    kept_base = int(base.sum())
    stats = {"rows": int(base.size), "base_keep": kept_base, "keep": int(keep.sum()),
             "extra_skip_frac_of_base": float(1 - keep.sum() / kept_base), "keep_frac": float(keep.mean()),
             "gap_median_of_base": float(np.median(gap[base]))}
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("T115_MASK " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_disagree"
    eval_name = f"{TAG}_student_disagree"
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
    print(f"T115_PREFLIGHT_OK head={head} control={CONTROL} gap_min={GAP_MIN}", flush=True)
    failures = []

    print("=== T115 STAGE mask (teacher gap on Task 106-kept rows) ===", flush=True)
    mask = build_mask(data, work)
    if not SKIP_RANGE[0] <= mask["extra_skip_frac_of_base"] <= SKIP_RANGE[1]:
        failures.append(f"extra skip {mask['extra_skip_frac_of_base']:.3f} outside {SKIP_RANGE}")

    if not (model_dir / "history.json").exists():
        print("=== T115 STAGE train (32M, Task 106 recipe, teacher-gap mask) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS, "--augment-mirror",
                  "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN,
                  "--row-mask", work / "mask/train_keep.npy", "--checkpoint-minutes", "10", "--resume", *stop_args],
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
    print(f"T115_VAL best_val_loss={hist['best_val_loss']:.1f} control={ctrl_hist['best_val_loss']:.1f}", flush=True)
    print(f"T115_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T115 STAGE standard frame (control, then candidate) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    pooled, per, bad_o = std_frame.delta(cand, ctrl, "(disagree - task106)")
    failures += bad_c + bad_k + bad_o
    for name, d in per.items():
        print(f"T115_FRAME {name} delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T115_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "DISAGREE_FILTER_HELPS" if lo > 0 else "DISAGREE_FILTER_HURTS" if hi < 0
               else "DISAGREE_FILTER_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 115, "git": {"head": head, "pushed": False}, "mask": mask,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T115_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
