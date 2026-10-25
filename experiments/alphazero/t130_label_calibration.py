#!/usr/bin/env python3
"""Task 130: calibrate the teacher's score distribution (quantile map onto gikou@d8), 32M.

docs/research_survey_2026-10-12.md proposal 1. tanuki- (2026-09) replaced each teacher
score by the mean score of a reference teacher at the same rank (rank-preserving,
tie-preserving quantile map) and gained +44.2 [+35.8, +52.6] over the same training
order uncalibrated (6,006 games). Here the reference is gikou@d8, the lineage of our own
teacher, so nothing outside the project's teacher policy enters: the ORDER of positions
is WS-T's, only the size of the scores changes.

Measured before pre-registration (2M sampled rows of the Task 128 labels, rows that were
not replaced, i.e. the same positions for both): WS-T quantiles 5/25/50/75/95% =
-2596 / -405 / -103 / +116 / +1329 cp, gikou@d8 = -3149 / -363 / 0 / +302 / +2003.
WS-T scores the side to move about 100 cp lower at the median and compresses large
values.

PRE-REGISTERED (committed before the run; no result is seen):
  * map: fitted on every train row that Task 119 did NOT replace (P = Q, so both scores
    belong to the same position): sort the WS-T labels and the gikou@d8 evals (clipped
    to +-5000, WS-T's clamp) separately; each distinct WS-T value v gets the mean of the
    gikou values at the ranks v occupies (ties kept together, so the map is monotone
    non-decreasing); values that do not occur in the fit set are linearly interpolated.
    The map is applied to every train and val label (Task 128's labels, relabelled
    leaves included).
  * one variable vs the control `task128_student_qsrelc` (Task 128 arm C; standard-frame
    games on the Task 127 yardstick reused): labels only. Same cache
    `caches/shared_32m_qs119`, no row mask, 098 NNUE_ARGS with lr 6e-5, --factorize,
    --augment-mirror, 12 epochs.
  * yardstick = Task 127 (tanuki- @20,000 / AobaNNUE @10,000, 100,000 nodes, seeds
    20272001-08, 800 games per anchor), pooled.
  * decision (calibrated - control): CI low > 0 -> CALIB_HELPS (next: the same at 64M vs
    the adopted eval) / CI high < 0 -> CALIB_HURTS / otherwise CALIB_NEUTRAL.
  * gates: map monotone, finite, |mapped| <= 5000; 0cp GREEN; no illegal move / engine
    failure; openings identical.
  * descriptive: map at -2000/-1000/-500/-200/0/+200/+500/+1000/+2000, label quantiles
    before / after, val loss (not comparable with the control: the labels differ).

Interruptible (scripts/job.sh pause|resume task130).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402

TAG = "task130"
CONTROL = "task128_student_qsrelc"
CACHE, LABELS, CHANGED = "caches/shared_32m_qs119", "work/task128/labels", "work/task119/changed.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
CLIP = 5000
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
PROBES = (-2000, -1000, -500, -200, 0, 200, 500, 1000, 2000)
QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def fit_map(labels, gikou):
    """Rank-preserving, tie-preserving quantile map: distinct label value -> mean reference value."""
    ls = np.sort(labels.astype(np.int64))
    gs = np.sort(np.clip(gikou.astype(np.int64), -CLIP, CLIP)).astype(np.float64)
    values, starts, counts = np.unique(ls, return_index=True, return_counts=True)
    csum = np.concatenate([[0.0], np.cumsum(gs)])
    means = (csum[starts + counts] - csum[starts]) / counts
    return values.astype(np.float64), means


def apply_map(x, values, means):
    return np.rint(np.interp(x.astype(np.float64), values, means)).astype(np.int16)


def build(data, work):
    out = work / "labels"
    if (out / "meta.json").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    train = np.load(data / LABELS / "train_dlshogi_evals.npy")
    val = np.load(data / LABELS / "val_dlshogi_evals.npy")
    gikou = np.load(data / CACHE / "train_evals.npy")
    keep = ~np.load(data / CHANGED)
    values, means = fit_map(train[keep], gikou[keep])
    if not (np.all(np.diff(means) >= 0) and np.isfinite(means).all() and np.abs(means).max() <= CLIP):
        raise AssertionError("map is not monotone / finite / bounded")
    new_train, new_val = apply_map(train, values, means), apply_map(val, values, means)
    stats = {"fit_rows": int(keep.sum()), "distinct_values": int(len(values)),
             "map": {str(p): float(np.interp(p, values, means)) for p in PROBES},
             "quantiles": {"q": list(QS),
                           "before": np.quantile(train, QS).tolist(), "after": np.quantile(new_train, QS).tolist(),
                           "gikou_fit_rows": np.quantile(np.clip(gikou[keep], -CLIP, CLIP), QS).tolist()}}
    tmp = work / "labels.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "train_dlshogi_evals.npy", new_train)
    np.save(tmp / "val_dlshogi_evals.npy", new_val)
    np.savez(tmp / "map.npz", values=values, means=means)
    (tmp / "meta.json").write_text(json.dumps({"source": LABELS, "reference": f"{CACHE}/train_evals (gikou@d8)",
                                               **stats}, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return stats


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    name = f"{TAG}_student_calib"
    model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / CHANGED,
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
    print(f"T130_PREFLIGHT_OK head={head} control={CONTROL} anchors={std_frame.ANCHORS}", flush=True)
    failures = []

    stats = build(data, work)
    print("T130_MAP " + " ".join(f"{k}->{v:+.0f}" for k, v in stats["map"].items()), flush=True)
    print(f"T130_QUANTILES {json.dumps(stats['quantiles'])}", flush=True)

    if not (model_dir / "history.json").exists():
        print("=== T130 STAGE train (32M, calibrated labels) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, "--factorize",
                  "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", work / "labels",
                  "--limit-train", LIMIT_TRAIN, "--checkpoint-minutes", "10", "--resume", *stop_args],
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
    print(f"T130_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f}", flush=True)

    print("=== T130 STAGE standard frame ===", flush=True)
    ctrl, bad = std_frame.measure(CONTROL, env)
    failures += bad
    cand, bad = std_frame.measure(name, env)
    failures += bad
    pooled, per, bad = std_frame.delta(cand, ctrl, "(calibrated - task128 C)")
    failures += bad
    print(f"T130_FRAME pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
          + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "CALIB_HELPS" if lo > 0 else "CALIB_HURTS" if hi < 0
               else "CALIB_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 130, "git": {"head": head, "pushed": False}, "map": stats,
                               "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T130_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
