#!/usr/bin/env python3
"""Task 116: random-drop control -- is it *what* was dropped, or dropping at all?

Task 114 (qsearch gain >= 300, 12.9% of the Task 106-kept rows) and Task 115 (teacher
gap >= 600 cp, 7.8%) are unrelated selections, yet both lost about 30 Elo on the
standard anchors (-30.5, -35.3) and both raised val loss. Two explanations:
  (V) volume: 12 fixed epochs over fewer rows = fewer updates; Task 109 (12 -> 24
      epochs, +43) shows the 32M students are not saturated.
  (H) hard positions: the dropped rows are the hard ones and they are worth learning.
Here: drop the same number of rows as Task 115 (1,568,143 of the Task 106-kept rows)
uniformly at random (numpy default_rng(SEED)), everything else identical.

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: the row mask. Screening scale = Task 106 exactly (`caches/shared_32m`,
    search labels `work/task104/rescore_search`, 098 NNUE_ARGS + --augment-mirror,
    12 epochs). Control = `task106_student_filter`; comparison arm =
    `task115_student_disagree`. Their standard-frame games are reused.
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, both evals at 100,000 nodes, seeds 20272001-08, 800 games per anchor), pooled.
  * Q1 (random - control): CI high < 0 -> RANDOM_DROP_HURTS / CI low > 0 ->
    RANDOM_DROP_HELPS / else RANDOM_DROP_NEUTRAL.
  * Q2 (Task 115 - random): CI high < 0 -> HARD_DROP_WORSE / CI low > 0 ->
    HARD_DROP_BETTER / else HARD_DROP_SAME.
  * reading: RANDOM_DROP_HURTS + HARD_DROP_SAME -> (V) volume explains Task 114/115;
    RANDOM_DROP_NEUTRAL + HARD_DROP_WORSE -> (H) hard positions matter; other
    combinations are reported as they are.
  * gates: 0cp GREEN, no illegal move / engine failure, dropped count exactly DROP_N.
  * descriptive: val loss vs control and Task 115.

Interruptible (scripts/job.sh pause|resume task116).
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

TAG = "task116"
CONTROL = "task106_student_filter"
HARD = "task115_student_disagree"
DROP_N = 20_117_842 - 18_549_699   # Task 115 extra skip
SEED = 20271007
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def build_mask(data, work):
    out = work / "mask"
    if (out / "train_keep.npy").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(data / BASE_MASK)
    if base.shape[0] != LIMIT_TRAIN:
        raise RuntimeError(f"mask rows {base.shape[0]} != {LIMIT_TRAIN}")
    drop = np.random.default_rng(SEED).choice(np.flatnonzero(base), size=DROP_N, replace=False)
    keep = base.copy()
    keep[drop] = False
    np.save(out / "train_keep.npy", keep)
    kept_base = int(base.sum())
    stats = {"rows": int(base.size), "base_keep": kept_base, "keep": int(keep.sum()), "dropped": kept_base - int(keep.sum()),
             "extra_skip_frac_of_base": float(1 - keep.sum() / kept_base), "seed": SEED}
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("T116_MASK " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_random"
    eval_name = f"{TAG}_student_random"
    eval_dir = data / "eval_dirs" / eval_name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK,
                 data / "eval_dirs" / CONTROL / "nn.bin", data / "eval_dirs" / HARD / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T116_PREFLIGHT_OK head={head} control={CONTROL} hard={HARD} drop_n={DROP_N}", flush=True)
    failures = []

    print("=== T116 STAGE mask (random drop of DROP_N Task 106-kept rows) ===", flush=True)
    mask = build_mask(data, work)
    if mask["dropped"] != DROP_N:
        failures.append(f"dropped {mask['dropped']} != {DROP_N}")

    if not (model_dir / "history.json").exists():
        print("=== T116 STAGE train (32M, Task 106 recipe, random-drop mask) ===", flush=True)
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
    hard_hist = json.loads((data / "models" / HARD / "history.json").read_text(encoding="utf-8"))
    print(f"T116_VAL best_val_loss={hist['best_val_loss']:.1f} control={ctrl_hist['best_val_loss']:.1f} "
          f"task115={hard_hist['best_val_loss']:.1f}", flush=True)
    print(f"T116_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T116 STAGE standard frame (control, Task 115, random) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    hard, bad_h = std_frame.measure(HARD, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    failures += bad_c + bad_h + bad_k
    res = {}
    for key, (x, y, label) in {"q1": (cand, ctrl, "(random - task106)"), "q2": (hard, cand, "(task115 - random)")}.items():
        pooled, per, bad = std_frame.delta(x, y, label)
        failures += bad
        res[key] = {"pooled": pooled, "per_anchor": per}
        print(f"T116_{key.upper()} {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    lo, hi = res["q1"]["pooled"]["ci95"]
    q1 = "INVALID" if failures else "RANDOM_DROP_HELPS" if lo > 0 else "RANDOM_DROP_HURTS" if hi < 0 else "RANDOM_DROP_NEUTRAL"
    lo, hi = res["q2"]["pooled"]["ci95"]
    q2 = "INVALID" if failures else "HARD_DROP_BETTER" if lo > 0 else "HARD_DROP_WORSE" if hi < 0 else "HARD_DROP_SAME"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 116, "git": {"head": head, "pushed": False}, "mask": mask,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "task115_best_val_loss": hard_hist.get("best_val_loss"),
                               "frame": res, "failures": failures, "q1": q1, "q2": q2}, indent=2), encoding="utf-8")
    print(f"T116_VERDICT q1={q1} q2={q2} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
