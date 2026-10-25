#!/usr/bin/env python3
"""Task 166 (strategy-night probe P5): does self-play wringing stack on top of the Suisho5 student?

Why: the Suisho5 student (Task 145b: 64M Suisho5 rows from scratch, 12 epochs, task145_suisho5_student_final) is
+41.5 [+26.2, +56.8] over the adopted task154_R_result30 on s280/s281/s282 (existing frames). The adopted eval got +28.6
from wringing on top of task142. If wringing also adds on top of the student, "student + own wringing" is a strength track.
Arms (from models/task145_suisho5_student/nnue_final.pt, the Task 154 recipe: 30% result, lr 1e-5, 2 epochs, --fast):
  S_R1  round 1 self-play data (work/task154/cache, games of task142, 10.28M rows)
  S_R2  round 2 self-play data (work/task159/cache, games of task154, 9.83M rows)
Reported only (a probe, no adoption): each arm - task154_R_result30 and each arm - task145_suisho5_student_final
(the wringing effect on top of the student), pooled over s280/s281/s282.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

TAG = "task166"
ADOPTED = "task154_R_result30"
STUDENT = "task145_suisho5_student_final"
R30 = ["--cp-result-lambda", "0.7", "--wdl-known-only"]
ARMS = {"S_R1": "task154", "S_R2": "task159"}


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    start = data / "models/task145_suisho5_student/nnue_final.pt"
    failures, names = [], []
    for key, src in ARMS.items():
        cache, comp = data / "work" / src / "cache", data / "work" / src / "cache_compact"
        n = json.loads((cache / "meta.json").read_text(encoding="utf-8"))["n_train"]
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T166 STAGE train {key} ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", cache, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", start, "--eval-npy", cache / "labels", "--limit-train", n,
                      "--compact-cache", comp, *R30], logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        names.append(name)
    base, bad = std_frame.measure_sets(ADOPTED, env)
    stud, bad2 = std_frame.measure_sets(STUDENT, env)
    failures += bad + bad2
    results = {}
    for name in names:
        cand, bad = std_frame.measure_sets(name, env)
        t1, per, ps, b1 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        t2, per2, _ps2, b2 = std_frame.delta_sets(cand, stud, f"({name} - {STUDENT})")
        failures += bad + b1 + b2
        results[name] = {"vs_adopted": t1, "vs_adopted_per_anchor": per, "vs_adopted_per_set": ps, "vs_student": t2}
        print(f"T166_DELTA {name}: vs {ADOPTED} {t1['delta_elo']:+.2f} ci={[round(v, 2) for v in t1['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items())
              + f" | vs {STUDENT} {t2['delta_elo']:+.2f} ci={[round(v, 2) for v in t2['ci95']]}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 166, "results": results, "failures": failures}, indent=2), encoding="utf-8")
    print(f"T166_DONE failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
