#!/usr/bin/env python3
"""Task 182: two variants of Task 181's mixed finish (M = task154 + mixed round-2 self-play and Suisho5, 2 epochs).

  M4 = the same manifest (work/task181/M.json), 4 epochs instead of 2
  MR = round-1 self-play added: B1 (work/task154/cache, task142 games, 30% result premixed by Task 172) + B2 + P, 2 epochs
Both from task154_R_result30 at lr 1e-5. Primary: pooled 3 sets, arm - adopted (descriptive: arm - M). No confirmation stage
here: a winner is confirmed in a separate step. Judging waits for the machines: MR on the Xeon (xeon_frame waits for its lock,
held by the Task 180 clock run), M4 here after the Task 180 14700KF half and the Task 181 confirmation have finished.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
import xeon_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402
from t172_build import dump, part_record  # noqa: E402
from t179_refinish import parallel  # noqa: E402
from trainer.multi_cache import FORMAT_MULTI, Manifest  # noqa: E402

TAG = "task182"
ADOPTED = "task154_R_result30"
M = "task181_M"
LOCAL_DONE = ["task180_local_20261024-032341.log", "task181_confirm_20261024-041124.log"]   # wait for these runs


def build_mr(data, path):
    t172 = data / "work/task172"
    b1 = part_record(data / "work/task154/cache", data / "work/task154/cache_compact", t172 / "B1_result30.npy",
                     10_282_217, "train", t172 / "B1_mask.npy", 0)
    m = json.loads((data / "work/task181/M.json").read_text(encoding="utf-8"))
    b2, p = (dict(part) for part in m["parts"])
    b2["offset"], p["offset"] = b1["rows"], b1["rows"] + b2["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, dict(format=FORMAT_MULTI, rows=b1["rows"] + b2["rows"] + p["rows"], parts=[b1, b2, p], val=m["val"], arm="MR",
                    provenance="B1: task142 selfplay, B2: task154 selfplay (30% result premixed, Task 172); P: Suisho5 validation 7.24M",
                    result_lambda_cli=1, result_mix=0.3))
    Manifest(path)


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    mr = data / "work" / TAG / "MR.json"
    if not mr.exists():
        build_mr(data, mr)
    arms = {"M4": (data / "work/task181/M.json", "4"), "MR": (mr, "2")}
    names = {}
    for key, (manifest, epochs) in arms.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T182 STAGE train {key} ===", flush=True)
            val_cache = json.loads(manifest.read_text(encoding="utf-8"))["val"]["cache"]
            t106.run([py, "-m", "trainer.train_nnue", val_cache, "--out", out, "--manifest", manifest, "--qat",
                      "--qat-tie", "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
                      "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
                      "--checkpoint-minutes", "10", "--resume", "--lr", "1e-5", "--epochs", epochs,
                      "--init-weights", data / "models" / ADOPTED / "nnue_final.pt"], logs / f"{name}.log", cwd=repo, env=env)
        failures = export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        if failures:
            raise RuntimeError(f"export gate failed: {failures}")
        names[key] = name

    def local(name):
        while not all("DETACHED_DONE" in (logs / f).read_text(encoding="utf-8") for f in LOCAL_DONE):
            time.sleep(120)
        print(f"T182_LOCAL_START {name}", flush=True)
        return std_frame.measure_sets(name, env)
    frames = parallel({"MR": lambda: xeon_frame.remote_measure_sets(names["MR"]), "M4": lambda: local(names["M4"])})
    base, bad = std_frame.measure_sets(ADOPTED, env)
    mframe, bad2 = std_frame.measure_sets(M, env)
    failures, results = bad + bad2, {}
    for key, name in names.items():
        cand, b = frames[key]
        total, per, per_set, b2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        vs_m = std_frame.delta_sets(cand, mframe, f"({name} - {M})")[0]
        failures += b + b2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set, "vs_M": vs_m}
        print(f"T182_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items())
              + f" | vs M {vs_m['delta_elo']:+.1f} {[round(v, 1) for v in vs_m['ci95']]}", flush=True)
    (logs / f"{TAG}_results.json").write_text(json.dumps({"results": results, "failures": failures}, indent=1,
                                                          default=str), encoding="utf-8")
    print(f"T182_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    main()
