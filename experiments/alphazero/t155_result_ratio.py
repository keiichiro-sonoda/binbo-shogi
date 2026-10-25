#!/usr/bin/env python3
"""Task 155: how much of the game result to mix in -- same Task 154 self-play data, other result fractions.

Why: Task 154's 30% result arm (now the adopted eval task154_R_result30) beat the eval-only arm by ~+11; 30% was a
single guess. Same data (work/task154/cache, 10.28M rows, every row with a result), same recipe (from
models/task142_finetune/nnue_final.pt, lr 1e-5, 2 epochs, final weights, --fast, --wdl-known-only), only the fraction:
  R50  --cp-result-lambda 0.5   (50% result)
  R15  --cp-result-lambda 0.85  (15% result)
Decision: new seed sets s280/s281/s282 pooled, paired, each arm - task154_R_result30 (its frames are reused from
Task 154). The best arm with CI low > 0 is confirmed on a fresh spaced set (std_frame.spaced_set(20288000), prefix
conf155) vs tanuki- + AobaNNUE AND vs micro: both point estimates > 0 -> ADOPT_CANDIDATE, else NOT_CONFIRMED. No arm with
CI low > 0 -> NO_BETTER (30% stays). Gates: 0cp GREEN, frame failures 0.
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

TAG = "task155"
ADOPTED = "task154_R_result30"
START = "task142_finetune"
ARMS = {"R50": ["--cp-result-lambda", "0.5", "--wdl-known-only"],
        "R15": ["--cp-result-lambda", "0.85", "--wdl-known-only"]}
CONF_PREFIX = "conf155"


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    cache = data / "work/task154/cache"
    comp = data / "work/task154/cache_compact"
    n = json.loads((cache / "meta.json").read_text(encoding="utf-8"))["n_train"]
    start = data / "models" / START / "nnue_final.pt"
    failures, names = [], []
    for key, extra in ARMS.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T155 STAGE train {key} ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", cache, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", start, "--eval-npy", cache / "labels", "--limit-train", n,
                      "--compact-cache", comp, *extra], logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        names.append(name)
    base, bad = std_frame.measure_sets(ADOPTED, env)
    failures += bad
    results = {}
    for name in names:
        cand, bad = std_frame.measure_sets(name, env)
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        failures += bad + bad2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T155_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = [x for x in names if results[x]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda x: results[x]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20288000)}
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T155_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        good = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if good else "NOT_CONFIRMED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 155, "results": results, "best": best, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T155_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
