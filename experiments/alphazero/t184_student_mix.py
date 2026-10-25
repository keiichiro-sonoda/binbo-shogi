#!/usr/bin/env python3
"""Task 184: Task 181's mixed finish on the Suisho5 student (the new base, decision 2026-10-24).

The student (Task 145b) + round-2 self-play wring is S_R2 (Task 166, +69.1 over task154, confirmed +76.0 / micro +45.1).
On task154, mixing the self-play data with the Suisho5 7.24M rows in one pass (M, +22.4) beat applying them one after
the other (F159, +8.4). Here the same manifests are used from the student (models/task145_suisho5_student/nnue_final.pt,
S_R2's start) at lr 1e-5, 2 epochs:
  S_M  = work/task181/M.json  (round-2 self-play 9.83M, 30% result premixed + Suisho5 7.24M)
  S_MR = work/task182/MR.json (round-1 + round-2 self-play + Suisho5 7.24M)
For the student the Suisho5 rows are a replay of its own teacher's distribution (rows it never trained on).
Primary: pooled 3 sets, arm - S_R2. CI low > 0 -> confirm the best on conf184 (spaced_set(20298000)), routine anchors and
micro, against S_R2; both pooled point estimates positive -> ADOPT_CANDIDATE. Descriptive: arm - task154_R_result30.
Judged on this machine (the Xeon generates Task 185's self-play).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

TAG = "task184"
BASE = "task166_S_R2"
OLD = "task154_R_result30"
START = "task145_suisho5_student"
ARMS = {"S_M": "work/task181/M.json", "S_MR": "work/task182/MR.json"}
CONF = {"conf184": std_frame.spaced_set(20298000)}


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    names = {}
    for key, manifest in ARMS.items():
        name, out, manifest = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}", data / manifest
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T184 STAGE train {key} ===", flush=True)
            val_cache = json.loads(manifest.read_text(encoding="utf-8"))["val"]["cache"]
            t106.run([py, "-m", "trainer.train_nnue", val_cache, "--out", out, "--manifest", manifest, "--qat",
                      "--qat-tie", "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
                      "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
                      "--checkpoint-minutes", "10", "--resume", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", data / "models" / START / "nnue_final.pt"], logs / f"{name}.log", cwd=repo, env=env)
        failures = export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        if failures:
            raise RuntimeError(f"export gate failed: {failures}")
        names[key] = name
    base, bad = std_frame.measure_sets(BASE, env)
    old, bad2 = std_frame.measure_sets(OLD, env)
    failures, results = bad + bad2, {}
    for key, name in names.items():
        cand, b = std_frame.measure_sets(name, env)
        total, per, per_set, b2 = std_frame.delta_sets(cand, base, f"({name} - {BASE})")
        vs_old = std_frame.delta_sets(cand, old, f"({name} - {OLD})")[0]
        failures += b + b2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set, "vs_task154": vs_old}
        print(f"T184_DELTA ({name} - {BASE}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items())
              + f" | vs task154 {vs_old['delta_elo']:+.1f}", flush=True)
    ok = [n for n in names.values() if results[n]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda n: results[n]["pooled"]["delta_elo"])
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(BASE, env, sets=CONF, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=CONF, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {BASE})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T184_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}",
                  flush=True)
        results[best]["confirm"] = conf
        confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    (logs / f"{TAG}_results.json").write_text(json.dumps({"verdict": verdict, "best": best, "results": results,
                                                          "failures": failures}, indent=1, default=str), encoding="utf-8")
    print(f"T184_VERDICT {verdict} best={best} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
