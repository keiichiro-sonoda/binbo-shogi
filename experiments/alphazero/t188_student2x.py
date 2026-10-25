#!/usr/bin/env python3
"""Task 188: the Suisho5 student on 2x the public data, then the S_MR finish, judged against S_MR (reports/task188_student2x.md).

  S2    = from scratch on chunks A + B (2 x 63,950,000 rows, work/task188/chunk_{a,b}.json) with --chunk-manifests
          (Task 188a), Task 145's recipe (098 NNUE_ARGS at lr 6e-5, 12 epochs, Factorizer, mirror, QAT, --fast); final weights.
  S2_MR = S2 + the S_MR finish (work/task182/MR.json: old self-play rounds 1+2 + Suisho5 7.24M), lr 1e-5, 2 epochs.
Primary: new yardsticks (Task 186), pooled 3 sets, S2_MR - S_MR, on the Xeon. CI low > 0 -> confirm on conf188
(spaced_set(20301000)) with routine anchors and micro; both pooled point estimates positive -> ADOPT_CANDIDATE.
Descriptive afterwards (only if the primary ran clean): S2 - task145_suisho5_student_final (before the finish).
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
import xeon_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402
from t179_refinish import parallel  # noqa: E402

TAG = "task188"
BASE = "task184_S_MR"
STUDENT = "task145_suisho5_student_final"
CONF = {"conf188": std_frame.spaced_set(20301000)}


def train(py, data, logs, repo, env, name, args):
    out = data / "models" / name
    if not (out / "history.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== T188 STAGE train {name} ===", flush=True)
        t106.run([py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", out, *args, "--factorize", "--augment-mirror",
                  "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--checkpoint-minutes", "10", "--resume"],
                 logs / f"{name}.log", cwd=repo, env=env)
    failures = export(name, out, "nnue_final.pt", data, logs, py, repo, env)
    if failures:
        raise RuntimeError(f"export gate failed: {failures}")


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    work = data / "work" / TAG
    chunks = ",".join(str(work / f"chunk_{k}.json") for k in ("a", "b"))
    s2, s2mr = f"{TAG}_S2", f"{TAG}_S2_MR"
    train(py, data, logs, repo, env, s2, [*t129.LR_ARGS, "--epochs", str(t129.EPOCHS), "--chunk-manifests", chunks])
    train(py, data, logs, repo, env, s2mr, ["--manifest", data / "work/task182/MR.json", "--qat", "--qat-tie",
                                            "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024",
                                            "--scale-cp", "600", "--lr", "1e-5", "--epochs", "2",
                                            "--init-weights", data / "models" / s2 / "nnue_final.pt"])
    print(f"T188_ANCHORS {std_frame.ANCHORS} confirm={std_frame.CONFIRM_ANCHORS}", flush=True)
    base, bad = xeon_frame.remote_measure_sets(BASE)
    cand, bad2 = xeon_frame.remote_measure_sets(s2mr)
    total, per, per_set, bad3 = std_frame.delta_sets(cand, base, f"({s2mr} - {BASE})")
    failures = bad + bad2 + bad3
    results = {"pooled": total, "per_anchor": per, "per_set": per_set}
    print(f"T188_DELTA ({s2mr} - {BASE}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
          + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    verdict = "INVALID" if failures else "NO_BETTER"
    if total["ci95"][0] > 0 and not failures:
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            got = parallel({"base": lambda w=which: xeon_frame.remote_measure_sets(BASE, sets=CONF, which=w),
                            "cand": lambda w=which: xeon_frame.remote_measure_sets(s2mr, sets=CONF, which=w)})
            (cb, b1), (cc, b2) = got["base"], got["cand"]
            ct, cper, _ps, b4 = std_frame.delta_sets(cc, cb, f"confirm {label} ({s2mr} - {BASE})")
            failures += b1 + b2 + b4
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T188_CONFIRM {label} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results["confirm"] = conf
        confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    record = {"verdict": verdict, "results": results, "failures": failures}
    (logs / f"{TAG}_results.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"T188_VERDICT {verdict} failures={failures}", flush=True)
    if not failures:
        a, _ = xeon_frame.remote_measure_sets(s2)
        b, _ = xeon_frame.remote_measure_sets(STUDENT)
        desc = std_frame.delta_sets(a, b, f"({s2} - {STUDENT})")[0]
        record["descriptive_s2_vs_student"] = desc
        (logs / f"{TAG}_results.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        print(f"T188_DESC ({s2} - {STUDENT}) pooled={desc['delta_elo']:+.2f} ci={[round(v, 2) for v in desc['ci95']]}",
              flush=True)


if __name__ == "__main__":
    main()
