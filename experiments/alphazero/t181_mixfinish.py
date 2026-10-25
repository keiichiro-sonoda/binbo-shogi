#!/usr/bin/env python3
"""Task 181: finish the adopted eval on the round-2 self-play data and the light Suisho5 data together.

Task 179 applies the two finishes one after the other. Here they are shuffled into one pass, so neither has to be
undone by the other: M = task154_R_result30 + 2 epochs over
  B2: the round-2 self-play (task159 cache, 9.83M, labels with 30% game result premixed by Task 172, its audit mask)
  P : Suisho5 validation 7.24M (Task 142 labels, no result mixing)
at lr 1e-5 (natural ratio ~58:42). Primary: pooled 3 sets, M - adopted; CI low > 0 -> confirm on conf181
(spaced_set(20297000)) with the routine anchors and micro. Gates: 0cp GREEN, frame failures 0.
Judging goes to the Xeon (xeon_frame); the confirmation splits base (Xeon) / candidate (here).
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
from t172_build import dump, part_record  # noqa: E402
from t179_refinish import parallel  # noqa: E402
from trainer.multi_cache import FORMAT_MULTI, Manifest  # noqa: E402

TAG = "task181"
ADOPTED = "task154_R_result30"
NAME = f"{TAG}_M"
CONF = {"conf181": std_frame.spaced_set(20297000)}


def build_manifest(data, path):
    t172 = data / "work/task172"
    b2 = part_record(data / "work/task159/cache", data / "work/task159/cache_compact", t172 / "B2_result30.npy",
                     9_829_926, "train", t172 / "B2_mask.npy", 0)
    pub = data / "caches/public_suisho5_val7m"
    p = part_record(pub, Path(str(pub) + "_compact"), data / "work/task142/labels_public/train_dlshogi_evals.npy",
                    7_240_673, "train", None, b2["rows"])
    val = json.loads((t172 / "R.json").read_text(encoding="utf-8"))["val"]
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, dict(format=FORMAT_MULTI, rows=b2["rows"] + p["rows"], parts=[b2, p], val=val, arm="M",
                    provenance="B2: task154 selfplay, 30% result premixed (Task 172); P: Suisho5 validation 7.24M (MIT)",
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
    manifest = data / "work" / TAG / "M.json"
    if not manifest.exists():
        build_manifest(data, manifest)
    out = data / "models" / NAME
    if not (out / "history.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        print("=== T181 STAGE train M ===", flush=True)
        val_cache = json.loads(manifest.read_text(encoding="utf-8"))["val"]["cache"]
        t106.run([py, "-m", "trainer.train_nnue", val_cache, "--out", out, "--manifest", manifest, "--qat", "--qat-tie",
                  "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize",
                  "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--checkpoint-minutes", "10",
                  "--resume", "--lr", "1e-5", "--epochs", "2",
                  "--init-weights", data / "models" / ADOPTED / "nnue_final.pt"], logs / f"{NAME}.log", cwd=repo, env=env)
    failures = export(NAME, out, "nnue_final.pt", data, logs, py, repo, env)
    if failures:
        raise RuntimeError(f"export gate failed: {failures}")
    base, bad = std_frame.measure_sets(ADOPTED, env)
    cand, bad2 = xeon_frame.remote_measure_sets(NAME)
    total, per, per_set, bad3 = std_frame.delta_sets(cand, base, f"({NAME} - {ADOPTED})")
    failures += bad + bad2 + bad3
    results = {"pooled": total, "per_anchor": per, "per_set": per_set}
    print(f"T181_DELTA ({NAME} - {ADOPTED}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
          + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    verdict = "INVALID" if failures else "NO_BETTER"
    if total["ci95"][0] > 0 and not failures:
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            got = parallel({"base": lambda w=which: xeon_frame.remote_measure_sets(ADOPTED, sets=CONF, which=w),
                            "cand": lambda w=which: std_frame.measure_sets(NAME, env, sets=CONF, which=w)})
            (cb, b1), (cc, b2) = got["base"], got["cand"]
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({NAME} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T181_CONFIRM {label} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}",
                  flush=True)
        results["confirm"] = conf
        confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    (logs / f"{TAG}_results.json").write_text(json.dumps({"verdict": verdict, "results": results,
                                                          "failures": failures}, indent=1, default=str),
                                             encoding="utf-8")
    print(f"T181_VERDICT {verdict} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
