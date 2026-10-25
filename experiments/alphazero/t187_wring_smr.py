#!/usr/bin/env python3
"""Task 187: the first wring of the student lineage with its own self-play (Task 185: S_R2 vs S_R2, 150k games),
mixed with the Suisho5 7.24M rows as in Tasks 181/184, from the adopted eval task184_S_MR (lr 1e-5, 2 epochs):
  R3M  = B3 (Task 185 self-play, 30% result premixed like Task 172) + P (Suisho5 7.24M)
  R3MR = B1 + B2 (the old self-play rounds, as in S_MR) + B3 + P
Judged with the recalibrated yardsticks of Task 186 (std_frame.ANCHORS / CONFIRM_ANCHORS after its commit):
primary pooled 3 sets, arm - S_MR; CI low > 0 -> confirm the best on conf187 (spaced_set(20300000)) with the routine
anchors and micro; both pooled point estimates positive -> ADOPT_CANDIDATE. Judging on the Xeon (xeon_frame) unless
it is busy for long; the 14700KF stays free for the event pass-through test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
import xeon_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402
from t172_build import dump, part_record, premix  # noqa: E402
from t179_refinish import parallel  # noqa: E402
from trainer.multi_cache import FORMAT_MULTI, Manifest  # noqa: E402

TAG = "task187"
BASE = "task184_S_MR"
CONF = {"conf187": std_frame.spaced_set(20300000)}


def build(data, work):
    cache = data / "work/task185/cache"
    raw = np.load(cache / "train_evals.npy", mmap_mode="r")
    wdl = np.load(cache / "train_wdl.npy", mmap_mode="r")
    labels = work / "B3_result30.npy"
    if not labels.exists():
        out = np.lib.format.open_memmap(work / "B3_result30.tmp.npy", mode="w+", dtype=np.int16, shape=raw.shape)
        for start in range(0, len(raw), 65536):
            out[start:start + 65536] = premix(raw[start:start + 65536], wdl[start:start + 65536])
        out.flush()
        del out
        os.replace(work / "B3_result30.tmp.npy", labels)
    b3 = dict(cache=cache, compact=data / "work/task185/cache_compact", labels=labels, rows=len(raw))
    m = json.loads((data / "work/task182/MR.json").read_text(encoding="utf-8"))   # B1, B2, P with their masks
    b1, b2, p = m["parts"]
    for arm, parts in {"R3M": [b3, p], "R3MR": [b1, b2, b3, p]}.items():
        path = work / f"{arm}.json"
        if path.exists():
            continue
        records, offset = [], 0
        for part in parts:
            if "source_stamp" in part:   # already a verified record from MR.json
                rec = dict(part, offset=offset)
            else:
                rec = part_record(part["cache"], part["compact"], part["labels"], part["rows"], "train", None, offset)
            records.append(rec)
            offset += rec["rows"]
        dump(path, dict(format=FORMAT_MULTI, rows=offset, parts=records, val=m["val"], arm=arm,
                        provenance="B1/B2: task142/task154 selfplay; B3: task166_S_R2 selfplay (Task 185); 30% result premixed; P: Suisho5 7.24M",
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
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    build(data, work)
    names = {}
    for arm in ("R3M", "R3MR"):
        name, out, manifest = f"{TAG}_{arm}", data / "models" / f"{TAG}_{arm}", work / f"{arm}.json"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T187 STAGE train {arm} ===", flush=True)
            val_cache = json.loads(manifest.read_text(encoding="utf-8"))["val"]["cache"]
            t106.run([py, "-m", "trainer.train_nnue", val_cache, "--out", out, "--manifest", manifest, "--qat",
                      "--qat-tie", "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
                      "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
                      "--checkpoint-minutes", "10", "--resume", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", data / "models" / BASE / "nnue_final.pt"], logs / f"{name}.log", cwd=repo, env=env)
        failures = export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        if failures:
            raise RuntimeError(f"export gate failed: {failures}")
        names[arm] = name
    print(f"T187_ANCHORS {std_frame.ANCHORS} confirm={std_frame.CONFIRM_ANCHORS}", flush=True)
    base, bad = xeon_frame.remote_measure_sets(BASE)
    failures, results = list(bad), {}
    for arm, name in names.items():
        cand, b = xeon_frame.remote_measure_sets(name)
        total, per, per_set, b2 = std_frame.delta_sets(cand, base, f"({name} - {BASE})")
        failures += b + b2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T187_DELTA ({name} - {BASE}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = [n for n in names.values() if results[n]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda n: results[n]["pooled"]["delta_elo"])
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            got = parallel({"base": lambda w=which: xeon_frame.remote_measure_sets(BASE, sets=CONF, which=w),
                            "cand": lambda w=which: xeon_frame.remote_measure_sets(best, sets=CONF, which=w)})
            (cb, b1), (cc, b2) = got["base"], got["cand"]
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {BASE})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T187_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}",
                  flush=True)
        results[best]["confirm"] = conf
        confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    (logs / f"{TAG}_results.json").write_text(json.dumps({"verdict": verdict, "best": best, "results": results,
                                                          "failures": failures}, indent=1, default=str), encoding="utf-8")
    print(f"T187_VERDICT {verdict} best={best} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
