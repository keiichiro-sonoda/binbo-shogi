#!/usr/bin/env python3
"""Task 159: wringing round 2 -- the same recipe as Task 154, now from the adopted task154_R_result30.

Round 2 of the Task 154 recipe (which gave +28.6). Changes (fixed before the data exist), following the Task 156 audit:
the self-play AND the qsearch replacement use the adopted eval task154_R_result30 explicitly (t153's defaults are
task142_finetune), a new work dir (work/task159) and a new generation seed (20271590); training starts from
models/task154_R_result30/nnue_final.pt. Arms: R30_e2 (30% result, 2 epochs, = Task 154's recipe) and R30_e4 (4 epochs).
Decision as in Task 154/155: s280/s281/s282 pooled vs task154_R_result30 (its frames reused); best arm with CI low > 0 is
confirmed on spaced_set(20289000) prefix conf159 vs tanuki- + AobaNNUE and vs micro; both points > 0 -> ADOPT_CANDIDATE.

--- Task 154's description, for reference ---
Task 154: proper "wringing" -- binbo's own self-play positions and game results, played until mate is read.

Why: relabelling Gikou2 self-play positions with binbo's depth 9 did not help (Task 147 / 150); Codex (Task 150 review)
listed the position distribution as a candidate cause, and those positions carry almost no game results (0.6%).
Data (Task 153 tool): the adopted eval task142_finetune vs itself, depth 9, Threads 1, hash 16 MB, random opening 6-16
plies (not recorded), recording from ply 16, max 320 plies (draw), repetition / declaration per the tool, and
**--eval-limit 32000 = the game ends only when the search reads a mate** (score mate -> +-32000), never on a large
evaluation (author's request: results must not depend on binbo's own judgement). 150,000 games (~10M kept rows expected),
26 processes, node cap 10M per search (added after the first run hit a depth-9 search that did not finish in 120 s,
like Task 146; the 9,174 games of that run were discarded). Then the qsearch replacement (changed rows dropped) and a trainer cache with results (train_wdl).

Arms (fixed before the data exist; both start from models/task142_finetune/nnue_final.pt, the Suisho5-finish amount:
lr 1e-5, 2 epochs, final weights, --fast):
  E  eval only (labels = self-play depth-9 scores)
  R  eval mixed with the game result: --cp-result-lambda 0.7 --wdl-known-only (30% result; draws keep the eval target)
Decision: new seed sets s280/s281/s282 pooled, paired, each arm - task142_finetune. The best arm with CI low > 0 is
confirmed on a fresh spaced set (std_frame.spaced_set(20287000), prefix conf154) vs tanuki- + AobaNNUE AND vs micro
(CONFIRM_ANCHORS) on the same set: both point estimates > 0 -> ADOPT_CANDIDATE, else NOT_CONFIRMED. No arm with
CI low > 0 -> NO_BETTER. Gates: 0cp GREEN, frame failures 0.
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

TAG = "task159"
ADOPTED = "task154_R_result30"
SEED = 20271590
GAMES, PROCS = 150_000, 26
R30 = ["--cp-result-lambda", "0.7", "--wdl-known-only"]
ARMS = {"R30_e2": ["--epochs", "2", *R30], "R30_e4": ["--epochs", "4", *R30]}
CONF_PREFIX = "conf159"


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    gen = repo / "experiments/alphazero/t153_selfplay_gen.py"
    evaldir = data / "eval_dirs" / ADOPTED
    work = data / "work" / TAG
    cache = work / "cache"
    comp = work / "cache_compact"
    if not (cache / "meta.json").exists():
        print("=== T159 STAGE gen (self-play to mate) ===", flush=True)
        t106.run([py, "-B", gen, "gen", "--games", GAMES, "--depth", "9", "--procs", PROCS, "--out", work,
                  "--eval-limit", "32000", "--max-ply", "320", "--nodes-cap", "10000000", "--eval-dir", evaldir,
                  "--seed", SEED], logs / f"{TAG}_gen.log", cwd=repo, env=env)
        print("=== T159 STAGE qs ===", flush=True)
        t106.run([py, "-B", gen, "qs", "--work", work, "--procs", PROCS, "--eval-dir", evaldir], logs / f"{TAG}_qs.log", cwd=repo, env=env)
        t106.run([py, "-B", gen, "assemble", "--work", work, "--out", cache], logs / f"{TAG}_assemble.log",
                 cwd=repo, env=env)
    if not (comp / "compact.json").exists():
        t106.run([py, "-B", "-m", "trainer.compact_cache", cache, comp], logs / f"{TAG}_compact.log", cwd=repo, env=env)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_train"]
    print(f"T159_DATA rows={n} meta_counts={meta.get('counts')}", flush=True)
    start = data / "models" / ADOPTED / "nnue_final.pt"
    failures, names = [], []
    for key, extra in ARMS.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T159 STAGE train {key} ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", cache, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--lr", "1e-5",
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
        print(f"T159_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = [x for x in names if results[x]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda x: results[x]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20289000)}
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T159_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        good = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if good else "NOT_CONFIRMED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 159, "data_rows": n, "results": results, "best": best, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T159_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
