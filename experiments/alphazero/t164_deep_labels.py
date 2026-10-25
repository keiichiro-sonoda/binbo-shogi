#!/usr/bin/env python3
"""Task 164: are deeper self-search labels better? -- depth 11 vs depth 9 on the same 3M self-play positions.

Why: after the first round (+28.6) the self-play wringing stopped adding (round 2 +12.0, merged +2.7, relabelled dropped
leaves +4.4/+5.7). Codex (Task 161, item 6): test the label quality, with a fresh depth-9 control on the same positions
(the stored labels came from the game search with a warm hash table, so a cold d9 differs from them as much as d11 does:
pilot on 26,000 rows, clamp +-2000: corr(stored, d9) 0.978, corr(d9, d11) 0.976, sign flips 7%).
Pilot speed (26 processes, task154_R_result30): d9 1,233 / d11 432 / d12 260 positions/s -> depth 11 for 3M (~1.9 h).
Positions: 3M rows of Task 159's round 2 training cache in a fixed random order (seed 20271640, work/task164/sample).
Both depths with the Task 146 relabeller (Threads 1, hash 16 MB, reset per 1,000-row part, node cap 10M).
Arms (fixed before the labels exist): Task 159's cache (9.83M rows, results kept for every row) with the 3M sampled rows'
labels replaced by the fresh label of the arm; from models/task154_R_result30/nnue_final.pt, 30% result, lr 1e-5,
2 epochs, --fast, final weights:  D9 (fresh depth 9)  /  D11 (depth 11).
Decision: s280/s281/s282 pooled vs task154_R_result30; best arm with CI low > 0 confirmed on spaced_set(20292000)
prefix conf164 vs tanuki- + AobaNNUE and vs micro; both points > 0 -> ADOPT_CANDIDATE. D11 - D9 pooled is reported
(the label-depth effect; descriptive). The 3M rows are 30% of the data (Codex proposed 1M; raised so the effect is
not diluted to 10%).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

TAG = "task164"
ADOPTED = "task154_R_result30"
EVAL = "task154_R_result30"
DEPTHS = {"D9": 9, "D11": 11}
R30 = ["--cp-result-lambda", "0.7", "--wdl-known-only"]
CONF_PREFIX = "conf164"


def label_dir(data, work, key, py, repo, env, logs):
    """Relabel the sample at the arm's depth; write a full-cache label dir with the sampled rows replaced."""
    sample, out = work / "sample", work / f"labels_{key}"
    if (out / "meta.json").exists():
        return out
    raw = work / f"raw_{key}"
    if not (raw / "meta.json").exists():
        rel = repo / "experiments/alphazero/t146_self_relabel.py"
        t106.run([py, "-B", rel, "run", "--work", work / f"relabel_{key}", "--cache", sample, "--eval-dir",
                  data / "eval_dirs" / EVAL, "--depth", str(DEPTHS[key]), "--node-cap", "10000000", "--procs", "26",
                  "--val-rows", "0"], logs / f"{TAG}_relabel_{key}.log", cwd=repo, env=env)
        t106.run([py, "-B", rel, "assemble", "--work", work / f"relabel_{key}", "--out", raw],
                 logs / f"{TAG}_relabel_{key}_assemble.log", cwd=repo, env=env)
    base = data / "work/task159/cache"
    rows = np.load(sample / "rows.npy")
    new = np.load(raw / "train_dlshogi_evals.npy")
    assert len(new) == len(rows)
    ev = np.array(np.load(base / "train_evals.npy"))
    ev[rows] = new
    tmp = work / f"labels_{key}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    np.save(tmp / "train_dlshogi_evals.npy", ev)
    os.symlink(os.readlink(base / "labels/val_dlshogi_evals.npy"), tmp / "val_dlshogi_evals.npy")
    (tmp / "meta.json").write_text(json.dumps(dict(task=164, arm=key, depth=DEPTHS[key], train=len(ev), replaced=len(rows),
                                                   cache=str(base), raw=str(raw))), encoding="utf-8")
    tmp.rename(out)
    return out


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    work = data / "work" / TAG
    cache, comp = data / "work/task159/cache", data / "work/task159/cache_compact"
    n = json.loads((cache / "meta.json").read_text(encoding="utf-8"))["n_train"]
    labels = {key: label_dir(data, work, key, py, repo, env, logs) for key in DEPTHS}
    rows = np.load(work / "sample/rows.npy")
    a, b = (np.clip(np.load(labels[k] / "train_dlshogi_evals.npy")[rows].astype(np.int32), -2000, 2000) for k in DEPTHS)
    print(f"T164_LABELS rows={len(rows)} corr(d9,d11)={np.corrcoef(a, b)[0, 1]:.4f} "
          f"mean_abs_diff={np.mean(np.abs(a - b)):.1f} sign_flip={np.mean(np.sign(a) != np.sign(b)):.4f}", flush=True)
    start = data / "models" / ADOPTED / "nnue_final.pt"
    failures, names = [], []
    for key in DEPTHS:
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T164 STAGE train {key} ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", cache, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", start, "--eval-npy", labels[key], "--limit-train", n,
                      "--compact-cache", comp, *R30], logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        names.append(name)
    base, bad = std_frame.measure_sets(ADOPTED, env)
    failures += bad
    results, frames = {}, {}
    for name in names:
        cand, bad = std_frame.measure_sets(name, env)
        frames[name] = cand
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        failures += bad + bad2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T164_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    depth_effect, _per, _ps, bad = std_frame.delta_sets(frames[f"{TAG}_D11"], frames[f"{TAG}_D9"], "(D11 - D9)")
    failures += bad
    print(f"T164_DEPTH_EFFECT (D11 - D9) pooled={depth_effect['delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in depth_effect['ci95']]}", flush=True)
    ok = [x for x in names if results[x]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda x: results[x]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20292000)}
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T164_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        good = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if good else "NOT_CONFIRMED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 164, "results": results, "depth_effect": depth_effect, "best": best,
                               "failures": failures, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T164_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
