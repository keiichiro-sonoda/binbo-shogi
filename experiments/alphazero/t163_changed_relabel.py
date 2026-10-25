#!/usr/bin/env python3
"""Task 163: put the dropped (changed) qsearch leaves back with a fresh depth-9 label.

Why: the qsearch step drops the rows whose quiet leaf differs from the recorded position (38%, biased against endgames,
Task 156). Task 162 found 3.18M of round 2's 6.71M changed leaves are new (not in the round 1 / round 2 training data,
unique), 59% new within 7 plies of the end. Codex (Task 161, item 4) proposed relabelling them.
Data: the novel leaves from work/task162/qs (round 2 games, qsearch with task154_R_result30), in a fixed random order
(seed 20271630). Each is searched again at depth 9 by task154_R_result30 (the Task 146 relabeller: Threads 1, hash 16 MB,
reset per 1,000-row part, node cap 10M), side to move, internal Value, mate +-32000. These rows get NO game result
(train_wdl 0): the leaf is a qsearch PV end, not a game position, so the game result is not mixed in (Codex's caution).
Training cache = Task 159's round 2 cache (9.83M rows, results) + all novel leaves; arms select rows with --row-mask.
Arms (fixed before the labels exist; from models/task154_R_result30/nnue_final.pt, 30% result on rows with a result,
lr 1e-5, 2 epochs, --fast, final weights; the control is Task 159's R30_e2 = the same without extra rows, +12.0):
  X1  round 2 + the first 1M novel leaves
  X3  round 2 + all 3.18M novel leaves
Decision as usual: s280/s281/s282 pooled vs task154_R_result30; best arm with CI low > 0 confirmed on
spaced_set(20291000) prefix conf163 vs tanuki- + AobaNNUE and vs micro; both points > 0 -> ADOPT_CANDIDATE.
Each arm's pooled difference vs task159_R30_e2 (same start, same base data) is also reported (descriptive).
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
from t160_wring_merged import CHUNK, row_hash  # noqa: E402

TAG = "task163"
ADOPTED = "task154_R_result30"
CONTROL = "task159_R30_e2"
EVAL = "task154_R_result30"
SUBSET = 1_000_000
R30 = ["--cp-result-lambda", "0.7", "--wdl-known-only"]
CONF_PREFIX = "conf163"


def novel_leaves(data):
    """Changed leaves of work/task162/qs that are in neither round's training data, unique, in a fixed random order."""
    mult = np.random.default_rng(20271600).integers(1, 2**63, size=77, dtype=np.uint64) | np.uint64(1)
    feats = [np.load(p)["features"][np.load(p)["changed"]] for p in sorted((data / "work/task162/qs").glob("game_*.npz"))]
    feats = np.concatenate(feats)
    h = row_hash(feats, mult)
    known = []
    for name in ("task159", "task154"):
        idx = np.load(data / "work" / name / "cache/train_indices.npy", mmap_mode="r")
        known.append(np.concatenate([row_hash(np.asarray(idx[i:i + CHUNK]), mult) for i in range(0, len(idx), CHUNK)]))
    _, first = np.unique(h, return_index=True)
    keep = np.zeros(len(h), bool)
    keep[first] = True
    keep &= ~np.isin(h, np.concatenate(known))
    rows = feats[keep]
    return rows[np.random.default_rng(20271630).permutation(len(rows))]


def build(data, work, py, repo, env, logs):
    """Novel-leaf mini cache -> Task 146 relabel -> training cache (round 2 + novel leaves) + row masks."""
    novel = work / "novel"
    if not (novel / "meta.json").exists():
        rows = novel_leaves(data)
        tmp = work / "novel.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / "train_indices.npy", rows)
        os.symlink(os.readlink(data / "work/task159/cache/val_indices.npy"), tmp / "val_indices.npy")
        (tmp / "meta.json").write_text(json.dumps(dict(task=163, n_train=len(rows), source="work/task162/qs changed, novel",
                                                       order_seed=20271630)), encoding="utf-8")
        tmp.rename(novel)
    n_novel = json.loads((novel / "meta.json").read_text(encoding="utf-8"))["n_train"]
    print(f"T163_NOVEL rows={n_novel}", flush=True)
    labels = work / "novel_labels"
    if not (labels / "meta.json").exists():
        rel = repo / "experiments/alphazero/t146_self_relabel.py"
        t106.run([py, "-B", rel, "run", "--work", work / "relabel", "--cache", novel, "--eval-dir", data / "eval_dirs" / EVAL,
                  "--depth", "9", "--node-cap", "10000000", "--procs", "26", "--val-rows", "0"],
                 logs / f"{TAG}_relabel.log", cwd=repo, env=env)
        t106.run([py, "-B", rel, "assemble", "--work", work / "relabel", "--out", labels],
                 logs / f"{TAG}_relabel_assemble.log", cwd=repo, env=env)
    cache = work / "cache"
    if not (cache / "meta.json").exists():
        base = data / "work/task159/cache"
        n_base = json.loads((base / "meta.json").read_text(encoding="utf-8"))["n_train"]
        extra_idx = np.load(novel / "train_indices.npy", mmap_mode="r")
        extra_ev = np.load(labels / "train_dlshogi_evals.npy", mmap_mode="r")
        assert len(extra_idx) == len(extra_ev) == n_novel
        n = n_base + n_novel
        tmp = work / "cache.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "labels").mkdir(parents=True)
        for name, dt, shp, extra in (("indices", np.int32, (n, 77), extra_idx), ("evals", np.int16, (n,), extra_ev),
                                     ("wdl", np.int8, (n,), None)):
            dst = np.lib.format.open_memmap(tmp / f"train_{name}.npy", mode="w+", dtype=dt, shape=shp)
            src = np.load(base / f"train_{name}.npy", mmap_mode="r")
            assert len(src) == n_base and src.dtype == dt
            for i in range(0, n_base, CHUNK):
                j = min(i + CHUNK, n_base)
                dst[i:j] = src[i:j]
            for i in range(0, n_novel, CHUNK):
                dst[n_base + i:n_base + i + CHUNK] = 0 if extra is None else extra[i:i + CHUNK]
            dst.flush()
            del dst
        for k in ("indices", "evals", "wdl"):
            os.symlink(os.readlink(base / f"val_{k}.npy"), tmp / f"val_{k}.npy")
        os.symlink("../train_evals.npy", tmp / "labels/train_dlshogi_evals.npy")
        os.symlink(os.readlink(base / "labels/val_dlshogi_evals.npy"), tmp / "labels/val_dlshogi_evals.npy")
        bm = json.loads((base / "meta.json").read_text(encoding="utf-8"))
        meta = dict(task=163, format="task163-relabel-v1", canonical=True, canonical_shared=True, features_per_position=77,
                    total_features=bm["total_features"], n_train=n, n_val=bm["n_val"], val_shards=bm["val_shards"],
                    counts=dict(base=n_base, novel=n_novel, val=bm["n_val"]), sources=[str(base), str(novel), str(labels)],
                    shared_validation=bm["shared_validation"],
                    labels="rows [0, base): round 2 rows with results; rows [base, n): novel leaves, depth-9 relabel, wdl 0")
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (tmp / "labels/meta.json").write_text(json.dumps(dict(task=163, train=n, val=bm["n_val"], cache=str(cache),
                                                             provenance="../meta.json")), encoding="utf-8")
        for arm, extra in (("X1", SUBSET), ("X3", n_novel)):
            mask = np.zeros(n, bool)
            mask[:n_base + min(extra, n_novel)] = True
            np.save(tmp / f"mask_{arm}.npy", mask)
        tmp.rename(cache)
    comp = work / "cache_compact"
    if not (comp / "compact.json").exists():
        t106.run([py, "-B", "-m", "trainer.compact_cache", cache, comp], logs / f"{TAG}_compact.log", cwd=repo, env=env)
    return cache, comp


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
    cache, comp = build(data, work, py, repo, env, logs)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_train"]
    ev = np.load(cache / "train_evals.npy", mmap_mode="r")[meta["counts"]["base"]:]
    print(f"T163_DATA rows={n} counts={meta['counts']} novel_label_mean={float(np.mean(ev)):.1f} "
          f"novel_abs_p50={float(np.median(np.abs(ev))):.0f} novel_mate_share={float(np.mean(np.abs(ev) == 32000)):.4f}",
          flush=True)
    start = data / "models" / ADOPTED / "nnue_final.pt"
    failures, names = [], []
    for key in ("X1", "X3"):
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T163 STAGE train {key} ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", cache, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--lr", "1e-5", "--epochs", "2",
                      "--init-weights", start, "--eval-npy", cache / "labels", "--limit-train", n,
                      "--row-mask", cache / f"mask_{key}.npy", "--compact-cache", comp, *R30],
                     logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        names.append(name)
    base, bad = std_frame.measure_sets(ADOPTED, env)
    ctrl, bad2 = std_frame.measure_sets(CONTROL, env)
    failures += bad + bad2
    results = {}
    for name in names:
        cand, bad = std_frame.measure_sets(name, env)
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        vs_ctrl, _per, _ps, bad3 = std_frame.delta_sets(cand, ctrl, f"({name} - {CONTROL})")
        failures += bad + bad2 + bad3
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set, "vs_control": vs_ctrl}
        print(f"T163_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items())
              + f" | vs {CONTROL} {vs_ctrl['delta_elo']:+.2f} ci={[round(v, 2) for v in vs_ctrl['ci95']]}", flush=True)
    ok = [x for x in names if results[x]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda x: results[x]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20291000)}
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T163_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        good = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if good else "NOT_CONFIRMED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 163, "data_rows": n, "results": results, "best": best, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T163_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
