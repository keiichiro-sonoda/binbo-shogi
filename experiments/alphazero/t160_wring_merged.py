#!/usr/bin/env python3
"""Task 160: wringing with the round 1 + round 2 self-play data merged (~20M rows), from task142_finetune.

Why: round 2 alone (Task 159, from task154_R_result30) gave only +12.0 [-3.3, +27.2], and more epochs on the same data
hurt. The adopted task154_R_result30 = task142_finetune + round 1 data (10.28M rows, 2 epochs). Starting again from
task142_finetune with both rounds isolates "twice the self-play data" (round 2 also comes from a stronger player).
Data: work/task154/cache (round 1, task142 self-play) + work/task159/cache (round 2, task154 self-play), concatenated in
that order, cross-set exact duplicates removed (64-bit row hash, first occurrence kept; each set is already deduped and
excludes the shared val and its mirrors). Arms (fixed before the merge): from models/task142_finetune/nnue_final.pt,
30% result (--cp-result-lambda 0.7 --wdl-known-only), lr 1e-5, --fast, final weights:
  M_e1  1 epoch  (same number of steps as Task 154's recipe)
  M_e2  2 epochs (same epochs as Task 154's recipe)
Decision: s280/s281/s282 pooled vs task154_R_result30 (frames reused); best arm with CI low > 0 is confirmed on
spaced_set(20290000) prefix conf160 vs tanuki- + AobaNNUE and vs micro; both points > 0 -> ADOPT_CANDIDATE.
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

TAG = "task160"
ADOPTED = "task154_R_result30"
START = "task142_finetune"
SOURCES = ("task154", "task159")
R30 = ["--cp-result-lambda", "0.7", "--wdl-known-only"]
ARMS = {"M_e1": ["--epochs", "1", *R30], "M_e2": ["--epochs", "2", *R30]}
CONF_PREFIX = "conf160"
CHUNK = 1 << 20


def row_hash(rows, mult):
    """64-bit polynomial hash of int32 feature rows (wrapping uint64 arithmetic)."""
    return (rows.astype(np.uint64) * mult).sum(axis=1, dtype=np.uint64)


def merge(data, out):
    """Concatenate the train splits of the source caches, drop cross-set duplicates, link the shared val."""
    srcs = [data / "work" / s / "cache" for s in SOURCES]
    metas = [json.loads((s / "meta.json").read_text(encoding="utf-8")) for s in srcs]
    assert all(m["features_per_position"] == 77 and m["val_shards"] == metas[0]["val_shards"] for m in metas)
    mult = np.random.default_rng(20271600).integers(1, 2**63, size=77, dtype=np.uint64) | np.uint64(1)
    hashes, counts = [], []
    for s in srcs:
        idx = np.load(s / "train_indices.npy", mmap_mode="r")
        hashes.append(np.concatenate([row_hash(np.asarray(idx[i:i + CHUNK]), mult) for i in range(0, len(idx), CHUNK)]))
        counts.append(len(idx))
    allh = np.concatenate(hashes)
    _, first = np.unique(allh, return_index=True)
    keep = np.zeros(len(allh), dtype=bool)
    keep[first] = True
    n = int(keep.sum())
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "labels").mkdir(parents=True)
    dst = {k: np.lib.format.open_memmap(tmp / f"train_{k}.npy", mode="w+", dtype=dt, shape=shp)
           for k, dt, shp in (("indices", np.int32, (n, 77)), ("evals", np.int16, (n,)), ("wdl", np.int8, (n,)))}
    pos, off = 0, 0
    for s, c in zip(srcs, counts):
        k = keep[off:off + c]
        off += c
        for name, arr in dst.items():
            src = np.load(s / f"train_{name}.npy", mmap_mode="r")
            assert len(src) == c and src.dtype == arr.dtype
            for i in range(0, c, CHUNK):
                part = np.asarray(src[i:i + CHUNK])[k[i:i + CHUNK]]
                j = pos + int(k[:i].sum())
                arr[j:j + len(part)] = part
        pos += int(k.sum())
    for arr in dst.values():
        arr.flush()
    for k in ("indices", "evals", "wdl"):
        os.symlink(os.readlink(srcs[0] / f"val_{k}.npy"), tmp / f"val_{k}.npy")
    os.symlink("../train_evals.npy", tmp / "labels/train_dlshogi_evals.npy")
    os.symlink(os.readlink(srcs[0] / "labels/val_dlshogi_evals.npy"), tmp / "labels/val_dlshogi_evals.npy")
    m0 = metas[0]
    meta = dict(task=160, format="task160-merged-v1", canonical=True, canonical_shared=True, features_per_position=77,
                total_features=m0["total_features"], n_train=n, n_val=m0["n_val"], val_shards=m0["val_shards"],
                counts=dict(sources={s: c for s, c in zip(SOURCES, counts)}, train=n,
                            cross_duplicates=int(len(allh) - n), val=m0["n_val"]),
                sources=[str(s) for s in srcs], shared_validation=m0["shared_validation"],
                labels="labels/; train score and WDL are leaf side-to-move (as in the sources)")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (tmp / "labels/meta.json").write_text(json.dumps(dict(task=160, train=n, val=m0["n_val"], cache=str(out),
                                                         provenance="../meta.json", score_unit="internal Value, STM")),
                                          encoding="utf-8")
    tmp.rename(out)
    print(f"T160_MERGE rows={n} sources={dict(zip(SOURCES, counts))} cross_duplicates={len(allh) - n}", flush=True)


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    work = data / "work" / TAG
    cache, comp = work / "cache", work / "cache_compact"
    if not (cache / "meta.json").exists():
        work.mkdir(parents=True, exist_ok=True)
        print("=== T160 STAGE merge ===", flush=True)
        merge(data, cache)
    if not (comp / "compact.json").exists():
        t106.run([py, "-B", "-m", "trainer.compact_cache", cache, comp], logs / f"{TAG}_compact.log", cwd=repo, env=env)
    n = json.loads((cache / "meta.json").read_text(encoding="utf-8"))["n_train"]
    print(f"T160_DATA rows={n}", flush=True)
    start = data / "models" / START / "nnue_final.pt"
    failures, names = [], []
    for key, extra in ARMS.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T160 STAGE train {key} ===", flush=True)
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
        print(f"T160_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = [x for x in names if results[x]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda x: results[x]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20290000)}
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cc, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T160_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        good = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if good else "NOT_CONFIRMED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 160, "data_rows": n, "results": results, "best": best, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T160_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
