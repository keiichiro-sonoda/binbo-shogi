#!/usr/bin/env python3
"""Task 162: how many of the dropped (changed) qsearch leaves are positions we already train on?

Why: the qsearch step drops the 38% of rows whose quiet leaf differs from the recorded position (Task 156: biased against
endgames). Before relabelling them (Codex Task 161 item 4), measure how many leaves are new: in self-play the capture
sequence is often actually played, so the leaf may equal a later recorded (kept) position of the same game.
Steps: a separate work dir that reuses Task 159's games (symlink) and manifest; qsearch with --changed inherit (same
engine / eval / options as Task 159, so the kept rows are identical); then compare the changed leaves' HalfKP rows
(exact 77-feature match = same board, hands and side to move after canonicalisation) with Task 159's and Task 154's
training caches, and among themselves. Reports counts overall and by distance to the game end (in recorded rows).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t160_wring_merged import CHUNK, row_hash  # noqa: E402

TAG = "task162"
SRC = "task159"
EVAL = "task154_R_result30"
BUCKETS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 10**9))


def hashes_of(path, mult):
    idx = np.load(path, mmap_mode="r")
    return np.concatenate([row_hash(np.asarray(idx[i:i + CHUNK]), mult) for i in range(0, len(idx), CHUNK)])


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    env = dict(os.environ, PYTHONPATH=str(repo))
    src, work = data / "work" / SRC, data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "games").exists():
        os.symlink(src / "games", work / "games")
        (work / "manifest.json").write_text((src / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    t106.run([py, "-B", repo / "experiments/alphazero/t153_selfplay_gen.py", "qs", "--work", work, "--procs", "26",
              "--eval-dir", data / "eval_dirs" / EVAL, "--changed", "inherit"], logs / f"{TAG}_qs.log", cwd=repo, env=env)
    mult = np.random.default_rng(20271600).integers(1, 2**63, size=77, dtype=np.uint64) | np.uint64(1)
    feats, dists = [], []
    kept_new = 0
    for p in sorted((work / "qs").glob("game_*.npz")):
        z = np.load(p)
        ch = z["changed"]
        kept_new += int((~ch).sum())
        if ch.any():
            feats.append(z["features"][ch])
            dists.append(int(z["input_rows"]) - 1 - z["source_rows"][ch])
    feats, dists = np.concatenate(feats), np.concatenate(dists)
    h = row_hash(feats, mult)
    known = {name: hashes_of(data / "work" / name / "cache/train_indices.npy", mult) for name in ("task159", "task154")}
    in159 = np.isin(h, known["task159"])
    in154 = np.isin(h, known["task154"])
    _, first = np.unique(h, return_index=True)
    uniq = np.zeros(len(h), bool)
    uniq[first] = True
    novel = ~in159 & ~in154 & uniq
    out = dict(task=162, changed_leaves=int(len(h)), kept_rows_recomputed=kept_new,
               in_task159_train=int(in159.sum()), in_task154_train=int((in154 & ~in159).sum()),
               duplicates_within_changed=int((~uniq).sum()), novel_unique=int(novel.sum()), by_distance_to_end={})
    for lo, hi in BUCKETS:
        m = (dists >= lo) & (dists < hi)
        out["by_distance_to_end"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = dict(
            changed=int(m.sum()), in_task159=int((m & in159).sum()), novel_unique=int((m & novel).sum()))
    (logs / f"{TAG}_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("T162_RESULT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
