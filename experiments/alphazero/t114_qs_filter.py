#!/usr/bin/env python3
"""Task 114: widen the tactical filter with a quiescence search (32M screening).

Task 106 skipped positions in check or whose gikou@d8 best move is a capture (+20 direct,
+48 on gikou@7) and is the base of every later eval. A position can still be tactically
unstable when a profitable capture exists that is not the best move. AobaNNUE's author
notes they rewrite positions that "move under quiescence search". Here: run a
capture-only material quiescence search (bench_qsearch.QSearch, MAX_PLY 16) on every row
Task 106 keeps and skip the row when the side to move gains >= 300 cp of material
(about a lance; pawn grabs are not counted). Calibration on 20,000 kept rows (design
only, no labels): 12.3% of kept rows reach 300 (23.0% gain anything).

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable: the row mask. Screening scale = Task 106 exactly: `caches/shared_32m`,
    search labels `work/task104/rescore_search`, recipe 098 NNUE_ARGS + --augment-mirror,
    12 epochs. Control = `task106_student_filter`. Candidate mask = Task 106 mask AND
    qsearch gain < 300.
  * judged per docs/evaluation_protocol.md: standard anchor frame (tanuki- @10,000,
    AobaNNUE @5,000, both evals at 100,000 nodes, seeds 20272001-08, 800 games per
    anchor per eval), pooled (candidate - control) over the 1,600 shared openings.
    No direct match.
  * gates: 0cp GREEN, no illegal move / engine failure, extra skip fraction within
    [5%, 25%] of the Task 106-kept rows.
  * decision: CI low > 0 -> QS_FILTER_HELPS (next: the same mask at 64M) /
    CI high < 0 -> QS_FILTER_HURTS / otherwise QS_FILTER_NEUTRAL.

Interruptible (scripts/job.sh pause|resume task114).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from bench_qsearch import MATE, QSearch, material  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402

TAG = "task114"
CONTROL = "task106_student_filter"
GAIN_MIN = 300
SKIP_RANGE = (0.05, 0.25)
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000


def gain_shard(job):
    run_dir, shard, keep_slice, out = job
    if os.path.exists(out):
        return out
    entries = np.fromfile(os.path.join(run_dir, shard["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
    gains = np.full(len(entries), -1, dtype=np.int16)        # -1 = not computed (row already skipped)
    board = cshogi.Board()
    for i in np.flatnonzero(keep_slice):
        board.set_hcp(np.asarray(entries[i]["hcp"]))
        value, _ = QSearch(material).search(board, -MATE, MATE, 0)
        gains[i] = int(min(max(value - material(board), 0), 30000))
    tmp = out + ".tmp.npz"
    np.savez(tmp, gains=gains)
    os.replace(tmp, out)
    return out


def build_mask(data, work):
    out = work / "mask"
    if (out / "train_keep.npy").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    (out / "parts").mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))
    primary, _ = shard_files(meta["run_dir"])
    groups = [(meta["run_dir"], s) for s in primary[:-len(meta["val_shards"])]] + \
             [(rd, s) for rd in meta["extra_runs"] for s in shard_files(rd)[0]]
    base = np.load(data / BASE_MASK)
    jobs, cur = [], 0
    for rd, shard in groups:
        n = shard["count"]
        jobs.append((rd, shard, base[cur:cur + n].copy(), str(out / "parts" / f"{Path(rd).name}_{shard['index']:04d}.npz")))
        cur += n
    if cur != base.shape[0]:
        raise RuntimeError(f"shards cover {cur} rows != mask {base.shape[0]}")
    with Pool(4) as pool:   # 14 and 8 both hit the thermal guard (90°C); 4 matches the safe match parallelism
        for k, _ in enumerate(pool.imap(gain_shard, jobs, chunksize=2), 1):
            if k % 200 == 0:
                print(f"T114_MASK {k}/{len(jobs)} shards", flush=True)
            if t106.stop_requested():
                pool.terminate()
                t106.pause_here(f"mask {k}/{len(jobs)}")
    gains = np.concatenate([np.load(j[3])["gains"] for j in jobs])
    keep = base & (gains < GAIN_MIN)
    np.save(out / "train_keep.npy", keep)
    np.save(out / "train_qs_gain.npy", gains)
    kept_base = int(base.sum())
    stats = {"rows": int(base.size), "base_keep": kept_base, "keep": int(keep.sum()),
             "extra_skip_frac_of_base": float(1 - keep.sum() / kept_base), "keep_frac": float(keep.mean()),
             "gain_any_frac_of_base": float((gains[base] > 0).mean())}
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("T114_MASK " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    model_dir = data / "models" / f"{TAG}_student_qsfilter"
    eval_name = f"{TAG}_student_qsfilter"
    eval_dir = data / "eval_dirs" / eval_name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK,
                 data / "eval_dirs" / CONTROL / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T114_PREFLIGHT_OK head={head} control={CONTROL} gain_min={GAIN_MIN}", flush=True)
    failures = []

    print("=== T114 STAGE mask (qsearch on Task 106-kept rows) ===", flush=True)
    mask = build_mask(data, work)
    if not SKIP_RANGE[0] <= mask["extra_skip_frac_of_base"] <= SKIP_RANGE[1]:
        failures.append(f"extra skip {mask['extra_skip_frac_of_base']:.3f} outside {SKIP_RANGE}")

    if not (model_dir / "history.json").exists():
        print("=== T114 STAGE train (32M, Task 106 recipe, wider mask) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS, "--augment-mirror",
                  "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN,
                  "--row-mask", work / "mask/train_keep.npy", "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                  "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                  "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{TAG}_yane.log", cwd=repo, env=env)
    gate = logs / f"{TAG}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    print(f"T114_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T114 STAGE standard frame (control, then candidate) ===", flush=True)
    ctrl, bad_c = std_frame.measure(CONTROL, env)
    cand, bad_k = std_frame.measure(eval_name, env)
    pooled, per, bad_o = std_frame.delta(cand, ctrl, "(qsfilter - task106)")
    failures += bad_c + bad_k + bad_o
    for name, d in per.items():
        print(f"T114_FRAME {name} delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T114_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "QS_FILTER_HELPS" if lo > 0 else "QS_FILTER_HURTS" if hi < 0
               else "QS_FILTER_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 114, "git": {"head": head, "pushed": False}, "mask": mask,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T114_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
