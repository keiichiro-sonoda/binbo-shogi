#!/usr/bin/env python3
"""Task 119: replace tactical positions by their qsearch leaf instead of dropping them (32M).

Task 106 drops positions in check or whose gikou@d8 best move is a capture (37% of the
32M rows) and helped (+20..+48). Tasks 114-116 then showed that dropping positions
costs strength (hard positions are worth learning). YaneuraOu's official distillation
pipeline does not drop them: `qsearch_psv` replaces each position by the leaf of its
qsearch PV and keeps the original (search-backed) label, negated when the PV has an
odd number of plies (yaneuraou-search.cpp `qsearch_psv_record`). tanuki- measured no
qsearch at -116 (docs/research_survey_2026-10-07.md §3.2). The NNUE is only asked to
evaluate quiet leaves in play (the search runs its own qsearch first), so a quiet leaf
with the value the teacher found by reading through the exchange is the cleaner sample.

Checks before pre-registration (100,000 rows, adopted eval driving the qsearch):
41.2% of rows are replaced (79% of the rows Task 106 drops, 19% of those it keeps),
27.4% flip side to move, max leaf ply 15, 0 decode errors / illegal PVs, outputs
identical for 1 and 4 workers (the hash does not change the leaves), 0 replaced
leaves in check (of 20,000), and features recomputed from the psv round trip match
the cache on 5,000 unchanged rows.

PRE-REGISTERED (committed before the run; no result is seen):
  * qsearch: YaneuraOu `qsearch_psv` with the adopted eval `task109_student_64m_e24`,
    4 workers, hash 16 MB cleared per file. Labels are inherited (search labels of
    `work/task104/rescore_search`, negated on odd PVs); nothing is relabelled. The
    gikou evals and side-to-move results in the cache are negated the same way. Val
    rows are NOT replaced (val loss stays comparable with the control).
  * arm A (decides): every row replaced, no row mask (31,950,000 rows per epoch).
  * arm B (descriptive): replaced rows AND the Task 106 mask (only the quieting effect
    on the rows Task 106 keeps; 20,117,842 rows per epoch).
  * recipe = Task 106: `caches/shared_32m` layout, 098 NNUE_ARGS + --augment-mirror,
    12 epochs. Control = `task106_student_filter` (standard-frame games reused).
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, 100,000 nodes, seeds 20272001-08, 800 games per anchor), pooled.
  * decision (arm A - control): CI low > 0 -> QS_REPLACE_HELPS (next: the same at
    64M, then the relabel variant) / CI high < 0 -> QS_REPLACE_HURTS / otherwise
    QS_REPLACE_NEUTRAL. Arm B: (B - control) and (A - B) reported, not decided.
  * gates: qsearch decode errors = illegal PVs = 0, replaced fraction within
    [0.25, 0.60], feature round trip exact on sampled unchanged rows, 0cp GREEN for
    each arm, no illegal move / engine failure, openings identical.

Interruptible (scripts/job.sh pause|resume task119).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

import qs_replace as qs  # noqa: E402
import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402
from trainer import kp  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402

TAG = "task119"
CONTROL = "task106_student_filter"
QS_EVAL = "task109_student_64m_e24"
CACHE, LABELS, BASE_MASK = "caches/shared_32m", "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
QS_CACHE = "caches/shared_32m_qs119"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
PART_ROWS = 1_000_000
REPLACED_RANGE = (0.25, 0.60)
ARMS = {"A": None, "B": BASE_MASK}   # arm -> row mask


def source_groups(data):
    meta = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))
    primary, _ = shard_files(meta["run_dir"])
    return [(meta["run_dir"], s) for s in primary[:-len(meta["val_shards"])]] + \
           [(rd, s) for rd in meta["extra_runs"] for s in shard_files(rd)[0]]


def build(data, work, yane):
    """qsearch-replace every train row; writes QS_CACHE and work/labels. Returns stats."""
    out_cache = data / QS_CACHE
    if (out_cache / "meta.json").exists():
        return json.loads((out_cache / "meta.json").read_text(encoding="utf-8"))["qsearch_replace"]
    psv_dir = work / "psv"
    psv_dir.mkdir(parents=True, exist_ok=True)

    parts, cur, buf, n_buf = [], 0, [], 0
    for rd, shard in source_groups(data):
        buf.append((rd, shard))
        n_buf += shard["count"]
        if n_buf >= PART_ROWS:
            parts.append((cur, cur + n_buf, buf))
            cur, buf, n_buf = cur + n_buf, [], 0
    if buf:
        parts.append((cur, cur + n_buf, buf))
        cur += n_buf
    if cur != LIMIT_TRAIN:
        raise RuntimeError(f"shards cover {cur} rows != {LIMIT_TRAIN}")

    engine, stats_all = None, []
    try:
        for k, (lo, hi, shards) in enumerate(parts):
            inp, outp = psv_dir / f"in_{k:03d}.psv", psv_dir / f"out_{k:03d}.psv"
            if not inp.exists():
                ents = np.concatenate([np.fromfile(os.path.join(rd, s["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
                                       for rd, s in shards])
                tmp = inp.with_suffix(".tmp")
                qs.hcpe_to_psv(ents).tofile(tmp)
                os.replace(tmp, inp)
            stat_path = psv_dir / f"out_{k:03d}.json"
            if not stat_path.exists():
                if engine is None:
                    engine = qs.QSearchEngine(yane, data / "eval_dirs" / QS_EVAL, threads=4, hash_mb=16)
                tmp = psv_dir / f"out_{k:03d}.tmp"
                st = engine.run(inp, tmp)
                if st["decode_errors"] or st["illegal_pv"] or st["records"] != hi - lo:
                    raise RuntimeError(f"part {k}: {st}")
                os.replace(tmp, outp)
                stat_path.write_text(json.dumps(st), encoding="utf-8")
            stats_all.append(json.loads(stat_path.read_text(encoding="utf-8")))
            print(f"T119_QS part {k + 1}/{len(parts)} rows={hi - lo} replaced={stats_all[-1]['replaced']}", flush=True)
            if t106.stop_requested():
                t106.pause_here(f"qsearch part {k + 1}/{len(parts)}")
    finally:
        if engine is not None:
            engine.close()

    print("=== T119 import (features of the replaced rows, negated labels) ===", flush=True)
    tmp_dir = data / (QS_CACHE + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    old_idx = np.load(data / CACHE / "train_indices.npy", mmap_mode="r")
    new_idx = np.lib.format.open_memmap(tmp_dir / "train_indices.npy", mode="w+", dtype=old_idx.dtype,
                                        shape=old_idx.shape)
    evals = np.load(data / CACHE / "train_evals.npy").copy()
    wdl = np.load(data / CACHE / "train_wdl.npy").copy()
    labels = np.load(data / LABELS / "train_dlshogi_evals.npy").copy()
    changed_all = np.zeros(LIMIT_TRAIN, dtype=bool)
    flip_all = np.zeros(LIMIT_TRAIN, dtype=bool)
    board = cshogi.Board()
    mismatch = checked = leaf_in_check = 0
    rng = np.random.default_rng(20271007)
    for k, (lo, hi, _) in enumerate(parts):
        inp = np.fromfile(psv_dir / f"in_{k:03d}.psv", dtype=qs.PSV)
        out = np.fromfile(psv_dir / f"out_{k:03d}.psv", dtype=qs.PSV)
        changed, flip = qs.leaf_rows(inp, out)
        new_idx[lo:hi] = old_idx[lo:hi]
        for i in rng.choice(np.flatnonzero(~changed), size=min(200, int((~changed).sum())), replace=False):
            board.set_psfen(inp[i]["sfen"])
            mismatch += not np.array_equal(np.asarray(kp.canonical_shared_row(board), dtype=np.int32), old_idx[lo + i])
            checked += 1
        for i in np.flatnonzero(changed):
            board.set_psfen(out[i]["sfen"])
            leaf_in_check += board.is_check()
            row = kp.canonical_shared_row(board)
            if len(row) != new_idx.shape[1]:
                raise AssertionError(f"row {lo + i}: {len(row)} features")
            new_idx[lo + i] = row
        sign = np.where(flip, -1, 1)
        evals[lo:hi] = (evals[lo:hi].astype(np.int32) * sign).astype(evals.dtype)
        wdl[lo:hi] = (wdl[lo:hi].astype(np.int32) * sign).astype(wdl.dtype)
        labels[lo:hi] = (labels[lo:hi].astype(np.int32) * sign).astype(labels.dtype)
        changed_all[lo:hi], flip_all[lo:hi] = changed, flip
        print(f"T119_IMPORT part {k + 1}/{len(parts)} changed={changed.mean():.4f} flip={flip.mean():.4f}", flush=True)
    new_idx.flush()
    del new_idx
    np.save(tmp_dir / "train_evals.npy", evals)
    np.save(tmp_dir / "train_wdl.npy", wdl)
    for name in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        os.symlink(data / CACHE / name, tmp_dir / name)

    base = np.load(data / BASE_MASK)
    stats = {"rows": LIMIT_TRAIN, "replaced_frac": float(changed_all.mean()), "flip_frac": float(flip_all.mean()),
             "replaced_frac_of_task106_kept": float(changed_all[base].mean()),
             "replaced_frac_of_task106_dropped": float(changed_all[~base].mean()),
             "max_leaf_ply": max(s["max_leaf_ply"] for s in stats_all),
             "decode_errors": sum(s["decode_errors"] for s in stats_all),
             "illegal_pv": sum(s["illegal_pv"] for s in stats_all),
             "roundtrip_checked": checked, "roundtrip_mismatch": mismatch, "leaf_in_check": int(leaf_in_check),
             "qs_eval": QS_EVAL}
    np.save(work / "changed.npy", changed_all)
    np.save(work / "flip.npy", flip_all)
    lab_dir = work / "labels"
    shutil.rmtree(lab_dir, ignore_errors=True)
    lab_dir.mkdir(parents=True)
    np.save(lab_dir / "train_dlshogi_evals.npy", labels)
    os.symlink(data / LABELS / "val_dlshogi_evals.npy", lab_dir / "val_dlshogi_evals.npy")
    (lab_dir / "meta.json").write_text(json.dumps({"source": LABELS, "negated_on_odd_qsearch_pv": True, **stats},
                                                  indent=2), encoding="utf-8")
    meta = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))
    meta["qsearch_replace"] = stats
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_dir, out_cache)
    for p in psv_dir.glob("*.psv"):
        p.unlink()   # regenerable in minutes; the json stats stay
    print("T119_BUILD " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def main():
    repo, data, logs, _binbo, yane, _anchors = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK,
                 data / "eval_dirs" / CONTROL / "nn.bin", data / "eval_dirs" / QS_EVAL / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T119_PREFLIGHT_OK head={head} control={CONTROL} qs_eval={QS_EVAL}", flush=True)
    failures = []

    print("=== T119 STAGE build (qsearch_psv on every train row) ===", flush=True)
    build_stats = build(data, work, yane)
    if build_stats["decode_errors"] or build_stats["illegal_pv"]:
        failures.append("qsearch decode errors / illegal PVs")
    if not REPLACED_RANGE[0] <= build_stats["replaced_frac"] <= REPLACED_RANGE[1]:
        failures.append(f"replaced fraction {build_stats['replaced_frac']:.3f} outside {REPLACED_RANGE}")
    if build_stats["roundtrip_mismatch"]:
        failures.append(f"feature round trip mismatch {build_stats['roundtrip_mismatch']}")

    ctrl, bad = std_frame.measure(CONTROL, env)
    failures += bad
    ctrl_hist = json.loads((data / "models" / CONTROL / "history.json").read_text(encoding="utf-8"))
    results = {}
    for arm, mask in ARMS.items():
        name = f"{TAG}_student_qsrep{arm.lower()}"
        model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
        if not (model_dir / "history.json").exists():
            print(f"=== T119 STAGE train arm {arm} (mask={mask}) ===", flush=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
            mask_args = ["--row-mask", data / mask] if mask else []
            t106.run([torch_py, "-m", "trainer.train_nnue", data / QS_CACHE, "--out", model_dir, *NNUE_ARGS,
                      "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", work / "labels",
                      "--limit-train", LIMIT_TRAIN, *mask_args, "--checkpoint-minutes", "10", "--resume", *stop_args],
                     logs / f"{TAG}_{arm}_student.log", cwd=repo, env=env)
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                      "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_{arm}_export.log", cwd=repo, env=env)
            t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                      "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                     logs / f"{TAG}_{arm}_yane.log", cwd=repo, env=env)
        gate = logs / f"{TAG}_{arm}_0cp.log"
        if not gate.exists():
            t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
        text = gate.read_text(encoding="utf-8")
        if "RED" in text or text.count("GREEN") < 2:
            failures.append(f"0cp gate RED arm {arm}")
        hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        print(f"T119_VAL arm={arm} best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f} "
              f"control={ctrl_hist['best_val_loss']:.1f}", flush=True)
        rows, bad = std_frame.measure(name, env)
        failures += bad
        pooled, per, bad = std_frame.delta(rows, ctrl, f"(arm {arm} - task106)")
        failures += bad
        results[arm] = {"eval": name, "rows": rows, "pooled": pooled, "per_anchor": per,
                        "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")}}
        print(f"T119_FRAME arm={arm} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        if arm == "A":
            lo, hi = pooled["ci95"]
            verdict = ("INVALID" if failures else "QS_REPLACE_HELPS" if lo > 0 else "QS_REPLACE_HURTS" if hi < 0
                       else "QS_REPLACE_NEUTRAL")
            print(f"T119_A_VERDICT verdict={verdict} failures={failures}", flush=True)

    ab, ab_per, bad = std_frame.delta(results["A"]["rows"], results["B"]["rows"], "(arm A - arm B)")
    failures += bad
    print(f"T119_A_MINUS_B pooled={ab['delta_elo']:+.2f} ci={[round(v, 2) for v in ab['ci95']]}", flush=True)
    lo, hi = results["A"]["pooled"]["ci95"]
    verdict = ("INVALID" if failures else "QS_REPLACE_HELPS" if lo > 0 else "QS_REPLACE_HURTS" if hi < 0
               else "QS_REPLACE_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 119, "git": {"head": head, "pushed": False}, "build": build_stats,
                               "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "arms": {a: {k: v for k, v in r.items() if k != "rows"} for a, r in results.items()},
                               "a_minus_b": {"pooled": ab, "per_anchor": ab_per},
                               "failures": failures, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T119_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
