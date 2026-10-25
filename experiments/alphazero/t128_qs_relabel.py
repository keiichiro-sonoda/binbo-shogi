#!/usr/bin/env python3
"""Task 128: qsearch replacement with RELABELLED leaves (32M, current recipe).

Task 119 replaced every train row by the leaf Q of its qsearch PV (YaneuraOu
`qsearch_psv`, 41% of rows changed) but kept the label of the original position P.
Replacing alone was neutral (arm B, Task 106 mask: +5.5), while also returning the rows
Task 106 drops hurt (A - B = -26.4 [-50.3, -2.6]). The suspected cause is the inherited
label: those rows are "gikou's best move is a capture" positions, and the label of P from
WS-T + MCTS 8 need not be the value of the quiet leaf Q. Here every changed row gets a
fresh label for Q itself from the same teacher (WS-T + Gumbel MCTS S=8, cp =
1200 atanh(v), Task 104's labeller), so position and label agree.

PRE-REGISTERED (committed before the run; no result is seen):
  * leaves: Task 119's replacement is regenerated (same YaneuraOu binary, same qsearch
    eval `task109_student_64m_e24`, 4 workers, hash 16 MB cleared per file; Task 119
    showed the leaves do not depend on threads or hash) and must be identical to Task
    119 (changed / flip masks equal on every row, features of 200 sampled changed rows
    per part equal to `caches/shared_32m_qs119`). Training reads that cache.
  * labels: changed rows = the teacher's S=8 search value of Q (side to move of Q);
    unchanged rows = `work/task104/rescore_search` as before. Val is not replaced (val
    loss stays comparable). Batch 128 as Task 104.
  * recipe = the control's: `task125_student_factor` (32M, Task 106 mask, search labels,
    098 NNUE_ARGS with lr 6e-5, --factorize, --augment-mirror, 12 epochs).
  * arm C (decides): replaced + relabelled, NO row mask (31,950,000 rows per epoch).
  * arm D (descriptive): replaced + relabelled + the Task 106 mask (20,117,842 rows).
  * yardstick = Task 127 (tanuki- @20,000 / AobaNNUE @10,000, 100,000 nodes, standard
    seeds 20272001-08, 800 games per anchor), pooled, paired on shared openings.
  * decision (C - control): CI low > 0 -> QS_RELABEL_HELPS (next: the same at 64M vs
    the adopted eval) / CI high < 0 -> QS_RELABEL_HURTS / otherwise QS_RELABEL_NEUTRAL.
    Descriptive: D - control, C - D. (Task 119's arms used the old recipe and the old
    yardstick, so C - A and D - B are not measured.)
  * gates: leaves identical to Task 119, labeller sign oracle, finite labels, corr(new,
    inherited) > 0.5 on changed rows (a sign error would make it negative), 0cp GREEN
    for each arm, no illegal move / engine failure, openings identical.
  * descriptive: label shift |new - inherited| on changed rows, split by Task 106 kept /
    dropped; val losses, best epochs.

Order: relabel (GPU, JAX, subprocess in venv_pgx) -> train C -> frames control, C
(verdict) -> train D -> frame D. Interruptible (scripts/job.sh pause|resume task128).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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

TAG = "task128"
CONTROL = "task125_student_factor"
QS_EVAL = "task109_student_64m_e24"   # Task 119's qsearch eval (the leaves must not change)
CACHE, QS_CACHE = "caches/shared_32m", "caches/shared_32m_qs119"
LABELS, BASE_MASK = "work/task104/rescore_search", "work/task106/mask/train_keep.npy"
T119 = "work/task119"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
PART_ROWS = 1_000_000
SIMS, BATCH = 8, 128
WST = Path.home() / "task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt"
PGX_PY = Path.home() / "task074_frameworks/venv_pgx/bin/python"
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
ARMS = {"C": None, "D": BASE_MASK}
CORR_MIN = 0.5


def parts_of(data):
    """Task 119's partition of the train rows: (lo, hi, [(run_dir, shard), ...])."""
    meta = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))
    primary, _ = shard_files(meta["run_dir"])
    groups = [(meta["run_dir"], s) for s in primary[:-len(meta["val_shards"])]] + \
             [(rd, s) for rd in meta["extra_runs"] for s in shard_files(rd)[0]]
    parts, cur, buf, n_buf = [], 0, [], 0
    for rd, shard in groups:
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
    return parts


def relabel(data, work, yane):
    """GPU stage (venv_pgx): regenerate the leaves, check them against Task 119, label Q."""
    from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp
    from m2_usi import load_model

    forward, model, _ = load_model(str(WST))
    sign_oracle(forward, model, SIMS)
    search = build_batched_search(forward, model, SIMS)
    changed_119 = np.load(data / T119 / "changed.npy", mmap_mode="r")
    flip_119 = np.load(data / T119 / "flip.npy", mmap_mode="r")
    qs_idx = np.load(data / QS_CACHE / "train_indices.npy", mmap_mode="r")
    psv_dir, part_dir = work / "psv", work / "parts"
    psv_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = parts_of(data)
    board = cshogi.Board()
    rng = np.random.default_rng(20271010)
    engine, started, done_rows = None, time.time(), 0
    try:
        for k, (lo, hi, shards) in enumerate(parts):
            part = part_dir / f"part_{k:03d}.npz"
            if part.exists():
                continue
            inp, outp = psv_dir / f"in_{k:03d}.psv", psv_dir / f"out_{k:03d}.psv"
            if not inp.exists():
                ents = np.concatenate([np.fromfile(os.path.join(rd, s["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
                                       for rd, s in shards])
                qs.hcpe_to_psv(ents).tofile(inp.with_suffix(".tmp"))
                os.replace(inp.with_suffix(".tmp"), inp)
            if not outp.exists():
                if engine is None:
                    engine = qs.QSearchEngine(yane, data / "eval_dirs" / QS_EVAL, threads=4, hash_mb=16)
                st = engine.run(inp, psv_dir / f"out_{k:03d}.tmp")
                if st["decode_errors"] or st["illegal_pv"] or st["records"] != hi - lo:
                    raise RuntimeError(f"part {k}: {st}")
                os.replace(psv_dir / f"out_{k:03d}.tmp", outp)
            a = np.fromfile(inp, dtype=qs.PSV)
            b = np.fromfile(outp, dtype=qs.PSV)
            changed, flip = qs.leaf_rows(a, b)
            if not (np.array_equal(changed, changed_119[lo:hi]) and np.array_equal(flip, flip_119[lo:hi])):
                raise AssertionError(f"part {k}: leaves differ from Task 119 "
                                     f"(changed {int((changed != changed_119[lo:hi]).sum())}, "
                                     f"flip {int((flip != flip_119[lo:hi]).sum())})")
            rows = np.flatnonzero(changed)
            for i in rng.choice(rows, size=min(200, len(rows)), replace=False):
                board.set_psfen(b[i]["sfen"])
                if not np.array_equal(np.asarray(kp.canonical_shared_row(board), dtype=np.int32), qs_idx[lo + i]):
                    raise AssertionError(f"part {k} row {lo + i}: features differ from {QS_CACHE}")
            sfens = []
            for i in rows:
                board.set_psfen(b[i]["sfen"])
                sfens.append(board.sfen())
            values, _q, _raw, _mv = score_sfens(search, sfens, BATCH)
            if not np.isfinite(values).all():
                raise AssertionError(f"part {k}: non-finite labels")
            tmp = part.with_name(part.name + ".tmp.npz")
            np.savez(tmp, rows=(rows + lo).astype(np.int64), cp=value_to_cp(values).astype(np.int16))
            os.replace(tmp, part)
            inp.unlink()
            outp.unlink()
            done_rows += len(rows)
            rate = done_rows / (time.time() - started)
            print(f"T128_RELABEL part {k + 1}/{len(parts)} changed={len(rows)} rate={rate:.0f}/s", flush=True)
            if t106.stop_requested():
                t106.pause_here(f"relabel part {k + 1}/{len(parts)}")
    finally:
        if engine is not None:
            engine.close()
    return 0


def assemble(data, work):
    """Write work/task128/labels (train: relabelled Q rows over the search labels) and stats."""
    lab_dir = work / "labels"
    if (lab_dir / "meta.json").exists():
        return json.loads((lab_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    labels = np.load(data / LABELS / "train_dlshogi_evals.npy").copy()
    inherited = np.load(data / T119 / "labels/train_dlshogi_evals.npy", mmap_mode="r")
    changed = np.load(data / T119 / "changed.npy")
    base = np.load(data / BASE_MASK)
    seen = np.zeros(LIMIT_TRAIN, dtype=bool)
    for k in range(len(parts_of(data))):
        z = np.load(work / "parts" / f"part_{k:03d}.npz")
        labels[z["rows"]] = z["cp"]
        seen[z["rows"]] = True
    if not np.array_equal(seen, changed):
        raise AssertionError("relabelled rows != Task 119 changed rows")
    new = labels[changed].astype(np.float64)
    old = np.asarray(inherited)[changed].astype(np.float64)
    keep = base[changed]
    shift = np.abs(new - old)
    stats = {"changed_rows": int(changed.sum()),
             "corr_new_inherited": float(np.corrcoef(new, old)[0, 1]),
             "median_abs_shift": float(np.median(shift)),
             "median_abs_shift_task106_kept": float(np.median(shift[keep])),
             "median_abs_shift_task106_dropped": float(np.median(shift[~keep])),
             "mean_abs_new": float(np.abs(new).mean()), "mean_abs_inherited": float(np.abs(old).mean())}
    tmp = work / "labels.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "train_dlshogi_evals.npy", labels)
    os.symlink(data / LABELS / "val_dlshogi_evals.npy", tmp / "val_dlshogi_evals.npy")
    (tmp / "meta.json").write_text(json.dumps({"source": LABELS, "leaves": QS_CACHE, "labeller": f"WS-T S={SIMS}",
                                               "stats": stats}, indent=2), encoding="utf-8")
    os.replace(tmp, lab_dir)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel-only", action="store_true", help="internal: the GPU stage in venv_pgx")
    args = parser.parse_args()
    repo, data, logs, _binbo, yane, _anchors = std_frame.paths()
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    if args.relabel_only:
        return relabel(data, work, yane)

    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for path in (data / QS_CACHE / "meta.json", data / LABELS / "meta.json", data / BASE_MASK,
                 data / T119 / "changed.npy", data / T119 / "flip.npy", data / T119 / "labels/train_dlshogi_evals.npy",
                 data / "eval_dirs" / CONTROL / "nn.bin", data / "eval_dirs" / QS_EVAL / "nn.bin", WST, PGX_PY):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T128_PREFLIGHT_OK head={head} control={CONTROL} anchors={std_frame.ANCHORS}", flush=True)
    failures = []

    if not (work / "labels/meta.json").exists():
        print("=== T128 STAGE relabel (qsearch leaves, WS-T S=8) ===", flush=True)
        t106.run([PGX_PY, os.path.realpath(__file__), "--relabel-only"], logs / f"{TAG}_relabel.log",
                 cwd=repo, env=env)
    stats = assemble(data, work)
    print("T128_LABELS " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    if not stats["corr_new_inherited"] > CORR_MIN:
        failures.append(f"corr(new, inherited) {stats['corr_new_inherited']:.3f} <= {CORR_MIN}")

    ctrl_hist = json.loads((data / "models" / CONTROL / "history.json").read_text(encoding="utf-8"))
    results, ctrl = {}, None
    for arm, mask in ARMS.items():
        name = f"{TAG}_student_qsrel{arm.lower()}"
        model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
        if not (model_dir / "history.json").exists():
            print(f"=== T128 STAGE train arm {arm} (mask={mask}) ===", flush=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
            mask_args = ["--row-mask", data / mask] if mask else []
            t106.run([torch_py, "-m", "trainer.train_nnue", data / QS_CACHE, "--out", model_dir, *LR_ARGS,
                      "--factorize", "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                      "--eval-npy", work / "labels", "--limit-train", LIMIT_TRAIN, *mask_args,
                      "--checkpoint-minutes", "10", "--resume", *stop_args],
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
        print(f"T128_VAL arm={arm} best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f} "
              f"control={ctrl_hist['best_val_loss']:.1f}", flush=True)
        if ctrl is None:
            ctrl, bad = std_frame.measure(CONTROL, env)
            failures += bad
        rows, bad = std_frame.measure(name, env)
        failures += bad
        pooled, per, bad = std_frame.delta(rows, ctrl, f"(arm {arm} - control)")
        failures += bad
        results[arm] = {"eval": name, "rows": rows, "pooled": pooled, "per_anchor": per,
                        "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")}}
        print(f"T128_FRAME arm={arm} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        if arm == "C":
            lo, hi = pooled["ci95"]
            verdict = ("INVALID" if failures else "QS_RELABEL_HELPS" if lo > 0 else "QS_RELABEL_HURTS" if hi < 0
                       else "QS_RELABEL_NEUTRAL")
            print(f"T128_C_VERDICT verdict={verdict} failures={failures}", flush=True)

    cd, cd_per, bad = std_frame.delta(results["C"]["rows"], results["D"]["rows"], "(arm C - arm D)")
    failures += bad
    print(f"T128_C_MINUS_D pooled={cd['delta_elo']:+.2f} ci={[round(v, 2) for v in cd['ci95']]}", flush=True)
    lo, hi = results["C"]["pooled"]["ci95"]
    verdict = ("INVALID" if failures else "QS_RELABEL_HELPS" if lo > 0 else "QS_RELABEL_HURTS" if hi < 0
               else "QS_RELABEL_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 128, "git": {"head": head, "pushed": False}, "labels": stats,
                               "anchors": std_frame.ANCHORS, "control_best_val_loss": ctrl_hist.get("best_val_loss"),
                               "arms": {a: {k: v for k, v in r.items() if k != "rows"} for a, r in results.items()},
                               "c_minus_d": {"pooled": cd, "per_anchor": cd_per},
                               "failures": failures, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T128_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
