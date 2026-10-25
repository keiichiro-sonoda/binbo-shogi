#!/usr/bin/env python3
"""Task 129: relabelled qsearch leaves at 64M, against the adopted eval.

Task 128 (32M): replacing every train row by its qsearch leaf Q and labelling Q with the
teacher (no row mask) beat the factorizer control by +63.9 [+40.9, +86.8]; the same with
the Task 106 mask gave +24.3 (C - D = +39.6). This is the 64M step of its decision table.

`caches/shared_64m` = `shared_32m` + `run32m_g` (train indices, evals and the Task 107
search labels of the first 31,950,000 rows are identical to the 32M ones; checked before
pre-registration). So the first half reuses Task 119's leaves (`shared_32m_qs119`) and
Task 128's labels; only run32m_g's 32,000,000 rows are replaced and relabelled here.

PRE-REGISTERED (committed before the run; no result is seen):
  * leaves for run32m_g: YaneuraOu `qsearch_psv` with `task109_student_64m_e24` (Task
    119's qsearch eval), 4 workers, hash 16 MB cleared per file; evals / wdl negated on
    odd PVs as Task 119. Labels: changed rows = WS-T + Gumbel MCTS S=8 value of Q (Task
    128's labeller, batch 128); unchanged rows = `work/task107/rescore_search`. Val is
    not replaced.
  * candidate = 64M, no row mask (63,950,000 rows per epoch), 098 NNUE_ARGS with lr 6e-5,
    --factorize, --augment-mirror, --mmap, 12 epochs (Task 126: 12 vs 24 epochs +2.5).
  * PRIMARY = candidate - adopted (`task126_student_64m_factor_e24`, 24 epochs) on the
    Task 127 yardstick (tanuki- @20,000 / AobaNNUE @10,000, 100,000 nodes, standard seeds
    20272001-08, 800 games per anchor; the adopted eval's games are reused), pooled:
      CI low > 0 -> confirm on FRESH seeds 20272501-08 (both evals, 800 games per anchor
        each): CI low > 0 -> ADOPT_CANDIDATE (the swap waits for approval) /
        otherwise NOT_CONFIRMED
      CI high < 0 -> QS_RELABEL_64M_WORSE / otherwise QS_RELABEL_64M_NEUTRAL
  * gates: prefix identity (32M vs 64M evals and labels on every row, indices on a
    sample), qsearch decode errors = illegal PVs = 0, run32m_g replaced fraction within
    [0.25, 0.60], features of sampled unchanged run32m_g rows equal the 64M cache,
    labeller sign oracle, finite labels, corr(new, inherited) > 0.5 on run32m_g's
    changed rows, 0cp GREEN, no illegal move / engine failure, openings identical.
  * descriptive: replaced fraction, label shift by Task 107 kept / dropped, val loss,
    best epoch.

Order: replace + relabel run32m_g (GPU, JAX, subprocess in venv_pgx) -> build the cache
and labels -> train -> frame -> confirm. Interruptible (scripts/job.sh pause|resume task129).
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

TAG = "task129"
ADOPTED = "task126_student_64m_factor_e24"
QS_EVAL = "task109_student_64m_e24"
CACHE32, CACHE64 = "caches/shared_32m", "caches/shared_64m"
QS32, QS64 = "caches/shared_32m_qs119", "caches/shared_64m_qs129"
LABELS32_REL = "work/task128/labels"           # Task 104 labels with Task 128's relabelled leaves
LABELS64, MASK64 = "work/task107/rescore_search", "work/task107/mask/train_keep.npy"
T119 = "work/task119"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
N32, N64 = 31_950_000, 63_950_000
NEW_RUN = "run32m_g"
PART_ROWS = 1_000_000
SIMS, BATCH = 8, 128
WST = Path.home() / "task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt"
PGX_PY = Path.home() / "task074_frameworks/venv_pgx/bin/python"
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
EPOCHS = 12
REPLACED_RANGE = (0.25, 0.60)
CORR_MIN = 0.5
CONFIRM_SEEDS = tuple(20272501 + i for i in range(8))
CONFIRM_PREFIX = "conf129"
CAND = f"{TAG}_student_64m_qsrel_e12"


def new_parts(data):
    """Partition of run32m_g's rows (64M cache rows N32..N64): (lo, hi, [(run_dir, shard), ...])."""
    meta = json.loads((data / CACHE64 / "meta.json").read_text(encoding="utf-8"))
    rd = meta["extra_runs"][-1]
    if Path(rd).name != NEW_RUN:
        raise RuntimeError(f"last extra run is {rd}, expected {NEW_RUN}")
    parts, cur, buf, n_buf = [], N32, [], 0
    for shard in shard_files(rd)[0]:
        buf.append((rd, shard))
        n_buf += shard["count"]
        if n_buf >= PART_ROWS:
            parts.append((cur, cur + n_buf, buf))
            cur, buf, n_buf = cur + n_buf, [], 0
    if buf:
        parts.append((cur, cur + n_buf, buf))
        cur += n_buf
    if cur != N64:
        raise RuntimeError(f"{NEW_RUN} ends at row {cur} != {N64}")
    return parts


def relabel(data, work, yane):
    """GPU stage (venv_pgx): qsearch-replace run32m_g, compute leaf features, label the leaves."""
    from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp
    from m2_usi import load_model

    forward, model, _ = load_model(str(WST))
    sign_oracle(forward, model, SIMS)
    search = build_batched_search(forward, model, SIMS)
    idx64 = np.load(data / CACHE64 / "train_indices.npy", mmap_mode="r")
    psv_dir, part_dir = work / "psv", work / "parts"
    psv_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = new_parts(data)
    board = cshogi.Board()
    rng = np.random.default_rng(20271011)
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
            mismatch = 0
            for i in rng.choice(np.flatnonzero(~changed), size=min(200, int((~changed).sum())), replace=False):
                board.set_psfen(a[i]["sfen"])
                mismatch += not np.array_equal(np.asarray(kp.canonical_shared_row(board), dtype=np.int32),
                                               idx64[lo + i])
            rows = np.flatnonzero(changed)
            feats = np.empty((len(rows), idx64.shape[1]), dtype=idx64.dtype)
            sfens, in_check = [], 0
            for j, i in enumerate(rows):
                board.set_psfen(b[i]["sfen"])
                in_check += board.is_check()
                feats[j] = kp.canonical_shared_row(board)
                sfens.append(board.sfen())
            values, _q, _raw, _mv = score_sfens(search, sfens, BATCH)
            if not np.isfinite(values).all():
                raise AssertionError(f"part {k}: non-finite labels")
            tmp = part.with_name(part.name + ".tmp.npz")
            np.savez(tmp, rows=(rows + lo).astype(np.int64), cp=value_to_cp(values).astype(np.int16), feats=feats,
                     flip=flip, mismatch=mismatch, in_check=in_check)
            os.replace(tmp, part)
            inp.unlink()
            outp.unlink()
            done_rows += len(rows)
            print(f"T129_RELABEL part {k + 1}/{len(parts)} rows={hi - lo} changed={len(rows)} "
                  f"rate={done_rows / (time.time() - started):.0f}/s mismatch={mismatch}", flush=True)
            if t106.stop_requested():
                t106.pause_here(f"relabel part {k + 1}/{len(parts)}")
    finally:
        if engine is not None:
            engine.close()
    return 0


def build(data, work, parts=None, out_cache=QS64):
    """Write the replaced 64M cache (indices / evals / wdl) and work/task129/labels. Returns stats.
    `parts` / `out_cache` are overridden only by the smoke test (rows N32..parts[-1].hi)."""
    lab_dir = work / "labels"
    if (data / out_cache / "meta.json").exists() and (lab_dir / "meta.json").exists():
        return json.loads((lab_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    parts = parts or new_parts(data)
    end = parts[-1][1]
    tmp_dir = data / (out_cache + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    src64 = np.load(data / CACHE64 / "train_indices.npy", mmap_mode="r")
    src32 = np.load(data / QS32 / "train_indices.npy", mmap_mode="r")
    idx = np.lib.format.open_memmap(tmp_dir / "train_indices.npy", mode="w+", dtype=src64.dtype, shape=(end, src64.shape[1]))
    step = 2_000_000
    for s in range(0, N32, step):
        idx[s:min(s + step, N32)] = src32[s:min(s + step, N32)]
    for s in range(N32, end, step):
        idx[s:min(s + step, end)] = src64[s:min(s + step, end)]
    evals = np.load(data / CACHE64 / "train_evals.npy")[:end].copy()
    wdl = np.load(data / CACHE64 / "train_wdl.npy")[:end].copy()
    evals[:N32] = np.load(data / QS32 / "train_evals.npy")
    wdl[:N32] = np.load(data / QS32 / "train_wdl.npy")
    labels = np.load(data / LABELS64 / "train_dlshogi_evals.npy")[:end].copy()
    labels[:N32] = np.load(data / LABELS32_REL / "train_dlshogi_evals.npy")
    inherited = np.load(data / LABELS64 / "train_dlshogi_evals.npy", mmap_mode="r")
    changed = np.zeros(end, dtype=bool)
    changed[:N32] = np.load(data / T119 / "changed.npy")
    flip_new = np.zeros(end, dtype=bool)
    mismatch = in_check = 0
    new_cp, old_cp = [], []
    for k, (lo, hi, _) in enumerate(parts):
        z = np.load(work / "parts" / f"part_{k:03d}.npz")
        rows, flip = z["rows"], z["flip"]
        if len(flip) != hi - lo or (len(rows) and (rows.min() < lo or rows.max() >= hi)):
            raise AssertionError(f"part {k}: rows / flip do not cover [{lo}, {hi})")
        idx[rows] = z["feats"]
        sign = np.where(flip, -1, 1)
        evals[lo:hi] = (evals[lo:hi].astype(np.int32) * sign).astype(evals.dtype)
        wdl[lo:hi] = (wdl[lo:hi].astype(np.int32) * sign).astype(wdl.dtype)
        old = np.asarray(inherited[lo:hi]).astype(np.int32) * sign   # inherited label, as Task 119 would have
        old_cp.append(old[rows - lo])
        new_cp.append(z["cp"].astype(np.int32))
        labels[rows] = z["cp"]
        changed[rows] = True
        flip_new[lo:hi] = flip
        mismatch += int(z["mismatch"])
        in_check += int(z["in_check"])
    idx.flush()
    del idx
    np.save(tmp_dir / "train_evals.npy", evals)
    np.save(tmp_dir / "train_wdl.npy", wdl)
    for name in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        os.symlink(data / CACHE64 / name, tmp_dir / name)
    new, old = np.concatenate(new_cp).astype(np.float64), np.concatenate(old_cp).astype(np.float64)
    mask = np.load(data / MASK64)[:end]
    keep_new = mask[N32:][changed[N32:]]
    shift = np.abs(new - old)
    stats = {"rows": end, "replaced_frac": float(changed.mean()), "replaced_frac_new": float(changed[N32:].mean()),
             "flip_frac_new": float(flip_new[N32:].mean()), "changed_new": int(changed[N32:].sum()),
             "feature_mismatch": mismatch, "leaf_in_check": in_check,
             "corr_new_inherited": float(np.corrcoef(new, old)[0, 1]), "median_abs_shift": float(np.median(shift)),
             "median_abs_shift_task107_kept": float(np.median(shift[keep_new])),
             "median_abs_shift_task107_dropped": float(np.median(shift[~keep_new]))}
    meta = json.loads((data / CACHE64 / "meta.json").read_text(encoding="utf-8"))
    meta["qsearch_replace"] = {**stats, "first_half": QS32, "qs_eval": QS_EVAL}
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(work / "changed.npy", changed)
    ltmp = work / "labels.tmp"
    shutil.rmtree(ltmp, ignore_errors=True)
    ltmp.mkdir(parents=True)
    np.save(ltmp / "train_dlshogi_evals.npy", labels)
    os.symlink(data / LABELS64 / "val_dlshogi_evals.npy", ltmp / "val_dlshogi_evals.npy")
    (ltmp / "meta.json").write_text(json.dumps({"source": LABELS64, "first_half": LABELS32_REL,
                                                "labeller": f"WS-T S={SIMS}", "stats": stats}, indent=2),
                                    encoding="utf-8")
    shutil.rmtree(data / out_cache, ignore_errors=True)
    os.replace(tmp_dir, data / out_cache)
    shutil.rmtree(lab_dir, ignore_errors=True)
    os.replace(ltmp, lab_dir)
    return stats


def prefix_gate(data):
    e32, e64 = np.load(data / CACHE32 / "train_evals.npy"), np.load(data / CACHE64 / "train_evals.npy", mmap_mode="r")
    l32 = np.load(data / "work/task104/rescore_search/train_dlshogi_evals.npy")
    l64 = np.load(data / LABELS64 / "train_dlshogi_evals.npy", mmap_mode="r")
    i32 = np.load(data / CACHE32 / "train_indices.npy", mmap_mode="r")
    i64 = np.load(data / CACHE64 / "train_indices.npy", mmap_mode="r")
    s = np.sort(np.random.default_rng(20271011).choice(N32, 20_000, replace=False))
    return np.array_equal(e32, e64[:N32]) and np.array_equal(l32, l64[:N32]) and np.array_equal(i32[s], i64[s])


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
    for path in (data / CACHE64 / "meta.json", data / QS32 / "meta.json", data / LABELS32_REL / "meta.json",
                 data / LABELS64 / "meta.json", data / MASK64, data / T119 / "changed.npy",
                 data / "eval_dirs" / ADOPTED / "nn.bin", data / "eval_dirs" / QS_EVAL / "nn.bin", WST, PGX_PY):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    if not prefix_gate(data):
        raise RuntimeError("64M prefix differs from 32M")
    print(f"T129_PREFLIGHT_OK head={head} adopted={ADOPTED} anchors={std_frame.ANCHORS} prefix=identical", flush=True)
    failures = []

    if not (work / "labels/meta.json").exists():
        if len(list((work / "parts").glob("part_*.npz"))) < len(new_parts(data)):
            print(f"=== T129 STAGE replace + relabel {NEW_RUN} ===", flush=True)
            t106.run([PGX_PY, os.path.realpath(__file__), "--relabel-only"], logs / f"{TAG}_relabel.log",
                     cwd=repo, env=env)
        print("=== T129 STAGE build cache + labels ===", flush=True)
    stats = build(data, work)
    print("T129_BUILD " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    if stats["feature_mismatch"]:
        failures.append(f"feature round trip mismatch {stats['feature_mismatch']}")
    if not REPLACED_RANGE[0] <= stats["replaced_frac_new"] <= REPLACED_RANGE[1]:
        failures.append(f"replaced fraction {stats['replaced_frac_new']:.3f} outside {REPLACED_RANGE}")
    if not stats["corr_new_inherited"] > CORR_MIN:
        failures.append(f"corr(new, inherited) {stats['corr_new_inherited']:.3f} <= {CORR_MIN}")
    if failures:
        raise RuntimeError(f"build gates failed: {failures}")

    model_dir, eval_dir = data / "models" / CAND, data / "eval_dirs" / CAND
    if not (model_dir / "history.json").exists():
        print(f"=== T129 STAGE train (64M, no mask, lr {LR}, factorizer, {EPOCHS} epochs) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / QS64, "--out", model_dir, *LR_ARGS, "--factorize",
                  "--epochs", EPOCHS, "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", work / "labels", "--limit-train", N64, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
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
    ad_hist = json.loads((data / "models" / ADOPTED / "history.json").read_text(encoding="utf-8"))
    print(f"T129_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f} "
          f"adopted={ad_hist['best_val_loss']:.1f}", flush=True)

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T129_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    def frame(name, seeds=std_frame.SEEDS, prefix="std"):
        rows, bad = std_frame.measure(name, env, seeds=seeds, prefix=prefix)
        failures.extend(bad)
        return rows

    print("=== T129 STAGE standard frame ===", flush=True)
    primary = compare(frame(CAND), frame(ADOPTED), "(64M qs relabel e12 - adopted)")
    lo, hi = primary["pooled"]["ci95"]
    confirm = None
    if failures:
        verdict = "INVALID"
    elif lo > 0:
        print(f"=== T129 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
        confirm = compare(frame(CAND, CONFIRM_SEEDS, CONFIRM_PREFIX), frame(ADOPTED, CONFIRM_SEEDS, CONFIRM_PREFIX),
                          "confirm (64M qs relabel e12 - adopted)")
        verdict = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                   else "NOT_CONFIRMED")
    elif hi < 0:
        verdict = "QS_RELABEL_64M_WORSE"
    else:
        verdict = "QS_RELABEL_64M_NEUTRAL"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 129, "git": {"head": head, "pushed": False}, "build": stats,
                               "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "adopted_best_val_loss": ad_hist.get("best_val_loss"),
                               "primary": primary, "confirm": confirm, "failures": failures, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"T129_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
