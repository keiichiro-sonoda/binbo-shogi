#!/usr/bin/env python3
"""Task 100: does a search-backed value label teach a better NNUE than the raw one?

Task 099 closed the last carriage path of the loop (d32 ~ 0: a +26 Elo policy
does not change the number search settles on). Its handoff (a) noted that the
one thing gold could NOT judge is whether "search moves the value AWAY from the
gikou manifold" is good or bad for a student -- gold is itself a gikou
distillate. The only instrument that does not go through gold is the Task 084
student test: train an NNUE on the labels and let the students play.

099 also showed the answer does not need the loop net (d32 ~ 0), so the
labeller here is WS-T + search, S=32 (the value 099 selected on mechanism
only). This is a question about AZ SEARCH LABELS in general, not about the loop.

WHAT IS COMPARED (one variable = the teacher label family):
    raw    : WS-T raw value head       -> cp = 1200*atanh(v)   (Task 098's ① labels,
                                          reused byte-for-byte from work/task098/rescore_wst)
    search : WS-T + Gumbel MCTS S=32   -> cp = 1200*atanh(summary.value)  (Task 099's ②)
Both students: the same 1,000,000 train positions (the first 100 shards of the
run2m_c cache, in cache order, via --limit-train), the same 40,000 val
positions (val is labelled by the SAME family as train so nnue_best.pt is
selected against its own teacher), the same NNUE recipe as 098 (NNUE_ARGS), the
same seed. 098's raw student was trained on 1.96M, so the raw arm is RETRAINED
here rather than reused -- reusing it would confound label family with data
size.

WHY 1M AND NOT 2M: the day's clock. The labeller runs ~187 pos/s single-process
(scratch probe on run2m_c shard 0; two processes on the GPU total 168/s, so
there is no parallel speed-up), i.e. 1.04M positions ~ 1.6 h, 2M ~ 3.0 h. With
training, export, 800 games and the write-up, 2M does not fit before 24:00 JST
with a margin; 1M does. Task 085's 2M -> 32M reversal applies with more force
at 1M: a positive here earns a 2M/32M confirmation, nothing more.

PRE-REGISTERED (committed locally before the run; results never seen):
  * primary = paired bootstrap (opening pairs, colour-swapped) of the
    search-student vs raw-student head-to-head, yane-vs-yane, 100 ms byoyomi,
    Threads 1, 6 random opening plies, max 320 plies -- Task 098's protocol.
    800 games run as 4 parallel workers x 200 games (distinct match seeds); each
    worker is a complete colour-swapped series, so the pair structure is intact.
    Disclosed operational deviation from 098 (one sequential process): four
    engines think at once on a 28-thread idle machine; both arms alternate
    colours inside every worker, so any frequency effect is symmetric.
  * clock rule (data-independent): if the match cannot start by 22:00 JST the
    workers run 150 games each (600 total) so the verdict still lands today.
  * hard gates (all before any Elo is seen): sign oracle at S=32 (099),
    per-shard gikou==cache alignment assert (084/098), the raw head reproducing
    Task 098's int16 raw labels on every scored row (a wiring gate: >= 50%
    exact and every row within 10 cp; the self-test measured 84.5% exact /
    max 3 cp of cross-graph TF32 rounding on shard 0), search labels having actually
    moved (median |cp_search - cp_raw| >= 100 cp; 099 stage2 measured 189 cp),
    0cp gate GREEN for both students, zero illegal/engine failures.
  * decision table below. Detection floor disclosed: 098 measured a +-23.1 Elo
    half-width at 800 games; a label effect below that cannot be resolved.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from m2_usi import load_model  # noqa: E402
from t094_pipeline import run_logged  # noqa: E402
from t098_pipeline import (  # noqa: E402
    DETECTION_FLOOR_400, NNUE_ARGS, load_yane_tsv, paired_h2h_delta, validate_h2h,
    yane_selftest,
)
from t099_search_label import (  # noqa: E402
    CLAMP_CP, SEARCH_BATCH, build_batched_search, score_sfens, sign_oracle, value_to_cp,
)

sys.path.insert(0, os.path.expanduser("~/GitHub/shogi/binbo-shogi"))
from trainer.prepare_data import shard_files  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))

# ---- pre-registered constants -------------------------------------------------
SIMS = 32                                 # 099's mechanism-selected S
LIMIT_TRAIN = 1_000_000                   # first 100 cache shards, cache order
H2H_WORKERS = 4
H2H_GAMES_PER_WORKER = 200                # 800 games = 400 colour-swapped pairs
H2H_GAMES_PER_WORKER_LATE = 150           # clock rule: 600 games if late
H2H_CLOCK_CUTOFF_JST = dt.time(22, 0)
H2H_SEEDS = (20270911, 20270912, 20270913, 20270914)
H2H_HALF_WIDTH_800 = 23.1                 # 098 measured at 800 games / 100 ms
# The raw head must reproduce 098's int16 raw labels row for row. This is a
# WIRING gate (order, decoder, side-to-move, cp transform): a fault there gives
# ~0% exact rows and |d| of hundreds of cp. It is NOT a bit-exactness gate: the
# search graph evaluates the root at batch 128 while az_rescore ran at batch
# 4096, and the self-test on shard 0 measured 84.5% exact / max |d| 3 cp
# between the two graphs (TF32/XLA kernel rounding, then int16 truncation).
RAW_EXACT_MIN = 0.50
RAW_TOLERANCE_CP = 10
LABEL_MOVED_MIN_CP = 100.0                # 099 stage2: median |search - raw| = 189 cp
SEARCH_ASYMMETRY_DEPTH = 0.30             # flags only (098: 14.90 vs 14.95)
SEARCH_ASYMMETRY_NODES = 0.05

DECISION_TABLE = (
    "0 prerequisites failed -> INVALID (non-terminal) | "
    "1 h2h CI low > 0 -> SEARCH_LABEL_TEACHES (1M; earns a 2M/32M confirmation only) | "
    "2 h2h CI high < 0 -> SEARCH_LABEL_HURTS | "
    "3 CI straddles 0 -> NULL_SEARCH_LABEL"
)


def now_jst():
    return dt.datetime.now(JST)


# ---- the labeller, cache-aligned ---------------------------------------------
class RescorePaused(Exception):
    """Raised between shards when `stop_file` exists (Task 106); every finished
    part is on disk, so the same call resumes where it stopped."""


def rescore_cache_with_search(cache, ckpt, out_dir, *, limit_train, sims, batch,
                              raw_dir, max_shards=None, splits=("train", "val"),
                              raw_ref_prefix=False, raw_out_dir=None, stop_file=None):
    """Score the cache's positions IN CACHE ORDER with the search-backed root
    value and write {train,val}_dlshogi_evals.npy the way az_rescore.py does.

    Rows past `limit_train` are never scored and are written as 0 (int16);
    meta.json records the valid row count and train_nnue is always run with
    --limit-train equal to it. Per-shard part files make the run resumable and
    let the alignment assert fail fast.

    Task 104 knobs (defaults reproduce Tasks 100-103 exactly):
      raw_ref_prefix -- the raw reference may cover only the first rows of the
          split (e.g. shared_8m's labels for shared_32m, whose first 7.95M train
          rows are shared_8m's). The raw-head wiring gate and the *_vs_098
          stats then run on that prefix; rows past it are protected by the
          per-shard gikou==cache alignment assert only.
      raw_out_dir -- also write the raw-head labels ({split}_dlshogi_evals.npy)
          there, aligned to the cache, so a raw-arm student can be trained
          without a second pass.
    """
    cache, out_dir, raw_dir = Path(cache), Path(out_dir), Path(raw_dir)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    run_dir, extras = meta["run_dir"], meta["extra_runs"]
    val_count = len(meta["val_shards"])
    primary, _ = shard_files(run_dir)
    # Same traversal as az_rescore.rescore_cache_aligned / prepare_data: the
    # primary run's train shards, then every extra run in order; val = the
    # primary run's last `val_count` shards. Rows are in this exact order.
    groups = {"train": [(run_dir, shard) for shard in primary[:-val_count]]
              + [(rd, shard) for rd in extras for shard in shard_files(rd)[0]],
              "val": [(run_dir, shard) for shard in primary[-val_count:]]}
    sizes = {"train": meta["n_train"], "val": meta["n_val"]}

    forward, model, info = load_model(str(ckpt))
    print(f"T100_LABELLER ckpt={ckpt} ({info}) sims={sims} batch={batch}", flush=True)
    search = build_batched_search(forward, model, sims)
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    board = cshogi.Board()
    stats = {}
    started = time.time()
    for split in splits:
        n = sizes[split]
        target = min(n, limit_train) if split == "train" else n
        cache_eval = np.load(cache / f"{split}_evals.npy")
        raw_ref = np.load(raw_dir / f"{split}_dlshogi_evals.npy")
        n_ref = raw_ref.shape[0]
        if cache_eval.shape[0] != n or (n_ref != n and not (raw_ref_prefix and n_ref < n)):
            raise RuntimeError(f"{split}: cache {cache_eval.shape[0]} / raw {n_ref} != {n}")
        search_cp = np.zeros(n, dtype=np.int16)
        raw_cp = np.zeros(n, dtype=np.int16)
        cur = 0
        n_no_legal = 0
        max_visits_sum = 0.0
        for shard_no, (rd, shard) in enumerate(groups[split]):
            if cur >= target:
                break
            if max_shards is not None and shard_no >= max_shards:
                break
            # Parts are keyed by (run, shard index), never by split: a run2m_c
            # shard that is val in cache_2m is train in shared_8m and its
            # search labels are the same numbers either way.
            part = parts / f"{Path(rd).name}_{shard['index']:04d}.npz"
            entries = np.fromfile(os.path.join(rd, shard["file"]),
                                  dtype=cshogi.HuffmanCodedPosAndEval)
            m = entries.shape[0]
            if cur + m > target:
                raise RuntimeError(f"{split}: limit {target} is not shard-aligned at row {cur}+{m}")
            gikou = entries["eval"].astype(np.int16)
            if not np.array_equal(gikou, cache_eval[cur:cur + m]):
                bad = int(np.sum(gikou != cache_eval[cur:cur + m]))
                raise AssertionError(f"{split} shard {shard['index']} ALIGNMENT MISMATCH: "
                                     f"{bad}/{m} gikou evals differ from the cache")
            if part.exists():
                saved = np.load(part)
                s_cp, r_cp, no_legal, mv = (saved["search_cp"], saved["raw_cp"],
                                            int(saved["n_no_legal"]), float(saved["max_visits_sum"]))
                if s_cp.shape[0] != m:
                    raise RuntimeError(f"{part}: {s_cp.shape[0]} rows != shard {m}")
            else:
                sfens = []
                no_legal = 0
                for entry in entries:
                    board.set_hcp(np.asarray(entry["hcp"]))
                    sfens.append(board.sfen())
                    if not any(True for _ in board.legal_moves):
                        no_legal += 1
                values, _visit_q, raw, max_visits = score_sfens(search, sfens, batch)
                if not (np.isfinite(values).all() and np.isfinite(raw).all()):
                    raise AssertionError(f"{split} shard {shard['index']}: non-finite label")
                s_cp = value_to_cp(values).astype(np.int16)
                r_cp = value_to_cp(raw).astype(np.int16)
                mv = float(np.sum(max_visits))
                # atomic (Task 106): a VM stop mid-write must not leave a torn part
                tmp = part.with_name(part.name + ".tmp.npz")
                np.savez(tmp, search_cp=s_cp, raw_cp=r_cp, n_no_legal=no_legal,
                         max_visits_sum=mv)
                os.replace(tmp, part)
                if stop_file is not None and Path(stop_file).exists():
                    raise RescorePaused(f"stop file {stop_file} after {part.name}")
            search_cp[cur:cur + m] = s_cp
            raw_cp[cur:cur + m] = r_cp
            n_no_legal += no_legal
            max_visits_sum += mv
            # raw-head wiring gate: the same corpus, decoder and cp transform as
            # 098 must reproduce 098's int16 labels (the value head is untouched).
            if cur + m <= n_ref:
                diff = np.abs(r_cp.astype(np.int32) - raw_ref[cur:cur + m].astype(np.int32))
                exact = float(np.mean(diff == 0))
                if exact < RAW_EXACT_MIN or int(diff.max()) > RAW_TOLERANCE_CP:
                    raise AssertionError(f"{split} shard {shard['index']}: raw head does not "
                                         f"reproduce 098 (exact {exact:.4f}, max |d| {int(diff.max())} cp)")
            elif cur < n_ref:
                raise RuntimeError(f"{split}: raw reference ends mid-shard at row {n_ref}")
            else:
                exact = float("nan")                  # past the reference prefix
            cur += m
            elapsed = time.time() - started
            rate = cur / max(1e-9, elapsed) if split == "train" else float("nan")
            remaining = (target - cur) / rate if split == "train" and rate > 0 else float("nan")
            print(f"T100_RESCORE {split} run={Path(rd).name} shard={shard['index']} rows={cur}/{target} "
                  f"elapsed={elapsed/60:.1f}min rate={rate:.0f}/s eta={remaining/60:.1f}min "
                  f"raw_exact={exact:.4f}", flush=True)
        if cur != target:
            raise AssertionError(f"{split}: scored {cur} rows, expected {target}")
        np.save(out_dir / f"{split}_dlshogi_evals.npy", search_cp)
        if raw_out_dir is not None:
            Path(raw_out_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(raw_out_dir) / f"{split}_dlshogi_evals.npy", raw_cp)
        scored = slice(0, cur)
        ref_rows = slice(0, min(cur, n_ref))
        gclip = np.clip(cache_eval[scored], -CLAMP_CP, CLAMP_CP).astype(np.float64)
        s64 = search_cp[scored].astype(np.float64)
        r64 = raw_cp[scored].astype(np.float64)
        ref64 = raw_ref[ref_rows].astype(np.float64)
        r64_ref = r64[ref_rows]
        stats[split] = {
            "rows": int(cur),
            "search_cp_std": float(s64.std()),
            "raw_cp_std": float(r64.std()),
            "gikou_clip_std": float(gclip.std()),
            "corr_search_gikou": float(np.corrcoef(s64, gclip)[0, 1]),
            "corr_raw_gikou": float(np.corrcoef(r64, gclip)[0, 1]),
            "abs_median_search_vs_gikou": float(np.median(np.abs(s64 - gclip))),
            "abs_median_raw_vs_gikou": float(np.median(np.abs(r64 - gclip))),
            "abs_median_search_vs_raw": float(np.median(np.abs(s64 - r64))),
            "corr_search_raw": float(np.corrcoef(s64, r64)[0, 1]),
            "raw_exact_vs_098": float(np.mean(r64_ref == ref64)),
            "raw_max_abs_vs_098": int(np.max(np.abs(r64_ref - ref64))),
            "raw_ref_rows": int(ref64.shape[0]),
            "n_no_legal_roots": int(n_no_legal),
            "mean_max_visits": float(max_visits_sum / max(1, cur)),
        }
        print(f"T100_LABELS {split} {json.dumps(stats[split])}", flush=True)
    stats["elapsed_s"] = time.time() - started
    stats["valid_train_rows"] = int(min(sizes["train"], limit_train))
    (out_dir / "meta.json").write_text(json.dumps({
        "labeller": "WS-T + gumbel MCTS root value (t099 build_batched_search)",
        "ckpt": str(ckpt), "sims": sims, "batch": batch,
        "valid_train_rows": stats["valid_train_rows"],
        "rows_beyond_valid_are_zero": True,
        "stats": stats,
    }, indent=2), encoding="utf-8")
    if raw_out_dir is not None:
        (Path(raw_out_dir) / "meta.json").write_text(json.dumps({
            "labeller": "WS-T raw value head (root of the t099 search graph, batch "
                        f"{batch}); cp = value_to_cp", "ckpt": str(ckpt),
            "valid_train_rows": stats["valid_train_rows"], "rows_beyond_valid_are_zero": True,
            "written_by": "t100 rescore_cache_with_search(raw_out_dir=...)",
        }, indent=2), encoding="utf-8")
    return stats


# ---- match helpers -----------------------------------------------------------
def search_conditions(path):
    """Mean depth / nodes per side from the 21-column rows (cand = search student)."""
    depth = {"cand": [], "base": []}
    nodes = {"cand": [], "base": []}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 21 or not fields[0].isdigit():
            continue
        depth["cand"].append(float(fields[8])); nodes["cand"].append(float(fields[9]))
        depth["base"].append(float(fields[12])); nodes["base"].append(float(fields[13]))
    return {side: {"avg_depth": float(np.mean(depth[side])), "avg_nodes": float(np.mean(nodes[side]))}
            for side in ("cand", "base")}


def run_parallel_matches(binbo, yane, eval_cand, eval_base, logs, tag, games, seeds, env, cwd):
    """4 complete colour-swapped series at once; each worker's TSV is its stdout."""
    procs = []
    for index, seed in enumerate(seeds):
        tsv = logs / f"{tag}_h2h_w{index}_seed{seed}_n{games}.tsv"
        if tsv.exists() and tsv.read_text(encoding="utf-8").count("elapsed_ms=") >= 1:
            procs.append((tsv, None))
            continue
        command = [str(binbo), "yane-vs-yane", str(yane), str(eval_cand), str(eval_base),
                   "100", str(seed), str(games), "6", "320", "0", "0", "0.05", "0.05"]
        print(f"RUN {' '.join(command)} log={tsv}", flush=True)
        handle = tsv.open("w", encoding="utf-8")
        procs.append((tsv, (subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                                             stderr=subprocess.STDOUT), handle)))
    for tsv, proc in procs:
        if proc is None:
            continue
        process, handle = proc
        rc = process.wait()
        handle.close()
        if rc != 0:
            raise RuntimeError(f"yane-vs-yane worker failed rc={rc}: {tsv}")
    return [tsv for tsv, _ in procs]


def combine_h2h(tsvs, games_per_worker):
    rows, offset = [], 0
    shapes = []
    for tsv in tsvs:
        part = load_yane_tsv(tsv)
        shapes.append(validate_h2h(part, tsv, expected_games=games_per_worker))
        for row in part:
            row = dict(row)
            row["game"] += offset
            rows.append(row)
        offset += len(part)
    shape = validate_h2h(rows, "combined", expected_games=games_per_worker * len(tsvs))
    shape["bad_reasons"] = [reason for s in shapes for reason in s["bad_reasons"]]
    return rows, shape


# ---- verdict -----------------------------------------------------------------
def analyse(h2h, labels, students, prerequisites, conditions, clock, limit_train=LIMIT_TRAIN,
            corpus_desc=None, label_moved_min_cp=LABEL_MOVED_MIN_CP, sims=SIMS):
    failures = prerequisites["failures"]
    low, high = h2h["ci95"]
    if failures:
        verdict, terminal = "INVALID", False
    elif low > 0.0:
        verdict, terminal = "SEARCH_LABEL_TEACHES", True
    elif high < 0.0:
        verdict, terminal = "SEARCH_LABEL_HURTS", True
    else:
        verdict, terminal = "NULL_SEARCH_LABEL", True
    flags = []
    if h2h["half_width"] > 1.5 * H2H_HALF_WIDTH_800:
        flags.append("UNDERPOWERED_VS_PLAN")
    if clock["late_rule_applied"]:
        flags.append("CLOCK_RULE_600_GAMES")
    depth_gap = abs(conditions["cand"]["avg_depth"] - conditions["base"]["avg_depth"])
    nodes_ratio = conditions["cand"]["avg_nodes"] / max(1.0, conditions["base"]["avg_nodes"])
    if depth_gap > SEARCH_ASYMMETRY_DEPTH or abs(nodes_ratio - 1.0) > SEARCH_ASYMMETRY_NODES:
        flags.append("SEARCH_ASYMMETRY")
    if labels["train"]["abs_median_search_vs_raw"] < label_moved_min_cp:
        flags.append("SEARCH_LABEL_BARELY_MOVED")
    return {
        "task": 100,
        "preregistration": {
            "one_variable": f"teacher label family: WS-T raw value vs WS-T + search (S={sims}) root value",
            "corpus": corpus_desc or f"run2m_c cache, first {limit_train} train rows (cache order) + 40k val, both arms",
            "recipe": "Task 098 NNUE_ARGS, both students retrained (raw arm NOT reused from 098)",
            "primary": "yane-vs-yane 100ms/Threads1, 4 workers x 200 games, paired bootstrap over opening pairs",
            "decision_table": DECISION_TABLE,
            "clock_rule": f"match start after {H2H_CLOCK_CUTOFF_JST} JST -> {H2H_GAMES_PER_WORKER_LATE} games/worker",
            "hard_gates": ["sign_oracle", "per_shard_alignment", "raw_head_reproduces_098",
                           "search_label_moved", "0cp_gate_both", "no_illegal"],
            "detection_floor_800": H2H_HALF_WIDTH_800,
            "promotion_caveat": "Task 085: 2M +237 reversed to 32M -36; a 1M win earns a 2M/32M confirmation only",
            "scope": "AZ search labels in general (099: d32 ~ 0, the loop net is not needed)",
        },
        "primary_h2h": h2h,
        "search_conditions": conditions,
        "label_stats": labels,
        "students": students,
        "clock": clock,
        "prerequisites": prerequisites,
        "flags": flags,
        "terminal": terminal,
        "verdict": verdict,
    }


def report(results):
    h2h = results["primary_h2h"]
    print(f"GATE primary h2h(search-student vs raw-student) n_pairs={h2h['n_pairs']} "
          f"W-D-L={h2h['wins']}-{h2h['draws']}-{h2h['losses']} score={h2h['score']:.4f} "
          f"elo={h2h['elo']:+.2f} ci={[round(v, 2) for v in h2h['ci95']]} "
          f"hw={h2h['half_width']:.1f}", flush=True)
    for split, stats in results["label_stats"].items():
        if isinstance(stats, dict):
            print(f"LABELS {split} search_std={stats['search_cp_std']:.0f} raw_std={stats['raw_cp_std']:.0f} "
                  f"corr_gikou search={stats['corr_search_gikou']:.3f} raw={stats['corr_raw_gikou']:.3f} "
                  f"|d|vs gikou search={stats['abs_median_search_vs_gikou']:.0f} "
                  f"raw={stats['abs_median_raw_vs_gikou']:.0f} |d|search-raw={stats['abs_median_search_vs_raw']:.0f} "
                  f"raw_exact_098={stats['raw_exact_vs_098']:.4f}", flush=True)
    cond = results["search_conditions"]
    print(f"SEARCH cand depth={cond['cand']['avg_depth']:.2f} nodes={cond['cand']['avg_nodes']:.0f} | "
          f"base depth={cond['base']['avg_depth']:.2f} nodes={cond['base']['avg_nodes']:.0f}", flush=True)
    print(f"FLAGS {results['flags']}", flush=True)


# ---- main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task100")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="gates + a 1-shard rescore into a scratch dir + a 4-game match smoke")
    parser.add_argument("--skip-cargo", action="store_true")
    parser.add_argument("--batch", type=int, default=SEARCH_BATCH)
    # Task 100b (2M confirmation) knobs; the defaults reproduce Task 100 exactly.
    parser.add_argument("--limit-train", type=int, default=LIMIT_TRAIN,
                        help="train rows for BOTH arms (shard-aligned; default = Task 100's 1M)")
    parser.add_argument("--raw-student", default=None,
                        help="existing raw-label student model dir to reuse instead of retraining "
                             "(only legitimate when it was trained on the same rows/recipe, e.g. "
                             "task098_student_wst for 1,960,000 rows)")
    parser.add_argument("--no-clock-rule", action="store_true",
                        help="disable the 22:00 JST 600-game rule (for runs that cross midnight)")
    parser.add_argument("--sims", type=int, default=SIMS,
                        help="MCTS simulations for the search label (default 32 = Tasks 099-102)")
    parser.add_argument("--label-moved-min-cp", type=float, default=LABEL_MOVED_MIN_CP,
                        help="gate: median |cp_search - cp_raw| must reach this (100 for S=32; "
                             "lower S moves the value less, e.g. 30 = 098's 'value moved' bar)")
    parser.add_argument("--raw-student-after", action="store_true",
                        help="train the raw-arm student AFTER the rescore instead of overlapping it "
                             "(GPU contention costs more than it saves for big corpora)")
    parser.add_argument("--cache", default=None,
                        help="prepare_data cache dir (default: work/task097/cache_2m = Task 100)")
    parser.add_argument("--raw-dir", default=None,
                        help="az_rescore output dir with the RAW WS-T labels aligned to --cache "
                             "(default: work/task098/rescore_wst = Task 100)")
    args = parser.parse_args()
    limit_train = args.limit_train

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    eval_dirs = data_root / "eval_dirs"
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    yane_eval = data_root / "engines/yaneuraou/eval"
    wst = clone / "t087_wst/ws2.ckpt"
    cache = Path(args.cache) if args.cache else data_root / "work/task097/cache_2m"
    raw_dir = Path(args.raw_dir) if args.raw_dir else data_root / "work/task098/rescore_wst"
    work = data_root / "work" / args.run_tag
    search_dir = work / "rescore_search"

    if args.dry_run:
        rate = 187.0
        print(f"DRYRUN work={work} labeller=WS-T+search sims={args.sims} batch={args.batch}")
        print(f"DRYRUN corpus first {args.limit_train} train rows + 40000 val of {cache}")
        print(f"DRYRUN students raw(retrained on {LIMIT_TRAIN}) vs search, NNUE_ARGS={NNUE_ARGS}")
        print(f"DRYRUN h2h {H2H_WORKERS}x{H2H_GAMES_PER_WORKER} games seeds={H2H_SEEDS}; "
              f"late rule after {H2H_CLOCK_CUTOFF_JST} JST -> {H2H_GAMES_PER_WORKER_LATE}/worker")
        print(f"DRYRUN decision table: {DECISION_TABLE}")
        print(f"DRYRUN est: rescore {(args.limit_train + 40000) / rate / 3600:.2f}h @ {rate:.0f}/s, "
              f"students 2 x ~7min (raw arm overlaps the rescore), export+gate ~1min, "
              f"h2h {H2H_GAMES_PER_WORKER * 11.0 / 60:.0f}min (11 s/game, 4 workers)")
        return 0

    # ---- stage 0: preflight + gates (before any result is observable) -------
    for name in ("m2_usi.py", "m2_distill_data.py", "t094_pipeline.py", "t095_gate.py",
                 "t095_pipeline.py", "t098_pipeline.py", "t099_search_label.py",
                 "t100_search_student.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    for path in (torch_py, binbo, yane, yane_eval / "nn.bin", wst, cache / "meta.json",
                 raw_dir / "train_dlshogi_evals.npy", raw_dir / "val_dlshogi_evals.npy"):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    adapter = yane_selftest(logs)
    forward, model, _ = load_model(str(wst))
    oracle = sign_oracle(forward, model, args.sims)
    del forward, model
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})

    if args.self_test:
        scratch = work / "selftest"
        stats = rescore_cache_with_search(cache, wst, scratch / "rescore_search",
                                          limit_train=10_000, sims=args.sims, batch=args.batch,
                                          raw_dir=raw_dir, max_shards=1, splits=("train",))
        if stats["train"]["abs_median_search_vs_raw"] < args.label_moved_min_cp:
            raise RuntimeError("self-test: search labels did not move away from raw")
        smoke_tsvs = run_parallel_matches(
            binbo, yane, eval_dirs / "task098_student_cand", eval_dirs / "task098_student_wst",
            scratch, "selftest", 4, H2H_SEEDS[:2], env, repo)
        rows, shape = combine_h2h(smoke_tsvs, 4)
        delta = paired_h2h_delta(rows, label="selftest")
        cond = search_conditions(smoke_tsvs[0])
        print(f"T100_SELFTEST_OK rescore_rate={10_000 / stats['elapsed_s']:.0f}/s "
              f"(incl. val 0 rows + JIT) h2h_games={shape['n_games']} elo={delta['elo']:+.1f} "
              f"depth={cond['cand']['avg_depth']:.1f}/{cond['base']['avg_depth']:.1f}", flush=True)
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    work.mkdir(parents=True, exist_ok=True)
    print(f"T100_PREFLIGHT_OK head={head} exp_dirty=0 start={now_jst():%H:%M} JST", flush=True)

    if not args.skip_cargo:
        run_logged(["cargo", "test", "--release"], logs / f"{args.run_tag}_cargo_test.log",
                   cwd=repo, env=env)
        print("T100_CARGO_TEST_OK", flush=True)

    # ---- stage 1: raw-arm student (overlaps the rescore) + search rescore -----
    def student_command(name, eval_npy):
        model_dir = models / f"{args.run_tag}_student_{name}"
        return model_dir, [str(torch_py), "-m", "trainer.train_nnue", str(cache), "--out",
                           str(model_dir), *NNUE_ARGS, "--eval-npy", str(eval_npy),
                           "--limit-train", str(limit_train)]

    raw_model, raw_cmd = student_command("raw", raw_dir)
    raw_proc = None
    if args.raw_student:
        raw_model = Path(args.raw_student)
        history = json.loads((raw_model / "history.json").read_text(encoding="utf-8"))
        if history.get("n_train") != limit_train:
            raise RuntimeError(f"--raw-student {raw_model} trained on {history.get('n_train')} "
                               f"rows, not {limit_train}")
        print(f"T100_RAW_STUDENT_REUSED {raw_model} n_train={history['n_train']}", flush=True)
    elif not (raw_model / "nnue_best.pt").exists() and not args.raw_student_after:
        raw_log = (logs / f"{args.run_tag}_student_raw.log").open("w", encoding="utf-8")
        print(f"RUN {' '.join(raw_cmd)} log={raw_log.name}", flush=True)
        raw_proc = subprocess.Popen(raw_cmd, cwd=repo, env=env, stdout=raw_log,
                                    stderr=subprocess.STDOUT)

    if not (search_dir / "meta.json").exists():
        labels = rescore_cache_with_search(cache, wst, search_dir, limit_train=limit_train,
                                           sims=args.sims, batch=args.batch, raw_dir=raw_dir)
    else:
        labels = json.loads((search_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    print(f"T100_RESCORE_DONE elapsed={labels['elapsed_s']/3600:.2f}h", flush=True)
    if raw_proc is not None:
        rc = raw_proc.wait()
        if rc != 0:
            raise RuntimeError(f"raw student training failed rc={rc}")
    if args.raw_student_after and not args.raw_student and not (raw_model / "nnue_best.pt").exists():
        run_logged(raw_cmd, logs / f"{args.run_tag}_student_raw.log", cwd=repo, env=env)
    print("T100_STUDENT_RAW_DONE", flush=True)

    # ---- stage 2: search-arm student ----------------------------------------
    search_model, search_cmd = student_command("search", search_dir)
    if not (search_model / "nnue_best.pt").exists():
        run_logged(search_cmd, logs / f"{args.run_tag}_student_search.log", cwd=repo, env=env)
    print("T100_STUDENT_SEARCH_DONE", flush=True)

    # ---- stage 3: export + 0cp gate, both arms ------------------------------
    students = {}
    for name, model_dir in (("raw", raw_model), ("search", search_model)):
        eval_dir = eval_dirs / f"{args.run_tag}_student_{name}"
        if name == "raw" and args.raw_student:
            eval_dir = eval_dirs / raw_model.name          # 098's nn.bin, 0cp GREEN of record
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            run_logged([str(torch_py), "-m", "trainer.export_nnue", str(model_dir),
                        "--weights", "nnue_best.pt", "--out", str(model_dir / "nnue_best.bnnq")],
                       logs / f"{args.run_tag}_export_{name}.log", cwd=repo, env=env)
            run_logged([str(torch_py), "-m", "trainer.binbo_to_yane",
                        "--from-bnnq", str(model_dir / "nnue_best.bnnq"),
                        "--template", str(yane_eval / "nn.bin"), "--out", str(eval_dir / "nn.bin")],
                       logs / f"{args.run_tag}_yane_{name}.log", cwd=repo, env=env)
        gate_log = logs / f"{args.run_tag}_0cp_{name}.log"
        if name == "raw" and args.raw_student and not gate_log.exists():
            prior = logs / f"{raw_model.name.split('_student_')[0]}_0cp_wst.log"
            if prior.exists():
                gate_log.write_text(prior.read_text(encoding="utf-8"), encoding="utf-8")
        if not gate_log.exists():
            run_logged(["bash", "scripts/binbo_to_yane.sh", str(model_dir / "nnue_best.bnnq")],
                       gate_log, cwd=repo, env=env)
        gate_text = gate_log.read_text(encoding="utf-8")
        if "RED" in gate_text or gate_text.count("GREEN") < 2:
            raise RuntimeError(f"0cp gate failed for {name}: see {gate_log}")
        summary = None
        if (model_dir / "history.json").exists():
            history = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
            summary = {key: history.get(key) for key in
                       ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")}
            summary["last_epoch"] = history.get("history", [None])[-1]
        students[name] = {"model_dir": str(model_dir), "eval_dir": str(eval_dir),
                          "gate": "GREEN", "training": summary}
    print("T100_STUDENTS_READY", flush=True)

    # ---- stage 4: head-to-head, 4 parallel colour-swapped series ------------
    start = now_jst()
    late = (start.time() >= H2H_CLOCK_CUTOFF_JST) and not args.no_clock_rule
    games = H2H_GAMES_PER_WORKER_LATE if late else H2H_GAMES_PER_WORKER
    clock = {"match_start_jst": start.strftime("%Y-%m-%d %H:%M"), "late_rule_applied": late,
             "games_per_worker": games, "workers": H2H_WORKERS}
    print(f"T100_H2H_START {clock}", flush=True)
    tsvs = run_parallel_matches(binbo, yane, students["search"]["eval_dir"],
                                students["raw"]["eval_dir"], logs, args.run_tag, games,
                                H2H_SEEDS[:H2H_WORKERS], env, repo)
    rows, shape = combine_h2h(tsvs, games)
    h2h = paired_h2h_delta(rows, label="student_search_minus_raw")
    conditions = {}
    per_worker = [search_conditions(tsv) for tsv in tsvs]
    for side in ("cand", "base"):
        conditions[side] = {key: float(np.mean([c[side][key] for c in per_worker]))
                            for key in ("avg_depth", "avg_nodes")}
    clock["match_end_jst"] = now_jst().strftime("%Y-%m-%d %H:%M")
    print("T100_H2H_DONE", flush=True)

    # ---- stage 5: verdict ---------------------------------------------------
    failures = []
    if shape["bad_reasons"]:
        failures.append(f"illegal_or_engine_fail_{len(shape['bad_reasons'])}")
    if labels["train"]["raw_exact_vs_098"] < RAW_EXACT_MIN:
        failures.append("raw_head_mismatch_098")
    if labels["train"]["abs_median_search_vs_raw"] < args.label_moved_min_cp:
        failures.append("search_label_not_applied")
    prerequisites = {"failures": failures, "adapter_selftest": adapter, "sign_oracle": oracle,
                     "h2h_shape": shape, "h2h_tsvs": [str(t) for t in tsvs]}
    cache_meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    results = analyse(h2h, labels, students, prerequisites, conditions, clock, limit_train,
                      corpus_desc=f"{cache.name}: first {limit_train} train rows (cache order) + "
                                  f"{cache_meta['n_val']} val, both arms",
                      label_moved_min_cp=args.label_moved_min_cp, sims=args.sims)
    results["limit_train"] = limit_train
    results["raw_student_reused"] = args.raw_student
    results["git"] = {"head": head, "exp_dirty": 0, "pushed": False}
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T100_PIPELINE_DONE verdict={results['verdict']} terminal={results['terminal']} "
          f"results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
