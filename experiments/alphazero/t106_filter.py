#!/usr/bin/env python3
"""Task 106: skip tactical training positions (best move is a capture, or in check).

The adopted eval (`task104_student_search`) learns every one of the 32M positions.
Stockfish's trainer skips positions whose best move is a capture or that are in
check ("smart fen skipping"): a static eval cannot be right in the middle of an
exchange, and those rows only teach it to guess. The qsearch micro-benchmark
(bench_qsearch.py) said that resolving exchanges barely moves the LABELS; this is
the other lever -- which POSITIONS the NNUE learns from. No new labels: the mask
comes from the hcpe's own best move (gikou@d8) and the position itself.

PRE-REGISTERED (committed before the run; no filtered-student result is seen):
  * one variable: the training row mask. Same corpus (`caches/shared_32m`), same
    labels (`work/task104/rescore_search`, the adopted eval's teacher), same recipe
    (Task 104: 098 NNUE_ARGS + --augment-mirror). keep = not (in_check(P) or
    best_move16(P) is a capture). Val is NOT filtered (same 50k as every task).
  * mask gates: cache alignment (each shard's gikou evals == the cache rows), every
    best_move16 decodes to a legal move, 5% <= skipped <= 60%.
  * PRIMARY = filtered student vs search student, yane-vs-yane 100 ms / Threads 1,
    1600 games as 16 chunks x 100 (seeds 20271201-16, 4 at a time), paired
    bootstrap over opening pairs (pairs never straddle a chunk).
  * FRAME = both students vs gikou@7, 800 games each on seeds 20270911-14 (the
    Task 105 anchor), measured in the same stage under the same load; frame delta
    (filtered - search) paired over the 400 shared openings.
  * gates: 0cp GREEN, no illegal move / engine failure, anchor depth 7 +- 0.1.
  * decision table (it also fixes Task 107's 64M recipe):
      0 a gate failed                                -> INVALID (non-terminal)
      1 primary CI low > 0 and frame CI low > 0      -> FILTER_HELPS   (64M uses the mask)
      2 primary CI high < 0                          -> FILTER_HURTS   (64M without)
      3 otherwise                                    -> FILTER_NEUTRAL (64M without; simpler)
  * flags only: TIME_ASYMMETRY when the yane nodes of two compared arms differ > 5%.

Interruptible (Task 106 infra): every stage skips finished outputs; with
BINBO_STOP_FILE (scripts/job.sh) the mask stops between shards, training saves
train_state.pt every 10 minutes and on request, matches stop between 100-game
chunks. Exit code 75 = paused; `scripts/job.sh resume task106` continues.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

from t098_pipeline import NNUE_ARGS, load_yane_tsv, paired_h2h_delta, validate_h2h  # noqa: E402
from t103_absolute import conditions, paired_series_delta  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402

SEARCH = "task104_student_search"
SEARCH_LABELS = "work/task104/rescore_search"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
ANCHOR_LEVEL, ANCHOR_THINK_MS = 7, 100
PRIMARY_SEEDS = tuple(20271201 + i for i in range(16))
CHUNK_GAMES, PARALLEL = 100, 4
FRAME_SEEDS, FRAME_GAMES = (20270911, 20270912, 20270913, 20270914), 200
SKIP_MIN, SKIP_MAX = 0.05, 0.60
NODES_TOL = 0.05
PAUSED = 75
STOP = os.environ.get("BINBO_STOP_FILE")


def stop_requested():
    return STOP is not None and os.path.exists(STOP)


def pause_here(where):
    print(f"T106_PAUSED at {where} (resume: scripts/job.sh resume <tag>)", flush=True)
    sys.exit(PAUSED)


def run(command, log_path, *, cwd, env):
    """run_logged with the pause code passed through instead of raised."""
    print(f"RUN {' '.join(map(str, command))} log={log_path}", flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        rc = subprocess.run([str(c) for c in command], cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc == PAUSED:
        pause_here(Path(log_path).name)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {command} (log {log_path})")


# ---- mask ----------------------------------------------------------------------
def mask_shard(job):
    run_dir, shard, cache_evals_slice, out = job
    if os.path.exists(out):
        return out
    entries = np.fromfile(os.path.join(run_dir, shard["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
    if not np.array_equal(entries["eval"].astype(np.int16), cache_evals_slice):
        raise AssertionError(f"{run_dir} shard {shard['index']}: ALIGNMENT MISMATCH with the cache")
    board = cshogi.Board()
    check = np.zeros(len(entries), dtype=bool)
    capture = np.zeros(len(entries), dtype=bool)
    bad = 0
    for i, entry in enumerate(entries):
        board.set_hcp(np.asarray(entry["hcp"]))
        check[i] = board.is_check()
        move = board.move_from_move16(int(entry["bestMove16"]) & 0xFFFF)
        if not board.is_legal(move):
            bad += 1
            continue
        capture[i] = cshogi.move_cap(move) != 0
    tmp = out + ".tmp.npz"
    np.savez(tmp, check=check, capture=capture, bad=bad)
    os.replace(tmp, out)
    return out


def build_mask(cache, work):
    out = work / "mask"
    if (out / "train_keep.npy").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    (out / "parts").mkdir(parents=True, exist_ok=True)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    primary, _ = shard_files(meta["run_dir"])
    val_count = len(meta["val_shards"])
    groups = [(meta["run_dir"], s) for s in primary[:-val_count]] + \
             [(rd, s) for rd in meta["extra_runs"] for s in shard_files(rd)[0]]
    evals = np.load(cache / "train_evals.npy", mmap_mode="r")
    jobs, cur = [], 0
    for rd, shard in groups:
        n = shard["count"]
        jobs.append((rd, shard, np.array(evals[cur:cur + n]), str(out / "parts" / f"{Path(rd).name}_{shard['index']:04d}.npz")))
        cur += n
    if cur != meta["n_train"]:
        raise RuntimeError(f"shards cover {cur} rows != n_train {meta['n_train']}")
    done = 0
    with Pool(12) as pool:
        for _ in pool.imap(mask_shard, jobs, chunksize=4):
            done += 1
            if done % 400 == 0:
                print(f"T106_MASK {done}/{len(jobs)} shards", flush=True)
            if stop_requested():
                pool.terminate()
                pause_here(f"mask {done}/{len(jobs)} shards")
    check = np.concatenate([np.load(j[3])["check"] for j in jobs])
    capture = np.concatenate([np.load(j[3])["capture"] for j in jobs])
    bad = int(sum(int(np.load(j[3])["bad"]) for j in jobs))
    keep = ~(check | capture)
    np.save(out / "train_keep.npy", keep)
    stats = {"rows": int(keep.size), "skip_frac": float(1 - keep.mean()), "in_check_frac": float(check.mean()),
             "best_capture_frac": float(capture.mean()), "both_frac": float((check & capture).mean()),
             "bad_best_moves": bad}
    (out / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("T106_MASK " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


# ---- matches in resumable chunks ------------------------------------------------
def complete(tsv):
    return tsv.exists() and "elapsed_ms=" in tsv.read_text(encoding="utf-8")


def run_chunks(commands, tsvs, env, cwd):
    """Run `PARALLEL` at a time; finished TSVs are skipped; stop between waves."""
    todo = [(c, t) for c, t in zip(commands, tsvs) if not complete(t)]
    for start in range(0, len(todo), PARALLEL):
        if stop_requested():
            pause_here(f"matches, {len(todo) - start} chunks left")
        procs = []
        for command, tsv in todo[start:start + PARALLEL]:
            print(f"RUN {' '.join(map(str, command))} log={tsv}", flush=True)
            handle = open(tsv, "w", encoding="utf-8")
            procs.append((subprocess.Popen([str(c) for c in command], cwd=cwd, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT), handle, tsv))
        for process, handle, tsv in procs:
            rc = process.wait()
            handle.close()
            if rc != 0:
                raise RuntimeError(f"match worker failed rc={rc}: {tsv}")


def combine(tsvs, games):
    rows, offset, bad = [], 0, []
    for tsv in tsvs:
        part = load_yane_tsv(tsv)
        bad += validate_h2h(part, tsv, expected_games=games)["bad_reasons"]
        for row in part:
            row = dict(row)
            row["game"] += offset
            rows.append(row)
        offset += len(part)
    validate_h2h(rows, "combined", expected_games=games * len(tsvs))
    return rows, bad


def anchor_series(p, eval_dir, tag, env):
    tsvs = [p["logs"] / f"{tag}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
    commands = [[p["binbo"], "yane-vs-gikou", p["yane"], eval_dir, p["gikou"], "100", ANCHOR_LEVEL, s, FRAME_GAMES,
                 "6", "320", ANCHOR_THINK_MS, "0", "0", "0.05", "0.05"] for s in FRAME_SEEDS]
    run_chunks(commands, tsvs, env, p["repo"])
    rows, bad = combine(tsvs, FRAME_GAMES)
    cond = conditions(tsvs)
    if abs(cond["b"]["avg_depth"] - ANCHOR_LEVEL) > 0.1:
        bad.append(f"{tag}: anchor depth {cond['b']['avg_depth']:.2f}")
    return rows, cond, bad, tsvs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default="task106")
    args = parser.parse_args()
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    p = {"repo": repo, "logs": Path(os.environ.get("LOGS_DIR", data / "logs")),
         "binbo": repo / "target/release/binbo-shogi", "yane": data / "engines/yaneuraou/source/YaneuraOu-by-gcc",
         "gikou": data / "engines/gikou/gikou"}
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    cache, work = data / "caches/shared_32m", data / "work" / args.run_tag
    model_dir = data / "models" / f"{args.run_tag}_student_filter"
    eval_dir = data / "eval_dirs" / f"{args.run_tag}_student_filter"
    search_eval = data / "eval_dirs" / SEARCH
    for path in (p["binbo"], p["yane"], p["gikou"], cache / "train_evals.npy", data / SEARCH_LABELS / "meta.json",
                 search_eval / "nn.bin", data / MIRROR_CACHE.replace(".npy", "_shared.npy")):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero or trainer is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
    print(f"T106_PREFLIGHT_OK head={head} stop_file={STOP}", flush=True)
    failures, flags = [], []

    print("=== T106 STAGE mask ===", flush=True)
    mask = build_mask(cache, work)
    if mask["bad_best_moves"] > 0:
        failures.append(f"mask: {mask['bad_best_moves']} best moves do not decode")
    if not SKIP_MIN <= mask["skip_frac"] <= SKIP_MAX:
        failures.append(f"mask: skip fraction {mask['skip_frac']:.3f} outside [{SKIP_MIN}, {SKIP_MAX}]")

    if not (model_dir / "history.json").exists():
        print("=== T106 STAGE train filtered student ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", STOP] if STOP else []
        run([torch_py, "-m", "trainer.train_nnue", cache, "--out", model_dir, *NNUE_ARGS, "--augment-mirror",
             "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / SEARCH_LABELS,
             "--limit-train", LIMIT_TRAIN, "--row-mask", work / "mask/train_keep.npy",
             "--checkpoint-minutes", "10", "--resume", *stop_args],
            p["logs"] / f"{args.run_tag}_student_filter.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
             "--out", model_dir / "nnue_best.bnnq"], p["logs"] / f"{args.run_tag}_export_filter.log", cwd=repo, env=env)
        run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
             "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
            p["logs"] / f"{args.run_tag}_yane_filter.log", cwd=repo, env=env)
    gate = p["logs"] / f"{args.run_tag}_0cp_filter.log"
    if not gate.exists():
        run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    print(f"T106_STUDENT_READY best_epoch={hist['best_epoch']} gate={'RED' if failures else 'GREEN'}", flush=True)

    print("=== T106 STAGE primary (filtered vs search student, 16 x 100 games) ===", flush=True)
    tsvs = [p["logs"] / f"{args.run_tag}_primary_c{i:02d}_seed{s}_n{CHUNK_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    commands = [[p["binbo"], "yane-vs-yane", p["yane"], eval_dir, search_eval, "100", s, CHUNK_GAMES,
                 "6", "320", "0", "0", "0.05", "0.05"] for s in PRIMARY_SEEDS]
    run_chunks(commands, tsvs, env, repo)
    rows, bad = combine(tsvs, CHUNK_GAMES)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="filtered_student_minus_search_student")
    print(f"T106_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} score={primary['score']:.4f} "
          f"elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]} "
          f"nodes={cond_p['a']['avg_nodes']:.0f}/{cond_p['b']['avg_nodes']:.0f}", flush=True)
    if abs(cond_p["a"]["avg_nodes"] / cond_p["b"]["avg_nodes"] - 1) > NODES_TOL:
        flags.append("TIME_ASYMMETRY_PRIMARY")

    print(f"=== T106 STAGE frame (both students vs gikou@{ANCHOR_LEVEL}) ===", flush=True)
    rf, cf, bf, tf = anchor_series(p, eval_dir, f"{args.run_tag}_filter_vs_gikou{ANCHOR_LEVEL}", env)
    rs, cs, bs, ts = anchor_series(p, search_eval, f"{args.run_tag}_search_vs_gikou{ANCHOR_LEVEL}", env)
    failures += bf + bs
    if any(a["opening"] != b["opening"] or a["color"] != b["color"] for a, b in zip(rf, rs)):
        failures.append("frame: openings differ between arms")
    if abs(cf["a"]["avg_nodes"] / cs["a"]["avg_nodes"] - 1) > NODES_TOL:
        flags.append("TIME_ASYMMETRY_FRAME")
    hf = paired_h2h_delta(rf, label=f"filtered_minus_gikou{ANCHOR_LEVEL}")
    hs = paired_h2h_delta(rs, label=f"search_minus_gikou{ANCHOR_LEVEL}")
    frame = paired_series_delta(rf, rs, label=f"(filtered - search) in the gikou@{ANCHOR_LEVEL} frame")
    print(f"T106_FRAME filtered={hf['elo']:+.2f} search={hs['elo']:+.2f} "
          f"delta={frame['delta_elo']:+.2f} ci={[round(v, 2) for v in frame['ci95']]}", flush=True)

    low, high = primary["ci95"]
    if failures:
        verdict = "INVALID"
    elif low > 0 and frame["ci95"][0] > 0:
        verdict = "FILTER_HELPS"
    elif high < 0:
        verdict = "FILTER_HURTS"
    else:
        verdict = "FILTER_NEUTRAL"
    out = p["logs"] / f"{args.run_tag}_results.json"
    out.write_text(json.dumps({
        "task": 106, "git": {"head": head, "pushed": False}, "mask": mask,
        "training": {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")},
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"filtered": hf, "search": hs, "delta": frame, "conditions": {"filtered": cf, "search": cs},
                  "tsvs": [str(t) for t in tf + ts]},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T106_VERDICT verdict={verdict} flags={flags} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
