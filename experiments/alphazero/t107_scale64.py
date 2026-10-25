#!/usr/bin/env python3
"""Task 107: double the filtered training set -- 64M raw positions (~40M after the Task 106 mask).

Task 106 made the adopted eval (`task106_student_filter`): search labels (WS-T +
Gumbel MCTS S=8) on the 32M corpus with tactical positions skipped (~20.1M rows
trained). The teacher run `run32m_g` (+32M gikou@d8 positions, Task 101 stage 1)
already exists; what it lacks is the search label. This builds the 64M cache by
APPENDING run32m_g after the 32M cache (so the 32M rows, their search labels and
their mask stay valid), re-scores only the new rows, and trains the same recipe.

PRE-REGISTERED (committed before the run; no 64M result is seen):
  * one variable: data size. 64M cache = `caches/shared_32m` train rows unchanged +
    `teacher/run32m_g` (3,200 shards) appended; val = the same 50k. Labels = the
    search label (S=8) for every row (32M part reused from Task 104 parts, new part
    scored here). Mask = the Task 106 rule (skip in-check / capture best move),
    recomputed on all rows. Recipe = Task 106 (098 NNUE_ARGS + --augment-mirror,
    12 epochs) plus --mmap (values identical to the in-RAM load).
  * gates: cache self-check (a 32M shard re-converted here == the 32M cache rows),
    new rows' gikou evals == hcpe, rescore alignment and raw-head wiring as Task 104,
    search label moved >= 30 cp (median) on the new rows, mask skip in [5%, 60%],
    0cp GREEN, no illegal move / engine failure, anchor depth 7 +- 0.1.
  * PRIMARY = 64M student vs the adopted `task106_student_filter`, yane-vs-yane
    100 ms / Threads 1, 1600 games = 16 chunks x 100 (seeds 20271301-16).
    FRAME = both vs gikou@7, 800 games each (seeds 20270911-14), same stage/load.
  * decision table:
      0 a gate failed                             -> INVALID (non-terminal)
      1 primary CI low > 0 and frame CI low > 0   -> SCALE_HELPS   (adoption candidate;
                                                     the swap waits for human approval)
      2 primary CI high < 0                       -> SCALE_HURTS
      3 otherwise                                 -> SCALE_NEUTRAL
  * flags only: TIME_ASYMMETRY (nodes of two compared arms differ > 5%),
    LAST_EPOCH_BEST (best epoch == 12: still improving, more epochs may help).

Interruptible like Task 106 (scripts/job.sh pause|resume task107).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS, paired_h2h_delta  # noqa: E402
from t100_search_student import RescorePaused, rescore_cache_with_search  # noqa: E402
from t103_absolute import conditions, paired_series_delta  # noqa: E402
from t104_decisive import BATCH, SIMS  # noqa: E402
from trainer import prepare_data  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402

ADOPTED = "task106_student_filter"
RUN_G = "teacher/run32m_g"
SRC_CACHE, CACHE = "caches/shared_32m", "caches/shared_64m"
SRC_PARTS = "work/task104/rescore_search/parts"
RAW_REF = "work/task102a/rescore_raw"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
WST = Path.home() / "task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt"
PRIMARY_SEEDS = tuple(20271301 + i for i in range(16))
LABEL_MOVED_MIN_CP = 30.0


# ---- 64M cache: copy the 32M rows, append run32m_g ------------------------------
def convert_job(job):
    run_dir, shard, cursor, tmp, n_train = job
    idx = np.load(tmp / "train_indices.npy", mmap_mode="r+")
    ev = np.load(tmp / "train_evals.npy", mmap_mode="r+")
    wdl = np.load(tmp / "train_wdl.npy", mmap_mode="r+")
    end = prepare_data.convert_shard(run_dir, shard, idx, ev, cursor, False, True, wdl)
    if end != cursor + shard["count"]:
        raise AssertionError(f"shard {shard['index']}: wrote {end - cursor} rows != {shard['count']}")
    idx.flush(); ev.flush(); wdl.flush()
    return shard["index"]


def build_cache(data):
    dst = data / CACHE
    if (dst / "meta.json").exists():
        return json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    src, run_g = data / SRC_CACHE, str(data / RUN_G)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("canonical_shared") or meta.get("canonical") or meta["features_per_position"] != 77:
        raise RuntimeError("the 32M cache is not the canonical-shared HalfKP layout this builder assumes")
    # self-check: re-convert the 32M cache's last shard and compare with its rows
    last_run = meta["extra_runs"][-1]
    last_shard = shard_files(last_run)[0][-1]
    probe_i = np.zeros((last_shard["count"], 77), np.int32)
    probe_e = np.zeros(last_shard["count"], np.int16)
    prepare_data.convert_shard(last_run, last_shard, probe_i, probe_e, 0, False, True, None)
    src_i = np.load(src / "train_indices.npy", mmap_mode="r")
    src_e = np.load(src / "train_evals.npy", mmap_mode="r")
    if not (np.array_equal(probe_i, src_i[-last_shard["count"]:]) and np.array_equal(probe_e, src_e[-last_shard["count"]:])):
        raise AssertionError("cache self-check failed: re-converted shard != 32M cache rows")
    print("T107_CACHE_SELFCHECK_OK (32M cache's last shard re-converted bit-exact)", flush=True)

    g_shards = shard_files(run_g)[0]
    n_src, n_g = meta["n_train"], sum(s["count"] for s in g_shards)
    n_train = n_src + n_g
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    out = {name: np.lib.format.open_memmap(tmp / f"train_{name}.npy", mode="w+", dtype=dt, shape=shape)
           for name, dt, shape in (("indices", np.int32, (n_train, 77)), ("evals", np.int16, (n_train,)),
                                   ("wdl", np.int8, (n_train,)))}
    src_w = np.load(src / "train_wdl.npy", mmap_mode="r")
    for start in range(0, n_src, 1_000_000):
        end = min(start + 1_000_000, n_src)
        out["indices"][start:end] = src_i[start:end]
        out["evals"][start:end] = src_e[start:end]
        out["wdl"][start:end] = src_w[start:end]
    for arr in out.values():
        arr.flush()
    del out
    print(f"T107_CACHE copied {n_src} rows in {time.time() - t0:.0f}s", flush=True)
    jobs, cursor = [], n_src
    for shard in g_shards:
        jobs.append((run_g, shard, cursor, tmp, n_train))
        cursor += shard["count"]
    with Pool(12) as pool:
        for k, _ in enumerate(pool.imap_unordered(convert_job, jobs, chunksize=8), 1):
            if k % 400 == 0:
                print(f"T107_CACHE converted {k}/{len(jobs)} shards", flush=True)
    for name in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        shutil.copyfile(src / name, tmp / name)
    # new rows: gikou evals straight from the hcpe, as an independent check
    ev = np.load(tmp / "train_evals.npy", mmap_mode="r")
    cur = n_src
    for shard in g_shards:
        if shard["index"] % 200 == 0:
            raw = np.fromfile(os.path.join(run_g, shard["file"]), dtype=prepare_data.cshogi.HuffmanCodedPosAndEval)
            if not np.array_equal(raw["eval"].astype(np.int16), ev[cur:cur + shard["count"]]):
                raise AssertionError(f"run32m_g shard {shard['index']}: evals differ from the hcpe")
        cur += shard["count"]
    idx = np.load(tmp / "train_indices.npy", mmap_mode="r")
    if int(idx[n_src::997].max()) >= meta["total_features"] or int(idx[n_src::997].min()) < 0:
        raise AssertionError("new rows have out-of-range feature indices")
    meta.update({"extra_runs": meta["extra_runs"] + [os.path.abspath(run_g)], "n_train": n_train,
                 "convert_seconds": round(time.time() - t0, 1),
                 "built_by": "t107_scale64.build_cache (32M cache rows copied, run32m_g appended)"})
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.rename(dst)
    print(f"T107_CACHE_DONE n_train={n_train} seconds={time.time() - t0:.0f}", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default="task107")
    args = parser.parse_args()
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    p = {"repo": repo, "logs": Path(os.environ.get("LOGS_DIR", data / "logs")),
         "binbo": repo / "target/release/binbo-shogi", "yane": data / "engines/yaneuraou/source/YaneuraOu-by-gcc",
         "gikou": data / "engines/gikou/gikou"}
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    work = data / "work" / args.run_tag
    labels, raw_dir = work / "rescore_search", work / "rescore_raw"
    model_dir = data / "models" / f"{args.run_tag}_student_64m"
    eval_dir = data / "eval_dirs" / f"{args.run_tag}_student_64m"
    adopted = data / "eval_dirs" / ADOPTED
    for path in (p["binbo"], p["yane"], p["gikou"], data / SRC_CACHE / "meta.json", data / RUN_G / "manifest.json",
                 data / SRC_PARTS, data / RAW_REF, adopted / "nn.bin", WST,
                 data / MIRROR_CACHE.replace(".npy", "_shared.npy")):
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
    print(f"T107_PREFLIGHT_OK head={head} stop_file={t106.STOP}", flush=True)
    failures, flags = [], []

    print("=== T107 STAGE cache 64M ===", flush=True)
    meta = build_cache(data)
    cache = data / CACHE
    n_train = meta["n_train"]

    print("=== T107 STAGE rescore (search label on the new rows) ===", flush=True)
    if not (labels / "meta.json").exists():
        parts = labels / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in (data / SRC_PARTS).glob("*.npz"):
            if not (parts / f.name).exists():
                shutil.copyfile(f, parts / f.name)
                copied += 1
        print(f"T107_PARTS reused {copied} parts from Task 104", flush=True)
        try:
            stats = rescore_cache_with_search(cache, WST, labels, limit_train=n_train, sims=SIMS, batch=BATCH,
                                              raw_dir=data / RAW_REF, raw_ref_prefix=True, raw_out_dir=raw_dir,
                                              stop_file=t106.STOP)
        except RescorePaused as e:
            t106.pause_here(f"rescore ({e})")
    else:
        stats = json.loads((labels / "meta.json").read_text(encoding="utf-8"))["stats"]
    tr = stats["train"]
    print(f"T107_RESCORE_DONE moved_median={tr['abs_median_search_vs_raw']} raw_exact={tr['raw_exact_vs_098']:.4f}",
          flush=True)
    if tr["raw_exact_vs_098"] < 0.50:
        failures.append("raw head does not reproduce 098 on the prefix")
    n_src = json.loads((data / SRC_CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]
    new = np.load(labels / "train_dlshogi_evals.npy", mmap_mode="r")[n_src:]
    new_raw = np.load(raw_dir / "train_dlshogi_evals.npy", mmap_mode="r")[n_src:]
    moved_new = float(np.median(np.abs(new[::7].astype(np.int32) - new_raw[::7].astype(np.int32))))
    print(f"T107_NEW_ROWS search-vs-raw median |d| = {moved_new:.0f} cp", flush=True)
    if moved_new < LABEL_MOVED_MIN_CP:
        failures.append(f"search label barely moved on the new rows ({moved_new:.0f} cp)")

    print("=== T107 STAGE mask ===", flush=True)
    mask = t106.build_mask(cache, work)
    if mask["bad_best_moves"] > 0:
        failures.append(f"mask: {mask['bad_best_moves']} best moves do not decode")
    if not t106.SKIP_MIN <= mask["skip_frac"] <= t106.SKIP_MAX:
        failures.append(f"mask: skip fraction {mask['skip_frac']:.3f}")

    if not (model_dir / "history.json").exists():
        print("=== T107 STAGE train 64M student (mmap) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", cache, "--out", model_dir, *NNUE_ARGS, "--augment-mirror",
                  "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", labels, "--limit-train", n_train,
                  "--row-mask", work / "mask/train_keep.npy", "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 p["logs"] / f"{args.run_tag}_student_64m.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                  "--out", model_dir / "nnue_best.bnnq"], p["logs"] / f"{args.run_tag}_export_64m.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                  "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 p["logs"] / f"{args.run_tag}_yane_64m.log", cwd=repo, env=env)
    gate = p["logs"] / f"{args.run_tag}_0cp_64m.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    if hist["best_epoch"] == hist["epochs"]:
        flags.append("LAST_EPOCH_BEST")
    print(f"T107_STUDENT_READY best_epoch={hist['best_epoch']}", flush=True)

    print("=== T107 STAGE primary (64M vs adopted, 16 x 100 games) ===", flush=True)
    tsvs = [p["logs"] / f"{args.run_tag}_primary_c{i:02d}_seed{s}_n{t106.CHUNK_GAMES}.tsv"
            for i, s in enumerate(PRIMARY_SEEDS)]
    commands = [[p["binbo"], "yane-vs-yane", p["yane"], eval_dir, adopted, "100", s, t106.CHUNK_GAMES,
                 "6", "320", "0", "0", "0.05", "0.05"] for s in PRIMARY_SEEDS]
    t106.run_chunks(commands, tsvs, env, repo)
    rows, bad = t106.combine(tsvs, t106.CHUNK_GAMES)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="student_64m_minus_adopted")
    print(f"T107_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} score={primary['score']:.4f} "
          f"elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]} "
          f"nodes={cond_p['a']['avg_nodes']:.0f}/{cond_p['b']['avg_nodes']:.0f}", flush=True)
    if abs(cond_p["a"]["avg_nodes"] / cond_p["b"]["avg_nodes"] - 1) > t106.NODES_TOL:
        flags.append("TIME_ASYMMETRY_PRIMARY")

    print(f"=== T107 STAGE frame (both vs gikou@{t106.ANCHOR_LEVEL}) ===", flush=True)
    r64, c64, b64, t64 = t106.anchor_series(p, eval_dir, f"{args.run_tag}_64m_vs_gikou{t106.ANCHOR_LEVEL}", env)
    ra, ca, ba, ta = t106.anchor_series(p, adopted, f"{args.run_tag}_adopted_vs_gikou{t106.ANCHOR_LEVEL}", env)
    failures += b64 + ba
    if any(a["opening"] != b["opening"] or a["color"] != b["color"] for a, b in zip(r64, ra)):
        failures.append("frame: openings differ between arms")
    if abs(c64["a"]["avg_nodes"] / ca["a"]["avg_nodes"] - 1) > t106.NODES_TOL:
        flags.append("TIME_ASYMMETRY_FRAME")
    h64 = paired_h2h_delta(r64, label="student_64m_minus_gikou7")
    ha = paired_h2h_delta(ra, label="adopted_minus_gikou7")
    frame = paired_series_delta(r64, ra, label="(64M - adopted) in the gikou@7 frame")
    print(f"T107_FRAME 64m={h64['elo']:+.2f} adopted={ha['elo']:+.2f} "
          f"delta={frame['delta_elo']:+.2f} ci={[round(v, 2) for v in frame['ci95']]}", flush=True)

    low, high = primary["ci95"]
    verdict = ("INVALID" if failures else "SCALE_HELPS" if low > 0 and frame["ci95"][0] > 0
               else "SCALE_HURTS" if high < 0 else "SCALE_NEUTRAL")
    out = p["logs"] / f"{args.run_tag}_results.json"
    out.write_text(json.dumps({
        "task": 107, "git": {"head": head, "pushed": False}, "cache": {"n_train": n_train}, "mask": mask,
        "label_stats": stats, "new_rows_moved_median": moved_new,
        "training": {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")},
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"student_64m": h64, "adopted": ha, "delta": frame, "conditions": {"64m": c64, "adopted": ca},
                  "tsvs": [str(t) for t in t64 + ta]},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T107_VERDICT verdict={verdict} flags={flags} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
