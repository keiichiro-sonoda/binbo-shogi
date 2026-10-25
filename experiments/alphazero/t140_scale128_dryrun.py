#!/usr/bin/env python3
"""Task 140: bounded CPU rehearsal, fresh Gikou games through the GPU boundary.

Run from this worktree after sourcing scripts/env.sh, with VENV_PY. No JAX,
teacher inference, training, downloads or writes to production data. The cache
contains Gikou diagnostics, NOT WS-T labels; model_ready remains false.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cshogi  # noqa: E402
import numpy as np  # noqa: E402
import qs_replace as qs  # noqa: E402
from trainer import compact_cache, kp  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bounded_run(argv, log, timeout=45):
    """Kill the whole process group on timeout, including teacher children."""
    with log.open("w") as output:
        proc = subprocess.Popen(list(map(str, argv)), cwd=REPO, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            require(proc.wait(timeout=timeout) == 0, f"command failed: see {log}")
        finally:
            # Also clean children if the parent exited unexpectedly.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def core(sfen):
    return " ".join(sfen.split()[:3])  # board, turn, hands; ignore move counter


def mirror_key(sfen):
    # flip_sfen is the repository's board-level augmentation oracle.
    from trainer.mirror import flip_sfen
    return min(core(sfen), core(flip_sfen(sfen)))


def seed_gate(teacher, seed):
    checked = 0
    for path in sorted(teacher.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        for shard in manifest["shards"]:
            require(int(shard["seed"]) not in (seed, seed + 1), f"seed collision: {path}")
        checked += 1
    return checked


def check_expected(run, shard):
    """Cross-check Rust's SFEN/eval/move/result sidecar using cshogi."""
    entries = np.fromfile(run / shard["file"], dtype=cshogi.HuffmanCodedPosAndEval)
    require(len(entries) == shard["count"], "hcpe record count mismatch")
    expected = (run / shard["file"]).with_suffix(".expected.tsv")
    with expected.open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    require(len(rows) == len(entries), "sidecar count mismatch")
    board = cshogi.Board()
    for entry, row in zip(entries, rows):
        board.set_hcp(entry["hcp"])
        require(core(board.sfen()) == row["sfen_core"], "Rust/cshogi SFEN mismatch")
        require(int(entry["eval"]) == int(row["eval"]), "Rust/cshogi eval mismatch")
        require(int(entry["gameResult"]) == int(row["gameResult"]), "result mismatch")
        require(int(entry["bestMove16"]) == int(row["bestMove16"]), "move16 mismatch")
        require(row["bestmove_usi"] in {cshogi.move_to_usi(m) for m in board.legal_moves},
                "sidecar bestmove is illegal")
    return entries


def replay_fixture(source, destination, positions):
    """Copy bounded prefixes from two independent shards; never call this generation."""
    shards, source_meta = shard_files(str(source))
    require(len(shards) >= 2, "replay requires two completed shards")
    destination.mkdir()
    copied = []
    for shard in shards[:2]:
        shard = dict(shard)
        count = positions // 2
        require(shard["count"] >= count, "source shard too small")
        a = np.fromfile(source / shard["file"], dtype=cshogi.HuffmanCodedPosAndEval, count=count)
        require(len(a) == count, "source shard truncated")
        a.tofile(destination / shard["file"])
        sidecar = Path(shard["file"]).with_suffix(".expected.tsv")
        with (source / sidecar).open() as inp, (destination / sidecar).open("w") as out:
            out.writelines(itertools.islice(inp, count + 1))
        shard.update(count=count, target_count=count)
        copied.append(shard)
    write_json(destination / "manifest.json", dict(meta=source_meta, shards=copied,
               task140_fixture=True, replayed_from=str(source.resolve())))


class ShortQSearch(qs.QSearchEngine):
    def _wait(self, pred, timeout):
        return super()._wait(pred, min(timeout, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=256, help="TOTAL rows, even, 32..512")
    parser.add_argument("--seed", type=int, default=20714000)
    parser.add_argument("--out", type=Path, help="NEW directory; default a temporary task140 fixture")
    parser.add_argument("--replay-run", type=Path, help="fixture only: existing run instead of fresh generation")
    parser.add_argument("--fixture-eval-dir", type=Path,
                        help="explicit alternate qsearch eval; marks the rehearsal as a different recipe")
    args = parser.parse_args()
    require(32 <= args.positions <= 512 and args.positions % 2 == 0,
            "--positions must be even and in 32..512; this is not a production generator")
    require(0 <= args.seed < 2**64 - 1, "seed outside u64 range")
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data")).resolve()
    teacher = Path(os.environ.get("TEACHER_DIR", data / "teacher"))
    binbo = Path(os.environ.get("BINBO", REPO / "target/release/binbo-shogi")).resolve()
    gikou = Path(os.environ.get("GIKOU", data / "engines/gikou/gikou")).resolve()
    yane = Path(os.environ.get("YANE", data / "engines/yaneuraou/source/YaneuraOu-by-gcc")).resolve()
    qs_eval = (args.fixture_eval_dir or data / "eval_dirs/task109_student_64m_e24").resolve()
    prerequisites = [yane, qs_eval / "nn.bin"]
    if not args.replay_run:
        require(binbo.is_relative_to(REPO), "BINBO must be built in this worktree")
        prerequisites += [binbo, gikou]
    for path in prerequisites:
        require(path.is_file(), f"missing prerequisite: {path}")
    manifests = seed_gate(teacher, args.seed) if not args.replay_run else 0
    if args.out:
        work = args.out.resolve()
        allowed = (REPO, Path(tempfile.gettempdir()).resolve(), data / "work/task140_dryrun")
        require(any(work.is_relative_to(root) for root in allowed), "output outside permitted fixture roots")
        work.mkdir(parents=True, exist_ok=False)  # never overwrite/resume a fixture
    else:
        work = Path(tempfile.mkdtemp(prefix="task140_dryrun_"))
    started = time.monotonic()
    write_json(work / "status.json", {"model_ready": False, "stage": "running"})
    print(f"T140 fixture={work} positions={args.positions} manifests_checked={manifests}", flush=True)
    run, raw, leaf = work / "teacher", work / "raw_cache", work / "leaf_cache_UNLABELLED"
    if args.replay_run:
        replay_fixture(args.replay_run, run, args.positions)
    else:
        bounded_run([binbo, "gen-teacher", gikou, run, args.positions,
                     "--shard-size", args.positions // 2, "--workers", 1, "--depth", 8,
                     "--random-plies", 6, "--max-plies", 256, "--seed", args.seed,
                     "--timeout-ms", 30000], work / "generate.log")
    shards, _ = shard_files(str(run))
    require(len(shards) == 2 and sum(s["count"] for s in shards) == args.positions,
            "incomplete generation")
    entries = [check_expected(run, shard) for shard in shards]
    bounded_run([sys.executable, "-m", "trainer.prepare_data", run, raw,
                 "--val-shards", 1, "--canonical-shared"], work / "prepare.log")
    a = qs.hcpe_to_psv(entries[0])  # train only; fixture val is never replaced
    a.tofile(work / "in.psv")
    engine = ShortQSearch.__new__(ShortQSearch)
    try:
        engine.__init__(yane, qs_eval, threads=4, hash_mb=16)
        stats = engine.run(work / "in.psv", work / "out.psv", timeout=20)
        require(stats["records"] == len(a) and stats["decode_errors"] == stats["illegal_pv"] == 0,
                f"qsearch failed: {stats}")
        engine.threads = 1
        engine.run(work / "in.psv", work / "out_1worker.psv", timeout=20)
        require((work / "out.psv").read_bytes() == (work / "out_1worker.psv").read_bytes(),
                "qsearch 1/4 worker outputs differ")
    finally:
        if hasattr(engine, "proc"):
            engine.close()
    b = np.fromfile(work / "out.psv", dtype=qs.PSV)
    require(len(b) == len(a), "qsearch output truncated")
    changed, flip = qs.leaf_rows(a, b)
    leaf.mkdir()
    board = cshogi.Board()
    sfens, raw_sfens = [], []
    feats = np.empty((len(b), 77), dtype=np.int32)
    raw_indices = np.load(raw / "train_indices.npy")
    for i, entry in enumerate(b):
        board.set_psfen(a[i]["sfen"])
        raw_sfens.append(board.sfen())
        require(np.array_equal(kp.canonical_shared_row(board), raw_indices[i]), "raw features differ")
        board.set_psfen(entry["sfen"])
        require(not board.is_check(), "qsearch leaf still in check")
        sfens.append(board.sfen())
        feats[i] = kp.canonical_shared_row(board)
        require(core(kp.canonical_shared_row_to_sfen(feats[i])) == core(board.sfen()),
                "leaf features do not reconstruct SFEN")
    for split in ("train", "val"):
        for name in ("indices", "evals", "wdl"):
            arr = np.load(raw / f"{split}_{name}.npy").copy()
            if split == "train":
                if name == "indices":
                    arr = feats
                else:
                    arr = (arr.astype(np.int32) * np.where(flip, -1, 1)).astype(arr.dtype)
            np.save(leaf / f"{split}_{name}.npy", arr)
    meta = json.loads((raw / "meta.json").read_text())
    meta.update(task140_fixture=True, model_ready=False, label_status="Gikou diagnostics only; WS-T TODO")
    write_json(leaf / "meta.json", meta)
    compact = compact_cache.convert(leaf, work / "compact_fixture", chunk_rows=64)
    # TODO GPU 1 (venv_pgx): load frozen WS-T; sign_oracle(..., sims=8);
    # build_batched_search(..., 8); score_sfens(..., batch=128); save finite
    # int16 value_to_cp labels keyed by split/row below. ALL new train rows need
    # labels once at their final Q, including unchanged P == Q. No zero fillers.
    with (work / "pending_gpu.jsonl").open("w") as output:
        for split in ("train", "val"):
            for i in range(meta[f"n_{split}"]):
                if split == "train":
                    sfen = sfens[i]
                else:
                    board.set_hcp(entries[1][i]["hcp"])
                    sfen = board.sfen()
                output.write(json.dumps(dict(split=split, row=i, sfen=sfen)) + "\n")
    # TODO GPU 2: assemble real --eval-npy files and verify every row/val hash;
    # train with Task 129 + Task 135 recipe AFTER implementing 128M residency.
    # TODO CPU after training: export, 0cp gate and fixed external frame.
    val_keys = set()
    for entry in entries[1]:
        board.set_hcp(entry["hcp"])
        val_keys.add(mirror_key(board.sfen()))
    keys = [mirror_key(sfen) for sfen in sfens]
    result = dict(stage="stopped_before_GPU", model_ready=False, positions=args.positions,
                  generation="replay_existing_hcpe" if args.replay_run else "fresh_gikou",
                  recipe_exact=not bool(args.replay_run or args.fixture_eval_dir), qsearch_eval=str(qs_eval),
                  train_rows=len(sfens), val_rows=len(entries[1]), changed=int(changed.sum()),
                  flipped=int(flip.sum()), qsearch=stats, qsearch_worker_identity=True,
                  raw_unique=len({core(s) for s in raw_sfens}), leaf_unique=len({core(s) for s in sfens}),
                  leaf_unique_with_mirror=len(set(keys)), fixture_val_overlap=sum(k in val_keys for k in keys),
                  production_overlap="NOT CHECKED: no full old-corpus scan in this tiny rehearsal",
                  compact_roundtrip={s: info["mismatches"] for s, info in compact["splits"].items()},
                  leaf_features_sha256=hashlib.sha256(feats.tobytes()).hexdigest(),
                  seconds=round(time.monotonic() - started, 3))
    write_json(work / "status.json", result)
    print("T140_CPU_OK " + json.dumps(result, ensure_ascii=False), flush=True)
    print(f"GPU TODO inputs: {work / 'pending_gpu.jsonl'}; no WS-T label files or model produced.")


if __name__ == "__main__":
    main()
