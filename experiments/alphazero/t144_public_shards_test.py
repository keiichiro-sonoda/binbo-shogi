#!/usr/bin/env python3
"""Task 144: bounded CPU integration tests with public PSV and independent board checks.

Source scripts/env.sh; run with VENV_PY and --out a NEW dryrun directory.
No training/GPU/network. Includes the real 7.24M-row exclusion index (minutes).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import cshogi  # noqa: E402
import numpy as np  # noqa: E402
import t144_public_shards as tool  # noqa: E402


def run(work, *argv, error=None):
    name = f"command_{len(list(work.glob('command_*.log'))):02d}.log"
    with (work / name).open("w") as log:
        result = subprocess.run([sys.executable, str(Path(tool.__file__).resolve()), *map(str, argv)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=tool.REPO, timeout=240)
    output = (work / name).read_text()
    assert result.returncode == 0 if error is None else result.returncode != 0 and error in output, output
    print(f"PASS {argv[0]} log={name}", flush=True)
    return output


def material(board):
    values = [0, 100, 300, 320, 450, 800, 1000, 550, 0, 550, 550, 550, 550, 950, 1150]
    total = sum(values[p & 15] * (1 if p < 16 else -1) for p in board.pieces if p)
    hand_values = (100, 300, 320, 450, 550, 800, 1000)
    total += sum((1 if color == 0 else -1) * sum(n*v for n, v in zip(hand, hand_values))
                 for color, hand in enumerate(board.pieces_in_hand))
    return total * (1 if board.turn == cshogi.BLACK else -1)


def audit(work, source, drop_part, inherit_part):
    raw = np.fromfile(source, dtype=tool.qs.PSV)
    leaf = np.fromfile(inherit_part / "qsearch_leaf.psv", dtype=tool.qs.PSV)
    marker = np.fromfile(inherit_part / "qsearch_input.psv", dtype=tool.qs.PSV)
    assert len(raw) == len(leaf) == len(marker) == 20_000
    changed, flip = tool.qs.leaf_rows(marker, leaf)
    feature = np.load(inherit_part / "features.npy", mmap_mode="r")
    score = np.load(inherit_part / "scores.npy")
    result = np.load(inherit_part / "results.npy")
    sign = np.where(flip, -1, 1)
    assert np.array_equal(score.astype(np.int32), raw["score"].astype(np.int32) * sign)
    assert np.array_equal(result.astype(np.int32), raw["game_result"].astype(np.int32) * sign)
    assert np.array_equal(np.load(drop_part / "source_rows.npy"), np.flatnonzero(~changed))
    assert np.array_equal(np.load(drop_part / "scores.npy"), raw["score"][~changed])
    assert not np.load(drop_part / "changed.npy").any()
    b = cshogi.Board()
    sample = np.random.default_rng(144).choice(len(raw), 500, replace=False)
    for i in sample:
        b.set_psfen(leaf[i]["sfen"])
        assert tool.kp.canonical_shared_row_to_sfen(feature[i]).split()[:3] == b.sfen().split()[:3]
        assert bytes(np.load(inherit_part / "hashes.npy", mmap_mode="r")[i]) == tool.row_hash(feature[i])
    before, after, turns_before, turns_after = [], [], [], []
    for orig, end, parity in zip(raw, leaf, flip):
        b.set_psfen(orig["sfen"])
        before.append(material(b))
        turns_before.append(b.turn)
        b.set_psfen(end["sfen"])
        after.append(material(b))
        turns_after.append(b.turn)
        assert (turns_before[-1] != turns_after[-1]) == bool(parity)
        assert not b.is_check()
    correlations = {}
    for stage, mat, turns, labels in (("original", before, turns_before, raw["score"]),
                                       ("leaf", after, turns_after, score)):
        for color in (0, 1):
            m = np.asarray(turns) == color
            corr = float(np.corrcoef(np.asarray(mat)[m], labels[m])[0, 1])
            assert corr > 0, (stage, color, corr)
            correlations[f"{stage}_{'black' if color == 0 else 'white'}"] = corr
    for color in (0, 1):
        m = flip & (np.asarray(turns_after) == color)
        corr = float(np.corrcoef(np.asarray(after)[m], score[m])[0, 1])
        naive = float(np.corrcoef(np.asarray(after)[m], raw["score"][m])[0, 1])
        assert corr > 0 and naive < 0
        correlations[f"flipped_leaf_{color}"] = corr
        correlations[f"flipped_without_sign_{color}"] = naive
    meta = tool.read_json(inherit_part / "part.json")
    assert meta["qsearch"]["records"] == 20_000
    assert meta["qsearch"]["decode_errors"] == meta["qsearch"]["illegal_pv"] == meta["in_check"] == 0
    return dict(rows=20_000, changed=int(changed.sum()), flipped=int(flip.sum()),
                default_kept=int((~changed).sum()), roundtrip_sample=500, correlations=correlations,
                qsearch=meta["qsearch"], in_check=meta["in_check"], part_seconds=meta["seconds"],
                qsearch_rows_per_second_per_thread=20_000/meta["qsearch_seconds"]/2,
                feature_rows_per_second_one_thread=20_000/meta["feature_seconds"],
                qsearch_seconds=meta["qsearch_seconds"], feature_seconds=meta["feature_seconds"])


def fixture_part(path, rows):
    path.mkdir()
    n = len(rows)
    arrays = dict(features=np.asarray(rows, dtype="<i4"), scores=np.arange(n, dtype="<i2"),
                  results=np.zeros(n, "i1"), hashes=np.asarray([np.frombuffer(tool.row_hash(r), dtype="u1") for r in rows]),
                  source_rows=np.arange(n, dtype="<i8"), changed=np.zeros(n, bool), flip=np.zeros(n, bool))
    files = {}
    for key, a in arrays.items():
        np.save(path / (key + ".npy"), a)
        files[key + ".npy"] = tool.sha256(path / (key + ".npy"))
    tool.atomic_json(path / "part.json", dict(format=tool.FORMAT, rows=n, files=files,
                     source={"path": "synthetic collision fixture", "sha256": "fixture"},
                     recipe={"changed": "drop"}, lo=0, hi=n))


def fixture_dedup(work, feature):
    """Task 144: independent board mirrors and forced hash collisions."""
    root = work / "fixtures"
    root.mkdir()
    public, val = root / "public", root / "val"
    public.mkdir()
    val.mkdir()
    # Pick distinct real boards; feature identity includes absolute side to move.
    a, b, c, d = [np.asarray(feature[i]).copy() for i in range(4)]
    b_board = cshogi.Board(tool.kp.canonical_shared_row_to_sfen(b))
    mb = np.asarray(tool.kp.canonical_shared_row(cshogi.Board(tool.mirror.flip_sfen(b_board.sfen()))), dtype="<i4")
    perm = tool.mirror.build_mirror_perm_shared()
    assert np.array_equal(tool.mirror_rows(b[None], perm)[0], mb)
    np.save(public / "train_indices.npy", np.asarray([a], dtype="<i4"))
    np.save(val / "val_indices.npy", np.asarray([b], dtype="<i4"))
    expected = np.array([False, False, False, True, False, True])
    results = {}
    for collision in (False, True):
        real_hash = tool.row_hash
        if collision:
            tool.row_hash = lambda row: b"x" * 16
        try:
            part, out = root / f"part_{collision}", root / f"dedup_{collision}"
            fixture_part(part, [a, b, mb, c, c, d])
            args = SimpleNamespace(parts=[part], out=out, existing=public, val_cache=val, chunk_rows=2)
            tool.dedup(args)
            first = tool.read_json(out / "manifest.json")
            assert np.array_equal(np.load(out / "keep_000000.npy"), expected)
            assert first["counts"]["kept"] == 2
            assert first["counts"]["val"] == first["counts"]["val_mirror"] == first["counts"]["existing"] == 1
            if collision:
                assert first["counts"]["hash_collisions"] > 0
            # Inject an uncommitted mask, like a crash between mask rename/DB commit.
            import sqlite3
            with sqlite3.connect(out / "index.sqlite3") as db:
                db.execute("DELETE FROM progress WHERE k='part0'")
                db.execute("DELETE FROM positions WHERE sid>=3")
            tool.dedup(args)
            assert tool.read_json(out / "manifest.json")["counts"] == first["counts"]
            assert np.array_equal(np.load(out / "keep_000000.npy"), expected)
            results[str(collision)] = first["counts"]
        finally:
            tool.row_hash = real_hash
    return results


def process_fixtures(work, source, inherit_part):
    """Task 144: actual part boundaries, empty parts and fail-closed labels."""
    raw = np.fromfile(source, dtype=tool.qs.PSV)
    small = work / "first128.psv"
    raw[:128].tofile(small)
    argv = ("process", small, "--out", work / "boundaries", "--part-rows", 32,
            "--changed", "inherit", "--threads", 1)
    run(work, *argv)
    parts = tool.expand_parts([work / "boundaries"])
    assert [(tool.read_json(p / "part.json")["lo"], tool.read_json(p / "part.json")["hi"])
            for p in parts] == [(0, 32), (32, 64), (64, 96), (96, 128)]
    assert np.array_equal(np.concatenate([np.load(p / "source_rows.npy") for p in parts]), np.arange(128))
    # Remove the last receipt and leave owned interrupted staging; resume must rebuild it.
    owner = tool.read_json(parts[-1] / ".task144-stage.json")
    shutil.rmtree(parts[-1])
    partial = Path(str(parts[-1]) + ".partial")
    tool.staging(partial, owner)
    (partial / "interrupted.raw").write_bytes(b"partial output")
    state_path = work / "boundaries/manifest.json"
    state = tool.read_json(state_path)
    state["sources"][0]["complete"] = False
    state["parts"].pop()
    tool.atomic_json(state_path, state)
    run(work, *argv)
    assert len(tool.expand_parts([work / "boundaries"])) == 4 and not partial.exists()
    run(work, *argv, "--changed", "drop", error="process recipe changed")
    changed = np.flatnonzero(np.load(inherit_part / "changed.npy"))
    all_changed = work / "all_changed.psv"
    raw[changed[:1]].tofile(all_changed)
    run(work, "process", all_changed, "--out", work / "empty", "--threads", 1)
    empty = tool.expand_parts([work / "empty"])[0]
    assert tool.load_part(empty)["rows"] == 0
    # Inject failures through the same engine interface; no new engine process.
    one = work / "one.psv"
    raw[:1].tofile(one)
    base_args = dict(psv=[one], yane=Path(os.environ["YANE"]),
                     eval_dir=tool.DATA / "eval_dirs/task129_student_64m_qsrel_e12",
                     threads=1, changed="inherit", part_rows=32, verify_rows=1,
                     timeout=10, keep_psv=False)
    checked = cshogi.Board()
    for move in ("7g7f", "3c3d", "8h3c+"):
        assert checked.is_legal(checked.move_from_usi(move))
        checked.push_usi(move)
    assert checked.is_check()
    packed_check = np.zeros(32, dtype=np.uint8)
    checked.to_psfen(packed_check)

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, inp, out, timeout):
            rows = np.fromfile(inp, dtype=tool.qs.PSV)
            if case == "check":
                rows["sfen"] = packed_check
            if case == "overflow":
                # Use a valid opposite-turn leaf, so -32768 must be rejected before casting.
                b = cshogi.Board()
                original = cshogi.Board()
                original.set_psfen(rows[0]["sfen"])
                if b.turn == original.turn:
                    b.push_usi("7g7f")
                assert b.turn != original.turn and not b.is_check()
                b.to_psfen(rows[0]["sfen"])
                rows["score"] = -1
            rows.tofile(out)
            return dict(records=len(rows), decode_errors=int(case == "decode"),
                        illegal_pv=int(case == "illegal"))

        def close(self):
            pass

    outcomes = {}
    for case, error in (("decode", "qsearch failed"), ("illegal", "qsearch failed"),
                        ("overflow", "outside int16"), ("check", None)):
        entry = raw[:1].copy()
        if case == "overflow":
            entry["score"] = -32768
        entry.tofile(one)
        args = SimpleNamespace(**base_args, out=work / f"failure_{case}")
        with patch.object(tool.qs, "QSearchEngine", FakeEngine):
            if error:
                try:
                    tool.process(args)
                except ValueError as exc:
                    assert error in str(exc), str(exc)
                else:
                    raise AssertionError(f"{case} should have failed")
                assert not list((args.out / "parts").glob("*/part.json"))
                assert not tool.read_json(args.out / "manifest.json")["sources"][0].get("complete")
            else:
                tool.process(args)
                meta = tool.load_part(tool.expand_parts([args.out])[0])
                assert meta["rows"] == 0 and meta["in_check"] == 1
            outcomes[case] = "pass"
    return dict(boundaries=4, rebuilt_partial=True, empty_part=True, **outcomes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = work / "first20000.psv"
    with (tool.DATA / "public/nodchip_suisho5_depth9/shuffled.bin").open("rb") as inp:
        source.write_bytes(inp.read(20_000*40))
    for policy in ("drop", "inherit"):
        run(work, "process", source, "--out", work / policy, "--changed", policy, "--threads", 2, "--keep-psv")
    drop_part = tool.expand_parts([work / "drop"])[0]
    inherit_part = tool.expand_parts([work / "inherit"])[0]
    stats = audit(work, source, drop_part, inherit_part)
    # Completed resume and recovery of a part published before the manifest update.
    before_sha = tool.sha256(inherit_part / "features.npy")
    run(work, "process", source, "--out", work / "inherit", "--changed", "inherit", "--threads", 2, "--keep-psv")
    state = tool.read_json(work / "inherit/manifest.json")
    state["sources"][0]["complete"] = False
    state["parts"] = []
    tool.atomic_json(work / "inherit/manifest.json", state)
    run(work, "process", source, "--out", work / "inherit", "--changed", "inherit", "--threads", 2, "--keep-psv")
    assert tool.sha256(inherit_part / "features.npy") == before_sha
    stats["fixtures"] = fixture_dedup(work, np.load(inherit_part / "features.npy", mmap_mode="r"))
    stats["process_fixtures"] = process_fixtures(work, source, inherit_part)
    # Run both repeated default part and repeated inherited part against production references.
    run(work, "dedup", drop_part, drop_part, inherit_part, inherit_part, "--out", work / "dedup")
    d = tool.read_json(work / "dedup/manifest.json")
    assert d["parts"][0]["counts"]["kept"] == d["parts"][1]["counts"]["kept"] == 0
    assert d["parts"][2]["counts"]["kept"] > 0
    assert d["parts"][3]["counts"]["kept"] == 0
    assert d["parts"][3]["counts"]["parts"] == (d["parts"][2]["counts"]["kept"]
                                                    + d["parts"][2]["counts"]["parts"])
    stats["dedup"] = d["counts"]
    stats["dedup_seconds"] = d["seconds"]
    run(work, "dedup", drop_part, drop_part, inherit_part, inherit_part, "--out", work / "dedup")
    assert tool.read_json(work / "dedup/manifest.json")["counts"] == d["counts"]
    run(work, "assemble", work / "dedup", "--out", work / "cache")
    scores = np.load(work / "cache/train_evals.npy")
    assert np.array_equal(scores, np.load(work / "cache_labels/train_dlshogi_evals.npy"))
    assert np.array_equal(np.load(work / "cache_labels/val_dlshogi_evals.npy"),
                          np.load(tool.DATA / "work/task129/labels/val_dlshogi_evals.npy"))
    for name in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        assert (work / "cache" / name).is_symlink()
    meta = tool.read_json(work / "cache/meta.json")
    assert all("recipe" in p for p in meta["source_parts"])
    # Recreate the exact state after labels rename and before cache rename.
    os.replace(work / "cache", work / "cache.partial")
    run(work, "assemble", work / "dedup", "--out", work / "cache")
    # Use the trainer converter unchanged; it verifies EVERY persisted feature.
    with (work / "compact.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "trainer.compact_cache", str(work / "cache"), str(work / "compact")],
                       cwd=tool.REPO, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    compact = tool.read_json(work / "compact/compact.json")
    assert all(x["mismatches"] == 0 and x["source_sha256"] == x["decoded_sha256"]
               for x in compact["splits"].values())
    stats["compact"] = compact["splits"]
    import torch
    from trainer.compact_cache import CompactCache
    torch.set_num_threads(2)
    resident = CompactCache(work / "compact", "train", device="cpu", source=work / "cache")
    sample = np.random.default_rng(144).choice(len(scores), min(500, len(scores)), replace=False)
    assert np.array_equal(resident[torch.tensor(sample)].numpy(),
                          np.load(work / "cache/train_indices.npy", mmap_mode="r")[sample])
    stats["compact_cpu_checked"] = len(sample)
    listing = work / "listing.json"
    tool.atomic_json(listing, [{"type": "file", "path": f"kifu.tag=train.depth=9.fixture{i}.bin", "size": 800_000}
                              for i in range(3)])
    output = run(work, "plan", "--dataset", "hao", "--target-rows", 15_000, "--listing", listing)
    planned = json.loads(output)
    assert len(planned["shards"]) == 2 and planned["expected_rows"] == 19_000
    fallback = json.loads(run(work, "plan", "--dataset", "tanuki", "--target-rows", 1_000_000))
    assert fallback["shards"][0]["sizes_and_names_estimated"]
    run(work, "plan", "--dataset", "hao", "--target-rows", 100_000,
        "--listing", listing, error="listing exhausted")
    tool.atomic_json(listing, tool.read_json(listing) * 2)
    run(work, "plan", "--dataset", "hao", "--target-rows", 1,
        "--listing", listing, error="duplicate shard paths")
    stats["seconds"] = time.monotonic() - start
    tool.atomic_json(work / "results.json", stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
