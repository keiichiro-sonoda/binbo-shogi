#!/usr/bin/env python3
"""Shard-aware verification for a sharded teacher-data run.

Reads `manifest.json` from a generation out_dir and verifies:

  * every shard is marked done, its file exists, and its byte length equals
    count * 38 (full check over all shards);
  * the summed shard counts reach the manifest target;
  * a deterministic sample of positions per shard reads back through cshogi and
    survives the KP roundtrip (局面 -> KP -> 局面, SFEN core unchanged) with the
    expected active-feature count.

Use `--full` to KP-verify every position instead of a per-shard sample.

The manifest's recorded anomaly tallies (mate / resign / max-ply cutoffs,
timeouts, engine restarts) are aggregated and printed so their frequency over a
big run is visible. Exit code is non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp

HCPE_ENTRY_SIZE = 38


def sample_indices(count: int, per_shard: int) -> list[int]:
    if per_shard <= 0 or count <= per_shard:
        return list(range(count))
    step = count / per_shard
    return sorted({min(count - 1, int(i * step)) for i in range(per_shard)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="generation output directory (with manifest.json)")
    parser.add_argument(
        "--sample-per-shard",
        type=int,
        default=200,
        help="positions to KP-verify per shard (ignored with --full)",
    )
    parser.add_argument("--full", action="store_true", help="KP-verify every position")
    args = parser.parse_args()

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    meta = manifest["meta"]
    shards = manifest["shards"]
    target = meta["target_positions"]
    print(f"manifest={manifest_path}")
    print(f"meta gikou_commit={meta['binbo_git_commit']} depth={meta['depth']} "
          f"random_plies={meta['random_plies']} base_seed={meta['base_seed']} "
          f"shard_size={meta['shard_size']} created_at={meta['created_at']}")
    print(f"shards={len(shards)} target_positions={target}")

    # --- full structural check over every shard ---
    size_errors = 0
    not_done = 0
    total_positions = 0
    total_games = 0
    anomalies = {
        "mate_cutoffs": 0,
        "resign_cutoffs": 0,
        "max_ply_cutoffs": 0,
        "crash_cutoffs": 0,
        "timeouts": 0,
        "engine_restarts": 0,
    }
    for shard in shards:
        if not shard["done"]:
            not_done += 1
            print(f"  shard {shard['index']}: NOT DONE", file=sys.stderr)
            continue
        total_positions += shard["count"]
        total_games += shard["games"]
        for key in anomalies:
            anomalies[key] += shard.get(key, 0)
        path = os.path.join(args.out_dir, shard["file"])
        expected_bytes = shard["count"] * HCPE_ENTRY_SIZE
        actual_bytes = os.path.getsize(path) if os.path.exists(path) else -1
        if actual_bytes != expected_bytes:
            size_errors += 1
            print(
                f"  shard {shard['index']}: size {actual_bytes} != expected {expected_bytes}",
                file=sys.stderr,
            )

    # --- sampled (or full) KP roundtrip check ---
    reversibility_mismatches = 0
    feature_count_mismatches = 0
    kp_checked = 0
    turn_black = 0
    turn_white = 0
    eval_min = None
    eval_max = None
    promoted_positions = 0
    hand_positions = 0
    mismatch_examples: list[str] = []

    board = cshogi.Board()
    for shard in shards:
        if not shard["done"]:
            continue
        path = os.path.join(args.out_dir, shard["file"])
        entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        if len(entries) != shard["count"]:
            size_errors += 1
            print(
                f"  shard {shard['index']}: entry count {len(entries)} != {shard['count']}",
                file=sys.stderr,
            )
        indices = range(len(entries)) if args.full else sample_indices(
            len(entries), args.sample_per_shard
        )
        for i in indices:
            entry = entries[i]
            board.set_hcp(entry["hcp"])
            kp_indices = kp.board_to_kp_indices(board)
            original = kp.sfen_core(board.sfen())
            try:
                restored = kp.sfen_core(kp.kp_indices_to_sfen(kp_indices))
            except Exception as err:  # noqa: BLE001
                restored = f"<error: {err}>"
            if restored != original:
                reversibility_mismatches += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        f"shard={shard['index']} i={i} orig='{original}' got='{restored}'"
                    )
            if len(kp_indices) != kp.expected_active_feature_count(board):
                feature_count_mismatches += 1

            kp_checked += 1
            if board.turn == cshogi.BLACK:
                turn_black += 1
            else:
                turn_white += 1
            ev = int(entry["eval"])
            eval_min = ev if eval_min is None else min(eval_min, ev)
            eval_max = ev if eval_max is None else max(eval_max, ev)
            if any((p & 0x0F) in kp.BOARD_PIECE_TO_ID and (p & 0x0F) >= cshogi.PROM_PAWN
                   for p in board.pieces if p != cshogi.NONE):
                promoted_positions += 1
            if sum(sum(h) for h in board.pieces_in_hand) > 0:
                hand_positions += 1

    print("")
    print("=== structural ===")
    print(f"shards_done={len(shards) - not_done}/{len(shards)} not_done={not_done}")
    print(f"size_errors={size_errors}")
    print(f"total_positions={total_positions} (target {target})")
    print(f"total_games={total_games}")

    print("")
    print("=== anomaly tallies (from manifest, done shards) ===")
    for key, value in anomalies.items():
        print(f"{key}={value}")

    print("")
    print("=== KP roundtrip ({} mode) ===".format("full" if args.full else "sampled"))
    print(f"kp_checked={kp_checked}")
    print(f"reversibility_mismatches={reversibility_mismatches}")
    print(f"feature_count_mismatches={feature_count_mismatches}")
    print(f"turn black_to_move={turn_black} white_to_move={turn_white}")
    print(f"eval_min={eval_min} eval_max={eval_max}")
    print(f"promoted_piece_positions={promoted_positions}")
    print(f"any_hand_positions={hand_positions}")
    if mismatch_examples:
        print("--- mismatch examples ---")
        for line in mismatch_examples:
            print(f"  {line}")

    ok = (
        not_done == 0
        and size_errors == 0
        and reversibility_mismatches == 0
        and feature_count_mismatches == 0
        and total_positions == target
    )
    print("")
    print("verify_shards=ok" if ok else "verify_shards=FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
