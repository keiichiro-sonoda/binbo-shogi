#!/usr/bin/env python3
"""Prepare a sharded teacher-data run for NNUE training.

Reads the hcpe shards of a `gen-teacher` run, converts every position to its KP
feature indices once (the expensive step), splits the shards into train/val
(shard-level, so train and val never share a game), and caches everything as
.npy so training can iterate cheaply.

Every legal shogi position has exactly 77 active KP features (2 * 38 non-king
pieces + 1 side feature; piece count is conserved), so the indices cache is a
dense (N, 77) int32 matrix. The script asserts this for every position.

Cache layout (in --out):
  train_indices.npy  int32 (n_train, 77)
  train_evals.npy    int16 (n_train,)
  val_indices.npy    int32 (n_val, 77)
  val_evals.npy      int16 (n_val,)
  meta.json          split description + counts
"""
from __future__ import annotations

import argparse
import array
import json
import os
import time

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp

# Every legal position has 2*(non-king pieces + hand pieces) + 1 = 2*38 + 1 = 77
# active KP features, because shogi conserves the 38 non-king pieces.
FEATURES_PER_POSITION = 77


def shard_files(run_dir: str) -> tuple[list[dict], dict]:
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    shards = [s for s in manifest["shards"] if s["done"]]
    shards.sort(key=lambda s: s["index"])
    return shards, manifest["meta"]


def convert_shard(run_dir: str, shard: dict, out_indices: np.ndarray, out_evals: np.ndarray,
                  start: int, canonical: bool = False, shared: bool = False,
                  out_wdl: np.ndarray | None = None, ka: bool = False,
                  kav2: bool = False, kpp_acc: dict | None = None) -> int:
    path = os.path.join(run_dir, shard["file"])
    entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
    board = cshogi.Board()
    cursor = start
    for entry in entries:
        board.set_hcp(entry["hcp"])
        # kav2: [own(40) ++ opp(39)] (HalfKA minus anchor king, Task 053);
        # ka: [own(41) ++ opp(40)] (kings as features, Task 052);
        # shared: [own(39) ++ opp(38)]; otherwise sorted 77-set (naive/two-band).
        if kav2:
            indices = kp.canonical_shared_kav2_row(board)
        elif ka:
            indices = kp.canonical_shared_ka_row(board)
        elif shared:
            indices = kp.canonical_shared_row(board)
        else:
            indices = kp.board_to_kp_indices(board, canonical=canonical)
        if len(indices) != FEATURES_PER_POSITION:
            raise AssertionError(
                f"shard {shard['index']} position {cursor - start}: "
                f"{len(indices)} features != {FEATURES_PER_POSITION} "
                f"(sfen={board.sfen()})"
            )
        out_indices[cursor] = indices
        if kpp_acc is not None:
            # king-zone KPP is variable-length -> CSR (flat values + per-position
            # lengths), no padding. KP base row above is the HalfKP shared row.
            own, opp = kp.kpp_zone_halves(board)
            kpp_acc["own_flat"].extend(own)
            kpp_acc["opp_flat"].extend(opp)
            kpp_acc["own_len"][cursor] = len(own)
            kpp_acc["opp_len"][cursor] = len(opp)
        out_evals[cursor] = int(entry["eval"])
        if out_wdl is not None:
            # gameResult is absolute (0=Draw,1=BlackWin,2=WhiteWin); store the
            # side-to-move outcome: +1 win / 0 draw / -1 loss for the player to move.
            gr = int(entry["gameResult"])
            if gr == 0:
                out_wdl[cursor] = 0
            else:
                winner_black = (gr == 1)
                stm_black = (board.turn == cshogi.BLACK)
                out_wdl[cursor] = 1 if (winner_black == stm_black) else -1
        cursor += 1
    return cursor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="gen-teacher output dir (with manifest.json)")
    parser.add_argument("out", help="cache output dir")
    parser.add_argument(
        "--val-shards",
        type=int,
        default=5,
        help="number of trailing shards of run_dir held out for validation",
    )
    parser.add_argument(
        "--extra-runs", nargs="*", default=[],
        help="additional gen-teacher run dirs whose ALL done shards go to train "
             "(val always comes only from the primary run_dir, for a stable held-out)",
    )
    parser.add_argument(
        "--canonical", action="store_true",
        help="convert with canonical HalfKP numbering (Task 046, two-band)",
    )
    parser.add_argument(
        "--canonical-shared", action="store_true",
        help="convert with true canonical shared-W1 HalfKP (Task 047); fixed column "
             "layout own(39)++opp(38), total_features = VIEW_DIM+2",
    )
    parser.add_argument(
        "--canonical-shared-ka", action="store_true",
        help="convert with HalfKA shared band (Task 052, kings as features); fixed "
             "column layout own(41)++opp(40), total_features = VIEW_DIM_KA+2",
    )
    parser.add_argument(
        "--canonical-shared-kav2", action="store_true",
        help="convert with HalfKAv2 (Task 053, HalfKA minus anchor king); fixed "
             "column layout own(40)++opp(39), total_features = VIEW_DIM_KA+2",
    )
    parser.add_argument(
        "--kpp-zone", action="store_true",
        help="ALSO emit king-zone KPP pair features (Task 053): HalfKP shared KP base "
             "(77) + ragged CSR KPP own/opp (kpp_{own,opp}_{flat,len}.npy).",
    )
    args = parser.parse_args()
    kpp_zone = args.kpp_zone
    kav2 = args.canonical_shared_kav2
    ka = args.canonical_shared_ka
    # KPP uses the HalfKP shared row as its KP base.
    shared = args.canonical_shared or kpp_zone
    global FEATURES_PER_POSITION
    if kav2:
        FEATURES_PER_POSITION = 79
    elif ka:
        FEATURES_PER_POSITION = 81

    shards, meta = shard_files(args.run_dir)
    if args.val_shards <= 0 or args.val_shards >= len(shards):
        raise ValueError(f"--val-shards must be in 1..{len(shards) - 1}")

    train_shards = shards[: -args.val_shards]
    val_shards = shards[-args.val_shards :]
    # Extra runs: every done shard becomes train (no val from them).
    extra_groups = [(rd, shard_files(rd)[0]) for rd in args.extra_runs]
    n_train = sum(s["count"] for s in train_shards) + sum(
        s["count"] for _, ss in extra_groups for s in ss
    )
    n_val = sum(s["count"] for s in val_shards)

    print(f"run_dir={args.run_dir} canonical={args.canonical} canonical_shared={shared} canonical_shared_ka={ka} canonical_shared_kav2={kav2}")
    print(f"extra_runs={args.extra_runs}")
    print(f"primary_train_shards={len(train_shards)} val_shards={len(val_shards)} "
          f"extra_shards={sum(len(ss) for _, ss in extra_groups)}")
    print(f"val_shard_indices={[s['index'] for s in val_shards]}")
    print(f"n_train={n_train} n_val={n_val} features_per_position={FEATURES_PER_POSITION}")

    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    # train = primary train shards + all extra-run shards; val = primary val shards.
    train_groups = [(args.run_dir, train_shards)] + extra_groups
    for name, groups, n in (("train", train_groups, n_train), ("val", [(args.run_dir, val_shards)], n_val)):
        indices = np.zeros((n, FEATURES_PER_POSITION), dtype=np.int32)
        evals = np.zeros((n,), dtype=np.int16)
        wdl = np.zeros((n,), dtype=np.int8)  # side-to-move outcome: +1/0/-1 (0 if teacher has no result)
        kpp_acc = None
        if kpp_zone:
            # array.array('i') = 4 bytes/int (vs ~28 for a Python list) so the ~200M
            # KPP indices of an 8M cache fit in RAM.
            kpp_acc = {"own_flat": array.array("i"), "opp_flat": array.array("i"),
                       "own_len": np.zeros((n,), dtype=np.int32),
                       "opp_len": np.zeros((n,), dtype=np.int32)}
        cursor = 0
        done_shards = 0
        total_shards = sum(len(ss) for _, ss in groups)
        for run_dir, group in groups:
            for shard in group:
                cursor = convert_shard(run_dir, shard, indices, evals, cursor, args.canonical, shared, wdl, ka, kav2, kpp_acc)
                done_shards += 1
                if done_shards % 20 == 0 or done_shards == total_shards:
                    rate = cursor / max(1e-9, time.time() - started)
                    print(f"  {name}: {done_shards}/{total_shards} shards, {cursor} positions ({rate:.0f}/s)")
        if cursor != n:
            raise AssertionError(f"{name}: filled {cursor} != expected {n}")
        np.save(os.path.join(args.out, f"{name}_indices.npy"), indices)
        np.save(os.path.join(args.out, f"{name}_evals.npy"), evals)
        np.save(os.path.join(args.out, f"{name}_wdl.npy"), wdl)
        print(f"  {name}: saved indices {indices.shape} evals {evals.shape} "
              f"wdl(win/draw/loss frac)={np.mean(wdl>0):.2f}/{np.mean(wdl==0):.2f}/{np.mean(wdl<0):.2f}")
        if kpp_acc is not None:
            for half in ("own", "opp"):
                np.save(os.path.join(args.out, f"{name}_kpp_{half}_flat.npy"),
                        np.asarray(kpp_acc[f"{half}_flat"], dtype=np.int32))
                np.save(os.path.join(args.out, f"{name}_kpp_{half}_len.npy"), kpp_acc[f"{half}_len"])
            print(f"  {name}: saved KPP-zone own/opp (mean pairs/half "
                  f"{kpp_acc['own_len'].mean():.1f}/{kpp_acc['opp_len'].mean():.1f}, "
                  f"max {kpp_acc['own_len'].max()}/{kpp_acc['opp_len'].max()})")

    elapsed = time.time() - started
    out_meta = {
        "run_dir": os.path.abspath(args.run_dir),
        "extra_runs": [os.path.abspath(r) for r in args.extra_runs],
        "canonical": bool(args.canonical),
        "canonical_shared": bool(shared),
        "canonical_shared_ka": bool(ka),
        "canonical_shared_kav2": bool(kav2),
        "kpp_zone": bool(kpp_zone),
        "kpp_zone_total": kp.KPP_ZONE_TOTAL if kpp_zone else 0,
        "source_meta": meta,
        "val_shards": [s["index"] for s in val_shards],
        "train_shards": [s["index"] for s in train_shards],
        "n_train": n_train,
        "n_val": n_val,
        "features_per_position": FEATURES_PER_POSITION,
        "total_features": (kp.CANONICAL_SHARED_KA_TOTAL if (ka or kav2)
                           else kp.CANONICAL_SHARED_TOTAL if shared else kp.TOTAL_FEATURES),
        "convert_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    print(f"convert_seconds={elapsed:.1f}")
    print(f"convert_rate_per_s={(n_train + n_val) / elapsed:.0f}")
    print(f"cache_dir={args.out}")
    print("prepare_data=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
