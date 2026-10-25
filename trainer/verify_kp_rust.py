#!/usr/bin/env python3
"""Cross-check the Rust KP conversion against the Python ground truth.

For a diverse sample of positions (taken from hcpe shards, stride-sampled within
each file so opening..endgame are covered), this:

  1. computes the Python KP index set via trainer/kp.py (the ground truth the
     NNUE weights were trained against),
  2. emits each position's SFEN,
  3. runs the Rust `kp-from-sfen` binary on those SFENs,
  4. asserts the Rust index set equals the Python index set for every position.

A single differing index would silently invalidate the learned evaluation
(train-time and inference-time feature meanings would diverge), so the bar is
exact set equality on every position.

Exit code is non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp


def stride_indices(n: int, want: int) -> list[int]:
    if want <= 0 or n <= want:
        return list(range(n))
    step = n / want
    return sorted({min(n - 1, int(i * step)) for i in range(want)})


def collect_positions(hcpe_files: list[str], limit: int, canonical: bool = False,
                      shared: bool = False, ka: bool = False, kav2: bool = False,
                      kpp: bool = False):
    """Yield (sfen, python_kp_set) for up to `limit` positions, spread across files.

    For `shared` (canonical-shared, Task 047) or `ka` (HalfKA shared, Task 052),
    python_kp_set is the pair (frozenset(own), frozenset(opp)) instead of a single
    set."""
    per_file = max(1, (limit + len(hcpe_files) - 1) // len(hcpe_files))
    board = cshogi.Board()
    sfens: list[str] = []
    py_sets: list[frozenset] = []
    diversity = {"hands": 0, "promoted": 0, "edge_king": 0, "black": 0, "white": 0}
    king_squares: set[int] = set()
    promoted_types = frozenset((
        cshogi.PROM_PAWN, cshogi.PROM_LANCE, cshogi.PROM_KNIGHT,
        cshogi.PROM_SILVER, cshogi.PROM_BISHOP, cshogi.PROM_ROOK,
    ))
    for path in hcpe_files:
        entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        for i in stride_indices(len(entries), per_file):
            if len(sfens) >= limit:
                break
            board.set_hcp(entries[i]["hcp"])
            sfens.append(board.sfen())
            if kpp:
                own, opp = kp.kpp_zone_halves(board)
                py_sets.append((frozenset(own), frozenset(opp)))
            elif kav2:
                own, opp = kp.canonical_shared_kav2_halves(board)
                py_sets.append((frozenset(own), frozenset(opp)))
            elif ka:
                own, opp = kp.canonical_shared_ka_halves(board)
                py_sets.append((frozenset(own), frozenset(opp)))
            elif shared:
                own, opp = kp.canonical_shared_halves(board)
                py_sets.append((frozenset(own), frozenset(opp)))
            else:
                py_sets.append(frozenset(kp.board_to_kp_indices(board, canonical=canonical)))
            # diversity bookkeeping
            if sum(sum(h) for h in board.pieces_in_hand) > 0:
                diversity["hands"] += 1
            if any((p & 0x0F) in promoted_types for p in board.pieces if p != cshogi.NONE):
                diversity["promoted"] += 1
            for color in (cshogi.BLACK, cshogi.WHITE):
                ks = board.king_square(color)
                king_squares.add(ks)
                if (ks // 9) in (0, 8) or (ks % 9) in (0, 8):
                    diversity["edge_king"] += 1
                    break
            diversity["black" if board.turn == cshogi.BLACK else "white"] += 1
        if len(sfens) >= limit:
            break
    return sfens, py_sets, diversity, len(king_squares)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hcpe", nargs="+", help="hcpe shard file(s) to sample from")
    parser.add_argument("--limit", type=int, default=50000, help="positions to check")
    parser.add_argument(
        "--binary",
        default="target/release/binbo-shogi",
        help="path to the binbo-shogi binary with kp-from-sfen",
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--canonical", action="store_true",
                        help="check canonical HalfKP numbering (Task 046, two-band)")
    parser.add_argument("--canonical-shared", action="store_true",
                        help="check true canonical shared-W1 HalfKP (Task 047)")
    parser.add_argument("--canonical-shared-ka", action="store_true",
                        help="check HalfKA shared band (Task 052, kings as features)")
    parser.add_argument("--canonical-shared-kav2", action="store_true",
                        help="check HalfKAv2 (Task 053, HalfKA minus anchor king)")
    parser.add_argument("--kpp-zone", action="store_true",
                        help="check king-zone KPP pair indices (Task 053)")
    args = parser.parse_args()

    kpp = args.kpp_zone
    kav2 = args.canonical_shared_kav2
    ka = args.canonical_shared_ka
    # KA/KAv2/KPP reuse the (own, opp) pair comparison path, so treat as `shared`.
    shared = args.canonical_shared or ka or kav2 or kpp
    print(f"binary={args.binary} canonical={args.canonical} canonical_shared={args.canonical_shared} ka={ka} kav2={kav2} kpp={kpp}")
    print(f"hcpe_files={len(args.hcpe)} limit={args.limit}")
    sfens, py_sets, diversity, n_king_sq = collect_positions(
        args.hcpe, args.limit, args.canonical, shared, ka, kav2, kpp)
    n = len(sfens)
    if n == 0:
        print("no positions", file=sys.stderr)
        return 1
    print(f"positions={n}")

    with tempfile.TemporaryDirectory() as tmp:
        sfen_path = os.path.join(tmp, "sfens.txt")
        out_path = os.path.join(tmp, "rust_kp.txt")
        with open(sfen_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sfens) + "\n")

        kp_cmd = [args.binary, "kp-from-sfen", sfen_path, out_path]
        if kpp:
            kp_cmd.append("kpp-zone")
        elif kav2:
            kp_cmd.append("canonical-shared-kav2")
        elif ka:
            kp_cmd.append("canonical-shared-ka")
        elif args.canonical_shared:
            kp_cmd.append("canonical-shared")
        elif args.canonical:
            kp_cmd.append("canonical")
        proc = subprocess.run(kp_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("rust kp-from-sfen failed:", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 1

        with open(out_path, encoding="utf-8") as f:
            rust_lines = f.read().splitlines()

    if len(rust_lines) != n:
        print(f"line count mismatch: rust={len(rust_lines)} python={n}", file=sys.stderr)
        return 1

    mismatches = 0
    examples: list[str] = []
    for i, (line, py_set) in enumerate(zip(rust_lines, py_sets)):
        if shared:
            # Rust line is "own_indices | opp_indices"; compare each half as a set.
            own_s, _, opp_s = line.partition("|")
            rust_pair = (
                frozenset(int(x) for x in own_s.split()),
                frozenset(int(x) for x in opp_s.split()),
            )
            ok = rust_pair == py_set
            if not ok and len(examples) < args.max_examples:
                examples.append(
                    f"pos {i} sfen='{sfens[i]}'\n"
                    f"    own only_py={sorted(py_set[0]-rust_pair[0])[:6]} only_rust={sorted(rust_pair[0]-py_set[0])[:6]}\n"
                    f"    opp only_py={sorted(py_set[1]-rust_pair[1])[:6]} only_rust={sorted(rust_pair[1]-py_set[1])[:6]}"
                )
            if not ok:
                mismatches += 1
            continue
        rust_set = frozenset(int(x) for x in line.split()) if line.strip() else frozenset()
        if rust_set != py_set:
            mismatches += 1
            if len(examples) < args.max_examples:
                only_py = sorted(py_set - rust_set)[:8]
                only_rust = sorted(rust_set - py_set)[:8]
                examples.append(
                    f"pos {i} sfen='{sfens[i]}'\n"
                    f"    only_in_python={only_py}\n"
                    f"    only_in_rust={only_rust}"
                )

    print("")
    print("=== diversity of checked positions ===")
    print(f"turn black={diversity['black']} white={diversity['white']}")
    print(f"with_hand_pieces={diversity['hands']} with_promoted={diversity['promoted']} "
          f"edge_king={diversity['edge_king']}")
    print(f"distinct_king_squares_seen={n_king_sq}/81")

    print("")
    print("=== KP cross-check (Rust vs Python) ===")
    print(f"checked={n}")
    print(f"mismatches={mismatches}")
    print(f"match_rate={n - mismatches}/{n} ({100.0 * (n - mismatches) / n:.4f}%)")
    if examples:
        print("--- mismatch examples ---")
        for e in examples:
            print(e)

    print("")
    if mismatches == 0:
        print("kp_rust_matches_python=ok")
        return 0
    print("kp_rust_matches_python=FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
