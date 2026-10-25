#!/usr/bin/env python3
"""Bulk KP reversibility verification over many diverse positions.

Reads an hcpe file (e.g. produced by `binbo-shogi hcpe-selfplay`), and for every
position checks:

  1. reversibility: 局面 -> KP インデックス集合 -> 局面 を復元し、SFEN core が元と一致。
  2. feature count: 立っている KP インデックス数が期待式と一致。

External answer tables are not needed: the roundtrip self-verifies, so the number
of positions can be scaled freely. The script also prints diversity statistics so
that "大量" を超えて "多様" であることを示す。

Exit code is non-zero if any position fails (reversibility ずれ or feature 数ずれ)。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp


PROMOTED_BOARD_TYPES = frozenset(
    (
        cshogi.PROM_PAWN,
        cshogi.PROM_LANCE,
        cshogi.PROM_KNIGHT,
        cshogi.PROM_SILVER,
        cshogi.PROM_BISHOP,
        cshogi.PROM_ROOK,
    )
)
HAND_KIND_NAMES = {
    cshogi.HPAWN: "P",
    cshogi.HLANCE: "L",
    cshogi.HKNIGHT: "N",
    cshogi.HSILVER: "S",
    cshogi.HGOLD: "G",
    cshogi.HBISHOP: "B",
    cshogi.HROOK: "R",
}


def square_on_edge(square: int) -> bool:
    file_index = square // 9
    rank_index = square % 9
    return file_index in (0, 8) or rank_index in (0, 8)


def bucketize(value: int, edges: tuple[int, ...]) -> str:
    """Return a human-readable bucket label for a histogram."""
    lo = 0
    for hi in edges:
        if value <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
        lo = hi + 1
    return f"{lo}+"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hcpe", help="hcpe file to verify")
    parser.add_argument("--limit", type=int, default=None, help="verify only the first N entries")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="how many mismatching positions to print",
    )
    args = parser.parse_args()

    entries = np.fromfile(args.hcpe, dtype=cshogi.HuffmanCodedPosAndEval)
    if args.limit is not None:
        entries = entries[: args.limit]
    total = len(entries)
    if total == 0:
        print("no entries to verify", file=sys.stderr)
        return 1

    print(f"kp_total_features={kp.TOTAL_FEATURES}")
    print(f"kp_view_dim={kp.VIEW_DIM}")
    print(f"verifying entries={total} from {args.hcpe}")

    reversibility_mismatches = 0
    feature_count_mismatches = 0
    mismatch_examples: list[str] = []

    distinct_positions: set[str] = set()
    turn_counter: Counter[int] = Counter()
    feature_counts: list[int] = []
    feature_count_hist: Counter[str] = Counter()
    hand_total_hist: Counter[str] = Counter()
    board_piece_hist: Counter[str] = Counter()

    positions_with_promoted = 0
    positions_with_any_hand = 0
    positions_with_big_hand = 0  # >= 8 pieces in hand: deep midgame/endgame proxy
    positions_with_king_on_edge = 0

    max_hand_total = 0
    max_hand_per_kind: dict[int, int] = {kind: 0 for kind in HAND_KIND_NAMES}
    self_king_squares: set[int] = set()
    opp_king_squares: set[int] = set()

    board = cshogi.Board()
    for index in range(total):
        board.set_hcp(entries[index]["hcp"])

        indices = kp.board_to_kp_indices(board)
        original = kp.sfen_core(board.sfen())
        try:
            restored = kp.sfen_core(kp.kp_indices_to_sfen(indices))
        except Exception as err:  # noqa: BLE001 - report which position broke
            restored = f"<error: {err}>"

        if restored != original:
            reversibility_mismatches += 1
            if len(mismatch_examples) < args.max_examples:
                mismatch_examples.append(
                    f"index={index} reversibility original='{original}' restored='{restored}'"
                )

        expected_count = kp.expected_active_feature_count(board)
        if len(indices) != expected_count:
            feature_count_mismatches += 1
            if len(mismatch_examples) < args.max_examples:
                mismatch_examples.append(
                    f"index={index} feature_count got={len(indices)} "
                    f"expected={expected_count} sfen='{original}'"
                )

        # --- diversity statistics ---
        distinct_positions.add(original)
        turn = board.turn
        turn_counter[turn] += 1
        feature_counts.append(len(indices))
        feature_count_hist[bucketize(len(indices), (40, 60, 80, 100, 120))] += 1

        has_promoted = False
        non_king_board_pieces = 0
        for piece in board.pieces:
            if piece == cshogi.NONE:
                continue
            piece_type = piece & 0x0F
            if piece_type == cshogi.KING:
                continue
            non_king_board_pieces += 1
            if piece_type in PROMOTED_BOARD_TYPES:
                has_promoted = True
        if has_promoted:
            positions_with_promoted += 1
        board_piece_hist[bucketize(non_king_board_pieces, (10, 20, 30, 38))] += 1

        hand_total = 0
        for color in (cshogi.BLACK, cshogi.WHITE):
            hand = board.pieces_in_hand[color]
            for kind, name in HAND_KIND_NAMES.items():
                count = hand[kind]
                hand_total += count
                if count > max_hand_per_kind[kind]:
                    max_hand_per_kind[kind] = count
        if hand_total > 0:
            positions_with_any_hand += 1
        if hand_total >= 8:
            positions_with_big_hand += 1
        max_hand_total = max(max_hand_total, hand_total)
        hand_total_hist[bucketize(hand_total, (0, 3, 7, 11, 15))] += 1

        self_king = kp.normalize_square(board.king_square(turn), turn)
        opp_king = kp.normalize_square(
            board.king_square(kp.opposite_color(turn)), turn
        )
        self_king_squares.add(self_king)
        opp_king_squares.add(opp_king)
        if square_on_edge(board.king_square(cshogi.BLACK)) or square_on_edge(
            board.king_square(cshogi.WHITE)
        ):
            positions_with_king_on_edge += 1

        if (index + 1) % 10000 == 0:
            print(f"  ...verified {index + 1}/{total}", file=sys.stderr)

    # --- report ---
    print("")
    print("=== reversibility ===")
    print(f"positions={total}")
    print(f"reversibility_mismatches={reversibility_mismatches}")
    print(f"feature_count_mismatches={feature_count_mismatches}")
    matched = total - reversibility_mismatches
    print(f"reversible_match_rate={matched}/{total} ({100.0 * matched / total:.4f}%)")

    print("")
    print("=== diversity ===")
    print(f"distinct_positions={len(distinct_positions)} ({100.0 * len(distinct_positions) / total:.2f}% of total)")
    print(
        f"turn black_to_move={turn_counter[cshogi.BLACK]} "
        f"white_to_move={turn_counter[cshogi.WHITE]}"
    )
    print(
        f"promoted_piece_positions={positions_with_promoted} "
        f"({100.0 * positions_with_promoted / total:.2f}%)"
    )
    print(
        f"any_hand_positions={positions_with_any_hand} "
        f"({100.0 * positions_with_any_hand / total:.2f}%)"
    )
    print(
        f"big_hand_positions(>=8 in hand)={positions_with_big_hand} "
        f"({100.0 * positions_with_big_hand / total:.2f}%)"
    )
    print(
        f"king_on_edge_positions={positions_with_king_on_edge} "
        f"({100.0 * positions_with_king_on_edge / total:.2f}%)"
    )
    print(f"max_hand_total={max_hand_total}")
    max_hand_desc = " ".join(
        f"{HAND_KIND_NAMES[kind]}={max_hand_per_kind[kind]}" for kind in HAND_KIND_NAMES
    )
    print(f"max_hand_per_kind {max_hand_desc}")
    print(
        f"distinct_self_king_squares={len(self_king_squares)}/81 "
        f"distinct_opponent_king_squares={len(opp_king_squares)}/81"
    )

    print("")
    print("=== feature count distribution ===")
    print(
        f"feature_count min={min(feature_counts)} max={max(feature_counts)} "
        f"mean={sum(feature_counts) / total:.2f}"
    )
    for label in sorted_buckets(feature_count_hist):
        print(f"  features {label}: {feature_count_hist[label]}")

    print("")
    print("=== pieces-in-hand distribution ===")
    for label in sorted_buckets(hand_total_hist):
        print(f"  hand_total {label}: {hand_total_hist[label]}")

    print("")
    print("=== board non-king piece-count distribution ===")
    for label in sorted_buckets(board_piece_hist):
        print(f"  board_pieces {label}: {board_piece_hist[label]}")

    if mismatch_examples:
        print("")
        print("=== mismatch examples ===")
        for line in mismatch_examples:
            print(f"  {line}")

    print("")
    if reversibility_mismatches == 0 and feature_count_mismatches == 0:
        print("kp_bulk_verification=ok")
        return 0
    print("kp_bulk_verification=FAILED")
    return 1


def sorted_buckets(hist: Counter[str]) -> list[str]:
    def key(label: str) -> int:
        head = label.split("-")[0].rstrip("+")
        return int(head)

    return sorted(hist, key=key)


if __name__ == "__main__":
    raise SystemExit(main())
