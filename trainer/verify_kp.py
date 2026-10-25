#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp


def load_expected(path: str | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def verify_hcpe(path: str, expected_tsv: str | None) -> None:
    entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
    expected = load_expected(expected_tsv)
    if expected is not None and len(expected) != len(entries):
        raise AssertionError(f"entry count mismatch: {len(entries)} != {len(expected)}")

    for index, entry in enumerate(entries):
        board = cshogi.Board()
        board.set_hcp(entry["hcp"])

        indices = kp.board_to_kp_indices(board)
        restored = kp.sfen_core(kp.kp_indices_to_sfen(indices))
        original = kp.sfen_core(board.sfen())
        if restored != original:
            raise AssertionError(
                f"reversible check failed index={index}: restored={restored} original={original}"
            )

        expected_count = kp.expected_active_feature_count(board)
        if len(indices) != expected_count:
            raise AssertionError(
                f"feature count mismatch index={index}: got={len(indices)} expected={expected_count}"
            )

        eval_value = int(entry["eval"])
        if expected is not None and eval_value != int(expected[index]["eval"]):
            raise AssertionError(
                f"eval mismatch index={index}: got={eval_value} expected={expected[index]['eval']}"
            )

        name = expected[index]["name"] if expected is not None else str(index)
        print(
            "ok "
            f"index={index} name={name} features={len(indices)} "
            f"eval={eval_value} sfen='{original}'"
        )


def verify_side_to_move_normalization() -> None:
    board_part = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPP1/1B5R1/LNSGKGSNL"
    black_turn = cshogi.Board(f"{board_part} b P 1")
    white_turn = cshogi.Board(f"{board_part} w P 1")

    black_indices = set(kp.board_to_kp_indices(black_turn))
    white_indices = set(kp.board_to_kp_indices(white_turn))

    black_self_king = kp.normalize_square(black_turn.king_square(cshogi.BLACK), cshogi.BLACK)
    white_self_king = kp.normalize_square(white_turn.king_square(cshogi.WHITE), cshogi.WHITE)

    black_hand_as_self = kp.hand_feature_index(
        view=kp.VIEW_SELF_KING,
        king_square=black_self_king,
        rel_owner=kp.SELF,
        hand_kind=cshogi.HPAWN,
        slot=0,
    )
    white_turn_black_hand_as_opponent = kp.hand_feature_index(
        view=kp.VIEW_SELF_KING,
        king_square=white_self_king,
        rel_owner=kp.OPPONENT,
        hand_kind=cshogi.HPAWN,
        slot=0,
    )

    black_pawn_7g_as_self = kp.board_feature_index(
        view=kp.VIEW_SELF_KING,
        king_square=black_self_king,
        rel_owner=kp.SELF,
        piece_type=cshogi.PAWN,
        piece_square=cshogi.G7,
    )
    white_turn_black_pawn_7g_as_opponent = kp.board_feature_index(
        view=kp.VIEW_SELF_KING,
        king_square=white_self_king,
        rel_owner=kp.OPPONENT,
        piece_type=cshogi.PAWN,
        piece_square=kp.normalize_square(cshogi.G7, cshogi.WHITE),
    )

    checks = [
        ("black_turn_black_hand_is_self", black_hand_as_self, black_indices),
        (
            "white_turn_black_hand_is_opponent",
            white_turn_black_hand_as_opponent,
            white_indices,
        ),
        ("black_turn_black_pawn_7g_is_self", black_pawn_7g_as_self, black_indices),
        (
            "white_turn_black_pawn_7g_is_opponent_rotated_to_3c",
            white_turn_black_pawn_7g_as_opponent,
            white_indices,
        ),
        (
            "black_side_feature",
            kp.SIDE_FEATURE_BASE + cshogi.BLACK,
            black_indices,
        ),
        (
            "white_side_feature",
            kp.SIDE_FEATURE_BASE + cshogi.WHITE,
            white_indices,
        ),
    ]
    for label, feature, indices in checks:
        if feature not in indices:
            raise AssertionError(f"missing side-to-move normalization feature: {label}")

    if black_hand_as_self in white_indices:
        raise AssertionError("black hand stayed SELF when side-to-move changed to white")
    if white_turn_black_hand_as_opponent in black_indices:
        raise AssertionError("opponent hand feature leaked into black-turn encoding")

    print(
        "ok side_to_move_normalization "
        "black_hand=self_when_black_to_move "
        "black_hand=opponent_when_white_to_move "
        "black_pawn_7g=opponent_3c_when_white_to_move"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hcpe")
    parser.add_argument("expected_tsv", nargs="?")
    args = parser.parse_args()

    print(f"kp_total_features={kp.TOTAL_FEATURES}")
    print(f"kp_view_dim={kp.VIEW_DIM}")
    print(f"kp_features_per_king={kp.FEATURES_PER_KING}")
    verify_hcpe(args.hcpe, args.expected_tsv)
    verify_side_to_move_normalization()
    print("kp_verification=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
