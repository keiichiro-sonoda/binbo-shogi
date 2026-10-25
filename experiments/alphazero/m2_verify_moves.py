# Task 077 M2: verify m2_moves USI<->action mapping against cshogi legal_moves
# (ground truth), the 057 lesson: numbering bugs are silent, so verify with power.
#
# For N positions from random cshogi games:
#   setA = { action_to_usi(a) : a in Pgx legal_action_mask }   (Pgx side)
#   setB = { move_to_usi(m)  : m in cshogi board.legal_moves } (ground truth)
#   assert setA == setB, plus usi_to_action(usi) round-trips into the mask.
# --negative-control flips one DIR_TABLE entry and expects a detection (RED).

import argparse
import sys

import numpy as np
import cshogi

import m2_moves
from pgx.experimental.shogi import from_sfen

STARTPOS_MOVES = 30  # known fact: shogi startpos has exactly 30 legal moves


def positions_from_random_games(n_positions, seed, max_plies=180, sample_every=7):
    rng = np.random.default_rng(seed)
    board = cshogi.Board()
    out = []
    while len(out) < n_positions:
        board.reset()
        for ply in range(max_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[rng.integers(len(moves))])
            if ply % sample_every == 0 and list(board.legal_moves):
                out.append(board.sfen())
                if len(out) >= n_positions:
                    break
    return out


def check_position(sfen, gt_board):
    state = from_sfen(sfen)
    color = int(state._x.color)
    board_pov = np.asarray(state._x.board)
    mask = np.asarray(state.legal_action_mask)
    actions = np.nonzero(mask)[0]

    set_pgx = {m2_moves.action_to_usi(int(a), board_pov, color) for a in actions}
    set_gt = {cshogi.move_to_usi(m) for m in gt_board.legal_moves}
    if set_pgx != set_gt:
        return f"legal-set mismatch: {sfen}\n  pgx-only={sorted(set_pgx - set_gt)}\n  gt-only={sorted(set_gt - set_pgx)}"

    for usi in set_gt:  # reverse direction + roundtrip
        a = m2_moves.usi_to_action(usi, color)
        if not mask[a]:
            return f"usi_to_action({usi}) -> {a} not legal in Pgx mask: {sfen}"
        if m2_moves.action_to_usi(a, board_pov, color) != usi:
            return f"roundtrip failed for {usi}: {sfen}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--negative-control", action="store_true")
    args = ap.parse_args()

    if args.negative_control:  # corrupt an entry that is guaranteed to be exercised
        frm, to = m2_moves.usi_to_sq("7g"), m2_moves.usi_to_sq("7f")  # startpos pawn push
        m2_moves.DIR_TABLE[frm, to] = (m2_moves.DIR_TABLE[frm, to] + 1) % 10

    # startpos: known legal-move count
    b = cshogi.Board()
    err = check_position(b.sfen(), b)
    n_start = len({cshogi.move_to_usi(m) for m in b.legal_moves})
    assert n_start == STARTPOS_MOVES, n_start
    if err:
        print(f"RED at startpos: {err}")
        sys.exit(1)
    print(f"startpos: {STARTPOS_MOVES} moves, sets match")

    sfens = positions_from_random_games(args.positions, args.seed)
    gt = cshogi.Board()
    n_hand = n_promo = 0
    for i, sfen in enumerate(sfens):
        gt.set_sfen(sfen)
        err = check_position(sfen, gt)
        if err:
            print(f"RED @{i}: {err}")
            sys.exit(1)
        parts = sfen.split()
        n_hand += parts[2] != "-"
        n_promo += "+" in parts[0]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(sfens)} ok", flush=True)
    print(f"GREEN: {len(sfens)} positions, legal sets identical + roundtrip ok "
          f"(with hands: {n_hand}, with promoted pieces on board: {n_promo})")


if __name__ == "__main__":
    main()
