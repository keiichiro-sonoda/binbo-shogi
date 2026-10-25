# Task 077 M2: USI move string <-> Pgx shogi action id (direction*81 + to).
#
# Pgx square index == cshogi/YaneuraOu square index (file-1)*9 + (rank-1); the board is
# side-to-move POV (flipped every ply), so white's squares mirror via Inv = 80 - sq.
# Directions are NOT hardcoded: DIR_TABLE is derived from Pgx's own LEGAL_FROM_IDX
# constant, and from-square resolution replicates Action._from_dlshogi_action
# (first masked candidate) in numpy. Verified against cshogi legal_moves as ground
# truth by m2_verify_moves.py (the 057 numbering-silent-failure lesson).

import numpy as np

from pgx._src.games.shogi import LEGAL_FROM_IDX

HAND_LETTERS = "PLNSBRG"  # drop piece ids 0-6 == direction - 20
_LEGAL_FROM = np.asarray(LEGAL_FROM_IDX)  # (10, 81, 8)

# (from_pov, to_pov) -> direction 0..9 (unique; assert while building)
DIR_TABLE = np.full((81, 81), -1, dtype=np.int8)
for _d in range(10):
    for _to in range(81):
        for _frm in _LEGAL_FROM[_d, _to]:
            if _frm >= 0:
                assert DIR_TABLE[_frm, _to] in (-1, _d), (_frm, _to, _d)
                DIR_TABLE[_frm, _to] = _d


def sq_to_usi(sq):
    """absolute square -> e.g. '7g'"""
    return f"{sq // 9 + 1}{chr(ord('a') + sq % 9)}"


def usi_to_sq(s):
    """'7g' -> absolute square"""
    return (int(s[0]) - 1) * 9 + (ord(s[1]) - ord("a"))


def pov(sq, color):
    """absolute <-> side-to-move POV (involution: white mirrors)"""
    return sq if color == 0 else 80 - sq


def usi_to_action(usi, color):
    """USI move -> Pgx action id for the side-to-move `color` (0=black, 1=white)."""
    if usi[1] == "*":  # drop, e.g. 'P*5e'
        piece = HAND_LETTERS.index(usi[0])
        return (20 + piece) * 81 + pov(usi_to_sq(usi[2:4]), color)
    frm = pov(usi_to_sq(usi[0:2]), color)
    to = pov(usi_to_sq(usi[2:4]), color)
    d = int(DIR_TABLE[frm, to])
    if d < 0:
        raise ValueError(f"no direction for {usi} (pov from={frm} to={to})")
    if usi.endswith("+"):
        d += 10
    return d * 81 + to


def action_to_usi(action, board_pov, color):
    """Pgx action id -> USI move. `board_pov` = GameState.board (side-to-move POV,
    numpy (81,)). From-square resolution replicates Action._from_dlshogi_action:
    first LEGAL_FROM_IDX candidate holding one of my pieces (0 <= piece < 14)."""
    direction, to = action // 81, action % 81
    if direction >= 20:  # drop
        return f"{HAND_LETTERS[direction - 20]}*{sq_to_usi(pov(to, color))}"
    cand = _LEGAL_FROM[direction % 10, to]
    mask = (cand >= 0) & (board_pov[cand] >= 0) & (board_pov[cand] < 14)
    if not mask.any():
        raise ValueError(f"no from-square for action {action}")
    frm = int(cand[int(np.argmax(mask))])
    suffix = "+" if 10 <= direction < 20 else ""
    return sq_to_usi(pov(frm, color)) + sq_to_usi(pov(to, color)) + suffix
