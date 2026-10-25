#!/usr/bin/env python3
"""HalfKP feature-index remap between binbo's KP numbering and YaneuraOu BonaPiece (Task 057).

This is the crux of the bridge: a wrong map loads fine but makes the eval silently
meaningless. We build ``R: binbo_shared_index -> YaneuraOu HalfKP feature index`` and
*verify* it by an arithmetic-free oracle: for many positions, binbo's active indices
(``kp.canonical_shared_halves``, the trusted side) remapped through R must equal an
INDEPENDENT YaneuraOu active-set computed straight from the board via BonaPiece.

Conventions that line up (verified in source, Task 057):
- Square numbering identical: cshogi/binbo and YaneuraOu both use sq=(file-1)*9+(rank-1),
  and 180-deg rotation is ``80-sq`` (YaneuraOu ``Inv``).
- binbo's side-to-move-frame normalization == YaneuraOu's per-perspective view
  (``piece_list_fb``/``piece_list_fw``). So binbo own half == YaneuraOu stm perspective,
  binbo opp half == YaneuraOu non-stm perspective.
- YaneuraOu HalfKP index = ``fe_end(1548) * king_sq + BonaPiece`` (half_kp.cpp MakeIndex).
- fe_end=1548 => DISTINGUISH_GOLDS OFF: +P/+L/+N/+S all map to GOLD (9 board types),
  while binbo keeps 13 distinct board piece types -> the map collapses pro-golds to gold.

BonaPiece bases are taken verbatim from ``source/evaluate.h`` (enum BonaPiece).
"""
from __future__ import annotations

import sys

import cshogi
import numpy as np

from trainer import kp

FE_END = 1548
SQ_NB = 81

# Board piece f_base (e_base = f_base + 81). evaluate.h: f_pawn=90, +81 per (f/e, type).
F_BOARD = {
    cshogi.PAWN: 90, cshogi.LANCE: 252, cshogi.KNIGHT: 414, cshogi.SILVER: 576,
    cshogi.GOLD: 738, cshogi.BISHOP: 900, cshogi.ROOK: 1224,
    cshogi.PROM_PAWN: 738, cshogi.PROM_LANCE: 738, cshogi.PROM_KNIGHT: 738,
    cshogi.PROM_SILVER: 738,                       # promoted small pieces -> gold
    cshogi.PROM_BISHOP: 1062,                      # horse
    cshogi.PROM_ROOK: 1386,                        # dragon
}
# Hand bases are NOT f+81 (Apery layout). evaluate.h f_hand_*/e_hand_*.
F_HAND = {cshogi.HPAWN: 1, cshogi.HLANCE: 39, cshogi.HKNIGHT: 49, cshogi.HSILVER: 59,
          cshogi.HGOLD: 69, cshogi.HBISHOP: 79, cshogi.HROOK: 85}
E_HAND = {cshogi.HPAWN: 20, cshogi.HLANCE: 44, cshogi.HKNIGHT: 54, cshogi.HSILVER: 64,
          cshogi.HGOLD: 74, cshogi.HBISHOP: 82, cshogi.HROOK: 88}

YANE_KIN = SQ_NB * FE_END  # 125388


def _inv(sq: int) -> int:
    return SQ_NB - 1 - sq


def build_remap() -> np.ndarray:
    """R[binbo_shared_index] -> YaneuraOu HalfKP index in [0, 125388), or -1 if none.

    Enumerates binbo's board+hand feature space (view=0 shared band) using binbo's own
    index formulas, and assigns the corresponding YaneuraOu index. binbo's side-feature
    rows (VIEW_DIM, VIEW_DIM+1) and impossible-but-unused rows stay -1 / mapped-but-zero.
    """
    R = np.full(kp.VIEW_DIM, -1, dtype=np.int64)
    for king in range(SQ_NB):
        for rel in (0, 1):
            for ptype in kp.BOARD_PIECE_TYPES:
                fb = F_BOARD[ptype] + (0 if rel == 0 else 81)
                for sq in range(SQ_NB):
                    bi = kp.board_feature_index(view=0, king_square=king, rel_owner=rel,
                                                piece_type=ptype, piece_square=sq)
                    R[bi] = FE_END * king + fb + sq
            for hk in kp.HAND_PIECE_TYPES:
                base = F_HAND[hk] if rel == 0 else E_HAND[hk]
                maxc = kp.HAND_MAX_COUNTS[kp.HAND_PIECE_TYPES.index(hk)]
                for slot in range(maxc):
                    bi = kp.hand_feature_index(view=0, king_square=king, rel_owner=rel,
                                               hand_kind=hk, slot=slot)
                    R[bi] = FE_END * king + base + slot
    return R


def yane_active(board: cshogi.Board) -> tuple[set[int], set[int]]:
    """Independent YaneuraOu HalfKP active-index sets (stm, non-stm) straight from the board.

    For perspective color P: king index = (P's king) seen from P (Inv if P is white);
    a piece is 'friend' (f_*) iff its color == P, else enemy (e_*); squares are Inv'd
    when P is white. Mirrors half_kp.cpp GetPieces/MakeIndex with piece_list_fb/fw.
    """
    persp: dict[int, set[int]] = {}
    for pcol in (cshogi.BLACK, cshogi.WHITE):
        k_abs = board.king_square(pcol)
        k = k_abs if pcol == cshogi.BLACK else _inv(k_abs)
        s: set[int] = set()
        for sq, piece in enumerate(board.pieces):
            if piece == cshogi.NONE:
                continue
            pt = piece & 0x0F
            if pt == cshogi.KING:
                continue
            c = kp.piece_color(piece)
            friend = (c == pcol)
            base = F_BOARD[pt] + (0 if friend else 81)
            psq = sq if pcol == cshogi.BLACK else _inv(sq)
            s.add(FE_END * k + base + psq)
        for c in (cshogi.BLACK, cshogi.WHITE):
            hand = board.pieces_in_hand[c]
            friend = (c == pcol)
            tbl = F_HAND if friend else E_HAND
            for hk in kp.HAND_PIECE_TYPES:
                for slot in range(hand[hk]):
                    s.add(FE_END * k + tbl[hk] + slot)
        persp[pcol] = s
    stm = board.turn
    return persp[stm], persp[1 - stm]


# Positions for the active-index correspondence oracle. Asymmetric ones matter most
# (startpos is 180-symmetric so it cannot catch a rotation bug).
TEST_SFENS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",  # startpos
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",  # after 7g7f-ish
    "ln1gkgsnl/1r5b1/p1pppp1pp/1p4p2/9/2P6/PPSPPPPPP/1B5R1/LN1GKGSNL b - 7",
    "l6nl/4g1gk1/2n1sp1p1/p1p1p1p1p/1P5P1/2P1P1P1P/PRSPSP3/2G2K3/LN1G3NL b Bbrs2p 49",  # midgame, hands, asymmetric
    "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124",  # late, promotions, hands
]


def main() -> int:
    R = build_remap()
    mapped = int((R >= 0).sum())
    print(f"remap built: VIEW_DIM={kp.VIEW_DIM} mapped={mapped} "
          f"unmapped(side/impossible)={kp.VIEW_DIM - mapped} "
          f"yane_range=[{R[R >= 0].min()},{R[R >= 0].max()}] (expect <{YANE_KIN})")
    assert R[R >= 0].max() < YANE_KIN

    ok = True
    for sfen in TEST_SFENS:
        b = cshogi.Board(sfen)
        own, opp = kp.canonical_shared_halves(b)
        r_own = {int(R[i]) for i in own if i < kp.VIEW_DIM and R[i] >= 0}
        r_opp = {int(R[i]) for i in opp if i < kp.VIEW_DIM and R[i] >= 0}
        ystm, ynonstm = yane_active(b)
        m_own = (r_own == ystm)
        m_opp = (r_opp == ynonstm)
        ok = ok and m_own and m_opp
        print(f"  own={len(r_own)}/{len(ystm)} {'OK' if m_own else 'MISMATCH'}  "
              f"opp={len(r_opp)}/{len(ynonstm)} {'OK' if m_opp else 'MISMATCH'}  "
              f"{sfen.split(' ')[0][:24]}")
        if not m_own:
            print(f"    own only-in-binbo: {sorted(r_own - ystm)[:6]}")
            print(f"    own only-in-yane : {sorted(ystm - r_own)[:6]}")
        if not m_opp:
            print(f"    opp only-in-binbo: {sorted(r_opp - ynonstm)[:6]}")
            print(f"    opp only-in-yane : {sorted(ynonstm - r_opp)[:6]}")
    print("numbering correspondence:", "GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
