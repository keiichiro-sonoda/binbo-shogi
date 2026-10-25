"""Left-right (horizontal) mirror of shogi positions, for data augmentation.

Shogi pieces are all left-right symmetric (no piece distinguishes left from
right), so mirroring the files (1<->9, 2<->8, 3<->7, 4<->6, 5 fixed; ranks and
hands and side-to-move unchanged) yields an exactly equivalent position with the
same evaluation. Task 036 uses this to augment training: a position and its
mirror share the same target cp, which doubles the data and forces a "mirror
invariance" that improves generalization (the diagnosed bottleneck, Task 035).

Two equivalent flips are provided and cross-checked in `verify_mirror.py`:

- `flip_sfen`: board-level mirror (reverse the files in each SFEN rank). Ground
  truth; reconstructs the board, so `board_to_kp_indices(flip)` is by definition
  the KP of the mirrored position.
- `build_mirror_perm`: a permutation of the 353486 KP feature indices, so a
  position's cached KP indices can be flipped with a single gather (fast, used in
  training). It mirrors the (normalized) king and piece squares in feature space;
  this matches the board-level flip because the horizontal mirror commutes with
  the 180-degree `normalize_square`.
"""
from __future__ import annotations

import os

import numpy as np

try:
    from . import kp
except ImportError:
    import kp


def mirror_square(square: int) -> int:
    """Horizontal mirror of a cshogi square index (file f -> 10-f, rank fixed).

    cshogi square = (file-1)*9 + rank, so `square // 9` is the file index and
    `square % 9` is the rank; mirroring flips the file index (0..8 -> 8-..)."""
    return (8 - square // 9) * 9 + (square % 9)


def mirror_feature_index(index: int) -> int:
    """Map a KP feature index to its left-right mirror. Side feature is unchanged;
    board/hand features keep view/owner/type and mirror the (normalized) king
    square (and piece square for board features)."""
    if index >= kp.SIDE_FEATURE_BASE:
        return index
    d = kp.decode_feature_index(index)
    mks = mirror_square(d.king_square)
    if d.kind == "board":
        return kp.board_feature_index(
            view=d.view, king_square=mks, rel_owner=d.rel_owner,
            piece_type=d.piece_type, piece_square=mirror_square(d.piece_square),
        )
    return kp.hand_feature_index(
        view=d.view, king_square=mks, rel_owner=d.rel_owner,
        hand_kind=d.hand_kind, slot=d.slot,
    )


def build_mirror_perm(cache_path: str | None = None) -> np.ndarray:
    """Build (or load) the length-TOTAL_FEATURES permutation `perm` such that
    flipping a position = mapping each active index `i -> perm[i]`."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    perm = np.empty(kp.TOTAL_FEATURES, dtype=np.int64)
    for i in range(kp.TOTAL_FEATURES):
        perm[i] = mirror_feature_index(i)
    if cache_path:
        np.save(cache_path, perm)
    return perm


def build_mirror_perm_shared(cache_path: str | None = None) -> np.ndarray:
    """Mirror permutation for the true canonical shared band (Task 047), length
    CANONICAL_SHARED_TOTAL. Body indices live in the single band [0, VIEW_DIM) with
    view=0; mirroring the (rotated-frame) king/piece squares is still correct
    because the horizontal mirror commutes with the canonical 180-rotation. The two
    side rows at the end are identity."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    perm = np.empty(kp.CANONICAL_SHARED_TOTAL, dtype=np.int64)
    for i in range(kp.VIEW_DIM):
        d = kp.decode_feature_index(i)  # view=0
        mks = mirror_square(d.king_square)
        if d.kind == "board":
            perm[i] = kp.board_feature_index(
                view=0, king_square=mks, rel_owner=d.rel_owner,
                piece_type=d.piece_type, piece_square=mirror_square(d.piece_square),
            )
        else:
            perm[i] = kp.hand_feature_index(
                view=0, king_square=mks, rel_owner=d.rel_owner,
                hand_kind=d.hand_kind, slot=d.slot,
            )
    for i in range(kp.VIEW_DIM, kp.CANONICAL_SHARED_TOTAL):
        perm[i] = i  # side feature: unchanged by mirroring
    if cache_path:
        np.save(cache_path, perm)
    return perm


def build_mirror_perm_shared_ka(cache_path: str | None = None) -> np.ndarray:
    """Mirror permutation for the HalfKA shared band (Task 052), length
    CANONICAL_SHARED_KA_TOTAL. Identical construction to the HalfKP-shared perm but
    over the larger KA band and its 14-kind decode/encode; the king features (id 13)
    mirror by their square just like any other board piece."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    perm = np.empty(kp.CANONICAL_SHARED_KA_TOTAL, dtype=np.int64)
    for i in range(kp.VIEW_DIM_KA):
        d = kp.decode_feature_index_ka(i)  # view=0
        mks = mirror_square(d.king_square)
        if d.kind == "board":
            perm[i] = kp.board_feature_index_ka(
                king_square=mks, rel_owner=d.rel_owner,
                piece_type=d.piece_type, piece_square=mirror_square(d.piece_square),
            )
        else:
            perm[i] = kp.hand_feature_index_ka(
                king_square=mks, rel_owner=d.rel_owner,
                hand_kind=d.hand_kind, slot=d.slot,
            )
    for i in range(kp.VIEW_DIM_KA, kp.CANONICAL_SHARED_KA_TOTAL):
        perm[i] = i  # side feature: unchanged by mirroring
    if cache_path:
        np.save(cache_path, perm)
    return perm


def _mirror_kpp_zone_state(state: int) -> int:
    """Horizontal mirror of a king-zone state: flip the zone file (the 5x5 zone's
    file index 0..4 -> 4..0); piece type and owner are unchanged."""
    owner = state % 2
    t = state // 2
    npt = len(kp.BOARD_PIECE_TYPES)
    pid = t % npt
    zone_rel = t // npt
    f, r = zone_rel // 5, zone_rel % 5
    return kp._kpp_zone_state((4 - f) * 5 + r, pid, owner)


def build_kpp_zone_mirror_perm(cache_path: str | None = None) -> np.ndarray:
    """Mirror permutation for the king-zone KPP pair table (Task 053), length
    KPP_ZONE_TOTAL. A horizontal board mirror flips each in-zone piece's file, hence
    its zone state; a pair {a,b} maps to {mirror(a), mirror(b)}."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    sm = [_mirror_kpp_zone_state(s) for s in range(kp.KPP_ZONE_STATES)]
    perm = np.empty(kp.KPP_ZONE_TOTAL, dtype=np.int64)
    for a in range(kp.KPP_ZONE_STATES):
        for b in range(a + 1, kp.KPP_ZONE_STATES):
            perm[kp._kpp_pair_index(a, b)] = kp._kpp_pair_index(sm[a], sm[b])
    if cache_path:
        np.save(cache_path, perm)
    return perm


def _flip_board_field(board_field: str) -> str:
    """Reverse the files within each SFEN rank (the board-level horizontal flip)."""
    out_ranks = []
    for rank in board_field.split("/"):
        cells: list[str] = []
        i = 0
        while i < len(rank):
            c = rank[i]
            if c.isdigit():
                cells.extend(["1"] * int(c))  # one placeholder per empty square
                i += 1
            elif c == "+":
                cells.append(rank[i : i + 2])  # promoted piece token, e.g. "+P"
                i += 2
            else:
                cells.append(c)
                i += 1
        if len(cells) != 9:
            raise ValueError(f"rank does not have 9 files: {rank!r} -> {cells}")
        cells.reverse()
        row, empty = [], 0
        for cell in cells:
            if cell == "1":
                empty += 1
            else:
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(cell)
        if empty:
            row.append(str(empty))
        out_ranks.append("".join(row))
    return "/".join(out_ranks)


def flip_sfen(sfen: str) -> str:
    """Horizontally mirror a full SFEN (board files reversed; turn/hands/move kept)."""
    parts = sfen.split()
    parts[0] = _flip_board_field(parts[0])
    return " ".join(parts)
