"""KP feature conversion (ground-truth implementation for training).

Canonical spec — numbering, dimensions, the "77 features" invariant, worked check
values, and the divergence from canonical HalfKP — lives in ``docs/kp_spec.md``.
This module is the authoritative numbering; ``src/eval/kp.rs`` (Rust inference) is
verified to match it exactly. Update ``docs/kp_spec.md`` together with any change
to the numbering here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cshogi
import numpy as np


SELF = 0
OPPONENT = 1
VIEW_SELF_KING = 0
VIEW_OPPONENT_KING = 1
VIEW_COUNT = 2
SQUARE_NB = 81

BOARD_PIECE_TYPES = (
    cshogi.PAWN,
    cshogi.LANCE,
    cshogi.KNIGHT,
    cshogi.SILVER,
    cshogi.BISHOP,
    cshogi.ROOK,
    cshogi.GOLD,
    cshogi.PROM_PAWN,
    cshogi.PROM_LANCE,
    cshogi.PROM_KNIGHT,
    cshogi.PROM_SILVER,
    cshogi.PROM_BISHOP,
    cshogi.PROM_ROOK,
)
BOARD_PIECE_TO_ID = {piece_type: index for index, piece_type in enumerate(BOARD_PIECE_TYPES)}

HAND_PIECE_TYPES = (
    cshogi.HPAWN,
    cshogi.HLANCE,
    cshogi.HKNIGHT,
    cshogi.HSILVER,
    cshogi.HGOLD,
    cshogi.HBISHOP,
    cshogi.HROOK,
)
HAND_MAX_COUNTS = tuple(cshogi.MAX_PIECES_IN_HAND)
HAND_KIND_OFFSETS: dict[int, int] = {}
_hand_offset = 0
for _hand_kind, _max_count in zip(HAND_PIECE_TYPES, HAND_MAX_COUNTS):
    HAND_KIND_OFFSETS[_hand_kind] = _hand_offset
    _hand_offset += _max_count

BOARD_FEATURES_PER_KING = len(BOARD_PIECE_TYPES) * 2 * SQUARE_NB
HAND_SLOTS_PER_OWNER = sum(HAND_MAX_COUNTS)
HAND_FEATURES_PER_KING = 2 * HAND_SLOTS_PER_OWNER
FEATURES_PER_KING = BOARD_FEATURES_PER_KING + HAND_FEATURES_PER_KING
VIEW_DIM = SQUARE_NB * FEATURES_PER_KING
TOTAL_FEATURES = VIEW_COUNT * VIEW_DIM + 2
SIDE_FEATURE_BASE = VIEW_COUNT * VIEW_DIM


@dataclass(frozen=True)
class KpExample:
    indices: tuple[int, ...]
    eval: int
    sfen_core: str


def read_hcpe(path: str) -> list[KpExample]:
    entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
    examples = []
    for entry in entries:
        board = cshogi.Board()
        board.set_hcp(entry["hcp"])
        examples.append(
            KpExample(
                indices=board_to_kp_indices(board),
                eval=int(entry["eval"]),
                sfen_core=sfen_core(board.sfen()),
            )
        )
    return examples


def rotate_square(square: int) -> int:
    """180-degree rotation within the side-to-move frame (X -> 80 - X)."""
    return SQUARE_NB - 1 - square


def board_to_kp_indices(board: cshogi.Board, canonical: bool = False) -> tuple[int, ...]:
    """Active KP feature indices for ``board``.

    ``canonical=False`` (default) = naive HalfKP (both halves in the side-to-move
    frame; view 1 only swaps the anchor king). Byte-identical to the original.

    ``canonical=True`` = canonical HalfKP (docs/kp_spec.md §7): the opponent half
    (view 1) is taken from the opponent's own perspective = the side-to-move frame
    rotated 180 degrees + relative-owner flipped. Numbering range and the 77-feature
    invariant are unchanged; only which view-1 indices light up changes.
    """
    turn = board.turn
    self_king = normalize_square(board.king_square(turn), turn)
    opponent_king = normalize_square(board.king_square(opposite_color(turn)), turn)
    active = set()

    # (view, anchor king square, flip) — flip rotates piece squares 180 and flips
    # the relative owner (the opponent's own perspective). Naive never flips.
    if canonical:
        views = (
            (VIEW_SELF_KING, self_king, False),
            (VIEW_OPPONENT_KING, rotate_square(opponent_king), True),
        )
    else:
        views = (
            (VIEW_SELF_KING, self_king, False),
            (VIEW_OPPONENT_KING, opponent_king, False),
        )

    for view, king_square, flip in views:
        for square, piece in enumerate(board.pieces):
            if piece == cshogi.NONE:
                continue
            piece_type = piece & 0x0F
            if piece_type == cshogi.KING:
                continue
            color = piece_color(piece)
            rel_owner = SELF if color == turn else OPPONENT
            piece_square = normalize_square(square, turn)
            if flip:
                rel_owner = 1 - rel_owner
                piece_square = rotate_square(piece_square)
            active.add(
                board_feature_index(
                    view=view,
                    king_square=king_square,
                    rel_owner=rel_owner,
                    piece_type=piece_type,
                    piece_square=piece_square,
                )
            )

        for color in (cshogi.BLACK, cshogi.WHITE):
            rel_owner = SELF if color == turn else OPPONENT
            if flip:
                rel_owner = 1 - rel_owner
            hand = board.pieces_in_hand[color]
            for hand_kind in HAND_PIECE_TYPES:
                count = hand[hand_kind]
                for slot in range(count):
                    active.add(
                        hand_feature_index(
                            view=view,
                            king_square=king_square,
                            rel_owner=rel_owner,
                            hand_kind=hand_kind,
                            slot=slot,
                        )
                    )

    active.add(SIDE_FEATURE_BASE + turn)
    return tuple(sorted(active))


def split_halves(indices: Iterable[int]) -> tuple[list[int], list[int]]:
    """Split active indices into (own_half, opp_half) for the 2-accumulator NNUE.

    own = view-0 body features + the side feature; opp = view-1 body features.
    The split is by index range, so it is identical for naive and canonical
    numbering (only the index *values* of the opp half differ between them).
    """
    own: list[int] = []
    opp: list[int] = []
    for index in indices:
        if index >= SIDE_FEATURE_BASE:
            own.append(index)  # side feature -> own (own perspective = side to move)
        elif index < VIEW_DIM:
            own.append(index)
        else:
            opp.append(index)
    return own, opp


# --- True canonical HalfKP (Task 047): a SINGLE shared feature table -----------
# The original NNUE paper (Nasu 2018, eq. 15-17) shares W1 between both
# perspectives: x^(active) and x^(opponent) index the SAME table. Here both halves
# index a single band [0, VIEW_DIM) (anchor = own king for the active half, the
# 180-rotated + owner-flipped opponent king for the opponent half), so the
# embedding row for a (king, piece) relation is shared no matter which side's king
# it belongs to. This is the weight sharing Task 046's two-band version lacked.
CANONICAL_SHARED_SIDE_BASE = VIEW_DIM            # 176742
CANONICAL_SHARED_TOTAL = VIEW_DIM + 2            # 176744 (half of naive's 353486 + side)


def canonical_shared_halves(board: cshogi.Board) -> tuple[list[int], list[int]]:
    """Active feature indices for true canonical HalfKP, returned as (own, opp).

    Both lists index the SAME shared band [0, VIEW_DIM); the side feature
    (CANONICAL_SHARED_SIDE_BASE + turn) is appended to `own`. own = 38 body + 1
    side = 39; opp = 38 body. own/opp may share index *values* (that is the point
    of sharing), so they are returned separately rather than merged.
    """
    turn = board.turn
    self_king = normalize_square(board.king_square(turn), turn)
    opp_king = rotate_square(normalize_square(board.king_square(opposite_color(turn)), turn))
    own: list[int] = []
    opp: list[int] = []

    for square, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
            continue
        piece_type = piece & 0x0F
        if piece_type == cshogi.KING:
            continue
        rel = SELF if piece_color(piece) == turn else OPPONENT
        psq = normalize_square(square, turn)
        # view=0 so both halves land in the single shared band [0, VIEW_DIM).
        own.append(board_feature_index(view=0, king_square=self_king, rel_owner=rel,
                                       piece_type=piece_type, piece_square=psq))
        opp.append(board_feature_index(view=0, king_square=opp_king, rel_owner=1 - rel,
                                       piece_type=piece_type, piece_square=rotate_square(psq)))

    for color in (cshogi.BLACK, cshogi.WHITE):
        rel = SELF if color == turn else OPPONENT
        hand = board.pieces_in_hand[color]
        for hand_kind in HAND_PIECE_TYPES:
            for slot in range(hand[hand_kind]):
                own.append(hand_feature_index(view=0, king_square=self_king, rel_owner=rel,
                                              hand_kind=hand_kind, slot=slot))
                opp.append(hand_feature_index(view=0, king_square=opp_king, rel_owner=1 - rel,
                                              hand_kind=hand_kind, slot=slot))

    own.append(CANONICAL_SHARED_SIDE_BASE + turn)
    return sorted(own), sorted(opp)


def vm_mirror_map() -> np.ndarray:
    """HalfKP_vm as weight tying (Task 132): ``m[b]`` for every row b of the shared band.

    YaneuraOu's HalfKP_vm mirrors a perspective left-right when its king stands on files
    6-9, so a king and its mirror image share one weight row. With HalfKP indexing that is
    the tie W[k, p] = W[mirror(k), mirror(p)]: a row whose king square has file index >= 5
    (cshogi square = file * 9 + rank, file 0 = the 1-file) is sent to the row of the
    mirrored king and the mirrored piece square; hand features keep their offset. Rows with
    the king on files 0-4 map to themselves, so ``m[m] == m``."""
    b = np.arange(VIEW_DIM, dtype=np.int64)
    king, off = b // FEATURES_PER_KING, b % FEATURES_PER_KING
    file, rank = king // 9, king % 9
    right = file >= 5
    king_m = np.where(right, (8 - file) * 9 + rank, king)
    board = off < BOARD_FEATURES_PER_KING
    sq = off % SQUARE_NB
    off_m = np.where(right & board, off - sq + (8 - sq // 9) * 9 + sq % 9, off)
    return king_m * FEATURES_PER_KING + off_m


def canonical_shared_row(board: cshogi.Board) -> tuple[int, ...]:
    """Fixed-layout cache row for canonical-shared: own(39) ++ opp(38) = 77 columns.

    own = columns [0:39] (38 body sorted, then the side feature, which is the
    largest), opp = columns [39:77]. Columns (not value range) carry the own/opp
    split, since the two halves can share index values."""
    own, opp = canonical_shared_halves(board)
    return tuple(own) + tuple(opp)


def canonical_shared_row_to_sfen(row: Iterable[int]) -> str:
    """Reconstruct the SFEN from a canonical-shared cache row (own half alone is
    sufficient; the opponent half is the same position rotated)."""
    row = tuple(row)
    own = row[:39]
    side = [i for i in own if i >= CANONICAL_SHARED_SIDE_BASE]
    if len(side) != 1:
        raise ValueError(f"expected exactly one side feature in own half, got {len(side)}")
    turn = side[0] - CANONICAL_SHARED_SIDE_BASE
    if turn not in (cshogi.BLACK, cshogi.WHITE):
        raise ValueError(f"invalid side feature: {side[0]}")

    self_king: int | None = None
    pieces: set[tuple[int, int, int]] = set()
    hands_slots: dict[tuple[int, int], set[int]] = {}
    for index in own:
        if index >= CANONICAL_SHARED_SIDE_BASE:
            continue
        d = decode_feature_index(index)  # view=0; king/owner/piece in the stm frame
        if self_king is None:
            self_king = d.king_square
        elif self_king != d.king_square:
            raise ValueError("inconsistent own-king square")
        if d.kind == "board":
            pieces.add((d.rel_owner, d.piece_type, d.piece_square))
        else:
            hands_slots.setdefault((d.rel_owner, d.hand_kind), set()).add(d.slot)
    if self_king is None:
        raise ValueError("no own-king anchor found")

    board = [cshogi.NONE] * SQUARE_NB
    board[denormalize_square(self_king, turn)] = colored_piece(cshogi.KING, turn)
    # The opponent king sits at the rotated own-king anchor's inverse; recover it
    # from the opponent half's shared anchor (all opp features share that king).
    opp = row[39:]
    opp_king_anchor = decode_feature_index(opp[0]).king_square if opp else rotate_square(self_king)
    opp_king_norm = rotate_square(opp_king_anchor)
    board[denormalize_square(opp_king_norm, turn)] = colored_piece(cshogi.KING, opposite_color(turn))

    for rel_owner, piece_type, nsq in pieces:
        color = color_from_rel_owner(rel_owner, turn)
        sq = denormalize_square(nsq, turn)
        if board[sq] != cshogi.NONE:
            raise ValueError(f"square collision at {cshogi.SQUARE_NAMES[sq]}")
        board[sq] = colored_piece(piece_type, color)

    hands = {cshogi.BLACK: [0] * len(HAND_PIECE_TYPES), cshogi.WHITE: [0] * len(HAND_PIECE_TYPES)}
    for (rel_owner, hand_kind), slots in hands_slots.items():
        if slots != set(range(max(slots) + 1)):
            raise ValueError(f"non-contiguous hand slots: {(rel_owner, hand_kind)} {slots}")
        hands[color_from_rel_owner(rel_owner, turn)][hand_kind] = len(slots)

    return make_sfen(board, hands, turn)


# --- HalfKA (Task 052): kings included as board piece features -----------------
# HalfKP omits the kings (they are only the anchors), so the network never sees
# the (own_king, enemy_king) relation = king safety. HalfKA ("A" = All) adds the
# kings as a board piece type (id 13), giving each half the enemy king's square as
# a feature. Implemented as a parallel canonical SHARED-band numbering (a single
# table indexed by both halves), exported with magic BNN4 / Rust Mode::SharedKA.
# The HalfKP numbering above is untouched (naive/two-band/shared all keep working).
BOARD_PIECE_TYPES_KA = BOARD_PIECE_TYPES + (cshogi.KING,)
BOARD_PIECE_TO_ID_KA = {pt: i for i, pt in enumerate(BOARD_PIECE_TYPES_KA)}  # KING -> 13
BOARD_PIECE_KINDS_KA = len(BOARD_PIECE_TYPES_KA)                       # 14
BOARD_FEATURES_PER_KING_KA = BOARD_PIECE_KINDS_KA * 2 * SQUARE_NB      # 2268
FEATURES_PER_KING_KA = BOARD_FEATURES_PER_KING_KA + HAND_FEATURES_PER_KING  # 2344
VIEW_DIM_KA = SQUARE_NB * FEATURES_PER_KING_KA                         # 189864
CANONICAL_SHARED_KA_SIDE_BASE = VIEW_DIM_KA                            # 189864
CANONICAL_SHARED_KA_TOTAL = VIEW_DIM_KA + 2                            # 189866


def board_feature_index_ka(*, king_square: int, rel_owner: int, piece_type: int,
                           piece_square: int) -> int:
    """Shared-band (view=0) HalfKA board feature index. `piece_type` may be KING."""
    piece_id = BOARD_PIECE_TO_ID_KA[piece_type]
    offset = (rel_owner * BOARD_PIECE_KINDS_KA + piece_id) * SQUARE_NB + piece_square
    return king_square * FEATURES_PER_KING_KA + offset


def hand_feature_index_ka(*, king_square: int, rel_owner: int, hand_kind: int,
                          slot: int) -> int:
    if slot < 0 or slot >= HAND_MAX_COUNTS[hand_kind]:
        raise ValueError(f"hand slot out of range: kind={hand_kind} slot={slot}")
    offset = (
        BOARD_FEATURES_PER_KING_KA
        + rel_owner * HAND_SLOTS_PER_OWNER
        + HAND_KIND_OFFSETS[hand_kind]
        + slot
    )
    return king_square * FEATURES_PER_KING_KA + offset


def decode_feature_index_ka(index: int) -> DecodedFeature:
    """Decode a HalfKA shared-band index (< CANONICAL_SHARED_KA_SIDE_BASE, view=0).
    Used to build the HalfKA mirror permutation."""
    if not 0 <= index < CANONICAL_SHARED_KA_SIDE_BASE:
        raise ValueError(f"not a HalfKA body feature index: {index}")
    king_square = index // FEATURES_PER_KING_KA
    feature = index % FEATURES_PER_KING_KA
    if feature < BOARD_FEATURES_PER_KING_KA:
        piece_bucket = feature // SQUARE_NB
        piece_square = feature % SQUARE_NB
        rel_owner = piece_bucket // BOARD_PIECE_KINDS_KA
        piece_type = BOARD_PIECE_TYPES_KA[piece_bucket % BOARD_PIECE_KINDS_KA]
        return DecodedFeature(view=0, king_square=king_square, kind="board",
                              rel_owner=rel_owner, piece_type=piece_type,
                              piece_square=piece_square)
    hand_feature = feature - BOARD_FEATURES_PER_KING_KA
    rel_owner = hand_feature // HAND_SLOTS_PER_OWNER
    hand_offset = hand_feature % HAND_SLOTS_PER_OWNER
    for hand_kind in HAND_PIECE_TYPES:
        start = HAND_KIND_OFFSETS[hand_kind]
        end = start + HAND_MAX_COUNTS[hand_kind]
        if start <= hand_offset < end:
            return DecodedFeature(view=0, king_square=king_square, kind="hand",
                                  rel_owner=rel_owner, hand_kind=hand_kind,
                                  slot=hand_offset - start)
    raise ValueError(f"invalid hand feature offset: {hand_offset}")


def canonical_shared_ka_halves(board: cshogi.Board) -> tuple[list[int], list[int]]:
    """HalfKA active features as (own, opp), both indexing the shared band
    [0, VIEW_DIM_KA). Like `canonical_shared_halves` but the two kings are INCLUDED
    as board pieces (id 13). own = 40 body + 1 side = 41; opp = 40 body = 40."""
    turn = board.turn
    self_king = normalize_square(board.king_square(turn), turn)
    opp_king = rotate_square(normalize_square(board.king_square(opposite_color(turn)), turn))
    own: list[int] = []
    opp: list[int] = []

    for square, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
            continue
        piece_type = piece & 0x0F  # KING kept (NOT skipped) -> id 13
        rel = SELF if piece_color(piece) == turn else OPPONENT
        psq = normalize_square(square, turn)
        own.append(board_feature_index_ka(king_square=self_king, rel_owner=rel,
                                          piece_type=piece_type, piece_square=psq))
        opp.append(board_feature_index_ka(king_square=opp_king, rel_owner=1 - rel,
                                          piece_type=piece_type, piece_square=rotate_square(psq)))

    for color in (cshogi.BLACK, cshogi.WHITE):
        rel = SELF if color == turn else OPPONENT
        hand = board.pieces_in_hand[color]
        for hand_kind in HAND_PIECE_TYPES:
            for slot in range(hand[hand_kind]):
                own.append(hand_feature_index_ka(king_square=self_king, rel_owner=rel,
                                                 hand_kind=hand_kind, slot=slot))
                opp.append(hand_feature_index_ka(king_square=opp_king, rel_owner=1 - rel,
                                                 hand_kind=hand_kind, slot=slot))

    own.append(CANONICAL_SHARED_KA_SIDE_BASE + turn)
    return sorted(own), sorted(opp)


def canonical_shared_ka_row(board: cshogi.Board) -> tuple[int, ...]:
    """Fixed-layout HalfKA cache row: own(41) ++ opp(40) = 81 columns."""
    own, opp = canonical_shared_ka_halves(board)
    return tuple(own) + tuple(opp)


def canonical_shared_kav2_halves(board: cshogi.Board) -> tuple[list[int], list[int]]:
    """HalfKAv2 (Task 053): like HalfKA but each half DROPS its own anchor king as a
    piece feature (the redundant `(anchor, anchor, K)` constant), keeping only the
    CROSS king (= the king-king relation). own half drops the side-to-move king,
    opp half drops the opponent king. own = 38 body + enemy-king + side = 40;
    opp = 38 body + own-king = 39. active = 79. Same shared KA table (the dropped
    anchor-king rows are simply never active)."""
    turn = board.turn
    self_king = normalize_square(board.king_square(turn), turn)
    opp_king = rotate_square(normalize_square(board.king_square(opposite_color(turn)), turn))
    own: list[int] = []
    opp: list[int] = []

    for square, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
            continue
        piece_type = piece & 0x0F
        rel = SELF if piece_color(piece) == turn else OPPONENT
        psq = normalize_square(square, turn)
        is_king = piece_type == cshogi.KING
        if not (is_king and rel == SELF):       # own half drops the own (anchor) king
            own.append(board_feature_index_ka(king_square=self_king, rel_owner=rel,
                                              piece_type=piece_type, piece_square=psq))
        if not (is_king and rel == OPPONENT):    # opp half drops the opp (anchor) king
            opp.append(board_feature_index_ka(king_square=opp_king, rel_owner=1 - rel,
                                              piece_type=piece_type, piece_square=rotate_square(psq)))

    for color in (cshogi.BLACK, cshogi.WHITE):
        rel = SELF if color == turn else OPPONENT
        hand = board.pieces_in_hand[color]
        for hand_kind in HAND_PIECE_TYPES:
            for slot in range(hand[hand_kind]):
                own.append(hand_feature_index_ka(king_square=self_king, rel_owner=rel,
                                                 hand_kind=hand_kind, slot=slot))
                opp.append(hand_feature_index_ka(king_square=opp_king, rel_owner=1 - rel,
                                                 hand_kind=hand_kind, slot=slot))

    own.append(CANONICAL_SHARED_KA_SIDE_BASE + turn)
    return sorted(own), sorted(opp)


def canonical_shared_kav2_row(board: cshogi.Board) -> tuple[int, ...]:
    """Fixed-layout HalfKAv2 cache row: own(40) ++ opp(39) = 79 columns."""
    own, opp = canonical_shared_kav2_halves(board)
    return tuple(own) + tuple(opp)


# --- king-zone KPP (Task 053): piece PAIRS within each king's 5x5 zone ---------
# A feasible reduced KPP (docs/kp_spec.md §9.2 / trainer/kpp_zone_probe.py). For
# each perspective king, the non-king board pieces within Chebyshev<=2 (a 5x5 zone)
# are encoded by their zone-relative state; unordered PAIRS of those states are the
# features. zone-relative (no king-square dimension) keeps the table tiny
# (C(650,2)=210925) and the active count <= KP, so no incremental update is needed.
# Numbered in the SIDE-TO-MOVE-normalized frame, consistent with canonical_shared:
# own = pairs near the side-to-move king; opp = pairs near the opponent king with the
# canonical 180-rotation (zone_rel -> 24-zone_rel) + owner flip (shared table).
KPP_ZONE_REL_NB = 25                                                 # 5x5 zone
KPP_ZONE_STATES = KPP_ZONE_REL_NB * len(BOARD_PIECE_TYPES) * 2       # 25*13*2 = 650
KPP_ZONE_TOTAL = KPP_ZONE_STATES * (KPP_ZONE_STATES - 1) // 2        # C(650,2) = 210925


def _kpp_zone_rel(psq_norm: int, king_norm: int) -> int | None:
    """Zone-relative index 0..24 of a normalized square within the king's 5x5 zone,
    or None if outside (Chebyshev > 2). cshogi square = file*9 + rank."""
    df = psq_norm // 9 - king_norm // 9
    dr = psq_norm % 9 - king_norm % 9
    if df < -2 or df > 2 or dr < -2 or dr > 2:
        return None
    return (df + 2) * 5 + (dr + 2)


def _kpp_zone_state(zone_rel: int, ptype_id: int, rel_owner: int) -> int:
    return (zone_rel * len(BOARD_PIECE_TYPES) + ptype_id) * 2 + rel_owner


def _kpp_pair_index(a: int, b: int) -> int:
    """Unordered combinatorial rank of the pair {a, b} (a != b) in [0, C(650,2))."""
    if a > b:
        a, b = b, a
    return a * KPP_ZONE_STATES - (a * (a + 1)) // 2 + (b - a - 1)


def _kpp_pairs(states: list[int]) -> list[int]:
    return sorted({_kpp_pair_index(states[i], states[j])
                   for i in range(len(states)) for j in range(i + 1, len(states))})


def kpp_zone_halves(board: cshogi.Board) -> tuple[list[int], list[int]]:
    """king-zone KPP pair indices (own, opp), side-to-move-normalized (see above)."""
    turn = board.turn
    self_king = normalize_square(board.king_square(turn), turn)
    opp_king = normalize_square(board.king_square(opposite_color(turn)), turn)
    own_states: list[int] = []
    opp_states: list[int] = []
    for square, piece in enumerate(board.pieces):
        if piece == cshogi.NONE:
            continue
        ptype = piece & 0x0F
        if ptype == cshogi.KING:
            continue
        psq = normalize_square(square, turn)
        rel = SELF if piece_color(piece) == turn else OPPONENT
        pid = BOARD_PIECE_TO_ID[ptype]
        zr_own = _kpp_zone_rel(psq, self_king)
        if zr_own is not None:
            own_states.append(_kpp_zone_state(zr_own, pid, rel))
        zr_opp = _kpp_zone_rel(psq, opp_king)
        if zr_opp is not None:                       # canonical 180-rotation + owner flip
            opp_states.append(_kpp_zone_state(24 - zr_opp, pid, 1 - rel))
    return _kpp_pairs(own_states), _kpp_pairs(opp_states)


def kp_indices_to_sfen(indices: Iterable[int], canonical: bool = False) -> str:
    indices = tuple(sorted(indices))
    turn_features = [index for index in indices if index >= SIDE_FEATURE_BASE]
    if len(turn_features) != 1:
        raise ValueError(f"expected exactly one side feature, got {len(turn_features)}")

    turn = turn_features[0] - SIDE_FEATURE_BASE
    if turn not in (cshogi.BLACK, cshogi.WHITE):
        raise ValueError(f"invalid side feature: {turn_features[0]}")

    view_king_squares: dict[int, int] = {}
    view_pieces: dict[int, set[tuple[int, int, int]]] = {
        VIEW_SELF_KING: set(),
        VIEW_OPPONENT_KING: set(),
    }
    view_hands: dict[int, set[tuple[int, int, int]]] = {
        VIEW_SELF_KING: set(),
        VIEW_OPPONENT_KING: set(),
    }

    for index in indices:
        if index >= SIDE_FEATURE_BASE:
            continue
        decoded = decode_feature_index(index)
        if canonical and decoded.view == VIEW_OPPONENT_KING:
            # Inverse of the canonical opponent-half transform: rotate 180 + flip
            # owner, back into the side-to-move frame, so the consistency check and
            # reconstruction below see both halves in the same frame.
            decoded = _untransform_canonical_opp(decoded)
        existing_king = view_king_squares.get(decoded.view)
        if existing_king is None:
            view_king_squares[decoded.view] = decoded.king_square
        elif existing_king != decoded.king_square:
            raise ValueError(f"view {decoded.view} has inconsistent king squares")

        if decoded.kind == "board":
            view_pieces[decoded.view].add(
                (decoded.rel_owner, decoded.piece_type, decoded.piece_square)
            )
        elif decoded.kind == "hand":
            view_hands[decoded.view].add((decoded.rel_owner, decoded.hand_kind, decoded.slot))
        else:
            raise ValueError(f"invalid feature kind: {decoded.kind}")

    if set(view_king_squares) != {VIEW_SELF_KING, VIEW_OPPONENT_KING}:
        raise ValueError(f"missing view king square: {view_king_squares}")
    if view_pieces[VIEW_SELF_KING] != view_pieces[VIEW_OPPONENT_KING]:
        raise ValueError("board piece features differ between the two views")
    if view_hands[VIEW_SELF_KING] != view_hands[VIEW_OPPONENT_KING]:
        raise ValueError("hand features differ between the two views")

    board = [cshogi.NONE] * SQUARE_NB
    self_king_square = denormalize_square(view_king_squares[VIEW_SELF_KING], turn)
    opponent_king_square = denormalize_square(view_king_squares[VIEW_OPPONENT_KING], turn)
    board[self_king_square] = colored_piece(cshogi.KING, turn)
    board[opponent_king_square] = colored_piece(cshogi.KING, opposite_color(turn))

    for rel_owner, piece_type, normalized_square in view_pieces[VIEW_SELF_KING]:
        color = color_from_rel_owner(rel_owner, turn)
        square = denormalize_square(normalized_square, turn)
        if board[square] != cshogi.NONE:
            raise ValueError(f"square collision at {cshogi.SQUARE_NAMES[square]}")
        board[square] = colored_piece(piece_type, color)

    hands = {
        cshogi.BLACK: [0] * len(HAND_PIECE_TYPES),
        cshogi.WHITE: [0] * len(HAND_PIECE_TYPES),
    }
    slots_by_owner_kind: dict[tuple[int, int], set[int]] = {}
    for rel_owner, hand_kind, slot in view_hands[VIEW_SELF_KING]:
        slots_by_owner_kind.setdefault((rel_owner, hand_kind), set()).add(slot)

    for (rel_owner, hand_kind), slots in slots_by_owner_kind.items():
        expected = set(range(max(slots) + 1))
        if slots != expected:
            raise ValueError(f"non-contiguous hand slots: {(rel_owner, hand_kind)} {slots}")
        color = color_from_rel_owner(rel_owner, turn)
        hands[color][hand_kind] = len(slots)

    return make_sfen(board, hands, turn)


def expected_active_feature_count(board: cshogi.Board) -> int:
    non_king_pieces = sum(
        1
        for piece in board.pieces
        if piece != cshogi.NONE and (piece & 0x0F) != cshogi.KING
    )
    hand_pieces = sum(sum(hand) for hand in board.pieces_in_hand)
    return 2 * (non_king_pieces + hand_pieces) + 1


def board_feature_index(
    *,
    view: int,
    king_square: int,
    rel_owner: int,
    piece_type: int,
    piece_square: int,
) -> int:
    piece_type_id = BOARD_PIECE_TO_ID[piece_type]
    offset = (rel_owner * len(BOARD_PIECE_TYPES) + piece_type_id) * SQUARE_NB + piece_square
    return view * VIEW_DIM + king_square * FEATURES_PER_KING + offset


def hand_feature_index(
    *,
    view: int,
    king_square: int,
    rel_owner: int,
    hand_kind: int,
    slot: int,
) -> int:
    if slot < 0 or slot >= HAND_MAX_COUNTS[hand_kind]:
        raise ValueError(f"hand slot out of range: kind={hand_kind} slot={slot}")
    offset = (
        BOARD_FEATURES_PER_KING
        + rel_owner * HAND_SLOTS_PER_OWNER
        + HAND_KIND_OFFSETS[hand_kind]
        + slot
    )
    return view * VIEW_DIM + king_square * FEATURES_PER_KING + offset


@dataclass(frozen=True)
class DecodedFeature:
    view: int
    king_square: int
    kind: str
    rel_owner: int
    piece_type: int | None = None
    piece_square: int | None = None
    hand_kind: int | None = None
    slot: int | None = None


def decode_feature_index(index: int) -> DecodedFeature:
    if not 0 <= index < SIDE_FEATURE_BASE:
        raise ValueError(f"not a KP body feature index: {index}")

    view = index // VIEW_DIM
    within_view = index % VIEW_DIM
    king_square = within_view // FEATURES_PER_KING
    feature = within_view % FEATURES_PER_KING

    if feature < BOARD_FEATURES_PER_KING:
        piece_bucket = feature // SQUARE_NB
        piece_square = feature % SQUARE_NB
        rel_owner = piece_bucket // len(BOARD_PIECE_TYPES)
        piece_type = BOARD_PIECE_TYPES[piece_bucket % len(BOARD_PIECE_TYPES)]
        return DecodedFeature(
            view=view,
            king_square=king_square,
            kind="board",
            rel_owner=rel_owner,
            piece_type=piece_type,
            piece_square=piece_square,
        )

    hand_feature = feature - BOARD_FEATURES_PER_KING
    rel_owner = hand_feature // HAND_SLOTS_PER_OWNER
    hand_offset = hand_feature % HAND_SLOTS_PER_OWNER
    for hand_kind in HAND_PIECE_TYPES:
        start = HAND_KIND_OFFSETS[hand_kind]
        end = start + HAND_MAX_COUNTS[hand_kind]
        if start <= hand_offset < end:
            return DecodedFeature(
                view=view,
                king_square=king_square,
                kind="hand",
                rel_owner=rel_owner,
                hand_kind=hand_kind,
                slot=hand_offset - start,
            )
    raise ValueError(f"invalid hand feature offset: {hand_offset}")


def _untransform_canonical_opp(decoded: "DecodedFeature") -> "DecodedFeature":
    """Map a canonical opponent-half decoded feature back to the side-to-move frame."""
    king_square = rotate_square(decoded.king_square)
    rel_owner = 1 - decoded.rel_owner
    if decoded.kind == "board":
        return DecodedFeature(
            view=decoded.view,
            king_square=king_square,
            kind="board",
            rel_owner=rel_owner,
            piece_type=decoded.piece_type,
            piece_square=rotate_square(decoded.piece_square),
        )
    return DecodedFeature(
        view=decoded.view,
        king_square=king_square,
        kind="hand",
        rel_owner=rel_owner,
        hand_kind=decoded.hand_kind,
        slot=decoded.slot,
    )


def normalize_square(square: int, turn: int) -> int:
    return square if turn == cshogi.BLACK else SQUARE_NB - 1 - square


def denormalize_square(square: int, turn: int) -> int:
    return normalize_square(square, turn)


def opposite_color(color: int) -> int:
    return cshogi.WHITE if color == cshogi.BLACK else cshogi.BLACK


def piece_color(piece: int) -> int:
    if piece == cshogi.NONE:
        raise ValueError("empty square has no color")
    return cshogi.WHITE if piece >= cshogi.WPAWN else cshogi.BLACK


def color_from_rel_owner(rel_owner: int, turn: int) -> int:
    if rel_owner == SELF:
        return turn
    if rel_owner == OPPONENT:
        return opposite_color(turn)
    raise ValueError(f"invalid rel_owner: {rel_owner}")


def colored_piece(piece_type: int, color: int) -> int:
    return piece_type if color == cshogi.BLACK else piece_type + 16


def sfen_core(sfen: str) -> str:
    return " ".join(sfen.split()[:3])


def make_sfen(board: list[int], hands: dict[int, list[int]], turn: int) -> str:
    ranks = []
    for rank in range(9):
        empty = 0
        row = []
        for file_number in range(9, 0, -1):
            square = (file_number - 1) * 9 + rank
            piece = board[square]
            if piece == cshogi.NONE:
                empty += 1
                continue
            if empty:
                row.append(str(empty))
                empty = 0
            row.append(piece_to_sfen_symbol(piece))
        if empty:
            row.append(str(empty))
        ranks.append("".join(row))

    hand_part = make_hand_sfen(hands)
    turn_part = "b" if turn == cshogi.BLACK else "w"
    return f"{'/'.join(ranks)} {turn_part} {hand_part} 1"


def piece_to_sfen_symbol(piece: int) -> str:
    color = piece_color(piece)
    piece_type = piece & 0x0F
    promoted = {
        cshogi.PROM_PAWN: "P",
        cshogi.PROM_LANCE: "L",
        cshogi.PROM_KNIGHT: "N",
        cshogi.PROM_SILVER: "S",
        cshogi.PROM_BISHOP: "B",
        cshogi.PROM_ROOK: "R",
    }
    plain = {
        cshogi.PAWN: "P",
        cshogi.LANCE: "L",
        cshogi.KNIGHT: "N",
        cshogi.SILVER: "S",
        cshogi.GOLD: "G",
        cshogi.BISHOP: "B",
        cshogi.ROOK: "R",
        cshogi.KING: "K",
    }
    if piece_type in promoted:
        symbol = "+" + promoted[piece_type]
    else:
        symbol = plain[piece_type]
    return symbol if color == cshogi.BLACK else symbol.lower()


def make_hand_sfen(hands: dict[int, list[int]]) -> str:
    order = (
        (cshogi.BLACK, cshogi.HROOK, "R"),
        (cshogi.BLACK, cshogi.HBISHOP, "B"),
        (cshogi.BLACK, cshogi.HGOLD, "G"),
        (cshogi.BLACK, cshogi.HSILVER, "S"),
        (cshogi.BLACK, cshogi.HKNIGHT, "N"),
        (cshogi.BLACK, cshogi.HLANCE, "L"),
        (cshogi.BLACK, cshogi.HPAWN, "P"),
        (cshogi.WHITE, cshogi.HROOK, "r"),
        (cshogi.WHITE, cshogi.HBISHOP, "b"),
        (cshogi.WHITE, cshogi.HGOLD, "g"),
        (cshogi.WHITE, cshogi.HSILVER, "s"),
        (cshogi.WHITE, cshogi.HKNIGHT, "n"),
        (cshogi.WHITE, cshogi.HLANCE, "l"),
        (cshogi.WHITE, cshogi.HPAWN, "p"),
    )
    parts = []
    for color, hand_kind, symbol in order:
        count = hands[color][hand_kind]
        if count == 0:
            continue
        parts.append(f"{count if count > 1 else ''}{symbol}")
    return "".join(parts) if parts else "-"
