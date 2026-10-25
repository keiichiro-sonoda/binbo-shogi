//! KP feature conversion for NNUE, kept byte-for-byte compatible with the Python
//! training implementation (`trainer/kp.py`). The training weights were learned
//! against that exact numbering, so inference here MUST produce the identical
//! index set for every position — a single mismatch silently invalidates the
//! learned evaluation.
//!
//! Canonical spec (numbering, dimensions, rationale, divergence from canonical
//! HalfKP, check values): `docs/kp_spec.md`. Update that document together with
//! any change here.
//!
//! Summary: two views (own-king based, opponent-king based), normalized to the
//! side-to-move's perspective. Square index follows cshogi/Apery:
//! `cshogi_square = shogi_core::Square::index() - 1 = (file-1)*9 + (rank-1)`,
//! with the side-to-move's view obtained by 180-degree rotation (`80 - sq`) for
//! White.

use shogi_core::{Color, PieceKind, Square};
use yasai::Position;

pub const SQUARE_NB: usize = 81;
pub const BOARD_PIECE_KINDS: usize = 13;
pub const BOARD_FEATURES_PER_KING: usize = BOARD_PIECE_KINDS * 2 * SQUARE_NB; // 2106
pub const HAND_SLOTS_PER_OWNER: usize = 38;
pub const HAND_FEATURES_PER_KING: usize = 2 * HAND_SLOTS_PER_OWNER; // 76
pub const FEATURES_PER_KING: usize = BOARD_FEATURES_PER_KING + HAND_FEATURES_PER_KING; // 2182
pub const VIEW_DIM: usize = SQUARE_NB * FEATURES_PER_KING; // 176742
#[allow(dead_code)] // public API constant (used by tests and future NNUE inference)
pub const TOTAL_FEATURES: usize = 2 * VIEW_DIM + 2; // 353486
pub const SIDE_FEATURE_BASE: usize = 2 * VIEW_DIM; // 353484

/// Hand piece kinds in the Python `HAND_PIECE_TYPES` order
/// (HPAWN, HLANCE, HKNIGHT, HSILVER, HGOLD, HBISHOP, HROOK).
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];
/// Cumulative slot offset per hand kind, from `MAX_PIECES_IN_HAND = [18,4,4,4,4,2,2]`.
const HAND_OFFSET: [usize; 7] = [0, 18, 22, 26, 30, 34, 36];

/// Board piece type id matching Python `BOARD_PIECE_TO_ID` (the index into
/// `BOARD_PIECE_TYPES`): note Gold is 6 (after Rook), promoted pieces are 7..=12.
fn board_piece_id(piece_kind: PieceKind) -> usize {
    match piece_kind {
        PieceKind::Pawn => 0,
        PieceKind::Lance => 1,
        PieceKind::Knight => 2,
        PieceKind::Silver => 3,
        PieceKind::Bishop => 4,
        PieceKind::Rook => 5,
        PieceKind::Gold => 6,
        PieceKind::ProPawn => 7,
        PieceKind::ProLance => 8,
        PieceKind::ProKnight => 9,
        PieceKind::ProSilver => 10,
        PieceKind::ProBishop => 11,
        PieceKind::ProRook => 12,
        PieceKind::King => unreachable!("king is encoded via king_square, not a board feature"),
    }
}

/// HalfKA board piece id: non-king kinds as in `board_piece_id`, plus King = 13
/// (Python `BOARD_PIECE_TO_ID_KA`). HalfKA includes the kings as board features.
fn board_piece_id_ka(piece_kind: PieceKind) -> usize {
    match piece_kind {
        PieceKind::King => 13,
        other => board_piece_id(other),
    }
}

fn cshogi_square(square: Square) -> usize {
    square.index() as usize - 1
}

fn color_index(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

/// Normalize a cshogi square to the side-to-move's perspective.
fn normalize(cshogi_sq: usize, turn: Color) -> usize {
    match turn {
        Color::Black => cshogi_sq,
        Color::White => SQUARE_NB - 1 - cshogi_sq,
    }
}

/// 180-degree rotation within the side-to-move frame (X -> 80 - X).
fn rotate(square: usize) -> usize {
    SQUARE_NB - 1 - square
}

/// Compute the sorted set of active naive-HalfKP feature indices for `pos`.
///
/// Matches `trainer/kp.py::board_to_kp_indices` (canonical=False). All indices are
/// unique by construction; the result is sorted to mirror Python's `sorted(set)`.
pub fn kp_indices(pos: &Position) -> Vec<u32> {
    kp_indices_with(pos, false)
}

/// Canonical HalfKP indices (Task 046, docs/kp_spec.md §7): the opponent half is
/// taken from the opponent's own perspective (side-to-move frame rotated 180 +
/// relative owner flipped). Matches `trainer/kp.py::board_to_kp_indices(canonical=True)`.
/// NOTE: this is the Task 046 TWO-BAND (non-shared W1) variant, kept for comparison.
pub fn kp_indices_canonical(pos: &Position) -> Vec<u32> {
    kp_indices_with(pos, true)
}

/// True canonical HalfKP shared band (Task 047): a SINGLE feature table of
/// `CANONICAL_SHARED_TOTAL` rows. Both halves index `[0, VIEW_DIM)` (anchor = own
/// king for `own`, the 180-rotated + owner-flipped opponent king for `opp`), so the
/// embedding row for a (king, piece) relation is shared across perspectives —
/// matching the original NNUE paper's shared W1. Returns `(own, opp)` separately
/// because the two halves can share index values. Matches
/// `trainer/kp.py::canonical_shared_halves`.
pub const CANONICAL_SHARED_SIDE_BASE: usize = VIEW_DIM; // 176742
pub const CANONICAL_SHARED_TOTAL: usize = VIEW_DIM + 2; // 176744

// --- HalfKA (Task 052): kings included as board piece features (id 13) ---------
// HalfKP omits the kings (anchors only). HalfKA adds them as a 14th board piece
// kind, so each half sees the enemy king's square = king safety. Parallel shared
// band numbering (Mode::SharedKA / magic BNN4); the HalfKP constants above and all
// existing models are untouched.
pub const BOARD_PIECE_KINDS_KA: usize = 14;
pub const BOARD_FEATURES_PER_KING_KA: usize = BOARD_PIECE_KINDS_KA * 2 * SQUARE_NB; // 2268
pub const FEATURES_PER_KING_KA: usize = BOARD_FEATURES_PER_KING_KA + HAND_FEATURES_PER_KING; // 2344
pub const VIEW_DIM_KA: usize = SQUARE_NB * FEATURES_PER_KING_KA; // 189864
pub const CANONICAL_SHARED_KA_SIDE_BASE: usize = VIEW_DIM_KA; // 189864
pub const CANONICAL_SHARED_KA_TOTAL: usize = VIEW_DIM_KA + 2; // 189866

pub fn kp_indices_canonical_shared(pos: &Position) -> (Vec<u32>, Vec<u32>) {
    let turn = pos.side_to_move();

    let mut board_pieces: Vec<(usize, PieceKind, Color)> = Vec::with_capacity(40);
    let mut self_king: Option<usize> = None;
    let mut opponent_king: Option<usize> = None;
    for square in Square::all() {
        let Some(piece) = pos.piece_at(square) else {
            continue;
        };
        let cshogi_sq = cshogi_square(square);
        let piece_kind = piece.piece_kind();
        let color = piece.color();
        if piece_kind == PieceKind::King {
            if color == turn {
                self_king = Some(cshogi_sq);
            } else {
                opponent_king = Some(cshogi_sq);
            }
        } else {
            board_pieces.push((cshogi_sq, piece_kind, color));
        }
    }

    let self_king = normalize(self_king.expect("missing side-to-move king"), turn);
    let opp_king = rotate(normalize(opponent_king.expect("missing opponent king"), turn));

    let mut own: Vec<u32> = Vec::with_capacity(40);
    let mut opp: Vec<u32> = Vec::with_capacity(40);
    // view=0 (no view multiplier) so both halves land in the shared band [0, VIEW_DIM).
    let own_base = self_king * FEATURES_PER_KING;
    let opp_base = opp_king * FEATURES_PER_KING;

    for &(cshogi_sq, piece_kind, color) in &board_pieces {
        let rel = if color == turn { 0 } else { 1 };
        let piece_id = board_piece_id(piece_kind);
        let psq = normalize(cshogi_sq, turn);
        let own_off = (rel * BOARD_PIECE_KINDS + piece_id) * SQUARE_NB + psq;
        let opp_off = ((1 - rel) * BOARD_PIECE_KINDS + piece_id) * SQUARE_NB + rotate(psq);
        own.push((own_base + own_off) as u32);
        opp.push((opp_base + opp_off) as u32);
    }

    for color in [Color::Black, Color::White] {
        let rel = if color == turn { 0 } else { 1 };
        let hand = pos.hand(color);
        for (kind_index, &hand_kind) in HAND_KINDS.iter().enumerate() {
            let count = hand.count(hand_kind).unwrap_or(0) as usize;
            for slot in 0..count {
                let own_off = BOARD_FEATURES_PER_KING + rel * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                let opp_off = BOARD_FEATURES_PER_KING + (1 - rel) * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                own.push((own_base + own_off) as u32);
                opp.push((opp_base + opp_off) as u32);
            }
        }
    }

    own.push((CANONICAL_SHARED_SIDE_BASE + color_index(turn)) as u32);
    own.sort_unstable();
    opp.sort_unstable();
    (own, opp)
}

/// HalfKA shared-band indices (Task 052): like `kp_indices_canonical_shared` but
/// the two kings are ALSO emitted as board features (id 13), so each half carries
/// the enemy king's square. own = 40 body + 1 side = 41; opp = 40 body. Matches
/// `trainer/kp.py::canonical_shared_ka_halves`.
pub fn kp_indices_canonical_shared_ka(pos: &Position) -> (Vec<u32>, Vec<u32>) {
    let turn = pos.side_to_move();

    let mut board_pieces: Vec<(usize, PieceKind, Color)> = Vec::with_capacity(40);
    let mut self_king: Option<usize> = None;
    let mut opponent_king: Option<usize> = None;
    for square in Square::all() {
        let Some(piece) = pos.piece_at(square) else {
            continue;
        };
        let cshogi_sq = cshogi_square(square);
        let piece_kind = piece.piece_kind();
        let color = piece.color();
        if piece_kind == PieceKind::King {
            if color == turn {
                self_king = Some(cshogi_sq);
            } else {
                opponent_king = Some(cshogi_sq);
            }
        }
        // HalfKA: every piece (kings included) is a board feature.
        board_pieces.push((cshogi_sq, piece_kind, color));
    }

    let self_king = normalize(self_king.expect("missing side-to-move king"), turn);
    let opp_king = rotate(normalize(opponent_king.expect("missing opponent king"), turn));

    let mut own: Vec<u32> = Vec::with_capacity(41);
    let mut opp: Vec<u32> = Vec::with_capacity(40);
    let own_base = self_king * FEATURES_PER_KING_KA;
    let opp_base = opp_king * FEATURES_PER_KING_KA;

    for &(cshogi_sq, piece_kind, color) in &board_pieces {
        let rel = if color == turn { 0 } else { 1 };
        let piece_id = board_piece_id_ka(piece_kind);
        let psq = normalize(cshogi_sq, turn);
        let own_off = (rel * BOARD_PIECE_KINDS_KA + piece_id) * SQUARE_NB + psq;
        let opp_off = ((1 - rel) * BOARD_PIECE_KINDS_KA + piece_id) * SQUARE_NB + rotate(psq);
        own.push((own_base + own_off) as u32);
        opp.push((opp_base + opp_off) as u32);
    }

    for color in [Color::Black, Color::White] {
        let rel = if color == turn { 0 } else { 1 };
        let hand = pos.hand(color);
        for (kind_index, &hand_kind) in HAND_KINDS.iter().enumerate() {
            let count = hand.count(hand_kind).unwrap_or(0) as usize;
            for slot in 0..count {
                let own_off = BOARD_FEATURES_PER_KING_KA + rel * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                let opp_off = BOARD_FEATURES_PER_KING_KA + (1 - rel) * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                own.push((own_base + own_off) as u32);
                opp.push((opp_base + opp_off) as u32);
            }
        }
    }

    own.push((CANONICAL_SHARED_KA_SIDE_BASE + color_index(turn)) as u32);
    own.sort_unstable();
    opp.sort_unstable();
    (own, opp)
}

/// HalfKAv2 shared-band indices (Task 053): like `kp_indices_canonical_shared_ka`
/// but each half DROPS its own anchor king as a piece feature (the redundant
/// `(anchor, anchor, K)` constant), keeping only the cross king. own = 38 body +
/// enemy-king + side = 40; opp = 38 body + own-king = 39. active = 79. Same KA
/// table. Matches `trainer/kp.py::canonical_shared_kav2_halves`.
pub fn kp_indices_canonical_shared_kav2(pos: &Position) -> (Vec<u32>, Vec<u32>) {
    let turn = pos.side_to_move();

    let mut board_pieces: Vec<(usize, PieceKind, Color)> = Vec::with_capacity(40);
    let mut self_king: Option<usize> = None;
    let mut opponent_king: Option<usize> = None;
    for square in Square::all() {
        let Some(piece) = pos.piece_at(square) else {
            continue;
        };
        let cshogi_sq = cshogi_square(square);
        let piece_kind = piece.piece_kind();
        let color = piece.color();
        if piece_kind == PieceKind::King {
            if color == turn {
                self_king = Some(cshogi_sq);
            } else {
                opponent_king = Some(cshogi_sq);
            }
        }
        board_pieces.push((cshogi_sq, piece_kind, color));
    }

    let self_king = normalize(self_king.expect("missing side-to-move king"), turn);
    let opp_king = rotate(normalize(opponent_king.expect("missing opponent king"), turn));

    let mut own: Vec<u32> = Vec::with_capacity(40);
    let mut opp: Vec<u32> = Vec::with_capacity(39);
    let own_base = self_king * FEATURES_PER_KING_KA;
    let opp_base = opp_king * FEATURES_PER_KING_KA;

    for &(cshogi_sq, piece_kind, color) in &board_pieces {
        let rel = if color == turn { 0 } else { 1 };
        let is_king = piece_kind == PieceKind::King;
        let piece_id = board_piece_id_ka(piece_kind);
        let psq = normalize(cshogi_sq, turn);
        if !(is_king && rel == 0) {
            // own half drops the own (anchor) king (rel == 0).
            let own_off = (rel * BOARD_PIECE_KINDS_KA + piece_id) * SQUARE_NB + psq;
            own.push((own_base + own_off) as u32);
        }
        if !(is_king && rel == 1) {
            // opp half drops the opponent (anchor) king (rel == 1).
            let opp_off = ((1 - rel) * BOARD_PIECE_KINDS_KA + piece_id) * SQUARE_NB + rotate(psq);
            opp.push((opp_base + opp_off) as u32);
        }
    }

    for color in [Color::Black, Color::White] {
        let rel = if color == turn { 0 } else { 1 };
        let hand = pos.hand(color);
        for (kind_index, &hand_kind) in HAND_KINDS.iter().enumerate() {
            let count = hand.count(hand_kind).unwrap_or(0) as usize;
            for slot in 0..count {
                let own_off = BOARD_FEATURES_PER_KING_KA + rel * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                let opp_off = BOARD_FEATURES_PER_KING_KA + (1 - rel) * HAND_SLOTS_PER_OWNER
                    + HAND_OFFSET[kind_index] + slot;
                own.push((own_base + own_off) as u32);
                opp.push((opp_base + opp_off) as u32);
            }
        }
    }

    own.push((CANONICAL_SHARED_KA_SIDE_BASE + color_index(turn)) as u32);
    own.sort_unstable();
    opp.sort_unstable();
    (own, opp)
}

// --- king-zone KPP (Task 053): piece PAIRS within each king's 5x5 zone ---------
// Matches `trainer/kp.py::kpp_zone_halves`. Side-to-move-normalized; own = pairs of
// non-king pieces within Chebyshev<=2 of the side-to-move king, opp = pairs near the
// opponent king with the canonical 180-rotation (zone_rel -> 24-zone_rel) + owner flip.
pub const KPP_ZONE_STATES: usize = 25 * BOARD_PIECE_KINDS * 2; // 650
pub const KPP_ZONE_TOTAL: usize = KPP_ZONE_STATES * (KPP_ZONE_STATES - 1) / 2; // 210925

fn kpp_zone_rel(psq_norm: usize, king_norm: usize) -> Option<usize> {
    let df = (psq_norm / 9) as i32 - (king_norm / 9) as i32;
    let dr = (psq_norm % 9) as i32 - (king_norm % 9) as i32;
    if !(-2..=2).contains(&df) || !(-2..=2).contains(&dr) {
        return None;
    }
    Some(((df + 2) * 5 + (dr + 2)) as usize)
}

fn kpp_zone_state(zone_rel: usize, ptype_id: usize, rel_owner: usize) -> usize {
    (zone_rel * BOARD_PIECE_KINDS + ptype_id) * 2 + rel_owner
}

fn kpp_pair_index(a: usize, b: usize) -> usize {
    let (a, b) = if a > b { (b, a) } else { (a, b) };
    a * KPP_ZONE_STATES - (a * (a + 1)) / 2 + (b - a - 1)
}

fn kpp_pairs(states: &[usize]) -> Vec<u32> {
    let mut set = std::collections::BTreeSet::new();
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            set.insert(kpp_pair_index(states[i], states[j]) as u32);
        }
    }
    set.into_iter().collect()
}

/// king-zone KPP pair indices `(own, opp)` for `pos`. Matches the Python reference.
pub fn kpp_zone_halves(pos: &Position) -> (Vec<u32>, Vec<u32>) {
    let turn = pos.side_to_move();
    let mut self_king: Option<usize> = None;
    let mut opponent_king: Option<usize> = None;
    for square in Square::all() {
        if let Some(piece) = pos.piece_at(square) {
            if piece.piece_kind() == PieceKind::King {
                if piece.color() == turn {
                    self_king = Some(cshogi_square(square));
                } else {
                    opponent_king = Some(cshogi_square(square));
                }
            }
        }
    }
    let self_king = normalize(self_king.expect("missing side-to-move king"), turn);
    let opp_king = normalize(opponent_king.expect("missing opponent king"), turn);

    let mut own_states: Vec<usize> = Vec::with_capacity(24);
    let mut opp_states: Vec<usize> = Vec::with_capacity(24);
    for square in Square::all() {
        let Some(piece) = pos.piece_at(square) else {
            continue;
        };
        let pk = piece.piece_kind();
        if pk == PieceKind::King {
            continue;
        }
        let psq = normalize(cshogi_square(square), turn);
        let rel = if piece.color() == turn { 0 } else { 1 };
        let pid = board_piece_id(pk);
        if let Some(zr) = kpp_zone_rel(psq, self_king) {
            own_states.push(kpp_zone_state(zr, pid, rel));
        }
        if let Some(zr) = kpp_zone_rel(psq, opp_king) {
            // canonical 180-rotation (24 - zone_rel) + owner flip (shared table).
            opp_states.push(kpp_zone_state(24 - zr, pid, 1 - rel));
        }
    }
    (kpp_pairs(&own_states), kpp_pairs(&opp_states))
}

fn kp_indices_with(pos: &Position, canonical: bool) -> Vec<u32> {
    let turn = pos.side_to_move();

    let mut board_pieces: Vec<(usize, PieceKind, Color)> = Vec::with_capacity(40);
    let mut self_king: Option<usize> = None;
    let mut opponent_king: Option<usize> = None;
    for square in Square::all() {
        let Some(piece) = pos.piece_at(square) else {
            continue;
        };
        let cshogi_sq = cshogi_square(square);
        let piece_kind = piece.piece_kind();
        let color = piece.color();
        if piece_kind == PieceKind::King {
            if color == turn {
                self_king = Some(cshogi_sq);
            } else {
                opponent_king = Some(cshogi_sq);
            }
        } else {
            board_pieces.push((cshogi_sq, piece_kind, color));
        }
    }

    let self_king = normalize(self_king.expect("missing side-to-move king"), turn);
    let opponent_king = normalize(opponent_king.expect("missing opponent king"), turn);

    // (view, anchor king square, flip). `flip` rotates piece squares 180 and flips
    // the relative owner = the opponent's own perspective. Naive never flips.
    let opp_king_anchor = if canonical { rotate(opponent_king) } else { opponent_king };
    let views = [
        (0_usize, self_king, false),
        (1_usize, opp_king_anchor, canonical),
    ];

    let mut indices: Vec<u32> = Vec::with_capacity(FEATURES_PER_KING.min(128));
    for (view, king_norm, flip) in views {
        let base = view * VIEW_DIM + king_norm * FEATURES_PER_KING;

        for &(cshogi_sq, piece_kind, color) in &board_pieces {
            let mut rel_owner = if color == turn { 0 } else { 1 };
            let piece_id = board_piece_id(piece_kind);
            let mut piece_square = normalize(cshogi_sq, turn);
            if flip {
                rel_owner = 1 - rel_owner;
                piece_square = rotate(piece_square);
            }
            let offset = (rel_owner * BOARD_PIECE_KINDS + piece_id) * SQUARE_NB + piece_square;
            indices.push((base + offset) as u32);
        }

        for color in [Color::Black, Color::White] {
            let mut rel_owner = if color == turn { 0 } else { 1 };
            if flip {
                rel_owner = 1 - rel_owner;
            }
            let hand = pos.hand(color);
            for (kind_index, &hand_kind) in HAND_KINDS.iter().enumerate() {
                let count = hand.count(hand_kind).unwrap_or(0) as usize;
                for slot in 0..count {
                    let offset = BOARD_FEATURES_PER_KING
                        + rel_owner * HAND_SLOTS_PER_OWNER
                        + HAND_OFFSET[kind_index]
                        + slot;
                    indices.push((base + offset) as u32);
                }
            }
        }
    }

    indices.push((SIDE_FEATURE_BASE + color_index(turn)) as u32);
    indices.sort_unstable();
    indices.dedup();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_position;

    #[test]
    fn dimensions_match_python() {
        assert_eq!(BOARD_FEATURES_PER_KING, 2106);
        assert_eq!(FEATURES_PER_KING, 2182);
        assert_eq!(VIEW_DIM, 176742);
        assert_eq!(TOTAL_FEATURES, 353486);
        assert_eq!(SIDE_FEATURE_BASE, 353484);
        // HalfKA (Task 052)
        assert_eq!(BOARD_FEATURES_PER_KING_KA, 2268);
        assert_eq!(FEATURES_PER_KING_KA, 2344);
        assert_eq!(VIEW_DIM_KA, 189864);
        assert_eq!(CANONICAL_SHARED_KA_TOTAL, 189866);
    }

    #[test]
    fn canonical_shared_ka_startpos_matches_python() {
        // Reference from `trainer/kp.py::canonical_shared_ka_halves` on the initial
        // position (Black to move). 81 active = own(40 body + side) + opp(40 body).
        // The two kings appear as board features (id 13): own king 104233, enemy
        // king 105359 (= the king-king relation HalfKP lacks). Startpos is
        // symmetric under 180-rotation + color swap, so own body == opp.
        let body: Vec<u32> = vec![
            103142, 103151, 103160, 103169, 103178, 103187, 103196, 103205, 103214, 103225,
            103297, 103315, 103369, 103405, 103441, 103530, 103557, 103657, 103675, 104233,
            104272, 104281, 104290, 104299, 104308, 104317, 104326, 104335, 104344, 104351,
            104423, 104441, 104495, 104531, 104567, 104604, 104739, 104783, 104801, 105359,
        ];
        let pos = initial_position();
        let (own, opp) = kp_indices_canonical_shared_ka(&pos);
        let mut expected_own = body.clone();
        expected_own.push(CANONICAL_SHARED_KA_SIDE_BASE as u32); // side, Black = 0
        assert_eq!(own, expected_own, "own half (Black to move)");
        assert_eq!(opp, body, "opp body == own body (shared rows in symmetric startpos)");
        assert_eq!(own.len(), 41);
        assert_eq!(opp.len(), 40);
        assert!(own.iter().all(|&i| (i as usize) < CANONICAL_SHARED_KA_TOTAL));
        assert!(opp.iter().all(|&i| (i as usize) < VIEW_DIM_KA));
    }

    #[test]
    fn canonical_shared_kav2_startpos_matches_python() {
        // HalfKAv2 (Task 053): HalfKA minus each half's anchor king. own drops the
        // own king 104233; opp drops the opp king; both keep the cross king 105359.
        // active 79 = own(38 body + enemy-king + side) + opp(38 body + own-king).
        // Reference from `trainer/kp.py::canonical_shared_kav2_halves` (startpos).
        let body: Vec<u32> = vec![
            103142, 103151, 103160, 103169, 103178, 103187, 103196, 103205, 103214, 103225,
            103297, 103315, 103369, 103405, 103441, 103530, 103557, 103657, 103675, 104272,
            104281, 104290, 104299, 104308, 104317, 104326, 104335, 104344, 104351, 104423,
            104441, 104495, 104531, 104567, 104604, 104739, 104783, 104801, 105359,
        ];
        let pos = initial_position();
        let (own, opp) = kp_indices_canonical_shared_kav2(&pos);
        let mut expected_own = body.clone();
        expected_own.push(CANONICAL_SHARED_KA_SIDE_BASE as u32);
        assert_eq!(own, expected_own, "own half (Black to move)");
        assert_eq!(opp, body, "opp body (symmetric startpos)");
        assert_eq!(own.len(), 40);
        assert_eq!(opp.len(), 39);
        // own king (104233) dropped from own; cross king (105359) kept in both.
        assert!(!own.contains(&104233));
        assert!(own.contains(&105359) && opp.contains(&105359));
    }

    #[test]
    fn startpos_matches_python_reference() {
        // Reference produced by `trainer/kp.py::board_to_kp_indices` on the
        // initial position (side to move = Black). 77 active features.
        let expected: Vec<u32> = vec![
            96014, 96023, 96032, 96041, 96050, 96059, 96068, 96077, 96086, 96097, 96169, 96187,
            96241, 96277, 96313, 96402, 96429, 96529, 96547, 97063, 97072, 97081, 97090, 97099,
            97108, 97117, 97126, 97135, 97142, 97214, 97232, 97286, 97322, 97358, 97395, 97530,
            97574, 97592, 255300, 255309, 255318, 255327, 255336, 255345, 255354, 255363, 255372,
            255383, 255455, 255473, 255527, 255563, 255599, 255688, 255715, 255815, 255833, 256349,
            256358, 256367, 256376, 256385, 256394, 256403, 256412, 256421, 256428, 256500, 256518,
            256572, 256608, 256644, 256681, 256816, 256860, 256878, 353484,
        ];
        let pos = initial_position();
        let got = kp_indices(&pos);
        assert_eq!(got.len(), 77, "startpos must have 77 active features");
        assert_eq!(got, expected);
    }

    #[test]
    fn canonical_shared_startpos_matches_python_and_shares_band() {
        // Initial position is symmetric under 180-rotation + color swap, so the own
        // and opponent halves share the EXACT same 38 body rows (the W1 sharing).
        let body: Vec<u32> = vec![
            96014, 96023, 96032, 96041, 96050, 96059, 96068, 96077, 96086, 96097, 96169, 96187,
            96241, 96277, 96313, 96402, 96429, 96529, 96547, 97063, 97072, 97081, 97090, 97099,
            97108, 97117, 97126, 97135, 97142, 97214, 97232, 97286, 97322, 97358, 97395, 97530,
            97574, 97592,
        ];
        let pos = initial_position();
        let (own, opp) = kp_indices_canonical_shared(&pos);
        // own = 38 body + side feature (CANONICAL_SHARED_SIDE_BASE + 0 for Black).
        let mut expected_own = body.clone();
        expected_own.push(CANONICAL_SHARED_SIDE_BASE as u32);
        assert_eq!(own, expected_own, "own half (Black to move)");
        assert_eq!(opp, body, "opp half == own body (shared rows in symmetric startpos)");
        // Both halves index the single shared band [0, VIEW_DIM) (+ the side row).
        assert!(opp.iter().all(|&i| (i as usize) < VIEW_DIM));
        assert!(own.iter().all(|&i| (i as usize) < CANONICAL_SHARED_TOTAL));
        assert_eq!(CANONICAL_SHARED_TOTAL, VIEW_DIM + 2);
    }

    #[test]
    fn kpp_zone_startpos_matches_python() {
        // king-zone KPP (Task 053). Reference from `trainer/kp.py::kpp_zone_halves`
        // on the initial position: 36 pairs/half; symmetric startpos -> own == opp.
        let pos = initial_position();
        let (own, opp) = kpp_zone_halves(&pos);
        assert_eq!(own.len(), 36);
        assert_eq!(opp.len(), 36);
        assert_eq!(own, opp, "symmetric startpos");
        assert_eq!(&own[..8], &[57, 129, 193, 259, 389, 453, 519, 577]);
        assert_eq!(&own[32..], &[177442, 191880, 191938, 202597]);
        assert_eq!(KPP_ZONE_STATES, 650);
        assert_eq!(KPP_ZONE_TOTAL, 210925);
        assert!(own.iter().all(|&i| (i as usize) < KPP_ZONE_TOTAL));
    }

    #[test]
    fn all_indices_in_range_and_one_side_feature() {
        let pos = initial_position();
        let indices = kp_indices(&pos);
        assert!(indices.iter().all(|&i| (i as usize) < TOTAL_FEATURES));
        let side = indices
            .iter()
            .filter(|&&i| (i as usize) >= SIDE_FEATURE_BASE)
            .count();
        assert_eq!(side, 1, "exactly one side-to-move feature");
    }
}
