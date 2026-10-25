//! Fixed-depth negamax search with alpha-beta pruning and a transposition table.
//!
//! Search steps so far (each measured against the previous baseline):
//! - Task 022: pure alpha-beta.
//! - Task 023: transposition table (this file). HashMap keyed by the Zobrist
//!   `pos.key()`, storing score + bound + depth + best move. A standard TT that
//!   reuses results from *deeper or equal* searches, so it is not bit-identical
//!   to the no-TT search (it can be more accurate); correctness is checked by
//!   self-play (TT must not lose to no-TT at the same depth), not by equality.
//!
//! - Task 024: iterative deepening (TT carried across depths).
//! - Task 025: hash-move ordering (TT best move tried first).
//! - Task 027: time management + per-iteration info (this revision). Iterative
//!   deepening is stopped within a time budget; a depth started but not finished
//!   in time is aborted and discarded, so the move returned is always from a
//!   fully completed depth.
//!
//! Still TODO (later, one at a time): MVV-LVA / killer / history ordering,
//! fixed-size array TT, MTD(f)/PVS, quiescence, `stop`, pv output.
//!
//! Leaves are evaluated by the NNUE (`src/eval/nnue.rs`); its f32 logit becomes
//! an order-preserving integer score. Search is negamax (each node scored from
//! the side-to-move's viewpoint, child scores negated).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use shogi_core::{Move, PieceKind};
use yasai::Position;

use crate::eval::nnue::Nnue;

/// Beyond any real score; the alpha-beta window starts at +/-this.
pub const VALUE_INF: i32 = 1_000_000;
/// Checkmate magnitude. A mate in `ply` half-moves scores `VALUE_MATE - ply`,
/// so shallower mates score higher and are clearly separated from evaluations.
pub const VALUE_MATE: i32 = 30_000;
/// Scores with magnitude >= this are mate scores (never plain evaluations).
/// Evaluations are clamped below it with a buffer, so mate distances up to
/// `VALUE_MATE - MATE_THRESHOLD` plies are representable without collision.
const MATE_THRESHOLD: i32 = VALUE_MATE - 1000;

/// Maps an NNUE logit to an integer score (order-preserving). The scale is the
/// training sigmoid scale (600), giving roughly centipawn-sized numbers, but the
/// only requirement here is that ordering is preserved. Clamped well below the
/// mate range so evaluations and mate scores never collide.
const EVAL_SCALE: f32 = 600.0;
const EVAL_CLAMP: i32 = MATE_THRESHOLD - 1000; // 28000

/// Maximum principal-variation length we keep/report. PVs rarely reach the search
/// depth; this is a safety cap (it also bounds the TT-walk fill against cycles).
const MAX_PV_LEN: usize = 64;

pub fn eval_to_int(logit: f32) -> i32 {
    let scaled = (logit * EVAL_SCALE).round() as i32;
    scaled.clamp(-EVAL_CLAMP, EVAL_CLAMP)
}

/// Format a score as a USI `score` value: `cp <n>` for evaluations, or
/// `mate <plies>` (we mate) / `mate -<plies>` (we are mated) for mate scores.
pub fn usi_score(score: i32) -> String {
    if score >= MATE_THRESHOLD {
        format!("mate {}", VALUE_MATE - score)
    } else if score <= -MATE_THRESHOLD {
        format!("mate -{}", VALUE_MATE + score)
    } else {
        format!("cp {score}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
struct TtEntry {
    score: i32,
    bound: Bound,
    depth: u16,
    /// Best move from this position; used as the hash move for ordering.
    best_move: Option<Move>,
}

/// Transposition table keyed by the Zobrist position key. HashMap keeps the full
/// 64-bit key, so there are no index collisions (only true Zobrist collisions,
/// negligible). Cleared at the start of every search.
pub struct Tt {
    table: HashMap<u64, TtEntry>,
}

impl Tt {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }

    #[allow(dead_code)] // entry count, for diagnostics
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Depth-preferred replacement: keep the entry searched to the greater
    /// remaining depth (ties overwrite, favoring the latest result).
    fn store(&mut self, key: u64, entry: TtEntry) {
        match self.table.get(&key) {
            Some(existing) if existing.depth > entry.depth => {}
            _ => {
                self.table.insert(key, entry);
            }
        }
    }
}

impl Default for Tt {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a (root-relative) mate score to a node-relative one for storage, so a
/// transposed position carries the correct mate distance from itself.
fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    }
}

/// Inverse of `score_to_tt`: restore a stored node-relative mate score to the
/// current node's root-relative score.
fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply
    } else if score <= -MATE_THRESHOLD {
        score + ply
    } else {
        score
    }
}

/// Search feature toggles, bundled so adding more later doesn't churn every
/// signature. `full()` is what the engine uses; the bench flips individual
/// flags to measure each feature in isolation.
#[derive(Debug, Clone, Copy)]
pub struct Opts {
    /// Alpha-beta cutoffs (false = full-width minimax, for node counts).
    pub prune: bool,
    /// Use the transposition table (probe/store).
    pub use_tt: bool,
    /// Hash-move ordering: try the TT best move first.
    pub use_ordering: bool,
    /// MVV-LVA ordering: after the hash move, try captures (most valuable
    /// victim, least valuable attacker) before quiet moves. Requires
    /// `use_ordering`. Only affects search order, never the result.
    pub use_mvv_lva: bool,
    /// Quiescence search at the leaf (Task 049): resolve captures / check
    /// evasions before evaluating, to dampen the horizon effect.
    pub use_quiescence: bool,
    /// Killer-move ordering: try quiet moves that caused a beta cutoff at the
    /// same ply first (search-order hint only; never changes the result).
    pub use_killers: bool,
    /// History-heuristic ordering: order remaining quiet moves by how often they
    /// have caused cutoffs (search-order hint only).
    pub use_history: bool,
    /// Check extension: search one ply deeper when the side to move is in check
    /// (do not drop into the leaf while in check).
    pub use_check_ext: bool,
    /// Principal-variation search (null-window scout + re-search). Same value as
    /// plain alpha-beta, fewer nodes with good move ordering.
    pub use_pvs: bool,
    /// Late move reductions: search late quiet non-killer moves at reduced depth
    /// first, re-searching at full depth only if they beat alpha. Trades a small
    /// risk of missing a deep quiet move for substantially deeper search.
    pub use_lmr: bool,
}

impl Opts {
    pub fn full() -> Self {
        Self {
            prune: true,
            use_tt: true,
            use_ordering: true,
            use_mvv_lva: true,
            use_quiescence: true,
            use_killers: true,
            use_history: true,
            use_check_ext: true,
            use_pvs: true,
            use_lmr: true,
        }
    }
}

/// Quiet moves that caused a beta cutoff (killers) are ordered just below
/// captures and above other quiet moves.
const ORDER_KILLER: i32 = 900;
/// Cap on a quiescence search's extra ply depth (captures are finite, but check
/// evasions could otherwise recurse; the time limit also bounds it).
const MAX_QUIESCENCE_PLY: i32 = 32;

/// Piece value used ONLY for move ordering (MVV-LVA). This is a search-order
/// hint, completely separate from the NNUE evaluation (see DESIGN.md: the
/// knowledge-zero rule constrains the evaluation, not ordering hints). Rough
/// values are fine; ordering does not change the search result.
fn order_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 1,
        PieceKind::Lance => 3,
        PieceKind::Knight => 3,
        PieceKind::Silver => 5,
        PieceKind::Gold => 6,
        PieceKind::ProPawn | PieceKind::ProLance | PieceKind::ProKnight | PieceKind::ProSilver => 6,
        PieceKind::Bishop => 8,
        PieceKind::Rook => 10,
        PieceKind::ProBishop => 10,
        PieceKind::ProRook => 12,
        PieceKind::King => 100, // never actually captured in a legal move
    }
}

const ORDER_HASH: i32 = 1_000_000;
const ORDER_CAPTURE_BASE: i32 = 1_000;

/// Ordering priority for a move: hash move first, then captures by MVV-LVA
/// (victim value dominant, attacker value as a small tie-breaker), then quiet
/// moves. Reads only the piece-value table and the board (no NNUE).
fn order_key(pos: &Position, mv: Move, hash_move: Option<Move>) -> i32 {
    if Some(mv) == hash_move {
        return ORDER_HASH;
    }
    // A capture is a board move (not a drop) onto an occupied square.
    if let Some(from) = mv.from() {
        if let Some(victim) = pos.piece_at(mv.to()) {
            let attacker = pos
                .piece_at(from)
                .map(|p| p.piece_kind())
                .unwrap_or(PieceKind::King);
            return ORDER_CAPTURE_BASE + order_value(victim.piece_kind()) * 32
                - order_value(attacker);
        }
    }
    0 // quiet move
}

/// Reorder the legal moves in place. With `use_mvv_lva`, sort by `order_key`
/// (hash move, then captures by MVV-LVA, then quiet); otherwise just move the
/// hash move to the front (Task 025 behavior). `moves` is taken as a slice so
/// the concrete `ArrayVec` type need not be named.
fn order_moves(moves: &mut [Move], pos: &Position, hash_move: Option<Move>, use_mvv_lva: bool) {
    if use_mvv_lva {
        moves.sort_unstable_by_key(|&mv| core::cmp::Reverse(order_key(pos, mv, hash_move)));
    } else if let Some(hm) = hash_move {
        if let Some(idx) = moves.iter().position(|&m| m == hm) {
            if idx != 0 {
                // Hash move to the front, preserving the order of the rest.
                moves[..=idx].rotate_right(1);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The chosen move, or `None` when there are no legal moves (resign).
    pub best_move: Option<Move>,
    pub score: i32,
    pub nodes: u64,
    #[allow(dead_code)] // searched depth, for callers that report it
    pub depth: u16,
    /// Principal variation (best line) from the root, in play order. Empty if no
    /// move (resign) or if the TT is disabled and nothing improved alpha.
    pub pv: Vec<Move>,
}

/// Reported once per completed iterative-deepening depth (for USI `info`).
pub struct IterInfo {
    pub depth: u16,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    /// Principal variation for this depth, in play order (for `info ... pv ...`).
    pub pv: Vec<Move>,
}

/// Don't start a new iteration once this fraction of the budget is spent (the
/// next depth typically costs more than everything so far).
const SOFT_FRACTION: f64 = 0.5;
/// Check the clock every this many nodes (Instant::now() is not free).
const TIME_CHECK_INTERVAL: u32 = 2048;

/// Holds the per-search state (net, TT, options, node count, time deadline) so
/// it does not have to be threaded through every recursive call.
struct Searcher<'a> {
    net: &'a Nnue,
    tt: &'a mut Tt,
    opts: Opts,
    nodes: u64,
    /// Hard abort deadline; `None` means never abort (no time limit / depth 1).
    deadline: Option<Instant>,
    aborted: bool,
    since_check: u32,
    /// Triangular PV table: `pv[ply]` holds the principal variation from `ply`
    /// downward (in play order). Built as the search finds improving moves, so
    /// it is exactly the line the search chose. Reading/writing it does not
    /// affect node counts or the returned score.
    pv: Vec<Vec<Move>>,
    /// Two killer moves per ply (quiet moves that caused a beta cutoff there).
    killers: Vec<[Option<Move>; 2]>,
    /// History heuristic, indexed by `history_index(mv)` (from*128 + to). Flat
    /// array (shogi_core::Move is not Hash in this build, and this is faster).
    history: Vec<i32>,
}

/// Index a move into the history table: `from_square * 128 + to_square`
/// (`Square::index()` is 1..=81; drops use from=0). Bounded by 81*128+81.
fn history_index(mv: Move) -> usize {
    let from = mv.from().map(|s| s.index() as usize).unwrap_or(0);
    from * 128 + mv.to().index() as usize
}
const HISTORY_SIZE: usize = 128 * 128;

impl<'a> Searcher<'a> {
    fn new(net: &'a Nnue, tt: &'a mut Tt, opts: Opts) -> Self {
        Self {
            net,
            tt,
            opts,
            nodes: 0,
            deadline: None,
            aborted: false,
            since_check: 0,
            pv: Vec::new(),
            killers: Vec::new(),
            history: vec![0; HISTORY_SIZE],
        }
    }

    fn ensure_killers_len(&mut self, len: usize) {
        if self.killers.len() < len {
            self.killers.resize(len, [None, None]);
        }
    }

    /// Static leaf evaluation as an order-preserving integer score. The f32 modes
    /// (BNN1..BNN6) return a logit, scaled by `EVAL_SCALE`; `Mode::Quant` (BNNQ)
    /// returns centipawns directly (bit-exact with YaneuraOu, Task 058), so it is only
    /// rounded/clamped, NOT re-scaled. This is what lets binbo's search run the same
    /// eval YaneuraOu uses (Task 060).
    fn leaf_eval(&self, pos: &Position) -> i32 {
        let v = self.net.forward(pos);
        if self.net.is_quant() {
            (v.round() as i32).clamp(-EVAL_CLAMP, EVAL_CLAMP)
        } else {
            eval_to_int(v)
        }
    }

    /// Record a quiet beta-cutoff move as a killer + history bonus for `ply`.
    fn record_cutoff(&mut self, pos: &Position, mv: Move, depth: u16, ply: usize) {
        // Only quiet moves (captures already order high via MVV-LVA).
        let is_capture = mv.from().is_some() && pos.piece_at(mv.to()).is_some();
        if is_capture {
            return;
        }
        if self.opts.use_killers {
            self.ensure_killers_len(ply + 1);
            if self.killers[ply][0] != Some(mv) {
                self.killers[ply][1] = self.killers[ply][0];
                self.killers[ply][0] = Some(mv);
            }
        }
        if self.opts.use_history {
            let bonus = (depth as i32) * (depth as i32);
            let e = &mut self.history[history_index(mv)];
            *e = (*e + bonus).min(ORDER_KILLER - 1); // keep below killers/captures
        }
    }

    /// Ordering key incorporating hash move, MVV-LVA captures, killers and history.
    fn order_key2(&self, pos: &Position, mv: Move, hash_move: Option<Move>, ply: usize) -> i32 {
        if Some(mv) == hash_move {
            return ORDER_HASH;
        }
        if self.opts.use_mvv_lva {
            if let Some(from) = mv.from() {
                if let Some(victim) = pos.piece_at(mv.to()) {
                    let attacker = pos
                        .piece_at(from)
                        .map(|p| p.piece_kind())
                        .unwrap_or(PieceKind::King);
                    return ORDER_CAPTURE_BASE + order_value(victim.piece_kind()) * 32
                        - order_value(attacker);
                }
            }
        }
        if self.opts.use_killers && ply < self.killers.len() {
            let k = self.killers[ply];
            if k[0] == Some(mv) {
                return ORDER_KILLER + 1;
            }
            if k[1] == Some(mv) {
                return ORDER_KILLER;
            }
        }
        if self.opts.use_history {
            return self.history[history_index(mv)];
        }
        0
    }

    /// Order moves in place using the full key (hash/MVV-LVA/killer/history).
    fn order_moves_at(&self, moves: &mut [Move], pos: &Position, hash_move: Option<Move>, ply: usize) {
        moves.sort_unstable_by_key(|&mv| core::cmp::Reverse(self.order_key2(pos, mv, hash_move, ply)));
    }

    /// Quiescence search: from a leaf, keep resolving captures (and, when in
    /// check, all evasions) until the position is "quiet", returning a stable
    /// stand-pat-bounded evaluation. Bounds the horizon effect without growing
    /// the main search depth. Captures strictly reduce material, so it
    /// terminates; the ply cap + time limit guard check-evasion recursion.
    fn quiescence(&mut self, pos: &mut Position, ply: i32, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0;
        }
        let in_check = pos.in_check();
        let mut best;
        if in_check {
            // Cannot stand pat while in check: must find an evasion.
            best = -VALUE_INF;
        } else {
            best = self.leaf_eval(pos);
            if best >= beta {
                return best;
            }
            if best > alpha {
                alpha = best;
            }
        }
        if ply >= MAX_QUIESCENCE_PLY {
            return if in_check { self.leaf_eval(pos) } else { best };
        }

        let legal = pos.legal_moves();
        if legal.is_empty() {
            // No legal move = checkmate (only reachable while in check; no stalemate).
            return -(VALUE_MATE - ply);
        }
        // In check: search all evasions. Quiet: only captures (resolve exchanges).
        let mut moves: Vec<Move> = if in_check {
            legal.iter().copied().collect()
        } else {
            legal
                .iter()
                .copied()
                .filter(|&mv| mv.from().is_some() && pos.piece_at(mv.to()).is_some())
                .collect()
        };
        if !in_check && moves.is_empty() {
            return best; // already quiet (no captures)
        }
        // MVV-LVA order (no killers/history in qsearch).
        moves.sort_unstable_by_key(|&mv| core::cmp::Reverse(order_key(pos, mv, None)));

        for mv in moves {
            pos.do_move(mv);
            let score = -self.quiescence(pos, ply + 1, -beta, -alpha);
            pos.undo_move(mv);
            if self.aborted {
                return best;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    /// Make sure the PV table has at least `len` ply slots.
    fn ensure_pv_len(&mut self, len: usize) {
        if self.pv.len() < len {
            self.pv.resize_with(len, Vec::new);
        }
    }

    /// Set `pv[ply] = mv` followed by the child's PV (`pv[ply + 1]`).
    fn update_pv(&mut self, ply: usize, mv: Move) {
        let (left, right) = self.pv.split_at_mut(ply + 1);
        let line = &mut left[ply];
        line.clear();
        line.push(mv);
        line.extend_from_slice(&right[0]);
        line.truncate(MAX_PV_LEN);
    }

    /// Rebuild `pv[ply]` by following TT best moves from `pos`. Used when an exact
    /// TT hit lets us skip a subtree, so the PV does not truncate at that node.
    /// Read-only on the TT; every move is legality-checked, and the walk stops on
    /// a repeated position or the length cap (so it cannot loop forever).
    fn fill_pv_from_tt(&mut self, pos: &Position, ply: usize) {
        self.pv[ply].clear();
        let mut work = pos.clone();
        let mut seen: Vec<u64> = Vec::new();
        while self.pv[ply].len() < MAX_PV_LEN {
            let key = work.key();
            if seen.contains(&key) {
                break;
            }
            seen.push(key);
            let Some(mv) = self.tt.table.get(&key).and_then(|entry| entry.best_move) else {
                break;
            };
            if !work.legal_moves().iter().any(|&m| m == mv) {
                break;
            }
            self.pv[ply].push(mv);
            work.do_move(mv);
        }
    }

    /// True once the hard deadline has passed (checked every few thousand nodes).
    fn time_up(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        let Some(deadline) = self.deadline else {
            return false;
        };
        self.since_check += 1;
        if self.since_check >= TIME_CHECK_INTERVAL {
            self.since_check = 0;
            if Instant::now() >= deadline {
                self.aborted = true;
            }
        }
        self.aborted
    }

    /// One root search at a fixed `depth`. Returns the best move and score; on a
    /// time abort the result is meaningless and the caller must check `aborted`.
    fn root_search(&mut self, pos: &Position, depth: u16) -> (Option<Move>, i32) {
        self.nodes += 1;
        self.ensure_pv_len(depth as usize + 2);
        self.pv[0].clear();
        let key = pos.key();
        let hash_move = if self.opts.use_tt {
            self.tt.table.get(&key).and_then(|entry| entry.best_move)
        } else {
            None
        };

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return (None, -VALUE_MATE);
        }
        if self.opts.use_ordering {
            order_moves(&mut moves, pos, hash_move, self.opts.use_mvv_lva);
        }

        let mut work = pos.clone();
        let mut alpha = -VALUE_INF;
        let beta = VALUE_INF;
        let mut best_move = moves[0];
        let mut best_score = -VALUE_INF;
        let child_depth = depth.saturating_sub(1);

        for mv in moves {
            work.do_move(mv);
            let score = -self.nega_alpha(&mut work, child_depth, 1, -beta, -alpha);
            work.undo_move(mv);
            if self.aborted {
                return (Some(best_move), best_score); // ignored: caller checks aborted
            }
            if score > best_score {
                best_score = score;
                best_move = mv;
                // Root window is full, so every improvement is a real PV move.
                self.update_pv(0, mv);
            }
            if score > alpha {
                alpha = score;
            }
        }

        if self.opts.use_tt {
            // Root is searched with a full window, so its value is exact. ply = 0.
            self.tt.store(
                key,
                TtEntry {
                    score: score_to_tt(best_score, 0),
                    bound: Bound::Exact,
                    depth,
                    best_move: Some(best_move),
                },
            );
        }

        (Some(best_move), best_score)
    }

    /// Negamax with alpha-beta and TT. Returns the side-to-move score. `ply` is
    /// the distance from the root (for mate distance and TT mate correction).
    fn nega_alpha(&mut self, pos: &mut Position, depth: u16, ply: i32, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0; // aborted; value ignored once `aborted` is set
        }
        // Check extension can take `ply` past the nominal `depth`, so grow the PV
        // table on demand (the root only sized it for the un-extended depth).
        self.ensure_pv_len(ply as usize + 2);
        // Start this node's PV empty so a leaf / terminal / fail-low return leaves
        // no stale line for the parent to splice in.
        self.pv[ply as usize].clear();

        // Check extension: search one ply deeper while in check (so the line after
        // escaping the check is seen at full depth, and we never evaluate in check).
        // Capped by ply so a perpetual check cannot recurse forever at depth 1
        // (which runs with no deadline).
        let depth = if self.opts.use_check_ext && ply < 48 && pos.in_check() {
            depth + 1
        } else {
            depth
        };

        let key = pos.key();
        let mut hash_move = None;
        if self.opts.use_tt {
            if let Some(entry) = self.tt.table.get(&key) {
                // Hash move orders regardless of depth; cutoffs need stored depth.
                hash_move = entry.best_move;
                if entry.depth >= depth {
                    let score = score_from_tt(entry.score, ply);
                    match entry.bound {
                        Bound::Exact => {
                            // Exact hits sit on the PV; rebuild the line from the
                            // TT so it does not truncate here.
                            self.fill_pv_from_tt(pos, ply as usize);
                            return score;
                        }
                        Bound::Lower if score >= beta => return score,
                        Bound::Upper if score <= alpha => return score,
                        _ => {}
                    }
                }
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            // No legal move = checkmate (shogi has no stalemate): lost.
            return -(VALUE_MATE - ply);
        }
        if depth == 0 {
            if self.opts.use_quiescence {
                return self.quiescence(pos, ply, alpha, beta);
            }
            return self.leaf_eval(pos);
        }
        if self.opts.use_ordering {
            self.order_moves_at(&mut moves, pos, hash_move, ply as usize);
        }

        let alpha_orig = alpha;
        let mut best = -VALUE_INF;
        let mut best_move = moves[0];
        let parent_in_check = pos.in_check();
        let child_depth = depth - 1;
        for (i, mv) in moves.into_iter().enumerate() {
            // Classify BEFORE do_move (the victim sits on mv.to() until then).
            let is_capture = mv.from().is_some() && pos.piece_at(mv.to()).is_some();
            let is_killer = self.opts.use_killers
                && (ply as usize) < self.killers.len()
                && (self.killers[ply as usize][0] == Some(mv)
                    || self.killers[ply as usize][1] == Some(mv));
            pos.do_move(mv);
            // PVS: first move full window; later moves a null-window scout, only
            // re-searched fully if they beat alpha. LMR reduces the scout depth of
            // late quiet non-killer moves (good ordering => they are likely bad).
            let score = if i == 0 || !self.opts.use_pvs {
                -self.nega_alpha(pos, child_depth, ply + 1, -beta, -alpha)
            } else {
                let reduce = self.opts.use_lmr
                    && i >= 3
                    && depth >= 3
                    && !is_capture
                    && !is_killer
                    && !parent_in_check
                    && !pos.in_check(); // do not reduce checking moves
                let red: u16 = if reduce {
                    // Reduce more for later moves / deeper nodes (search deeper).
                    let mut r = 1u16;
                    if i >= 6 {
                        r += 1;
                    }
                    if i >= 12 && depth >= 6 {
                        r += 1;
                    }
                    r.min(child_depth.saturating_sub(1).max(1))
                } else {
                    0
                };
                let mut s = -self.nega_alpha(
                    pos,
                    child_depth.saturating_sub(red),
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                );
                if red > 0 && s > alpha && !self.aborted {
                    // Reduced scout beat alpha: re-search at full depth (null window).
                    s = -self.nega_alpha(pos, child_depth, ply + 1, -(alpha + 1), -alpha);
                }
                if s > alpha && s < beta && !self.aborted {
                    // PVS full-window re-search.
                    s = -self.nega_alpha(pos, child_depth, ply + 1, -beta, -alpha);
                }
                s
            };
            pos.undo_move(mv);
            if self.aborted {
                return best; // unwind without storing a partial result
            }
            if score > best {
                best = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    // Exact improvement inside the window: record the PV move and
                    // the child's continuation. (>= beta would be a fail-high,
                    // not part of the final PV, so it is skipped.)
                    if score < beta {
                        self.update_pv(ply as usize, mv);
                    }
                }
            }
            if self.opts.prune && alpha >= beta {
                // Beta cutoff: reward this move for ordering future siblings
                // (killers + history). Quiet moves only (captures order high already).
                if self.opts.use_killers || self.opts.use_history {
                    self.record_cutoff(pos, mv, depth, ply as usize);
                }
                break;
            }
        }

        if self.opts.use_tt {
            let bound = if best <= alpha_orig {
                Bound::Upper
            } else if best >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt.store(
                key,
                TtEntry {
                    score: score_to_tt(best, ply),
                    bound,
                    depth,
                    best_move: Some(best_move),
                },
            );
        }

        best
    }
}

/// Search the root position to a fixed `depth` and return the best move (no time
/// limit). The TT (if enabled) is cleared so each search starts fresh.
pub fn search_root(net: &Nnue, pos: &Position, depth: u16, opts: Opts, tt: &mut Tt) -> SearchResult {
    if opts.use_tt {
        tt.clear();
    }
    let mut searcher = Searcher::new(net, tt, opts);
    let (best_move, score) = searcher.root_search(pos, depth);
    let pv = searcher.pv.first().cloned().unwrap_or_default();
    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
        depth,
        pv,
    }
}

/// Iterative deepening with no time limit (searches every depth 1..=max_depth).
/// Used by benchmarks and self-play, where determinism matters.
pub fn search_iddfs(net: &Nnue, pos: &Position, max_depth: u16, opts: Opts, tt: &mut Tt) -> SearchResult {
    let mut noop = |_: &IterInfo| {};
    search_iddfs_timed(net, pos, max_depth, opts, tt, None, &mut noop)
}

/// Iterative deepening within a time `budget` (the total time allowed for this
/// move; `None` = no limit). Carries the TT across depths. `on_iter` is called
/// after each completed depth (for USI `info`). A depth that cannot finish in
/// time is aborted and discarded, so the returned move is always from a fully
/// completed depth (depth 1 always completes, guaranteeing a move).
pub fn search_iddfs_timed(
    net: &Nnue,
    pos: &Position,
    max_depth: u16,
    opts: Opts,
    tt: &mut Tt,
    budget: Option<Duration>,
    on_iter: &mut dyn FnMut(&IterInfo),
) -> SearchResult {
    if opts.use_tt {
        tt.clear();
    }
    let start = Instant::now();
    let hard_deadline = budget.map(|b| start + b);
    let mut searcher = Searcher::new(net, tt, opts);

    let mut best_move = None;
    let mut score = 0;
    let mut reached = 0;
    let mut best_pv: Vec<Move> = Vec::new();
    for depth in 1..=max_depth.max(1) {
        // Soft cutoff: don't start a new depth if most of the budget is gone.
        if depth > 1 {
            if let Some(b) = budget {
                if start.elapsed().as_secs_f64() >= b.as_secs_f64() * SOFT_FRACTION {
                    break;
                }
            }
        }
        // Depth 1 always completes (guarantees a move); deeper depths can abort.
        searcher.deadline = if depth > 1 { hard_deadline } else { None };
        searcher.aborted = false;

        let (bm, sc) = searcher.root_search(pos, depth);
        if searcher.aborted {
            break; // discard the unfinished depth; keep the previous result
        }
        best_move = bm;
        score = sc;
        reached = depth;
        best_pv = searcher.pv.first().cloned().unwrap_or_default();
        on_iter(&IterInfo {
            depth,
            score: sc,
            nodes: searcher.nodes,
            elapsed: start.elapsed(),
            pv: best_pv.clone(),
        });
        if bm.is_none() || sc.abs() >= MATE_THRESHOLD {
            break; // resign or settled mate: deeper search cannot change it
        }
    }

    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
        depth: reached,
        pv: best_pv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_to_int_is_order_preserving() {
        let logits = [-6.0_f32, -1.0, -0.001, 0.0, 0.001, 1.0, 6.0];
        let scores: Vec<i32> = logits.iter().map(|&l| eval_to_int(l)).collect();
        assert!(scores.windows(2).all(|w| w[0] <= w[1]), "{scores:?}");
        assert_eq!(eval_to_int(0.0), 0);
        assert!(eval_to_int(1.0) > 0 && eval_to_int(-1.0) < 0);
    }

    #[test]
    fn value_ranges_do_not_collide() {
        // Evaluations stay strictly inside the mate band, which stays inside INF.
        assert!(EVAL_CLAMP < MATE_THRESHOLD);
        assert!(MATE_THRESHOLD < VALUE_MATE);
        assert!(VALUE_MATE < VALUE_INF);
        // Even an extreme logit clamps below the mate band.
        assert!(eval_to_int(100.0) < MATE_THRESHOLD);
        assert!(eval_to_int(-100.0) > -MATE_THRESHOLD);
        // A mate-in-1 outscores any evaluation.
        assert!(VALUE_MATE - 1 > eval_to_int(100.0));
    }

    #[test]
    fn mate_score_tt_round_trip() {
        // A mate score survives store/retrieve, re-anchored to the node's ply.
        let root_relative = VALUE_MATE - 5; // mate in 5 from the root
        let ply = 2;
        let stored = score_to_tt(root_relative, ply);
        // Stored value is node-relative: mate in 3 from this node.
        assert_eq!(stored, VALUE_MATE - 3);
        // Retrieved at the same ply restores the root-relative score.
        assert_eq!(score_from_tt(stored, ply), root_relative);
        // Plain evaluations are unchanged by the correction.
        assert_eq!(score_to_tt(1234, ply), 1234);
        assert_eq!(score_from_tt(-1234, ply), -1234);
    }
}
