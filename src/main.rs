mod eval;
mod search;
mod sprt;
mod teacher;
mod usi_engine;

use shogi_core::{Color, Hand, Move, PartialPosition, Piece, PieceKind, Square, ToUsi};
use shogi_usi_parser::FromUsi;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sprt::{bounds, elo_from_score, score_ci, sprt, SprtConfig, SprtVerdict};
use teacher::gen::{self, GenConfig};
use teacher::gikou::{GikouEngine, GikouError, GikouEval, GikouOptions, GikouScore};
use teacher::hcpe::{self, HcpeEntry, HcpeGameResult, HCPE_ENTRY_SIZE};
use usi_engine::{UsiEngine, UsiOptions};
use yasai::Position;

const ENGINE_NAME: &str = "binbo-shogi";
const ENGINE_AUTHOR: &str = "sonoda";
const GIKOU_ANALYSIS_TIMEOUT: Duration = Duration::from_secs(30);
const INITIAL_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
const EXPECTED_PERFT: &[(usize, u64)] = &[
    (1, 30),
    (2, 900),
    (3, 25_470),
    (4, 719_731),
    (5, 19_861_490),
];

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();

    let result = match args.as_slice() {
        [] => run_usi().map_err(|err| err.to_string()),
        [command] if command == "usi" => run_usi().map_err(|err| err.to_string()),
        [command] if command == "verify" => {
            print_build_features();
            verify();
            Ok(())
        }
        [command, depth] if command == "perft" => parse_usize(depth)
            .map(|depth| {
                print_build_features();
                let mut pos = initial_position();
                let started = Instant::now();
                let nodes = perft(&mut pos, depth);
                print_single_result(depth, nodes, started.elapsed());
            })
            .map_err(|err| format!("invalid depth: {err}")),
        [command, depth] if command == "divide" => parse_usize(depth)
            .map(|depth| {
                let mut pos = initial_position();
                let total = divide(&mut pos, depth);
                println!("total: {total}");
            })
            .map_err(|err| format!("invalid depth: {err}")),
        [command, depth] if command == "bench" => parse_usize(depth)
            .map(|depth| {
                print_build_features();
                benchmark(depth, 5);
            })
            .map_err(|err| format!("invalid depth: {err}")),
        [command, depth, reps] if command == "bench" => {
            match (parse_usize(depth), parse_usize(reps)) {
                (Ok(depth), Ok(reps)) => {
                    print_build_features();
                    benchmark(depth, reps);
                    Ok(())
                }
                (Err(err), _) => Err(format!("invalid depth: {err}")),
                (_, Err(err)) => Err(format!("invalid reps: {err}")),
            }
        }
        [command, path] if command == "hcpe-dummy" => write_hcpe_dummy(path),
        [command, path, games] if command == "hcpe-selfplay" => {
            run_hcpe_selfplay(path, games, "512", "1", "1")
        }
        [command, path, games, max_plies] if command == "hcpe-selfplay" => {
            run_hcpe_selfplay(path, games, max_plies, "1", "1")
        }
        [command, path, games, max_plies, seed] if command == "hcpe-selfplay" => {
            run_hcpe_selfplay(path, games, max_plies, seed, "1")
        }
        [command, path, games, max_plies, seed, sample_every] if command == "hcpe-selfplay" => {
            run_hcpe_selfplay(path, games, max_plies, seed, sample_every)
        }
        [command, gikou_path, output, positions] if command == "hcpe-gikou" => {
            run_hcpe_gikou(gikou_path, output, positions, "8", "6", "512", "1", "false")
        }
        [command, gikou_path, output, positions, depth] if command == "hcpe-gikou" => {
            run_hcpe_gikou(
                gikou_path, output, positions, depth, "6", "512", "1", "false",
            )
        }
        [command, gikou_path, output, positions, depth, random_plies]
            if command == "hcpe-gikou" =>
        {
            run_hcpe_gikou(
                gikou_path,
                output,
                positions,
                depth,
                random_plies,
                "512",
                "1",
                "false",
            )
        }
        [command, gikou_path, output, positions, depth, random_plies, max_plies]
            if command == "hcpe-gikou" =>
        {
            run_hcpe_gikou(
                gikou_path,
                output,
                positions,
                depth,
                random_plies,
                max_plies,
                "1",
                "false",
            )
        }
        [command, gikou_path, output, positions, depth, random_plies, max_plies, seed]
            if command == "hcpe-gikou" =>
        {
            run_hcpe_gikou(
                gikou_path,
                output,
                positions,
                depth,
                random_plies,
                max_plies,
                seed,
                "false",
            )
        }
        [command, gikou_path, output, positions, depth, random_plies, max_plies, seed, own_book]
            if command == "hcpe-gikou" =>
        {
            run_hcpe_gikou(
                gikou_path,
                output,
                positions,
                depth,
                random_plies,
                max_plies,
                seed,
                own_book,
            )
        }
        [command, in_path, out_path] if command == "kp-from-sfen" => {
            run_kp_from_sfen(in_path, out_path, "naive")
        }
        [command, in_path, out_path, mode] if command == "kp-from-sfen" => {
            run_kp_from_sfen(in_path, out_path, mode)
        }
        [command, weights, in_path, out_path] if command == "nnue-eval" => {
            run_nnue_eval(weights, in_path, out_path)
        }
        [command, weights, sfen @ ..] if command == "nnue-bestmove" && !sfen.is_empty() => {
            run_nnue_bestmove(weights, &sfen.join(" "))
        }
        [command, rest @ ..] if command == "nnue-match" => run_nnue_match(rest),
        [command, rest @ ..] if command == "nnue-selfmatch" => run_nnue_selfmatch(rest),
        [command, rest @ ..] if command == "nnue-ttmatch" => run_nnue_ttmatch(rest),
        [command, rest @ ..] if command == "nnue-idmatch" => run_nnue_idmatch(rest),
        [command, rest @ ..] if command == "nnue-ordmatch" => run_nnue_ordmatch(rest),
        [command, rest @ ..] if command == "nnue-mvvmatch" => run_nnue_mvvmatch(rest),
        [command, rest @ ..] if command == "nnue-vs-gikou" => run_nnue_vs_gikou(rest),
        [command, rest @ ..] if command == "nnue-vs-yane" => run_nnue_vs_yane(rest),
        [command, rest @ ..] if command == "yane-vs-yane" => run_yane_vs_yane(rest),
        [command, rest @ ..] if command == "yane-vs-yane-ex" => run_yane_vs_yane_ex(rest),
        [command, rest @ ..] if command == "yane-vs-gikou" => run_yane_vs_gikou(rest),
        [command, rest @ ..] if command == "nnue-diag" => run_nnue_diag(rest),
        [command, rest @ ..] if command == "nnue-bench" => run_nnue_bench(rest),
        [command, rest @ ..] if command == "gen-teacher" => run_gen_teacher_cli(rest),
        [command, rest @ ..] if command == "gikou-recheck" => run_gikou_recheck_cli(rest),
        [command] if command == "selfplay" => run_selfplay(512, default_seed()),
        [command, max_plies] if command == "selfplay" => match parse_usize(max_plies) {
            Ok(max_plies) => run_selfplay(max_plies, default_seed()),
            Err(err) => Err(format!("invalid max plies: {err}")),
        },
        [command, max_plies, seed] if command == "selfplay" => {
            match (parse_usize(max_plies), parse_u64(seed)) {
                (Ok(max_plies), Ok(seed)) => run_selfplay(max_plies, seed),
                (Err(err), _) => Err(format!("invalid max plies: {err}")),
                (_, Err(err)) => Err(format!("invalid seed: {err}")),
            }
        }
        _ => Err(usage()),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(2);
    }
}

fn usage() -> String {
    [
        "usage:",
        "  cargo run --release                 # USI engine",
        "  cargo run --release -- usi",
        "  cargo run --release -- verify",
        "  cargo run --release -- perft <depth>",
        "  cargo run --release -- divide <depth>",
        "  cargo run --release -- bench <depth> [reps]",
        "  cargo run --release -- hcpe-dummy <path>",
        "  cargo run --release -- hcpe-selfplay <path> <games> [max_plies] [seed] [sample_every]",
        "  cargo run --release -- hcpe-gikou <gikou_path> <output> <positions> [depth] [random_plies] [max_plies] [seed] [own_book]",
        "  cargo run --release -- gen-teacher <gikou_path> <out_dir> <target> [--shard-size N] [--workers N] [--depth N] [--random-plies N] [--max-plies N] [--seed N] [--timeout-ms N] [--own-book BOOL] [--max-retries N]",
        "  cargo run --release -- gikou-recheck <gikou_path> <expected_tsv> <sample_n> [depth] [seed] [timeout_ms]",
        "  cargo run --release -- kp-from-sfen <sfen_in> <indices_out> [naive|canonical|canonical-shared|canonical-shared-ka|canonical-shared-kav2]",
        "  cargo run --release -- nnue-eval <weights_bin> <sfen_in> <logit_out>",
        "  cargo run --release -- nnue-bestmove <weights_bin> <sfen>",
        "  cargo run --release -- nnue-match <weights_bin> <games> [seed] [max_plies] [depth]",
        "  cargo run --release -- nnue-selfmatch <weights_bin> <depth_a> <depth_b> <games> [seed] [max_plies]",
        "  cargo run --release -- nnue-ttmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]",
        "  cargo run --release -- nnue-idmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]",
        "  cargo run --release -- nnue-ordmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]",
        "  cargo run --release -- nnue-mvvmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]",
        "  cargo run --release -- nnue-vs-gikou <weights_bin> <gikou_path> <level> <games> [seed] [max_plies] [opening_plies] [depth] [binbo_think_ms] [gikou_think_ms]",
        "  cargo run --release -- nnue-vs-yane <bnnq> <yane_bin> <eval_dir> [movetime_ms] [seed] [max_games] [opening_plies] [max_plies] [depth] [elo0=0] [elo1=-50] [alpha] [beta]",
        "  cargo run --release -- yane-vs-yane <yane_bin> <eval_dir_a> <eval_dir_b> [movetime_ms] [seed] [max_games] [opening_plies] [max_plies] [elo0=0] [elo1=10] [alpha] [beta]",
        "  cargo run --release -- yane-vs-yane-ex --a-eval <dir> --b-eval <dir> [--a-bin|--b-bin <yane_bin>] [--a-opt|--b-opt Name=Value]... [--a-nodes|--b-nodes N | --a-movetime|--b-movetime MS] [--seed S] [--games N] [--opening-plies 6] [--max-plies 320]",
        "  cargo run --release -- yane-vs-gikou <yane_bin> <champion_eval_dir> <gikou_path> [movetime_ms] [gikou_level] [seed] [max_games] [opening_plies] [max_plies] [gikou_think_ms] [elo0] [elo1] [alpha] [beta]",
        "  cargo run --release -- nnue-diag <weights_bin> <gikou_path> [seed] [student_ms] [gikou_depth] [opening_plies] [max_plies]",
        "  cargo run --release -- nnue-bench <weights_bin> [max_depth]",
        "  cargo run --release -- selfplay [max_plies] [seed]",
    ]
    .join("\n")
}

fn parse_usize(value: &str) -> Result<usize, std::num::ParseIntError> {
    value.parse::<usize>()
}

fn parse_u64(value: &str) -> Result<u64, std::num::ParseIntError> {
    value.parse::<u64>()
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(format!("invalid bool: {other}")),
    }
}

fn run_usi() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut engine = Engine::new();

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !engine.handle_command(line, &mut stdout)? {
            break;
        }
    }
    Ok(())
}

struct Engine {
    pos: Position,
    thinker: Box<dyn ThinkingEngine>,
    seed: u64,
    eval_file: Option<String>,
    eval_kind: EvalKind,
    depth: u16,
    log_file: Option<String>,
}

const DEFAULT_SEARCH_DEPTH: u16 = 3;
const DEFAULT_VS_GIKOU_DEPTH: u16 = 12;
/// Depth cap for the movetime-governed vs-yane match: high enough that wall-clock movetime
/// (not depth) is the binding constraint on binbo's internal search.
const DEFAULT_VS_YANE_DEPTH: u16 = 24;

impl Engine {
    fn new() -> Self {
        Self {
            pos: initial_position(),
            thinker: Box::new(RandomEngine::new(default_seed())),
            seed: default_seed(),
            eval_file: None,
            // Default to NNUE; with no EvalFile this falls back to random on
            // isready, so behavior with no options is unchanged (random).
            eval_kind: EvalKind::Nnue,
            depth: DEFAULT_SEARCH_DEPTH,
            log_file: None,
        }
    }

    /// (Re)build the thinker from the current options. Heavy work (loading the
    /// ~45 MB NNUE weights) happens here, at isready, per the USI convention.
    fn rebuild_thinker(&mut self) {
        self.thinker = match self.eval_kind {
            EvalKind::Random => Box::new(RandomEngine::new(self.seed)),
            EvalKind::Nnue => match &self.eval_file {
                Some(path) => match NnueEngine::load(path, self.depth) {
                    Ok(mut engine) => {
                        engine.emit_info = true; // stream info to the GUI over USI
                        if let Some(log_path) = &self.log_file {
                            match std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(log_path)
                            {
                                Ok(file) => {
                                    engine.log = Some(file);
                                    eprintln!("info string thinking log -> {log_path}");
                                }
                                Err(err) => {
                                    eprintln!("info string could not open LogFile {log_path}: {err}");
                                }
                            }
                        }
                        eprintln!(
                            "info string loaded NNUE eval from {path} (depth {})",
                            self.depth
                        );
                        Box::new(engine)
                    }
                    Err(err) => {
                        eprintln!("info string NNUE load failed ({err}); using random");
                        Box::new(RandomEngine::new(self.seed))
                    }
                },
                None => {
                    eprintln!("info string no EvalFile set; using random");
                    Box::new(RandomEngine::new(self.seed))
                }
            },
        };
    }

    fn handle_command<W: Write>(&mut self, line: &str, out: &mut W) -> io::Result<bool> {
        let command = line.split_whitespace().next().unwrap_or_default();
        match command {
            "usi" => {
                write_usi_line(out, &format!("id name {ENGINE_NAME}"))?;
                write_usi_line(out, &format!("id author {ENGINE_AUTHOR}"))?;
                write_usi_line(
                    out,
                    "option name Seed type spin default 0 min 0 max 2147483647",
                )?;
                write_usi_line(out, "option name Eval type combo default nnue var nnue var random")?;
                write_usi_line(out, "option name EvalFile type string default <empty>")?;
                write_usi_line(out, "option name Depth type spin default 3 min 1 max 12")?;
                write_usi_line(out, "option name LogFile type string default <empty>")?;
                write_usi_line(out, "usiok")?;
            }
            "isready" => {
                self.rebuild_thinker();
                write_usi_line(out, "readyok")?;
            }
            "setoption" => {
                if let Err(err) = self.set_option(line) {
                    eprintln!("info string ignored setoption: {err}");
                }
            }
            "usinewgame" => {
                self.pos = initial_position();
                self.thinker.new_game();
            }
            "position" => match parse_position(line) {
                Ok(pos) => self.pos = pos,
                Err(err) => eprintln!("info string failed to parse position: {err}"),
            },
            "go" => {
                let limits = GoLimits::from_go_command(line);
                let result = self.thinker.think(&self.pos, &limits);
                write_usi_line(
                    out,
                    &format!("bestmove {}", result.bestmove.to_usi_string()),
                )?;
            }
            "d" => {
                debug_position(&self.pos)?;
            }
            "stop" => {
                self.thinker.stop();
            }
            "ponderhit" | "gameover" | "debug" => {}
            "quit" => return Ok(false),
            _ => {
                eprintln!("info string ignored command: {line}");
            }
        }
        Ok(true)
    }

    fn set_option(&mut self, line: &str) -> Result<(), String> {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let Some(name_pos) = tokens.iter().position(|token| *token == "name") else {
            return Err("missing name".to_string());
        };
        let value_pos = tokens.iter().position(|token| *token == "value");
        let name_end = value_pos.unwrap_or(tokens.len());
        let name = tokens[name_pos + 1..name_end].join(" ");
        // The value can contain spaces (e.g. a file path), so join the rest.
        let value = value_pos.map(|pos| tokens[pos + 1..].join(" "));

        // Options take effect at the next isready (rebuild_thinker).
        match name.as_str() {
            "Seed" => {
                let value = value.ok_or_else(|| "Seed value is missing".to_string())?;
                let seed = value
                    .parse::<u64>()
                    .map_err(|err| format!("invalid Seed value: {err}"))?;
                self.seed = if seed == 0 { default_seed() } else { seed };
            }
            "EvalFile" => {
                let value = value.unwrap_or_default();
                let trimmed = value.trim();
                self.eval_file = if trimmed.is_empty() || trimmed == "<empty>" {
                    None
                } else {
                    Some(trimmed.to_string())
                };
            }
            "Eval" => {
                let value = value.unwrap_or_default();
                self.eval_kind = match value.trim() {
                    "random" => EvalKind::Random,
                    "nnue" => EvalKind::Nnue,
                    other => return Err(format!("unknown Eval value: {other}")),
                };
            }
            "Depth" => {
                let value = value.ok_or_else(|| "Depth value is missing".to_string())?;
                let depth = value
                    .trim()
                    .parse::<u16>()
                    .map_err(|err| format!("invalid Depth value: {err}"))?;
                self.depth = depth.clamp(1, 12);
            }
            "LogFile" => {
                let value = value.unwrap_or_default();
                let trimmed = value.trim();
                self.log_file = if trimmed.is_empty() || trimmed == "<empty>" {
                    None
                } else {
                    Some(trimmed.to_string())
                };
            }
            _ => {}
        }
        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct GoLimits {
    btime: Option<u64>,
    wtime: Option<u64>,
    binc: Option<u64>,
    winc: Option<u64>,
    byoyomi: Option<u64>,
    movetime: Option<u64>,
    depth: Option<usize>,
    nodes: Option<u64>,
    infinite: bool,
    searchmoves: Vec<String>,
}

impl GoLimits {
    fn from_go_command(line: &str) -> Self {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let mut limits = Self::default();
        let mut index = 1;

        while index < tokens.len() {
            match tokens[index] {
                "btime" => {
                    limits.btime = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "wtime" => {
                    limits.wtime = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "binc" => {
                    limits.binc = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "winc" => {
                    limits.winc = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "byoyomi" => {
                    limits.byoyomi = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "movetime" => {
                    limits.movetime = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "depth" => {
                    limits.depth = tokens.get(index + 1).and_then(|value| value.parse().ok());
                    index += 2;
                }
                "nodes" => {
                    limits.nodes = parse_next_u64(&tokens, index);
                    index += 2;
                }
                "infinite" => {
                    limits.infinite = true;
                    index += 1;
                }
                "searchmoves" => {
                    index += 1;
                    while index < tokens.len() && !is_go_limit_keyword(tokens[index]) {
                        limits.searchmoves.push(tokens[index].to_string());
                        index += 1;
                    }
                }
                _ => index += 1,
            }
        }

        limits
    }
}

fn parse_next_u64(tokens: &[&str], index: usize) -> Option<u64> {
    tokens.get(index + 1).and_then(|value| value.parse().ok())
}

fn is_go_limit_keyword(token: &str) -> bool {
    matches!(
        token,
        "btime"
            | "wtime"
            | "binc"
            | "winc"
            | "byoyomi"
            | "movetime"
            | "depth"
            | "nodes"
            | "infinite"
            | "searchmoves"
    )
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ThinkResult {
    bestmove: BestMove,
    ponder: Option<Move>,
    score: Option<i32>,
    depth: Option<usize>,
    nodes: Option<u64>,
    pv: Vec<Move>,
}

impl ThinkResult {
    fn bestmove(bestmove: BestMove) -> Self {
        Self {
            bestmove,
            ponder: None,
            score: None,
            depth: None,
            nodes: None,
            pv: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BestMove {
    Move(Move),
    Resign,
}

impl BestMove {
    fn to_usi_string(self) -> String {
        match self {
            BestMove::Move(mv) => mv.to_usi_owned(),
            BestMove::Resign => "resign".to_string(),
        }
    }
}

trait ThinkingEngine {
    fn new_game(&mut self);
    fn think(&mut self, pos: &Position, limits: &GoLimits) -> ThinkResult;
    fn stop(&mut self) {}
}

struct RandomEngine {
    rng: SplitMix64,
}

impl RandomEngine {
    fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }
}

impl ThinkingEngine for RandomEngine {
    fn new_game(&mut self) {}

    fn think(&mut self, pos: &Position, _limits: &GoLimits) -> ThinkResult {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            ThinkResult::bestmove(BestMove::Resign)
        } else {
            ThinkResult::bestmove(BestMove::Move(moves[self.rng.gen_index(moves.len())]))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EvalKind {
    Random,
    Nnue,
}

/// Thinking engine that picks moves with the trained NNUE evaluator via a
/// fixed-depth negamax + alpha-beta search (`src/search.rs`). NNUE evaluates the
/// leaves; negamax keeps the best line from our own viewpoint. Search depth is
/// set by the `Depth` USI option. A wrong sign would make it choose the worst
/// move, so it is verified explicitly (stage B / `nnue-match` / `nnue-selfmatch`).
struct NnueEngine {
    net: eval::nnue::Nnue,
    depth: u16,
    use_tt: bool,
    use_iddfs: bool,
    use_ordering: bool,
    use_mvv_lva: bool,
    /// Emit USI `info` lines per completed depth (true only for the USI engine).
    emit_info: bool,
    /// Optional thinking-log file: per-move depth/score/nodes/time, for review.
    log: Option<File>,
    move_count: u32,
    tt: search::Tt,
}

impl NnueEngine {
    fn load(path: &str, depth: u16) -> Result<Self, String> {
        Ok(Self {
            net: eval::nnue::Nnue::load(path)?,
            depth: depth.max(1),
            use_tt: true,
            use_iddfs: true,
            use_ordering: true,
            use_mvv_lva: true,
            emit_info: false,
            log: None,
            move_count: 0,
            tt: search::Tt::new(),
        })
    }

    fn opts(&self) -> search::Opts {
        search::Opts {
            use_tt: self.use_tt,
            use_ordering: self.use_ordering,
            use_mvv_lva: self.use_mvv_lva,
            ..search::Opts::full()
        }
    }
}

impl ThinkingEngine for NnueEngine {
    fn new_game(&mut self) {
        self.move_count = 0;
        self.tt.clear();
        if let Some(file) = self.log.as_mut() {
            let _ = writeln!(file, "\n# === usinewgame ===");
        }
    }

    fn think(&mut self, pos: &Position, limits: &GoLimits) -> ThinkResult {
        let opts = self.opts();
        let emit = self.emit_info;
        let want_log = self.log.is_some();
        // Lines collected here are written to the log file after the search, so
        // the file borrow is not held across the (borrowing) search call.
        let mut log_lines: Vec<String> = Vec::new();

        let result = if self.use_iddfs {
            // Depth is the safety upper bound; time (if any) cuts off earlier.
            let budget = budget_from_limits(limits, pos.side_to_move());
            let net = &self.net;
            let tt = &mut self.tt;
            let mut on_iter = |info: &search::IterInfo| {
                let mut line = format!(
                    "info depth {} score {} nodes {} time {}",
                    info.depth,
                    search::usi_score(info.score),
                    info.nodes,
                    info.elapsed.as_millis()
                );
                if !info.pv.is_empty() {
                    line.push_str(" pv ");
                    line.push_str(&format_pv(&info.pv));
                }
                if emit {
                    // Flush each line so the GUI shows progress live (pipes are
                    // block-buffered, so println! alone would not flush).
                    let mut out = io::stdout();
                    let _ = writeln!(out, "{line}");
                    let _ = out.flush();
                }
                if want_log {
                    log_lines.push(line);
                }
            };
            search::search_iddfs_timed(net, pos, self.depth, opts, tt, budget, &mut on_iter)
        } else {
            search::search_root(&self.net, pos, self.depth, opts, &mut self.tt)
        };

        let best = match result.best_move {
            Some(mv) => BestMove::Move(mv),
            None => BestMove::Resign,
        };

        self.move_count += 1;
        if let Some(file) = self.log.as_mut() {
            let _ = writeln!(
                file,
                "# move {} side {} sfen {}",
                self.move_count,
                pos.side_to_move().to_usi_owned(),
                sfen_core(&position_to_sfen(pos))
            );
            for line in &log_lines {
                let _ = writeln!(file, "{line}");
            }
            let _ = writeln!(file, "bestmove {}", best.to_usi_string());
            let _ = file.flush();
        }

        ThinkResult {
            bestmove: best,
            ponder: result.pv.get(1).copied(),
            score: Some(result.score),
            depth: Some(result.depth as usize),
            nodes: Some(result.nodes),
            pv: result.pv,
        }
    }
}

/// Format a principal variation as space-separated USI moves (for `info ... pv`).
fn format_pv(pv: &[Move]) -> String {
    pv.iter()
        .map(|mv| mv.to_usi_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute the time budget for this move from the `go` limits, or `None` for no
/// limit (infinite, or no time info at all -> fall back to the Depth cap).
/// Deliberately simple: byoyomi + a coarse slice of main time + increment, minus
/// a safety margin so bestmove is sent before the clock runs out.
fn budget_from_limits(limits: &GoLimits, side: Color) -> Option<Duration> {
    const MARGIN_MS: u64 = 500;
    const MOVES_LEFT: u64 = 30;
    const MIN_BUDGET_MS: u64 = 10;

    if limits.infinite {
        return None;
    }
    if let Some(movetime) = limits.movetime {
        let ms = movetime.saturating_sub(MARGIN_MS).max(MIN_BUDGET_MS);
        return Some(Duration::from_millis(ms));
    }
    let (main_time, inc) = match side {
        Color::Black => (limits.btime.unwrap_or(0), limits.binc.unwrap_or(0)),
        Color::White => (limits.wtime.unwrap_or(0), limits.winc.unwrap_or(0)),
    };
    let byoyomi = limits.byoyomi.unwrap_or(0);
    if main_time == 0 && byoyomi == 0 && inc == 0 {
        // Plain `go` (CLI/self-play/smoke): no clock -> use the Depth cap.
        return None;
    }
    let raw = main_time / MOVES_LEFT + inc + byoyomi;
    let ms = raw.saturating_sub(MARGIN_MS).max(MIN_BUDGET_MS);
    Some(Duration::from_millis(ms))
}

fn write_usi_line<W: Write>(out: &mut W, line: &str) -> io::Result<()> {
    writeln!(out, "{line}")?;
    out.flush()
}

fn parse_position(line: &str) -> Result<Position, String> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.first() != Some(&"position") {
        return Err("not a position command".to_string());
    }

    let Some(kind) = tokens.get(1) else {
        return Err("missing position kind".to_string());
    };

    let (mut pos, moves) = match *kind {
        "startpos" => {
            let moves = match tokens.get(2) {
                None => &[][..],
                Some(&"moves") => &tokens[3..],
                Some(other) => return Err(format!("unexpected token after startpos: {other}")),
            };
            (initial_position(), moves)
        }
        "sfen" => {
            let moves_pos = tokens.iter().position(|token| *token == "moves");
            let sfen_end = moves_pos.unwrap_or(tokens.len());
            if sfen_end <= 2 {
                return Err("missing SFEN fields".to_string());
            }
            let sfen = tokens[2..sfen_end].join(" ");
            let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
                .map_err(|err| format!("invalid SFEN: {err:?}"))?;
            let moves = moves_pos
                .map(|index| &tokens[index + 1..])
                .unwrap_or(&[][..]);
            (Position::new(partial), moves)
        }
        other => return Err(format!("unsupported position kind: {other}")),
    };

    for move_usi in moves {
        apply_usi_move(&mut pos, move_usi)?;
    }
    Ok(pos)
}

fn apply_usi_move(pos: &mut Position, move_usi: &str) -> Result<(), String> {
    let legal_move = pos
        .legal_moves()
        .into_iter()
        .find(|mv| mv.to_usi_owned() == move_usi)
        .ok_or_else(|| format!("illegal move for current position: {move_usi}"))?;
    pos.do_move(legal_move);
    Ok(())
}

fn write_hcpe_dummy(path: &str) -> Result<(), String> {
    struct DummyCase {
        name: &'static str,
        sfen: &'static str,
        eval: i16,
        best_move: &'static str,
        result: HcpeGameResult,
    }

    let cases = [
        DummyCase {
            name: "startpos",
            sfen: INITIAL_SFEN,
            eval: 123,
            best_move: "7g7f",
            result: HcpeGameResult::BlackWin,
        },
        DummyCase {
            name: "midgame",
            sfen: "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
            eval: -456,
            best_move: "2g2f",
            result: HcpeGameResult::WhiteWin,
        },
        DummyCase {
            name: "black_hand_drop",
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPP1/1B5R1/LNSGKGSNL b P 42",
            eval: 0,
            best_move: "P*1g",
            result: HcpeGameResult::Draw,
        },
        DummyCase {
            name: "promoted_piece",
            sfen: "lnsgkgsnl/1r5b1/ppppppp+Pp/9/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL b - 1",
            eval: 789,
            best_move: "8h2b+",
            result: HcpeGameResult::BlackWin,
        },
    ];

    let hcpe_path = Path::new(path);
    if let Some(parent) = hcpe_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| format!("failed to create parent dir: {err}"))?;
    }

    let mut entries = Vec::with_capacity(cases.len());
    let mut expected = String::from(
        "index\tname\tsfen_core\teval\tbestmove_usi\tbestMove16\tgameResult\tdummy\thcp_hex\n",
    );

    for (index, case) in cases.iter().enumerate() {
        let pos = parse_position(&format!("position sfen {}", case.sfen))?;
        let best_move = parse_raw_usi_move(case.best_move)?;
        let entry = HcpeEntry::from_position(&pos, case.eval, best_move, case.result)
            .map_err(|err| format!("failed to encode hcpe entry {}: {err}", case.name))?;
        expected.push_str(&format!(
            "{index}\t{}\t{}\t{}\t{}\t{}\t{}\t0\t{}\n",
            case.name,
            sfen_core(case.sfen),
            case.eval,
            case.best_move,
            entry.best_move16,
            case.result as i8,
            hex(&entry.hcp)
        ));
        entries.push(entry);
    }

    let mut file = File::create(hcpe_path)
        .map_err(|err| format!("failed to create {}: {err}", hcpe_path.display()))?;
    hcpe::write_entries(&mut file, &entries)
        .map_err(|err| format!("failed to write hcpe entries: {err}"))?;
    file.flush()
        .map_err(|err| format!("failed to flush {}: {err}", hcpe_path.display()))?;

    let expected_path = expected_path_for(hcpe_path);
    fs::write(&expected_path, expected)
        .map_err(|err| format!("failed to write {}: {err}", expected_path.display()))?;

    let file_size = fs::metadata(hcpe_path)
        .map_err(|err| format!("failed to stat {}: {err}", hcpe_path.display()))?
        .len();
    let expected_size = (entries.len() * HCPE_ENTRY_SIZE) as u64;
    if file_size != expected_size {
        return Err(format!(
            "hcpe file size mismatch: got {file_size}, expected {expected_size}"
        ));
    }

    println!("hcpe_path={}", hcpe_path.display());
    println!("expected_path={}", expected_path.display());
    println!("entries={}", entries.len());
    println!("hcpe_entry_size={HCPE_ENTRY_SIZE}");
    println!("file_size={file_size}");
    println!("status=ok");
    Ok(())
}

fn run_hcpe_selfplay(
    path: &str,
    games: &str,
    max_plies: &str,
    seed: &str,
    sample_every: &str,
) -> Result<(), String> {
    let games = parse_usize(games).map_err(|err| format!("invalid games: {err}"))?;
    let max_plies = parse_usize(max_plies).map_err(|err| format!("invalid max plies: {err}"))?;
    let base_seed = parse_u64(seed).map_err(|err| format!("invalid seed: {err}"))?;
    let sample_every =
        parse_usize(sample_every).map_err(|err| format!("invalid sample_every: {err}"))?;
    if sample_every == 0 {
        return Err("sample_every must be >= 1".to_string());
    }

    let hcpe_path = Path::new(path);
    if let Some(parent) = hcpe_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| format!("failed to create parent dir: {err}"))?;
    }

    let file = File::create(hcpe_path)
        .map_err(|err| format!("failed to create {}: {err}", hcpe_path.display()))?;
    let mut writer = BufWriter::new(file);

    let mut positions: u64 = 0;
    let mut terminal_games: u64 = 0;
    let mut max_game_plies: usize = 0;

    // Each game uses an independent SplitMix64 stream so the games diverge after the
    // first move (30 legal opening moves), giving a wide spread of positions.
    for game in 0..games {
        let mut rng = SplitMix64::new(base_seed.wrapping_add(game as u64).wrapping_add(1));
        let mut pos = initial_position();
        let mut reached = 0;
        for ply_index in 0..max_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                terminal_games += 1;
                break;
            }
            let mv = moves[rng.gen_index(moves.len())];

            // Skip the literal initial position (ply_index == 0): it is identical across
            // games and already covered by Task 012. Sample every `sample_every` plies after.
            if ply_index > 0 && ply_index % sample_every == 0 {
                let entry = HcpeEntry::from_position(&pos, 0, mv, HcpeGameResult::Draw)
                    .map_err(|err| format!("failed to encode hcpe entry: {err}"))?;
                writer
                    .write_all(&entry.encode())
                    .map_err(|err| format!("failed to write hcpe entry: {err}"))?;
                positions += 1;
            }

            pos.do_move(mv);
            reached = ply_index + 1;
        }
        max_game_plies = max_game_plies.max(reached);
    }

    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", hcpe_path.display()))?;

    let file_size = fs::metadata(hcpe_path)
        .map_err(|err| format!("failed to stat {}: {err}", hcpe_path.display()))?
        .len();
    let expected_size = positions * HCPE_ENTRY_SIZE as u64;
    if file_size != expected_size {
        return Err(format!(
            "hcpe file size mismatch: got {file_size}, expected {expected_size}"
        ));
    }

    println!("hcpe_path={}", hcpe_path.display());
    println!("games={games}");
    println!("max_plies={max_plies}");
    println!("seed={base_seed}");
    println!("sample_every={sample_every}");
    println!("positions={positions}");
    println!("terminal_games={terminal_games}");
    println!("max_game_plies={max_game_plies}");
    println!("hcpe_entry_size={HCPE_ENTRY_SIZE}");
    println!("file_size={file_size}");
    println!("status=ok");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Read SFENs (one per line) and write each position's KP feature indices
/// (space-separated, sorted) to the output file, one line per position. Used to
/// cross-check the Rust KP numbering against the Python `trainer/kp.py`.
fn run_kp_from_sfen(in_path: &str, out_path: &str, mode: &str) -> Result<(), String> {
    use std::io::BufReader;

    let input = File::open(in_path).map_err(|err| format!("failed to open {in_path}: {err}"))?;
    let reader = BufReader::new(input);
    let output = File::create(out_path).map_err(|err| format!("failed to create {out_path}: {err}"))?;
    let mut writer = BufWriter::new(output);

    let mut count = 0_usize;
    let mut line_buf = String::new();
    for line in reader.lines() {
        let line = line.map_err(|err| format!("read error: {err}"))?;
        let sfen = line.trim();
        if sfen.is_empty() {
            continue;
        }
        let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
            .map_err(|err| format!("invalid SFEN '{sfen}': {err:?}"))?;
        let pos = Position::new(partial);
        line_buf.clear();
        if mode == "canonical-shared" || mode == "canonical-shared-ka"
            || mode == "canonical-shared-kav2" || mode == "kpp-zone" {
            // Output "own_indices | opp_indices" (the two shared-band halves).
            let (own, opp) = match mode {
                "canonical-shared-ka" => eval::kp::kp_indices_canonical_shared_ka(&pos),
                "canonical-shared-kav2" => eval::kp::kp_indices_canonical_shared_kav2(&pos),
                "kpp-zone" => eval::kp::kpp_zone_halves(&pos),
                _ => eval::kp::kp_indices_canonical_shared(&pos),
            };
            for (i, index) in own.iter().enumerate() {
                if i > 0 {
                    line_buf.push(' ');
                }
                line_buf.push_str(&index.to_string());
            }
            line_buf.push_str(" |");
            for index in &opp {
                line_buf.push(' ');
                line_buf.push_str(&index.to_string());
            }
            line_buf.push('\n');
            writer
                .write_all(line_buf.as_bytes())
                .map_err(|err| format!("write error: {err}"))?;
            count += 1;
            continue;
        }
        let indices = if mode == "canonical" {
            eval::kp::kp_indices_canonical(&pos)
        } else {
            eval::kp::kp_indices(&pos)
        };
        for (i, index) in indices.iter().enumerate() {
            if i > 0 {
                line_buf.push(' ');
            }
            line_buf.push_str(&index.to_string());
        }
        line_buf.push('\n');
        writer
            .write_all(line_buf.as_bytes())
            .map_err(|err| format!("write error: {err}"))?;
        count += 1;
    }
    writer.flush().map_err(|err| format!("flush error: {err}"))?;
    eprintln!("kp_from_sfen positions={count} out={out_path}");
    Ok(())
}

/// Read SFENs (one per line) and write the NNUE logit for each, one per line.
/// Cross-checked against Python in `trainer/verify_nnue_rust.py`.
fn run_nnue_eval(weights_path: &str, in_path: &str, out_path: &str) -> Result<(), String> {
    use std::io::BufReader;

    let net = eval::nnue::Nnue::load(weights_path)?;
    let input = File::open(in_path).map_err(|err| format!("failed to open {in_path}: {err}"))?;
    let reader = BufReader::new(input);
    let output =
        File::create(out_path).map_err(|err| format!("failed to create {out_path}: {err}"))?;
    let mut writer = BufWriter::new(output);

    let mut count = 0_usize;
    for line in reader.lines() {
        let line = line.map_err(|err| format!("read error: {err}"))?;
        let sfen = line.trim();
        if sfen.is_empty() {
            continue;
        }
        let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
            .map_err(|err| format!("invalid SFEN '{sfen}': {err:?}"))?;
        let pos = Position::new(partial);
        let logit = net.forward(&pos);
        writeln!(writer, "{logit}").map_err(|err| format!("write error: {err}"))?;
        count += 1;
    }
    writer.flush().map_err(|err| format!("flush error: {err}"))?;
    eprintln!("nnue_eval positions={count} out={out_path}");
    Ok(())
}

/// Print every legal move scored from the side-to-move's viewpoint
/// (`-NNUE(child)`), sorted best-first, plus the chosen bestmove. Used to verify
/// the evaluation viewpoint/sign on hand-picked positions (stage B).
fn run_nnue_bestmove(weights_path: &str, sfen: &str) -> Result<(), String> {
    let net = eval::nnue::Nnue::load(weights_path)?;
    let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
        .map_err(|err| format!("invalid SFEN '{sfen}': {err:?}"))?;
    let pos = Position::new(partial);

    let moves = pos.legal_moves();
    println!("side_to_move={}", pos.side_to_move().to_usi_owned());
    println!("legal_moves={}", moves.len());
    if moves.is_empty() {
        println!("bestmove resign");
        return Ok(());
    }

    let mut work = pos.clone();
    let mut scored: Vec<(f32, Move)> = Vec::with_capacity(moves.len());
    for mv in moves {
        work.do_move(mv);
        let our_score = if work.legal_moves().is_empty() {
            f32::INFINITY
        } else {
            -net.forward(&work)
        };
        work.undo_move(mv);
        scored.push((our_score, mv));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let show = scored.len().min(10);
    println!("top {show} moves (our-view score, higher=better for side to move):");
    for (rank, (score, mv)) in scored.iter().take(show).enumerate() {
        println!("  {:>2}. {:<7} {:+.4}", rank + 1, mv.to_usi_owned(), score);
    }
    if scored.len() > show {
        let (worst_score, worst_mv) = scored.last().unwrap();
        println!("  worst: {:<7} {:+.4}", worst_mv.to_usi_owned(), worst_score);
    }
    println!("bestmove {}", scored[0].1.to_usi_owned());
    Ok(())
}

/// Play `games` games of NNUE vs RandomEngine (alternating colors) and report
/// the win/loss/draw tally (stage C: evidence the learned eval beats random).
fn run_nnue_match(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-match <weights_bin> <games> [seed] [max_plies] [depth]".to_string()
    })?;
    let games: u64 = args
        .get(1)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(2) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let depth: u16 = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid depth: {err}"))?,
        None => DEFAULT_SEARCH_DEPTH,
    };

    let mut nnue = NnueEngine::load(weights, depth)?;
    let limits = GoLimits::default();
    let (mut nnue_wins, mut nnue_losses, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        let nnue_is_black = game % 2 == 0;
        let mut pos = initial_position();
        let mut random = RandomEngine::new(seed.wrapping_add(game));
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let nnue_to_move = (pos.side_to_move() == Color::Black) == nnue_is_black;
            let bestmove = if nnue_to_move {
                nnue.think(&pos, &limits).bestmove
            } else {
                random.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    // Side to move has no legal move: it loses.
                    outcome = if nnue_to_move { "loss" } else { "win" };
                    break;
                }
            }
        }

        match outcome {
            "win" => nnue_wins += 1,
            "loss" => nnue_losses += 1,
            _ => draws += 1,
        }
    }

    let decisive = nnue_wins + nnue_losses;
    let win_rate = if games > 0 {
        100.0 * nnue_wins as f64 / games as f64
    } else {
        0.0
    };
    let decisive_rate = if decisive > 0 {
        100.0 * nnue_wins as f64 / decisive as f64
    } else {
        0.0
    };
    println!("nnue-match weights={weights} games={games} seed={seed} max_plies={max_plies} depth={depth}");
    println!("nnue_wins={nnue_wins} nnue_losses={nnue_losses} draws={draws}");
    println!("nnue_win_rate={win_rate:.1}% (of all games)");
    println!("nnue_win_rate_decisive={decisive_rate:.1}% (of {decisive} decided games)");
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VsGikouOutcome {
    BinboWin,
    BinboLoss,
    Draw,
    Error,
}

impl VsGikouOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::BinboWin => "win",
            Self::BinboLoss => "loss",
            Self::Draw => "draw",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Default)]
struct SearchSideStats {
    moves: u64,
    depth_sum: u64,
    depth_samples: u64,
    max_depth: usize,
    nodes_sum: u128,
    nodes_samples: u64,
    max_nodes: u64,
    time_ms_sum: f64,
}

impl SearchSideStats {
    fn record(&mut self, depth: Option<usize>, nodes: Option<u64>, elapsed: Duration) {
        self.moves += 1;
        self.time_ms_sum += elapsed.as_secs_f64() * 1000.0;
        if let Some(depth) = depth {
            self.depth_sum += depth as u64;
            self.depth_samples += 1;
            self.max_depth = self.max_depth.max(depth);
        }
        if let Some(nodes) = nodes {
            self.nodes_sum += nodes as u128;
            self.nodes_samples += 1;
            self.max_nodes = self.max_nodes.max(nodes);
        }
    }

    fn merge(&mut self, other: &Self) {
        self.moves += other.moves;
        self.depth_sum += other.depth_sum;
        self.depth_samples += other.depth_samples;
        self.max_depth = self.max_depth.max(other.max_depth);
        self.nodes_sum += other.nodes_sum;
        self.nodes_samples += other.nodes_samples;
        self.max_nodes = self.max_nodes.max(other.max_nodes);
        self.time_ms_sum += other.time_ms_sum;
    }

    fn avg_depth(&self) -> Option<f64> {
        (self.depth_samples > 0).then(|| self.depth_sum as f64 / self.depth_samples as f64)
    }

    fn avg_nodes(&self) -> Option<f64> {
        (self.nodes_samples > 0).then(|| self.nodes_sum as f64 / self.nodes_samples as f64)
    }

    fn avg_time_ms(&self) -> Option<f64> {
        (self.moves > 0).then(|| self.time_ms_sum / self.moves as f64)
    }
}

struct VsGikouGameResult {
    outcome: VsGikouOutcome,
    reason: String,
    plies: usize,
    opening: Vec<String>,
    elapsed: Duration,
    binbo_stats: SearchSideStats,
    gikou_stats: SearchSideStats,
}

fn run_nnue_vs_gikou(args: &[String]) -> Result<(), String> {
    let usage = "usage: nnue-vs-gikou <weights_bin> <gikou_path> <level> <games> [seed] [max_plies] [opening_plies] [depth] [binbo_think_ms] [gikou_think_ms]";
    let weights = args.first().ok_or_else(|| usage.to_string())?;
    let gikou_path = args.get(1).ok_or_else(|| usage.to_string())?;
    let level: usize = args.get(2)
        .ok_or_else(|| "missing <level>".to_string())?
        .parse()
        .map_err(|err| format!("invalid level: {err}"))?;
    if !(1..=100).contains(&level) {
        return Err("level must be in 1..=100".to_string());
    }
    let games: u64 = args.get(3)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260622,
    };
    let max_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(6) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 6,
    };
    let depth: u16 = match args.get(7) {
        Some(value) => value.parse().map_err(|err| format!("invalid depth: {err}"))?,
        None => DEFAULT_VS_GIKOU_DEPTH,
    };
    let binbo_think_ms: u64 = match args.get(8) {
        Some(value) => value.parse().map_err(|err| format!("invalid binbo_think_ms: {err}"))?,
        None => 1000,
    };
    let gikou_think_ms: u64 = match args.get(9) {
        Some(value) => value.parse().map_err(|err| format!("invalid gikou_think_ms: {err}"))?,
        None => binbo_think_ms,
    };

    let mut nnue = NnueEngine::load(weights, depth)?;
    let limits = GoLimits {
        movetime: Some(binbo_think_ms.saturating_add(500)),
        ..GoLimits::default()
    };

    let started = Instant::now();
    let mut wins = 0_u64;
    let mut losses = 0_u64;
    let mut draws = 0_u64;
    let mut errors = 0_u64;
    let mut total_plies = 0_usize;
    let mut binbo_stats = SearchSideStats::default();
    let mut gikou_stats = SearchSideStats::default();

    println!("nnue-vs-gikou weights={weights} gikou_path={gikou_path} level={level} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies} depth={depth} binbo_think_ms={binbo_think_ms} gikou_think_ms={gikou_think_ms}");
    println!("gikou_options=OwnBook:false,Threads:1,DepthLimit:{level},MinThinkingTime:10");
    println!("time_control=binbo_movetime_ms:{} gikou_go='go btime 0 wtime 0 byoyomi {gikou_think_ms}'", binbo_think_ms.saturating_add(500));
    println!("game\tbinbo_color\topening_seed\tresult\treason\tplies\telapsed_ms\tbinbo_moves\tbinbo_avg_depth\tbinbo_avg_nodes\tbinbo_avg_ms\tgikou_moves\tgikou_avg_depth\tgikou_avg_nodes\tgikou_avg_ms\topening");

    for game in 0..games {
        nnue.new_game();
        let mut gikou = start_gikou_match_engine(gikou_path, level)
            .map_err(|err| format!("failed to start Gikou: {err}"))?;
        let binbo_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = play_nnue_vs_gikou_game(
            &mut nnue,
            &mut gikou,
            &limits,
            binbo_is_black,
            opening_seed,
            opening_plies,
            max_plies,
            gikou_think_ms,
        );

        match result.outcome {
            VsGikouOutcome::BinboWin => wins += 1,
            VsGikouOutcome::BinboLoss => losses += 1,
            VsGikouOutcome::Draw => draws += 1,
            VsGikouOutcome::Error => errors += 1,
        }
        total_plies += result.plies;
        binbo_stats.merge(&result.binbo_stats);
        gikou_stats.merge(&result.gikou_stats);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            game + 1,
            if binbo_is_black { "black" } else { "white" },
            opening_seed,
            result.outcome.as_str(),
            result.reason,
            result.plies,
            result.elapsed.as_secs_f64() * 1000.0,
            result.binbo_stats.moves,
            fmt_avg(result.binbo_stats.avg_depth()),
            fmt_avg(result.binbo_stats.avg_nodes()),
            fmt_avg(result.binbo_stats.avg_time_ms()),
            result.gikou_stats.moves,
            fmt_avg(result.gikou_stats.avg_depth()),
            fmt_avg(result.gikou_stats.avg_nodes()),
            fmt_avg(result.gikou_stats.avg_time_ms()),
            result.opening.join(",")
        );
    }

    let scored_games = games.saturating_sub(errors);
    let score = wins as f64 + 0.5 * draws as f64;
    let score_rate = if scored_games > 0 {
        100.0 * score / scored_games as f64
    } else {
        0.0
    };
    let decisive = wins + losses;
    let decisive_rate = if decisive > 0 {
        100.0 * wins as f64 / decisive as f64
    } else {
        0.0
    };
    let avg_plies = if games > 0 {
        total_plies as f64 / games as f64
    } else {
        0.0
    };
    let elapsed = started.elapsed();

    println!("summary_level={level}");
    println!("binbo_record={wins}-{losses}-{draws} errors={errors}");
    println!("binbo_score_rate={score_rate:.1}% (wins + 0.5*draws over {scored_games} scored games)");
    println!("binbo_win_rate_decisive={decisive_rate:.1}% (of {decisive} decided games)");
    println!("avg_plies={avg_plies:.1}");
    println!("binbo_moves={}", binbo_stats.moves);
    println!("binbo_avg_depth={}", fmt_avg(binbo_stats.avg_depth()));
    println!("binbo_max_depth={}", binbo_stats.max_depth);
    println!("binbo_avg_nodes={}", fmt_avg(binbo_stats.avg_nodes()));
    println!("binbo_max_nodes={}", binbo_stats.max_nodes);
    println!("binbo_avg_time_ms={}", fmt_avg(binbo_stats.avg_time_ms()));
    println!("gikou_moves={}", gikou_stats.moves);
    println!("gikou_avg_depth={}", fmt_avg(gikou_stats.avg_depth()));
    println!("gikou_max_depth={}", gikou_stats.max_depth);
    println!("gikou_avg_nodes={}", fmt_avg(gikou_stats.avg_nodes()));
    println!("gikou_max_nodes={}", gikou_stats.max_nodes);
    println!("gikou_avg_time_ms={}", fmt_avg(gikou_stats.avg_time_ms()));
    println!("elapsed_ms={:.1}", elapsed.as_secs_f64() * 1000.0);
    println!("status={}", if errors == 0 { "ok" } else { "incomplete" });
    Ok(())
}

fn start_gikou_match_engine(gikou_path: &str, level: usize) -> Result<GikouEngine, GikouError> {
    let mut options = GikouOptions::new(Path::new(gikou_path), false, 1);
    options.depth_limit = Some(level);
    options.min_thinking_time_ms = Some(10);
    GikouEngine::with_options(options)
}

fn play_nnue_vs_gikou_game(
    nnue: &mut NnueEngine,
    gikou: &mut GikouEngine,
    limits: &GoLimits,
    binbo_is_black: bool,
    opening_seed: u64,
    opening_plies: usize,
    max_plies: usize,
    gikou_think_ms: u64,
) -> VsGikouGameResult {
    let started = Instant::now();
    let mut pos = initial_position();
    let opening = apply_random_opening(&mut pos, opening_plies, opening_seed);
    let mut binbo_stats = SearchSideStats::default();
    let mut gikou_stats = SearchSideStats::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(pos.key(), 1);

    let mut plies = opening.len();
    let go_command = format!("go btime 0 wtime 0 byoyomi {gikou_think_ms}");
    let timeout = Duration::from_millis(gikou_think_ms.saturating_mul(4).saturating_add(10_000));

    while plies < max_plies {
        if pos.legal_moves().is_empty() {
            let binbo_to_move = (pos.side_to_move() == Color::Black) == binbo_is_black;
            return make_vs_gikou_result(
                if binbo_to_move {
                    VsGikouOutcome::BinboLoss
                } else {
                    VsGikouOutcome::BinboWin
                },
                "no_legal_moves",
                plies,
                opening,
                started,
                binbo_stats,
                gikou_stats,
            );
        }

        let binbo_to_move = (pos.side_to_move() == Color::Black) == binbo_is_black;
        let mv = if binbo_to_move {
            let think_started = Instant::now();
            let result = nnue.think(&pos, limits);
            binbo_stats.record(result.depth, result.nodes, think_started.elapsed());
            match result.bestmove {
                BestMove::Move(mv) => mv,
                BestMove::Resign => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboLoss,
                        "binbo_resign",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        gikou_stats,
                    );
                }
            }
        } else {
            let sfen = position_to_sfen(&pos);
            let analysis = match gikou.bestmove_sfen(&sfen, &go_command, timeout) {
                Ok(value) => value,
                Err(err) if err.requires_restart() => {
                    if let Err(restart_err) = gikou.restart() {
                        return make_vs_gikou_result(
                            VsGikouOutcome::Error,
                            &compact_reason(&format!("gikou_restart_failed:{restart_err}")),
                            plies,
                            opening,
                            started,
                            binbo_stats,
                            gikou_stats,
                        );
                    }
                    match gikou.bestmove_sfen(&sfen, &go_command, timeout) {
                        Ok(value) => value,
                        Err(retry_err) => {
                            return make_vs_gikou_result(
                                VsGikouOutcome::Error,
                                &compact_reason(&format!("gikou_retry_failed:{retry_err}")),
                                plies,
                                opening,
                                started,
                                binbo_stats,
                                gikou_stats,
                            );
                        }
                    }
                }
                Err(err) => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::Error,
                        &compact_reason(&format!("gikou_error:{err}")),
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        gikou_stats,
                    );
                }
            };
            gikou_stats.record(analysis.reported_depth, analysis.nodes, analysis.elapsed);
            match analysis.bestmove.as_str() {
                "resign" => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboWin,
                        "gikou_resign",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        gikou_stats,
                    );
                }
                "win" => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboLoss,
                        "gikou_win_declaration",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        gikou_stats,
                    );
                }
                bestmove => match find_legal_move(&pos, bestmove) {
                    Some(mv) => mv,
                    None => {
                        return make_vs_gikou_result(
                            VsGikouOutcome::Error,
                            &compact_reason(&format!("gikou_illegal_bestmove:{bestmove}")),
                            plies,
                            opening,
                            started,
                            binbo_stats,
                            gikou_stats,
                        );
                    }
                },
            }
        };

        pos.do_move(mv);
        plies += 1;
        let count = repetitions.entry(pos.key()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 4 {
            return make_vs_gikou_result(
                VsGikouOutcome::Draw,
                "repetition_key4",
                plies,
                opening,
                started,
                binbo_stats,
                gikou_stats,
            );
        }
    }

    make_vs_gikou_result(
        VsGikouOutcome::Draw,
        "max_plies",
        plies,
        opening,
        started,
        binbo_stats,
        gikou_stats,
    )
}

fn make_vs_gikou_result(
    outcome: VsGikouOutcome,
    reason: &str,
    plies: usize,
    opening: Vec<String>,
    started: Instant,
    binbo_stats: SearchSideStats,
    gikou_stats: SearchSideStats,
) -> VsGikouGameResult {
    VsGikouGameResult {
        outcome,
        reason: reason.to_string(),
        plies,
        opening,
        elapsed: started.elapsed(),
        binbo_stats,
        gikou_stats,
    }
}

/// Search-engine comparison match (Task 060): the SAME eval (BNNQ `Mode::Quant`, which
/// returns centipawns) drives both sides — binbo's internal search vs YaneuraOu's search,
/// with YaneuraOu loading the bit-exact nn.bin written from the same weights. The only
/// variable is the search engine. Stops early on an SPRT verdict (trinomial/BayesElo).
fn run_nnue_vs_yane(args: &[String]) -> Result<(), String> {
    let usage = "usage: nnue-vs-yane <bnnq> <yane_bin> <eval_dir> [movetime_ms=100] [seed] [max_games=200] [opening_plies=6] [max_plies=320] [depth=24] [elo0=0] [elo1=-50] [alpha=0.05] [beta=0.05]";
    let bnnq = args.first().ok_or_else(|| usage.to_string())?;
    let yane_bin = args.get(1).ok_or_else(|| usage.to_string())?;
    let eval_dir = args.get(2).ok_or_else(|| usage.to_string())?;
    let movetime_ms: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid movetime_ms: {err}"))?,
        None => 100,
    };
    let seed: u64 = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260622,
    };
    let max_games: u64 = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_games: {err}"))?,
        None => 200,
    };
    let opening_plies: usize = match args.get(6) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 6,
    };
    let max_plies: usize = match args.get(7) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let depth: u16 = match args.get(8) {
        Some(value) => value.parse().map_err(|err| format!("invalid depth: {err}"))?,
        None => DEFAULT_VS_YANE_DEPTH,
    };
    let elo0: f64 = match args.get(9) {
        Some(value) => value.parse().map_err(|err| format!("invalid elo0: {err}"))?,
        None => 0.0,
    };
    // H1 default is NEGATIVE: the experiment expects binbo's simple search to be WEAKER than
    // YaneuraOu's refined search (same eval), so "accept H1" = "binbo's search is >=50 elo
    // worse = refined search significantly stronger" — the meaningful headline to establish.
    let elo1: f64 = match args.get(10) {
        Some(value) => value.parse().map_err(|err| format!("invalid elo1: {err}"))?,
        None => -50.0,
    };
    let alpha: f64 = match args.get(11) {
        Some(value) => value.parse().map_err(|err| format!("invalid alpha: {err}"))?,
        None => 0.05,
    };
    let beta: f64 = match args.get(12) {
        Some(value) => value.parse().map_err(|err| format!("invalid beta: {err}"))?,
        None => 0.05,
    };
    let sprt_cfg = SprtConfig { elo0, elo1, alpha, beta };
    let (llr_lower, llr_upper) = bounds(&sprt_cfg);

    let mut nnue = NnueEngine::load(bnnq, depth)?;
    // budget_from_limits subtracts a 500ms safety margin, so add it back to honor movetime.
    let limits = GoLimits {
        movetime: Some(movetime_ms.saturating_add(500)),
        ..GoLimits::default()
    };
    let mut yane = start_yane_match_engine(yane_bin, eval_dir)
        .map_err(|err| format!("failed to start YaneuraOu: {err}"))?;

    let started = Instant::now();
    let mut wins = 0_u64;
    let mut losses = 0_u64;
    let mut draws = 0_u64;
    let mut errors = 0_u64;
    let mut total_plies = 0_usize;
    let mut binbo_stats = SearchSideStats::default();
    let mut yane_stats = SearchSideStats::default();

    println!("nnue-vs-yane bnnq={bnnq} yane_bin={yane_bin} eval_dir={eval_dir} movetime_ms={movetime_ms} seed={seed} max_games={max_games} opening_plies={opening_plies} max_plies={max_plies} depth={depth}");
    println!("yane_options=EvalDir:{eval_dir},USI_OwnBook:false,Threads:1,MinimumThinkingTime:0,NetworkDelay:0,NetworkDelay2:0");
    println!("time_control=binbo_movetime_ms:{} yane_go='go btime 0 wtime 0 byoyomi {movetime_ms}'", movetime_ms.saturating_add(500));
    println!("sprt elo0={elo0} elo1={elo1} alpha={alpha} beta={beta} llr_bounds=[{llr_lower:.3},{llr_upper:.3}]");
    println!("game\tbinbo_color\topening_seed\tresult\treason\tplies\telapsed_ms\tbinbo_moves\tbinbo_avg_depth\tbinbo_avg_nodes\tbinbo_avg_ms\tyane_moves\tyane_avg_depth\tyane_avg_nodes\tyane_avg_ms\tW\tD\tL\tllr\tverdict\topening");

    let mut verdict = SprtVerdict::Continue;
    let mut played = 0_u64;
    for game in 0..max_games {
        nnue.new_game();
        if let Err(err) = yane.new_game() {
            // A failed usinewgame means the child is unhealthy; restart before continuing.
            if let Err(restart_err) = yane.restart() {
                return Err(format!("YaneuraOu restart failed: {restart_err}"));
            }
            let _ = err;
        }
        let binbo_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = play_nnue_vs_yane_game(
            &mut nnue,
            &mut yane,
            &limits,
            binbo_is_black,
            opening_seed,
            opening_plies,
            max_plies,
            movetime_ms,
        );

        match result.outcome {
            VsGikouOutcome::BinboWin => wins += 1,
            VsGikouOutcome::BinboLoss => losses += 1,
            VsGikouOutcome::Draw => draws += 1,
            VsGikouOutcome::Error => errors += 1,
        }
        total_plies += result.plies;
        binbo_stats.merge(&result.binbo_stats);
        yane_stats.merge(&result.gikou_stats);
        played += 1;

        // SPRT is computed only over decided/drawn games (errors are excluded, not scored).
        let (llr, v) = sprt(wins, draws, losses, &sprt_cfg);
        verdict = v;
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}",
            game + 1,
            if binbo_is_black { "black" } else { "white" },
            opening_seed,
            result.outcome.as_str(),
            result.reason,
            result.plies,
            result.elapsed.as_secs_f64() * 1000.0,
            result.binbo_stats.moves,
            fmt_avg(result.binbo_stats.avg_depth()),
            fmt_avg(result.binbo_stats.avg_nodes()),
            fmt_avg(result.binbo_stats.avg_time_ms()),
            result.gikou_stats.moves,
            fmt_avg(result.gikou_stats.avg_depth()),
            fmt_avg(result.gikou_stats.avg_nodes()),
            fmt_avg(result.gikou_stats.avg_time_ms()),
            wins,
            draws,
            losses,
            llr,
            v.as_str(),
            result.opening.join(",")
        );

        if v != SprtVerdict::Continue {
            break;
        }
    }

    let scored_games = wins + draws + losses;
    let (score, ci_lo, ci_hi) = score_ci(wins, draws, losses);
    let elo = elo_from_score(score);
    let elo_lo = elo_from_score(ci_lo);
    let elo_hi = elo_from_score(ci_hi);
    let avg_plies = if played > 0 {
        total_plies as f64 / played as f64
    } else {
        0.0
    };
    let (final_llr, _) = sprt(wins, draws, losses, &sprt_cfg);
    let elapsed = started.elapsed();

    println!("summary_movetime_ms={movetime_ms}");
    println!("binbo_record={wins}-{losses}-{draws} errors={errors} games_played={played}");
    println!("binbo_score_rate={:.1}% (wins + 0.5*draws over {scored_games} scored games)", 100.0 * score);
    println!("binbo_score_ci95=[{:.1}%,{:.1}%]", 100.0 * ci_lo, 100.0 * ci_hi);
    println!("binbo_elo={elo:.1} elo_ci95=[{elo_lo:.1},{elo_hi:.1}]");
    println!("sprt_llr={final_llr:.3} llr_bounds=[{llr_lower:.3},{llr_upper:.3}] verdict={}", verdict.as_str());
    println!("avg_plies={avg_plies:.1}");
    println!("binbo_moves={}", binbo_stats.moves);
    println!("binbo_avg_depth={}", fmt_avg(binbo_stats.avg_depth()));
    println!("binbo_max_depth={}", binbo_stats.max_depth);
    println!("binbo_avg_nodes={}", fmt_avg(binbo_stats.avg_nodes()));
    println!("binbo_max_nodes={}", binbo_stats.max_nodes);
    println!("binbo_avg_time_ms={}", fmt_avg(binbo_stats.avg_time_ms()));
    println!("yane_moves={}", yane_stats.moves);
    println!("yane_avg_depth={}", fmt_avg(yane_stats.avg_depth()));
    println!("yane_max_depth={}", yane_stats.max_depth);
    println!("yane_avg_nodes={}", fmt_avg(yane_stats.avg_nodes()));
    println!("yane_max_nodes={}", yane_stats.max_nodes);
    println!("yane_avg_time_ms={}", fmt_avg(yane_stats.avg_time_ms()));
    println!("elapsed_ms={:.1}", elapsed.as_secs_f64() * 1000.0);
    println!("status={}", if errors == 0 { "ok" } else { "incomplete" });
    Ok(())
}

fn start_yane_match_engine(yane_bin: &str, eval_dir: &str) -> Result<UsiEngine, crate::usi_engine::UsiError> {
    start_yane_match_engine_with(yane_bin, eval_dir, &[])
}

/// `start_yane_match_engine` plus caller-supplied setoptions sent after the defaults (so they
/// win), e.g. `NodesLimit` / `FV_SCALE` for an external anchor (Task 108).
fn start_yane_match_engine_with(
    yane_bin: &str,
    eval_dir: &str,
    extra: &[(String, String)],
) -> Result<UsiEngine, crate::usi_engine::UsiError> {
    let mut setoptions = vec![
        ("EvalDir".to_string(), eval_dir.to_string()),
        ("USI_OwnBook".to_string(), "false".to_string()),
        ("Threads".to_string(), "1".to_string()),
        ("MinimumThinkingTime".to_string(), "0".to_string()),
        ("NetworkDelay".to_string(), "0".to_string()),
        ("NetworkDelay2".to_string(), "0".to_string()),
    ];
    setoptions.extend(extra.iter().cloned());
    let options = UsiOptions::new(Path::new(yane_bin), setoptions);
    UsiEngine::start(options)
}

/// Parallel of `play_nnue_vs_gikou_game` for a generic USI opponent (YaneuraOu). Reuses
/// `VsGikouOutcome`/`SearchSideStats`/repetition tracking; the opponent's stats land in the
/// `gikou_stats` slot of `VsGikouGameResult` (relabelled "yane" by the caller).
fn play_nnue_vs_yane_game(
    nnue: &mut NnueEngine,
    yane: &mut UsiEngine,
    limits: &GoLimits,
    binbo_is_black: bool,
    opening_seed: u64,
    opening_plies: usize,
    max_plies: usize,
    movetime_ms: u64,
) -> VsGikouGameResult {
    let started = Instant::now();
    let mut pos = initial_position();
    let opening = apply_random_opening(&mut pos, opening_plies, opening_seed);
    let mut binbo_stats = SearchSideStats::default();
    let mut yane_stats = SearchSideStats::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(pos.key(), 1);

    let mut plies = opening.len();
    let go_command = format!("go btime 0 wtime 0 byoyomi {movetime_ms}");
    let timeout = Duration::from_millis(movetime_ms.saturating_mul(4).saturating_add(10_000));

    while plies < max_plies {
        if pos.legal_moves().is_empty() {
            let binbo_to_move = (pos.side_to_move() == Color::Black) == binbo_is_black;
            return make_vs_gikou_result(
                if binbo_to_move {
                    VsGikouOutcome::BinboLoss
                } else {
                    VsGikouOutcome::BinboWin
                },
                "no_legal_moves",
                plies,
                opening,
                started,
                binbo_stats,
                yane_stats,
            );
        }

        let binbo_to_move = (pos.side_to_move() == Color::Black) == binbo_is_black;
        let mv = if binbo_to_move {
            let think_started = Instant::now();
            let result = nnue.think(&pos, limits);
            binbo_stats.record(result.depth, result.nodes, think_started.elapsed());
            match result.bestmove {
                BestMove::Move(mv) => mv,
                BestMove::Resign => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboLoss,
                        "binbo_resign",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        yane_stats,
                    );
                }
            }
        } else {
            let sfen = position_to_sfen(&pos);
            let analysis = match yane.bestmove(&sfen, &go_command, timeout) {
                Ok(value) => value,
                Err(err) if err.requires_restart() => {
                    if let Err(restart_err) = yane.restart() {
                        return make_vs_gikou_result(
                            VsGikouOutcome::Error,
                            &compact_reason(&format!("yane_restart_failed:{restart_err}")),
                            plies,
                            opening,
                            started,
                            binbo_stats,
                            yane_stats,
                        );
                    }
                    match yane.bestmove(&sfen, &go_command, timeout) {
                        Ok(value) => value,
                        Err(retry_err) => {
                            return make_vs_gikou_result(
                                VsGikouOutcome::Error,
                                &compact_reason(&format!("yane_retry_failed:{retry_err}")),
                                plies,
                                opening,
                                started,
                                binbo_stats,
                                yane_stats,
                            );
                        }
                    }
                }
                Err(err) => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::Error,
                        &compact_reason(&format!("yane_error:{err}")),
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        yane_stats,
                    );
                }
            };
            yane_stats.record(analysis.depth, analysis.nodes, analysis.elapsed);
            match analysis.bestmove.as_str() {
                "resign" => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboWin,
                        "yane_resign",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        yane_stats,
                    );
                }
                "win" => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::BinboLoss,
                        "yane_win_declaration",
                        plies,
                        opening,
                        started,
                        binbo_stats,
                        yane_stats,
                    );
                }
                bestmove => match find_legal_move(&pos, bestmove) {
                    Some(mv) => mv,
                    None => {
                        return make_vs_gikou_result(
                            VsGikouOutcome::Error,
                            &compact_reason(&format!("yane_illegal_bestmove:{bestmove}")),
                            plies,
                            opening,
                            started,
                            binbo_stats,
                            yane_stats,
                        );
                    }
                },
            }
        };

        pos.do_move(mv);
        plies += 1;
        let count = repetitions.entry(pos.key()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 4 {
            return make_vs_gikou_result(
                VsGikouOutcome::Draw,
                "repetition_key4",
                plies,
                opening,
                started,
                binbo_stats,
                yane_stats,
            );
        }
    }

    make_vs_gikou_result(
        VsGikouOutcome::Draw,
        "max_plies",
        plies,
        opening,
        started,
        binbo_stats,
        yane_stats,
    )
}

/// Eval-vs-eval generation comparison (Task 061): TWO YaneuraOu instances, the SAME search,
/// each loading a different nn.bin (EvalDir) — candidate (A) vs baseline (B). The only
/// variable is the eval, so the result measures eval strength (the opposite axis from Task
/// 060's search comparison). Stops early on an SPRT verdict (H1 = candidate is BETTER).
fn run_yane_vs_yane(args: &[String]) -> Result<(), String> {
    let usage = "usage: yane-vs-yane <yane_bin> <eval_dir_a> <eval_dir_b> [movetime_ms=100] [seed] [max_games=400] [opening_plies=6] [max_plies=320] [elo0=0] [elo1=10] [alpha=0.05] [beta=0.05]";
    let yane_bin = args.first().ok_or_else(|| usage.to_string())?;
    let eval_dir_a = args.get(1).ok_or_else(|| usage.to_string())?;
    let eval_dir_b = args.get(2).ok_or_else(|| usage.to_string())?;
    let movetime_ms: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid movetime_ms: {err}"))?,
        None => 100,
    };
    let seed: u64 = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260622,
    };
    let max_games: u64 = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_games: {err}"))?,
        None => 400,
    };
    let opening_plies: usize = match args.get(6) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 6,
    };
    let max_plies: usize = match args.get(7) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    // H1 default POSITIVE: we want to detect that the candidate (A) is BETTER than the
    // baseline (B), so "accept H1" = "candidate >= elo1 stronger" = adopt.
    let elo0: f64 = match args.get(8) {
        Some(value) => value.parse().map_err(|err| format!("invalid elo0: {err}"))?,
        None => 0.0,
    };
    let elo1: f64 = match args.get(9) {
        Some(value) => value.parse().map_err(|err| format!("invalid elo1: {err}"))?,
        None => 10.0,
    };
    let alpha: f64 = match args.get(10) {
        Some(value) => value.parse().map_err(|err| format!("invalid alpha: {err}"))?,
        None => 0.05,
    };
    let beta: f64 = match args.get(11) {
        Some(value) => value.parse().map_err(|err| format!("invalid beta: {err}"))?,
        None => 0.05,
    };
    let sprt_cfg = SprtConfig { elo0, elo1, alpha, beta };
    let (llr_lower, llr_upper) = bounds(&sprt_cfg);

    let mut engine_a = start_yane_match_engine(yane_bin, eval_dir_a)
        .map_err(|err| format!("failed to start YaneuraOu A (candidate): {err}"))?;
    let mut engine_b = start_yane_match_engine(yane_bin, eval_dir_b)
        .map_err(|err| format!("failed to start YaneuraOu B (baseline): {err}"))?;


    println!("yane-vs-yane yane_bin={yane_bin} eval_dir_a(cand)={eval_dir_a} eval_dir_b(base)={eval_dir_b} movetime_ms={movetime_ms} seed={seed} max_games={max_games} opening_plies={opening_plies} max_plies={max_plies}");
    println!("yane_options=USI_OwnBook:false,Threads:1,MinimumThinkingTime:0,NetworkDelay:0,NetworkDelay2:0 go='go btime 0 wtime 0 byoyomi {movetime_ms}'");
    println!("sprt elo0={elo0} elo1={elo1} alpha={alpha} beta={beta} llr_bounds=[{llr_lower:.3},{llr_upper:.3}] (W/D/L from candidate A perspective)");
    let go_command = format!("go btime 0 wtime 0 byoyomi {movetime_ms}");
    let timeout = Duration::from_millis(movetime_ms.saturating_mul(4).saturating_add(10_000));
    run_yane_pair_series(
        &mut engine_a,
        &mut engine_b,
        (&go_command, timeout),
        (&go_command, timeout),
        seed,
        max_games,
        opening_plies,
        max_plies,
        &sprt_cfg,
        &format!("summary_movetime_ms={movetime_ms}"),
    )
}

/// The game loop + TSV rows + summary shared by `yane-vs-yane` and `yane-vs-yane-ex`
/// (output format unchanged: 21 tab-separated columns per game, then the summary block).
#[allow(clippy::too_many_arguments)]
fn run_yane_pair_series(
    engine_a: &mut UsiEngine,
    engine_b: &mut UsiEngine,
    go_a: (&str, Duration),
    go_b: (&str, Duration),
    seed: u64,
    max_games: u64,
    opening_plies: usize,
    max_plies: usize,
    sprt_cfg: &SprtConfig,
    summary_label: &str,
) -> Result<(), String> {
    let (llr_lower, llr_upper) = bounds(sprt_cfg);
    let started = Instant::now();
    let mut wins = 0_u64; // candidate (A) wins
    let mut losses = 0_u64;
    let mut draws = 0_u64;
    let mut errors = 0_u64;
    let mut total_plies = 0_usize;
    let mut a_stats = SearchSideStats::default();
    let mut b_stats = SearchSideStats::default();
    println!("game\tcand_color\topening_seed\tresult\treason\tplies\telapsed_ms\tcand_moves\tcand_avg_depth\tcand_avg_nodes\tcand_avg_ms\tbase_moves\tbase_avg_depth\tbase_avg_nodes\tbase_avg_ms\tW\tD\tL\tllr\tverdict\topening");

    let mut verdict = SprtVerdict::Continue;
    let mut played = 0_u64;
    for game in 0..max_games {
        if let Err(err) = engine_a.new_game() {
            engine_a.restart().map_err(|e| format!("YaneuraOu A restart failed: {e}"))?;
            let _ = err;
        }
        if let Err(err) = engine_b.new_game() {
            engine_b.restart().map_err(|e| format!("YaneuraOu B restart failed: {e}"))?;
            let _ = err;
        }
        let a_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = play_yane_pair_game(
            engine_a,
            engine_b,
            a_is_black,
            opening_seed,
            opening_plies,
            max_plies,
            go_a,
            go_b,
        );

        match result.outcome {
            VsGikouOutcome::BinboWin => wins += 1,
            VsGikouOutcome::BinboLoss => losses += 1,
            VsGikouOutcome::Draw => draws += 1,
            VsGikouOutcome::Error => errors += 1,
        }
        total_plies += result.plies;
        a_stats.merge(&result.binbo_stats);
        b_stats.merge(&result.gikou_stats);
        played += 1;

        let (llr, v) = sprt(wins, draws, losses, sprt_cfg);
        verdict = v;
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}",
            game + 1,
            if a_is_black { "black" } else { "white" },
            opening_seed,
            result.outcome.as_str(),
            result.reason,
            result.plies,
            result.elapsed.as_secs_f64() * 1000.0,
            result.binbo_stats.moves,
            fmt_avg(result.binbo_stats.avg_depth()),
            fmt_avg(result.binbo_stats.avg_nodes()),
            fmt_avg(result.binbo_stats.avg_time_ms()),
            result.gikou_stats.moves,
            fmt_avg(result.gikou_stats.avg_depth()),
            fmt_avg(result.gikou_stats.avg_nodes()),
            fmt_avg(result.gikou_stats.avg_time_ms()),
            wins,
            draws,
            losses,
            llr,
            v.as_str(),
            result.opening.join(",")
        );

        if v != SprtVerdict::Continue {
            break;
        }
    }

    let scored_games = wins + draws + losses;
    let (score, ci_lo, ci_hi) = score_ci(wins, draws, losses);
    let elo = elo_from_score(score);
    let elo_lo = elo_from_score(ci_lo);
    let elo_hi = elo_from_score(ci_hi);
    let draw_rate = if scored_games > 0 { 100.0 * draws as f64 / scored_games as f64 } else { 0.0 };
    let avg_plies = if played > 0 { total_plies as f64 / played as f64 } else { 0.0 };
    let (final_llr, _) = sprt(wins, draws, losses, &sprt_cfg);
    let elapsed = started.elapsed();

    println!("{summary_label}");
    println!("candidate_record(W-L-D)={wins}-{losses}-{draws} errors={errors} games_played={played}");
    println!("candidate_score_rate={:.1}% (wins + 0.5*draws over {scored_games} scored games)", 100.0 * score);
    println!("candidate_score_ci95=[{:.1}%,{:.1}%]", 100.0 * ci_lo, 100.0 * ci_hi);
    println!("candidate_elo={elo:.1} elo_ci95=[{elo_lo:.1},{elo_hi:.1}]");
    println!("draw_rate={draw_rate:.1}%");
    println!("sprt_llr={final_llr:.3} llr_bounds=[{llr_lower:.3},{llr_upper:.3}] verdict={}", verdict.as_str());
    println!("avg_plies={avg_plies:.1}");
    println!("cand_moves={} cand_avg_depth={} cand_avg_nodes={} cand_avg_time_ms={}",
        a_stats.moves, fmt_avg(a_stats.avg_depth()), fmt_avg(a_stats.avg_nodes()), fmt_avg(a_stats.avg_time_ms()));
    println!("base_moves={} base_avg_depth={} base_avg_nodes={} base_avg_time_ms={}",
        b_stats.moves, fmt_avg(b_stats.avg_depth()), fmt_avg(b_stats.avg_nodes()), fmt_avg(b_stats.avg_time_ms()));
    println!("elapsed_ms={:.1}", elapsed.as_secs_f64() * 1000.0);
    println!("status={}", if errors == 0 { "ok" } else { "incomplete" });
    Ok(())
}

/// `yane-vs-yane` with a per-side engine spec (Task 108, external anchors): each side has its
/// own YaneuraOu binary (an anchor may need a different NNUE build), EvalDir, extra setoptions
/// (e.g. `FV_SCALE`) and search limit -- a fixed node budget (`go nodes N`, independent of
/// CPU load) or a byoyomi in ms. Output is the same 21-column TSV + summary as `yane-vs-yane`.
fn run_yane_vs_yane_ex(args: &[String]) -> Result<(), String> {
    let usage = "usage: yane-vs-yane-ex --a-eval <dir> --b-eval <dir> [--a-bin|--b-bin <yane_bin>] [--a-opt|--b-opt Name=Value]... [--a-nodes|--b-nodes N | --a-movetime|--b-movetime MS] [--seed S] [--games N] [--opening-plies 6] [--max-plies 320]";
    #[derive(Default)]
    struct Side {
        bin: Option<String>,
        eval: Option<String>,
        opts: Vec<(String, String)>,
        nodes: Option<u64>,
        movetime: Option<u64>,
    }
    let (mut a, mut b) = (Side::default(), Side::default());
    let (mut seed, mut games, mut opening_plies, mut max_plies) = (20260622_u64, 400_u64, 6_usize, 320_usize);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value\n{usage}"))?;
        let num = |v: &str| v.parse::<u64>().map_err(|e| format!("invalid {flag} {v}: {e}"));
        let opt = |v: &str| {
            v.split_once('=')
                .map(|(k, x)| (k.to_string(), x.to_string()))
                .ok_or_else(|| format!("{flag} expects Name=Value, got {v}"))
        };
        match flag.as_str() {
            "--a-bin" => a.bin = Some(value.clone()),
            "--b-bin" => b.bin = Some(value.clone()),
            "--a-eval" => a.eval = Some(value.clone()),
            "--b-eval" => b.eval = Some(value.clone()),
            "--a-opt" => a.opts.push(opt(value)?),
            "--b-opt" => b.opts.push(opt(value)?),
            "--a-nodes" => a.nodes = Some(num(value)?),
            "--b-nodes" => b.nodes = Some(num(value)?),
            "--a-movetime" => a.movetime = Some(num(value)?),
            "--b-movetime" => b.movetime = Some(num(value)?),
            "--seed" => seed = num(value)?,
            "--games" => games = num(value)?,
            "--opening-plies" => opening_plies = num(value)? as usize,
            "--max-plies" => max_plies = num(value)? as usize,
            _ => return Err(format!("unknown flag {flag}\n{usage}")),
        }
    }
    let default_bin = std::env::var("YANE").unwrap_or_default();
    let spec = |side: &Side, name: &str| -> Result<(String, String, String, Duration), String> {
        let bin = side.bin.clone().unwrap_or_else(|| default_bin.clone());
        if bin.is_empty() {
            return Err(format!("--{name}-bin missing (or set $YANE)\n{usage}"));
        }
        let eval = side.eval.clone().ok_or_else(|| format!("--{name}-eval missing\n{usage}"))?;
        let (go, timeout) = match (side.nodes, side.movetime) {
            (Some(_), Some(_)) => return Err(format!("--{name}-nodes and --{name}-movetime are exclusive")),
            (Some(n), None) => (format!("go nodes {n}"), Duration::from_secs(120)),
            (None, ms) => {
                let ms = ms.unwrap_or(100);
                (format!("go btime 0 wtime 0 byoyomi {ms}"),
                 Duration::from_millis(ms.saturating_mul(4).saturating_add(10_000)))
            }
        };
        Ok((bin, eval, go, timeout))
    };
    let (bin_a, eval_a, go_a, timeout_a) = spec(&a, "a")?;
    let (bin_b, eval_b, go_b, timeout_b) = spec(&b, "b")?;
    let fmt_opts = |o: &[(String, String)]| o.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",");

    let mut engine_a = start_yane_match_engine_with(&bin_a, &eval_a, &a.opts)
        .map_err(|err| format!("failed to start A (candidate): {err}"))?;
    let mut engine_b = start_yane_match_engine_with(&bin_b, &eval_b, &b.opts)
        .map_err(|err| format!("failed to start B (anchor/baseline): {err}"))?;
    let sprt_cfg = SprtConfig { elo0: 0.0, elo1: 10.0, alpha: 0.05, beta: 0.05 };
    let (llr_lower, llr_upper) = bounds(&sprt_cfg);
    println!("yane-vs-yane-ex bin_a={bin_a} eval_dir_a(cand)={eval_a} bin_b={bin_b} eval_dir_b(base)={eval_b} seed={seed} max_games={games} opening_plies={opening_plies} max_plies={max_plies}");
    println!("yane_options=USI_OwnBook:false,Threads:1,MinimumThinkingTime:0,NetworkDelay:0,NetworkDelay2:0 extra_a=[{}] extra_b=[{}] go_a='{go_a}' go_b='{go_b}'", fmt_opts(&a.opts), fmt_opts(&b.opts));
    println!("sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 llr_bounds=[{llr_lower:.3},{llr_upper:.3}] (W/D/L from candidate A perspective, recorded only)");
    run_yane_pair_series(
        &mut engine_a,
        &mut engine_b,
        (&go_a, timeout_a),
        (&go_b, timeout_b),
        seed,
        games,
        opening_plies,
        max_plies,
        &sprt_cfg,
        &format!("summary_go_a='{go_a}' go_b='{go_b}'"),
    )
}

/// External-engine anchor match (Task 066): champion eval in YaneuraOu (full search, A) vs
/// gikou/技巧2 (B), the FIXED external reference for the self-reinforcement loop. Same shape as
/// `yane-vs-yane` but side B is gikou. W/D/L is from the champion (A) perspective; running both a
/// new generation and the previous champion against this anchor detects self-poisoning (a gen
/// that beats the previous gen but loses ground vs the fixed anchor).
fn run_yane_vs_gikou(args: &[String]) -> Result<(), String> {
    let usage = "usage: yane-vs-gikou <yane_bin> <champion_eval_dir> <gikou_path> [movetime_ms=100] [gikou_level=32] [seed] [max_games=400] [opening_plies=6] [max_plies=320] [gikou_think_ms=movetime] [elo0=0] [elo1=10] [alpha=0.05] [beta=0.05]";
    let yane_bin = args.first().ok_or_else(|| usage.to_string())?;
    let eval_dir_a = args.get(1).ok_or_else(|| usage.to_string())?;
    let gikou_path = args.get(2).ok_or_else(|| usage.to_string())?;
    let movetime_ms: u64 = match args.get(3) { Some(v) => v.parse().map_err(|e| format!("invalid movetime_ms: {e}"))?, None => 100 };
    // gikou (技巧2) reliably EOF-crashes mid-game when its search reaches depth >= ~8 (known
    // warm-state instability; gen-teacher sees ~80% crash at depth 10). DepthLimit 6 finishes in
    // a few ms and is rock-stable (0 errors), so it is the default for a usable fixed anchor.
    let gikou_level: usize = match args.get(4) { Some(v) => v.parse().map_err(|e| format!("invalid gikou_level: {e}"))?, None => 6 };
    let seed: u64 = match args.get(5) { Some(v) => v.parse().map_err(|e| format!("invalid seed: {e}"))?, None => 20260622 };
    let max_games: u64 = match args.get(6) { Some(v) => v.parse().map_err(|e| format!("invalid max_games: {e}"))?, None => 400 };
    let opening_plies: usize = match args.get(7) { Some(v) => v.parse().map_err(|e| format!("invalid opening_plies: {e}"))?, None => 6 };
    let max_plies: usize = match args.get(8) { Some(v) => v.parse().map_err(|e| format!("invalid max_plies: {e}"))?, None => 320 };
    let gikou_think_ms: u64 = match args.get(9) { Some(v) => v.parse().map_err(|e| format!("invalid gikou_think_ms: {e}"))?, None => movetime_ms };
    let elo0: f64 = match args.get(10) { Some(v) => v.parse().map_err(|e| format!("invalid elo0: {e}"))?, None => 0.0 };
    let elo1: f64 = match args.get(11) { Some(v) => v.parse().map_err(|e| format!("invalid elo1: {e}"))?, None => 10.0 };
    let alpha: f64 = match args.get(12) { Some(v) => v.parse().map_err(|e| format!("invalid alpha: {e}"))?, None => 0.05 };
    let beta: f64 = match args.get(13) { Some(v) => v.parse().map_err(|e| format!("invalid beta: {e}"))?, None => 0.05 };
    let sprt_cfg = SprtConfig { elo0, elo1, alpha, beta };
    let (llr_lower, llr_upper) = bounds(&sprt_cfg);

    let mut engine_a = start_yane_match_engine(yane_bin, eval_dir_a)
        .map_err(|err| format!("failed to start YaneuraOu A (champion): {err}"))?;
    let mut gikou = start_gikou_match_engine(gikou_path, gikou_level)
        .map_err(|err| format!("failed to start gikou (anchor): {err}"))?;

    let started = Instant::now();
    let mut wins = 0_u64;
    let mut losses = 0_u64;
    let mut draws = 0_u64;
    let mut errors = 0_u64;
    let mut total_plies = 0_usize;
    let mut a_stats = SearchSideStats::default();
    let mut g_stats = SearchSideStats::default();

    println!("yane-vs-gikou yane_bin={yane_bin} champion_eval_dir={eval_dir_a} gikou_path={gikou_path} movetime_ms={movetime_ms} gikou_level={gikou_level} gikou_think_ms={gikou_think_ms} seed={seed} max_games={max_games} opening_plies={opening_plies} max_plies={max_plies}");
    println!("sprt elo0={elo0} elo1={elo1} alpha={alpha} beta={beta} llr_bounds=[{llr_lower:.3},{llr_upper:.3}] (W/D/L from champion A perspective)");
    println!("game\tchamp_color\topening_seed\tresult\treason\tplies\telapsed_ms\tchamp_moves\tchamp_avg_depth\tchamp_avg_nodes\tchamp_avg_ms\tgikou_moves\tgikou_avg_depth\tgikou_avg_nodes\tgikou_avg_ms\tW\tD\tL\tllr\tverdict\topening");

    let mut verdict = SprtVerdict::Continue;
    let mut played = 0_u64;
    for game in 0..max_games {
        if let Err(err) = engine_a.new_game() {
            engine_a.restart().map_err(|e| format!("YaneuraOu A restart failed: {e}"))?;
            let _ = err;
        }
        let a_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = play_yane_vs_gikou_game(
            &mut engine_a, &mut gikou, a_is_black, opening_seed, opening_plies, max_plies, movetime_ms, gikou_think_ms,
        );
        match result.outcome {
            VsGikouOutcome::BinboWin => wins += 1,
            VsGikouOutcome::BinboLoss => losses += 1,
            VsGikouOutcome::Draw => draws += 1,
            VsGikouOutcome::Error => errors += 1,
        }
        total_plies += result.plies;
        a_stats.merge(&result.binbo_stats);
        g_stats.merge(&result.gikou_stats);
        played += 1;

        let (llr, v) = sprt(wins, draws, losses, &sprt_cfg);
        verdict = v;
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}",
            game + 1,
            if a_is_black { "black" } else { "white" },
            opening_seed,
            result.outcome.as_str(),
            result.reason,
            result.plies,
            result.elapsed.as_secs_f64() * 1000.0,
            result.binbo_stats.moves,
            fmt_avg(result.binbo_stats.avg_depth()),
            fmt_avg(result.binbo_stats.avg_nodes()),
            fmt_avg(result.binbo_stats.avg_time_ms()),
            result.gikou_stats.moves,
            fmt_avg(result.gikou_stats.avg_depth()),
            fmt_avg(result.gikou_stats.avg_nodes()),
            fmt_avg(result.gikou_stats.avg_time_ms()),
            wins, draws, losses, llr, v.as_str(),
            result.opening.join(",")
        );
        if v != SprtVerdict::Continue { break; }
    }

    let scored_games = wins + draws + losses;
    let (score, ci_lo, ci_hi) = score_ci(wins, draws, losses);
    let elo = elo_from_score(score);
    let elo_lo = elo_from_score(ci_lo);
    let elo_hi = elo_from_score(ci_hi);
    let draw_rate = if scored_games > 0 { 100.0 * draws as f64 / scored_games as f64 } else { 0.0 };
    let avg_plies = if played > 0 { total_plies as f64 / played as f64 } else { 0.0 };
    let (final_llr, _) = sprt(wins, draws, losses, &sprt_cfg);
    let elapsed = started.elapsed();

    println!("summary_movetime_ms={movetime_ms}");
    println!("champion_record(W-L-D)={wins}-{losses}-{draws} errors={errors} games_played={played}");
    println!("champion_score_rate={:.1}% (wins + 0.5*draws over {scored_games} scored games)", 100.0 * score);
    println!("champion_score_ci95=[{:.1}%,{:.1}%]", 100.0 * ci_lo, 100.0 * ci_hi);
    println!("champion_elo={elo:.1} elo_ci95=[{elo_lo:.1},{elo_hi:.1}]");
    println!("draw_rate={draw_rate:.1}%");
    println!("sprt_llr={final_llr:.3} llr_bounds=[{llr_lower:.3},{llr_upper:.3}] verdict={}", verdict.as_str());
    println!("avg_plies={avg_plies:.1}");
    println!("champ_moves={} champ_avg_depth={} champ_avg_nodes={} champ_avg_time_ms={}",
        a_stats.moves, fmt_avg(a_stats.avg_depth()), fmt_avg(a_stats.avg_nodes()), fmt_avg(a_stats.avg_time_ms()));
    println!("gikou_moves={} gikou_avg_depth={} gikou_avg_nodes={} gikou_avg_time_ms={}",
        g_stats.moves, fmt_avg(g_stats.avg_depth()), fmt_avg(g_stats.avg_nodes()), fmt_avg(g_stats.avg_time_ms()));
    println!("elapsed_ms={:.1}", elapsed.as_secs_f64() * 1000.0);
    println!("status={}", if errors == 0 { "ok" } else { "incomplete" });
    Ok(())
}

/// One game: champion in YaneuraOu (A) vs gikou (B, the anchor). Shared binbo `Position` enforces
/// legality/repetition. W/D/L from A. A's stats land in `binbo_stats`, gikou's in `gikou_stats`.
#[allow(clippy::too_many_arguments)]
fn play_yane_vs_gikou_game(
    engine_a: &mut UsiEngine,
    gikou: &mut GikouEngine,
    a_is_black: bool,
    opening_seed: u64,
    opening_plies: usize,
    max_plies: usize,
    movetime_ms: u64,
    gikou_think_ms: u64,
) -> VsGikouGameResult {
    let started = Instant::now();
    let mut pos = initial_position();
    let opening = apply_random_opening(&mut pos, opening_plies, opening_seed);
    let mut a_stats = SearchSideStats::default();
    let mut g_stats = SearchSideStats::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(pos.key(), 1);

    let mut plies = opening.len();
    let a_go = format!("go btime 0 wtime 0 byoyomi {movetime_ms}");
    let g_go = format!("go btime 0 wtime 0 byoyomi {gikou_think_ms}");
    let a_timeout = Duration::from_millis(movetime_ms.saturating_mul(4).saturating_add(10_000));
    let g_timeout = Duration::from_millis(gikou_think_ms.saturating_mul(4).saturating_add(10_000));

    while plies < max_plies {
        let a_to_move = (pos.side_to_move() == Color::Black) == a_is_black;
        if pos.legal_moves().is_empty() {
            return make_vs_gikou_result(
                if a_to_move { VsGikouOutcome::BinboLoss } else { VsGikouOutcome::BinboWin },
                "no_legal_moves", plies, opening, started, a_stats, g_stats,
            );
        }
        let sfen = position_to_sfen(&pos);
        let mv = if a_to_move {
            let analysis = match usi_bestmove_retry(engine_a, &sfen, &a_go, a_timeout) {
                Ok(value) => value,
                Err(reason) => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::Error,
                        &compact_reason(&format!("champ_{reason}")),
                        plies, opening, started, a_stats, g_stats,
                    );
                }
            };
            a_stats.record(analysis.depth, analysis.nodes, analysis.elapsed);
            match analysis.bestmove.as_str() {
                "resign" => return make_vs_gikou_result(VsGikouOutcome::BinboLoss, "champ_resign", plies, opening, started, a_stats, g_stats),
                "win" => return make_vs_gikou_result(VsGikouOutcome::BinboWin, "champ_win_declaration", plies, opening, started, a_stats, g_stats),
                bestmove => match find_legal_move(&pos, bestmove) {
                    Some(mv) => mv,
                    None => return make_vs_gikou_result(VsGikouOutcome::Error, &compact_reason(&format!("champ_illegal_bestmove:{bestmove}")), plies, opening, started, a_stats, g_stats),
                },
            }
        } else {
            let analysis = match gikou.bestmove_sfen(&sfen, &g_go, g_timeout) {
                Ok(value) => value,
                Err(err) if err.requires_restart() => {
                    if let Err(restart_err) = gikou.restart() {
                        return make_vs_gikou_result(VsGikouOutcome::Error, &compact_reason(&format!("gikou_restart_failed:{restart_err}")), plies, opening, started, a_stats, g_stats);
                    }
                    match gikou.bestmove_sfen(&sfen, &g_go, g_timeout) {
                        Ok(value) => value,
                        Err(retry_err) => return make_vs_gikou_result(VsGikouOutcome::Error, &compact_reason(&format!("gikou_retry_failed:{retry_err}")), plies, opening, started, a_stats, g_stats),
                    }
                }
                Err(err) => return make_vs_gikou_result(VsGikouOutcome::Error, &compact_reason(&format!("gikou_error:{err}")), plies, opening, started, a_stats, g_stats),
            };
            g_stats.record(analysis.reported_depth, analysis.nodes, analysis.elapsed);
            match analysis.bestmove.as_str() {
                "resign" => return make_vs_gikou_result(VsGikouOutcome::BinboWin, "gikou_resign", plies, opening, started, a_stats, g_stats),
                "win" => return make_vs_gikou_result(VsGikouOutcome::BinboLoss, "gikou_win_declaration", plies, opening, started, a_stats, g_stats),
                bestmove => match find_legal_move(&pos, bestmove) {
                    Some(mv) => mv,
                    None => return make_vs_gikou_result(VsGikouOutcome::Error, &compact_reason(&format!("gikou_illegal_bestmove:{bestmove}")), plies, opening, started, a_stats, g_stats),
                },
            }
        };
        pos.do_move(mv);
        plies += 1;
        let count = repetitions.entry(pos.key()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 4 {
            return make_vs_gikou_result(VsGikouOutcome::Draw, "repetition_key4", plies, opening, started, a_stats, g_stats);
        }
    }
    make_vs_gikou_result(VsGikouOutcome::Draw, "max_plies", plies, opening, started, a_stats, g_stats)
}

/// One external-engine move with a single restart-and-retry on a hang/EOF; `Err(reason)` on
/// definitive failure (the game is then scored as an error). Shared by both yane-vs-yane sides.
fn usi_bestmove_retry(
    engine: &mut UsiEngine,
    sfen: &str,
    go_command: &str,
    timeout: Duration,
) -> Result<crate::usi_engine::UsiMove, String> {
    match engine.bestmove(sfen, go_command, timeout) {
        Ok(value) => Ok(value),
        Err(err) if err.requires_restart() => {
            engine.restart().map_err(|e| format!("restart_failed:{e}"))?;
            engine
                .bestmove(sfen, go_command, timeout)
                .map_err(|e| format!("retry_failed:{e}"))
        }
        Err(err) => Err(format!("error:{err}")),
    }
}

/// Play one game between two external YaneuraOu engines (A=candidate, B=baseline). The shared
/// binbo `Position` enforces legality/repetition and produces the sfen each engine is queried
/// with. W/D/L is from A's perspective. A's stats land in `binbo_stats`, B's in `gikou_stats`.
#[allow(clippy::too_many_arguments)]
/// One game between two USI engines, each with its own `go` command and timeout (Task 108:
/// e.g. the candidate on `byoyomi` or `nodes N`, a node-limited anchor on `nodes M`).
#[allow(clippy::too_many_arguments)]
fn play_yane_pair_game(
    engine_a: &mut UsiEngine,
    engine_b: &mut UsiEngine,
    a_is_black: bool,
    opening_seed: u64,
    opening_plies: usize,
    max_plies: usize,
    go_a: (&str, Duration),
    go_b: (&str, Duration),
) -> VsGikouGameResult {
    let started = Instant::now();
    let mut pos = initial_position();
    let opening = apply_random_opening(&mut pos, opening_plies, opening_seed);
    let mut a_stats = SearchSideStats::default();
    let mut b_stats = SearchSideStats::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(pos.key(), 1);

    let mut plies = opening.len();

    while plies < max_plies {
        let a_to_move = (pos.side_to_move() == Color::Black) == a_is_black;
        if pos.legal_moves().is_empty() {
            // Side to move is checkmated (shogi has no stalemate draw): that side loses.
            return make_vs_gikou_result(
                if a_to_move { VsGikouOutcome::BinboLoss } else { VsGikouOutcome::BinboWin },
                "no_legal_moves",
                plies,
                opening,
                started,
                a_stats,
                b_stats,
            );
        }

        let sfen = position_to_sfen(&pos);
        let analysis = if a_to_move {
            usi_bestmove_retry(engine_a, &sfen, go_a.0, go_a.1)
        } else {
            usi_bestmove_retry(engine_b, &sfen, go_b.0, go_b.1)
        };
        let who = if a_to_move { "cand" } else { "base" };
        let analysis = match analysis {
            Ok(value) => value,
            Err(reason) => {
                return make_vs_gikou_result(
                    VsGikouOutcome::Error,
                    &compact_reason(&format!("{who}_{reason}")),
                    plies,
                    opening,
                    started,
                    a_stats,
                    b_stats,
                );
            }
        };
        if a_to_move {
            a_stats.record(analysis.depth, analysis.nodes, analysis.elapsed);
        } else {
            b_stats.record(analysis.depth, analysis.nodes, analysis.elapsed);
        }

        let mv = match analysis.bestmove.as_str() {
            // The engine to move concedes -> that side loses (mapped to A's perspective).
            "resign" => {
                return make_vs_gikou_result(
                    if a_to_move { VsGikouOutcome::BinboLoss } else { VsGikouOutcome::BinboWin },
                    if a_to_move { "cand_resign" } else { "base_resign" },
                    plies,
                    opening,
                    started,
                    a_stats,
                    b_stats,
                );
            }
            // Declaration of victory (nyugyoku) -> that side wins.
            "win" => {
                return make_vs_gikou_result(
                    if a_to_move { VsGikouOutcome::BinboWin } else { VsGikouOutcome::BinboLoss },
                    if a_to_move { "cand_win_declaration" } else { "base_win_declaration" },
                    plies,
                    opening,
                    started,
                    a_stats,
                    b_stats,
                );
            }
            bestmove => match find_legal_move(&pos, bestmove) {
                Some(mv) => mv,
                None => {
                    return make_vs_gikou_result(
                        VsGikouOutcome::Error,
                        &compact_reason(&format!("{who}_illegal_bestmove:{bestmove}")),
                        plies,
                        opening,
                        started,
                        a_stats,
                        b_stats,
                    );
                }
            },
        };

        pos.do_move(mv);
        plies += 1;
        let count = repetitions.entry(pos.key()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 4 {
            return make_vs_gikou_result(
                VsGikouOutcome::Draw,
                "repetition_key4",
                plies,
                opening,
                started,
                a_stats,
                b_stats,
            );
        }
    }

    make_vs_gikou_result(
        VsGikouOutcome::Draw,
        "max_plies",
        plies,
        opening,
        started,
        a_stats,
        b_stats,
    )
}

fn apply_random_opening(pos: &mut Position, opening_plies: usize, seed: u64) -> Vec<String> {
    let mut rng = SplitMix64::new(seed);
    let mut opening = Vec::new();
    for _ in 0..opening_plies {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            break;
        }
        let mv = moves[rng.gen_index(moves.len())];
        opening.push(mv.to_usi_owned());
        pos.do_move(mv);
    }
    opening
}

fn compact_reason(reason: &str) -> String {
    reason
        .chars()
        .map(|ch| if ch.is_whitespace() { '_' } else { ch })
        .collect()
}

fn fmt_avg(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "-".to_string(),
    }
}

/// Play NNUE(depth_a) vs NNUE(depth_b) and report engine A's tally (stage C:
/// evidence that deeper search is stronger).
///
/// Both engines are deterministic, so games are diversified with a random
/// opening of `opening_plies` moves. Each distinct opening is played twice with
/// swapped colors (game pair `2k`, `2k+1`), so opening luck and color are
/// balanced.
fn run_nnue_selfmatch(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-selfmatch <weights_bin> <depth_a> <depth_b> <games> [seed] [max_plies] [opening_plies]"
            .to_string()
    })?;
    let depth_a: u16 = args
        .get(1)
        .ok_or_else(|| "missing <depth_a>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth_a: {err}"))?;
    let depth_b: u16 = args
        .get(2)
        .ok_or_else(|| "missing <depth_b>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth_b: {err}"))?;
    let games: u64 = args
        .get(3)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(6) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 8,
    };

    let mut engine_a = NnueEngine::load(weights, depth_a)?;
    let mut engine_b = NnueEngine::load(weights, depth_b)?;
    let limits = GoLimits::default();
    let (mut a_wins, mut b_wins, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        // Engine A is Black on even games, White on odd (fair color split).
        let a_is_black = game % 2 == 0;
        // Games 2k and 2k+1 share the same random opening (colors swapped).
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut pos = initial_position();
        let mut opening = SplitMix64::new(opening_seed);
        for _ in 0..opening_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.do_move(moves[opening.gen_index(moves.len())]);
        }
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let a_to_move = (pos.side_to_move() == Color::Black) == a_is_black;
            let bestmove = if a_to_move {
                engine_a.think(&pos, &limits).bestmove
            } else {
                engine_b.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    outcome = if a_to_move { "b_win" } else { "a_win" };
                    break;
                }
            }
        }
        match outcome {
            "a_win" => a_wins += 1,
            "b_win" => b_wins += 1,
            _ => draws += 1,
        }
    }

    let decisive = a_wins + b_wins;
    let a_rate = if games > 0 {
        100.0 * a_wins as f64 / games as f64
    } else {
        0.0
    };
    let a_decisive = if decisive > 0 {
        100.0 * a_wins as f64 / decisive as f64
    } else {
        0.0
    };
    println!("nnue-selfmatch weights={weights} depth_a={depth_a} depth_b={depth_b} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies}");
    println!("a(depth{depth_a})_wins={a_wins} b(depth{depth_b})_wins={b_wins} draws={draws}");
    println!("a_win_rate={a_rate:.1}% (of all games)");
    println!("a_win_rate_decisive={a_decisive:.1}% (of {decisive} decided games)");
    Ok(())
}

/// Play NNUE-with-TT vs NNUE-without-TT at the *same* depth, with random
/// openings, and report the TT side's tally. The quality check for the TT
/// (Task 023): reusing deeper results must not make it weaker, so the TT side
/// must not lose to the no-TT side at equal depth.
fn run_nnue_ttmatch(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-ttmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]"
            .to_string()
    })?;
    let depth: u16 = args
        .get(1)
        .ok_or_else(|| "missing <depth>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth: {err}"))?;
    let games: u64 = args
        .get(2)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 8,
    };

    let mut tt_engine = NnueEngine::load(weights, depth)?;
    let mut no_tt_engine = NnueEngine::load(weights, depth)?;
    no_tt_engine.use_tt = false;
    let limits = GoLimits::default();
    let (mut tt_wins, mut no_tt_wins, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        // TT side is Black on even games, White on odd; games 2k/2k+1 share an
        // opening with colors swapped.
        let tt_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut pos = initial_position();
        let mut opening = SplitMix64::new(opening_seed);
        for _ in 0..opening_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.do_move(moves[opening.gen_index(moves.len())]);
        }
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let tt_to_move = (pos.side_to_move() == Color::Black) == tt_is_black;
            let bestmove = if tt_to_move {
                tt_engine.think(&pos, &limits).bestmove
            } else {
                no_tt_engine.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    outcome = if tt_to_move { "no_tt_win" } else { "tt_win" };
                    break;
                }
            }
        }
        match outcome {
            "tt_win" => tt_wins += 1,
            "no_tt_win" => no_tt_wins += 1,
            _ => draws += 1,
        }
    }

    let decisive = tt_wins + no_tt_wins;
    let tt_rate = if games > 0 {
        100.0 * tt_wins as f64 / games as f64
    } else {
        0.0
    };
    println!("nnue-ttmatch weights={weights} depth={depth} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies}");
    println!("tt_wins={tt_wins} no_tt_wins={no_tt_wins} draws={draws}");
    println!("tt_win_rate={tt_rate:.1}% (of all games; >=50% means TT is not weaker)");
    println!("tt_vs_no_tt_decisive={tt_wins}-{no_tt_wins} (of {decisive} decided)");
    Ok(())
}

/// Play iterative-deepening(depth) vs fixed-depth(depth), both with TT, at the
/// same depth with random openings, and report the iddfs side's tally. The
/// quality check for iterative deepening (Task 024): it reads to the same final
/// depth, so it must not lose to the fixed-depth search.
fn run_nnue_idmatch(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-idmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]"
            .to_string()
    })?;
    let depth: u16 = args
        .get(1)
        .ok_or_else(|| "missing <depth>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth: {err}"))?;
    let games: u64 = args
        .get(2)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 8,
    };

    let mut idd_engine = NnueEngine::load(weights, depth)?;
    let mut fixed_engine = NnueEngine::load(weights, depth)?;
    fixed_engine.use_iddfs = false;
    let limits = GoLimits::default();
    let (mut idd_wins, mut fixed_wins, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        let idd_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut pos = initial_position();
        let mut opening = SplitMix64::new(opening_seed);
        for _ in 0..opening_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.do_move(moves[opening.gen_index(moves.len())]);
        }
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let idd_to_move = (pos.side_to_move() == Color::Black) == idd_is_black;
            let bestmove = if idd_to_move {
                idd_engine.think(&pos, &limits).bestmove
            } else {
                fixed_engine.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    outcome = if idd_to_move { "fixed_win" } else { "idd_win" };
                    break;
                }
            }
        }
        match outcome {
            "idd_win" => idd_wins += 1,
            "fixed_win" => fixed_wins += 1,
            _ => draws += 1,
        }
    }

    let decisive = idd_wins + fixed_wins;
    let idd_rate = if games > 0 {
        100.0 * idd_wins as f64 / games as f64
    } else {
        0.0
    };
    println!("nnue-idmatch weights={weights} depth={depth} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies}");
    println!("iddfs_wins={idd_wins} fixed_wins={fixed_wins} draws={draws}");
    println!("iddfs_win_rate={idd_rate:.1}% (of all games; >=50% means iddfs is not weaker)");
    println!("iddfs_vs_fixed_decisive={idd_wins}-{fixed_wins} (of {decisive} decided)");
    Ok(())
}

/// Play hash-move-ordering ON vs OFF (both iterative deepening + TT) at the same
/// depth with random openings, and report the ordering side's tally. Quality
/// check for hash-move ordering (Task 025): it only reorders the search, so it
/// must not lose to the no-ordering search.
fn run_nnue_ordmatch(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-ordmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]"
            .to_string()
    })?;
    let depth: u16 = args
        .get(1)
        .ok_or_else(|| "missing <depth>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth: {err}"))?;
    let games: u64 = args
        .get(2)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 8,
    };

    let mut ord_engine = NnueEngine::load(weights, depth)?;
    let mut noord_engine = NnueEngine::load(weights, depth)?;
    noord_engine.use_ordering = false;
    let limits = GoLimits::default();
    let (mut ord_wins, mut noord_wins, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        let ord_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut pos = initial_position();
        let mut opening = SplitMix64::new(opening_seed);
        for _ in 0..opening_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.do_move(moves[opening.gen_index(moves.len())]);
        }
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let ord_to_move = (pos.side_to_move() == Color::Black) == ord_is_black;
            let bestmove = if ord_to_move {
                ord_engine.think(&pos, &limits).bestmove
            } else {
                noord_engine.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    outcome = if ord_to_move { "noord_win" } else { "ord_win" };
                    break;
                }
            }
        }
        match outcome {
            "ord_win" => ord_wins += 1,
            "noord_win" => noord_wins += 1,
            _ => draws += 1,
        }
    }

    let decisive = ord_wins + noord_wins;
    let ord_rate = if games > 0 {
        100.0 * ord_wins as f64 / games as f64
    } else {
        0.0
    };
    println!("nnue-ordmatch weights={weights} depth={depth} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies}");
    println!("ord_wins={ord_wins} noord_wins={noord_wins} draws={draws}");
    println!("ord_win_rate={ord_rate:.1}% (of all games; >=50% means ordering is not weaker)");
    println!("ord_vs_noord_decisive={ord_wins}-{noord_wins} (of {decisive} decided)");
    Ok(())
}

/// Play MVV-LVA ordering ON vs OFF (both iterative deepening + TT + hash-move
/// ordering) at the same depth with random openings, and report the MVV side's
/// tally. Quality check for MVV-LVA (Task 028): it only reorders the search, so
/// it must not lose to the no-MVV search.
fn run_nnue_mvvmatch(args: &[String]) -> Result<(), String> {
    let weights = args.first().ok_or_else(|| {
        "usage: nnue-mvvmatch <weights_bin> <depth> <games> [seed] [max_plies] [opening_plies]"
            .to_string()
    })?;
    let depth: u16 = args
        .get(1)
        .ok_or_else(|| "missing <depth>".to_string())?
        .parse()
        .map_err(|err| format!("invalid depth: {err}"))?;
    let games: u64 = args
        .get(2)
        .ok_or_else(|| "missing <games>".to_string())?
        .parse()
        .map_err(|err| format!("invalid games: {err}"))?;
    let seed: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260621,
    };
    let max_plies: usize = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 320,
    };
    let opening_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 8,
    };

    let mut mvv_engine = NnueEngine::load(weights, depth)?;
    let mut nomvv_engine = NnueEngine::load(weights, depth)?;
    nomvv_engine.use_mvv_lva = false;
    let limits = GoLimits::default();
    let (mut mvv_wins, mut nomvv_wins, mut draws) = (0u64, 0u64, 0u64);

    for game in 0..games {
        let mvv_is_black = game % 2 == 0;
        let opening_seed = seed.wrapping_add(game / 2).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut pos = initial_position();
        let mut opening = SplitMix64::new(opening_seed);
        for _ in 0..opening_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.do_move(moves[opening.gen_index(moves.len())]);
        }
        let mut outcome = "draw";

        for _ply in 1..=max_plies {
            let mvv_to_move = (pos.side_to_move() == Color::Black) == mvv_is_black;
            let bestmove = if mvv_to_move {
                mvv_engine.think(&pos, &limits).bestmove
            } else {
                nomvv_engine.think(&pos, &limits).bestmove
            };
            match bestmove {
                BestMove::Move(mv) => pos.do_move(mv),
                BestMove::Resign => {
                    outcome = if mvv_to_move { "nomvv_win" } else { "mvv_win" };
                    break;
                }
            }
        }
        match outcome {
            "mvv_win" => mvv_wins += 1,
            "nomvv_win" => nomvv_wins += 1,
            _ => draws += 1,
        }
    }

    let decisive = mvv_wins + nomvv_wins;
    let mvv_rate = if games > 0 {
        100.0 * mvv_wins as f64 / games as f64
    } else {
        0.0
    };
    println!("nnue-mvvmatch weights={weights} depth={depth} games={games} seed={seed} max_plies={max_plies} opening_plies={opening_plies}");
    println!("mvv_wins={mvv_wins} nomvv_wins={nomvv_wins} draws={draws}");
    println!("mvv_win_rate={mvv_rate:.1}% (of all games; >=50% means MVV-LVA is not weaker)");
    println!("mvv_vs_nomvv_decisive={mvv_wins}-{nomvv_wins} (of {decisive} decided)");
    Ok(())
}

/// Benchmark search on a few built-in positions: per-depth nodes/time for
/// iterative-deepening+TT with hash-move ordering only vs + MVV-LVA (same
/// position+depth), the reduction, best move and score.
fn run_nnue_bench(args: &[String]) -> Result<(), String> {
    let weights = args
        .first()
        .ok_or_else(|| "usage: nnue-bench <weights_bin> [max_depth]".to_string())?;
    let max_depth: u16 = match args.get(1) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_depth: {err}"))?,
        None => 4,
    };

    let positions: [(&str, &str); 3] = [
        ("startpos", INITIAL_SFEN),
        (
            "midgame",
            "l3+S3l/5+R3/p7p/1p+B3p2/2K2p1g1/P1p4sP/1P1PPPN2/6S+nk/L4GP1+r b GS2NL3Pbg2p 1",
        ),
        ("endgame", "4k4/9/4g4/9/9/9/4R4/9/4K4 b - 1"),
    ];

    let net = eval::nnue::Nnue::load(weights)?;
    let mut tt = search::Tt::new();
    // Both use iterative deepening + TT + hash-move ordering; compare adding
    // Baseline = hash-move ordering only (Task 025), all Task 028/049 search
    // features off; compared against the full modern search (MVV-LVA + quiescence
    // + killers + history + check extension).
    let hash_only = search::Opts {
        use_mvv_lva: false,
        use_quiescence: false,
        use_killers: false,
        use_history: false,
        use_check_ext: false,
        use_pvs: false,
        use_lmr: false,
        ..search::Opts::full()
    };
    let hash_mvv = search::Opts::full();
    println!("nnue-bench weights={weights} max_depth={max_depth}");
    println!("(iddfs + TT: hash-move ordering only (Task 025) vs full search (Task 028/049))");

    for (name, sfen) in positions {
        let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
            .map_err(|err| format!("invalid SFEN '{sfen}': {err:?}"))?;
        let pos = Position::new(partial);
        println!("\n== {name} : {sfen}");
        for depth in 1..=max_depth {
            // Hash-move ordering only (Task 025 baseline).
            let t0 = Instant::now();
            let off = search::search_iddfs(&net, &pos, depth, hash_only, &mut tt);
            let ms0 = t0.elapsed().as_secs_f64() * 1000.0;

            // Hash-move + MVV-LVA ordering (Task 028).
            let t1 = Instant::now();
            let on = search::search_iddfs(&net, &pos, depth, hash_mvv, &mut tt);
            let ms1 = t1.elapsed().as_secs_f64() * 1000.0;

            // Reduction from turning ordering on.
            let node_red = if off.nodes > 0 {
                100.0 * (1.0 - on.nodes as f64 / off.nodes as f64)
            } else {
                0.0
            };
            let time_red = if ms0 > 0.0 {
                100.0 * (1.0 - ms1 / ms0)
            } else {
                0.0
            };
            let off_best = off
                .best_move
                .map(|mv| mv.to_usi_owned())
                .unwrap_or_else(|| "resign".to_string());
            let on_best = on
                .best_move
                .map(|mv| mv.to_usi_owned())
                .unwrap_or_else(|| "resign".to_string());
            // Correctness: ordering only changes search order, not the result.
            let mismatch = if off.score == on.score && off_best == on_best {
                String::new()
            } else {
                format!(
                    " MISMATCH(off best={off_best} score={}, on best={on_best} score={})",
                    off.score, on.score
                )
            };
            println!(
                "  depth={depth} nodes {} -> {} ({:.1}%) time_ms {:.1} -> {:.1} ({:.1}%) best={on_best} score={}{}",
                off.nodes, on.nodes, node_red, ms0, ms1, time_red, on.score, mismatch
            );
        }
    }
    println!("\nnnue_bench=ok");
    Ok(())
}

/// One diagnosed position: student NNUE (static + time-bounded search with PV)
/// next to Gikou's depth-1 and depth-N evaluations, for the Task 031 diagnosis.
struct DiagRow {
    context: String,
    side: char,
    student_static_cp: i32,
    student_score: i32,
    student_depth: u16,
    student_nodes: u64,
    student_best: String,
    student_pv: String,
    g1_score: String,
    g1_best: String,
    gn_score: String,
    gn_best: String,
    gn_raw: Option<GikouScore>,
}

impl DiagRow {
    /// Sign of the student's search score (side-to-move view): + good, - bad.
    fn student_sign(&self) -> i32 {
        self.student_score.signum()
    }

    /// True when student and Gikou-depth-N agree on who is better (sign match, or
    /// either is ~0). None when Gikou's eval is missing.
    fn sign_agrees(&self) -> Option<bool> {
        let gn = self.gn_raw?;
        let gs = gikou_score_sign(gn);
        let ss = self.student_sign();
        Some(gs == 0 || ss == 0 || gs == ss)
    }

    /// True when the student's chosen move equals Gikou's depth-N best move.
    fn best_matches_gn(&self) -> bool {
        self.gn_best == self.student_best && self.gn_best != "ERR"
    }
}

fn gikou_score_str(score: GikouScore) -> String {
    match score {
        GikouScore::Cp(value) => format!("cp{value}"),
        GikouScore::Mate(value) => format!("mate{value}"),
    }
}

fn gikou_score_sign(score: GikouScore) -> i32 {
    match score {
        GikouScore::Cp(value) => value.signum(),
        GikouScore::Mate(value) => value.signum(),
    }
}

/// Evaluate one position with Gikou to `depth`, restarting once on a transient
/// failure. Returns `None` if Gikou cannot produce a score.
fn gikou_eval_raw(
    gikou: &mut GikouEngine,
    sfen: &str,
    depth: usize,
    timeout: Duration,
) -> Option<GikouEval> {
    match gikou.eval_sfen(sfen, depth, timeout) {
        Ok(eval) => Some(eval),
        Err(err) if err.requires_restart() => {
            let _ = gikou.restart();
            gikou.eval_sfen(sfen, depth, timeout).ok()
        }
        Err(_) => None,
    }
}

/// Build a `DiagRow` for `pos`: student static eval + a time-bounded search (with
/// PV), and Gikou's depth-1 and depth-N evaluations of the same position.
#[allow(clippy::too_many_arguments)]
fn diag_analyze(
    net: &eval::nnue::Nnue,
    gikou: &mut GikouEngine,
    tt: &mut search::Tt,
    pos: &Position,
    context: &str,
    student_ms: u64,
    gikou_depth: usize,
    gikou_timeout: Duration,
) -> DiagRow {
    let sfen = position_to_sfen(pos);
    let side = match pos.side_to_move() {
        Color::Black => 'b',
        Color::White => 'w',
    };

    let student_static_cp = search::eval_to_int(net.forward(pos));
    let mut noop = |_: &search::IterInfo| {};
    let res = search::search_iddfs_timed(
        net,
        pos,
        16,
        search::Opts::full(),
        tt,
        Some(Duration::from_millis(student_ms)),
        &mut noop,
    );
    let student_best = res
        .best_move
        .map(|mv| mv.to_usi_owned())
        .unwrap_or_else(|| "resign".to_string());
    let student_pv = format_pv(&res.pv);

    let g1 = gikou_eval_raw(gikou, &sfen, 1, gikou_timeout);
    let gn = gikou_eval_raw(gikou, &sfen, gikou_depth, gikou_timeout);
    let to_score = |e: &Option<GikouEval>| {
        e.as_ref()
            .map(|v| gikou_score_str(v.score))
            .unwrap_or_else(|| "ERR".to_string())
    };
    let to_best = |e: &Option<GikouEval>| {
        e.as_ref()
            .map(|v| v.bestmove.clone())
            .unwrap_or_else(|| "ERR".to_string())
    };

    DiagRow {
        context: context.to_string(),
        side,
        student_static_cp,
        student_score: res.score,
        student_depth: res.depth,
        student_nodes: res.nodes,
        student_best,
        student_pv,
        g1_score: to_score(&g1),
        g1_best: to_best(&g1),
        gn_score: to_score(&gn),
        gn_best: to_best(&gn),
        gn_raw: gn.map(|e| e.score),
    }
}

fn print_diag_header(gikou_depth: usize) {
    println!(
        "ctx\tside\tst_static\tst_score\tst_d\tst_nodes\tst_best\tst_pv | g1_score\tg1_best | g{gikou_depth}_score\tg{gikou_depth}_best\tsignOK\tbest=gN"
    );
}

fn print_diag_row(row: &DiagRow) {
    let sign_ok = match row.sign_agrees() {
        Some(true) => "yes",
        Some(false) => "NO",
        None => "-",
    };
    let best_eq = if row.best_matches_gn() { "yes" } else { "no" };
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{} | {}\t{} | {}\t{}\t{}\t{}",
        row.context,
        row.side,
        row.student_static_cp,
        row.student_score,
        row.student_depth,
        row.student_nodes,
        row.student_best,
        row.student_pv,
        row.g1_score,
        row.g1_best,
        row.gn_score,
        row.gn_best,
        sign_ok,
        best_eq,
    );
}

/// Task 031 evaluation diagnosis: compare the student NNUE to Gikou (depth 1 and
/// depth N) on (1) a curated set of representative positions, then (2) a real
/// student(black) vs Gikou(depth 1) game, dumping the student's PV/score and
/// Gikou's eval at every student move so the losing line can be read off.
fn run_nnue_diag(args: &[String]) -> Result<(), String> {
    let usage =
        "usage: nnue-diag <weights_bin> <gikou_path> [seed] [student_ms] [gikou_depth] [opening_plies] [max_plies]";
    let weights = args.first().ok_or_else(|| usage.to_string())?;
    let gikou_path = args.get(1).ok_or_else(|| usage.to_string())?;
    let seed: u64 = match args.get(2) {
        Some(value) => value.parse().map_err(|err| format!("invalid seed: {err}"))?,
        None => 20260623,
    };
    let student_ms: u64 = match args.get(3) {
        Some(value) => value.parse().map_err(|err| format!("invalid student_ms: {err}"))?,
        None => 2000,
    };
    let gikou_depth: usize = match args.get(4) {
        Some(value) => value.parse().map_err(|err| format!("invalid gikou_depth: {err}"))?,
        None => 8,
    };
    let opening_plies: usize = match args.get(5) {
        Some(value) => value.parse().map_err(|err| format!("invalid opening_plies: {err}"))?,
        None => 6,
    };
    let max_plies: usize = match args.get(6) {
        Some(value) => value.parse().map_err(|err| format!("invalid max_plies: {err}"))?,
        None => 80,
    };

    let net = eval::nnue::Nnue::load(weights)?;
    let mut tt = search::Tt::new();
    let gikou_timeout = Duration::from_secs(60);
    let mut gikou = GikouEngine::start(Path::new(gikou_path), false, 1)
        .map_err(|err| format!("failed to start Gikou: {err}"))?;

    println!("nnue-diag weights={weights} gikou_path={gikou_path} seed={seed} student_ms={student_ms} gikou_depth={gikou_depth} opening_plies={opening_plies} max_plies={max_plies}");
    println!("note: student score = logit*600 (side-to-move cp-ish); Gikou score is side-to-move cp/mate. Compare sign and trend, not exact scale.");

    // (1) Curated representative positions (opening / midgame / endgame / clearly
    // decided material), to check whether the student gets the sign and the
    // magnitude trend right across position types.
    println!("\n=== B-1: representative positions (student static+search vs Gikou d1/d{gikou_depth}) ===");
    let curated: [(&str, &str); 6] = [
        ("open_startpos", INITIAL_SFEN),
        (
            "open_quiet",
            "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
        ),
        (
            "mid_tactical",
            "l3+S3l/5+R3/p7p/1p+B3p2/2K2p1g1/P1p4sP/1P1PPPN2/6S+nk/L4GP1+r b GS2NL3Pbg2p 1",
        ),
        ("end_bench", "4k4/9/4g4/9/9/9/4R4/9/4K4 b - 1"),
        (
            "black_up_rook",
            "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b R 1",
        ),
        (
            "black_down_rook",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B7/LNSGKGSNL b r 1",
        ),
    ];
    print_diag_header(gikou_depth);
    for (label, sfen) in curated {
        let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
            .map_err(|err| format!("invalid curated SFEN '{sfen}': {err:?}"))?;
        let pos = Position::new(partial);
        let row = diag_analyze(
            &net, &mut gikou, &mut tt, &pos, label, student_ms, gikou_depth, gikou_timeout,
        );
        print_diag_row(&row);
    }

    // (2) A real game: student is Black, Gikou plays its depth-1 move (≈ Lv1).
    // At every student move we diagnose the position, so the losing line is
    // visible move by move (student PV/score vs Gikou's strong eval).
    println!("\n=== B-2: student(black) vs Gikou(depth1) game, diagnosing every student move ===");
    let mut pos = initial_position();
    let opening = apply_random_opening(&mut pos, opening_plies, seed);
    println!("opening={}", opening.join(" "));
    print_diag_header(gikou_depth);

    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(pos.key(), 1);
    let mut plies = opening.len();
    let mut sign_agree = 0_u32;
    let mut sign_total = 0_u32;
    let mut best_agree = 0_u32;
    let mut student_moves = 0_u32;
    let mut outcome = "max_plies";
    let go1_timeout = Duration::from_secs(20);

    while plies < max_plies {
        if pos.legal_moves().is_empty() {
            // Side to move is checkmated.
            outcome = if pos.side_to_move() == Color::Black {
                "student_mated"
            } else {
                "gikou_mated"
            };
            break;
        }
        let student_to_move = pos.side_to_move() == Color::Black;
        let mv = if student_to_move {
            let ctx = format!("ply{plies}");
            let row = diag_analyze(
                &net, &mut gikou, &mut tt, &pos, &ctx, student_ms, gikou_depth, gikou_timeout,
            );
            print_diag_row(&row);
            student_moves += 1;
            if let Some(agree) = row.sign_agrees() {
                sign_total += 1;
                if agree {
                    sign_agree += 1;
                }
            }
            if row.best_matches_gn() {
                best_agree += 1;
            }
            match row.student_best.as_str() {
                "resign" => {
                    outcome = "student_resign";
                    break;
                }
                best => match find_legal_move(&pos, best) {
                    Some(mv) => mv,
                    None => {
                        outcome = "student_illegal";
                        break;
                    }
                },
            }
        } else {
            let sfen = position_to_sfen(&pos);
            let Some(eval) = gikou_eval_raw(&mut gikou, &sfen, 1, go1_timeout) else {
                outcome = "gikou_error";
                break;
            };
            match eval.bestmove.as_str() {
                "resign" => {
                    outcome = "gikou_resign";
                    break;
                }
                "win" => {
                    outcome = "gikou_win_decl";
                    break;
                }
                best => match find_legal_move(&pos, best) {
                    Some(mv) => mv,
                    None => {
                        outcome = "gikou_illegal";
                        break;
                    }
                },
            }
        };

        pos.do_move(mv);
        plies += 1;
        let count = repetitions.entry(pos.key()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 4 {
            outcome = "repetition";
            break;
        }
    }

    println!("\n=== summary ===");
    println!("outcome={outcome} plies={plies}");
    println!("student_moves_diagnosed={student_moves}");
    println!(
        "sign_agreement={sign_agree}/{sign_total} (student vs Gikou-d{gikou_depth} agree on who is better)"
    );
    println!(
        "student_best_equals_gikou_d{gikou_depth}_best={best_agree}/{student_moves}"
    );
    println!("nnue_diag=ok");
    Ok(())
}

fn run_gen_teacher_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(usage());
    }
    let gikou_path = PathBuf::from(&args[0]);
    let out_dir = PathBuf::from(&args[1]);
    let target_positions =
        parse_usize(&args[2]).map_err(|err| format!("invalid target: {err}"))?;

    let mut shard_size = 10_000_usize;
    let mut workers = 4_usize;
    let mut depth = 8_usize;
    let mut random_plies = 6_usize;
    let mut max_plies = 256_usize;
    let mut base_seed = 20_260_621_u64;
    let mut timeout_ms = 30_000_u64;
    let mut own_book = false;
    let mut max_shard_retries = 3_usize;

    let mut index = 3;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--shard-size" => {
                shard_size = parse_usize(value).map_err(|err| format!("invalid shard-size: {err}"))?
            }
            "--workers" => {
                workers = parse_usize(value).map_err(|err| format!("invalid workers: {err}"))?
            }
            "--depth" => depth = parse_usize(value).map_err(|err| format!("invalid depth: {err}"))?,
            "--random-plies" => {
                random_plies =
                    parse_usize(value).map_err(|err| format!("invalid random-plies: {err}"))?
            }
            "--max-plies" => {
                max_plies = parse_usize(value).map_err(|err| format!("invalid max-plies: {err}"))?
            }
            "--seed" => {
                base_seed = parse_u64(value).map_err(|err| format!("invalid seed: {err}"))?
            }
            "--timeout-ms" => {
                timeout_ms = parse_u64(value).map_err(|err| format!("invalid timeout-ms: {err}"))?
            }
            "--own-book" => own_book = parse_bool(value)?,
            "--max-retries" => {
                max_shard_retries =
                    parse_usize(value).map_err(|err| format!("invalid max-retries: {err}"))?
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        index += 2;
    }

    let config = GenConfig {
        gikou_path,
        out_dir,
        target_positions,
        shard_size,
        workers,
        depth,
        random_plies,
        max_plies,
        base_seed,
        timeout: Duration::from_millis(timeout_ms),
        own_book,
        max_shard_retries,
    };
    gen::run_gen_teacher(config)
}

/// Re-run Gikou on a deterministic sample of already-generated positions and
/// confirm the stored eval (turn-side cp) matches a fresh analysis. This
/// re-verifies both the手番側視点 of stored evals and Gikou determinism.
fn run_gikou_recheck_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(usage());
    }
    let gikou_path = &args[0];
    let expected_tsv = &args[1];
    let sample_n = parse_usize(&args[2]).map_err(|err| format!("invalid sample_n: {err}"))?;
    let depth = args
        .get(3)
        .map(|value| parse_usize(value))
        .transpose()
        .map_err(|err| format!("invalid depth: {err}"))?
        .unwrap_or(8);
    let seed = args
        .get(4)
        .map(|value| parse_u64(value))
        .transpose()
        .map_err(|err| format!("invalid seed: {err}"))?
        .unwrap_or(1);
    let timeout_ms = args
        .get(5)
        .map(|value| parse_u64(value))
        .transpose()
        .map_err(|err| format!("invalid timeout_ms: {err}"))?
        .unwrap_or(30_000);
    let timeout = Duration::from_millis(timeout_ms);

    let text = fs::read_to_string(expected_tsv)
        .map_err(|err| format!("failed to read {expected_tsv}: {err}"))?;
    let mut rows: Vec<(String, i16)> = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line_no == 0 {
            continue; // header
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        // columns: index name sfen_core eval bestmove_usi bestMove16 gameResult dummy hcp_hex
        if fields.len() < 4 {
            continue;
        }
        let sfen_core = fields[2].to_string();
        let eval = fields[3]
            .parse::<i16>()
            .map_err(|err| format!("invalid eval on line {}: {err}", line_no + 1))?;
        rows.push((sfen_core, eval));
    }
    if rows.is_empty() {
        return Err("no rows to recheck".to_string());
    }

    let sample_n = sample_n.min(rows.len());
    let mut rng = SplitMix64::new(seed);
    let mut picked = Vec::with_capacity(sample_n);
    let mut chosen = std::collections::BTreeSet::new();
    while chosen.len() < sample_n {
        let idx = rng.gen_index(rows.len());
        if chosen.insert(idx) {
            picked.push(idx);
        }
    }

    let mut gikou = GikouEngine::start(Path::new(gikou_path), false, 1)
        .map_err(|err| format!("failed to start Gikou: {err}"))?;

    // The stored eval was computed with the engine warmed by the whole game and
    // the real ply number; a cold single-position recheck differs in magnitude
    // (and, near zero, occasionally in sign) — that is search noise, not a
    // viewpoint error. So the viewpoint verdict is keyed on *decisive* positions
    // (|stored eval| >= 500cp), where the side-to-move advantage is unambiguous
    // and the sign must always agree. A real per-color sign flip would break
    // these (and ~half of one turn).
    const DECISIVE_CP: i16 = 500;
    let mut exact = 0_usize;
    let mut sign_ok = 0_usize;
    let mut decisive_checked = 0_usize;
    let mut decisive_sign_ok = 0_usize;
    let mut max_abs_diff = 0_i32;
    let mut checked = 0_usize;
    let mut mate_skips = 0_usize;
    let mut shown = 0_usize;
    for &row_index in &picked {
        let (sfen_core, stored_eval) = &rows[row_index];
        let sfen = format!("{sfen_core} 1");
        let analysis = match gikou.analyze_sfen(&sfen, depth, timeout) {
            Ok(analysis) => analysis,
            Err(GikouError::MateScore(_)) => {
                mate_skips += 1;
                continue;
            }
            Err(err) => return Err(format!("recheck analysis failed: {err}")),
        };
        checked += 1;
        let fresh = analysis.eval_cp;
        let diff = (i32::from(fresh) - i32::from(*stored_eval)).abs();
        max_abs_diff = max_abs_diff.max(diff);
        let same_sign = (fresh >= 0) == (*stored_eval >= 0);
        if fresh == *stored_eval {
            exact += 1;
        }
        if same_sign {
            sign_ok += 1;
        }
        if stored_eval.abs() >= DECISIVE_CP {
            decisive_checked += 1;
            if same_sign {
                decisive_sign_ok += 1;
            } else if shown < 10 {
                shown += 1;
                eprintln!(
                    "DECISIVE sign mismatch row={row_index} stored={stored_eval} fresh={fresh} sfen='{sfen_core}'"
                );
            }
        }
    }

    println!("gikou_recheck_source={expected_tsv}");
    println!("rows_total={}", rows.len());
    println!("sample_requested={sample_n}");
    println!("checked={checked}");
    println!("mate_skips={mate_skips}");
    println!("exact_eval_match={exact}");
    println!("sign_match={sign_ok}");
    println!("decisive_checked={decisive_checked}");
    println!("decisive_sign_match={decisive_sign_ok}");
    println!("max_abs_diff={max_abs_diff}");
    drop(gikou);
    if checked == 0 {
        println!("gikou_recheck=no_samples");
    } else if decisive_checked == 0 {
        println!("gikou_recheck=inconclusive_no_decisive_samples");
    } else if decisive_sign_ok == decisive_checked {
        println!("gikou_recheck=ok_viewpoint");
    } else {
        println!("gikou_recheck=VIEWPOINT_MISMATCH");
    }
    Ok(())
}

fn run_hcpe_gikou(
    gikou_path: &str,
    output: &str,
    positions: &str,
    depth: &str,
    random_plies: &str,
    max_plies: &str,
    seed: &str,
    own_book: &str,
) -> Result<(), String> {
    let target_positions =
        parse_usize(positions).map_err(|err| format!("invalid positions: {err}"))?;
    let depth = parse_usize(depth).map_err(|err| format!("invalid depth: {err}"))?;
    let random_plies =
        parse_usize(random_plies).map_err(|err| format!("invalid random_plies: {err}"))?;
    let max_plies = parse_usize(max_plies).map_err(|err| format!("invalid max_plies: {err}"))?;
    let seed = parse_u64(seed).map_err(|err| format!("invalid seed: {err}"))?;
    let own_book = parse_bool(own_book)?;
    if target_positions == 0 {
        return Err("positions must be >= 1".to_string());
    }
    if depth == 0 {
        return Err("depth must be >= 1".to_string());
    }
    if random_plies >= max_plies {
        return Err("random_plies must be smaller than max_plies".to_string());
    }

    let hcpe_path = Path::new(output);
    if let Some(parent) = hcpe_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| format!("failed to create parent dir: {err}"))?;
    }

    let mut gikou = GikouEngine::start(Path::new(gikou_path), own_book, 1)
        .map_err(|err| format!("failed to start Gikou: {err}"))?;
    let file = File::create(hcpe_path)
        .map_err(|err| format!("failed to create {}: {err}", hcpe_path.display()))?;
    let mut writer = BufWriter::new(file);
    let expected_path = expected_path_for(hcpe_path);
    let mut expected = BufWriter::new(
        File::create(&expected_path)
            .map_err(|err| format!("failed to create {}: {err}", expected_path.display()))?,
    );
    writeln!(
        expected,
        "index\tname\tsfen_core\teval\tbestmove_usi\tbestMove16\tgameResult\tdummy\thcp_hex"
    )
    .map_err(|err| format!("failed to write expected header: {err}"))?;

    let started = Instant::now();
    let mut positions_written = 0_usize;
    let mut games = 0_usize;
    let mut rng = SplitMix64::new(seed);
    let mut analysis_elapsed = Duration::ZERO;
    let mut eval_min = i16::MAX;
    let mut eval_max = i16::MIN;
    let mut reported_depth_min = usize::MAX;
    let mut reported_depth_max = 0_usize;
    let mut mate_cutoffs = 0_usize;
    let max_games = target_positions.saturating_mul(10).max(100);

    while positions_written < target_positions {
        if games >= max_games {
            return Err(format!(
                "too many games without reaching target positions: games={games}, positions={positions_written}, target={target_positions}"
            ));
        }
        games += 1;
        let mut pos = initial_position();

        for _ in 0..random_plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = moves[rng.gen_index(moves.len())];
            pos.do_move(mv);
        }

        for ply in random_plies..max_plies {
            if positions_written >= target_positions {
                break;
            }

            let sfen = position_to_sfen(&pos);
            let analysis = match gikou.analyze_sfen(&sfen, depth, GIKOU_ANALYSIS_TIMEOUT) {
                Ok(analysis) => analysis,
                Err(GikouError::MateScore(_)) => {
                    mate_cutoffs += 1;
                    break;
                }
                Err(err) => {
                    return Err(format!(
                        "Gikou analysis failed at game={games} ply={ply}: {err}"
                    ));
                }
            };
            if analysis.bestmove == "resign" || analysis.bestmove == "win" {
                break;
            }
            let best_move = find_legal_move(&pos, &analysis.bestmove).ok_or_else(|| {
                format!(
                    "Gikou returned illegal bestmove at game={games} ply={ply}: {} sfen={sfen}",
                    analysis.bestmove
                )
            })?;
            let entry =
                HcpeEntry::from_position(&pos, analysis.eval_cp, best_move, HcpeGameResult::Draw)
                    .map_err(|err| format!("failed to encode hcpe entry: {err}"))?;

            writer
                .write_all(&entry.encode())
                .map_err(|err| format!("failed to write hcpe entry: {err}"))?;
            writeln!(
                expected,
                "{}\tgikou_g{}_p{}\t{}\t{}\t{}\t{}\t{}\t0\t{}",
                positions_written,
                games,
                ply,
                sfen_core(&sfen),
                analysis.eval_cp,
                analysis.bestmove,
                entry.best_move16,
                HcpeGameResult::Draw as i8,
                hex(&entry.hcp)
            )
            .map_err(|err| format!("failed to write expected row: {err}"))?;

            analysis_elapsed += analysis.elapsed;
            eval_min = eval_min.min(analysis.eval_cp);
            eval_max = eval_max.max(analysis.eval_cp);
            if let Some(reported_depth) = analysis.reported_depth {
                reported_depth_min = reported_depth_min.min(reported_depth);
                reported_depth_max = reported_depth_max.max(reported_depth);
            }
            positions_written += 1;
            pos.do_move(best_move);
        }
    }

    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", hcpe_path.display()))?;
    expected
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", expected_path.display()))?;

    let file_size = fs::metadata(hcpe_path)
        .map_err(|err| format!("failed to stat {}: {err}", hcpe_path.display()))?
        .len();
    let expected_size = (positions_written * HCPE_ENTRY_SIZE) as u64;
    if file_size != expected_size {
        return Err(format!(
            "hcpe file size mismatch: got {file_size}, expected {expected_size}"
        ));
    }

    let elapsed = started.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / positions_written as f64;
    let avg_analysis_ms = analysis_elapsed.as_secs_f64() * 1000.0 / positions_written as f64;
    let estimate_1m_hours = avg_ms * 1_000_000.0 / 1000.0 / 3600.0;
    let estimate_5m_hours = avg_ms * 5_000_000.0 / 1000.0 / 3600.0;

    println!("hcpe_path={}", hcpe_path.display());
    println!("expected_path={}", expected_path.display());
    println!("gikou_path={gikou_path}");
    println!("positions={positions_written}");
    println!("games={games}");
    println!("depth={depth}");
    println!("own_book={own_book}");
    println!("random_plies={random_plies}");
    println!("max_plies={max_plies}");
    println!("seed={seed}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("avg_ms_per_position={avg_ms:.3}");
    println!(
        "analysis_elapsed_ms={:.3}",
        analysis_elapsed.as_secs_f64() * 1000.0
    );
    println!("avg_analysis_ms_per_position={avg_analysis_ms:.3}");
    println!("eval_min={eval_min}");
    println!("eval_max={eval_max}");
    println!("mate_cutoffs={mate_cutoffs}");
    if reported_depth_min != usize::MAX {
        println!("reported_depth_min={reported_depth_min}");
        println!("reported_depth_max={reported_depth_max}");
    }
    println!("hcpe_entry_size={HCPE_ENTRY_SIZE}");
    println!("file_size={file_size}");
    println!("estimate_1m_hours={estimate_1m_hours:.3}");
    println!("estimate_5m_hours={estimate_5m_hours:.3}");
    println!("disk_1m_mb=38.0");
    println!("disk_5m_mb=190.0");
    println!("status=ok");
    Ok(())
}

fn expected_path_for(hcpe_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.expected.tsv", hcpe_path.to_string_lossy()))
}

fn sfen_core(sfen: &str) -> String {
    sfen.split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_raw_usi_move(usi: &str) -> Result<Move, String> {
    let chars = usi.chars().collect::<Vec<_>>();
    if chars.len() == 4 && chars[1] == '*' {
        let piece_kind = drop_piece_kind(chars[0])?;
        let to = parse_usi_square(chars[2], chars[3])?;
        return Ok(Move::Drop {
            piece: Piece::new(piece_kind, Color::Black),
            to,
        });
    }

    if chars.len() == 4 || chars.len() == 5 {
        let from = parse_usi_square(chars[0], chars[1])?;
        let to = parse_usi_square(chars[2], chars[3])?;
        let promote = match chars.get(4) {
            None => false,
            Some('+') => true,
            Some(other) => return Err(format!("invalid promote marker: {other}")),
        };
        return Ok(Move::Normal { from, to, promote });
    }

    Err(format!("invalid USI move: {usi}"))
}

fn find_legal_move(pos: &Position, move_usi: &str) -> Option<Move> {
    pos.legal_moves()
        .into_iter()
        .find(|mv| mv.to_usi_owned() == move_usi)
}

fn position_to_sfen(pos: &Position) -> String {
    let mut board = String::new();
    for rank in 1..=9 {
        if rank > 1 {
            board.push('/');
        }
        let mut empty = 0;
        for file in (1..=9).rev() {
            let square = Square::new(file, rank).expect("valid board square");
            match pos.piece_at(square) {
                None => empty += 1,
                Some(piece) => {
                    if empty > 0 {
                        board.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    board.push_str(&piece.to_usi_owned());
                }
            }
        }
        if empty > 0 {
            board.push(char::from(b'0' + empty));
        }
    }

    format!(
        "{} {} {} {}",
        board,
        pos.side_to_move().to_usi_owned(),
        [pos.hand(Color::Black), pos.hand(Color::White)].to_usi_owned(),
        pos.ply()
    )
}

fn parse_usi_square(file: char, rank: char) -> Result<Square, String> {
    let file = file
        .to_digit(10)
        .filter(|value| (1..=9).contains(value))
        .ok_or_else(|| format!("invalid file: {file}"))? as u8;
    let rank = match rank {
        'a'..='i' => rank as u8 - b'a' + 1,
        _ => return Err(format!("invalid rank: {rank}")),
    };
    Square::new(file, rank).ok_or_else(|| format!("invalid square: {file}{rank}"))
}

fn drop_piece_kind(piece: char) -> Result<PieceKind, String> {
    match piece {
        'P' => Ok(PieceKind::Pawn),
        'L' => Ok(PieceKind::Lance),
        'N' => Ok(PieceKind::Knight),
        'S' => Ok(PieceKind::Silver),
        'G' => Ok(PieceKind::Gold),
        'B' => Ok(PieceKind::Bishop),
        'R' => Ok(PieceKind::Rook),
        _ => Err(format!("invalid drop piece: {piece}")),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let upper = upper as u128;
        let bound = ((1_u128 << 64) / upper) * upper;
        loop {
            let value = self.next_u64() as u128;
            if value < bound {
                return (value % upper) as usize;
            }
        }
    }
}

fn default_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ u64::from(process::id()).rotate_left(32)
}

fn debug_position(pos: &Position) -> io::Result<()> {
    let mut err = io::stderr().lock();
    writeln!(err, "=== board japanese ===")?;
    writeln!(err, "     9    8    7    6    5    4    3    2    1")?;
    for rank in 1..=9 {
        write!(err, "{} ", rank_label(rank))?;
        for file in (1..=9).rev() {
            let square = Square::new(file, rank).expect("valid board square");
            write!(err, "{:>4}", piece_japanese(pos.piece_at(square)))?;
        }
        writeln!(err)?;
    }
    writeln!(err, "手番: {}", color_japanese(pos.side_to_move()))?;
    writeln!(
        err,
        "持ち駒(先手): {}",
        hand_japanese(pos.hand(Color::Black))
    )?;
    writeln!(
        err,
        "持ち駒(後手): {}",
        hand_japanese(pos.hand(Color::White))
    )?;
    writeln!(err, "ply: {}", pos.ply())?;
    writeln!(err, "key: {}", pos.key())?;
    writeln!(err)?;
    writeln!(err, "=== board alphabet ===")?;
    writeln!(err, "    9  8  7  6  5  4  3  2  1")?;
    for rank in 1..=9 {
        write!(err, "{} ", rank_label(rank))?;
        for file in (1..=9).rev() {
            let square = Square::new(file, rank).expect("valid board square");
            write!(err, "{:>3}", piece_alphabet(pos.piece_at(square)))?;
        }
        writeln!(err)?;
    }
    writeln!(err, "side: {}", pos.side_to_move().to_usi_owned())?;
    writeln!(
        err,
        "hands: {}",
        [pos.hand(Color::Black), pos.hand(Color::White)].to_usi_owned()
    )?;
    writeln!(
        err,
        "TODO: 千日手は未実装。同一 pos.key() 4回で引き分け検出する。"
    )?;
    err.flush()
}

fn rank_label(rank: u8) -> char {
    char::from(b'a' + rank - 1)
}

fn piece_japanese(piece: Option<Piece>) -> String {
    match piece {
        None => "・".to_string(),
        Some(piece) => {
            let symbol = piece_kind_japanese(piece.piece_kind());
            match piece.color() {
                Color::Black => format!(" {symbol}"),
                Color::White => format!("v{symbol}"),
            }
        }
    }
}

fn piece_kind_japanese(piece_kind: PieceKind) -> &'static str {
    match piece_kind {
        PieceKind::Pawn => "歩",
        PieceKind::Lance => "香",
        PieceKind::Knight => "桂",
        PieceKind::Silver => "銀",
        PieceKind::Gold => "金",
        PieceKind::Bishop => "角",
        PieceKind::Rook => "飛",
        PieceKind::King => "玉",
        PieceKind::ProPawn => "と",
        PieceKind::ProLance => "成香",
        PieceKind::ProKnight => "成桂",
        PieceKind::ProSilver => "成銀",
        PieceKind::ProBishop => "馬",
        PieceKind::ProRook => "竜",
    }
}

fn piece_alphabet(piece: Option<Piece>) -> String {
    match piece {
        None => ".".to_string(),
        Some(piece) => piece.to_usi_owned(),
    }
}

fn color_japanese(color: Color) -> &'static str {
    match color {
        Color::Black => "先手",
        Color::White => "後手",
    }
}

fn hand_japanese(hand: Hand) -> String {
    let parts = Hand::all_hand_pieces()
        .filter_map(|piece_kind| {
            let count = hand.count(piece_kind).unwrap_or(0);
            if count == 0 {
                None
            } else {
                Some(format!("{}x{count}", piece_kind_japanese(piece_kind)))
            }
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        "なし".to_string()
    } else {
        parts.join(" ")
    }
}

fn run_selfplay(max_plies: usize, seed: u64) -> Result<(), String> {
    let mut pos = initial_position();
    let mut thinker = RandomEngine::new(seed);
    let limits = GoLimits::default();
    let mut moves = Vec::new();
    println!("selfplay: max_plies={max_plies}, seed={seed}");
    println!(
        "TODO: repetition draw is not implemented yet; track pos.key() counts for sennichite."
    );

    for ply in 1..=max_plies {
        let bestmove = match thinker.think(&pos, &limits).bestmove {
            BestMove::Move(mv) => mv,
            BestMove::Resign => {
                println!(
                    "result: resign side={} ply={} moves={}",
                    pos.side_to_move().to_usi_owned(),
                    ply,
                    moves.join(" ")
                );
                return Ok(());
            }
        };
        let move_usi = bestmove.to_usi_owned();
        println!("{ply}: {} {move_usi}", pos.side_to_move().to_usi_owned());
        pos.do_move(bestmove);
        moves.push(move_usi);
    }

    println!(
        "result: draw max_plies={max_plies} moves={}",
        moves.join(" ")
    );
    Ok(())
}

fn print_build_features() {
    println!(
        "build target features: avx2={} bmi2={}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "bmi2")
    );
    println!("yasai: git rev 51c23780109a4ac7258c27d8965640d4f8b78e9e, feature simd");
}

fn initial_position() -> Position {
    Position::new(
        PartialPosition::from_usi(&format!("sfen {INITIAL_SFEN}"))
            .expect("failed to parse initial SFEN"),
    )
}

fn verify() {
    println!("initial sfen: {INITIAL_SFEN}");
    println!("perft verification:");
    println!("depth,nodes,expected,status,elapsed_ms,nps");

    for &(depth, expected) in EXPECTED_PERFT {
        let mut pos = initial_position();
        let initial_key = pos.key();
        let started = Instant::now();
        let nodes = perft(&mut pos, depth);
        let elapsed = started.elapsed();
        let final_key = pos.key();
        let status = if nodes == expected && initial_key == final_key {
            "ok"
        } else {
            "ng"
        };
        println!(
            "{depth},{nodes},{expected},{status},{:.3},{}",
            elapsed.as_secs_f64() * 1000.0,
            nps(nodes, elapsed)
        );
        assert_eq!(expected, nodes, "perft depth {depth} mismatch");
        assert_eq!(
            initial_key, final_key,
            "key changed after perft depth {depth}"
        );
    }
}

fn benchmark(depth: usize, reps: usize) {
    if reps == 0 {
        println!("benchmark skipped: reps=0");
        return;
    }

    println!("benchmark: depth={depth}, reps={reps}");
    let mut runs = Vec::with_capacity(reps);
    for rep in 1..=reps {
        let mut pos = initial_position();
        let initial_key = pos.key();
        let started = Instant::now();
        let nodes = perft(&mut pos, depth);
        let elapsed = started.elapsed();
        assert_eq!(
            initial_key,
            pos.key(),
            "key changed after benchmark run {rep}"
        );
        println!(
            "run {rep}: nodes={nodes}, elapsed_ms={:.3}, nps={}",
            elapsed.as_secs_f64() * 1000.0,
            nps(nodes, elapsed)
        );
        runs.push((elapsed, nodes));
    }

    runs.sort_by_key(|&(elapsed, _)| elapsed);
    let best = runs[0];
    let median = runs[runs.len() / 2];
    let total_elapsed = runs
        .iter()
        .fold(Duration::ZERO, |acc, &(elapsed, _)| acc + elapsed);
    let total_nodes = runs.iter().map(|&(_, nodes)| nodes).sum::<u64>();

    println!(
        "summary: best_ms={:.3}, best_nps={}, median_ms={:.3}, median_nps={}, avg_ms={:.3}, avg_nps={}",
        best.0.as_secs_f64() * 1000.0,
        nps(best.1, best.0),
        median.0.as_secs_f64() * 1000.0,
        nps(median.1, median.0),
        total_elapsed.as_secs_f64() * 1000.0 / reps as f64,
        nps(total_nodes, total_elapsed)
    );
}

fn print_single_result(depth: usize, nodes: u64, elapsed: Duration) {
    println!(
        "perft {depth}: nodes={nodes}, elapsed_ms={:.3}, nps={}",
        elapsed.as_secs_f64() * 1000.0,
        nps(nodes, elapsed)
    );
}

fn nps(nodes: u64, elapsed: Duration) -> u128 {
    let nanos = elapsed.as_nanos();
    if nanos == 0 {
        0
    } else {
        u128::from(nodes) * 1_000_000_000 / nanos
    }
}

fn perft(pos: &mut Position, depth: usize) -> u64 {
    match depth {
        0 => 1,
        1 => pos.legal_moves().len() as u64,
        _ => {
            let mut nodes = 0;
            for mv in pos.legal_moves() {
                pos.do_move(mv);
                nodes += perft(pos, depth - 1);
                pos.undo_move(mv);
            }
            nodes
        }
    }
}

fn divide(pos: &mut Position, depth: usize) -> u64 {
    let mut total = 0;
    for mv in pos.legal_moves() {
        let nodes = if depth <= 1 {
            1
        } else {
            pos.do_move(mv);
            let nodes = perft(pos, depth - 1);
            pos.undo_move(mv);
            nodes
        };
        total += nodes;
        println!("{}: {nodes}", mv.to_usi_owned());
    }
    total
}
