use std::fmt;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct GikouAnalysis {
    pub eval_cp: i16,
    pub bestmove: String,
    pub elapsed: Duration,
    pub reported_depth: Option<usize>,
}

/// Raw analysis result that keeps the score as-is (cp or mate). Unlike
/// `GikouAnalysis` it does not reject mate scores, so it is usable for diagnostics
/// where a mate verdict is information, not an error.
#[derive(Debug, Clone)]
pub struct GikouEval {
    pub score: GikouScore,
    pub bestmove: String,
    pub elapsed: Duration,
    pub reported_depth: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct GikouMove {
    pub bestmove: String,
    pub elapsed: Duration,
    pub reported_depth: Option<usize>,
    pub nodes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GikouScore {
    Cp(i32),
    Mate(i32),
}

#[derive(Debug)]
pub enum GikouError {
    Io(io::Error),
    Spawn { path: PathBuf, source: io::Error },
    UnexpectedEof(String),
    Timeout { marker: String, waited: Duration },
    MissingScore,
    MissingBestmove,
    MateScore(i32),
    EvalOutOfRange(i32),
    ProcessExit(String),
}

impl GikouError {
    /// Errors that mean the child process is in a bad/unknown state and the
    /// engine should be restarted (killed and respawned) before retrying.
    pub fn requires_restart(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. } | Self::UnexpectedEof(_) | Self::Io(_) | Self::ProcessExit(_)
        )
    }
}

impl fmt::Display for GikouError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Spawn { path, source } => {
                write!(f, "failed to spawn {}: {source}", path.display())
            }
            Self::UnexpectedEof(marker) => {
                write!(f, "unexpected EOF while waiting for {marker}")
            }
            Self::Timeout { marker, waited } => {
                write!(f, "timeout after {waited:?} while waiting for {marker}")
            }
            Self::MissingScore => write!(f, "bestmove received before any usable score"),
            Self::MissingBestmove => write!(f, "missing bestmove"),
            Self::MateScore(score) => {
                write!(f, "score mate {score} is not accepted as cp teacher label")
            }
            Self::EvalOutOfRange(score) => write!(f, "score cp out of i16 range: {score}"),
            Self::ProcessExit(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GikouError {}

impl From<io::Error> for GikouError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Options used to (re)start a Gikou child process. Kept so the engine can be
/// respawned identically after a hang or crash.
#[derive(Debug, Clone)]
pub struct GikouOptions {
    pub path: PathBuf,
    pub own_book: bool,
    pub threads: usize,
    pub resign_score: i32,
    pub depth_limit: Option<usize>,
    pub min_thinking_time_ms: Option<u64>,
    pub handshake_timeout: Duration,
}

impl GikouOptions {
    pub fn new(path: &Path, own_book: bool, threads: usize) -> Self {
        Self {
            path: path.to_path_buf(),
            own_book,
            threads,
            resign_score: -32600,
            depth_limit: None,
            min_thinking_time_ms: None,
            handshake_timeout: Duration::from_secs(60),
        }
    }
}

pub struct GikouEngine {
    options: GikouOptions,
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    reader: Option<JoinHandle<()>>,
}

impl GikouEngine {
    pub fn start(path: &Path, own_book: bool, threads: usize) -> Result<Self, GikouError> {
        Self::with_options(GikouOptions::new(path, own_book, threads))
    }

    pub fn with_options(options: GikouOptions) -> Result<Self, GikouError> {
        let (child, stdin, rx, reader) = spawn_child(&options)?;
        let mut engine = Self {
            options,
            child,
            stdin,
            rx,
            reader: Some(reader),
        };
        engine.handshake()?;
        Ok(engine)
    }

    /// Kill the current child and spawn a fresh one with the same options.
    /// Used to recover from a Gikou crash or hang mid-shard.
    pub fn restart(&mut self) -> Result<(), GikouError> {
        self.kill_child();
        let (child, stdin, rx, reader) = spawn_child(&self.options)?;
        self.child = child;
        self.stdin = stdin;
        self.rx = rx;
        self.reader = Some(reader);
        self.handshake()
    }

    fn handshake(&mut self) -> Result<(), GikouError> {
        let timeout = self.options.handshake_timeout;
        self.send_line("usi")?;
        self.recv_until("usiok", timeout)?;
        self.send_line(&format!(
            "setoption name OwnBook value {}",
            if self.options.own_book { "true" } else { "false" }
        ))?;
        self.send_line(&format!(
            "setoption name ResignScore value {}",
            self.options.resign_score
        ))?;
        self.send_line(&format!(
            "setoption name Threads value {}",
            self.options.threads
        ))?;
        if let Some(depth_limit) = self.options.depth_limit {
            self.send_line(&format!("setoption name DepthLimit value {depth_limit}"))?;
        }
        if let Some(min_thinking_time_ms) = self.options.min_thinking_time_ms {
            self.send_line(&format!(
                "setoption name MinThinkingTime value {min_thinking_time_ms}"
            ))?;
        }
        self.send_line("isready")?;
        self.recv_until("readyok", timeout)?;
        self.send_line("usinewgame")?;
        Ok(())
    }

    pub fn analyze_sfen(
        &mut self,
        sfen: &str,
        depth: usize,
        timeout: Duration,
    ) -> Result<GikouAnalysis, GikouError> {
        let eval = self.eval_sfen(sfen, depth, timeout)?;
        let eval_cp = match eval.score {
            GikouScore::Cp(value) => {
                i16::try_from(value).map_err(|_| GikouError::EvalOutOfRange(value))?
            }
            GikouScore::Mate(value) => return Err(GikouError::MateScore(value)),
        };
        Ok(GikouAnalysis {
            eval_cp,
            bestmove: eval.bestmove,
            elapsed: eval.elapsed,
            reported_depth: eval.reported_depth,
        })
    }

    /// Analyze a position to a fixed depth and return the raw score (cp or mate),
    /// the bestmove, and the reported depth. Mate scores are returned, not
    /// rejected (for diagnostics). Used by `analyze_sfen` after the cp/mate split.
    pub fn eval_sfen(
        &mut self,
        sfen: &str,
        depth: usize,
        timeout: Duration,
    ) -> Result<GikouEval, GikouError> {
        // Drop any stale lines left over from a previous search before issuing
        // a new one, so a late `bestmove` cannot be mistaken for this position.
        self.drain();

        let started = Instant::now();
        self.send_line(&format!("position sfen {sfen}"))?;
        self.send_line(&format!("go depth {depth}"))?;

        let deadline = started + timeout;
        let mut score = None;
        let mut reported_depth = None;

        let bestmove = loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(GikouError::Timeout {
                    marker: "bestmove".to_string(),
                    waited: started.elapsed(),
                });
            }
            let trimmed = match self.rx.recv_timeout(deadline - now) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(GikouError::Timeout {
                        marker: "bestmove".to_string(),
                        waited: started.elapsed(),
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(GikouError::UnexpectedEof("bestmove".to_string()));
                }
            };
            if trimmed.starts_with("info ") {
                if info_is_primary_pv(&trimmed) {
                    if let Some(parsed_depth) = parse_info_depth(&trimmed) {
                        reported_depth = Some(parsed_depth);
                    }
                    if let Some(parsed_score) = parse_info_score(&trimmed) {
                        score = Some(parsed_score);
                    }
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("bestmove ") {
                break rest
                    .split_whitespace()
                    .next()
                    .map(str::to_string)
                    .ok_or(GikouError::MissingBestmove)?;
            }
        };

        let Some(score) = score else {
            return Err(GikouError::MissingScore);
        };

        Ok(GikouEval {
            score,
            bestmove,
            elapsed: started.elapsed(),
            reported_depth,
        })
    }

    pub fn bestmove_sfen(
        &mut self,
        sfen: &str,
        go_command: &str,
        timeout: Duration,
    ) -> Result<GikouMove, GikouError> {
        self.drain();

        let started = Instant::now();
        self.send_line(&format!("position sfen {sfen}"))?;
        self.send_line(go_command)?;

        let deadline = started + timeout;
        let mut reported_depth = None;
        let mut nodes = None;

        let bestmove = loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(GikouError::Timeout {
                    marker: "bestmove".to_string(),
                    waited: started.elapsed(),
                });
            }
            let trimmed = match self.rx.recv_timeout(deadline - now) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(GikouError::Timeout {
                        marker: "bestmove".to_string(),
                        waited: started.elapsed(),
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(GikouError::UnexpectedEof("bestmove".to_string()));
                }
            };
            if trimmed.starts_with("info ") {
                if info_is_primary_pv(&trimmed) {
                    if let Some(parsed_depth) = parse_info_depth(&trimmed) {
                        reported_depth = Some(parsed_depth);
                    }
                    if let Some(parsed_nodes) = parse_info_nodes(&trimmed) {
                        nodes = Some(parsed_nodes);
                    }
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("bestmove ") {
                break rest
                    .split_whitespace()
                    .next()
                    .map(str::to_string)
                    .ok_or(GikouError::MissingBestmove)?;
            }
        };

        Ok(GikouMove {
            bestmove,
            elapsed: started.elapsed(),
            reported_depth,
            nodes,
        })
    }

    fn drain(&mut self) {
        while self.rx.try_recv().is_ok() {}
    }

    fn send_line(&mut self, line: &str) -> Result<(), GikouError> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv_until(&mut self, marker: &str, timeout: Duration) -> Result<(), GikouError> {
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(GikouError::Timeout {
                    marker: marker.to_string(),
                    waited: timeout,
                });
            }
            match self.rx.recv_timeout(deadline - now) {
                Ok(line) => {
                    if line.trim() == marker {
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(GikouError::Timeout {
                        marker: marker.to_string(),
                        waited: timeout,
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(GikouError::UnexpectedEof(marker.to_string()));
                }
            }
        }
    }

    fn kill_child(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // The reader thread exits once the child's stdout reaches EOF.
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for GikouEngine {
    fn drop(&mut self) {
        // Best-effort graceful quit, then ensure the child and reader are gone.
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        self.kill_child();
    }
}

type SpawnedChild = (Child, ChildStdin, Receiver<String>, JoinHandle<()>);

fn spawn_child(options: &GikouOptions) -> Result<SpawnedChild, GikouError> {
    let working_dir = options
        .path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut child = Command::new(&options.path)
        .current_dir(working_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|source| GikouError::Spawn {
            path: options.path.clone(),
            source,
        })?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| GikouError::ProcessExit("failed to open Gikou stdin".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| GikouError::ProcessExit("failed to open Gikou stdout".to_string()))?;

    let (tx, rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tx.send(line.trim().to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    Ok((child, stdin, rx, reader))
}

fn info_is_primary_pv(line: &str) -> bool {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    match tokens.iter().position(|token| *token == "multipv") {
        None => true,
        Some(index) => tokens.get(index + 1) == Some(&"1"),
    }
}

fn parse_info_depth(line: &str) -> Option<usize> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    tokens
        .iter()
        .position(|token| *token == "depth")
        .and_then(|index| tokens.get(index + 1))
        .and_then(|value| value.parse().ok())
}

fn parse_info_nodes(line: &str) -> Option<u64> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    tokens
        .iter()
        .position(|token| *token == "nodes")
        .and_then(|index| tokens.get(index + 1))
        .and_then(|value| value.parse().ok())
}

fn parse_info_score(line: &str) -> Option<GikouScore> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let score_index = tokens.iter().position(|token| *token == "score")?;
    let kind = *tokens.get(score_index + 1)?;
    let value = tokens.get(score_index + 2)?.trim_start_matches('+');
    let value = value.parse::<i32>().ok()?;
    match kind {
        "cp" => Some(GikouScore::Cp(value)),
        "mate" => Some(GikouScore::Mate(value)),
        _ => None,
    }
}
