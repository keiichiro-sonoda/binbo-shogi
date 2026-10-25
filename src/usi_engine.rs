//! Minimal generic USI engine driver for head-to-head matches (Task 060).
//!
//! Mirrors the process plumbing of `src/teacher/gikou.rs` (spawn + background reader
//! thread + `bestmove` request loop) but takes a caller-supplied `setoption` list, so it
//! can drive any USI engine. Used to run YaneuraOu as the opponent in `nnue-vs-yane`
//! (the gikou-specific wrapper hard-codes gikou's option names, e.g. it lacks `EvalDir`).

use std::fmt;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct UsiMove {
    pub bestmove: String,
    pub elapsed: Duration,
    pub depth: Option<usize>,
    pub nodes: Option<u64>,
}

#[derive(Debug)]
pub enum UsiError {
    Io(io::Error),
    Spawn { path: PathBuf, source: io::Error },
    Eof(String),
    Timeout { marker: String, waited: Duration },
    MissingBestmove,
    ProcessExit(String),
}

impl UsiError {
    /// The child is in an unknown/bad state and should be restarted before retrying.
    pub fn requires_restart(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. } | Self::Eof(_) | Self::Io(_) | Self::ProcessExit(_)
        )
    }
}

impl fmt::Display for UsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Spawn { path, source } => write!(f, "failed to spawn {}: {source}", path.display()),
            Self::Eof(marker) => write!(f, "unexpected EOF while waiting for {marker}"),
            Self::Timeout { marker, waited } => {
                write!(f, "timeout after {waited:?} while waiting for {marker}")
            }
            Self::MissingBestmove => write!(f, "missing bestmove"),
            Self::ProcessExit(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for UsiError {}

impl From<io::Error> for UsiError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// (Re)start parameters: kept so the engine can be respawned identically after a hang.
#[derive(Debug, Clone)]
pub struct UsiOptions {
    pub path: PathBuf,
    /// `setoption name <0> value <1>` lines sent between `usiok` and `isready`.
    pub setoptions: Vec<(String, String)>,
    pub handshake_timeout: Duration,
}

impl UsiOptions {
    pub fn new(path: &Path, setoptions: Vec<(String, String)>) -> Self {
        Self {
            path: path.to_path_buf(),
            setoptions,
            handshake_timeout: Duration::from_secs(60),
        }
    }
}

pub struct UsiEngine {
    options: UsiOptions,
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    reader: Option<JoinHandle<()>>,
}

impl UsiEngine {
    pub fn start(options: UsiOptions) -> Result<Self, UsiError> {
        let (child, stdin, rx, reader) = spawn_child(&options)?;
        let mut engine = Self { options, child, stdin, rx, reader: Some(reader) };
        engine.handshake()?;
        Ok(engine)
    }

    pub fn restart(&mut self) -> Result<(), UsiError> {
        self.kill_child();
        let (child, stdin, rx, reader) = spawn_child(&self.options)?;
        self.child = child;
        self.stdin = stdin;
        self.rx = rx;
        self.reader = Some(reader);
        self.handshake()
    }

    fn handshake(&mut self) -> Result<(), UsiError> {
        let timeout = self.options.handshake_timeout;
        self.send_line("usi")?;
        self.recv_until("usiok", timeout)?;
        for (name, value) in &self.options.setoptions.clone() {
            self.send_line(&format!("setoption name {name} value {value}"))?;
        }
        self.send_line("isready")?;
        self.recv_until("readyok", timeout)?;
        self.send_line("usinewgame")?;
        Ok(())
    }

    /// `usinewgame` to clear per-game state (TT, etc.) between games.
    pub fn new_game(&mut self) -> Result<(), UsiError> {
        self.send_line("usinewgame")
    }

    /// Send `position sfen <sfen>` + `go_command`, returning the bestmove (+ depth/nodes).
    pub fn bestmove(
        &mut self,
        sfen: &str,
        go_command: &str,
        timeout: Duration,
    ) -> Result<UsiMove, UsiError> {
        self.drain();
        let started = Instant::now();
        self.send_line(&format!("position sfen {sfen}"))?;
        self.send_line(go_command)?;

        let deadline = started + timeout;
        let mut depth = None;
        let mut nodes = None;
        let bestmove = loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(UsiError::Timeout { marker: "bestmove".to_string(), waited: started.elapsed() });
            }
            let trimmed = match self.rx.recv_timeout(deadline - now) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(UsiError::Timeout { marker: "bestmove".to_string(), waited: started.elapsed() })
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(UsiError::Eof("bestmove".to_string()))
                }
            };
            if trimmed.starts_with("info ") {
                if info_is_primary_pv(&trimmed) {
                    if let Some(d) = parse_info_field(&trimmed, "depth") {
                        depth = Some(d as usize);
                    }
                    if let Some(n) = parse_info_field(&trimmed, "nodes") {
                        nodes = Some(n);
                    }
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("bestmove ") {
                break rest
                    .split_whitespace()
                    .next()
                    .map(str::to_string)
                    .ok_or(UsiError::MissingBestmove)?;
            }
        };
        Ok(UsiMove { bestmove, elapsed: started.elapsed(), depth, nodes })
    }

    fn drain(&mut self) {
        while self.rx.try_recv().is_ok() {}
    }

    fn send_line(&mut self, line: &str) -> Result<(), UsiError> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv_until(&mut self, marker: &str, timeout: Duration) -> Result<(), UsiError> {
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(UsiError::Timeout { marker: marker.to_string(), waited: timeout });
            }
            match self.rx.recv_timeout(deadline - now) {
                Ok(line) if line.trim() == marker => return Ok(()),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    return Err(UsiError::Timeout { marker: marker.to_string(), waited: timeout })
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(UsiError::Eof(marker.to_string()))
                }
            }
        }
    }

    fn kill_child(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for UsiEngine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        self.kill_child();
    }
}

type SpawnedChild = (Child, ChildStdin, Receiver<String>, JoinHandle<()>);

fn spawn_child(options: &UsiOptions) -> Result<SpawnedChild, UsiError> {
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
        .map_err(|source| UsiError::Spawn { path: options.path.clone(), source })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| UsiError::ProcessExit("failed to open engine stdin".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| UsiError::ProcessExit("failed to open engine stdout".to_string()))?;
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
    match tokens.iter().position(|t| *t == "multipv") {
        None => true,
        Some(i) => tokens.get(i + 1) == Some(&"1"),
    }
}

fn parse_info_field(line: &str, field: &str) -> Option<u64> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    tokens
        .iter()
        .position(|t| *t == field)
        .and_then(|i| tokens.get(i + 1))
        .and_then(|v| v.parse().ok())
}
