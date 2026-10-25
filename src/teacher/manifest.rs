//! Generation manifest: a human-readable JSON目録 describing a sharded teacher
//! data run. It records the global generation conditions (for reproducibility)
//! and, per shard, the seed, target/actual counts, completion flag, and anomaly
//! tallies. It is the single source of truth for resume: a shard is only skipped
//! when the manifest says `done` (written *after* the shard file is committed).
//!
//! `serde_json` is not available offline, so this module hand-rolls a small JSON
//! reader/writer. The writer emits from typed structs (full control over number
//! formatting); the reader is a minimal recursive-descent parser used only for
//! resume.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;
pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone, PartialEq)]
pub struct GenMeta {
    pub schema_version: u32,
    pub gikou_path: String,
    pub depth: usize,
    pub random_plies: usize,
    pub max_plies: usize,
    pub base_seed: u64,
    pub shard_size: usize,
    pub target_positions: usize,
    pub own_book: bool,
    pub threads_per_engine: usize,
    pub timeout_ms: u64,
    pub hcpe_entry_size: usize,
    pub binbo_git_commit: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShardRecord {
    pub index: usize,
    pub file: String,
    pub seed: u64,
    pub target_count: usize,
    pub count: usize,
    pub done: bool,
    pub games: usize,
    pub mate_cutoffs: usize,
    pub resign_cutoffs: usize,
    pub max_ply_cutoffs: usize,
    pub crash_cutoffs: usize,
    pub timeouts: usize,
    pub engine_restarts: usize,
    pub elapsed_ms: f64,
    pub eval_min: i32,
    pub eval_max: i32,
}

impl ShardRecord {
    pub fn pending(index: usize, file: String, seed: u64, target_count: usize) -> Self {
        Self {
            index,
            file,
            seed,
            target_count,
            count: 0,
            done: false,
            games: 0,
            mate_cutoffs: 0,
            resign_cutoffs: 0,
            max_ply_cutoffs: 0,
            crash_cutoffs: 0,
            timeouts: 0,
            engine_restarts: 0,
            elapsed_ms: 0.0,
            eval_min: 0,
            eval_max: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub meta: GenMeta,
    pub shards: Vec<ShardRecord>,
}

impl Manifest {
    /// Serialize to pretty JSON.
    pub fn to_json_string(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str("  \"schema_version\": ");
        write!(s, "{}", self.meta.schema_version).unwrap();
        s.push_str(",\n");
        s.push_str("  \"meta\": {\n");
        let m = &self.meta;
        push_obj_field(&mut s, 4, "gikou_path", &json_str(&m.gikou_path), true);
        push_obj_field(&mut s, 4, "depth", &m.depth.to_string(), true);
        push_obj_field(&mut s, 4, "random_plies", &m.random_plies.to_string(), true);
        push_obj_field(&mut s, 4, "max_plies", &m.max_plies.to_string(), true);
        push_obj_field(&mut s, 4, "base_seed", &m.base_seed.to_string(), true);
        push_obj_field(&mut s, 4, "shard_size", &m.shard_size.to_string(), true);
        push_obj_field(
            &mut s,
            4,
            "target_positions",
            &m.target_positions.to_string(),
            true,
        );
        push_obj_field(&mut s, 4, "own_book", &m.own_book.to_string(), true);
        push_obj_field(
            &mut s,
            4,
            "threads_per_engine",
            &m.threads_per_engine.to_string(),
            true,
        );
        push_obj_field(&mut s, 4, "timeout_ms", &m.timeout_ms.to_string(), true);
        push_obj_field(
            &mut s,
            4,
            "hcpe_entry_size",
            &m.hcpe_entry_size.to_string(),
            true,
        );
        push_obj_field(
            &mut s,
            4,
            "binbo_git_commit",
            &json_str(&m.binbo_git_commit),
            true,
        );
        push_obj_field(&mut s, 4, "created_at", &json_str(&m.created_at), false);
        s.push_str("  },\n");
        s.push_str("  \"shards\": [\n");
        for (i, shard) in self.shards.iter().enumerate() {
            let last = i + 1 == self.shards.len();
            s.push_str("    {\n");
            push_obj_field(&mut s, 6, "index", &shard.index.to_string(), true);
            push_obj_field(&mut s, 6, "file", &json_str(&shard.file), true);
            push_obj_field(&mut s, 6, "seed", &shard.seed.to_string(), true);
            push_obj_field(
                &mut s,
                6,
                "target_count",
                &shard.target_count.to_string(),
                true,
            );
            push_obj_field(&mut s, 6, "count", &shard.count.to_string(), true);
            push_obj_field(&mut s, 6, "done", &shard.done.to_string(), true);
            push_obj_field(&mut s, 6, "games", &shard.games.to_string(), true);
            push_obj_field(
                &mut s,
                6,
                "mate_cutoffs",
                &shard.mate_cutoffs.to_string(),
                true,
            );
            push_obj_field(
                &mut s,
                6,
                "resign_cutoffs",
                &shard.resign_cutoffs.to_string(),
                true,
            );
            push_obj_field(
                &mut s,
                6,
                "max_ply_cutoffs",
                &shard.max_ply_cutoffs.to_string(),
                true,
            );
            push_obj_field(
                &mut s,
                6,
                "crash_cutoffs",
                &shard.crash_cutoffs.to_string(),
                true,
            );
            push_obj_field(&mut s, 6, "timeouts", &shard.timeouts.to_string(), true);
            push_obj_field(
                &mut s,
                6,
                "engine_restarts",
                &shard.engine_restarts.to_string(),
                true,
            );
            push_obj_field(
                &mut s,
                6,
                "elapsed_ms",
                &format!("{:.3}", shard.elapsed_ms),
                true,
            );
            push_obj_field(&mut s, 6, "eval_min", &shard.eval_min.to_string(), true);
            push_obj_field(&mut s, 6, "eval_max", &shard.eval_max.to_string(), false);
            s.push_str(if last { "    }\n" } else { "    },\n" });
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }

    /// Atomically write the manifest into `dir` (tmp file + rename).
    pub fn save_atomic(&self, dir: &Path) -> Result<(), String> {
        let final_path = dir.join(MANIFEST_FILE);
        let tmp_path = dir.join(format!("{MANIFEST_FILE}.tmp"));
        fs::write(&tmp_path, self.to_json_string())
            .map_err(|err| format!("failed to write {}: {err}", tmp_path.display()))?;
        fs::rename(&tmp_path, &final_path)
            .map_err(|err| format!("failed to rename manifest into place: {err}"))?;
        Ok(())
    }

    pub fn path_in(dir: &Path) -> PathBuf {
        dir.join(MANIFEST_FILE)
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join(MANIFEST_FILE);
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let value = parse_json(text)?;
        let meta_obj = value.get("meta").ok_or("manifest missing meta")?;
        let meta = GenMeta {
            schema_version: value.get("schema_version").and_then(Json::as_u64).unwrap_or(0) as u32,
            gikou_path: meta_obj.get_str("gikou_path")?,
            depth: meta_obj.get_u64("depth")? as usize,
            random_plies: meta_obj.get_u64("random_plies")? as usize,
            max_plies: meta_obj.get_u64("max_plies")? as usize,
            base_seed: meta_obj.get_u64("base_seed")?,
            shard_size: meta_obj.get_u64("shard_size")? as usize,
            target_positions: meta_obj.get_u64("target_positions")? as usize,
            own_book: meta_obj.get_bool("own_book")?,
            threads_per_engine: meta_obj.get_u64("threads_per_engine")? as usize,
            timeout_ms: meta_obj.get_u64("timeout_ms")?,
            hcpe_entry_size: meta_obj.get_u64("hcpe_entry_size")? as usize,
            binbo_git_commit: meta_obj.get_str("binbo_git_commit")?,
            created_at: meta_obj.get_str("created_at")?,
        };
        let shards_value = value.get("shards").ok_or("manifest missing shards")?;
        let shards_arr = shards_value.as_array().ok_or("shards is not an array")?;
        let mut shards = Vec::with_capacity(shards_arr.len());
        for shard in shards_arr {
            shards.push(ShardRecord {
                index: shard.get_u64("index")? as usize,
                file: shard.get_str("file")?,
                seed: shard.get_u64("seed")?,
                target_count: shard.get_u64("target_count")? as usize,
                count: shard.get_u64("count")? as usize,
                done: shard.get_bool("done")?,
                games: shard.get_u64("games")? as usize,
                mate_cutoffs: shard.get_u64("mate_cutoffs")? as usize,
                resign_cutoffs: shard.get_u64("resign_cutoffs")? as usize,
                max_ply_cutoffs: shard.get_u64("max_ply_cutoffs")? as usize,
                crash_cutoffs: shard.get_u64("crash_cutoffs")? as usize,
                timeouts: shard.get_u64("timeouts")? as usize,
                engine_restarts: shard.get_u64("engine_restarts")? as usize,
                elapsed_ms: shard.get_f64("elapsed_ms")?,
                eval_min: shard.get_i64("eval_min")? as i32,
                eval_max: shard.get_i64("eval_max")? as i32,
            });
        }
        Ok(Manifest { meta, shards })
    }
}

fn push_obj_field(out: &mut String, indent: usize, key: &str, value: &str, comma: bool) {
    for _ in 0..indent {
        out.push(' ');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(value);
    out.push_str(if comma { ",\n" } else { "\n" });
}

fn json_str(value: &str) -> String {
    let mut s = String::with_capacity(value.len() + 2);
    s.push('"');
    for ch in value.chars() {
        match ch {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(s, "\\u{:04x}", c as u32).unwrap();
            }
            c => s.push(c),
        }
    }
    s.push('"');
    s
}

// --- minimal JSON reader (objects, arrays, strings, numbers, bool, null) ---

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(raw) => raw.parse::<u64>().ok(),
            _ => None,
        }
    }

    fn get_str(&self, key: &str) -> Result<String, String> {
        match self.get(key) {
            Some(Json::Str(s)) => Ok(s.clone()),
            _ => Err(format!("missing or non-string field: {key}")),
        }
    }

    fn get_u64(&self, key: &str) -> Result<u64, String> {
        match self.get(key) {
            Some(Json::Num(raw)) => raw
                .parse::<u64>()
                .map_err(|_| format!("field {key} is not a u64: {raw}")),
            _ => Err(format!("missing or non-number field: {key}")),
        }
    }

    fn get_i64(&self, key: &str) -> Result<i64, String> {
        match self.get(key) {
            Some(Json::Num(raw)) => raw
                .parse::<i64>()
                .map_err(|_| format!("field {key} is not an i64: {raw}")),
            _ => Err(format!("missing or non-number field: {key}")),
        }
    }

    fn get_f64(&self, key: &str) -> Result<f64, String> {
        match self.get(key) {
            Some(Json::Num(raw)) => raw
                .parse::<f64>()
                .map_err(|_| format!("field {key} is not an f64: {raw}")),
            _ => Err(format!("missing or non-number field: {key}")),
        }
    }

    fn get_bool(&self, key: &str) -> Result<bool, String> {
        match self.get(key) {
            Some(Json::Bool(b)) => Ok(*b),
            _ => Err(format!("missing or non-bool field: {key}")),
        }
    }
}

struct Parser<'a> {
    chars: &'a [u8],
    pos: usize,
}

fn parse_json(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        chars: text.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.chars.len() {
        return Err(format!("trailing data at byte {}", parser.pos));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(format!("unexpected token {other:?} at byte {}", self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.pos += 1; // consume '{'
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected key string at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                other => return Err(format!("expected ',' or '}}', got {other:?}")),
            }
        }
        Ok(Json::Obj(fields))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.pos += 1; // consume '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                other => return Err(format!("expected ',' or ']', got {other:?}")),
            }
        }
        Ok(Json::Arr(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.pos += 1; // consume opening quote
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".to_string()),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'b') => out.push('\u{0008}'),
                        Some(b'f') => out.push('\u{000C}'),
                        Some(b'u') => {
                            let hex: String = self
                                .chars
                                .get(self.pos + 1..self.pos + 5)
                                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                                .ok_or("truncated \\u escape")?;
                            let code = u32::from_str_radix(&hex, 16)
                                .map_err(|_| "invalid \\u escape".to_string())?;
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                            self.pos += 4;
                        }
                        other => return Err(format!("invalid escape {other:?}")),
                    }
                    self.pos += 1;
                }
                Some(_) => {
                    // Consume one UTF-8 codepoint.
                    let start = self.pos;
                    let len = utf8_len(self.chars[start]);
                    let slice = self
                        .chars
                        .get(start..start + len)
                        .ok_or("truncated UTF-8 in string")?;
                    out.push_str(&String::from_utf8_lossy(slice));
                    self.pos += len;
                }
            }
        }
        Ok(out)
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let raw = String::from_utf8_lossy(&self.chars[start..self.pos]).into_owned();
        Ok(Json::Num(raw))
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.chars[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.chars[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.chars[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        Manifest {
            meta: GenMeta {
                schema_version: SCHEMA_VERSION,
                gikou_path: "/tmp/gikou/release".to_string(),
                depth: 8,
                random_plies: 6,
                max_plies: 256,
                base_seed: 20260621,
                shard_size: 20000,
                target_positions: 1_000_000,
                own_book: false,
                threads_per_engine: 1,
                timeout_ms: 30000,
                hcpe_entry_size: 38,
                binbo_git_commit: "abc1234".to_string(),
                created_at: "2026-06-21T00:00:00Z".to_string(),
            },
            shards: vec![
                ShardRecord {
                    index: 0,
                    file: "shard_0000.hcpe".to_string(),
                    seed: 20260621,
                    target_count: 20000,
                    count: 20000,
                    done: true,
                    games: 210,
                    mate_cutoffs: 12,
                    resign_cutoffs: 0,
                    max_ply_cutoffs: 3,
                    crash_cutoffs: 2,
                    timeouts: 1,
                    engine_restarts: 1,
                    elapsed_ms: 81234.5,
                    eval_min: -9000,
                    eval_max: 8000,
                },
                ShardRecord::pending(1, "shard_0001.hcpe".to_string(), 20260622, 15000),
            ],
        }
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = sample_manifest();
        let text = manifest.to_json_string();
        let parsed = Manifest::from_json_str(&text).expect("parse round-trip");
        assert_eq!(manifest, parsed);
    }

    #[test]
    fn pending_shard_is_not_done() {
        let shard = ShardRecord::pending(7, "shard_0007.hcpe".to_string(), 42, 20000);
        assert!(!shard.done);
        assert_eq!(shard.count, 0);
        assert_eq!(shard.target_count, 20000);
    }

    #[test]
    fn parses_with_extra_whitespace_and_unicode() {
        let text = r#"
        {
          "schema_version": 1,
          "meta": {
            "gikou_path": "/path/技巧",
            "depth": 8, "random_plies": 6, "max_plies": 256,
            "base_seed": 18446744073709551615,
            "shard_size": 1, "target_positions": 1, "own_book": false,
            "threads_per_engine": 1, "timeout_ms": 1, "hcpe_entry_size": 38,
            "binbo_git_commit": "x", "created_at": "t"
          },
          "shards": []
        }"#;
        let parsed = Manifest::from_json_str(text).expect("parse");
        assert_eq!(parsed.meta.gikou_path, "/path/技巧");
        assert_eq!(parsed.meta.base_seed, u64::MAX);
        assert!(parsed.shards.is_empty());
    }
}
