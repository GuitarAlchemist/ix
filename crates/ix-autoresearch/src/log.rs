//! JSONL run log — append-only, schema-versioned, replay-tolerant.
//!
//! Every entry is one of [`LogEvent::RunStart`], [`LogEvent::Iteration`],
//! [`LogEvent::RunComplete`]. The first line of every run is `RunStart`;
//! the absence of `RunComplete` means the run was interrupted.
//!
//! ## Durability contract
//!
//! Every `append` (a) serializes the event to a single `Vec<u8>`,
//! (b) writes it with one `write_all` of `[buf, b"\n"].concat()`,
//! (c) calls `sync_all` per the configured checkpoint policy. We do **not**
//! use `BufWriter` + `writeln!` — on Windows NTFS, a 1–2 KB JSON line can
//! split across two `WriteFile` calls and produce a torn line under crash.
//!
//! ## Replay contract
//!
//! `read_log` parses each line under `serde_json::from_str`. A trailing
//! parse failure (interpreted as crash-truncated) is silently discarded;
//! any parse failure on a non-trailing line is a hard error (corruption
//! must not be hidden mid-stream).
//!
//! ## Schema version
//!
//! Every event carries `schema_version: u32`. v1 → v1.x changes are
//! additive, with `#[serde(default)]` on every new optional field.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::AutoresearchError;

/// Wire-format version. Bump when schema changes are not additive.
pub const SCHEMA_VERSION: u32 = 1;

/// Tagged JSONL event. Every line in the log is exactly one of these.
///
/// The explicit `bound` attribute on the derive prevents serde from
/// adding spurious `C: Default` / `S: Default` bounds when our
/// `#[serde(default)]` annotations target `Option<…>` or other types
/// that already implement `Default` independently of `C` / `S`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "event",
    rename_all = "snake_case",
    bound(serialize = "C: Serialize, S: Serialize"),
    bound(deserialize = "C: Deserialize<'de>, S: Deserialize<'de>")
)]
pub enum LogEvent<C, S> {
    /// First line of every run.
    RunStart {
        #[serde(default = "default_schema_version")]
        schema_version: u32,
        run_id: String,
        timestamp: DateTime<Utc>,
        target: String,
        strategy: serde_json::Value,
        seed: u64,
        #[serde(default)]
        git_sha: Option<String>,
        #[serde(default)]
        git_sha_reason: Option<String>,
        baseline_config: C,
        #[serde(default)]
        eval_inputs_hash: Option<String>,
    },

    /// One per evaluated candidate.
    Iteration {
        #[serde(default = "default_schema_version")]
        schema_version: u32,
        iteration: usize,
        timestamp: DateTime<Utc>,
        config: C,
        config_hash: String,
        #[serde(default)]
        score: Option<S>,
        #[serde(default)]
        reward: Option<f64>,
        accepted: bool,
        #[serde(default)]
        previous_hash: Option<String>,
        #[serde(default)]
        error: Option<String>,
        elapsed_ms: u64,
        /// Strategy state at decision time (e.g. SA temperature). Logged for
        /// forensics; v1 does not consume this on resume — see plan §SA
        /// temperature resume.
        #[serde(default)]
        strategy_state: Option<serde_json::Value>,
        /// `true` if this iteration's score came from the cache.
        #[serde(default)]
        cache_hit: bool,
    },

    /// Last line on graceful exit. Absence ⇒ run was interrupted.
    RunComplete {
        #[serde(default = "default_schema_version")]
        schema_version: u32,
        timestamp: DateTime<Utc>,
        iterations: usize,
        accepted: usize,
        #[serde(default)]
        best_iteration: Option<usize>,
        #[serde(default)]
        best_reward: Option<f64>,
        /// `Some(n)` if the run aborted because of n consecutive hard kills.
        #[serde(default)]
        consecutive_kills_at_abort: Option<usize>,
        /// Cost ledger (harness-engineering pattern). All fields are
        /// aggregated across the iteration entries.
        #[serde(default)]
        cost: Option<CostLedger>,
    },
}

/// Aggregated cost over a run. Per the harness-engineering "cost ledger"
/// pattern (Uni-CLI), surface this on `RunComplete` so users can answer
/// "did this overnight run cost what we expected?" without parsing every
/// iteration line.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostLedger {
    /// Sum of every iteration's `elapsed_ms` (wall clock per iter).
    pub total_elapsed_ms: u64,
    /// How many iterations short-circuited via the cache.
    pub cache_hit_count: u32,
    /// How many iterations failed evaluation (error.is_some()).
    pub eval_failure_count: u32,
    /// How many iterations were rejected by the strategy.
    pub rejected_count: u32,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Frequency policy for `sync_all` on the underlying file.
#[derive(Debug, Clone, Copy)]
pub enum FsyncPolicy {
    /// Sync after every event. Maximum durability, ~5 ms × N overhead.
    Every,
    /// Sync after every Nth event AND on every accepted iteration.
    /// Default `N = 10`. Recommended.
    Checkpoint { every_n: u32 },
    /// Never sync from inside the kernel. Eval is responsible.
    Never,
}

impl Default for FsyncPolicy {
    fn default() -> Self {
        Self::Checkpoint { every_n: 10 }
    }
}

/// Append-only JSONL log writer. One per run.
pub struct JsonlLog {
    file: File,
    path: PathBuf,
    policy: FsyncPolicy,
    appended_since_sync: u32,
}

impl JsonlLog {
    /// Open `path` in append mode (creates if missing). Parent dirs must
    /// already exist; this fails fast on missing parent or unwritable target.
    pub fn open(path: impl Into<PathBuf>, policy: FsyncPolicy) -> Result<Self, AutoresearchError> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            file,
            path,
            policy,
            appended_since_sync: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a single event with atomic write semantics.
    ///
    /// `accepted_hint` is `Some(true)` when this entry's `accepted` field
    /// is true — used by the `Checkpoint` policy to fsync on every accept.
    pub fn append<C, S>(
        &mut self,
        event: &LogEvent<C, S>,
        accepted_hint: bool,
    ) -> Result<(), AutoresearchError>
    where
        C: Serialize,
        S: Serialize,
    {
        // Serialize first → single buffer → single write_all.
        let mut buf = serde_json::to_vec(event)?;
        buf.push(b'\n');
        self.file.write_all(&buf)?;

        // Sync per policy.
        self.appended_since_sync = self.appended_since_sync.saturating_add(1);
        let should_sync = match self.policy {
            FsyncPolicy::Every => true,
            FsyncPolicy::Checkpoint { every_n } => {
                accepted_hint || self.appended_since_sync >= every_n
            }
            FsyncPolicy::Never => false,
        };
        if should_sync {
            self.file.sync_all()?;
            self.appended_since_sync = 0;
        }
        Ok(())
    }

    /// Force-sync on close. Idempotent.
    pub fn finalize(self) -> Result<(), AutoresearchError> {
        self.file.sync_all()?;
        Ok(())
    }
}

/// Replay a log file. Trailing-parse-failure is treated as a crash-truncated
/// last line and discarded silently; mid-stream parse failure is a hard error.
///
/// Returns the parsed events in file order.
pub fn read_log<C, S>(path: &Path) -> Result<Vec<LogEvent<C, S>>, AutoresearchError>
where
    C: for<'de> Deserialize<'de>,
    S: for<'de> Deserialize<'de>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?;

    let mut out: Vec<LogEvent<C, S>> = Vec::with_capacity(lines.len());
    let total = lines.len();
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LogEvent<C, S>>(line) {
            Ok(ev) => out.push(ev),
            Err(e) => {
                if i + 1 == total {
                    // Trailing parse failure: assume crash-truncated.
                    break;
                }
                return Err(AutoresearchError::Serde(e));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MockConfig(f64);

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MockScore(f64);

    fn run_start() -> LogEvent<MockConfig, MockScore> {
        LogEvent::RunStart {
            schema_version: SCHEMA_VERSION,
            run_id: "test-run".to_string(),
            timestamp: Utc::now(),
            target: "mock".to_string(),
            strategy: serde_json::json!({ "kind": "Greedy" }),
            seed: 42,
            git_sha: None,
            git_sha_reason: Some("test".to_string()),
            baseline_config: MockConfig(1.0),
            eval_inputs_hash: None,
        }
    }

    fn iteration(i: usize, accepted: bool) -> LogEvent<MockConfig, MockScore> {
        LogEvent::Iteration {
            schema_version: SCHEMA_VERSION,
            iteration: i,
            timestamp: Utc::now(),
            config: MockConfig(i as f64 * 0.1),
            config_hash: format!("hash-{i}"),
            score: Some(MockScore(0.5)),
            reward: Some(0.5),
            accepted,
            previous_hash: None,
            error: None,
            elapsed_ms: 5,
            strategy_state: None,
            cache_hit: false,
        }
    }

    fn run_complete() -> LogEvent<MockConfig, MockScore> {
        LogEvent::RunComplete {
            schema_version: SCHEMA_VERSION,
            timestamp: Utc::now(),
            iterations: 10,
            accepted: 3,
            best_iteration: Some(5),
            best_reward: Some(0.8),
            consecutive_kills_at_abort: None,
            cost: Some(CostLedger {
                total_elapsed_ms: 50,
                cache_hit_count: 1,
                eval_failure_count: 0,
                rejected_count: 7,
            }),
        }
    }

    #[test]
    fn jsonl_roundtrip_preserves_event_order() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("log.jsonl");
        let mut log = JsonlLog::open(&path, FsyncPolicy::Every).unwrap();
        log.append(&run_start(), false).unwrap();
        for i in 0..5 {
            log.append(&iteration(i, i == 2), i == 2).unwrap();
        }
        log.append(&run_complete(), false).unwrap();
        log.finalize().unwrap();

        let events: Vec<LogEvent<MockConfig, MockScore>> = read_log(&path).unwrap();
        assert_eq!(events.len(), 7);
        assert!(matches!(events[0], LogEvent::RunStart { .. }));
        assert!(matches!(events[6], LogEvent::RunComplete { .. }));
    }

    #[test]
    fn truncated_trailing_line_is_silently_discarded() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("log.jsonl");
        {
            let mut log = JsonlLog::open(&path, FsyncPolicy::Every).unwrap();
            log.append(&run_start(), false).unwrap();
            log.append(&iteration(0, false), false).unwrap();
            log.finalize().unwrap();
        }
        // Manually truncate the last line in the middle.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 30);
        std::fs::write(&path, &bytes).unwrap();

        let events: Vec<LogEvent<MockConfig, MockScore>> = read_log(&path).unwrap();
        // First line parses; trailing partial line is dropped.
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn mid_stream_parse_failure_is_a_hard_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("log.jsonl");
        let mut bad = String::new();
        // First line is malformed run_start (missing required fields).
        bad.push_str("{ \"event\": \"run_start\", \"run_id\": \"x\" }\n");
        // Second line has a junk-but-not-trailing payload that fails parse.
        bad.push_str("not a json line\n");
        // Third line is a valid run_complete (now requires `cost: null`).
        let valid_complete = serde_json::to_string(&run_complete()).unwrap();
        bad.push_str(&valid_complete);
        bad.push('\n');
        std::fs::write(&path, bad).unwrap();
        let res: Result<Vec<LogEvent<MockConfig, MockScore>>, _> = read_log(&path);
        assert!(res.is_err());
    }

    #[test]
    fn fsync_checkpoint_policy_does_not_sync_every_line() {
        // We can't observe fsync calls directly, but we can verify the
        // policy doesn't error and produces a readable log.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("log.jsonl");
        let mut log = JsonlLog::open(&path, FsyncPolicy::Checkpoint { every_n: 5 }).unwrap();
        for i in 0..12 {
            log.append(&iteration(i, false), false).unwrap();
        }
        log.finalize().unwrap();
        let events: Vec<LogEvent<MockConfig, MockScore>> = read_log(&path).unwrap();
        assert_eq!(events.len(), 12);
    }

    #[test]
    fn schema_version_is_present_on_every_event() {
        let json_runstart = serde_json::to_string(&run_start()).unwrap();
        let json_iter = serde_json::to_string(&iteration(0, true)).unwrap();
        let json_complete = serde_json::to_string(&run_complete()).unwrap();
        assert!(json_runstart.contains("\"schema_version\":1"));
        assert!(json_iter.contains("\"schema_version\":1"));
        assert!(json_complete.contains("\"schema_version\":1"));
    }
}
