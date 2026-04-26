//! CLI wrapper: read cargo JSON, emit SessionEvent JSONL.
//!
//! Usage:
//!
//! ```text
//! cargo test --format=json --report-time 2>&1 | \
//!   ix-harness-cargo --round <N> > observations.jsonl
//! ```
//!
//! Defaults:
//! - `--input` defaults to stdin
//! - `--output` defaults to stdout
//! - `--round` is required
//! - `--stats` prints summary stats to stderr instead of emitting observations
//!
//! Exit codes:
//! - 0: success
//! - 1: argument or I/O error
//! - 2: adapter error (UTF-8 failure)

use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use ix_harness_cargo::{cargo_to_observations, compute_stats};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ix-harness-cargo: {}", err.message());
            ExitCode::from(err.exit_code())
        }
    }
}

enum CliError {
    Usage(String),
    Io(String),
    Adapter(String),
}

impl CliError {
    fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) | Self::Io(_) => 1,
            Self::Adapter(_) => 2,
        }
    }
    fn message(&self) -> &str {
        match self {
            Self::Usage(s) | Self::Io(s) | Self::Adapter(s) => s,
        }
    }
}

fn run() -> Result<(), CliError> {
    let args: Vec<String> = std::env::args().collect();
    let mut round: Option<u32> = None;
    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut stats_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--round" => {
                i += 1;
                let s = args
                    .get(i)
                    .ok_or_else(|| CliError::Usage("--round requires a value".to_string()))?;
                round = Some(
                    s.parse()
                        .map_err(|_| CliError::Usage(format!("invalid round: {s}")))?,
                );
            }
            "--input" => {
                i += 1;
                input_path = Some(
                    args.get(i)
                        .ok_or_else(|| CliError::Usage("--input requires a path".to_string()))?
                        .clone(),
                );
            }
            "--output" => {
                i += 1;
                output_path = Some(
                    args.get(i)
                        .ok_or_else(|| CliError::Usage("--output requires a path".to_string()))?
                        .clone(),
                );
            }
            "--stats" => stats_mode = true,
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => return Err(CliError::Usage(format!("unknown argument: {other}"))),
        }
        i += 1;
    }

    // Read input bytes.
    let input_bytes: Vec<u8> = match input_path {
        Some(path) => fs::read(&path).map_err(|e| CliError::Io(format!("read {path}: {e}")))?,
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| CliError::Io(format!("read stdin: {e}")))?;
            buf
        }
    };

    if stats_mode {
        let stats = compute_stats(&input_bytes).map_err(|e| CliError::Adapter(e.to_string()))?;
        eprintln!(
            "cargo-harness stats: total={} passed={} failed={} ignored={} slow={}",
            stats.total_tests, stats.passed, stats.failed, stats.ignored, stats.slow
        );
        return Ok(());
    }

    let round = round.ok_or_else(|| {
        CliError::Usage(
            "missing required --round <N>. Pass the current remediation round number.".to_string(),
        )
    })?;

    let events =
        cargo_to_observations(&input_bytes, round).map_err(|e| CliError::Adapter(e.to_string()))?;

    let mut out: Box<dyn Write> = match output_path {
        Some(path) => Box::new(
            fs::File::create(&path).map_err(|e| CliError::Io(format!("create {path}: {e}")))?,
        ),
        None => Box::new(io::stdout()),
    };
    for event in &events {
        let line = serde_json::to_string(event)
            .map_err(|e| CliError::Adapter(format!("serialize: {e}")))?;
        writeln!(out, "{line}").map_err(|e| CliError::Io(format!("write: {e}")))?;
    }
    Ok(())
}

fn print_usage() {
    eprintln!(
        "ix-harness-cargo — harness adapter: cargo test JSON → SessionEvent JSONL\n\
         \n\
         Usage:\n\
         \x20   ix-harness-cargo --round <N> [--input <path>] [--output <path>]\n\
         \x20   ix-harness-cargo --stats [--input <path>]\n\
         \n\
         Options:\n\
         \x20   --round <N>     required unless --stats; remediation round\n\
         \x20   --input <path>  read cargo JSON from path (default: stdin)\n\
         \x20   --output <path> write SessionEvent JSONL (default: stdout)\n\
         \x20   --stats         print summary stats to stderr, no JSONL output\n\
         \x20   -h, --help      show this message\n\
         \n\
         Example:\n\
         \x20   cargo test --format=json --report-time 2>&1 | \\\n\
         \x20       ix-harness-cargo --round 3 > round-3.jsonl\n\
         \n\
         Projection rules: demerzel/logic/harness-cargo.md\n"
    );
}
