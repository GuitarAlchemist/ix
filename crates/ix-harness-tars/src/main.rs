//! CLI wrapper: read tars diagnostic JSON, emit SessionEvent JSONL.
//!
//! Usage:
//!
//! ```text
//! ix-harness-tars --round <N> [--input <path>] [--output <path>]
//! ```
//!
//! Defaults:
//! - `--input` defaults to stdin
//! - `--output` defaults to stdout
//! - `--round` is required
//!
//! Output format: one SessionEvent per line (JSONL), matching the
//! schema at `demerzel/schemas/session-event.schema.json`. Each line
//! is independently parseable and can be appended to any SessionLog
//! file.
//!
//! Exit codes:
//! - 0: success (observations written, may be empty)
//! - 1: argument error (missing --round, bad file path)
//! - 2: adapter error (malformed input)

use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use ix_harness_tars::tars_to_observations;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let code = err.exit_code();
            eprintln!("ix-harness-tars: {}", err.message());
            ExitCode::from(code)
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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--round" => {
                i += 1;
                let s = args.get(i).ok_or_else(|| {
                    CliError::Usage("--round requires a value".to_string())
                })?;
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
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => {
                return Err(CliError::Usage(format!("unknown argument: {other}")));
            }
        }
        i += 1;
    }

    let round = round.ok_or_else(|| {
        CliError::Usage(
            "missing required --round <N>. Pass the current remediation round number."
                .to_string(),
        )
    })?;

    // Read input bytes from file or stdin.
    let input_bytes: Vec<u8> = match input_path {
        Some(path) => {
            fs::read(&path).map_err(|e| CliError::Io(format!("read {path}: {e}")))?
        }
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| CliError::Io(format!("read stdin: {e}")))?;
            buf
        }
    };

    // Project.
    let events = tars_to_observations(&input_bytes, round)
        .map_err(|e| CliError::Adapter(e.to_string()))?;

    // Write JSONL to file or stdout.
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
        "ix-harness-tars — harness adapter: tars diagnostics → SessionEvent JSONL\n\
         \n\
         Usage:\n\
         \x20   ix-harness-tars --round <N> [--input <path>] [--output <path>]\n\
         \n\
         Options:\n\
         \x20   --round <N>       required; remediation round number\n\
         \x20   --input <path>    read native tars JSON from path (default: stdin)\n\
         \x20   --output <path>   write SessionEvent JSONL to path (default: stdout)\n\
         \x20   -h, --help        show this message\n\
         \n\
         Example:\n\
         \x20   tars diagnose --json | ix-harness-tars --round 3 > round-3.jsonl\n\
         \n\
         Projection rules are specified in\n\
         demerzel/logic/harness-tars.md and consumed as\n\
         prior_observations by ix_triage_session.\n"
    );
}
