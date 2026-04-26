//! CLI wrapper for ix-harness-clippy.

use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use ix_harness_clippy::clippy_to_observations;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ix-harness-clippy: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut round: Option<u32> = None;
    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--round" => {
                i += 1;
                round = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("invalid round: {}", args[i]))?,
                );
            }
            "--input" => {
                i += 1;
                input_path = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(args[i].clone());
            }
            "-h" | "--help" => {
                eprintln!(
                    "ix-harness-clippy — clippy JSON → SessionEvent JSONL\n\
                     \n\
                     Usage:\n\
                     \x20   cargo clippy --message-format=json 2>&1 | \\\n\
                     \x20       ix-harness-clippy --round <N>\n"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
        i += 1;
    }
    let round = round.ok_or_else(|| "missing required --round <N>".to_string())?;

    let input_bytes: Vec<u8> = match input_path {
        Some(p) => fs::read(&p).map_err(|e| format!("read {p}: {e}"))?,
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            buf
        }
    };

    let events = clippy_to_observations(&input_bytes, round).map_err(|e| e.to_string())?;

    let mut out: Box<dyn Write> = match output_path {
        Some(p) => Box::new(fs::File::create(&p).map_err(|e| format!("create {p}: {e}"))?),
        None => Box::new(io::stdout()),
    };
    for event in &events {
        let line = serde_json::to_string(event).map_err(|e| format!("serialize: {e}"))?;
        writeln!(out, "{line}").map_err(|e| format!("write: {e}"))?;
    }
    Ok(())
}
