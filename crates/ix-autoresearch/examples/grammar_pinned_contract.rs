//! `grammar_pinned_contract` — emit a real grammar autoresearch log
//! conforming to the pinned JSONL contract in `SCHEMA.md`.
//!
//! This example is the **canonical producer** for the Phase 1 IX→Hari
//! boundary contract. It runs the in-process `GrammarTarget` smoke test
//! for a bounded number of iterations and writes the resulting JSONL log
//! to a caller-supplied directory. Output is consumed by:
//!
//! 1. `tests/jsonl_contract.rs` — structural + determinism validation on
//!    the IX side (CI green ⇒ contract holds for the producer).
//! 2. `hari-extractor`'s `hari-from-ix-autoresearch` bin — projects each
//!    `iteration` line into a Hari `ResearchEvent` (Phase 2).
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example grammar_pinned_contract -- \
//!     --out-dir state/autoresearch/contract \
//!     --iterations 20 \
//!     --seed 42
//! ```
//!
//! The log will be written to `<out-dir>/<run-id>/log.jsonl` (the kernel
//! picks the run-id). The path is printed on stdout in a `LOG_PATH=`
//! line so callers can scrape it.
//!
//! ## Why grammar
//!
//! Per the plan and README, grammar is the in-process, sub-second
//! smoke-test target. No shell-out, no GA build dependency. A 20-iter
//! greedy run finishes in well under a second on a laptop and reliably
//! produces both `accepted = true` and `accepted = false` lines — which
//! is what we need to exercise Hari's contradictory-evidence semantics
//! downstream.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use ix_autoresearch::{run_experiment, GrammarTarget, Strategy, TimeBudget};

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    std::fs::create_dir_all(&args.out_dir).expect("create out-dir");

    let mut target = GrammarTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::SimulatedAnnealing {
            initial_temperature: Some(0.05),
            cooling_rate: 0.95,
        },
        args.iterations,
        TimeBudget::soft(Duration::from_secs(5)),
        &args.out_dir,
        args.seed,
    )
    .expect("run_experiment");

    println!("LOG_PATH={}", outcome.log_path.display());
    println!("RUN_ID={}", outcome.run_id);
    println!("ITERATIONS={}", outcome.iterations);
    println!("ACCEPTED={}", outcome.accepted);
    if let Some(r) = outcome.best_reward {
        println!("BEST_REWARD={r:.6}");
    }

    ExitCode::SUCCESS
}

struct Args {
    out_dir: PathBuf,
    iterations: usize,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut out_dir: Option<PathBuf> = None;
    let mut iterations: usize = 20;
    let mut seed: u64 = 42;

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--out-dir" => out_dir = argv.next().map(PathBuf::from),
            "--iterations" => {
                iterations = argv
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--iterations expects a usize".to_string())?;
            }
            "--seed" => {
                seed = argv
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--seed expects a u64".to_string())?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    let out_dir = out_dir.ok_or_else(|| "--out-dir <path> is required".to_string())?;
    Ok(Args {
        out_dir,
        iterations,
        seed,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: grammar_pinned_contract --out-dir <path> [--iterations N] [--seed N]\n\
         \n\
         Runs the GrammarTarget smoke-test for `iterations` iterations of SA\n\
         and writes the JSONL log to `<out-dir>/<run-id>/log.jsonl`. The log\n\
         conforms to the pinned contract in crates/ix-autoresearch/SCHEMA.md."
    );
}
