//! `ix-quality-validate <root>` — CI gate for dashboard-envelope snapshots.
//!
//! Walks the given root directory, validating every `*.json` file against
//! `docs/contracts/quality-snapshot.schema.json`. Prints a per-file
//! pass/fail report and exits `0` on full success or `1` on any failure.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use ix_quality_validate::{build_validator, walk_and_validate, FileReport, FileStatus};

#[derive(Parser, Debug)]
#[command(
    name = "ix-quality-validate",
    about = "Validate state/quality/*.json snapshots against the canonical dashboard envelope schema.",
    long_about = "Recursively scans <root> for *.json files and validates each against \
                  docs/contracts/quality-snapshot.schema.json (canonical dashboard envelope). \
                  Files whose name begins with '_' (e.g. _schema.json) are skipped.\n\n\
                  Exit code 0 = all files pass. Exit code 1 = at least one failure. \
                  Exit code 2 = could not load or compile the schema."
)]
struct Args {
    /// Root directory to scan (e.g. `state/quality`).
    root: PathBuf,

    /// Suppress per-file PASS lines (still prints failures and the summary).
    #[arg(long)]
    quiet: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let validator = match build_validator() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    if !args.root.exists() {
        eprintln!("error: root path does not exist: {}", args.root.display());
        return ExitCode::from(2);
    }

    let reports = walk_and_validate(&args.root, &validator);

    let mut passed = 0usize;
    let mut failed = 0usize;
    for report in &reports {
        print_report(report, args.quiet);
        match report.status {
            FileStatus::Pass => passed += 1,
            _ => failed += 1,
        }
    }

    println!(
        "\nix-quality-validate: {} file(s) — {} pass, {} fail",
        reports.len(),
        passed,
        failed
    );

    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn print_report(report: &FileReport, quiet: bool) {
    match &report.status {
        FileStatus::Pass => {
            if !quiet {
                println!("PASS  {}", report.path.display());
            }
        }
        FileStatus::Fail(errors) => {
            println!("FAIL  {}", report.path.display());
            for err in errors {
                println!("        - {err}");
            }
        }
        FileStatus::Unreadable(msg) => {
            println!("ERR   {}  ({msg})", report.path.display());
        }
    }
}
