//! ix-invariant-coverage — measure invariant-catalog optimality.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

use ix_invariant_coverage::{
    coverage::{CoverageMatrix, Firings, OptimalityVerdict},
    invariant::parse_catalog,
    report::render_markdown,
};

#[derive(Parser, Debug)]
#[command(name = "ix-invariant-coverage")]
#[command(about = "Rank, redundancy, orphan and coverage-gap analysis over an invariant catalog.")]
struct Args {
    /// Path to the invariants catalog markdown file.
    #[arg(long)]
    catalog: PathBuf,

    /// Optional path to a firings JSON from an external test run.
    /// If omitted, the report shows catalog structure only (no evidence verdict).
    #[arg(long)]
    firings: Option<PathBuf>,

    /// Where to write the markdown report. Defaults to stdout.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Exit non-zero if the verdict is Suboptimal. Use in CI.
    #[arg(long)]
    strict: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let invariants = match parse_catalog(&args.catalog) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let firings = match args.firings.as_deref() {
        Some(path) => match Firings::from_path(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        None => Firings::default(),
    };

    let matrix = CoverageMatrix::new(invariants, firings);
    let report = matrix.report();
    let markdown = render_markdown(&matrix);

    match args.out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &markdown) {
                eprintln!("error: writing {}: {e}", path.display());
                return ExitCode::from(2);
            }
            eprintln!(
                "Report: {} invariants × {} exemplars, rank={}, verdict={:?}",
                matrix.invariants.len(),
                matrix.exemplars.len(),
                report.rank,
                report.verdict,
            );
        }
        None => {
            print!("{markdown}");
        }
    }

    if args.strict && report.verdict == OptimalityVerdict::Suboptimal {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
