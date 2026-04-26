//! ix-invariant-produce — emit firings.json from the Phase 1 checker library.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

use ix_invariant_coverage::producer::produce_firings;

#[derive(Parser, Debug)]
#[command(name = "ix-invariant-produce")]
#[command(
    about = "Runs the built-in invariant checkers against an exemplar corpus and writes firings.json for ix-invariant-coverage to ingest."
)]
struct Args {
    /// Output path for the firings JSON. Writes to stdout if omitted.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Pretty-print the JSON output.
    #[arg(long)]
    pretty: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let firings = produce_firings();
    let json = if args.pretty {
        serde_json::to_string_pretty(&firings)
    } else {
        serde_json::to_string(&firings)
    };
    let json = match json {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error serializing firings: {e}");
            return ExitCode::from(2);
        }
    };
    match args.out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &json) {
                eprintln!("error writing {}: {e}", path.display());
                return ExitCode::from(2);
            }
            eprintln!(
                "Wrote firings: {} exemplars × {} invariants → {}",
                firings.exemplars.len(),
                firings.fired.len(),
                path.display()
            );
        }
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}
