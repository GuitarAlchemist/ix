//! ix-invariant-produce — emit firings.json from the Phase 1 checker library.
//!
//! Output is wrapped in the canonical dashboard envelope
//! (`{domain, emitted_at, metric_name, metric_value, oracle_status,
//! summary, problems}`) consumed by the GA dev dashboard at
//! `/test#dev/summary`. The legacy `exemplars` and `fired` fields are
//! flattened at the top level so existing readers
//! (`ix-invariant-coverage`) continue to deserialize unchanged.

use chrono::SecondsFormat;
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

use ix_invariant_coverage::coverage::FiringsEnvelope;
use ix_invariant_coverage::producer::produce_firings;

#[derive(Parser, Debug)]
#[command(name = "ix-invariant-produce")]
#[command(
    about = "Runs the built-in invariant checkers against an exemplar corpus and writes a canonical-envelope firings JSON for the dashboard and ix-invariant-coverage to ingest."
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
    let exemplar_count = firings.exemplars.len();
    let invariant_count = firings.fired.len();
    let emitted_at = chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let envelope = FiringsEnvelope::wrap(firings, emitted_at);

    let json = if args.pretty {
        serde_json::to_string_pretty(&envelope)
    } else {
        serde_json::to_string(&envelope)
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
                "Wrote firings: {} exemplars × {} invariants (status={}, metric={:.3}) → {}",
                exemplar_count,
                invariant_count,
                envelope.oracle_status,
                envelope.metric_value,
                path.display()
            );
        }
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}
