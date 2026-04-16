//! `ix-voicings` CLI — Phase A entry point.
//!
//! Runs `enumerate` + `featurize` for one or more instruments and prints
//! a short manifest per instrument. Phase B will add `run` subcommand
//! with the full DAG (cluster/topology/transitions/progressions/book).

use std::process::ExitCode;

use clap::Parser;
use ix_voicings::{enumerate, featurize, run_manifest, Instrument, VoicingsError};

#[derive(Parser, Debug)]
#[command(name = "ix-voicings", version, about = "Voicings study — Phase A (enumerate + featurize)")]
struct Cli {
    /// Comma-separated instrument list. Accepts guitar, bass, ukulele.
    #[arg(long, default_value = "guitar,bass,ukulele")]
    instruments: String,

    /// Cap on voicings per instrument (forwards to GA `--export-max`).
    /// Omit for unlimited (the full corpus; slow).
    #[arg(long)]
    export_max: Option<usize>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let instruments: Result<Vec<Instrument>, VoicingsError> = cli
        .instruments
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(Instrument::parse)
        .collect();

    let instruments = match instruments {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            eprintln!("error: --instruments must list at least one instrument");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    for instrument in instruments {
        match run_one(instrument, cli.export_max) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("ix-voicings failed for {}: {e}", instrument.as_str());
                return ExitCode::from(1);
            }
        }
    }
    ExitCode::SUCCESS
}

fn run_one(instrument: Instrument, export_max: Option<usize>) -> Result<(), VoicingsError> {
    let enum_out = enumerate(instrument, export_max)?;
    let feat_out = featurize(instrument)?;
    let manifest = run_manifest(instrument, &enum_out, &feat_out);
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
