//! `ix-voicings` CLI — Phase A + B entry point.
//!
//! Phase A: `enumerate` + `featurize` for one or more instruments.
//! Phase B: `cluster`, `topology`, `transitions`, `progressions`, `render_book`.
//! The `run` subcommand runs the full Phase B pipeline (assumes features exist).

use std::collections::HashMap;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use ix_voicings::{
    build_pipeline, cluster, enumerate, featurize, progressions, render_book,
    run_manifest, topology, transitions, viz_precompute, Instrument, VoicingsError,
};

#[derive(Parser, Debug)]
#[command(name = "ix-voicings", version, about = "Voicings study pipeline (Phase A + B)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Comma-separated instrument list. Accepts guitar, bass, ukulele.
    #[arg(long, default_value = "guitar,bass,ukulele")]
    instruments: String,

    /// Cap on voicings per instrument (forwards to GA `--export-max`).
    /// Omit for unlimited (the full corpus; slow).
    #[arg(long)]
    export_max: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run Phase A only (enumerate + featurize)
    PhaseA,
    /// Run Phase B only (cluster + topology + transitions + progressions + book).
    /// Assumes features already exist on disk.
    PhaseB,
    /// Run the full pipeline via PipelineBuilder (Phase B DAG only).
    Pipeline,
    /// Run Phase C: precompute the Harmonic Nebula viz inputs from Phase B
    /// artifacts. Writes `state/viz/{cluster-layout,voicing-layout,neighbors,manifest}.json`.
    /// See `ga/docs/plans/2026-04-20-optick-corpus-viz-plan.md`.
    VizPrecompute,
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

    match cli.command {
        None | Some(Commands::PhaseA) => {
            for &instrument in &instruments {
                match run_phase_a(instrument, cli.export_max) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("ix-voicings phase-a failed for {}: {e}", instrument.as_str());
                        return ExitCode::from(1);
                    }
                }
            }
        }
        Some(Commands::PhaseB) => {
            for &instrument in &instruments {
                match run_phase_b(instrument) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("ix-voicings phase-b failed for {}: {e}", instrument.as_str());
                        return ExitCode::from(1);
                    }
                }
            }
            // Render the book across all instruments
            match render_book(&instruments) {
                Ok(path) => {
                    eprintln!("Book written to: {}", path.display());
                }
                Err(e) => {
                    eprintln!("ix-voicings render_book failed: {e}");
                    return ExitCode::from(1);
                }
            }
        }
        Some(Commands::VizPrecompute) => {
            match viz_precompute::run_viz_precompute(&instruments) {
                Ok(manifest) => {
                    eprintln!(
                        "ix-voicings viz-precompute: {} clusters, {} voicings across {:?}",
                        manifest.cluster_count, manifest.voicing_count, manifest.instruments
                    );
                    eprintln!("Output: {}", viz_precompute::viz_root().display());
                }
                Err(e) => {
                    eprintln!("ix-voicings viz-precompute failed: {e}");
                    return ExitCode::from(1);
                }
            }
        }
        Some(Commands::Pipeline) => {
            for &instrument in &instruments {
                let dag = build_pipeline(instrument);
                let inputs = HashMap::new();
                match ix_pipeline::executor::execute(&dag, &inputs, &ix_pipeline::executor::NoCache) {
                    Ok(result) => {
                        eprintln!(
                            "Pipeline for {} completed in {:?}",
                            instrument.as_str(),
                            result.total_duration
                        );
                        for (level_idx, level) in result.execution_order.iter().enumerate() {
                            eprintln!("  level {}: {:?}", level_idx, level);
                        }
                    }
                    Err(e) => {
                        eprintln!("Pipeline failed for {}: {e}", instrument.as_str());
                        return ExitCode::from(1);
                    }
                }
            }
        }
    }

    ExitCode::SUCCESS
}

fn run_phase_a(instrument: Instrument, export_max: Option<usize>) -> Result<(), VoicingsError> {
    let enum_out = enumerate(instrument, export_max)?;
    let feat_out = featurize(instrument)?;
    let manifest = run_manifest(instrument, &enum_out, &feat_out);
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn run_phase_b(instrument: Instrument) -> Result<(), VoicingsError> {
    eprintln!("[{}] Running cluster...", instrument.as_str());
    match cluster(instrument) {
        Ok(ca) => eprintln!(
            "  cluster: k={}, silhouette={:.4}",
            ca.k, ca.silhouette
        ),
        Err(e) => eprintln!("  cluster FAILED: {e}"),
    }

    eprintln!("[{}] Running topology...", instrument.as_str());
    match topology(instrument) {
        Ok(ta) => eprintln!(
            "  topology: betti_0={}, betti_1={}, pairs={}",
            ta.betti_0,
            ta.betti_1,
            ta.persistence_pairs.len()
        ),
        Err(e) => eprintln!("  topology FAILED: {e}"),
    }

    eprintln!("[{}] Running transitions...", instrument.as_str());
    match transitions(instrument) {
        Ok(ta) => eprintln!(
            "  transitions: edges={}, paths={}",
            ta.edges.len(),
            ta.shortest_paths.len()
        ),
        Err(e) => eprintln!("  transitions FAILED: {e}"),
    }

    eprintln!("[{}] Running progressions...", instrument.as_str());
    match progressions(instrument) {
        Ok(pa) => {
            let total: usize = pa.parse_counts.values().sum();
            eprintln!("  progressions: total_parses={}", total);
        }
        Err(e) => eprintln!("  progressions FAILED: {e}"),
    }

    Ok(())
}
