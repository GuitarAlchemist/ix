//! `ix-quality-trend` — CLI entry point.
//!
//! Reads a directory of timestamped JSON snapshots (embeddings, voicing
//! analysis, chatbot QA), computes trends and regressions, and writes an
//! executive-readable markdown report.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use ix_quality_trend::report;
use ix_quality_trend::snapshot::load_all;

/// Aggregate quality snapshots and emit a trend report.
#[derive(Debug, Parser)]
#[command(name = "ix-quality-trend", version, about)]
struct Cli {
    /// Root directory containing `embeddings/`, `voicing-analysis/`,
    /// `chatbot-qa/` subdirectories of `YYYY-MM-DD.json` files.
    #[arg(long)]
    snapshots_dir: PathBuf,

    /// Destination markdown file (directory will be created if needed).
    #[arg(long)]
    out: PathBuf,

    /// Optional machine-readable health artifact JSON path.
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Optional explicit baseline date (informational; the report always uses
    /// the most recent snapshot as the comparison anchor).
    #[arg(long)]
    baseline: Option<String>,

    /// Regression flag threshold in percent absolute Δ vs 7-day average.
    #[arg(long, default_value_t = 5.0)]
    regression_threshold_pct: f64,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Some(b) = &cli.baseline {
        if chrono::NaiveDate::parse_from_str(b, "%Y-%m-%d").is_err() {
            eprintln!("ix-quality-trend: --baseline must be YYYY-MM-DD (got {b:?})");
            return ExitCode::from(2);
        }
    }

    if !cli.snapshots_dir.exists() {
        eprintln!(
            "ix-quality-trend: --snapshots-dir {:?} does not exist",
            cli.snapshots_dir
        );
        return ExitCode::from(2);
    }

    let set = match load_all(&cli.snapshots_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ix-quality-trend: load error: {e}");
            return ExitCode::from(1);
        }
    };

    let summary = report::summarize(&set, cli.regression_threshold_pct);
    let artifact = report::build_health_artifact(&summary, cli.regression_threshold_pct);
    let md = report::render(&set, &cli.snapshots_dir, cli.regression_threshold_pct);

    if let Some(parent) = cli.out.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("ix-quality-trend: cannot create {parent:?}: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if let Err(e) = std::fs::write(&cli.out, md.as_bytes()) {
        eprintln!("ix-quality-trend: cannot write {:?}: {e}", cli.out);
        return ExitCode::from(1);
    }

    if let Some(out_json) = &cli.out_json {
        if let Some(parent) = out_json.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("ix-quality-trend: cannot create {parent:?}: {e}");
                    return ExitCode::from(1);
                }
            }
        }
        let json = match serde_json::to_vec_pretty(&artifact) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("ix-quality-trend: cannot serialize health artifact: {e}");
                return ExitCode::from(1);
            }
        };
        if let Err(e) = std::fs::write(out_json, json) {
            eprintln!("ix-quality-trend: cannot write {:?}: {e}", out_json);
            return ExitCode::from(1);
        }
    }

    eprintln!(
        "ix-quality-trend: wrote {}{} ({} embeddings / {} voicing / {} chatbot snapshots)",
        cli.out.display(),
        cli.out_json
            .as_ref()
            .map(|p| format!(", {}", p.display()))
            .unwrap_or_default(),
        set.embeddings.len(),
        set.voicing.len(),
        set.chatbot.len()
    );
    ExitCode::SUCCESS
}
