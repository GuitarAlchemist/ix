//! Jarvis J4 tracer bullet (GuitarAlchemist/ix#221):
//! GBDT vs persistence baseline on ONE ga `state/quality/` metric series.
//!
//! Reproduce:
//!
//! ```bash
//! cargo run -p ix-quality-trend --example gbdt_vs_persistence -- \
//!     --snapshots-dir ../ga/state/quality \
//!     --out state/jarvis/2026-07-06-j4-gbdt-vs-persistence.eval.json
//! ```
//!
//! With no `--category`, every series is evaluated and the combined report is
//! written to `--out` (if given). Persistence == "last value carries forward";
//! in the direction-classification decision space that is "always predict
//! Flat". If GBDT cannot strictly beat it, the epic pauses (honest-pause rule).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use ix_quality_trend::forecast::{evaluate, load_series, Category, EvalConfig, ForecastEval, Verdict};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "gbdt_vs_persistence",
    about = "Jarvis J4 tracer: GBDT vs persistence on a ga state/quality series (#221)"
)]
struct Cli {
    /// Path to ga `state/quality/` (the sibling clone read path).
    #[arg(long)]
    snapshots_dir: PathBuf,

    /// Which series: chatbot-qa | embeddings | voicing-analysis. Omit for all.
    #[arg(long)]
    category: Option<String>,

    /// Optional path to write the combined JSON report.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Report {
    issue: &'static str,
    tracer: &'static str,
    generated_by: &'static str,
    snapshots_dir: String,
    baseline: &'static str,
    model: &'static str,
    evals: Vec<ForecastEval>,
}

fn print_eval(e: &ForecastEval) {
    println!("── {} [{}] ──", e.category.dir_name(), e.metric);
    println!(
        "   snapshots={}  real={}  carried/degraded={}  distinct_values={}  transitions={}",
        e.n_snapshots_seen, e.n_real, e.n_carried_or_degraded, e.n_distinct_values, e.n_transition_events
    );
    match e.verdict {
        Verdict::PausedInsufficientData => {
            println!("   VERDICT: PAUSED — insufficient data (honest-pause rule).");
        }
        Verdict::Ran => {
            println!(
                "   VERDICT: RAN  train={} test={}  persistence_acc={:.3}  gbdt_acc={:.3}  gbdt_beats_persistence={:?}",
                e.train_len,
                e.test_len,
                e.persistence_accuracy.unwrap_or(f64::NAN),
                e.gbdt_accuracy.unwrap_or(f64::NAN),
                e.gbdt_beats_persistence
            );
        }
    }
    for n in &e.notes {
        println!("   note: {n}");
    }
    println!();
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let cfg = EvalConfig::default();

    let categories: Vec<Category> = match cli.category.as_deref() {
        None => Category::ALL.to_vec(),
        Some(s) => match Category::parse(s) {
            Some(c) => vec![c],
            None => {
                eprintln!("unknown category '{s}' (expected chatbot-qa|embeddings|voicing-analysis)");
                return ExitCode::FAILURE;
            }
        },
    };

    let mut evals = Vec::new();
    for cat in categories {
        let obs = match load_series(&cli.snapshots_dir, cat) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("skip {}: {e}", cat.dir_name());
                continue;
            }
        };
        let eval = evaluate(cat, &obs, &cfg);
        print_eval(&eval);
        evals.push(eval);
    }

    // Fail fast rather than writing empty "successful" evidence: a wrong/missing
    // --snapshots-dir (or a requested category that can't load) must not become a
    // committed report with zero evaluated series.
    if evals.is_empty() {
        eprintln!(
            "no series could be evaluated from '{}' — refusing to write empty evidence",
            cli.snapshots_dir.display()
        );
        return ExitCode::FAILURE;
    }

    let report = Report {
        issue: "GuitarAlchemist/ix#221",
        tracer: "jarvis-j4-gbdt-vs-persistence",
        generated_by: "ix-quality-trend/examples/gbdt_vs_persistence.rs",
        snapshots_dir: cli.snapshots_dir.display().to_string(),
        baseline: "persistence (last value carries forward == always predict Flat)",
        model: "ix_ensemble::GradientBoostedClassifier over 6 recency features",
        evals,
    };

    if let Some(out) = &cli.out {
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                if let Err(e) = std::fs::write(out, json) {
                    eprintln!("failed to write {}: {e}", out.display());
                    return ExitCode::FAILURE;
                }
                println!("wrote report -> {}", out.display());
            }
            Err(e) => {
                eprintln!("serialize error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}
