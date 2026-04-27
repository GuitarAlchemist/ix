//! `ix-autoresearch` CLI — multi-verb dispatcher around the kernel.
//!
//! Verbs (per plan §Phase 3):
//!
//! - `run --target grammar [...]` — fresh experiment.
//! - `resume --log <path> [...]` — continue an interrupted run.
//! - `list [--include-milestones]` — enumerate runs and milestones under
//!   `state/autoresearch/`.
//! - `promote <run-id> --note <slug> [--force]` — copy run → milestone.
//! - `revert <run-id> --to <iteration>` — print the config from a
//!   specific iteration (read-only; no mutation).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

use ix_autoresearch::{
    is_complete_milestone, log::read_log, promote_run, resume_experiment, run_experiment,
    GrammarConfig, GrammarScore, GrammarTarget, LogEvent, RunId, Strategy, TimeBudget,
    MCP_ITERATION_CAP,
};

/// Default state-dir under cwd. Can be overridden via `--state-dir`.
const DEFAULT_STATE_DIR: &str = "state/autoresearch";

#[derive(Parser, Debug)]
#[command(
    name = "ix-autoresearch",
    version,
    about = "Karpathy-style edit-eval-iterate kernel for IX subsystems"
)]
struct Cli {
    /// State directory for runs and milestones.
    #[arg(long, default_value = DEFAULT_STATE_DIR, global = true)]
    state_dir: PathBuf,

    /// Suppress non-essential output.
    #[arg(long, short = 'q', global = true)]
    quiet: bool,

    #[command(subcommand)]
    verb: Verb,
}

#[derive(Subcommand, Debug)]
enum Verb {
    /// Start a fresh experiment.
    Run(RunArgs),

    /// Resume an interrupted run from its existing log file.
    Resume(ResumeArgs),

    /// List runs and milestones.
    List {
        /// Also show entries under `milestones/`.
        #[arg(long, default_value_t = true)]
        include_milestones: bool,
    },

    /// Copy a run into the milestones tree (atomic, sanitized).
    Promote(PromoteArgs),

    /// Print the config from a specific iteration of a run (read-only).
    Revert(RevertArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Adapter to invoke.
    #[arg(long, value_enum)]
    target: TargetKind,

    /// Number of iterations to run.
    #[arg(long, default_value_t = 50)]
    iterations: usize,

    /// Acceptance strategy.
    #[arg(long, value_enum, default_value_t = StrategyKind::Greedy)]
    strategy: StrategyKind,

    /// SA initial temperature; required when `--strategy sa` and you
    /// want to skip auto-calibration.
    #[arg(long)]
    initial_temperature: Option<f64>,

    /// SA cooling rate; only meaningful with `--strategy sa`.
    #[arg(long, default_value_t = 0.95)]
    cooling_rate: f64,

    /// Per-iter soft deadline in seconds.
    #[arg(long, default_value_t = 300.0)]
    soft_seconds: f64,

    /// Per-iter hard timeout in seconds; off by default.
    #[arg(long)]
    hard_seconds: Option<f64>,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(clap::Args, Debug)]
struct ResumeArgs {
    /// Path to an existing log.jsonl.
    #[arg(long)]
    log: PathBuf,

    /// Additional iterations beyond the existing log's last entry.
    #[arg(long)]
    additional_iterations: usize,

    /// Adapter (must match the original run).
    #[arg(long, value_enum)]
    target: TargetKind,

    /// Soft deadline.
    #[arg(long, default_value_t = 300.0)]
    soft_seconds: f64,

    /// Hard timeout (off by default).
    #[arg(long)]
    hard_seconds: Option<f64>,

    /// RNG seed for the resumed iterations (independent from original).
    #[arg(long, default_value_t = 43)]
    seed: u64,
}

#[derive(clap::Args, Debug)]
struct PromoteArgs {
    /// Source run identifier (UUIDv7 string).
    run_id: String,

    /// Milestone slug (regex `^[a-z0-9][a-z0-9-]{0,63}$`).
    #[arg(long)]
    note: String,

    /// Overwrite an existing milestone with the same slug.
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(clap::Args, Debug)]
struct RevertArgs {
    /// Source run identifier.
    run_id: String,

    /// Iteration whose config should be printed.
    #[arg(long)]
    to: usize,
}

#[derive(ValueEnum, Clone, Debug)]
enum TargetKind {
    Grammar,
    // Phase 4 will add Chatbot; Phase 5 will add OpticK. Listed here as
    // future variants would require recompile, so we stick to one for v1.
}

#[derive(ValueEnum, Clone, Debug)]
enum StrategyKind {
    Greedy,
    Sa,
    Random,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let runs_root = cli.state_dir.join("runs");
    let milestones_root = cli.state_dir.join("milestones");

    let result: Result<(), String> = match cli.verb {
        Verb::Run(args) => verb_run(&runs_root, args, cli.quiet),
        Verb::Resume(args) => verb_resume(args, cli.quiet),
        Verb::List {
            include_milestones,
        } => verb_list(&runs_root, &milestones_root, include_milestones),
        Verb::Promote(args) => verb_promote(&runs_root, &milestones_root, args, cli.quiet),
        Verb::Revert(args) => verb_revert(&runs_root, args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ix-autoresearch: {e}");
            ExitCode::from(1)
        }
    }
}

// ───────────────────────── verb impls ─────────────────────────

fn verb_run(runs_root: &std::path::Path, args: RunArgs, quiet: bool) -> Result<(), String> {
    let strategy = build_strategy(args.strategy, args.initial_temperature, args.cooling_rate);
    let budget = build_budget(args.soft_seconds, args.hard_seconds)?;
    std::fs::create_dir_all(runs_root)
        .map_err(|e| format!("cannot create runs dir {}: {e}", runs_root.display()))?;

    match args.target {
        TargetKind::Grammar => {
            let mut target = GrammarTarget::default_smoke();
            let outcome = run_experiment(
                &mut target,
                strategy,
                args.iterations,
                budget,
                runs_root,
                args.seed,
            )
            .map_err(|e| format!("run failed: {e}"))?;
            if !quiet {
                print_outcome("grammar", &outcome.run_id, &outcome.log_path, outcome.iterations,
                              outcome.accepted, outcome.best_reward, outcome.aborted_kills);
            }
        }
    }
    Ok(())
}

fn verb_resume(args: ResumeArgs, quiet: bool) -> Result<(), String> {
    let budget = build_budget(args.soft_seconds, args.hard_seconds)?;
    match args.target {
        TargetKind::Grammar => {
            let mut target = GrammarTarget::default_smoke();
            let outcome = resume_experiment(
                &mut target,
                &args.log,
                args.additional_iterations,
                budget,
                args.seed,
            )
            .map_err(|e| format!("resume failed: {e}"))?;
            if !quiet {
                print_outcome("grammar", &outcome.run_id, &outcome.log_path, outcome.iterations,
                              outcome.accepted, outcome.best_reward, outcome.aborted_kills);
            }
        }
    }
    Ok(())
}

fn verb_list(
    runs_root: &std::path::Path,
    milestones_root: &std::path::Path,
    include_milestones: bool,
) -> Result<(), String> {
    let mut runs: Vec<String> = Vec::new();
    if runs_root.is_dir() {
        for entry in std::fs::read_dir(runs_root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name == ".gitkeep" {
                    continue;
                }
                if RunId::parse(&name).is_ok() {
                    runs.push(name);
                }
            }
        }
    }
    runs.sort();

    if runs.is_empty() {
        eprintln!("no runs yet — try: ix-autoresearch run --target grammar");
    } else {
        println!("runs ({}):", runs.len());
        for id in &runs {
            println!("  {id}");
        }
    }

    if include_milestones {
        let mut milestones: Vec<String> = Vec::new();
        if milestones_root.is_dir() {
            for entry in std::fs::read_dir(milestones_root).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if name == ".gitkeep" {
                    continue;
                }
                // Only show *complete* milestones — half-built dirs are skipped.
                if is_complete_milestone(&path) {
                    milestones.push(name);
                }
            }
        }
        milestones.sort();
        if milestones.is_empty() {
            if !runs.is_empty() {
                println!("milestones: none (use `promote <run-id> --note <slug>`)");
            }
        } else {
            println!("milestones ({}):", milestones.len());
            for m in milestones {
                println!("  {m}");
            }
        }
    }
    Ok(())
}

fn verb_promote(
    runs_root: &std::path::Path,
    milestones_root: &std::path::Path,
    args: PromoteArgs,
    quiet: bool,
) -> Result<(), String> {
    let dst = promote_run(runs_root, milestones_root, &args.run_id, &args.note, args.force)
        .map_err(|e| format!("promote failed: {e}"))?;
    if !quiet {
        println!("promoted: {}", dst.display());
    }
    Ok(())
}

fn verb_revert(runs_root: &std::path::Path, args: RevertArgs) -> Result<(), String> {
    // Validate the run-id (defends against path traversal).
    let canonical = RunId::parse(&args.run_id)
        .map_err(|e| format!("invalid run-id: {e}"))?
        .as_string();
    let log_path = runs_root.join(&canonical).join("log.jsonl");
    if !log_path.is_file() {
        return Err(format!(
            "no log at {} (does the run exist?)",
            log_path.display()
        ));
    }
    let events: Vec<LogEvent<GrammarConfig, GrammarScore>> =
        read_log(&log_path).map_err(|e| format!("cannot read log: {e}"))?;
    for ev in &events {
        if let LogEvent::Iteration {
            iteration, config, ..
        } = ev
        {
            if *iteration == args.to {
                let json = serde_json::to_string_pretty(config)
                    .map_err(|e| format!("cannot serialize config: {e}"))?;
                println!("{json}");
                return Ok(());
            }
        }
    }
    Err(format!(
        "iteration {} not found in run {}",
        args.to, canonical
    ))
}

// ───────────────────────── helpers ─────────────────────────

fn build_strategy(
    kind: StrategyKind,
    initial_temperature: Option<f64>,
    cooling_rate: f64,
) -> Strategy {
    match kind {
        StrategyKind::Greedy => Strategy::Greedy,
        StrategyKind::Sa => Strategy::SimulatedAnnealing {
            initial_temperature,
            cooling_rate,
        },
        StrategyKind::Random => Strategy::RandomSearch,
    }
}

fn build_budget(soft_seconds: f64, hard_seconds: Option<f64>) -> Result<TimeBudget, String> {
    if !soft_seconds.is_finite() || soft_seconds <= 0.0 {
        return Err(format!("--soft-seconds must be > 0, got {soft_seconds}"));
    }
    let soft = Duration::from_secs_f64(soft_seconds);
    match hard_seconds {
        None => Ok(TimeBudget::soft(soft)),
        Some(h) if h.is_finite() && h > 0.0 => {
            Ok(TimeBudget::soft_and_hard(soft, Duration::from_secs_f64(h)))
        }
        Some(h) => Err(format!("--hard-seconds must be > 0, got {h}")),
    }
}

fn print_outcome(
    target: &str,
    run_id: &RunId,
    log_path: &std::path::Path,
    iterations: usize,
    accepted: usize,
    best_reward: Option<f64>,
    aborted_kills: Option<usize>,
) {
    println!("run_id:       {run_id}");
    println!("target:       {target}");
    println!("iterations:   {iterations}");
    println!("accepted:     {accepted}");
    println!(
        "best_reward:  {}",
        best_reward
            .map(|r| format!("{r:.6}"))
            .unwrap_or_else(|| "<none>".to_string())
    );
    if let Some(n) = aborted_kills {
        println!("ABORTED:      {n} consecutive hard kills");
    }
    println!("log:          {}", log_path.display());
    println!();
    println!("Sanity:       MCP iteration cap is {MCP_ITERATION_CAP}; CLI is unconstrained.");
}
