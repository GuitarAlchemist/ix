//! Phase 7 validation harness — live OPTIC-K weight tuning.
//!
//! Runs `OpticKTarget::ci_reduced` (real GA CLI + invariants + diag
//! binaries) under `Strategy::SimulatedAnnealing` for N iterations and
//! K seeds. Reports per-seed baseline-vs-final delta on
//! `structure_leak_pct` and the lex-best config across all runs.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example phase7_validate -- \
//!     --ga-cli       <path-to-FretboardVoicingsCLI[.exe]> \
//!     --invariants   <path-to-ix-optick-invariants[.exe]> \
//!     --diagnostics  <path-to-baseline-diagnostics[.exe]> \
//!     --iterations 30 \
//!     --seeds 3 \
//!     --export-max 2000
//! ```
//!
//! ## Smoke run
//!
//! `--iterations 3 --seeds 1` is the smoke configuration — completes
//! in ~15 s and verifies the live pipeline parses scores cleanly.
//!
//! ## What "conclusive" looks like
//!
//! For each seed:
//! - record `structure_leak_pct` of the *baseline* config (uniform 1/6)
//!   on iter 0,
//! - record the lex-best config's leak across all iters,
//! - delta = baseline_leak − best_leak (positive ⇒ improvement).
//!
//! A single seed showing 1pp+ improvement is suggestive but noisy. The
//! multi-seed table makes the claim falsifiable: if the seed-to-seed
//! stddev of "delta" is comparable to the mean, the loop is just chasing
//! noise. If stddev ≪ mean, the optimization is real.

use std::path::PathBuf;
use std::time::Duration;

use ix_autoresearch::{
    log::read_log, run_experiment, CIReducedConfig, LogEvent, OpticKConfig, OpticKScore,
    OpticKTarget, Strategy, TimeBudget,
};
use tempfile::TempDir;

#[derive(Clone, Copy, Debug)]
enum Baseline {
    Production,
    Uniform,
}

struct Args {
    ga_cli: PathBuf,
    invariants: PathBuf,
    diagnostics: PathBuf,
    iterations: u32,
    seeds: u32,
    export_max: u32,
    base_seed: u64,
    baseline: Baseline,
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut ga_cli: Option<PathBuf> = None;
    let mut invariants: Option<PathBuf> = None;
    let mut diagnostics: Option<PathBuf> = None;
    let mut iterations: u32 = 30;
    let mut seeds: u32 = 3;
    let mut export_max: u32 = 2000;
    let mut base_seed: u64 = 20_260_429;
    // Default to production weights — that's the question worth asking
    // ("can the loop beat what's deployed?"). Pass --baseline uniform to
    // reproduce the original 2026-04-29 weak-signal run.
    let mut baseline = Baseline::Production;
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--ga-cli" => ga_cli = argv.next().map(PathBuf::from),
            "--invariants" => invariants = argv.next().map(PathBuf::from),
            "--diagnostics" => diagnostics = argv.next().map(PathBuf::from),
            "--iterations" => iterations = argv.next().and_then(|s| s.parse().ok()).unwrap_or(30),
            "--seeds" => seeds = argv.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "--export-max" => export_max = argv.next().and_then(|s| s.parse().ok()).unwrap_or(2000),
            "--base-seed" => {
                base_seed = argv.next().and_then(|s| s.parse().ok()).unwrap_or(base_seed)
            }
            "--baseline" => {
                baseline = match argv.next().as_deref() {
                    Some("production") => Baseline::Production,
                    Some("uniform") => Baseline::Uniform,
                    Some(other) => panic!("--baseline must be 'production' or 'uniform'; got '{other}'"),
                    None => panic!("--baseline requires an argument"),
                }
            }
            other => panic!("unknown flag: {other}"),
        }
    }
    Args {
        ga_cli: ga_cli.expect("--ga-cli required"),
        invariants: invariants.expect("--invariants required"),
        diagnostics: diagnostics.expect("--diagnostics required"),
        iterations,
        seeds,
        export_max,
        base_seed,
        baseline,
    }
}

#[derive(Debug, Clone)]
struct SeedResult {
    seed: u64,
    baseline_leak: f64,
    best_leak: f64,
    best_cfg: OpticKConfig,
    accepted: u32,
    eval_failures: u32,
    elapsed_secs: f64,
    iters: Vec<(f64, f64)>, // (leak, retr) per iter
}

fn run_seed(args: &Args, seed: u64, idx: u32) -> SeedResult {
    let workdir = TempDir::new().expect("workdir");
    let log_dir = TempDir::new().expect("logdir");

    let mut cfg = CIReducedConfig::new(
        &args.ga_cli,
        &args.invariants,
        &args.diagnostics,
        workdir.path(),
    );
    cfg.export_max = args.export_max;

    let baseline_cfg = match args.baseline {
        Baseline::Production => OpticKTarget::production_weights(),
        Baseline::Uniform => OpticKConfig::from_array([1.0 / 6.0; 6]),
    };
    let mut target = OpticKTarget::ci_reduced(cfg).with_baseline(baseline_cfg);

    let start = std::time::Instant::now();
    eprintln!(
        "  seed {} ({}/{}): running {} iters of SimulatedAnnealing...",
        seed,
        idx + 1,
        args.seeds,
        args.iterations
    );
    let outcome = run_experiment(
        &mut target,
        Strategy::SimulatedAnnealing {
            initial_temperature: None, // calibrated from first eval
            cooling_rate: 0.95,
        },
        args.iterations as usize,
        TimeBudget::soft(Duration::from_secs(args.iterations as u64 * 30)),
        log_dir.path(),
        seed,
    )
    .expect("run_experiment");
    let elapsed_secs = start.elapsed().as_secs_f64();

    let events: Vec<LogEvent<OpticKConfig, OpticKScore>> =
        read_log(&outcome.log_path).expect("read_log");

    let mut baseline_leak: f64 = f64::NAN;
    let mut best_leak = f64::INFINITY;
    let mut best_cfg = OpticKConfig::from_array([1.0 / 6.0; 6]);
    let mut iters: Vec<(f64, f64)> = Vec::new();
    let mut iter_idx = 0usize;
    for ev in &events {
        if let LogEvent::Iteration {
            score: Some(score),
            config,
            ..
        } = ev
        {
            if iter_idx == 0 {
                baseline_leak = score.structure_leak_pct;
            }
            iters.push((score.structure_leak_pct, score.retrieval_match_pct));
            if score.structure_leak_pct < best_leak {
                best_leak = score.structure_leak_pct;
                best_cfg = config.clone();
            }
            iter_idx += 1;
        }
    }
    if !best_leak.is_finite() {
        // No accepted iter produced a score — fall back to baseline.
        best_leak = baseline_leak;
    }

    SeedResult {
        seed,
        baseline_leak,
        best_leak,
        best_cfg,
        accepted: outcome.accepted as u32,
        eval_failures: outcome.cost.eval_failure_count,
        elapsed_secs,
        iters,
    }
}

fn fmt_cfg(c: &OpticKConfig) -> String {
    format!(
        "S={:.3} M={:.3} C={:.3} Y={:.3} L={:.3} R={:.3}",
        c.structure_weight,
        c.morphology_weight,
        c.context_weight,
        c.symbolic_weight,
        c.modal_weight,
        c.root_weight,
    )
}

fn mean_stddev(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn main() {
    let args = parse_args();
    for (label, p) in [
        ("--ga-cli", &args.ga_cli),
        ("--invariants", &args.invariants),
        ("--diagnostics", &args.diagnostics),
    ] {
        if !p.exists() {
            eprintln!("error: {label} {p:?} does not exist");
            std::process::exit(2);
        }
    }

    println!("=== Phase 7 validation: live OPTIC-K weight tuning ===");
    println!("  GA CLI:        {}", args.ga_cli.display());
    println!("  invariants:    {}", args.invariants.display());
    println!("  diagnostics:   {}", args.diagnostics.display());
    println!("  iterations:    {} per seed", args.iterations);
    println!("  seeds:         {}", args.seeds);
    println!("  --export-max:  {}", args.export_max);
    println!(
        "  baseline:      {} ({})",
        match args.baseline {
            Baseline::Production => "production",
            Baseline::Uniform => "uniform 1/6",
        },
        match args.baseline {
            Baseline::Production => "S=0.30 M=0.25 C=0.15 Y=0.10 L=0.15 R=0.05",
            Baseline::Uniform => "S=0.167 M=0.167 C=0.167 Y=0.167 L=0.167 R=0.167",
        }
    );
    println!();

    let mut results: Vec<SeedResult> = Vec::with_capacity(args.seeds as usize);
    for i in 0..args.seeds {
        let seed = args.base_seed.wrapping_add(i as u64 * 1_000_003);
        results.push(run_seed(&args, seed, i));
    }

    println!();
    println!("=== Per-seed trajectories (structure_leak_pct each iter) ===");
    for r in &results {
        let traj: String = r
            .iters
            .iter()
            .map(|(leak, _)| format!("{leak:.3}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  seed {}: {traj}", r.seed);
    }

    println!();
    println!("=== Per-seed summary ===");
    println!(
        "  {:>16}  {:>10}  {:>10}  {:>10}  {:>5}  {:>4}  {:>8}",
        "seed", "baseline", "best", "delta(pp)", "acc", "fail", "elapsed"
    );
    for r in &results {
        let delta_pp = (r.baseline_leak - r.best_leak) * 100.0;
        println!(
            "  {:>16}  {:>10.4}  {:>10.4}  {:>10.2}  {:>5}  {:>4}  {:>7.1}s",
            r.seed,
            r.baseline_leak,
            r.best_leak,
            delta_pp,
            r.accepted,
            r.eval_failures,
            r.elapsed_secs
        );
    }

    let baselines: Vec<f64> = results.iter().map(|r| r.baseline_leak).collect();
    let bests: Vec<f64> = results.iter().map(|r| r.best_leak).collect();
    let deltas_pp: Vec<f64> = results
        .iter()
        .map(|r| (r.baseline_leak - r.best_leak) * 100.0)
        .collect();
    let (b_mean, b_std) = mean_stddev(&baselines);
    let (best_mean, best_std) = mean_stddev(&bests);
    let (d_mean, d_std) = mean_stddev(&deltas_pp);

    println!();
    println!("=== Across seeds ===");
    println!("  baseline leak: μ={b_mean:.4}  σ={b_std:.4}");
    println!("  best leak:     μ={best_mean:.4}  σ={best_std:.4}");
    println!("  delta (pp):    μ={d_mean:.2}    σ={d_std:.2}");
    let signal_to_noise = if d_std > 1e-9 {
        d_mean.abs() / d_std
    } else {
        f64::INFINITY
    };
    println!(
        "  signal/noise:  {signal_to_noise:.2}  ({})",
        if signal_to_noise >= 2.0 {
            "real signal (≥ 2σ)"
        } else if signal_to_noise >= 1.0 {
            "weak signal (1-2σ)"
        } else {
            "noise — loop did NOT find a real improvement"
        }
    );

    if let Some(overall_best) = results
        .iter()
        .min_by(|a, b| a.best_leak.partial_cmp(&b.best_leak).unwrap())
    {
        println!();
        println!("=== Lex-best config across all seeds ===");
        println!("  seed:        {}", overall_best.seed);
        println!("  leak_pct:    {:.4}", overall_best.best_leak);
        println!("  cfg:         {}", fmt_cfg(&overall_best.best_cfg));
    }
}
