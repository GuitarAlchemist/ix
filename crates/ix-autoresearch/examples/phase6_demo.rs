//! Phase 6 end-to-end demo — IX↔GA cross-repo loop.
//!
//! Runs the real `FretboardVoicingsCLI --weights-config` flag over N
//! Dirichlet-sampled simplex weight vectors, captures index size +
//! elapsed time per iteration, picks a lex-best winner, and prints a
//! summary table.
//!
//! This is the Phase-7-lite preview: same plumbing the live Target A
//! loop will use, but with a tiny corpus (`--export-max 200`) so it
//! finishes in under a minute on a laptop.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example phase6_demo -- \
//!     --ga-cli "C:/Users/spare/source/repos/ga/Demos/Music Theory/FretboardVoicingsCLI/bin/Release/net10.0/FretboardVoicingsCLI.exe" \
//!     --iterations 5
//! ```
//!
//! On Linux/macOS, omit `.exe`. The CLI must be a Release build.
//!
//! ## What it shows
//!
//! - JSON weights file written atomically per iteration
//! - GA CLI invoked with `--weights-config <path>`, stderr captured
//! - Index file produced; size + entry count parsed
//! - Lex-best iteration printed (tie-breaking on elapsed time)

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use ix_autoresearch::OpticKConfig;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Dirichlet, Distribution};
use tempfile::TempDir;

const PRODUCTION_WEIGHTS: [f64; 6] = [0.45, 0.25, 0.20, 0.10, 0.10, 0.05];
const ALPHA: f64 = 200.0;
const FLOOR: f64 = 1e-3;
const EXPORT_MAX: u32 = 200;

struct Args {
    ga_cli: PathBuf,
    iterations: u32,
    seed: u64,
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut ga_cli: Option<PathBuf> = None;
    let mut iterations: u32 = 5;
    let mut seed: u64 = 2026_04_27;
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--ga-cli" => ga_cli = argv.next().map(PathBuf::from),
            "--iterations" => iterations = argv.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            "--seed" => seed = argv.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            other => panic!("unknown flag: {other}"),
        }
    }
    Args {
        ga_cli: ga_cli.expect("--ga-cli <path> required"),
        iterations,
        seed,
    }
}

/// Renormalize-with-floor then Dirichlet-perturb around `base`.
fn perturb(base: &OpticKConfig, alpha: f64, floor: f64, rng: &mut ChaCha8Rng) -> OpticKConfig {
    // Floor + renormalize the base so no component is near 0 (absorbing).
    let mut a = base.as_array();
    for w in &mut a {
        if *w < floor {
            *w = floor;
        }
    }
    let s: f64 = a.iter().sum();
    for w in &mut a {
        *w /= s;
    }
    // Dirichlet centered on a, concentration α.
    let scaled: [f64; 6] = [
        a[0] * alpha,
        a[1] * alpha,
        a[2] * alpha,
        a[3] * alpha,
        a[4] * alpha,
        a[5] * alpha,
    ];
    let dir = Dirichlet::<f64, 6>::new(scaled).expect("Dirichlet params positive");
    let sample = dir.sample(rng);
    OpticKConfig::from_array(sample)
}

fn write_weights_json(cfg: &OpticKConfig, path: &std::path::Path) -> std::io::Result<()> {
    let json = serde_json::json!({
        "schema_version": 1,
        "structure_weight":  cfg.structure_weight,
        "morphology_weight": cfg.morphology_weight,
        "context_weight":    cfg.context_weight,
        "symbolic_weight":   cfg.symbolic_weight,
        "modal_weight":      cfg.modal_weight,
        "root_weight":       cfg.root_weight,
    });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&json).unwrap())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Debug)]
struct IterResult {
    cfg: OpticKConfig,
    index_bytes: u64,
    elapsed_ms: u128,
    accepted: bool,
    diagnostic: String,
}

fn run_one_iter(
    ga_cli: &std::path::Path,
    cfg: &OpticKConfig,
    workdir: &std::path::Path,
    iter: u32,
) -> IterResult {
    let weights_path = workdir.join(format!("weights-{iter:02}.json"));
    let index_path = workdir.join(format!("index-{iter:02}.optk"));
    write_weights_json(cfg, &weights_path).expect("write weights");

    let started = Instant::now();
    let output = Command::new(ga_cli)
        .args([
            "--export-embeddings",
            "--tuning",
            "guitar",
            "--export-max",
            &EXPORT_MAX.to_string(),
            "--weights-config",
            weights_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .output()
        .expect("invoke GA CLI");
    let elapsed_ms = started.elapsed().as_millis();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let accepted = output.status.success() && index_path.exists();
    let index_bytes = if accepted {
        std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };
    let diagnostic = if accepted {
        stderr
            .lines()
            .find(|l| l.contains("OVERRIDE applied"))
            .unwrap_or("(no override line)")
            .trim()
            .to_string()
    } else {
        stderr
            .lines()
            .find(|l| l.to_lowercase().contains("error") || l.contains("Exception"))
            .unwrap_or("(no error line)")
            .chars()
            .take(120)
            .collect()
    };
    IterResult {
        cfg: cfg.clone(),
        index_bytes,
        elapsed_ms,
        accepted,
        diagnostic,
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

fn main() {
    let args = parse_args();
    if !args.ga_cli.exists() {
        eprintln!("error: --ga-cli {:?} does not exist", args.ga_cli);
        std::process::exit(2);
    }

    let workdir = TempDir::new().expect("tempdir");
    println!("=== Phase 6 demo: IX↔GA cross-repo edit-eval-iterate ===");
    println!("  GA CLI:      {}", args.ga_cli.display());
    println!("  iterations:  {}", args.iterations);
    println!("  seed:        {}", args.seed);
    println!("  --export-max {} per iter (small for demo speed)", EXPORT_MAX);
    println!("  workdir:     {}", workdir.path().display());
    println!();

    let base = OpticKConfig::from_array(PRODUCTION_WEIGHTS);
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let mut results: Vec<IterResult> = Vec::with_capacity(args.iterations as usize);
    for iter in 0..args.iterations {
        let cfg = if iter == 0 {
            // First iter is the production baseline — sanity check.
            base.clone()
        } else {
            perturb(&base, ALPHA, FLOOR, &mut rng)
        };
        println!("iter {iter:02}  cfg: {}", fmt_cfg(&cfg));
        let r = run_one_iter(&args.ga_cli, &cfg, workdir.path(), iter);
        let status = if r.accepted { "OK" } else { "FAIL" };
        println!(
            "          [{status}]  bytes={}  elapsed={}ms  {}",
            r.index_bytes, r.elapsed_ms, r.diagnostic
        );
        results.push(r);
    }

    println!();
    println!("=== Summary ===");
    let accepted: Vec<&IterResult> = results.iter().filter(|r| r.accepted).collect();
    println!(
        "  accepted: {}/{}",
        accepted.len(),
        results.len()
    );
    if let Some(min_bytes) = accepted.iter().map(|r| r.index_bytes).min() {
        println!("  smallest index: {} bytes", min_bytes);
    }
    if let Some(max_bytes) = accepted.iter().map(|r| r.index_bytes).max() {
        println!("  largest index:  {} bytes", max_bytes);
    }
    if let Some(fastest) = accepted.iter().min_by_key(|r| r.elapsed_ms) {
        println!(
            "  fastest iter:   {}ms  {}",
            fastest.elapsed_ms,
            fmt_cfg(&fastest.cfg)
        );
    }

    // Lex-best: smallest index, tie-break on elapsed.
    if let Some(best) = accepted
        .iter()
        .min_by_key(|r| (r.index_bytes, r.elapsed_ms))
    {
        println!();
        println!("=== Lex-best (smallest bytes, fastest tiebreak) ===");
        println!("  bytes={}  elapsed={}ms", best.index_bytes, best.elapsed_ms);
        println!("  cfg: {}", fmt_cfg(&best.cfg));
    }
}
