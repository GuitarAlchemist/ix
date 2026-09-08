use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::Utc;
use clap::{Parser, Subcommand};
use uuid::{NoContext, Timestamp, Uuid};

use ix_optick_sae::trainer::{
    default_python_bin, run_python_trainer, TrainConfig, TrainerError, EXIT_DEAD_FEATURES,
};
use ix_optick_sae::{validate_artifact, SaeArtifact, DEAD_FEATURES_PCT_GUARDRAIL};

// Resolved at compile time so the binary always knows where its Python trainer lives.
const DEFAULT_PYTHON_SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/python/train.py");
// Same trick for the coverage reconciler used by `verify`.
const RECONCILER_SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/python/optick_coverage.py");

#[derive(Parser)]
#[command(
    name = "ix-optick-sae",
    about = "OPTIC-K Sparse Autoencoder orchestrator",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a TopK SAE over an OPTIC-K index (or synthetic corpus) and persist the artifact.
    Train(TrainArgs),

    /// Audit an existing snapshot: does its artifact declare what its parquet covers,
    /// and does the parquet actually match that declaration? (ix #248)
    Verify(VerifyArgs),
}

#[derive(clap::Args)]
struct VerifyArgs {
    /// Snapshot directory holding optick-sae-artifact.json and feature_activations.parquet.
    /// Convention: <ga>/state/quality/optick-sae/<YYYY-MM-DD>/.
    #[arg(long)]
    snapshot: PathBuf,

    /// Live OPTIC-K corpus row count, if known. When given, the artifact's declared
    /// `corpus_n` must match it — this is the declared-vs-world check that catches a
    /// declaration gone stale against an index rebuild. Read it from
    /// `ga_voicing_index_info` (`totalVoicings`).
    #[arg(long)]
    corpus_n: Option<u64>,

    /// Skip the physical parquet reconciliation and report on the declaration only.
    /// Use when the parquet is legitimately unavailable (it is gitignored, so any
    /// fresh checkout lacks it) and you want the declaration verdict anyway.
    #[arg(long)]
    skip_physical: bool,

    /// Python interpreter used for the parquet read (needs pandas + pyarrow).
    #[arg(long, default_value_t = default_python_bin().to_string())]
    python_bin: String,
}

#[derive(clap::Args)]
struct TrainArgs {
    /// Path to optick.index, a .npy dump, or the literal string "synthetic".
    /// When the file does not exist or "synthetic" is passed, a 1 000-voicing
    /// synthetic corpus is generated and the deviation is documented in the artifact.
    #[arg(long, default_value = "synthetic")]
    index: String,

    /// Output directory. Artifact, parquet, weights, and log are written here.
    /// Convention: state/quality/optick-sae/<YYYY-MM-DD>/ in the GA repo.
    #[arg(long)]
    output: PathBuf,

    /// SAE dictionary size for the first training attempt.
    #[arg(long, default_value_t = 1024)]
    dict_size: u32,

    /// Top-k sparsity constraint.
    #[arg(long, default_value_t = 32)]
    k_sparse: u32,

    /// Training epochs.
    #[arg(long, default_value_t = 50)]
    epochs: u32,

    /// Mini-batch size.
    #[arg(long, default_value_t = 256)]
    batch_size: u32,

    /// Adam learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// RNG seed (trainer + synthetic corpus).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Fraction of corpus held out for reconstruction metrics.
    #[arg(long, default_value_t = 0.05)]
    held_out_pct: f64,

    /// Path to the Python trainer script.
    /// Defaults to the python/train.py next to this crate's Cargo.toml.
    #[arg(long, default_value = DEFAULT_PYTHON_SCRIPT)]
    python_script: PathBuf,

    /// Python interpreter to invoke. Default is platform-appropriate
    /// (`python` on Windows because `python3` typically resolves to the
    /// Microsoft Store stub; `python3` on POSIX per PEP 394).
    /// Override to point at a venv: `--python-bin .venv/bin/python`.
    #[arg(long, default_value_t = default_python_bin().to_string())]
    python_bin: String,

    /// Weight on AuxK ghost-grad auxiliary loss (0 = disabled).
    /// Empirical baseline: 0.10 keeps dead_features_pct ≤ 30% on real 313k-voicing corpus.
    /// The Anthropic default of 0.03 is too weak for this corpus.
    #[arg(long, default_value_t = 0.10)]
    aux_alpha: f64,

    /// Top-k_aux features from the dead-feature pool per batch (AuxK ghost grads).
    #[arg(long, default_value_t = 64)]
    aux_k: u32,

    /// artifact_id this run supersedes (written to links.supersedes in the artifact).
    /// Set to the canonical 2026-05-04 baseline ID for production runs.
    #[arg(long)]
    supersedes: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            if let Err(e) = run_train(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Verify(args) => std::process::exit(run_verify(args)),
    }
}

// Exit codes for `verify`. Kept distinct so a caller can tell "this snapshot is
// wrong" from "I could not check it" — collapsing those is how a coverage gap
// goes quiet in the first place.
/// The declaration itself is invalid (or absent) — the ix #248 shape.
const EXIT_DECLARATION_INVALID: i32 = 1;
/// The parquet on disk contradicts the declaration.
const EXIT_RECONCILIATION_FAILED: i32 = 4;
/// The physical check could not be run at all. NOT a pass.
const EXIT_NOT_VERIFIED: i32 = 5;

/// Audits a snapshot in two independent passes.
///
/// 1. **Declaration** — parse `optick-sae-artifact.json` and run the contract
///    guardrails, including `activations_coverage` (`validate_artifact`).
/// 2. **Physical** — hand the snapshot to `python/optick_coverage.py`, which
///    reads the parquet's `optick_row` column back and reconciles it against
///    what the artifact promised.
///
/// Pass 2 is not merely "extra": pass 1 can only check the declaration against
/// itself. A snapshot whose artifact says 297,395 rows while the parquet holds
/// 200,000 is green on pass 1 and red on pass 2.
fn run_verify(args: VerifyArgs) -> i32 {
    let artifact_path = args.snapshot.join("optick-sae-artifact.json");
    eprintln!("verifying snapshot: {}", args.snapshot.display());

    let json = match fs::read_to_string(&artifact_path) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("FAIL: cannot read {}: {e}", artifact_path.display());
            return EXIT_DECLARATION_INVALID;
        }
    };

    let artifact: SaeArtifact = match serde_json::from_str(&json) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FAIL: artifact JSON is malformed: {e}");
            return EXIT_DECLARATION_INVALID;
        }
    };

    let mut declaration_ok = true;
    if let Err(e) = validate_artifact(&artifact) {
        eprintln!("FAIL declaration: {e}");
        declaration_ok = false;
    }

    // Declared-vs-world: the corpus the artifact describes must be the corpus
    // that exists now. An index rebuild between train and consume silently
    // invalidates every optick_row in the parquet.
    if let (Some(live), Some(coverage)) = (args.corpus_n, artifact.activations_coverage.as_ref()) {
        if coverage.corpus_n != live {
            eprintln!(
                "FAIL declaration: activations_coverage.corpus_n ({}) != live corpus ({live}) \
                 — the declaration is stale against the current optick.index",
                coverage.corpus_n
            );
            declaration_ok = false;
        }
    }

    if declaration_ok {
        let coverage = artifact
            .activations_coverage
            .as_ref()
            .expect("validate_artifact rejects a missing coverage block");
        eprintln!(
            "ok declaration: {} split covers {}/{} voicings ({:.2}%); {} held out",
            coverage.optick_row_split,
            coverage.n_train,
            coverage.corpus_n,
            coverage.coverage_pct,
            coverage.n_val,
        );
    }

    if args.skip_physical {
        eprintln!("SKIPPED physical reconciliation (--skip-physical).");
        return if declaration_ok {
            EXIT_NOT_VERIFIED
        } else {
            EXIT_DECLARATION_INVALID
        };
    }

    match run_reconciler(&args.snapshot, &args.python_bin) {
        Ok(true) => {
            if declaration_ok {
                eprintln!("✓ snapshot verified: declaration valid and parquet reconciles.");
                0
            } else {
                EXIT_DECLARATION_INVALID
            }
        }
        // A bad declaration is the root cause when both are red, so report that
        // code — the reconciler's verdicts are printed either way.
        Ok(false) if declaration_ok => EXIT_RECONCILIATION_FAILED,
        Ok(false) => EXIT_DECLARATION_INVALID,
        Err(e) => {
            // Absence of the checker is never evidence of health. Say "not
            // verified" and exit non-zero so no pipeline reads this as green.
            eprintln!(
                "NOT VERIFIED: could not run the parquet reconciler ({e}). \
                 This is not a pass — install pandas + pyarrow for '{}', or pass \
                 --skip-physical to accept a declaration-only verdict.",
                args.python_bin
            );
            EXIT_NOT_VERIFIED
        }
    }
}

/// Shells out to `python/optick_coverage.py <snapshot>`; `Ok(true)` when every
/// reconciliation assertion is green.
fn run_reconciler(snapshot: &Path, python_bin: &str) -> Result<bool, String> {
    let script = PathBuf::from(RECONCILER_SCRIPT);
    let status = Command::new(python_bin)
        .arg(&script)
        .arg(snapshot)
        .status()
        .map_err(|e| format!("failed to launch {python_bin} {}: {e}", script.display()))?;

    // The reconciler's codes mirror the constants in optick_coverage.py. Code 5
    // ("nothing was checked") must map to Err, not Ok(false): a fresh checkout
    // has no parquet, and reporting that as a contradiction would make the
    // distinction exit 5 exists for meaningless.
    match status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        Some(5) => Err("the reconciler could not evaluate this snapshot".to_string()),
        Some(code) => Err(format!("reconciler exited with unexpected code {code}")),
        None => Err("reconciler was killed by a signal".to_string()),
    }
}

fn run_train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let artifact_id = make_artifact_id();
    eprintln!("artifact_id: {artifact_id}");

    fs::create_dir_all(&args.output)?;

    let mut config = TrainConfig {
        index_path: args.index.clone(),
        output_dir: args.output.clone(),
        artifact_id: artifact_id.clone(),
        dict_size: args.dict_size,
        k_sparse: args.k_sparse,
        epochs: args.epochs,
        batch_size: args.batch_size,
        lr: args.lr,
        seed: args.seed,
        held_out_pct: args.held_out_pct,
        retry_note: None,
        aux_alpha: args.aux_alpha,
        aux_k: args.aux_k,
        supersedes: args.supersedes.clone(),
    };

    match run_python_trainer(&args.python_script, &config, &args.python_bin) {
        Ok(()) => finish(args.output, &artifact_id),
        // Surfaced with its own code rather than collapsed into the generic
        // error exit: a reconciliation failure means the outputs on disk are
        // real but undeclared, which a caller may want to handle differently
        // from a crash. Mirrors the Python trainer's exit 4.
        Err(TrainerError::ReconciliationFailed) => {
            eprintln!("FAIL: {}", TrainerError::ReconciliationFailed);
            std::process::exit(4);
        }
        Err(TrainerError::MseGuardrailExceeded) => {
            eprintln!(
                "FAIL: reconstruction_mse > {:.2} — no artifact emitted.",
                ix_optick_sae::RECONSTRUCTION_MSE_GUARDRAIL
            );
            std::process::exit(2);
        }
        Err(TrainerError::UnexpectedExitCode { code }) if code == EXIT_DEAD_FEATURES => {
            // dead_features_pct > 30% — retry with dict_size=512 per contract §5.
            let retry_dict = 512u32;
            eprintln!(
                "dead_features_pct > {:.0}% — retrying with dict_size={retry_dict} (contract §5).",
                DEAD_FEATURES_PCT_GUARDRAIL
            );
            config.dict_size = retry_dict;
            config.retry_note = Some(format!(
                "Initial run had dead_features_pct > {:.0}%; retrained with dict_size={retry_dict}.",
                DEAD_FEATURES_PCT_GUARDRAIL
            ));

            match run_python_trainer(&args.python_script, &config, &args.python_bin) {
                Ok(()) => finish(args.output, &artifact_id),
                // Same code on the retry path, so `exit 4 == reconciliation` holds
                // however the run got here.
                Err(TrainerError::ReconciliationFailed) => {
                    eprintln!("FAIL: {}", TrainerError::ReconciliationFailed);
                    std::process::exit(4);
                }
                Err(TrainerError::MseGuardrailExceeded) => {
                    eprintln!("FAIL: reconstruction_mse > 0.05 on retry — no artifact emitted.");
                    std::process::exit(2);
                }
                Err(TrainerError::UnexpectedExitCode { code }) if code == EXIT_DEAD_FEATURES => {
                    eprintln!(
                        "FAIL: dead_features_pct > {:.0}% even after retry with dict_size={retry_dict}. \
                         No artifact emitted. Corpus may be too small or too low-rank.",
                        DEAD_FEATURES_PCT_GUARDRAIL
                    );
                    std::process::exit(3);
                }
                Err(e) => Err(e.into()),
            }
        }
        Err(e) => Err(e.into()),
    }
}

/// Reads the artifact JSON written by the Python trainer, validates it,
/// and prints a summary. The file is already in the output dir.
fn finish(output_dir: PathBuf, artifact_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let artifact_path = output_dir.join("optick-sae-artifact.json");
    let json = fs::read_to_string(&artifact_path)
        .map_err(|e| format!("cannot read artifact at {}: {e}", artifact_path.display()))?;

    let artifact: SaeArtifact =
        serde_json::from_str(&json).map_err(|e| format!("artifact JSON is malformed: {e}"))?;

    validate_artifact(&artifact).map_err(|e| format!("artifact validation failed: {e}"))?;

    eprintln!("✓ artifact validated: {artifact_id}");
    eprintln!(
        "  reconstruction_mse={:.6}  dead_features_pct={:.1}%  alive={}/{}",
        artifact.metrics.reconstruction_mse,
        artifact.metrics.dead_features_pct,
        artifact.features_summary.alive,
        artifact.features_summary.total,
    );
    eprintln!("  written to: {}", artifact_path.display());

    Ok(())
}

fn make_artifact_id() -> String {
    let ts = Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let uuid = Uuid::new_v7(Timestamp::now(NoContext));
    let short = &uuid.simple().to_string()[..8];
    format!("optick-sae-{ts}-{short}-topk-sae")
}
