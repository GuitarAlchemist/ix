use std::fs;
use std::path::PathBuf;

use chrono::Utc;
use clap::{Parser, Subcommand};
use uuid::{NoContext, Timestamp, Uuid};

use ix_optick_sae::trainer::{
    run_python_trainer, TrainConfig, TrainerError, EXIT_DEAD_FEATURES,
};
use ix_optick_sae::{validate_artifact, SaeArtifact, DEAD_FEATURES_PCT_GUARDRAIL};

// Resolved at compile time so the binary always knows where its Python trainer lives.
const DEFAULT_PYTHON_SCRIPT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/python/train.py");

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
    #[arg(long, default_value_t = 100)]
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
    };

    match run_python_trainer(&args.python_script, &config) {
        Ok(()) => finish(args.output, &artifact_id),
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

            match run_python_trainer(&args.python_script, &config) {
                Ok(()) => finish(args.output, &artifact_id),
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

    let artifact: SaeArtifact = serde_json::from_str(&json)
        .map_err(|e| format!("artifact JSON is malformed: {e}"))?;

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
