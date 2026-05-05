use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub index_path: String,
    pub output_dir: PathBuf,
    pub artifact_id: String,
    pub dict_size: u32,
    pub k_sparse: u32,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub seed: u64,
    pub held_out_pct: f64,
    pub retry_note: Option<String>,
}

/// Platform-appropriate default Python interpreter name.
/// Windows ships a `python3.exe` Store stub that has no packages; users typically
/// install via the python.org installer which provides `python.exe`. POSIX
/// systems follow PEP 394 — `python3` is the canonical name.
pub const fn default_python_bin() -> &'static str {
    if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    }
}

/// Exit code the Python trainer emits when dead_features_pct > 30%.
pub const EXIT_DEAD_FEATURES: i32 = 3;
/// Exit code the Python trainer emits when reconstruction_mse > 0.05.
pub const EXIT_MSE_FAIL: i32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    #[error("failed to launch Python subprocess: {0}")]
    Spawn(#[from] std::io::Error),

    #[error(
        "reconstruction_mse exceeded guardrail 0.05 — Python trainer exited with code 2. \
         No artifact was written. Check training.log for diagnostics."
    )]
    MseGuardrailExceeded,

    #[error(
        "dead_features_pct > 30% on retry run (dict_size already reduced). \
         No artifact emitted. This corpus may be too small or too low-rank for the SAE."
    )]
    DeadFeaturesAfterRetry,

    #[error("Python trainer exited with unexpected code {code}")]
    UnexpectedExitCode { code: i32 },

    #[error("Python trainer was killed by signal")]
    KilledBySignal,
}

/// Runs the Python SAE trainer as a subprocess.
///
/// Returns:
/// - `Ok(())` — artifact written to `config.output_dir/optick-sae-artifact.json`
/// - `Err(TrainerError::MseGuardrailExceeded)` — caller should abort
/// - `Err(TrainerError::DeadFeaturesAfterRetry)` — dead features exit, caller decides on retry
///
/// If dead_features_pct > 30% the Python process exits with code `EXIT_DEAD_FEATURES` (3)
/// *without* writing an artifact. The CLI handles the retry loop.
pub fn run_python_trainer(
    script: &Path,
    config: &TrainConfig,
    python_bin: &str,
) -> Result<(), TrainerError> {
    let mut cmd = Command::new(python_bin);
    cmd.arg(script)
        .arg("--index")
        .arg(&config.index_path)
        .arg("--output-dir")
        .arg(&config.output_dir)
        .arg("--artifact-id")
        .arg(&config.artifact_id)
        .arg("--dict-size")
        .arg(config.dict_size.to_string())
        .arg("--k-sparse")
        .arg(config.k_sparse.to_string())
        .arg("--epochs")
        .arg(config.epochs.to_string())
        .arg("--batch-size")
        .arg(config.batch_size.to_string())
        .arg("--lr")
        .arg(config.lr.to_string())
        .arg("--seed")
        .arg(config.seed.to_string())
        .arg("--held-out-pct")
        .arg(config.held_out_pct.to_string());

    if let Some(note) = &config.retry_note {
        cmd.arg("--retry-note").arg(note);
    }

    // Inherit stdout/stderr so training progress appears in the terminal.
    let status = cmd.status()?;

    match status.code() {
        Some(0) => Ok(()),
        Some(EXIT_MSE_FAIL) => Err(TrainerError::MseGuardrailExceeded),
        Some(EXIT_DEAD_FEATURES) => Err(TrainerError::UnexpectedExitCode { code: EXIT_DEAD_FEATURES }),
        Some(code) => Err(TrainerError::UnexpectedExitCode { code }),
        None => Err(TrainerError::KilledBySignal),
    }
}
