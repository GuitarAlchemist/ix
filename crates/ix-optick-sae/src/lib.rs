pub mod trainer;

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const RECONSTRUCTION_MSE_GUARDRAIL: f64 = 0.05;
pub const DEAD_FEATURES_PCT_GUARDRAIL: f64 = 30.0;

// Canonical Phase 1 partition set — similarity-relevant only.
// IDENTITY (0..6) is excluded because it encodes the lowest pitch's (octave, pitch_class)
// as identity tags, not similarity features. ROOT (228..240) is included because it
// carries chord-root identity which is critical for similarity comparisons.
// Source: state/quality/optick-sae/2026-05-04/optick-sae-artifact.json (canonical baseline).
pub const PHASE1_PARTITIONS: &[&str] = &[
    "STRUCTURE", "MORPHOLOGY", "CONTEXT", "SYMBOLIC", "MODAL", "ROOT",
];

// ── Artifact JSON shape (mirrors optick-sae-artifact.schema.json v0.1) ─────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaeArtifact {
    pub schema_version: u32,
    pub artifact_id: String,
    pub trained_at: String,
    /// Enum: "ix-optick-sae" | "manual"
    pub trainer: String,
    pub trainer_version: String,
    pub input: InputMeta,
    pub model: ModelConfig,
    pub metrics: SaeMetrics,
    pub features_summary: FeaturesSummary,
    pub links: ArtifactLinks,
    pub narrative: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMeta {
    pub optick_index_path: String,
    /// Must match `^sha256:[0-9a-f]{64}$`
    pub optick_index_sha: String,
    /// 240 for OPTIC-K v1.8. Read from EmbeddingSchema.TotalDimension; do not hardcode.
    pub optick_dim: u32,
    /// "OPTIC-K-v1.8"
    pub schema_version: String,
    pub corpus_size: u64,
    pub partitions_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// "topk_sae" | "relu_sae" | "gated_sae"
    pub kind: String,
    pub dict_size: u32,
    pub k_sparse: u32,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub seed: u64,
    pub loss_final: f64,
    pub sparsity_actual_mean: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaeMetrics {
    /// Guardrail: must be ≤ 0.05 or artifact is not emitted.
    pub reconstruction_mse: f64,
    pub reconstruction_r2: f64,
    pub active_features_per_voicing_p50: u32,
    pub active_features_per_voicing_p95: u32,
    /// Guardrail: > 30% triggers retry with dict_size=512.
    pub dead_features_pct: f64,
    pub feature_partition_purity_mean: f64,
    pub feature_partition_purity_p10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesSummary {
    pub total: u32,
    pub alive: u32,
    pub high_frequency_count: u32,
    pub low_frequency_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLinks {
    pub feature_activations_parquet: String,
    pub feature_manifest_jsonl: String,
    pub training_log: String,
    pub model_weights: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("schema_version must be {expected}, got {got}")]
    WrongSchemaVersion { expected: u32, got: u32 },

    #[error(
        "artifact_id '{id}' is not filename-safe (must not contain ':' or '/')"
    )]
    InvalidArtifactId { id: String },

    #[error(
        "reconstruction_mse {mse:.8} exceeds guardrail {guardrail:.2} — artifact not emitted"
    )]
    ReconstructionMseTooHigh { mse: f64, guardrail: f64 },

    #[error(
        "dead_features_pct {pct:.2}% exceeds guardrail {guardrail:.1}% — artifact not emitted"
    )]
    DeadFeaturesTooHigh { pct: f64, guardrail: f64 },

    #[error("trainer must be 'ix-optick-sae' or 'manual', got '{0}'")]
    UnknownTrainer(String),

    #[error("narrative is empty or exceeds 500 characters (len={0})")]
    InvalidNarrative(usize),
}

/// Validates the artifact against contract guardrails.
/// Called by the CLI after reading the JSON written by the Python subprocess.
pub fn validate_artifact(artifact: &SaeArtifact) -> Result<(), ValidationError> {
    if artifact.schema_version != SCHEMA_VERSION {
        return Err(ValidationError::WrongSchemaVersion {
            expected: SCHEMA_VERSION,
            got: artifact.schema_version,
        });
    }

    if artifact.artifact_id.contains(':') || artifact.artifact_id.contains('/') {
        return Err(ValidationError::InvalidArtifactId {
            id: artifact.artifact_id.clone(),
        });
    }

    if !matches!(artifact.trainer.as_str(), "ix-optick-sae" | "manual") {
        return Err(ValidationError::UnknownTrainer(artifact.trainer.clone()));
    }

    let n = artifact.narrative.len();
    if n == 0 || n > 500 {
        return Err(ValidationError::InvalidNarrative(n));
    }

    if artifact.metrics.reconstruction_mse > RECONSTRUCTION_MSE_GUARDRAIL {
        return Err(ValidationError::ReconstructionMseTooHigh {
            mse: artifact.metrics.reconstruction_mse,
            guardrail: RECONSTRUCTION_MSE_GUARDRAIL,
        });
    }

    if artifact.metrics.dead_features_pct > DEAD_FEATURES_PCT_GUARDRAIL {
        return Err(ValidationError::DeadFeaturesTooHigh {
            pct: artifact.metrics.dead_features_pct,
            guardrail: DEAD_FEATURES_PCT_GUARDRAIL,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub_artifact(mse: f64, dead_pct: f64) -> SaeArtifact {
        SaeArtifact {
            schema_version: 1,
            artifact_id: "optick-sae-2026-05-03T12-00-00Z-abcd1234-topk-sae".into(),
            trained_at: "2026-05-03T12:00:00Z".into(),
            trainer: "ix-optick-sae".into(),
            trainer_version: "0.1.0".into(),
            input: InputMeta {
                optick_index_path: "synthetic".into(),
                optick_index_sha: format!("sha256:{}", "a".repeat(64)),
                optick_dim: 240,
                schema_version: "OPTIC-K-v1.8".into(),
                corpus_size: 1000,
                partitions_used: PHASE1_PARTITIONS.iter().map(|s| s.to_string()).collect(),
            },
            model: ModelConfig {
                kind: "topk_sae".into(),
                dict_size: 1024,
                k_sparse: 32,
                training: TrainingConfig {
                    epochs: 100,
                    batch_size: 256,
                    lr: 1e-3,
                    seed: 42,
                    loss_final: 0.001,
                    sparsity_actual_mean: 0.03125,
                },
            },
            metrics: SaeMetrics {
                reconstruction_mse: mse,
                reconstruction_r2: 0.95,
                active_features_per_voicing_p50: 32,
                active_features_per_voicing_p95: 32,
                dead_features_pct: dead_pct,
                feature_partition_purity_mean: 0.6,
                feature_partition_purity_p10: 0.3,
            },
            features_summary: FeaturesSummary {
                total: 1024,
                alive: 900,
                high_frequency_count: 200,
                low_frequency_count: 50,
            },
            links: ArtifactLinks {
                feature_activations_parquet: "feature_activations.parquet".into(),
                feature_manifest_jsonl: "feature_manifest.jsonl".into(),
                training_log: "training.log".into(),
                model_weights: "sae_weights.safetensors".into(),
                supersedes: None,
            },
            narrative: "Phase 1 smoke run.".into(),
        }
    }

    #[test]
    fn valid_artifact_passes() {
        assert!(validate_artifact(&stub_artifact(0.01, 10.0)).is_ok());
    }

    #[test]
    fn mse_at_guardrail_boundary_passes() {
        assert!(validate_artifact(&stub_artifact(0.05, 10.0)).is_ok());
    }

    #[test]
    fn mse_above_guardrail_fails() {
        let err = validate_artifact(&stub_artifact(0.051, 10.0)).unwrap_err();
        assert!(matches!(err, ValidationError::ReconstructionMseTooHigh { .. }));
    }

    #[test]
    fn dead_features_at_guardrail_boundary_passes() {
        assert!(validate_artifact(&stub_artifact(0.01, 30.0)).is_ok());
    }

    #[test]
    fn dead_features_above_guardrail_fails() {
        let err = validate_artifact(&stub_artifact(0.01, 30.1)).unwrap_err();
        assert!(matches!(err, ValidationError::DeadFeaturesTooHigh { .. }));
    }

    #[test]
    fn colon_in_artifact_id_fails() {
        let mut a = stub_artifact(0.01, 10.0);
        a.artifact_id = "optick-sae-2026-05-03T12:00:00Z-abc-topk-sae".into();
        assert!(matches!(
            validate_artifact(&a).unwrap_err(),
            ValidationError::InvalidArtifactId { .. }
        ));
    }

    #[test]
    fn roundtrip_serde() {
        let a = stub_artifact(0.01, 10.0);
        let json = serde_json::to_string_pretty(&a).unwrap();
        let back: SaeArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.artifact_id, a.artifact_id);
        assert!((back.metrics.reconstruction_mse - a.metrics.reconstruction_mse).abs() < 1e-12);
    }
}
