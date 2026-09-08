pub mod trainer;

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const RECONSTRUCTION_MSE_GUARDRAIL: f64 = 0.05;
pub const DEAD_FEATURES_PCT_GUARDRAIL: f64 = 30.0;

/// Minimum share of the corpus a train-split artifact may declare covered.
///
/// `feature_activations.parquet` holds the train split only, so it never covers
/// 100% of the corpus — but the shortfall must be small and stated. The trainer
/// holds out 5% by default and is policy-capped at 10%, so anything under 90%
/// means the split changed materially and every consumer's "the parquet is
/// ~the corpus" assumption is broken.
///
/// Mirrors `MIN_COVERAGE_PCT` in `python/optick_coverage.py`; the two are kept
/// in step by `coverage_floor_matches_python_producer`.
/// Revisit trigger: a deliberate `held_out_pct > 0.10` — bump both constants in
/// the same PR. Never widen a consumer join to hide the gap (ix #248).
pub const MIN_COVERAGE_PCT: f64 = 90.0;

/// The only split `feature_activations.parquet` is allowed to key on today.
pub const OPTICK_ROW_SPLIT: &str = "train";

// Canonical Phase 1 partition set — similarity-relevant only.
// IDENTITY (0..6) is excluded because it encodes the lowest pitch's (octave, pitch_class)
// as identity tags, not similarity features. ROOT (228..240) is included because it
// carries chord-root identity which is critical for similarity comparisons.
// Source: state/quality/optick-sae/2026-05-04/optick-sae-artifact.json (canonical baseline).
pub const PHASE1_PARTITIONS: &[&str] = &[
    "STRUCTURE",
    "MORPHOLOGY",
    "CONTEXT",
    "SYMBOLIC",
    "MODAL",
    "ROOT",
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
    /// What `feature_activations.parquet` actually covers.
    ///
    /// `Option` only so the two pre-#248 snapshots (2026-06-14, 2026-07-20)
    /// still parse and can be *reported on*; `validate_artifact` rejects
    /// `None`. An artifact that does not declare its coverage is bug #248 by
    /// definition — a consumer joining `optick_row` against the full index has
    /// no way to know how many rows it silently drops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activations_coverage: Option<ActivationsCoverage>,
    pub model: ModelConfig,
    pub metrics: SaeMetrics,
    pub features_summary: FeaturesSummary,
    pub links: ArtifactLinks,
    pub narrative: String,
}

/// Declared corpus coverage of `feature_activations.parquet`.
///
/// Produced by `python/optick_coverage.py::activations_coverage`. The parquet
/// carries one row per *train* voicing, keyed by `optick_row` — its stable
/// position in the full OPTIC-K index — so `n_train` rows join against an
/// `n_train + n_val` corpus and the val split is legitimately absent. Stating
/// that here is what turns a silent 5% join gap into an assertable fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationsCoverage {
    /// Which split `optick_row` enumerates. Always `"train"` today.
    pub optick_row_split: String,
    /// Rows present in `feature_activations.parquet`.
    pub n_train: u64,
    /// Held-out rows, absent from the parquet by design.
    pub n_val: u64,
    /// Corpus size the split partitions — must equal `input.corpus_size`.
    pub corpus_n: u64,
    /// `100 * n_train / corpus_n`, rounded to 2dp.
    pub coverage_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMeta {
    pub optick_index_path: String,
    /// Must match `^sha256:[0-9a-f]{64}$`
    pub optick_index_sha: String,
    /// OPTIC-K embedding TOTAL dimension. 240 for v1.8. Read from
    /// EmbeddingSchema.TotalDimension; do not hardcode.
    pub optick_dim: u32,
    /// Dimension the SAE actually trained on (compact OPTK = 124 for v1.8).
    /// Optional in contract v0.1.x; equal to `optick_dim` if the trainer
    /// used the full embedding. Disambiguates total-vs-training dim that
    /// caused PR #82's "118-dim" narrative bug. See GA contract v0.1.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_training_dim: Option<u32>,
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

    #[error("artifact_id '{id}' is not filename-safe (must not contain ':' or '/')")]
    InvalidArtifactId { id: String },

    #[error("reconstruction_mse {mse:.8} exceeds guardrail {guardrail:.2} — artifact not emitted")]
    ReconstructionMseTooHigh { mse: f64, guardrail: f64 },

    #[error(
        "dead_features_pct {pct:.2}% exceeds guardrail {guardrail:.1}% — artifact not emitted"
    )]
    DeadFeaturesTooHigh { pct: f64, guardrail: f64 },

    #[error("trainer must be 'ix-optick-sae' or 'manual', got '{0}'")]
    UnknownTrainer(String),

    #[error("narrative is empty or exceeds 500 characters (len={0})")]
    InvalidNarrative(usize),

    #[error(
        "artifact declares no activations_coverage block — feature_activations.parquet's \
         corpus coverage is undeclared, so a consumer joining optick_row against the full \
         index cannot know how many rows it silently drops (ix #248)"
    )]
    MissingActivationsCoverage,

    #[error(
        "activations_coverage.optick_row_split is '{got}', expected '{expected}' — the \
         coverage guardrails do not describe an artifact keyed on any other split"
    )]
    UnknownRowSplit { got: String, expected: String },

    #[error(
        "activations_coverage split does not partition the corpus: \
         n_train({n_train}) + n_val({n_val}) = {sum} != corpus_n({corpus_n})"
    )]
    NonAdditiveSplit {
        n_train: u64,
        n_val: u64,
        sum: u64,
        corpus_n: u64,
    },

    #[error(
        "activations_coverage.corpus_n ({coverage}) != input.corpus_size ({input}) — the \
         coverage block describes a different corpus than the one that was trained on"
    )]
    CoverageCorpusMismatch { coverage: u64, input: u64 },

    #[error(
        "activations_coverage.coverage_pct is {declared:.2}, but 100 * {n_train} / {corpus_n} \
         = {recomputed:.2} — the declared percentage is not the one the counts imply"
    )]
    CoveragePctInconsistent {
        declared: f64,
        recomputed: f64,
        n_train: u64,
        corpus_n: u64,
    },

    #[error(
        "activations_coverage is {pct:.2}% of the corpus, below the {floor:.1}% floor — \
         {missing} of {corpus_n} voicings are absent from feature_activations.parquet. \
         Fix the split or raise the floor deliberately; do not widen consumer joins (ix #248)"
    )]
    CoverageBelowFloor {
        pct: f64,
        floor: f64,
        missing: u64,
        corpus_n: u64,
    },
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

    validate_coverage(artifact)?;

    Ok(())
}

/// Checks the `activations_coverage` declaration for internal consistency and
/// against the coverage floor.
///
/// This is the *declaration-level* half of the #248 guard: everything it can
/// decide from the artifact JSON alone, with no parquet in hand. It deliberately
/// cannot tell whether the parquet matches the declaration — that needs the
/// bytes, and lives in `python/optick_coverage.py::reconcile` (run at produce
/// time and by `ix-optick-sae verify`). Keeping the split explicit is what makes
/// each half honest about its own blind spot.
pub fn validate_coverage(artifact: &SaeArtifact) -> Result<(), ValidationError> {
    let coverage = artifact
        .activations_coverage
        .as_ref()
        .ok_or(ValidationError::MissingActivationsCoverage)?;

    if coverage.optick_row_split != OPTICK_ROW_SPLIT {
        return Err(ValidationError::UnknownRowSplit {
            got: coverage.optick_row_split.clone(),
            expected: OPTICK_ROW_SPLIT.to_string(),
        });
    }

    let sum = coverage.n_train + coverage.n_val;
    if sum != coverage.corpus_n {
        return Err(ValidationError::NonAdditiveSplit {
            n_train: coverage.n_train,
            n_val: coverage.n_val,
            sum,
            corpus_n: coverage.corpus_n,
        });
    }

    // The coverage block and the input block must be talking about one corpus.
    // A mismatch means the declaration went stale against an index rebuild.
    if coverage.corpus_n != artifact.input.corpus_size {
        return Err(ValidationError::CoverageCorpusMismatch {
            coverage: coverage.corpus_n,
            input: artifact.input.corpus_size,
        });
    }

    let recomputed = recomputed_coverage_pct(coverage.n_train, coverage.corpus_n);
    // Both sides round to 2dp, so an exact-enough comparison is right here; the
    // epsilon only absorbs float formatting, not a genuine disagreement.
    if (coverage.coverage_pct - recomputed).abs() > 0.005 {
        return Err(ValidationError::CoveragePctInconsistent {
            declared: coverage.coverage_pct,
            recomputed,
            n_train: coverage.n_train,
            corpus_n: coverage.corpus_n,
        });
    }

    if recomputed < MIN_COVERAGE_PCT {
        return Err(ValidationError::CoverageBelowFloor {
            pct: recomputed,
            floor: MIN_COVERAGE_PCT,
            missing: coverage.corpus_n.saturating_sub(coverage.n_train),
            corpus_n: coverage.corpus_n,
        });
    }

    Ok(())
}

/// `100 * n_train / corpus_n`, rounded to 2dp — the one canonical formula,
/// matching `optick_coverage.coverage_pct` in the Python producer.
///
/// `round_ties_even`, not `round`, is load-bearing. Python's `round` breaks ties
/// to even while Rust's `f64::round` breaks them away from zero, so a value
/// landing exactly on a half at the third decimal disagrees across the two. With
/// `n_train = 721, corpus_n = 800` the quotient is exactly 90.125: Python emits
/// `90.12`, and `f64::round` would recompute `90.13` — a 0.01 gap that exceeds
/// the comparison epsilon and rejects a producer-approved artifact *after* a
/// full training run. Rare, but a hard failure with a baffling message.
fn recomputed_coverage_pct(n_train: u64, corpus_n: u64) -> f64 {
    let denominator = corpus_n.max(1) as f64;
    (100.0 * n_train as f64 / denominator * 100.0).round_ties_even() / 100.0
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
                compact_training_dim: Some(124),
                schema_version: "OPTIC-K-v1.8".into(),
                corpus_size: 1000,
                partitions_used: PHASE1_PARTITIONS.iter().map(|s| s.to_string()).collect(),
            },
            activations_coverage: Some(ActivationsCoverage {
                optick_row_split: OPTICK_ROW_SPLIT.into(),
                n_train: 950,
                n_val: 50,
                corpus_n: 1000,
                coverage_pct: 95.0,
            }),
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
        assert!(matches!(
            err,
            ValidationError::ReconstructionMseTooHigh { .. }
        ));
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

    // ── activations_coverage (ix #248) ────────────────────────────────────────
    //
    // One negative control per rule. The shipped 2026-07-20 snapshot is the
    // motivating case for the first of these: its parquet was fine, but nothing
    // declared what it covered, so a consumer joining optick_row against the
    // 313,047-row index silently dropped the 15,652-row val split (5.0%).

    /// Coverage as declared by the real 2026-07-20 training run.
    fn real_coverage() -> ActivationsCoverage {
        ActivationsCoverage {
            optick_row_split: OPTICK_ROW_SPLIT.into(),
            n_train: 297_395,
            n_val: 15_652,
            corpus_n: 313_047,
            coverage_pct: 95.0,
        }
    }

    fn artifact_with_coverage(coverage: Option<ActivationsCoverage>) -> SaeArtifact {
        let mut a = stub_artifact(0.01, 10.0);
        if let Some(c) = coverage.as_ref() {
            a.input.corpus_size = c.corpus_n;
        }
        a.activations_coverage = coverage;
        a
    }

    #[test]
    fn real_2026_07_20_split_passes_coverage_validation() {
        // Positive control on measured production shapes: 297_395 + 15_652
        // = 313_047, the live optick.index row count.
        assert!(validate_artifact(&artifact_with_coverage(Some(real_coverage()))).is_ok());
    }

    #[test]
    fn missing_coverage_block_fails() {
        // Exactly the shipped 2026-07-20 artifact: valid in every other respect,
        // silent about what its parquet covers.
        let err = validate_artifact(&artifact_with_coverage(None)).unwrap_err();
        assert!(matches!(err, ValidationError::MissingActivationsCoverage));
    }

    #[test]
    fn unknown_row_split_fails() {
        let mut c = real_coverage();
        c.optick_row_split = "val".into();
        assert!(matches!(
            validate_artifact(&artifact_with_coverage(Some(c))).unwrap_err(),
            ValidationError::UnknownRowSplit { .. }
        ));
    }

    #[test]
    fn non_additive_split_fails() {
        let mut c = real_coverage();
        c.n_val -= 1; // one voicing belongs to neither split
        assert!(matches!(
            validate_artifact(&artifact_with_coverage(Some(c))).unwrap_err(),
            ValidationError::NonAdditiveSplit { .. }
        ));
    }

    #[test]
    fn coverage_describing_a_different_corpus_fails() {
        // Declaration gone stale against an index rebuild: internally additive,
        // but not about the corpus input says was trained on.
        let mut a = artifact_with_coverage(Some(real_coverage()));
        a.input.corpus_size = 313_046;
        assert!(matches!(
            validate_artifact(&a).unwrap_err(),
            ValidationError::CoverageCorpusMismatch { .. }
        ));
    }

    #[test]
    fn inconsistent_coverage_pct_fails() {
        let mut c = real_coverage();
        c.coverage_pct = 100.0; // the number a hand-edit would reach for
        assert!(matches!(
            validate_artifact(&artifact_with_coverage(Some(c))).unwrap_err(),
            ValidationError::CoveragePctInconsistent { .. }
        ));
    }

    #[test]
    fn coverage_below_floor_fails() {
        // Additive, self-consistent, and still unacceptable: half the corpus
        // held out means a consumer's join loses half its rows.
        let c = ActivationsCoverage {
            optick_row_split: OPTICK_ROW_SPLIT.into(),
            n_train: 156_523,
            n_val: 156_524,
            corpus_n: 313_047,
            coverage_pct: 50.0,
        };
        assert!(matches!(
            validate_artifact(&artifact_with_coverage(Some(c))).unwrap_err(),
            ValidationError::CoverageBelowFloor { .. }
        ));
    }

    #[test]
    fn coverage_at_floor_passes() {
        let c = ActivationsCoverage {
            optick_row_split: OPTICK_ROW_SPLIT.into(),
            n_train: 900,
            n_val: 100,
            corpus_n: 1000,
            coverage_pct: 90.0,
        };
        assert!(validate_artifact(&artifact_with_coverage(Some(c))).is_ok());
    }

    #[test]
    fn coverage_just_below_floor_fails() {
        let c = ActivationsCoverage {
            optick_row_split: OPTICK_ROW_SPLIT.into(),
            n_train: 8_999,
            n_val: 1_001,
            corpus_n: 10_000,
            coverage_pct: 89.99,
        };
        assert!(matches!(
            validate_artifact(&artifact_with_coverage(Some(c))).unwrap_err(),
            ValidationError::CoverageBelowFloor { .. }
        ));
    }

    #[test]
    fn coverage_floor_matches_python_producer() {
        // The Rust validator and the Python producer must agree on the floor, or
        // the trainer emits artifacts its own orchestrator rejects (or worse,
        // the other way round). Read the constant out of the producer source
        // rather than restating it here — a restated literal drifts silently.
        let source = include_str!("../python/optick_coverage.py");
        let declared = source
            .lines()
            .find_map(|l| l.strip_prefix("MIN_COVERAGE_PCT = "))
            .expect("MIN_COVERAGE_PCT not found in python/optick_coverage.py")
            .trim()
            .parse::<f64>()
            .expect("MIN_COVERAGE_PCT in optick_coverage.py is not a float literal");
        assert_eq!(
            declared, MIN_COVERAGE_PCT,
            "coverage floor drifted between python/optick_coverage.py ({declared}) \
             and src/lib.rs ({MIN_COVERAGE_PCT}) — bump both in the same PR",
        );
    }

    #[test]
    fn coverage_pct_breaks_ties_the_way_python_does() {
        // 100 * 721 / 800 is exactly 90.125. Python's round() goes to even
        // (90.12); f64::round() would go away from zero (90.13), and the 0.01
        // gap exceeds the comparison epsilon — so a producer-approved artifact
        // would be rejected after a full training run. Guards that agreement.
        assert_eq!(recomputed_coverage_pct(721, 800), 90.12);
        let c = ActivationsCoverage {
            optick_row_split: OPTICK_ROW_SPLIT.into(),
            n_train: 721,
            n_val: 79,
            corpus_n: 800,
            coverage_pct: 90.12, // what the Python producer emits
        };
        assert!(validate_artifact(&artifact_with_coverage(Some(c))).is_ok());
    }

    #[test]
    fn legacy_artifact_without_coverage_still_parses() {
        // The two pre-#248 snapshots must remain readable so tooling can report
        // on them; only validate_artifact refuses them.
        let mut json = serde_json::to_value(stub_artifact(0.01, 10.0)).unwrap();
        json.as_object_mut().unwrap().remove("activations_coverage");
        let parsed: SaeArtifact = serde_json::from_value(json).unwrap();
        assert!(parsed.activations_coverage.is_none());
        assert!(matches!(
            validate_artifact(&parsed).unwrap_err(),
            ValidationError::MissingActivationsCoverage
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
