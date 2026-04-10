//! Layer 5 — Risk Delta Governance Gates.
//!
//! Compares before/after [`CodeMetrics`] snapshots of a code entity (file or
//! function), computes a composite risk delta, and maps the delta to a
//! hexavalent verdict (`T`, `P`, `U`, `D`, `F`, `C`) that Demerzel's alignment
//! policy can translate into an [`EscalationLevel`].
//!
//! This module is gated behind the `gates` feature because it pulls in
//! `ix-governance` and `ix-types`.

use crate::metrics::CodeMetrics;
use ix_governance::EscalationLevel;
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};

/// Change in a single named metric between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    /// Metric name (e.g. `"cyclomatic"`, `"h_volume"`).
    pub name: String,
    /// Value before the proposed change.
    pub before: f64,
    /// Value after the proposed change.
    pub after: f64,
    /// Raw delta: `after - before`.
    pub delta: f64,
    /// Delta normalized by `max(|before|, 1.0)` so it is scale-invariant.
    pub normalized_delta: f64,
}

/// A risk signal observed while scoring the delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Stable signal identifier (e.g. `"cyclomatic_regression"`).
    pub name: String,
    /// Severity on a 0.0 (info) to 1.0 (critical) scale.
    pub severity: f64,
    /// Human-readable description.
    pub description: String,
}

/// Provenance metadata for a risk-delta report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Version of the code analyzer that produced the inputs.
    pub analyzer_version: String,
    /// Version of the feature schema used.
    pub feature_schema_version: String,
    /// Version of the risk model applied.
    pub model_version: String,
    /// ISO-8601 timestamp of the scoring run.
    pub timestamp: String,
    /// Hash of the input pair for audit reproducibility.
    pub input_hash: String,
    /// Persona that produced the verdict (e.g. `"skeptical-auditor"`).
    pub governance_persona: String,
}

impl Provenance {
    /// Default provenance used when the caller does not provide one.
    pub fn stub() -> Self {
        Self {
            analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
            feature_schema_version: "1".to_string(),
            model_version: "risk-delta-heuristic-v1".to_string(),
            timestamp: String::new(),
            input_hash: String::new(),
            governance_persona: "skeptical-auditor".to_string(),
        }
    }
}

/// Full risk-delta report for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDelta {
    /// Entity identifier (file path or function name).
    pub entity: String,
    /// Per-metric deltas.
    pub metric_deltas: Vec<MetricDelta>,
    /// Composite normalized risk delta in `[-inf, +inf]`.
    pub composite_risk_delta: f64,
    /// Hexavalent verdict derived from `composite_risk_delta` and `confidence`.
    pub verdict: Hexavalent,
    /// Confidence in the verdict in `[0, 1]`.
    pub confidence: f64,
    /// Signals contributing to the verdict.
    pub signals: Vec<Signal>,
    /// Provenance metadata.
    pub provenance: Provenance,
}

/// Point-in-time risk scorecard for a single entity (no before snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    /// Entity identifier.
    pub entity: String,
    /// Signals observed on the entity.
    pub signals: Vec<Signal>,
    /// Aggregate risk score in `[0, 1]`.
    pub risk_score: f64,
    /// Hexavalent verdict.
    pub verdict: Hexavalent,
    /// Confidence in the verdict in `[0, 1]`.
    pub confidence: f64,
}

/// Metric weights for the composite risk score. Regressions (positive deltas)
/// in these metrics worsen risk; improvements (negative deltas) reduce it.
const METRIC_WEIGHTS: &[(&str, f64)] = &[
    ("cyclomatic", 0.35),
    ("cognitive", 0.25),
    ("h_volume", 0.10),
    ("h_difficulty", 0.10),
    ("sloc", 0.05),
    ("n_exits", 0.05),
    ("n_args", 0.05),
    // Maintainability is inverted (higher is better) — handled in
    // `compute_risk_delta` below.
    ("maintainability_index", 0.05),
];

/// Compute a risk-delta report from two [`CodeMetrics`] snapshots.
pub fn compute_risk_delta(before: &CodeMetrics, after: &CodeMetrics) -> RiskDelta {
    let pairs: Vec<(&'static str, f64, f64)> = vec![
        ("cyclomatic", before.cyclomatic, after.cyclomatic),
        ("cognitive", before.cognitive, after.cognitive),
        ("h_volume", before.h_volume, after.h_volume),
        ("h_difficulty", before.h_difficulty, after.h_difficulty),
        ("sloc", before.sloc, after.sloc),
        ("n_exits", before.n_exits, after.n_exits),
        ("n_args", before.n_args, after.n_args),
        (
            "maintainability_index",
            before.maintainability_index,
            after.maintainability_index,
        ),
    ];

    let mut metric_deltas = Vec::with_capacity(pairs.len());
    let mut composite = 0.0f64;
    let mut signals = Vec::new();

    for (name, b, a) in pairs {
        // NaN / infinity guard: if either side of the comparison is not
        // finite, the metric provides no signal. Record a zeroed delta
        // with a warning-tagged signal and skip the composite contribution.
        if !b.is_finite() || !a.is_finite() {
            metric_deltas.push(MetricDelta {
                name: name.to_string(),
                before: b,
                after: a,
                delta: 0.0,
                normalized_delta: 0.0,
            });
            signals.push(Signal {
                name: "non_finite_metric".to_string(),
                severity: 0.0,
                description: format!("{name} has non-finite value (NaN or Inf)"),
            });
            continue;
        }

        let delta = a - b;
        let denom = b.abs().max(1.0);
        let normalized = delta / denom;
        metric_deltas.push(MetricDelta {
            name: name.to_string(),
            before: b,
            after: a,
            delta,
            normalized_delta: normalized,
        });

        let weight = METRIC_WEIGHTS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        // Maintainability: higher is better → invert sign so improvements
        // reduce the composite risk.
        let signed = if name == "maintainability_index" {
            -normalized
        } else {
            normalized
        };
        composite += weight * signed;

        if name == "cyclomatic" && delta >= 5.0 {
            signals.push(Signal {
                name: "cyclomatic_regression".to_string(),
                severity: (delta / 20.0).min(1.0),
                description: format!("cyclomatic increased by {:.1}", delta),
            });
        }
        if name == "cognitive" && delta >= 5.0 {
            signals.push(Signal {
                name: "cognitive_regression".to_string(),
                severity: (delta / 20.0).min(1.0),
                description: format!("cognitive complexity increased by {:.1}", delta),
            });
        }
        if name == "maintainability_index" && delta <= -10.0 {
            signals.push(Signal {
                name: "maintainability_drop".to_string(),
                severity: ((-delta) / 40.0).min(1.0),
                description: format!("maintainability dropped by {:.1}", -delta),
            });
        }
    }

    // Confidence: we trust the verdict more when the before snapshot has
    // enough signal (non-trivial complexity) to compare against. A function
    // going from 1 → 1 carries little information.
    let base_signal = before.cyclomatic.max(before.cognitive).max(1.0);
    let confidence = (base_signal / 10.0).clamp(0.2, 0.95);

    let verdict = verdict_from_delta(composite, confidence);

    RiskDelta {
        entity: after.name.clone(),
        metric_deltas,
        composite_risk_delta: composite,
        verdict,
        confidence,
        signals,
        provenance: Provenance::stub(),
    }
}

/// Map a composite risk delta and confidence to a [`Hexavalent`] verdict.
///
/// | Condition                                       | Verdict |
/// |-------------------------------------------------|---------|
/// | `confidence < 0.3`                              | `U`     |
/// | `delta <= 0.0`                                  | `T`     |
/// | `delta < 0.15` and `confidence > 0.7`           | `P`     |
/// | `delta < 0.15` and `confidence <= 0.7`          | `U`     |
/// | `0.15 <= delta <= 0.30`                         | `D`     |
/// | `delta > 0.30`                                  | `F`     |
///
/// The `C` (Contradictory) verdict is reserved for callers that detect a
/// conflict between signals (e.g. high complexity paired with high test
/// coverage) and request it explicitly via [`verdict_with_contradiction`].
pub fn verdict_from_delta(composite: f64, confidence: f64) -> Hexavalent {
    // NaN / non-finite guard: a verdict computed from garbage inputs is
    // itself garbage. Return Unknown so downstream policy escalates for
    // human review instead of silently treating the input as "safe" or
    // "unsafe" (the old behavior collapsed NaN comparisons to False).
    if !composite.is_finite() || !confidence.is_finite() {
        return Hexavalent::Unknown;
    }
    if confidence < 0.3 {
        return Hexavalent::Unknown;
    }
    if composite <= 0.0 {
        return Hexavalent::True;
    }
    if composite < 0.15 {
        if confidence > 0.7 {
            return Hexavalent::Probable;
        }
        return Hexavalent::Unknown;
    }
    if composite <= 0.30 {
        return Hexavalent::Doubtful;
    }
    Hexavalent::False
}

/// Variant of [`verdict_from_delta`] that allows callers to force a
/// `Contradictory` verdict when they detect conflicting signals (e.g. a
/// risky cyclomatic regression paired with improved test coverage).
pub fn verdict_with_contradiction(
    composite: f64,
    confidence: f64,
    contradictory: bool,
) -> Hexavalent {
    if contradictory {
        return Hexavalent::Contradictory;
    }
    verdict_from_delta(composite, confidence)
}

/// Map a hexavalent verdict to a Demerzel [`EscalationLevel`].
///
/// - `T`, `P` → `Autonomous`
/// - `U`      → `ProceedWithNote`
/// - `D`      → `AskConfirmation`
/// - `F`, `C` → `Escalate`
pub fn map_verdict_to_escalation(v: Hexavalent) -> EscalationLevel {
    match v {
        Hexavalent::True | Hexavalent::Probable => EscalationLevel::Autonomous,
        Hexavalent::Unknown => EscalationLevel::ProceedWithNote,
        Hexavalent::Doubtful => EscalationLevel::AskConfirmation,
        Hexavalent::False | Hexavalent::Contradictory => EscalationLevel::Escalate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> CodeMetrics {
        CodeMetrics {
            name: "f".to_string(),
            start_line: 1,
            end_line: 20,
            cyclomatic: 5.0,
            cognitive: 5.0,
            n_exits: 1.0,
            n_args: 2.0,
            sloc: 20.0,
            ploc: 25.0,
            lloc: 18.0,
            cloc: 2.0,
            blank: 3.0,
            h_u_ops: 10.0,
            h_u_opnds: 8.0,
            h_total_ops: 30.0,
            h_total_opnds: 25.0,
            h_vocabulary: 18.0,
            h_length: 55.0,
            h_volume: 229.0,
            h_difficulty: 15.6,
            h_effort: 3572.0,
            h_bugs: 0.076,
            maintainability_index: 110.0,
        }
    }

    #[test]
    fn test_verdict_safe_on_zero_delta() {
        // Confidence must be above the unknown floor (0.3) for the zero-delta
        // branch to resolve to True; we use confidence 0.8.
        let v = verdict_from_delta(0.0, 0.8);
        assert_eq!(v, Hexavalent::True);
    }

    #[test]
    fn test_verdict_fail_on_high_delta() {
        let v = verdict_from_delta(0.5, 0.9);
        assert_eq!(v, Hexavalent::False);
    }

    #[test]
    fn test_verdict_unknown_on_low_confidence() {
        let v = verdict_from_delta(0.05, 0.1);
        assert_eq!(v, Hexavalent::Unknown);
    }

    #[test]
    fn test_verdict_doubtful_midrange() {
        assert_eq!(verdict_from_delta(0.20, 0.9), Hexavalent::Doubtful);
    }

    #[test]
    fn test_verdict_probable_small_positive() {
        assert_eq!(verdict_from_delta(0.05, 0.9), Hexavalent::Probable);
    }

    #[test]
    fn test_verdict_unknown_on_nan_composite() {
        // NaN inputs must never collapse to True or False — that would hide
        // data-quality errors inside a plausible-looking verdict.
        assert_eq!(verdict_from_delta(f64::NAN, 0.9), Hexavalent::Unknown);
        assert_eq!(verdict_from_delta(0.1, f64::NAN), Hexavalent::Unknown);
        assert_eq!(
            verdict_from_delta(f64::INFINITY, 0.9),
            Hexavalent::Unknown
        );
        assert_eq!(
            verdict_from_delta(0.1, f64::NEG_INFINITY),
            Hexavalent::Unknown
        );
    }

    #[test]
    fn test_risk_delta_skips_non_finite_metrics() {
        // If any metric is NaN, compute_risk_delta should emit a
        // non_finite_metric signal and still produce a finite composite.
        let before = baseline();
        let mut after = baseline();
        after.cyclomatic = f64::NAN;
        let report = compute_risk_delta(&before, &after);
        assert!(
            report.composite_risk_delta.is_finite(),
            "composite must remain finite, got {}",
            report.composite_risk_delta
        );
        assert!(
            report.signals.iter().any(|s| s.name == "non_finite_metric"),
            "expected non_finite_metric signal"
        );
        // Verdict should not be False (which would be silent corruption)
        assert_ne!(report.verdict, Hexavalent::False);
    }

    #[test]
    fn test_contradictory_override() {
        assert_eq!(
            verdict_with_contradiction(0.5, 0.9, true),
            Hexavalent::Contradictory
        );
    }

    #[test]
    fn test_map_verdict_to_escalation_all() {
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::True),
            EscalationLevel::Autonomous
        );
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::Probable),
            EscalationLevel::Autonomous
        );
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::Unknown),
            EscalationLevel::ProceedWithNote
        );
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::Doubtful),
            EscalationLevel::AskConfirmation
        );
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::False),
            EscalationLevel::Escalate
        );
        assert_eq!(
            map_verdict_to_escalation(Hexavalent::Contradictory),
            EscalationLevel::Escalate
        );
    }

    #[test]
    fn test_risk_delta_identifies_regression() {
        let before = baseline();
        let mut after = baseline();
        after.cyclomatic = 20.0;
        after.cognitive = 18.0;
        after.maintainability_index = 80.0;

        let report = compute_risk_delta(&before, &after);
        assert!(
            report.composite_risk_delta > 0.30,
            "expected high composite risk, got {}",
            report.composite_risk_delta
        );
        assert_eq!(report.verdict, Hexavalent::False);
        assert!(
            report.signals.iter().any(|s| s.name == "cyclomatic_regression"),
            "expected cyclomatic_regression signal"
        );
        assert_eq!(
            map_verdict_to_escalation(report.verdict),
            EscalationLevel::Escalate
        );
    }

    #[test]
    fn test_risk_delta_neutral_on_identity() {
        let before = baseline();
        let after = baseline();
        let report = compute_risk_delta(&before, &after);
        assert_eq!(report.composite_risk_delta, 0.0);
        assert_eq!(report.verdict, Hexavalent::True);
    }

    #[test]
    fn test_risk_delta_improvement() {
        let before = baseline();
        let mut after = baseline();
        after.cyclomatic = 3.0;
        after.cognitive = 3.0;
        after.maintainability_index = 130.0;
        let report = compute_risk_delta(&before, &after);
        assert!(report.composite_risk_delta < 0.0);
        assert_eq!(report.verdict, Hexavalent::True);
    }
}
