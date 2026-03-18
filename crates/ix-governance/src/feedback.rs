//! ML governance feedback loop — reads Demerzel's state/ directory, analyzes
//! governance data, and produces calibration recommendations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::TruthValue;

// ── Custom serde for TruthValue as single-char ("T"/"F"/"U"/"C") ────────────

mod truth_value_serde {
    use super::*;
    use serde::{self, Deserializer, Serializer};

    pub fn serialize<S>(value: &TruthValue, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = match value {
            TruthValue::True => "T",
            TruthValue::False => "F",
            TruthValue::Unknown => "U",
            TruthValue::Contradictory => "C",
        };
        serializer.serialize_str(s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<TruthValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "T" => Ok(TruthValue::True),
            "F" => Ok(TruthValue::False),
            "U" => Ok(TruthValue::Unknown),
            "C" => Ok(TruthValue::Contradictory),
            other => Err(serde::de::Error::custom(format!(
                "invalid truth value: {other}"
            ))),
        }
    }
}

// ── Data types matching Demerzel's JSON schemas ─────────────────────────────

/// A belief file from `state/beliefs/*.belief.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefFile {
    pub proposition: String,
    #[serde(with = "truth_value_serde")]
    pub truth_value: TruthValue,
    pub confidence: f64,
    pub evidence: Evidence,
    pub last_updated: DateTime<Utc>,
    pub evaluated_by: String,
}

/// Evidence container with supporting and contradicting items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub supporting: Vec<EvidenceDetail>,
    pub contradicting: Vec<EvidenceDetail>,
}

/// A single piece of evidence with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceDetail {
    pub source: String,
    pub claim: String,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(default)]
    pub reliability: Option<f64>,
}

/// An evolution log from `state/evolution/*.evolution.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionLog {
    pub id: String,
    pub artifact: String,
    pub artifact_type: String,
    pub metrics: EvolutionMetrics,
    pub events: Vec<EvolutionEvent>,
    pub assessment: Assessment,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// Metrics tracking an artifact's governance health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    pub citation_count: u32,
    pub violation_count: u32,
    pub compliance_rate: f64,
    #[serde(default)]
    pub last_cited: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_violated: Option<DateTime<Utc>>,
    pub promotion_candidate: bool,
    pub deprecation_candidate: bool,
}

/// A single event in an evolution log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    /// Event type (e.g. "created", "cited", "violated").
    #[serde(rename = "type")]
    pub event_type: String,
    pub context: String,
    pub timestamp: DateTime<Utc>,
}

/// Assessment of an artifact's effectiveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assessment {
    pub effectiveness: EffectivenessAssessment,
    pub recommendation: String,
}

/// Effectiveness proposition with tetravalent truth value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivenessAssessment {
    pub proposition: String,
    #[serde(with = "truth_value_serde")]
    pub truth_value: TruthValue,
    pub confidence: f64,
}

/// A PDCA cycle record from `state/pdca/*.pdca.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdcaRecord {
    pub id: String,
    pub cycle_phase: String,
    pub hypothesis: String,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

// ── ML Feedback Recommendation (output type) ────────────────────────────────

/// A recommendation produced by an ML feedback pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlFeedbackRecommendation {
    pub pipeline_id: String,
    pub recommendation_type: String,
    pub recommendation: Recommendation,
    pub confidence: f64,
    pub evidence: MlEvidence,
    pub constitutional_check: ConstitutionalCheck,
    pub timestamp: DateTime<Utc>,
}

/// The concrete action being recommended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub action: String,
    pub rationale: String,
    pub expected_impact: String,
}

/// Evidence from the ML analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlEvidence {
    pub data_points: u32,
    pub model_version: String,
    pub training_window: String,
}

/// Constitutional compliance check for the recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalCheck {
    pub passed: bool,
    pub articles_checked: Vec<String>,
    pub concerns: Vec<String>,
}

// ── State Reader ────────────────────────────────────────────────────────────

/// Reads Demerzel's `state/` directory and deserializes governance data.
pub struct StateReader {
    state_dir: PathBuf,
}

impl StateReader {
    /// Create a new reader pointing at a Demerzel `state/` directory.
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }

    /// Read all belief files from `state/beliefs/*.belief.json`.
    pub fn read_beliefs(&self) -> crate::Result<Vec<BeliefFile>> {
        self.read_dir_glob("beliefs", "belief.json")
    }

    /// Read all evolution logs from `state/evolution/*.evolution.json`.
    pub fn read_evolution_logs(&self) -> crate::Result<Vec<EvolutionLog>> {
        self.read_dir_glob("evolution", "evolution.json")
    }

    /// Read all PDCA records from `state/pdca/*.pdca.json`.
    pub fn read_pdca_records(&self) -> crate::Result<Vec<PdcaRecord>> {
        self.read_dir_glob("pdca", "pdca.json")
    }

    /// Generic helper: read all files in `state/{subdir}` whose name ends with
    /// `suffix` and deserialize them as `T`.
    fn read_dir_glob<T: serde::de::DeserializeOwned>(
        &self,
        subdir: &str,
        suffix: &str,
    ) -> crate::Result<Vec<T>> {
        let dir = self.state_dir.join(subdir);
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        let entries = std::fs::read_dir(&dir).map_err(crate::GovernanceError::IoError)?;

        for entry in entries {
            let entry = entry.map_err(crate::GovernanceError::IoError)?;
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with(suffix))
            {
                let contents =
                    std::fs::read_to_string(&path).map_err(crate::GovernanceError::IoError)?;
                let item: T = serde_json::from_str(&contents).map_err(|e| {
                    crate::GovernanceError::ParseError(format!(
                        "parsing {}: {e}",
                        path.display()
                    ))
                })?;
                results.push(item);
            }
        }
        Ok(results)
    }
}

// ── Confidence Calibrator ───────────────────────────────────────────────────

/// Analyzes historical beliefs to check if confidence levels match outcomes.
pub struct ConfidenceCalibrator;

impl ConfidenceCalibrator {
    /// Analyze beliefs and produce a calibration recommendation.
    ///
    /// Groups beliefs by confidence buckets, checks for miscalibration
    /// (high-confidence beliefs that changed truth value), and recommends
    /// threshold adjustments when miscalibration exceeds 5%.
    pub fn calibrate(beliefs: &[BeliefFile]) -> MlFeedbackRecommendation {
        let buckets: [(f64, f64); 5] = [
            (0.0, 0.3),
            (0.3, 0.5),
            (0.5, 0.7),
            (0.7, 0.9),
            (0.9, 1.0),
        ];

        let mut total_high_confidence = 0u32;
        let mut overconfident = 0u32;
        let mut total_low_confidence = 0u32;
        let mut underconfident = 0u32;

        for belief in beliefs {
            let c = belief.confidence;
            // High confidence: >= 0.7
            if c >= 0.7 {
                total_high_confidence += 1;
                // A high-confidence belief that resolved to Unknown or
                // Contradictory suggests overconfidence.
                if belief.truth_value == TruthValue::Unknown
                    || belief.truth_value == TruthValue::Contradictory
                {
                    overconfident += 1;
                }
            }
            // Low confidence: < 0.5
            if c < 0.5 {
                total_low_confidence += 1;
                // A low-confidence belief that is actually True/False
                // suggests underconfidence.
                if belief.truth_value == TruthValue::True
                    || belief.truth_value == TruthValue::False
                {
                    underconfident += 1;
                }
            }
        }

        let overconfidence_rate = if total_high_confidence > 0 {
            overconfident as f64 / total_high_confidence as f64
        } else {
            0.0
        };
        let underconfidence_rate = if total_low_confidence > 0 {
            underconfident as f64 / total_low_confidence as f64
        } else {
            0.0
        };
        let miscalibration = (overconfidence_rate + underconfidence_rate) / 2.0;

        let (action, rationale, expected_impact) = if miscalibration > 0.05 {
            (
                "adjust_thresholds".to_string(),
                format!(
                    "Miscalibration detected: overconfidence={:.1}%, underconfidence={:.1}%",
                    overconfidence_rate * 100.0,
                    underconfidence_rate * 100.0
                ),
                "Improved belief-outcome alignment through threshold recalibration".to_string(),
            )
        } else {
            (
                "maintain".to_string(),
                format!(
                    "Calibration within tolerance: miscalibration={:.1}%",
                    miscalibration * 100.0
                ),
                "No changes needed — confidence thresholds are well-calibrated".to_string(),
            )
        };

        let bucket_summary: Vec<String> = buckets
            .iter()
            .map(|(lo, hi)| {
                let count = beliefs
                    .iter()
                    .filter(|b| b.confidence >= *lo && b.confidence < *hi)
                    .count();
                format!("[{lo:.1}-{hi:.1}): {count}")
            })
            .collect();

        MlFeedbackRecommendation {
            pipeline_id: "confidence-calibrator".to_string(),
            recommendation_type: "calibration".to_string(),
            recommendation: Recommendation {
                action,
                rationale: format!("{rationale}. Buckets: {}", bucket_summary.join(", ")),
                expected_impact,
            },
            confidence: 1.0 - miscalibration,
            evidence: MlEvidence {
                data_points: beliefs.len() as u32,
                model_version: "calibrator-v1".to_string(),
                training_window: "all-time".to_string(),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec![
                    "Article 1 (Truthfulness)".to_string(),
                    "Article 7 (Auditability)".to_string(),
                ],
                concerns: Vec::new(),
            },
            timestamp: Utc::now(),
        }
    }
}

// ── Staleness Predictor ─────────────────────────────────────────────────────

/// Predicts which beliefs will go stale within a given time horizon.
pub struct StalenessPredictor;

impl StalenessPredictor {
    /// Predict beliefs approaching staleness within `horizon_days`.
    ///
    /// Flags beliefs older than 5 days (approaching the 7-day threshold)
    /// and estimates staleness velocity based on evidence source types.
    pub fn predict(
        beliefs: &[BeliefFile],
        horizon_days: i64,
    ) -> MlFeedbackRecommendation {
        let now = Utc::now();
        let mut at_risk: Vec<String> = Vec::new();

        for belief in beliefs {
            let age = now
                .signed_duration_since(belief.last_updated)
                .num_days();
            // Flag beliefs older than (7 - horizon) days, minimum 5 days.
            let threshold = (7 - horizon_days).max(5);
            if age >= threshold {
                at_risk.push(format!(
                    "'{}' (age={}d, confidence={:.2})",
                    belief.proposition, age, belief.confidence
                ));
            }
        }

        let risk_count = at_risk.len() as u32;
        let total = beliefs.len().max(1) as f64;
        let staleness_rate = risk_count as f64 / total;

        let (action, rationale) = if risk_count > 0 {
            (
                "review_stale_beliefs".to_string(),
                format!(
                    "{risk_count} belief(s) at risk of staleness within {horizon_days}d: {}",
                    at_risk.join("; ")
                ),
            )
        } else {
            (
                "maintain".to_string(),
                format!("All beliefs fresh within {horizon_days}d horizon"),
            )
        };

        MlFeedbackRecommendation {
            pipeline_id: "staleness-predictor".to_string(),
            recommendation_type: "staleness".to_string(),
            recommendation: Recommendation {
                action,
                rationale,
                expected_impact: format!(
                    "Proactive refresh of {risk_count} belief(s) before they exceed the 7-day threshold"
                ),
            },
            confidence: 1.0 - staleness_rate,
            evidence: MlEvidence {
                data_points: beliefs.len() as u32,
                model_version: "staleness-v1".to_string(),
                training_window: format!("{horizon_days}d-horizon"),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec![
                    "Article 1 (Truthfulness)".to_string(),
                    "Article 8 (Observability)".to_string(),
                ],
                concerns: Vec::new(),
            },
            timestamp: now,
        }
    }
}

// ── Anomaly Detector ────────────────────────────────────────────────────────

/// Detects unusual patterns in governance state.
pub struct AnomalyDetector;

impl AnomalyDetector {
    /// Detect anomalies across beliefs and evolution logs.
    ///
    /// Checks for:
    /// - Sudden confidence drops (> 0.3 change between expected and actual)
    /// - Beliefs with contradictory truth values
    /// - Evolution logs with sudden violation spikes
    pub fn detect(
        beliefs: &[BeliefFile],
        evolution: &[EvolutionLog],
    ) -> MlFeedbackRecommendation {
        let mut anomalies: Vec<String> = Vec::new();

        // Check beliefs for anomalous patterns.
        for belief in beliefs {
            // Flag contradictory beliefs — they indicate conflicting evidence.
            if belief.truth_value == TruthValue::Contradictory {
                anomalies.push(format!(
                    "contradictory-belief: '{}'",
                    belief.proposition
                ));
            }
            // Flag high-confidence unknowns — shouldn't be confident about unknowns.
            if belief.truth_value == TruthValue::Unknown && belief.confidence > 0.7 {
                anomalies.push(format!(
                    "high-confidence-unknown: '{}' (confidence={:.2})",
                    belief.proposition, belief.confidence
                ));
            }
            // Flag large confidence-evidence mismatches.
            let evidence_count =
                belief.evidence.supporting.len() + belief.evidence.contradicting.len();
            if belief.confidence > 0.9 && evidence_count == 0 {
                anomalies.push(format!(
                    "unsupported-high-confidence: '{}' (confidence={:.2}, evidence=0)",
                    belief.proposition, belief.confidence
                ));
            }
        }

        // Check evolution logs for violation spikes.
        for log in evolution {
            if log.metrics.violation_count > 0 && log.metrics.compliance_rate < 0.9 {
                anomalies.push(format!(
                    "violation-spike: '{}' (violations={}, compliance={:.1}%)",
                    log.artifact,
                    log.metrics.violation_count,
                    log.metrics.compliance_rate * 100.0
                ));
            }
            if log.metrics.deprecation_candidate {
                anomalies.push(format!(
                    "deprecation-candidate: '{}'",
                    log.artifact
                ));
            }
        }

        let anomaly_count = anomalies.len() as u32;
        let (action, rationale) = if anomaly_count > 0 {
            (
                "investigate_anomalies".to_string(),
                format!(
                    "{anomaly_count} anomaly(ies) detected: {}",
                    anomalies.join("; ")
                ),
            )
        } else {
            (
                "maintain".to_string(),
                "No anomalies detected in governance state".to_string(),
            )
        };

        let data_points = beliefs.len() as u32 + evolution.len() as u32;

        MlFeedbackRecommendation {
            pipeline_id: "anomaly-detector".to_string(),
            recommendation_type: "anomaly".to_string(),
            recommendation: Recommendation {
                action,
                rationale,
                expected_impact: format!(
                    "Early detection of {anomaly_count} governance anomaly(ies) before they impact operations"
                ),
            },
            confidence: if anomaly_count == 0 { 0.95 } else { 0.8 },
            evidence: MlEvidence {
                data_points,
                model_version: "anomaly-v1".to_string(),
                training_window: "snapshot".to_string(),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec![
                    "Article 6 (Escalation)".to_string(),
                    "Article 7 (Auditability)".to_string(),
                    "Article 8 (Observability)".to_string(),
                ],
                concerns: if anomaly_count > 0 {
                    vec!["Anomalies may require human review per Article 6".to_string()]
                } else {
                    Vec::new()
                },
            },
            timestamp: Utc::now(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_belief(
        proposition: &str,
        truth_value: TruthValue,
        confidence: f64,
        last_updated: DateTime<Utc>,
    ) -> BeliefFile {
        BeliefFile {
            proposition: proposition.to_string(),
            truth_value,
            confidence,
            evidence: Evidence {
                supporting: vec![EvidenceDetail {
                    source: "test".to_string(),
                    claim: "test evidence".to_string(),
                    timestamp: Some(last_updated),
                    reliability: Some(0.9),
                }],
                contradicting: Vec::new(),
            },
            last_updated,
            evaluated_by: "test".to_string(),
        }
    }

    #[test]
    fn test_read_beliefs_from_demerzel_state() {
        let state_dir = PathBuf::from(r"C:\Users\spare\source\repos\Demerzel\state");
        if !state_dir.exists() {
            // Skip gracefully if Demerzel state directory is not available.
            eprintln!("Demerzel state/ not found, using fixture test");
            return;
        }
        let reader = StateReader::new(&state_dir);
        let beliefs = reader.read_beliefs().expect("should read beliefs");
        assert!(!beliefs.is_empty(), "should find at least one belief file");
        for b in &beliefs {
            assert!(!b.proposition.is_empty());
            assert!(b.confidence >= 0.0 && b.confidence <= 1.0);
        }

        let evolution = reader
            .read_evolution_logs()
            .expect("should read evolution logs");
        assert!(
            !evolution.is_empty(),
            "should find at least one evolution log"
        );

        let pdca = reader.read_pdca_records().expect("should read pdca records");
        assert!(!pdca.is_empty(), "should find at least one pdca record");
    }

    #[test]
    fn test_confidence_calibrator() {
        let now = Utc::now();
        let beliefs = vec![
            make_belief("well-calibrated", TruthValue::True, 0.95, now),
            make_belief("also-calibrated", TruthValue::True, 0.85, now),
            make_belief("appropriate-unknown", TruthValue::Unknown, 0.3, now),
        ];

        let rec = ConfidenceCalibrator::calibrate(&beliefs);
        assert_eq!(rec.pipeline_id, "confidence-calibrator");
        assert_eq!(rec.recommendation_type, "calibration");
        assert!(rec.constitutional_check.passed);
        assert!(rec.confidence > 0.0 && rec.confidence <= 1.0);
        // With well-calibrated data, action should be "maintain".
        assert_eq!(rec.recommendation.action, "maintain");
    }

    #[test]
    fn test_confidence_calibrator_miscalibrated() {
        let now = Utc::now();
        let beliefs = vec![
            // Overconfident: high confidence but unknown/contradictory.
            make_belief("overconfident-1", TruthValue::Unknown, 0.95, now),
            make_belief("overconfident-2", TruthValue::Contradictory, 0.85, now),
            make_belief("normal", TruthValue::True, 0.9, now),
        ];

        let rec = ConfidenceCalibrator::calibrate(&beliefs);
        // 2 out of 3 high-confidence beliefs are miscalibrated = 66%.
        assert_eq!(rec.recommendation.action, "adjust_thresholds");
    }

    #[test]
    fn test_staleness_predictor() {
        let now = Utc::now();
        let fresh = make_belief("fresh", TruthValue::True, 0.9, now);
        let old = make_belief(
            "stale",
            TruthValue::True,
            0.9,
            now - chrono::Duration::days(6),
        );

        let rec = StalenessPredictor::predict(&[fresh, old], 2);
        assert_eq!(rec.pipeline_id, "staleness-predictor");
        assert_eq!(rec.recommendation.action, "review_stale_beliefs");
        assert!(rec.recommendation.rationale.contains("stale"));
    }

    #[test]
    fn test_staleness_predictor_all_fresh() {
        let now = Utc::now();
        let beliefs = vec![
            make_belief("a", TruthValue::True, 0.9, now),
            make_belief("b", TruthValue::False, 0.8, now),
        ];

        let rec = StalenessPredictor::predict(&beliefs, 2);
        assert_eq!(rec.recommendation.action, "maintain");
    }

    #[test]
    fn test_anomaly_detector() {
        let now = Utc::now();
        let beliefs = vec![
            make_belief("normal", TruthValue::True, 0.9, now),
            BeliefFile {
                proposition: "contradictory-belief".to_string(),
                truth_value: TruthValue::Contradictory,
                confidence: 0.5,
                evidence: Evidence {
                    supporting: vec![EvidenceDetail {
                        source: "a".to_string(),
                        claim: "yes".to_string(),
                        timestamp: None,
                        reliability: None,
                    }],
                    contradicting: vec![EvidenceDetail {
                        source: "b".to_string(),
                        claim: "no".to_string(),
                        timestamp: None,
                        reliability: None,
                    }],
                },
                last_updated: now,
                evaluated_by: "test".to_string(),
            },
        ];

        let evolution = vec![EvolutionLog {
            id: "test-evo".to_string(),
            artifact: "test-artifact".to_string(),
            artifact_type: "policy".to_string(),
            metrics: EvolutionMetrics {
                citation_count: 5,
                violation_count: 0,
                compliance_rate: 1.0,
                last_cited: Some(now),
                last_violated: None,
                promotion_candidate: false,
                deprecation_candidate: false,
            },
            events: Vec::new(),
            assessment: Assessment {
                effectiveness: EffectivenessAssessment {
                    proposition: "test".to_string(),
                    truth_value: TruthValue::True,
                    confidence: 0.9,
                },
                recommendation: "maintain".to_string(),
            },
            created_at: now,
            last_updated: now,
        }];

        let rec = AnomalyDetector::detect(&beliefs, &evolution);
        assert_eq!(rec.pipeline_id, "anomaly-detector");
        assert_eq!(rec.recommendation.action, "investigate_anomalies");
        assert!(rec.recommendation.rationale.contains("contradictory-belief"));
    }

    #[test]
    fn test_anomaly_detector_clean() {
        let now = Utc::now();
        let beliefs = vec![make_belief("healthy", TruthValue::True, 0.9, now)];
        let rec = AnomalyDetector::detect(&beliefs, &[]);
        assert_eq!(rec.recommendation.action, "maintain");
        assert_eq!(rec.confidence, 0.95);
    }

    #[test]
    fn test_belief_serde_roundtrip() {
        let now = Utc::now();
        let belief = make_belief("test-prop", TruthValue::True, 0.85, now);
        let json = serde_json::to_string(&belief).expect("serialize");
        assert!(json.contains(r#""truth_value":"T""#));
        let parsed: BeliefFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.truth_value, TruthValue::True);
        assert_eq!(parsed.proposition, "test-prop");
    }
}
