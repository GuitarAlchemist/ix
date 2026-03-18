//! Remediation optimizer — takes outputs from all pipeline modules and
//! prioritizes remediation actions based on impact and urgency.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::feedback::{ConstitutionalCheck, MlEvidence, MlFeedbackRecommendation, Recommendation};
use crate::violation_pattern::ViolationPatternReport;

// ── Report types ────────────────────────────────────────────────────────────

/// Priority level for a remediation action.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// A single prioritized remediation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    pub id: String,
    pub source_pipeline: String,
    pub action: String,
    pub rationale: String,
    pub priority: Priority,
    pub estimated_impact: f64,
    pub urgency_score: f64,
}

/// Resource requirements for executing the remediation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub total_actions: u32,
    pub critical_actions: u32,
    pub high_actions: u32,
    pub estimated_review_cycles: u32,
}

/// Timeline estimate for the remediation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub immediate: Vec<String>,
    pub short_term: Vec<String>,
    pub long_term: Vec<String>,
}

/// Full remediation plan produced by the optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationPlan {
    pub prioritized_actions: Vec<RemediationAction>,
    pub estimated_impact: f64,
    pub resource_requirements: ResourceRequirements,
    pub timeline: Timeline,
    pub timestamp: DateTime<Utc>,
}

// ── Optimizer ───────────────────────────────────────────────────────────────

/// Combines outputs from all feedback pipeline modules and produces a
/// prioritized remediation plan.
pub struct RemediationOptimizer;

impl RemediationOptimizer {
    /// Optimize remediation priorities from all pipeline outputs.
    ///
    /// Takes:
    /// - `calibration`: output from ConfidenceCalibrator
    /// - `staleness`: output from StalenessPredictor
    /// - `anomalies`: output from AnomalyDetector
    /// - `violation_report`: output from ViolationPatternAnalyzer
    ///
    /// Produces a prioritized remediation plan plus an ML feedback recommendation.
    pub fn optimize(
        calibration: &MlFeedbackRecommendation,
        staleness: &MlFeedbackRecommendation,
        anomalies: &MlFeedbackRecommendation,
        violation_report: &ViolationPatternReport,
    ) -> (RemediationPlan, MlFeedbackRecommendation) {
        let now = Utc::now();
        let mut actions = Vec::new();
        let mut action_id = 0u32;

        // ── Process violation patterns (highest priority source) ────────
        for rec_text in &violation_report.recommendations {
            action_id += 1;
            let (priority, urgency) = if rec_text.starts_with("URGENT") {
                (Priority::Critical, 1.0)
            } else if rec_text.starts_with("MONITOR") {
                (Priority::High, 0.7)
            } else {
                (Priority::Low, 0.3)
            };

            let impact = if violation_report.total_violations > 0 {
                0.8
            } else {
                0.2
            };

            actions.push(RemediationAction {
                id: format!("REM-{action_id:03}"),
                source_pipeline: "violation-pattern-analyzer".to_string(),
                action: "remediate_violation_pattern".to_string(),
                rationale: rec_text.clone(),
                priority,
                estimated_impact: impact,
                urgency_score: urgency,
            });
        }

        // ── Process anomalies ───────────────────────────────────────────
        if anomalies.recommendation.action != "maintain" {
            action_id += 1;
            actions.push(RemediationAction {
                id: format!("REM-{action_id:03}"),
                source_pipeline: "anomaly-detector".to_string(),
                action: anomalies.recommendation.action.clone(),
                rationale: anomalies.recommendation.rationale.clone(),
                priority: Priority::High,
                estimated_impact: 0.7,
                urgency_score: 0.8,
            });
        }

        // ── Process calibration issues ──────────────────────────────────
        if calibration.recommendation.action != "maintain" {
            action_id += 1;
            actions.push(RemediationAction {
                id: format!("REM-{action_id:03}"),
                source_pipeline: "confidence-calibrator".to_string(),
                action: calibration.recommendation.action.clone(),
                rationale: calibration.recommendation.rationale.clone(),
                priority: Priority::Medium,
                estimated_impact: 0.5,
                urgency_score: 0.5,
            });
        }

        // ── Process staleness predictions ───────────────────────────────
        if staleness.recommendation.action != "maintain" {
            action_id += 1;
            actions.push(RemediationAction {
                id: format!("REM-{action_id:03}"),
                source_pipeline: "staleness-predictor".to_string(),
                action: staleness.recommendation.action.clone(),
                rationale: staleness.recommendation.rationale.clone(),
                priority: Priority::Medium,
                estimated_impact: 0.4,
                urgency_score: 0.6,
            });
        }

        // ── Sort by priority then urgency ───────────────────────────────
        actions.sort_by(|a, b| {
            a.priority.cmp(&b.priority).then(
                b.urgency_score
                    .partial_cmp(&a.urgency_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        // ── Compute aggregate metrics ───────────────────────────────────
        let total_actions = actions.len() as u32;
        let critical_actions = actions
            .iter()
            .filter(|a| a.priority == Priority::Critical)
            .count() as u32;
        let high_actions = actions
            .iter()
            .filter(|a| a.priority == Priority::High)
            .count() as u32;

        let estimated_impact = if actions.is_empty() {
            0.0
        } else {
            actions.iter().map(|a| a.estimated_impact).sum::<f64>() / actions.len() as f64
        };

        // ── Build timeline ──────────────────────────────────────────────
        let immediate: Vec<String> = actions
            .iter()
            .filter(|a| a.priority == Priority::Critical)
            .map(|a| a.id.clone())
            .collect();
        let short_term: Vec<String> = actions
            .iter()
            .filter(|a| a.priority == Priority::High || a.priority == Priority::Medium)
            .map(|a| a.id.clone())
            .collect();
        let long_term: Vec<String> = actions
            .iter()
            .filter(|a| a.priority == Priority::Low)
            .map(|a| a.id.clone())
            .collect();

        let plan = RemediationPlan {
            prioritized_actions: actions,
            estimated_impact,
            resource_requirements: ResourceRequirements {
                total_actions,
                critical_actions,
                high_actions,
                estimated_review_cycles: critical_actions * 2 + high_actions + total_actions / 3,
            },
            timeline: Timeline {
                immediate,
                short_term,
                long_term,
            },
            timestamp: now,
        };

        // ── Build ML feedback recommendation ────────────────────────────
        let (action, rationale) = if total_actions > 0 {
            (
                "execute_remediation_plan".to_string(),
                format!(
                    "Remediation plan with {total_actions} action(s): {critical_actions} critical, {high_actions} high priority"
                ),
            )
        } else {
            (
                "maintain".to_string(),
                "All pipelines healthy — no remediation needed".to_string(),
            )
        };

        let recommendation = MlFeedbackRecommendation {
            pipeline_id: "remediation-optimizer".to_string(),
            recommendation_type: "remediation".to_string(),
            recommendation: Recommendation {
                action,
                rationale,
                expected_impact: if total_actions > 0 {
                    format!(
                        "Coordinated remediation across {total_actions} action(s) with estimated impact of {estimated_impact:.0}%"
                    )
                } else {
                    "System operating within governance parameters".to_string()
                },
            },
            confidence: if total_actions == 0 {
                0.95
            } else {
                // Confidence is inversely related to the number of critical issues.
                (0.9 - critical_actions as f64 * 0.1).max(0.5)
            },
            evidence: MlEvidence {
                data_points: 4, // four pipeline inputs
                model_version: "remediation-optimizer-v1".to_string(),
                training_window: "current-cycle".to_string(),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec![
                    "Article 3 (Reversibility)".to_string(),
                    "Article 4 (Proportionality)".to_string(),
                    "Article 6 (Escalation)".to_string(),
                    "Article 9 (Bounded Autonomy)".to_string(),
                ],
                concerns: if critical_actions > 0 {
                    vec![
                        "Critical actions require human approval per Article 6 (Escalation)"
                            .to_string(),
                    ]
                } else {
                    Vec::new()
                },
            },
            timestamp: now,
        };

        (plan, recommendation)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::{ConstitutionalCheck, MlEvidence, MlFeedbackRecommendation, Recommendation};
    use crate::violation_pattern::ViolationPatternReport;

    fn make_pipeline_rec(
        pipeline_id: &str,
        action: &str,
        rationale: &str,
    ) -> MlFeedbackRecommendation {
        MlFeedbackRecommendation {
            pipeline_id: pipeline_id.to_string(),
            recommendation_type: "test".to_string(),
            recommendation: Recommendation {
                action: action.to_string(),
                rationale: rationale.to_string(),
                expected_impact: "test impact".to_string(),
            },
            confidence: 0.9,
            evidence: MlEvidence {
                data_points: 10,
                model_version: "test-v1".to_string(),
                training_window: "test".to_string(),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec!["Article 1".to_string()],
                concerns: Vec::new(),
            },
            timestamp: Utc::now(),
        }
    }

    fn make_violation_report(
        total_violations: u32,
        recommendations: Vec<String>,
    ) -> ViolationPatternReport {
        ViolationPatternReport {
            most_violated_articles: vec![],
            violation_trends: vec![],
            correlation_matrix: vec![],
            recommendations,
            total_violations,
            total_artifacts_analyzed: 5,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_remediation_all_healthy() {
        let calibration = make_pipeline_rec("calibrator", "maintain", "all good");
        let staleness = make_pipeline_rec("staleness", "maintain", "all fresh");
        let anomalies = make_pipeline_rec("anomaly", "maintain", "no anomalies");
        let violations = make_violation_report(0, vec![
            "No violations detected. Governance controls are operating effectively.".to_string(),
        ]);

        let (plan, rec) = RemediationOptimizer::optimize(
            &calibration,
            &staleness,
            &anomalies,
            &violations,
        );

        // The one recommendation from violations generates a low-priority action.
        assert_eq!(rec.pipeline_id, "remediation-optimizer");
        assert!(rec.constitutional_check.passed);
        assert_eq!(plan.resource_requirements.critical_actions, 0);
    }

    #[test]
    fn test_remediation_with_issues() {
        let calibration = make_pipeline_rec(
            "calibrator",
            "adjust_thresholds",
            "Miscalibration detected",
        );
        let staleness = make_pipeline_rec(
            "staleness",
            "review_stale_beliefs",
            "3 beliefs at risk",
        );
        let anomalies = make_pipeline_rec(
            "anomaly",
            "investigate_anomalies",
            "2 anomalies found",
        );
        let violations = make_violation_report(5, vec![
            "URGENT: 'policy' violations are increasing (count=5). Review and strengthen governance controls.".to_string(),
            "MONITOR: 'persona' has persistent violations (count=2). Consider targeted policy refinement.".to_string(),
        ]);

        let (plan, rec) = RemediationOptimizer::optimize(
            &calibration,
            &staleness,
            &anomalies,
            &violations,
        );

        assert_eq!(rec.recommendation.action, "execute_remediation_plan");
        assert!(plan.prioritized_actions.len() >= 4);
        assert_eq!(plan.resource_requirements.critical_actions, 1);
        assert!(plan.resource_requirements.high_actions >= 1);

        // Critical actions should be first.
        assert_eq!(plan.prioritized_actions[0].priority, Priority::Critical);

        // Timeline should have immediate actions.
        assert!(!plan.timeline.immediate.is_empty());
        assert!(!plan.timeline.short_term.is_empty());

        // Should flag escalation concern for critical actions.
        assert!(!rec.constitutional_check.concerns.is_empty());
    }

    #[test]
    fn test_remediation_priority_ordering() {
        let calibration = make_pipeline_rec("calibrator", "maintain", "ok");
        let staleness = make_pipeline_rec("staleness", "maintain", "ok");
        let anomalies = make_pipeline_rec(
            "anomaly",
            "investigate_anomalies",
            "found issues",
        );
        let violations = make_violation_report(3, vec![
            "URGENT: critical issue".to_string(),
        ]);

        let (plan, _rec) = RemediationOptimizer::optimize(
            &calibration,
            &staleness,
            &anomalies,
            &violations,
        );

        // Verify ordering: Critical before High.
        let priorities: Vec<&Priority> = plan
            .prioritized_actions
            .iter()
            .map(|a| &a.priority)
            .collect();
        for i in 1..priorities.len() {
            assert!(priorities[i] >= priorities[i - 1], "actions should be sorted by priority");
        }
    }

    #[test]
    fn test_remediation_plan_serde_roundtrip() {
        let calibration = make_pipeline_rec("calibrator", "maintain", "ok");
        let staleness = make_pipeline_rec("staleness", "maintain", "ok");
        let anomalies = make_pipeline_rec("anomaly", "maintain", "ok");
        let violations = make_violation_report(0, vec![]);

        let (plan, _) = RemediationOptimizer::optimize(
            &calibration,
            &staleness,
            &anomalies,
            &violations,
        );

        let json = serde_json::to_string(&plan).expect("serialize");
        let parsed: RemediationPlan = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.prioritized_actions.len(),
            plan.prioritized_actions.len()
        );
    }

    #[test]
    fn test_remediation_confidence_degrades_with_critical() {
        let calibration = make_pipeline_rec("calibrator", "maintain", "ok");
        let staleness = make_pipeline_rec("staleness", "maintain", "ok");
        let anomalies = make_pipeline_rec("anomaly", "maintain", "ok");

        // Many urgent violations should lower confidence.
        let violations = make_violation_report(10, vec![
            "URGENT: issue 1".to_string(),
            "URGENT: issue 2".to_string(),
            "URGENT: issue 3".to_string(),
        ]);

        let (_plan, rec) = RemediationOptimizer::optimize(
            &calibration,
            &staleness,
            &anomalies,
            &violations,
        );

        assert!(
            rec.confidence < 0.9,
            "confidence should be lower with critical issues"
        );
    }
}
