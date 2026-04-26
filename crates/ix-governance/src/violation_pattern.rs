//! Violation pattern analysis — reads evolution state files and identifies
//! patterns in constitutional violations (frequency, severity, correlations).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::feedback::{ConstitutionalCheck, MlEvidence, Recommendation};
use crate::feedback::{EvolutionLog, MlFeedbackRecommendation};

// ── Report types ────────────────────────────────────────────────────────────

/// A trend direction for violation frequency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// A single article's violation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleViolationSummary {
    pub article: String,
    pub violation_count: u32,
    pub compliance_rate: f64,
    pub trend: TrendDirection,
}

/// Correlation between two artifacts that share violation patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationCorrelation {
    pub artifact_a: String,
    pub artifact_b: String,
    /// Correlation strength from 0.0 (none) to 1.0 (perfect).
    pub strength: f64,
}

/// Full report produced by the ViolationPatternAnalyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPatternReport {
    pub most_violated_articles: Vec<ArticleViolationSummary>,
    pub violation_trends: Vec<ArticleViolationSummary>,
    pub correlation_matrix: Vec<ViolationCorrelation>,
    pub recommendations: Vec<String>,
    pub total_violations: u32,
    pub total_artifacts_analyzed: u32,
    pub timestamp: DateTime<Utc>,
}

// ── Analyzer ────────────────────────────────────────────────────────────────

/// Analyzes evolution logs to identify patterns in constitutional violations.
pub struct ViolationPatternAnalyzer;

impl ViolationPatternAnalyzer {
    /// Analyze evolution logs and produce a violation pattern report plus
    /// an ML feedback recommendation.
    ///
    /// Identifies:
    /// - Which artifact types have the most violations
    /// - Violation frequency trends (based on event timestamps)
    /// - Correlations between artifacts that violate together
    /// - Actionable recommendations for reducing violations
    pub fn analyze(
        evolution: &[EvolutionLog],
    ) -> (ViolationPatternReport, MlFeedbackRecommendation) {
        let now = Utc::now();

        // ── Aggregate violations by artifact type ───────────────────────
        let mut type_violations: HashMap<String, (u32, f64, u32)> = HashMap::new(); // (violations, compliance_sum, count)

        for log in evolution {
            let entry = type_violations
                .entry(log.artifact_type.clone())
                .or_insert((0, 0.0, 0));
            entry.0 += log.metrics.violation_count;
            entry.1 += log.metrics.compliance_rate;
            entry.2 += 1;
        }

        // ── Build article violation summaries ───────────────────────────
        let mut summaries: Vec<ArticleViolationSummary> = type_violations
            .iter()
            .map(|(artifact_type, (violations, compliance_sum, count))| {
                let avg_compliance = if *count > 0 {
                    compliance_sum / *count as f64
                } else {
                    1.0
                };

                // Determine trend from violation events in the logs.
                let trend = Self::compute_trend(evolution, artifact_type);

                ArticleViolationSummary {
                    article: artifact_type.clone(),
                    violation_count: *violations,
                    compliance_rate: avg_compliance,
                    trend,
                }
            })
            .collect();

        // Sort by violation count descending.
        summaries.sort_by_key(|s| std::cmp::Reverse(s.violation_count));

        // ── Compute correlation matrix ──────────────────────────────────
        let correlations = Self::compute_correlations(evolution);

        // ── Total violations ────────────────────────────────────────────
        let total_violations: u32 = evolution.iter().map(|l| l.metrics.violation_count).sum();
        let total_artifacts = evolution.len() as u32;

        // ── Generate recommendations ────────────────────────────────────
        let mut recommendations = Vec::new();

        for summary in &summaries {
            if summary.violation_count > 0 {
                let rec = match summary.trend {
                    TrendDirection::Increasing => format!(
                        "URGENT: '{}' violations are increasing (count={}). Review and strengthen governance controls.",
                        summary.article, summary.violation_count
                    ),
                    TrendDirection::Stable => format!(
                        "MONITOR: '{}' has persistent violations (count={}). Consider targeted policy refinement.",
                        summary.article, summary.violation_count
                    ),
                    TrendDirection::Decreasing => format!(
                        "POSITIVE: '{}' violations are decreasing (count={}). Current remediation is working.",
                        summary.article, summary.violation_count
                    ),
                };
                recommendations.push(rec);
            }
        }

        if total_violations == 0 {
            recommendations.push(
                "No violations detected. Governance controls are operating effectively."
                    .to_string(),
            );
        }

        // ── Build the report ────────────────────────────────────────────
        let report = ViolationPatternReport {
            most_violated_articles: summaries.clone(),
            violation_trends: summaries,
            correlation_matrix: correlations,
            recommendations: recommendations.clone(),
            total_violations,
            total_artifacts_analyzed: total_artifacts,
            timestamp: now,
        };

        // ── Build ML feedback recommendation ────────────────────────────
        let (action, rationale) = if total_violations > 0 {
            let top_violators: Vec<String> = report
                .most_violated_articles
                .iter()
                .filter(|s| s.violation_count > 0)
                .take(3)
                .map(|s| format!("{}({})", s.article, s.violation_count))
                .collect();
            (
                "address_violation_patterns".to_string(),
                format!(
                    "{total_violations} violation(s) across {total_artifacts} artifact(s). Top: {}",
                    top_violators.join(", ")
                ),
            )
        } else {
            (
                "maintain".to_string(),
                format!("No violations across {total_artifacts} artifact(s)"),
            )
        };

        let recommendation = MlFeedbackRecommendation {
            pipeline_id: "violation-pattern-analyzer".to_string(),
            recommendation_type: "violation-pattern".to_string(),
            recommendation: Recommendation {
                action,
                rationale,
                expected_impact: if total_violations > 0 {
                    format!("Targeted remediation of {} violation pattern(s) to improve overall compliance", recommendations.len())
                } else {
                    "Continued compliance through proactive monitoring".to_string()
                },
            },
            confidence: if total_violations == 0 { 0.95 } else { 0.85 },
            evidence: MlEvidence {
                data_points: total_artifacts,
                model_version: "violation-pattern-v1".to_string(),
                training_window: "all-time".to_string(),
            },
            constitutional_check: ConstitutionalCheck {
                passed: true,
                articles_checked: vec![
                    "Article 7 (Auditability)".to_string(),
                    "Article 8 (Observability)".to_string(),
                    "Article 11 (Ethical Stewardship)".to_string(),
                ],
                concerns: if total_violations > 0 {
                    vec!["Active violations require remediation per Article 11".to_string()]
                } else {
                    Vec::new()
                },
            },
            timestamp: now,
        };

        (report, recommendation)
    }

    /// Compute the violation trend for a given artifact type by examining
    /// the temporal distribution of "violated" events.
    fn compute_trend(evolution: &[EvolutionLog], artifact_type: &str) -> TrendDirection {
        let mut violation_timestamps: Vec<DateTime<Utc>> = Vec::new();

        for log in evolution
            .iter()
            .filter(|l| l.artifact_type == artifact_type)
        {
            for event in &log.events {
                if event.event_type == "violated" {
                    violation_timestamps.push(event.timestamp);
                }
            }
        }

        if violation_timestamps.len() < 2 {
            return TrendDirection::Stable;
        }

        violation_timestamps.sort();

        // Split the time range into two halves and count events in each.
        let first = violation_timestamps[0];
        let last = *violation_timestamps.last().unwrap();
        let midpoint = first + (last - first) / 2;

        let early_count = violation_timestamps
            .iter()
            .filter(|t| **t <= midpoint)
            .count();
        let late_count = violation_timestamps
            .iter()
            .filter(|t| **t > midpoint)
            .count();

        if late_count > early_count {
            TrendDirection::Increasing
        } else if early_count > late_count {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    /// Compute correlations between artifacts that tend to be violated together.
    fn compute_correlations(evolution: &[EvolutionLog]) -> Vec<ViolationCorrelation> {
        let violated: Vec<&EvolutionLog> = evolution
            .iter()
            .filter(|l| l.metrics.violation_count > 0)
            .collect();

        let mut correlations = Vec::new();

        // For each pair of violated artifacts, compute a simple correlation
        // based on whether they share similar compliance rates (co-degradation).
        for i in 0..violated.len() {
            for j in (i + 1)..violated.len() {
                let a = violated[i];
                let b = violated[j];

                // Correlation strength: how similar are their compliance rates?
                // Artifacts degrading together have similar (low) compliance rates.
                let rate_diff = (a.metrics.compliance_rate - b.metrics.compliance_rate).abs();
                let strength = 1.0 - rate_diff;

                if strength > 0.5 {
                    correlations.push(ViolationCorrelation {
                        artifact_a: a.artifact.clone(),
                        artifact_b: b.artifact.clone(),
                        strength,
                    });
                }
            }
        }

        correlations.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        correlations
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::{Assessment, EffectivenessAssessment, EvolutionEvent, EvolutionMetrics};
    use crate::TruthValue;

    fn make_evolution_log(
        artifact: &str,
        artifact_type: &str,
        violation_count: u32,
        compliance_rate: f64,
        events: Vec<EvolutionEvent>,
    ) -> EvolutionLog {
        let now = Utc::now();
        EvolutionLog {
            id: format!("{artifact}-evo"),
            artifact: artifact.to_string(),
            artifact_type: artifact_type.to_string(),
            metrics: EvolutionMetrics {
                citation_count: 5,
                violation_count,
                compliance_rate,
                last_cited: Some(now),
                last_violated: if violation_count > 0 { Some(now) } else { None },
                promotion_candidate: false,
                deprecation_candidate: false,
            },
            events,
            assessment: Assessment {
                effectiveness: EffectivenessAssessment {
                    proposition: format!("{artifact} is effective"),
                    truth_value: TruthValue::True,
                    confidence: 0.9,
                },
                recommendation: "maintain".to_string(),
            },
            created_at: now,
            last_updated: now,
        }
    }

    #[test]
    fn test_violation_pattern_no_violations() {
        let logs = vec![
            make_evolution_log("policy-a", "policy", 0, 1.0, vec![]),
            make_evolution_log("policy-b", "policy", 0, 1.0, vec![]),
        ];

        let (report, rec) = ViolationPatternAnalyzer::analyze(&logs);

        assert_eq!(report.total_violations, 0);
        assert_eq!(report.total_artifacts_analyzed, 2);
        assert!(report.correlation_matrix.is_empty());
        assert_eq!(rec.pipeline_id, "violation-pattern-analyzer");
        assert_eq!(rec.recommendation.action, "maintain");
        assert_eq!(rec.confidence, 0.95);
        assert!(rec.constitutional_check.passed);
    }

    #[test]
    fn test_violation_pattern_with_violations() {
        let now = Utc::now();
        let logs = vec![
            make_evolution_log(
                "alignment-policy",
                "policy",
                3,
                0.7,
                vec![
                    EvolutionEvent {
                        event_type: "violated".to_string(),
                        context: "test violation 1".to_string(),
                        timestamp: now - chrono::Duration::days(5),
                    },
                    EvolutionEvent {
                        event_type: "violated".to_string(),
                        context: "test violation 2".to_string(),
                        timestamp: now - chrono::Duration::days(3),
                    },
                    EvolutionEvent {
                        event_type: "violated".to_string(),
                        context: "test violation 3".to_string(),
                        timestamp: now - chrono::Duration::days(1),
                    },
                ],
            ),
            make_evolution_log("safe-persona", "persona", 1, 0.85, vec![]),
            make_evolution_log("clean-policy", "policy", 0, 1.0, vec![]),
        ];

        let (report, rec) = ViolationPatternAnalyzer::analyze(&logs);

        assert_eq!(report.total_violations, 4);
        assert_eq!(report.total_artifacts_analyzed, 3);
        assert!(!report.most_violated_articles.is_empty());
        assert_eq!(rec.recommendation.action, "address_violation_patterns");
        assert!(rec.recommendation.rationale.contains("4 violation(s)"));
        assert!(rec.constitutional_check.passed);
        assert!(!rec.constitutional_check.concerns.is_empty());
    }

    #[test]
    fn test_violation_pattern_correlations() {
        let logs = vec![
            make_evolution_log("artifact-a", "policy", 2, 0.75, vec![]),
            make_evolution_log("artifact-b", "policy", 3, 0.78, vec![]),
            make_evolution_log("artifact-c", "persona", 0, 1.0, vec![]),
        ];

        let (report, _rec) = ViolationPatternAnalyzer::analyze(&logs);

        // artifact-a and artifact-b both have violations and similar compliance
        // rates, so they should be correlated.
        assert!(
            !report.correlation_matrix.is_empty(),
            "should find correlated violation patterns"
        );
        let corr = &report.correlation_matrix[0];
        assert!(corr.strength > 0.5);
    }

    #[test]
    fn test_violation_pattern_recommendations() {
        let logs = vec![make_evolution_log(
            "failing-policy",
            "policy",
            5,
            0.5,
            vec![],
        )];

        let (report, _rec) = ViolationPatternAnalyzer::analyze(&logs);

        assert!(!report.recommendations.is_empty());
        assert!(report.recommendations[0].contains("policy"));
    }

    #[test]
    fn test_violation_pattern_trend_detection() {
        let now = Utc::now();
        // Create events concentrated in the second half (increasing trend).
        let events = vec![
            EvolutionEvent {
                event_type: "violated".to_string(),
                context: "old".to_string(),
                timestamp: now - chrono::Duration::days(10),
            },
            EvolutionEvent {
                event_type: "violated".to_string(),
                context: "recent-1".to_string(),
                timestamp: now - chrono::Duration::days(2),
            },
            EvolutionEvent {
                event_type: "violated".to_string(),
                context: "recent-2".to_string(),
                timestamp: now - chrono::Duration::days(1),
            },
            EvolutionEvent {
                event_type: "violated".to_string(),
                context: "recent-3".to_string(),
                timestamp: now,
            },
        ];

        let logs = vec![make_evolution_log("worsening", "policy", 4, 0.6, events)];

        let (report, _rec) = ViolationPatternAnalyzer::analyze(&logs);

        // The trend should be detected based on event distribution.
        let policy_summary = report
            .violation_trends
            .iter()
            .find(|s| s.article == "policy")
            .expect("should have policy summary");
        assert_eq!(policy_summary.trend, TrendDirection::Increasing);
    }

    #[test]
    fn test_violation_report_serde_roundtrip() {
        let logs = vec![make_evolution_log(
            "test-artifact",
            "policy",
            2,
            0.8,
            vec![],
        )];
        let (report, _) = ViolationPatternAnalyzer::analyze(&logs);
        let json = serde_json::to_string(&report).expect("serialize");
        let parsed: ViolationPatternReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.total_violations, report.total_violations);
    }
}
