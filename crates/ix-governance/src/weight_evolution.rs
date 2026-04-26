//! ML-driven weight training — accumulates research cycle outcomes to produce
//! per-department weight recommendations for hypothesis and test methods,
//! and detects cross-department transfer opportunities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── Data types ──────────────────────────────────────────────────────────────

/// A single research cycle outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchOutcome {
    pub department: String,
    pub hypothesis_method: String,
    pub test_method: String,
    pub success: bool,
    pub confidence: f64,
}

/// Weight recommendation for a department's research strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRecommendation {
    pub department: String,
    pub hypothesis_weights: Vec<(String, f64)>,
    pub test_weights: Vec<(String, f64)>,
    pub confidence: f64,
    pub training_samples: usize,
    pub recommendation: String,
}

/// A cross-department transfer opportunity — a method works well in one
/// department but poorly in another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferOpportunity {
    pub source_department: String,
    pub target_department: String,
    pub method: String,
    pub source_success_rate: f64,
    pub target_success_rate: f64,
    pub improvement_potential: f64,
}

// ── Evolver ─────────────────────────────────────────────────────────────────

/// Trains on accumulated research outcomes and produces weight recommendations.
pub struct WeightEvolver;

impl WeightEvolver {
    /// Train on accumulated outcomes, produce weight recommendations per department.
    ///
    /// For each department, computes success rate per hypothesis method and test
    /// method, normalises to probability weights, and derives an asymptotic
    /// confidence score: `sqrt(n) / (sqrt(n) + 5.0)`.
    pub fn train(outcomes: &[ResearchOutcome]) -> Vec<WeightRecommendation> {
        // Group by department.
        let mut by_dept: HashMap<String, Vec<&ResearchOutcome>> = HashMap::new();
        for o in outcomes {
            by_dept.entry(o.department.clone()).or_default().push(o);
        }

        let mut recommendations: Vec<WeightRecommendation> = Vec::new();

        for (dept, dept_outcomes) in &by_dept {
            let training_samples = dept_outcomes.len();

            let hypothesis_weights = Self::compute_weights(
                dept_outcomes
                    .iter()
                    .map(|o| (o.hypothesis_method.as_str(), o.success)),
            );
            let test_weights = Self::compute_weights(
                dept_outcomes
                    .iter()
                    .map(|o| (o.test_method.as_str(), o.success)),
            );

            let n = training_samples as f64;
            let confidence = n.sqrt() / (n.sqrt() + 5.0);

            let top_hyp = hypothesis_weights
                .first()
                .map(|(m, _)| m.as_str())
                .unwrap_or("none");
            let top_test = test_weights
                .first()
                .map(|(m, _)| m.as_str())
                .unwrap_or("none");

            let recommendation = format!(
                "Prefer {top_hyp} hypotheses with {top_test} testing (confidence: {confidence:.2})."
            );

            recommendations.push(WeightRecommendation {
                department: dept.clone(),
                hypothesis_weights,
                test_weights,
                confidence,
                training_samples,
                recommendation,
            });
        }

        recommendations
    }

    /// Detect cross-department transfer opportunities.
    ///
    /// For each method, if department A has success_rate > 0.7 and department B
    /// has success_rate < 0.3 for the same method, report a transfer opportunity
    /// from A to B.
    pub fn detect_transfers(recommendations: &[WeightRecommendation]) -> Vec<TransferOpportunity> {
        // Collect all method→(department, weight) for both hypothesis and test.
        let mut method_rates: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for rec in recommendations {
            for (method, weight) in &rec.hypothesis_weights {
                method_rates
                    .entry(method.clone())
                    .or_default()
                    .push((rec.department.clone(), *weight));
            }
            for (method, weight) in &rec.test_weights {
                method_rates
                    .entry(method.clone())
                    .or_default()
                    .push((rec.department.clone(), *weight));
            }
        }

        let mut transfers = Vec::new();

        for (method, dept_rates) in &method_rates {
            for (source_dept, source_rate) in dept_rates {
                if *source_rate <= 0.7 {
                    continue;
                }
                for (target_dept, target_rate) in dept_rates {
                    if source_dept == target_dept || *target_rate >= 0.3 {
                        continue;
                    }
                    transfers.push(TransferOpportunity {
                        source_department: source_dept.clone(),
                        target_department: target_dept.clone(),
                        method: method.clone(),
                        source_success_rate: *source_rate,
                        target_success_rate: *target_rate,
                        improvement_potential: source_rate - target_rate,
                    });
                }
            }
        }

        transfers
    }

    /// Read research cycle logs from state directory.
    pub fn read_outcomes(state_dir: &Path, department: &str) -> Vec<ResearchOutcome> {
        let dir = state_dir.join("streeling").join("research");
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let prefix = format!("{department}-");
        let mut outcomes = Vec::new();

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && name_str.ends_with(".cycle.json") {
                if let Ok(contents) = std::fs::read_to_string(entry.path()) {
                    if let Ok(mut parsed) = serde_json::from_str::<Vec<ResearchOutcome>>(&contents)
                    {
                        outcomes.append(&mut parsed);
                    }
                }
            }
        }

        outcomes
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Compute normalised success-rate weights for a set of (method, success) pairs.
    /// Returns a vec sorted by weight descending.
    fn compute_weights<'a>(entries: impl Iterator<Item = (&'a str, bool)>) -> Vec<(String, f64)> {
        let mut counts: HashMap<String, (usize, usize)> = HashMap::new(); // (successes, total)
        for (method, success) in entries {
            let entry = counts.entry(method.to_string()).or_insert((0, 0));
            if success {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        let mut rates: Vec<(String, f64)> = counts
            .into_iter()
            .map(|(method, (succ, total))| (method, succ as f64 / total as f64))
            .collect();

        // Normalise so weights sum to 1.0.
        let sum: f64 = rates.iter().map(|(_, r)| *r).sum();
        if sum > 0.0 {
            for (_, r) in &mut rates {
                *r /= sum;
            }
        }

        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rates
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(dept: &str, hyp: &str, test: &str, success: bool) -> ResearchOutcome {
        ResearchOutcome {
            department: dept.to_string(),
            hypothesis_method: hyp.to_string(),
            test_method: test.to_string(),
            success,
            confidence: if success { 0.9 } else { 0.3 },
        }
    }

    #[test]
    fn train_empty() {
        let recs = WeightEvolver::train(&[]);
        assert!(recs.is_empty());
    }

    #[test]
    fn train_single_dept() {
        let outcomes = vec![
            make_outcome("psychohistory", "inductive", "empirical", true),
            make_outcome("psychohistory", "inductive", "simulation", false),
            make_outcome("psychohistory", "deductive", "empirical", true),
        ];
        let recs = WeightEvolver::train(&outcomes);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].department, "psychohistory");
        assert_eq!(recs[0].training_samples, 3);
        assert!(!recs[0].hypothesis_weights.is_empty());
        assert!(!recs[0].test_weights.is_empty());
    }

    #[test]
    fn train_normalizes() {
        let outcomes = vec![
            make_outcome("psychohistory", "inductive", "empirical", true),
            make_outcome("psychohistory", "deductive", "simulation", true),
            make_outcome("psychohistory", "abductive", "formal_proof", false),
        ];
        let recs = WeightEvolver::train(&outcomes);
        assert_eq!(recs.len(), 1);

        let hyp_sum: f64 = recs[0].hypothesis_weights.iter().map(|(_, w)| w).sum();
        assert!(
            (hyp_sum - 1.0).abs() < 1e-9,
            "hypothesis weights should sum to 1.0, got {hyp_sum}"
        );

        let test_sum: f64 = recs[0].test_weights.iter().map(|(_, w)| w).sum();
        assert!(
            (test_sum - 1.0).abs() < 1e-9,
            "test weights should sum to 1.0, got {test_sum}"
        );
    }

    #[test]
    fn detect_transfers() {
        let recs = vec![
            WeightRecommendation {
                department: "psychohistory".to_string(),
                hypothesis_weights: vec![("inductive".to_string(), 0.9)],
                test_weights: vec![("empirical".to_string(), 0.8)],
                confidence: 0.5,
                training_samples: 10,
                recommendation: String::new(),
            },
            WeightRecommendation {
                department: "linguistics".to_string(),
                hypothesis_weights: vec![("inductive".to_string(), 0.2)],
                test_weights: vec![("empirical".to_string(), 0.1)],
                confidence: 0.5,
                training_samples: 10,
                recommendation: String::new(),
            },
        ];
        let transfers = WeightEvolver::detect_transfers(&recs);
        assert!(!transfers.is_empty());
        let t = &transfers[0];
        assert_eq!(t.source_department, "psychohistory");
        assert_eq!(t.target_department, "linguistics");
        assert!(t.improvement_potential > 0.5);
    }

    #[test]
    fn serde_roundtrip() {
        let rec = WeightRecommendation {
            department: "psychohistory".to_string(),
            hypothesis_weights: vec![
                ("inductive".to_string(), 0.6),
                ("deductive".to_string(), 0.4),
            ],
            test_weights: vec![("empirical".to_string(), 1.0)],
            confidence: 0.71,
            training_samples: 25,
            recommendation: "Prefer inductive hypotheses.".to_string(),
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: WeightRecommendation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.department, rec.department);
        assert_eq!(back.training_samples, rec.training_samples);
        assert!((back.confidence - rec.confidence).abs() < 1e-9);
    }
}
