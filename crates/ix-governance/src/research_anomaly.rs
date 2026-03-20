//! Anomaly clustering for Streeling paradigm detection — groups research
//! anomalies by shared production paths and domain context, then assesses
//! paradigm health per department using a Kuhnian escalation ladder.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── Data types ──────────────────────────────────────────────────────────────

/// A single research anomaly recorded during a Streeling research cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchAnomaly {
    pub anomaly_id: String,
    pub cycle_id: String,
    pub department: String,
    pub production_path: Vec<String>,
    pub hypothesis: String,
    pub failure_mode: String,
    pub domain_context: Vec<String>,
    pub severity: f64,
    pub cluster_id: Option<String>,
    pub paradigm_state: String,
}

/// A cluster of related anomalies that share production path elements and
/// domain context within a single department.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCluster {
    pub cluster_id: String,
    pub department: String,
    pub anomaly_count: usize,
    pub anomaly_ids: Vec<String>,
    pub shared_path_elements: Vec<String>,
    pub shared_context: Vec<String>,
    pub avg_severity: f64,
    pub paradigm_state: String,
}

/// Paradigm health assessment for a department.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmAssessment {
    pub department: String,
    pub state: String,
    pub cluster_count: usize,
    pub total_anomalies: usize,
    pub recommendation: String,
}

// ── Clusterer ───────────────────────────────────────────────────────────────

/// Groups anomalies into clusters and assesses paradigm health.
pub struct AnomalyClusterer;

impl AnomalyClusterer {
    /// Cluster anomalies by shared production path elements and domain context.
    ///
    /// Grouping rules:
    /// - Group by department first.
    /// - Within a department, two anomalies belong together when they share
    ///   >= 2 production path elements AND >= 1 domain context element.
    /// - Paradigm state is derived from cluster size:
    ///   1 anomaly → `normal`, 2 → `watch`, 3+ → `tension`,
    ///   avg_severity > 0.7 → `crisis`.
    pub fn cluster(anomalies: &[ResearchAnomaly]) -> Vec<AnomalyCluster> {
        // Group by department.
        let mut by_dept: HashMap<String, Vec<&ResearchAnomaly>> = HashMap::new();
        for a in anomalies {
            by_dept.entry(a.department.clone()).or_default().push(a);
        }

        let mut clusters: Vec<AnomalyCluster> = Vec::new();
        let mut cluster_counter: usize = 0;

        for (dept, dept_anomalies) in &by_dept {
            // Track which anomalies have been assigned to a cluster.
            let mut assigned: Vec<bool> = vec![false; dept_anomalies.len()];

            for i in 0..dept_anomalies.len() {
                if assigned[i] {
                    continue;
                }
                let mut group_indices = vec![i];

                for j in (i + 1)..dept_anomalies.len() {
                    if assigned[j] {
                        continue;
                    }
                    // Check if anomaly j shares enough with anomaly i.
                    if Self::shares_enough(dept_anomalies[i], dept_anomalies[j]) {
                        group_indices.push(j);
                    }
                }

                // Need at least 2 anomalies to form a cluster.
                if group_indices.len() < 2 {
                    continue;
                }

                for &idx in &group_indices {
                    assigned[idx] = true;
                }

                cluster_counter += 1;
                let cluster_id = format!("cluster-{cluster_counter}");
                let group: Vec<&ResearchAnomaly> =
                    group_indices.iter().map(|&idx| dept_anomalies[idx]).collect();

                let anomaly_ids: Vec<String> = group.iter().map(|a| a.anomaly_id.clone()).collect();
                let shared_path = Self::shared_elements(
                    &group.iter().map(|a| &a.production_path).collect::<Vec<_>>(),
                );
                let shared_ctx = Self::shared_elements(
                    &group.iter().map(|a| &a.domain_context).collect::<Vec<_>>(),
                );
                let avg_severity: f64 =
                    group.iter().map(|a| a.severity).sum::<f64>() / group.len() as f64;

                let paradigm_state = if avg_severity > 0.7 {
                    "crisis".to_string()
                } else if group.len() >= 3 {
                    "tension".to_string()
                } else {
                    "watch".to_string()
                };

                clusters.push(AnomalyCluster {
                    cluster_id,
                    department: dept.clone(),
                    anomaly_count: group.len(),
                    anomaly_ids,
                    shared_path_elements: shared_path,
                    shared_context: shared_ctx,
                    avg_severity,
                    paradigm_state,
                });
            }
        }

        clusters
    }

    /// Assess paradigm health for a single department.
    pub fn assess_paradigm(clusters: &[AnomalyCluster], department: &str) -> ParadigmAssessment {
        let dept_clusters: Vec<&AnomalyCluster> =
            clusters.iter().filter(|c| c.department == department).collect();

        let cluster_count = dept_clusters.len();
        let total_anomalies: usize = dept_clusters.iter().map(|c| c.anomaly_count).sum();

        let state = if dept_clusters.iter().any(|c| c.avg_severity > 0.7) {
            "crisis"
        } else if dept_clusters.iter().any(|c| c.anomaly_count >= 3) {
            "tension"
        } else if dept_clusters.iter().any(|c| c.anomaly_count >= 2) {
            "watch"
        } else {
            "normal"
        };

        let recommendation = match state {
            "crisis" => "Immediate paradigm review required — high-severity anomaly cluster detected.".to_string(),
            "tension" => "Schedule paradigm review — recurring anomaly pattern emerging.".to_string(),
            "watch" => "Monitor anomaly cluster — potential pattern forming.".to_string(),
            _ => "No action needed — paradigm operating normally.".to_string(),
        };

        ParadigmAssessment {
            department: department.to_string(),
            state: state.to_string(),
            cluster_count,
            total_anomalies,
            recommendation,
        }
    }

    /// Read anomaly files from Demerzel state directory.
    pub fn read_anomalies(state_dir: &Path, department: &str) -> Vec<ResearchAnomaly> {
        let path = state_dir
            .join("streeling")
            .join("research")
            .join(format!("{department}-anomalies.json"));
        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        serde_json::from_str(&contents).unwrap_or_default()
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Two anomalies share enough when they have >= 2 common production path
    /// elements and >= 1 common domain context element.
    fn shares_enough(a: &ResearchAnomaly, b: &ResearchAnomaly) -> bool {
        let shared_path = a
            .production_path
            .iter()
            .filter(|e| b.production_path.contains(e))
            .count();
        let shared_ctx = a
            .domain_context
            .iter()
            .filter(|e| b.domain_context.contains(e))
            .count();
        shared_path >= 2 && shared_ctx >= 1
    }

    /// Find elements common to all provided vectors.
    fn shared_elements(sets: &[&Vec<String>]) -> Vec<String> {
        if sets.is_empty() {
            return Vec::new();
        }
        let first = sets[0];
        first
            .iter()
            .filter(|e| sets[1..].iter().all(|s| s.contains(e)))
            .cloned()
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_anomaly(
        id: &str,
        dept: &str,
        path: Vec<&str>,
        ctx: Vec<&str>,
        severity: f64,
    ) -> ResearchAnomaly {
        ResearchAnomaly {
            anomaly_id: id.to_string(),
            cycle_id: "cycle-1".to_string(),
            department: dept.to_string(),
            production_path: path.into_iter().map(String::from).collect(),
            hypothesis: "test hypothesis".to_string(),
            failure_mode: "unexpected result".to_string(),
            domain_context: ctx.into_iter().map(String::from).collect(),
            severity,
            cluster_id: None,
            paradigm_state: "normal".to_string(),
        }
    }

    #[test]
    fn cluster_empty() {
        let clusters = AnomalyClusterer::cluster(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn cluster_single() {
        let anomalies = vec![make_anomaly(
            "a1",
            "psychohistory",
            vec!["observe", "hypothesize", "test"],
            vec!["math"],
            0.5,
        )];
        let clusters = AnomalyClusterer::cluster(&anomalies);
        assert!(clusters.is_empty(), "single anomaly should not form a cluster");
    }

    #[test]
    fn cluster_shared_path() {
        let anomalies = vec![
            make_anomaly("a1", "psychohistory", vec!["observe", "hypothesize", "test"], vec!["math", "stats"], 0.4),
            make_anomaly("a2", "psychohistory", vec!["observe", "hypothesize", "conclude"], vec!["math"], 0.5),
            make_anomaly("a3", "psychohistory", vec!["observe", "hypothesize", "validate"], vec!["stats", "math"], 0.3),
        ];
        let clusters = AnomalyClusterer::cluster(&anomalies);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].anomaly_count, 3);
        assert!(clusters[0].shared_path_elements.contains(&"observe".to_string()));
        assert!(clusters[0].shared_path_elements.contains(&"hypothesize".to_string()));
    }

    #[test]
    fn assess_normal() {
        let assessment = AnomalyClusterer::assess_paradigm(&[], "psychohistory");
        assert_eq!(assessment.state, "normal");
        assert_eq!(assessment.cluster_count, 0);
        assert_eq!(assessment.total_anomalies, 0);
    }

    #[test]
    fn assess_tension() {
        let cluster = AnomalyCluster {
            cluster_id: "cluster-1".to_string(),
            department: "psychohistory".to_string(),
            anomaly_count: 4,
            anomaly_ids: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
            shared_path_elements: vec!["observe".into(), "hypothesize".into()],
            shared_context: vec!["math".into()],
            avg_severity: 0.5,
            paradigm_state: "tension".to_string(),
        };
        let assessment = AnomalyClusterer::assess_paradigm(&[cluster], "psychohistory");
        assert_eq!(assessment.state, "tension");
        assert_eq!(assessment.total_anomalies, 4);
    }

    #[test]
    fn assess_crisis() {
        let cluster = AnomalyCluster {
            cluster_id: "cluster-1".to_string(),
            department: "psychohistory".to_string(),
            anomaly_count: 2,
            anomaly_ids: vec!["a1".into(), "a2".into()],
            shared_path_elements: vec!["observe".into(), "hypothesize".into()],
            shared_context: vec!["math".into()],
            avg_severity: 0.85,
            paradigm_state: "crisis".to_string(),
        };
        let assessment = AnomalyClusterer::assess_paradigm(&[cluster], "psychohistory");
        assert_eq!(assessment.state, "crisis");
    }
}
