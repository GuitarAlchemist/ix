//! Reconciles raw extractor output against external evidence.
//!
//! Given a `Vec<Annotation>` (from [`crate::walker::extract`]) and a list of
//! known test files, the reconciler:
//!
//! 1. **Test-coverage match.** Tries to find a test file path that "exercises"
//!    each annotation. Heuristic (necessarily fuzzy):
//!    - test file path contains the annotation file's basename (or a `_test` /
//!      `tests/` variant), AND
//!    - test file contains at least one significant word from the claim.
//! 2. **Contradiction promotion.** Two annotations at the same `(path, line,
//!    kind)` with different truth values get promoted to `C` (Contradictory).
//!    The originals stay in the output; the synthesized C-promoted annotation
//!    is appended with `reconciliation.promoted_to_c_from` populated.
//! 3. **Staleness.** Compares file mtime against `annotation.updated_at`. If
//!    the file was touched more than `stale_threshold_days` (default 7) after
//!    the annotation's `updated_at`, the annotation is flagged `stale = true`.
//! 4. **Multi-source weighted aggregation.** Multiple annotations claiming the
//!    same thing (same `(path, line, kind, claim)` but different `source.author`)
//!    are aggregated using confidence-weighted hexavalent voting.

use crate::types::{Annotation, AnnotationKind, Reconciliation, TruthValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Configuration for one reconciliation run.
#[derive(Debug, Clone)]
pub struct ReconcilerConfig {
    /// Files (relative to workspace) considered "tests" for coverage matching.
    pub test_files: Vec<PathBuf>,
    /// Workspace root, used for resolving relative paths in [`Annotation::location`].
    pub workspace: PathBuf,
    /// Stale threshold in days. Default 7.
    pub stale_threshold_days: i64,
}

impl ReconcilerConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            test_files: Vec::new(),
            workspace: workspace.into(),
            stale_threshold_days: 7,
        }
    }

    pub fn with_test_files(mut self, files: Vec<PathBuf>) -> Self {
        self.test_files = files;
        self
    }
}

/// Aggregate report of one reconciliation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub generated_at: String,
    pub total_annotations: usize,
    pub by_truth_value: HashMap<String, usize>,
    pub by_certainty: HashMap<String, usize>,
    pub by_kind: HashMap<String, usize>,
    pub verified_by_test: usize,
    pub stale: usize,
    pub contradictory: usize,
    /// All annotations after reconciliation (originals + any synthesized C-promotions).
    pub annotations: Vec<Annotation>,
}

/// Reconcile a vec of annotations. Returns a new vec (input untouched) plus
/// summary stats.
pub fn reconcile(mut annotations: Vec<Annotation>, cfg: &ReconcilerConfig) -> ReconciliationReport {
    // 1. Test-coverage match
    apply_test_matches(&mut annotations, cfg);
    // 2. Staleness from file mtime
    apply_staleness(&mut annotations, cfg);
    // 3. Contradiction promotion + multi-source weighted aggregation
    let synthesized = synthesize_contradictions_and_weighted(&annotations);
    annotations.extend(synthesized);

    build_report(annotations)
}

fn build_report(annotations: Vec<Annotation>) -> ReconciliationReport {
    let mut by_truth_value: HashMap<String, usize> = HashMap::new();
    let mut by_certainty: HashMap<String, usize> = HashMap::new();
    let mut by_kind: HashMap<String, usize> = HashMap::new();
    let mut verified_by_test = 0usize;
    let mut stale = 0usize;
    let mut contradictory = 0usize;

    for a in &annotations {
        *by_truth_value
            .entry(a.truth_value.as_str().to_string())
            .or_insert(0) += 1;
        let cert_key = serde_json::to_value(a.certainty)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        *by_certainty.entry(cert_key).or_insert(0) += 1;
        let kind_key = serde_json::to_value(a.kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        *by_kind.entry(kind_key).or_insert(0) += 1;

        if a.truth_value == TruthValue::T
            && a.reconciliation
                .as_ref()
                .and_then(|r| r.test_match.as_ref())
                .is_some()
        {
            verified_by_test += 1;
        }
        if a.stale {
            stale += 1;
        }
        if a.truth_value == TruthValue::C {
            contradictory += 1;
        }
    }

    ReconciliationReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        total_annotations: annotations.len(),
        by_truth_value,
        by_certainty,
        by_kind,
        verified_by_test,
        stale,
        contradictory,
        annotations,
    }
}

fn apply_test_matches(annotations: &mut [Annotation], cfg: &ReconcilerConfig) {
    if cfg.test_files.is_empty() {
        return;
    }
    for a in annotations.iter_mut() {
        if let Some(match_path) = find_test_match(a, cfg) {
            let r = a.reconciliation.get_or_insert_with(Reconciliation::default);
            r.test_match = Some(match_path);
        }
    }
}

fn find_test_match(a: &Annotation, cfg: &ReconcilerConfig) -> Option<String> {
    let basename = Path::new(&a.location.path)
        .file_stem()?
        .to_string_lossy()
        .to_lowercase();
    let keywords = significant_words(&a.claim);
    if keywords.is_empty() {
        return None;
    }

    for test_path in &cfg.test_files {
        let test_str = test_path.to_string_lossy().to_lowercase();
        // Must reference the same basename OR live under tests/ for the same parent dir.
        let path_match = test_str.contains(&basename)
            || (test_str.contains("tests/") || test_str.contains("tests\\") || test_str.contains("/tests/"));
        if !path_match {
            continue;
        }
        // Read the test file and look for any significant keyword from the claim.
        let abs = cfg.workspace.join(test_path);
        let body = match fs::read_to_string(&abs) {
            Ok(s) => s.to_lowercase(),
            Err(_) => continue,
        };
        if keywords.iter().any(|w| body.contains(w)) {
            return Some(test_path.to_string_lossy().replace('\\', "/"));
        }
    }
    None
}

fn significant_words(claim: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
        "on", "at", "for", "with", "by", "from", "and", "or", "not", "no", "do", "does", "did",
        "this", "that", "it", "its", "as", "if", "then", "than", "but", "so", "we", "i", "you",
        "they", "them", "all", "any", "some", "must", "should", "will", "can", "could", "would",
    ];
    claim
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                .to_lowercase()
        })
        .filter(|w| w.len() >= 3 && !STOP.contains(&w.as_str()))
        .collect()
}

fn apply_staleness(annotations: &mut [Annotation], cfg: &ReconcilerConfig) {
    let threshold = chrono::Duration::days(cfg.stale_threshold_days);
    for a in annotations.iter_mut() {
        let abs = cfg.workspace.join(&a.location.path);
        let Ok(meta) = fs::metadata(&abs) else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        let mtime_dt = system_time_to_utc(mtime);
        let Ok(updated) = chrono::DateTime::parse_from_rfc3339(&a.updated_at) else {
            continue;
        };
        let updated_utc = updated.with_timezone(&chrono::Utc);
        if mtime_dt > updated_utc + threshold {
            a.stale = true;
        }
    }
}

fn system_time_to_utc(t: SystemTime) -> chrono::DateTime<chrono::Utc> {
    let dur = t
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    chrono::DateTime::<chrono::Utc>::from_timestamp(dur.as_secs() as i64, dur.subsec_nanos())
        .unwrap_or_else(chrono::Utc::now)
}

/// Group annotations by (path, line_start, kind). Where a group spans multiple
/// truth values, synthesize a `C` (Contradictory) annotation. Where a group has
/// multiple sources with the same truth value, run confidence-weighted
/// aggregation and emit a synthesized `weighted_*` reconciliation block.
fn synthesize_contradictions_and_weighted(annotations: &[Annotation]) -> Vec<Annotation> {
    let mut groups: HashMap<(String, u32, AnnotationKind), Vec<&Annotation>> = HashMap::new();
    for a in annotations {
        groups
            .entry((a.location.path.clone(), a.location.line_start, a.kind))
            .or_default()
            .push(a);
    }

    let mut synthesized = Vec::new();
    for ((path, line, kind), members) in groups {
        if members.len() < 2 {
            continue;
        }
        let truth_values: Vec<TruthValue> = members.iter().map(|a| a.truth_value).collect();
        let distinct: std::collections::BTreeSet<_> = truth_values.iter().collect();
        if distinct.len() > 1 {
            // Contradiction!
            let claim = format!(
                "[reconciler] {} sources disagree at {}:{} ({:?})",
                members.len(),
                path,
                line,
                kind
            );
            let now = chrono::Utc::now().to_rfc3339();
            let id =
                crate::types::annotation_id(&path, line, AnnotationKind::Decision, &claim);
            let weighted = weighted_truth(members.iter().copied());
            let mut promoted_to_c_from: Vec<TruthValue> = distinct.into_iter().copied().collect();
            promoted_to_c_from.sort_by_key(|t| t.as_str());
            synthesized.push(Annotation {
                schema_version: crate::types::SCHEMA_VERSION,
                id,
                kind: AnnotationKind::Decision,
                claim,
                truth_value: TruthValue::C,
                certainty: crate::types::Certainty::Inferred,
                confidence: weighted.1,
                source: crate::types::Source {
                    author: "auto".to_string(),
                    model: Some("ix-ai-annotations-reconciler".to_string()),
                    evidence: None,
                },
                location: crate::types::Location {
                    path: path.clone(),
                    line_start: line,
                    line_end: line,
                },
                created_at: now.clone(),
                updated_at: now,
                stale: false,
                reconciliation: Some(Reconciliation {
                    test_match: None,
                    promoted_to_c_from,
                    weighted_truth_value: Some(weighted.0),
                    weighted_confidence: Some(weighted.1),
                }),
            });
        } else {
            // Multiple sources, consistent value -> emit a weighted aggregate.
            let weighted = weighted_truth(members.iter().copied());
            let now = chrono::Utc::now().to_rfc3339();
            let claim = format!(
                "[reconciler] weighted across {} sources at {}:{}",
                members.len(),
                path,
                line
            );
            let id =
                crate::types::annotation_id(&path, line, AnnotationKind::Decision, &claim);
            synthesized.push(Annotation {
                schema_version: crate::types::SCHEMA_VERSION,
                id,
                kind: AnnotationKind::Decision,
                claim,
                truth_value: weighted.0,
                certainty: crate::types::Certainty::Inferred,
                confidence: weighted.1,
                source: crate::types::Source {
                    author: "auto".to_string(),
                    model: Some("ix-ai-annotations-reconciler".to_string()),
                    evidence: None,
                },
                location: crate::types::Location {
                    path,
                    line_start: line,
                    line_end: line,
                },
                created_at: now.clone(),
                updated_at: now,
                stale: false,
                reconciliation: Some(Reconciliation {
                    test_match: None,
                    promoted_to_c_from: Vec::new(),
                    weighted_truth_value: Some(weighted.0),
                    weighted_confidence: Some(weighted.1),
                }),
            });
        }
    }
    synthesized
}

/// Confidence-weighted hexavalent voting. Returns `(argmax_truth_value, avg_confidence)`.
/// Tie-break order matches Demerzel: C > U > T > F (P/D pull toward T/F respectively).
fn weighted_truth<'a, I: Iterator<Item = &'a Annotation>>(it: I) -> (TruthValue, f64) {
    let mut buckets: HashMap<TruthValue, f64> = HashMap::new();
    let mut total = 0.0;
    let mut count = 0;
    for a in it {
        *buckets.entry(a.truth_value).or_insert(0.0) += a.confidence;
        total += a.confidence;
        count += 1;
    }
    let avg = if count > 0 { total / count as f64 } else { 0.0 };
    // Tie-break: C > U > F > D > T > P (escalation-favoring)
    let order = [
        TruthValue::C,
        TruthValue::U,
        TruthValue::F,
        TruthValue::D,
        TruthValue::T,
        TruthValue::P,
    ];
    let max_weight = buckets.values().copied().fold(0.0_f64, f64::max);
    let winner = order
        .into_iter()
        .find(|tv| buckets.get(tv).copied().unwrap_or(0.0) == max_weight)
        .unwrap_or(TruthValue::U);
    (winner, avg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Certainty, Location, Source};

    fn fake(
        path: &str,
        line: u32,
        kind: AnnotationKind,
        tv: TruthValue,
        conf: f64,
        author: &str,
    ) -> Annotation {
        let claim = "claim".to_string();
        Annotation {
            schema_version: crate::types::SCHEMA_VERSION,
            id: crate::types::annotation_id(path, line, kind, &claim),
            kind,
            claim,
            truth_value: tv,
            certainty: Certainty::Assumed,
            confidence: conf,
            source: Source {
                author: author.to_string(),
                model: None,
                evidence: None,
            },
            location: Location {
                path: path.to_string(),
                line_start: line,
                line_end: line,
            },
            created_at: "2026-05-24T00:00:00Z".to_string(),
            updated_at: "2026-05-24T00:00:00Z".to_string(),
            stale: false,
            reconciliation: None,
        }
    }

    #[test]
    fn contradictory_pair_promotes_to_c() {
        let anns = vec![
            fake("a.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.9, "claude"),
            fake("a.rs", 1, AnnotationKind::Invariant, TruthValue::F, 0.7, "codex"),
        ];
        let cfg = ReconcilerConfig::new(".");
        let report = reconcile(anns, &cfg);
        assert_eq!(report.contradictory, 1);
        let c = report
            .annotations
            .iter()
            .find(|a| a.truth_value == TruthValue::C)
            .expect("C-promotion synthesized");
        let from = &c.reconciliation.as_ref().unwrap().promoted_to_c_from;
        assert!(from.contains(&TruthValue::T) && from.contains(&TruthValue::F));
    }

    #[test]
    fn consistent_pair_weighted_aggregate() {
        let anns = vec![
            fake("b.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.9, "claude"),
            fake("b.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.7, "codex"),
        ];
        let cfg = ReconcilerConfig::new(".");
        let report = reconcile(anns, &cfg);
        let synth = report
            .annotations
            .iter()
            .find(|a| {
                a.kind == AnnotationKind::Decision
                    && a.claim.starts_with("[reconciler] weighted")
            })
            .expect("weighted synth present");
        assert_eq!(synth.truth_value, TruthValue::T);
        let conf = synth.reconciliation.as_ref().unwrap().weighted_confidence.unwrap();
        assert!((conf - 0.8).abs() < 1e-9);
    }

    #[test]
    fn solo_annotation_no_synthesis() {
        let anns = vec![fake("c.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.9, "claude")];
        let cfg = ReconcilerConfig::new(".");
        let report = reconcile(anns, &cfg);
        assert_eq!(report.total_annotations, 1);
    }

    #[test]
    fn weighted_truth_tiebreak_prefers_c() {
        let anns = [
            fake("d.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.5, "claude"),
            fake("d.rs", 1, AnnotationKind::Invariant, TruthValue::C, 0.5, "codex"),
        ];
        let (w, _) = weighted_truth(anns.iter());
        assert_eq!(w, TruthValue::C);
    }

    #[test]
    fn report_counts_truth_values() {
        let anns = vec![
            fake("e.rs", 1, AnnotationKind::Invariant, TruthValue::T, 0.9, "x"),
            fake("e.rs", 2, AnnotationKind::Hint, TruthValue::U, 0.5, "y"),
        ];
        let cfg = ReconcilerConfig::new(".");
        let report = reconcile(anns, &cfg);
        assert_eq!(report.by_truth_value.get("T"), Some(&1));
        assert_eq!(report.by_truth_value.get("U"), Some(&1));
    }
}
