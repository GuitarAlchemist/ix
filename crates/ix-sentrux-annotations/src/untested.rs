//! Convert sentrux `test_gaps` per-file rankings into `ai-annotation-v1`
//! records, filtered to the intersection with `@ai:business-value`
//! annotations.
//!
//! ## Coverage strategy
//!
//! Sentrux reports thousands of untested files on real codebases (4,186 in
//! ga, 1,142 in ix as of 2026-05-24). Emitting one annotation per untested
//! file would drown the operator. Instead, we surface ONLY untested files
//! that the operator has already marked as **high business value** —
//! the intersection becomes the "REFACTOR FIRST" quadrant on the
//! value × complexity heatmap, now with the additional "untested" signal
//! making them the *single most actionable* set of files in the codebase.
//!
//! ## Annotation shape
//!
//! For each (untested ∩ business-value) file:
//!
//! - `kind: smell`
//! - `claim: "no test coverage detected by sentrux"`
//! - `truth_value: F` (the implicit "this code is tested" claim is refuted)
//! - `certainty: detected-by-sentrux`
//! - `confidence: 1.0`
//! - `source.author: sentrux`
//! - `source.evidence: sentrux-test-gaps@<ISO8601>`
//! - `location: { path, line_start: 1, line_end: 1 }` (whole-file scope)

use crate::SENTRUX_AUTHOR;
use ix_ai_annotations::types::{
    annotation_id, Annotation, AnnotationKind, Certainty, Location, Source, TruthValue,
    SCHEMA_VERSION,
};
use std::collections::HashSet;
use std::path::Path;

/// Sentinel claim text used on every untested-smell annotation produced by
/// the bridge. Stable so the reconciler can dedupe across reruns.
pub const UNTESTED_CLAIM: &str = "no test coverage detected by sentrux";

/// Build one [`Annotation`] for an untested file path. The annotation has
/// whole-file scope (`line_start = line_end = 1`) because sentrux operates
/// at file granularity for coverage.
pub fn untested_file_to_annotation(path: &str, now: &str) -> Annotation {
    let normalized = path.replace('\\', "/");
    let kind = AnnotationKind::Smell;
    let id = annotation_id(&normalized, 1, kind, UNTESTED_CLAIM);
    Annotation {
        schema_version: SCHEMA_VERSION,
        id,
        kind,
        claim: UNTESTED_CLAIM.to_string(),
        truth_value: TruthValue::F,
        certainty: Certainty::DetectedBySentrux,
        confidence: 1.0,
        source: Source {
            author: SENTRUX_AUTHOR.to_string(),
            model: None,
            evidence: Some(format!("sentrux-test-gaps@{now}")),
        },
        location: Location {
            path: normalized,
            line_start: 1,
            line_end: 1,
        },
        created_at: now.to_string(),
        updated_at: now.to_string(),
        stale: false,
        reconciliation: None,
    }
}

/// Extract the set of repo-relative paths that carry at least one
/// `kind=business-value` annotation. Path strings are normalized to
/// forward slashes so they match the form sentrux emits.
pub fn business_value_paths(annotations: &[Annotation]) -> HashSet<String> {
    annotations
        .iter()
        .filter(|a| a.kind == AnnotationKind::BusinessValue)
        .map(|a| a.location.path.replace('\\', "/"))
        .collect()
}

/// Compute the intersection of (untested files reported by sentrux) and
/// (files carrying a `business-value` annotation). One annotation is
/// emitted per file in the intersection.
///
/// `untested_paths` is the list sentrux returned via `test_gaps.top_untested`.
/// `business_value_files` is the precomputed set from
/// [`business_value_paths`]. `now` is the shared ISO-8601 timestamp.
pub fn untested_high_value_annotations(
    untested_paths: &[String],
    business_value_files: &HashSet<String>,
    now: &str,
) -> Vec<Annotation> {
    untested_paths
        .iter()
        .filter_map(|raw| {
            let normalized = raw.replace('\\', "/");
            if business_value_files.contains(&normalized) {
                Some(untested_file_to_annotation(&normalized, now))
            } else {
                None
            }
        })
        .collect()
}

/// Convenience: load every annotation found under `<workspace>` (via
/// the same walker `ix-ai-annotations` uses), then return only the
/// `business-value` subset's paths.
///
/// Returns an empty set on any walker error — coverage detection should
/// be best-effort, not a hard failure.
pub fn business_value_paths_from_workspace(workspace: &Path) -> HashSet<String> {
    match ix_ai_annotations::walker::extract(workspace) {
        Ok(annos) => business_value_paths(&annos),
        Err(_) => HashSet::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_ai_annotations::types::{Source, SCHEMA_VERSION};

    fn make_business_value(path: &str) -> Annotation {
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, 10, AnnotationKind::BusinessValue, "core"),
            kind: AnnotationKind::BusinessValue,
            claim: "core".into(),
            truth_value: TruthValue::T,
            certainty: Certainty::ManuallyReviewed,
            confidence: 0.95,
            source: Source {
                author: "human".into(),
                model: None,
                evidence: None,
            },
            location: Location {
                path: path.into(),
                line_start: 10,
                line_end: 10,
            },
            created_at: "2026-05-24T18:00:00Z".into(),
            updated_at: "2026-05-24T18:00:00Z".into(),
            stale: false,
            reconciliation: None,
        }
    }

    fn make_other_kind(path: &str, kind: AnnotationKind) -> Annotation {
        let mut a = make_business_value(path);
        a.kind = kind;
        a
    }

    #[test]
    fn untested_annotation_carries_canonical_metadata() {
        let a = untested_file_to_annotation("src/foo.rs", "2026-05-24T18:00:00Z");
        assert_eq!(a.kind, AnnotationKind::Smell);
        assert_eq!(a.claim, UNTESTED_CLAIM);
        assert_eq!(a.truth_value, TruthValue::F);
        assert_eq!(a.certainty, Certainty::DetectedBySentrux);
        assert_eq!(a.confidence, 1.0);
        assert_eq!(a.source.author, "sentrux");
        assert_eq!(
            a.source.evidence.as_deref(),
            Some("sentrux-test-gaps@2026-05-24T18:00:00Z")
        );
        assert_eq!(a.location.line_start, 1);
        assert_eq!(a.location.line_end, 1);
    }

    #[test]
    fn windows_backslashes_normalized_to_forward_slashes() {
        let a = untested_file_to_annotation("crates\\foo\\src\\lib.rs", "now");
        assert_eq!(a.location.path, "crates/foo/src/lib.rs");
    }

    #[test]
    fn id_is_stable_across_runs() {
        let a = untested_file_to_annotation("src/foo.rs", "now-1");
        let b = untested_file_to_annotation("src/foo.rs", "now-2");
        // Same path+line+kind+claim => same id; timestamp doesn't change it.
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn business_value_paths_extracts_only_business_value_kind() {
        let annos = vec![
            make_business_value("a.rs"),
            make_other_kind("b.rs", AnnotationKind::Smell),
            make_business_value("c.rs"),
            make_other_kind("d.rs", AnnotationKind::Invariant),
        ];
        let paths = business_value_paths(&annos);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains("a.rs"));
        assert!(paths.contains("c.rs"));
        assert!(!paths.contains("b.rs"));
        assert!(!paths.contains("d.rs"));
    }

    #[test]
    fn business_value_paths_dedupes_repeated_paths() {
        // Two business-value annotations on the same file => one path entry.
        let mut a1 = make_business_value("hot.rs");
        a1.location.line_start = 10;
        let mut a2 = make_business_value("hot.rs");
        a2.location.line_start = 50;
        let paths = business_value_paths(&[a1, a2]);
        assert_eq!(paths.len(), 1);
        assert!(paths.contains("hot.rs"));
    }

    #[test]
    fn intersection_keeps_only_business_value_untested_files() {
        // The spec's worked example: 2 business-value files + 5 untested
        // files => the intersection should be exactly the 2 business-value
        // files that ALSO appear in the untested list.
        let bv_annos = vec![
            make_business_value("crates/foo/src/lib.rs"),
            make_business_value("crates/bar/src/main.rs"),
        ];
        let untested = vec![
            "crates/foo/src/lib.rs".to_string(),
            "crates/baz/src/lib.rs".to_string(),
            "crates/bar/src/main.rs".to_string(),
            "crates/quux/src/lib.rs".to_string(),
            "crates/zonk/src/lib.rs".to_string(),
        ];
        let bv_paths = business_value_paths(&bv_annos);
        let intersection = untested_high_value_annotations(&untested, &bv_paths, "now");
        assert_eq!(intersection.len(), 2);
        let paths: Vec<&str> = intersection
            .iter()
            .map(|a| a.location.path.as_str())
            .collect();
        assert!(paths.contains(&"crates/foo/src/lib.rs"));
        assert!(paths.contains(&"crates/bar/src/main.rs"));
    }

    #[test]
    fn intersection_with_path_separator_mismatch_still_matches() {
        // Sentrux on Windows might emit `crates\foo\src\lib.rs`; our
        // business-value annotation has `crates/foo/src/lib.rs`. The
        // intersection must normalize both sides.
        let bv = vec![make_business_value("crates/foo/src/lib.rs")];
        let bv_paths = business_value_paths(&bv);
        let untested = vec!["crates\\foo\\src\\lib.rs".to_string()];
        let intersection = untested_high_value_annotations(&untested, &bv_paths, "now");
        assert_eq!(intersection.len(), 1);
        assert_eq!(intersection[0].location.path, "crates/foo/src/lib.rs");
    }

    #[test]
    fn empty_business_value_yields_no_intersection() {
        // No business-value annotations => zero untested-smells, regardless
        // of how many untested files sentrux reports. This is the safety
        // gate that prevents the bridge from spamming 1,142 annotations
        // on a codebase with no business-value tags.
        let bv_paths = HashSet::new();
        let untested = vec!["a.rs".into(), "b.rs".into(), "c.rs".into()];
        let intersection = untested_high_value_annotations(&untested, &bv_paths, "now");
        assert!(intersection.is_empty());
    }

    #[test]
    fn empty_untested_yields_empty_intersection() {
        let bv_paths: HashSet<String> = ["a.rs".into(), "b.rs".into()].iter().cloned().collect();
        let intersection = untested_high_value_annotations(&[], &bv_paths, "now");
        assert!(intersection.is_empty());
    }
}
