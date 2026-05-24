//! Convert sentrux `RuleViolation`s into `ai-annotation-v1` records.
//!
//! Each violation becomes one [`ix_ai_annotations::Annotation`] with:
//! - `kind = Smell` (or `Invariant` for cycle-detection-style rules — see
//!   [`kind_for_rule`])
//! - `truth_value = F` (the rule is *violated*; the implicit claim "this
//!   rule holds here" is refuted)
//! - `certainty = DetectedBySentrux`
//! - `confidence = 1.0` (sentrux runs deterministic checks)
//! - `source.author = "sentrux"`, `source.evidence = "sentrux-check@<ts>"`
//!
//! Line number resolution: sentrux gives us a symbol (e.g. function name),
//! but no line. We do a cheap text-scan of the file to find the declaration
//! line. Falls back to line 1 if not found.

use crate::rules_response::{RuleViolation, ViolationLocation};
use crate::SENTRUX_AUTHOR;
use ix_ai_annotations::types::{
    annotation_id, Annotation, AnnotationKind, Certainty, Location, Source, TruthValue,
    SCHEMA_VERSION,
};
use std::fs;
use std::path::Path;

/// Map a sentrux rule name to an annotation `kind`. The default is `Smell`;
/// rules that capture structural invariants (cycles, layering) map to
/// `Invariant` because the implicit claim is stronger.
pub fn kind_for_rule(rule: &str) -> AnnotationKind {
    match rule {
        // Structural invariants — the rule asserts a property of the graph.
        "acyclic_modules" | "no_cycles" | "layering" | "no_upward_imports" => {
            AnnotationKind::Invariant
        }
        // Everything else is a smell (size limits, redundancy, complexity, …).
        _ => AnnotationKind::Smell,
    }
}

/// Convert one violation into one or more annotations (one per file entry).
///
/// `workspace` is the repo root (used only to resolve absolute paths so
/// the line-number scan works). `now` is an ISO-8601 timestamp shared
/// across the whole batch for stable evidence references.
pub fn violation_to_annotation(
    workspace: &Path,
    violation: &RuleViolation,
    now: &str,
) -> Vec<Annotation> {
    let mut out = Vec::with_capacity(violation.files.len().max(1));
    for raw in &violation.files {
        let loc = ViolationLocation::parse(raw);
        let line = resolve_line(workspace, &loc.path, loc.symbol.as_deref()).unwrap_or(1);
        out.push(build_annotation(violation, &loc, line, now));
    }
    out
}

fn build_annotation(
    violation: &RuleViolation,
    loc: &ViolationLocation,
    line: u32,
    now: &str,
) -> Annotation {
    let kind = kind_for_rule(&violation.rule);
    let claim = build_claim(violation, loc);
    let id = annotation_id(&loc.path, line, kind, &claim);
    Annotation {
        schema_version: SCHEMA_VERSION,
        id,
        kind,
        claim,
        truth_value: TruthValue::F,
        certainty: Certainty::DetectedBySentrux,
        confidence: 1.0,
        source: Source {
            author: SENTRUX_AUTHOR.to_string(),
            model: None,
            evidence: Some(format!("sentrux-check@{now}")),
        },
        location: Location {
            path: loc.path.clone(),
            line_start: line,
            line_end: line,
        },
        created_at: now.to_string(),
        updated_at: now.to_string(),
        stale: false,
        reconciliation: None,
    }
}

fn build_claim(violation: &RuleViolation, loc: &ViolationLocation) -> String {
    // Keep the claim under the 500-char contract limit.
    let symbol_part = loc
        .symbol
        .as_deref()
        .map(|s| format!(" `{s}`"))
        .unwrap_or_default();
    let detail_part = loc
        .detail
        .as_deref()
        .map(|d| format!(" ({d})"))
        .unwrap_or_default();
    let base = format!(
        "sentrux rule `{rule}` violated{symbol_part}{detail_part}: {msg}",
        rule = violation.rule,
        msg = violation.message,
    );
    if base.len() <= 500 {
        base
    } else {
        let mut s = base;
        s.truncate(497);
        s.push_str("...");
        s
    }
}

/// Best-effort line-number resolver. If sentrux gave us a symbol, scan the
/// file for the first line that looks like a declaration of that symbol.
/// Otherwise, return `None` and the caller falls back to line 1.
fn resolve_line(workspace: &Path, rel_path: &str, symbol: Option<&str>) -> Option<u32> {
    let symbol = symbol?;
    let full = workspace.join(rel_path);
    let text = fs::read_to_string(&full).ok()?;
    for (idx, line) in text.lines().enumerate() {
        if looks_like_decl(line, symbol) {
            return Some((idx + 1) as u32);
        }
    }
    None
}

/// Heuristic: does `line` contain a declaration of `symbol`?
///
/// We look for the symbol preceded by one of a handful of declaration
/// keywords. This isn't a full parser, but sentrux's rule names map
/// 1:1 to a small set of artifacts (functions, modules, types) and the
/// false-positive cost is low — a wrong line number doesn't break the
/// schema, only the in-IDE jump-to-source experience.
fn looks_like_decl(line: &str, symbol: &str) -> bool {
    if !line.contains(symbol) {
        return false;
    }
    // Anchor on common decl keywords across the languages sentrux supports.
    const KEYWORDS: &[&str] = &[
        "fn ", "pub fn ", "async fn ", "pub async fn ", "let ", "const ", "static ",
        "struct ", "enum ", "trait ", "impl ", "mod ", "pub mod ",
        "function ", "def ", "class ", "type ", "interface ",
    ];
    for kw in KEYWORDS {
        if let Some(pos) = line.find(kw) {
            let after = &line[pos + kw.len()..];
            if after.trim_start().starts_with(symbol) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_response::RuleViolation;
    use tempfile::tempdir;

    fn sample_violation() -> RuleViolation {
        RuleViolation {
            rule: "max_fn_lines".into(),
            severity: "Error".into(),
            message: "1 function(s) exceed max length of 400 lines".into(),
            files: vec!["src/foo.rs:my_huge_fn (420 lines)".into()],
        }
    }

    #[test]
    fn produces_false_truth_value_and_sentrux_author() {
        let dir = tempdir().unwrap();
        let annos = violation_to_annotation(dir.path(), &sample_violation(), "2026-05-24T18:00:00Z");
        assert_eq!(annos.len(), 1);
        let a = &annos[0];
        assert_eq!(a.truth_value, TruthValue::F);
        assert_eq!(a.certainty, Certainty::DetectedBySentrux);
        assert_eq!(a.confidence, 1.0);
        assert_eq!(a.source.author, "sentrux");
        assert_eq!(
            a.source.evidence.as_deref(),
            Some("sentrux-check@2026-05-24T18:00:00Z")
        );
    }

    #[test]
    fn id_is_stable_for_same_inputs() {
        let dir = tempdir().unwrap();
        let v = sample_violation();
        let a = &violation_to_annotation(dir.path(), &v, "2026-05-24T18:00:00Z")[0];
        let b = &violation_to_annotation(dir.path(), &v, "2026-05-24T19:00:00Z")[0];
        // Same path+line+kind+claim => same id; timestamp doesn't change it.
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn claim_mentions_rule_and_symbol() {
        let dir = tempdir().unwrap();
        let a = &violation_to_annotation(dir.path(), &sample_violation(), "now")[0];
        assert!(a.claim.contains("max_fn_lines"));
        assert!(a.claim.contains("my_huge_fn"));
        assert!(a.claim.contains("420 lines"));
    }

    #[test]
    fn resolves_line_when_decl_present() {
        let dir = tempdir().unwrap();
        let src = "fn other() {}\n\nfn my_huge_fn() {\n    // long body\n}\n";
        std::fs::write(dir.path().join("src.rs"), src).unwrap();
        // Use a violation referencing the file we just wrote.
        let v = RuleViolation {
            rule: "max_fn_lines".into(),
            severity: "Error".into(),
            message: "too long".into(),
            files: vec!["src.rs:my_huge_fn (10 lines)".into()],
        };
        let a = &violation_to_annotation(dir.path(), &v, "now")[0];
        assert_eq!(a.location.line_start, 3);
    }

    #[test]
    fn falls_back_to_line_1_when_symbol_not_found() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("src.rs"), "// nothing here").unwrap();
        let v = RuleViolation {
            rule: "max_fn_lines".into(),
            severity: "Error".into(),
            message: "too long".into(),
            files: vec!["src.rs:not_in_file (10 lines)".into()],
        };
        let a = &violation_to_annotation(dir.path(), &v, "now")[0];
        assert_eq!(a.location.line_start, 1);
    }

    #[test]
    fn cycle_rule_becomes_invariant_kind() {
        assert_eq!(kind_for_rule("acyclic_modules"), AnnotationKind::Invariant);
        assert_eq!(kind_for_rule("max_fn_lines"), AnnotationKind::Smell);
        assert_eq!(kind_for_rule("unknown_rule"), AnnotationKind::Smell);
    }

    #[test]
    fn multiple_files_yield_multiple_annotations() {
        let dir = tempdir().unwrap();
        let v = RuleViolation {
            rule: "redundancy".into(),
            severity: "Warning".into(),
            message: "duplicate imports".into(),
            files: vec!["a.rs".into(), "b.rs".into(), "c.rs".into()],
        };
        let annos = violation_to_annotation(dir.path(), &v, "now");
        assert_eq!(annos.len(), 3);
    }
}
