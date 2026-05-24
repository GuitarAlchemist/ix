//! Emit modes: sidecar JSONL (preferred default) or inline source patch.
//!
//! `Sidecar` is non-invasive — it writes one JSONL line per annotation to
//! `state/quality/ai-annotations-sentrux.jsonl`. The reconciler picks
//! these up alongside annotations from any other author.
//!
//! `Inline` patches the violating source file, inserting an `@ai:smell`
//! comment immediately above the affected line. It skips files that
//! already carry a sentrux annotation at the target location, so reruns
//! are idempotent.

use crate::Error;
use ix_ai_annotations::types::Annotation;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Result of an emit run. Both numbers are useful for the CLI summary.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmitOutcome {
    /// Annotations that were newly written (sidecar lines or inline patches).
    pub written: usize,
    /// Annotations skipped because they were already present (inline mode
    /// only; sidecar always overwrites).
    pub skipped: usize,
}

/// Write every annotation as a JSONL line under `state/quality/ai-annotations-sentrux.jsonl`.
///
/// The file is **overwritten** on each run — sentrux is a live verifier,
/// so the JSONL always represents the current set of violations.
pub fn emit_sidecar(
    workspace: &Path,
    annotations: &[Annotation],
    out_override: Option<&Path>,
) -> Result<EmitOutcome, Error> {
    let out_path = match out_override {
        Some(p) => p.to_path_buf(),
        None => workspace.join(crate::DEFAULT_SIDECAR_PATH),
    };
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&out_path)?;
    let mut written = 0usize;
    for a in annotations {
        serde_json::to_writer(&mut f, a)?;
        f.write_all(b"\n")?;
        written += 1;
    }
    f.flush()?;
    Ok(EmitOutcome {
        written,
        skipped: 0,
    })
}

/// Patch each violating source file with an `// @ai:smell` comment above
/// the violation line. Idempotent: if the line above the target already
/// contains a `@ai:smell` annotation, the patch is skipped.
///
/// When `dry_run` is true, no files are mutated; counts still report what
/// **would** be written.
pub fn emit_inline(
    workspace: &Path,
    annotations: &[Annotation],
    dry_run: bool,
) -> Result<EmitOutcome, Error> {
    let mut written = 0usize;
    let mut skipped = 0usize;
    for a in annotations {
        let full_path = workspace.join(&a.location.path);
        let source = match fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(_) => {
                // Source missing or non-UTF-8; skip silently.
                skipped += 1;
                continue;
            }
        };
        let comment = comment_for_path(&a.location.path);
        let marker_line = format_marker_line(comment, a);

        // Find a per-file unique sentinel so two reruns don't stack
        // identical markers. We key on the claim text — the claim already
        // embeds the sentrux rule name and (where available) the symbol,
        // so it uniquely identifies the violation regardless of how the
        // line numbers shift after an earlier patch.
        let sentinel = format!(
            "@ai:{kind} {claim} ",
            kind = kind_serialize(a),
            claim = a.claim
        );
        if source.contains(&sentinel) {
            skipped += 1;
            continue;
        }

        let lines: Vec<&str> = source.lines().collect();
        let target_idx = a.location.line_start.saturating_sub(1) as usize;
        if target_idx >= lines.len() {
            skipped += 1;
            continue;
        }

        if dry_run {
            written += 1;
            continue;
        }

        // Insert the marker line before `target_idx`. Preserve original
        // line endings as best we can by re-joining with \n (sentrux'
        // primary targets are Rust/C#/TS sources — LF on Windows and Unix
        // alike per .gitattributes in the GA ecosystem).
        let mut new_lines: Vec<String> = Vec::with_capacity(lines.len() + 1);
        for (i, line) in lines.iter().enumerate() {
            if i == target_idx {
                new_lines.push(marker_line.clone());
            }
            new_lines.push((*line).to_string());
        }
        let trailing = if source.ends_with('\n') { "\n" } else { "" };
        let joined = format!("{}{trailing}", new_lines.join("\n"));
        fs::write(&full_path, joined)?;
        written += 1;
    }
    Ok(EmitOutcome { written, skipped })
}

fn comment_for_path(path: &str) -> &'static str {
    let p = path.to_ascii_lowercase();
    if p.ends_with(".py")
        || p.ends_with(".rb")
        || p.ends_with(".sh")
        || p.ends_with(".ps1")
        || p.ends_with(".yml")
        || p.ends_with(".yaml")
        || p.ends_with(".toml")
    {
        "#"
    } else if p.ends_with(".lua") || p.ends_with(".sql") || p.ends_with(".hs") {
        "--"
    } else if p.ends_with(".html") || p.ends_with(".htm") || p.ends_with(".xml") || p.ends_with(".md") {
        "<!--"
    } else {
        "//"
    }
}

fn format_marker_line(comment: &'static str, a: &Annotation) -> String {
    let tail = match a.source.evidence.as_deref() {
        Some(e) => format!(" src:{e}"),
        None => String::new(),
    };
    let body = format!(
        "@ai:{kind} {claim} [F:detected-by-sentrux conf:{conf}{tail}]",
        kind = kind_serialize(a),
        claim = a.claim,
        conf = format_conf(a.confidence),
    );
    if comment == "<!--" {
        format!("<!-- {body} -->")
    } else {
        format!("{comment} {body}")
    }
}

fn format_conf(c: f64) -> String {
    // Trim trailing zeros / "1" stays as "1.0".
    if (c - c.round()).abs() < f64::EPSILON {
        format!("{:.1}", c)
    } else {
        format!("{}", c)
    }
}

fn kind_serialize(a: &Annotation) -> String {
    // The serde rename_all = "kebab-case" Round-trips to lowercase variant
    // names for our small enum; the simplest portable path is to serialize
    // the value to JSON and strip quotes.
    match serde_json::to_value(a.kind) {
        Ok(serde_json::Value::String(s)) => s,
        _ => "smell".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_ai_annotations::types::{
        annotation_id, Annotation, AnnotationKind, Certainty, Location, Source, TruthValue,
        SCHEMA_VERSION,
    };
    use tempfile::tempdir;

    fn make(path: &str, line: u32, claim: &str) -> Annotation {
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, line, AnnotationKind::Smell, claim),
            kind: AnnotationKind::Smell,
            claim: claim.to_string(),
            truth_value: TruthValue::F,
            certainty: Certainty::DetectedBySentrux,
            confidence: 1.0,
            source: Source {
                author: "sentrux".into(),
                model: None,
                evidence: Some("sentrux-check@now".into()),
            },
            location: Location {
                path: path.into(),
                line_start: line,
                line_end: line,
            },
            created_at: "2026-05-24T18:00:00Z".into(),
            updated_at: "2026-05-24T18:00:00Z".into(),
            stale: false,
            reconciliation: None,
        }
    }

    #[test]
    fn sidecar_writes_jsonl_one_per_line() {
        let dir = tempdir().unwrap();
        let annos = vec![
            make("a.rs", 10, "claim a"),
            make("b.rs", 20, "claim b"),
        ];
        let outcome = emit_sidecar(dir.path(), &annos, None).unwrap();
        assert_eq!(outcome.written, 2);
        let sidecar = dir.path().join(crate::DEFAULT_SIDECAR_PATH);
        let body = fs::read_to_string(sidecar).unwrap();
        assert_eq!(body.lines().count(), 2);
        for line in body.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["source"]["author"], "sentrux");
            assert_eq!(v["truth_value"], "F");
            assert_eq!(v["certainty"], "detected-by-sentrux");
        }
    }

    #[test]
    fn sidecar_overwrites_on_rerun() {
        let dir = tempdir().unwrap();
        emit_sidecar(dir.path(), &[make("a.rs", 1, "x")], None).unwrap();
        emit_sidecar(dir.path(), &[make("b.rs", 2, "y"), make("c.rs", 3, "z")], None).unwrap();
        let body = fs::read_to_string(dir.path().join(crate::DEFAULT_SIDECAR_PATH)).unwrap();
        assert_eq!(body.lines().count(), 2);
        assert!(body.contains("b.rs"));
        assert!(!body.contains("a.rs"), "rerun must overwrite, not append");
    }

    #[test]
    fn inline_inserts_marker_above_violating_line() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        let src = "fn ok() {}\nfn bad() {\n    todo!()\n}\n";
        std::fs::write(dir.path().join("src/foo.rs"), src).unwrap();
        let a = make("src/foo.rs", 2, "sentrux rule `max_fn_lines` violated");
        let outcome = emit_inline(dir.path(), &[a], false).unwrap();
        assert_eq!(outcome.written, 1);
        let body = std::fs::read_to_string(dir.path().join("src/foo.rs")).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        // Marker should be on line 2, original "fn bad() {" pushed to line 3.
        assert!(lines[1].contains("@ai:smell"), "got line2={:?}", lines[1]);
        assert!(lines[1].contains("[F:detected-by-sentrux"));
        assert!(lines[2].starts_with("fn bad()"));
    }

    #[test]
    fn inline_is_idempotent() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/foo.rs"),
            "fn ok() {}\nfn bad() {\n    todo!()\n}\n",
        )
        .unwrap();
        let a = make("src/foo.rs", 2, "sentrux rule `max_fn_lines` violated");
        emit_inline(dir.path(), &[a.clone()], false).unwrap();
        let after_first = std::fs::read_to_string(dir.path().join("src/foo.rs")).unwrap();
        let second = emit_inline(dir.path(), &[a], false).unwrap();
        assert_eq!(second.written, 0);
        assert_eq!(second.skipped, 1);
        let after_second = std::fs::read_to_string(dir.path().join("src/foo.rs")).unwrap();
        assert_eq!(after_first, after_second, "second run must not patch again");
    }

    #[test]
    fn inline_dry_run_does_not_touch_files() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        let original = "fn bad() {}\n";
        std::fs::write(dir.path().join("src/foo.rs"), original).unwrap();
        let a = make("src/foo.rs", 1, "sentrux rule `x` violated");
        let outcome = emit_inline(dir.path(), &[a], true).unwrap();
        assert_eq!(outcome.written, 1);
        let after = std::fs::read_to_string(dir.path().join("src/foo.rs")).unwrap();
        assert_eq!(after, original);
    }

    #[test]
    fn inline_picks_correct_comment_marker_for_python() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("foo.py"), "def bad():\n    pass\n").unwrap();
        let a = make("foo.py", 1, "python smell");
        emit_inline(dir.path(), &[a], false).unwrap();
        let body = std::fs::read_to_string(dir.path().join("foo.py")).unwrap();
        let first = body.lines().next().unwrap();
        assert!(first.starts_with("# @ai:smell"), "got {first:?}");
    }
}
