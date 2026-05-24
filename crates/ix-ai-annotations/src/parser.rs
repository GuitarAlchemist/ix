//! Parser for `@ai:` markers inside source comments.
//!
//! Recognizes the form documented in
//! `docs/contracts/2026-05-24-ai-annotation.contract.md`:
//!
//! ```text
//! <comment-marker> @ai:<kind> <claim> [<T>:<certainty> conf:<confidence> src:<evidence>]
//! ```
//!
//! Supported comment markers (single-line and single-line block):
//! - `//`  (C-family, Rust, JS/TS, C#, F#, Java, Go, Swift)
//! - `#`   (Python, Ruby, Shell, PowerShell, YAML, TOML)
//! - `--`  (Lua, SQL, Haskell)
//! - `/* ... */`   (single-line block)
//! - `<!-- ... -->` (single-line HTML/XML/MD block)

use crate::types::{AnnotationKind, Certainty, TruthValue};
use regex::Regex;
use std::sync::OnceLock;

/// One parsed `@ai:` marker — language-agnostic intermediate form.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedMarker {
    pub kind: AnnotationKind,
    pub claim: String,
    pub truth_value: TruthValue,
    pub certainty: Certainty,
    pub confidence: Option<f64>,
    pub evidence: Option<String>,
}

// The single regex that captures everything after we've stripped comment markers.
// We accept a payload like:
//
//     @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]
//
// Capture groups:
// 1. kind            (`invariant`)
// 2. claim           (`arr is sorted ascending`)
// 3. truth_value     (`T`)
// 4. certainty       (`test`)
// 5. confidence      (`0.95`, optional)
// 6. evidence        (`test_search.rs:42`, optional)
fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            @ai:(?P<kind>[a-zA-Z_-]+)
            \s+
            (?P<claim>.+?)
            \s+
            \[
              (?P<tv>[TPUDFC])
              :
              (?P<cert>[a-z][a-z\-]*)
              (?:\s+conf:(?P<conf>[0-9]*\.?[0-9]+))?
              (?:\s+src:(?P<src>[^\]]+?))?
            \]
            ",
        )
        .expect("static regex compiles")
    })
}

/// Strip the leading comment marker from a line (returning the text after the
/// marker), or `None` if the line has no comment marker we recognize.
fn strip_comment_marker(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    // Block comments: /* ... */ — accept only single-line form.
    if let Some(rest) = trimmed.strip_prefix("/*") {
        let rest = rest.trim_start();
        if let Some(closed) = rest.rsplit_once("*/") {
            return Some(closed.0);
        }
        // No closing marker on the same line: still try to parse the rest.
        return Some(rest);
    }
    // HTML comment: <!-- ... -->
    if let Some(rest) = trimmed.strip_prefix("<!--") {
        let rest = rest.trim_start();
        if let Some(closed) = rest.rsplit_once("-->") {
            return Some(closed.0);
        }
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("///") {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("//!") {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("//") {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("#!") {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix('#') {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("--") {
        return Some(rest);
    }
    None
}

/// Parse one source line. Returns `Some` if the line contains a well-formed
/// `@ai:` marker.
pub fn parse_line(line: &str) -> Option<ParsedMarker> {
    let payload = strip_comment_marker(line)?;
    // Quick reject: must contain "@ai:" before regex.
    if !payload.contains("@ai:") {
        return None;
    }
    let caps = marker_re().captures(payload)?;
    let kind = AnnotationKind::parse(caps.name("kind")?.as_str())?;
    let claim = caps.name("claim")?.as_str().trim().to_string();
    let tv_char = caps.name("tv")?.as_str().chars().next()?;
    let truth_value = TruthValue::from_char(tv_char)?;
    let certainty = Certainty::parse(caps.name("cert")?.as_str())?;
    let confidence = caps
        .name("conf")
        .and_then(|m| m.as_str().parse::<f64>().ok());
    let evidence = caps.name("src").map(|m| m.as_str().trim().to_string());

    Some(ParsedMarker {
        kind,
        claim,
        truth_value,
        certainty,
        confidence,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_double_slash_full_marker() {
        let line = "    // @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Invariant);
        assert_eq!(m.claim, "arr is sorted ascending");
        assert_eq!(m.truth_value, TruthValue::T);
        assert_eq!(m.certainty, Certainty::Test);
        assert_eq!(m.confidence, Some(0.95));
        assert_eq!(m.evidence.as_deref(), Some("test_search.rs:42"));
    }

    #[test]
    fn python_hash_assumption_no_evidence() {
        let line = "# @ai:assumption caller holds the lock [P:assumed conf:0.7]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Assumption);
        assert_eq!(m.truth_value, TruthValue::P);
        assert_eq!(m.certainty, Certainty::Assumed);
        assert_eq!(m.confidence, Some(0.7));
        assert!(m.evidence.is_none());
    }

    #[test]
    fn lua_double_dash() {
        let line = "-- @ai:hypothesis race-free under MIRI [U:uncertain conf:0.4]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Hypothesis);
        assert_eq!(m.truth_value, TruthValue::U);
        assert_eq!(m.certainty, Certainty::Uncertain);
    }

    #[test]
    fn rust_block_comment_single_line() {
        let line = "/* @ai:smell deep nesting [D:inferred conf:0.6] */";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Smell);
        assert_eq!(m.truth_value, TruthValue::D);
    }

    #[test]
    fn html_comment() {
        let line = "<!-- @ai:hint prefer KaTeX over MathJax [U:uncertain] -->";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Hint);
        assert!(m.confidence.is_none());
    }

    #[test]
    fn rust_triple_slash_doc_comment() {
        let line = "/// @ai:contract returns non-empty iff input is non-empty [T:manually-reviewed conf:0.85 src:PR#313]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.kind, AnnotationKind::Contract);
        assert_eq!(m.certainty, Certainty::ManuallyReviewed);
        assert_eq!(m.evidence.as_deref(), Some("PR#313"));
    }

    #[test]
    fn missing_marker_yields_none() {
        assert!(parse_line("// just a normal comment").is_none());
        assert!(parse_line("let x = 5;").is_none());
        assert!(parse_line("# also normal").is_none());
    }

    #[test]
    fn malformed_bracket_rejected() {
        // missing certainty
        assert!(parse_line("// @ai:invariant something [T:]").is_none());
        // wrong truth value letter
        assert!(parse_line("// @ai:invariant something [X:test]").is_none());
    }

    #[test]
    fn refuted_false_value() {
        let line = "// @ai:hypothesis no allocator in hot path [F:test conf:0.92 src:bench.rs:18]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.truth_value, TruthValue::F);
    }

    #[test]
    fn contradictory_value() {
        let line = "// @ai:invariant len > 0 [C:test conf:0.5]";
        let m = parse_line(line).expect("parses");
        assert_eq!(m.truth_value, TruthValue::C);
    }
}
