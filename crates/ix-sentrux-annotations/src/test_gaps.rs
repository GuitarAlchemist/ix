//! Parse the JSON envelope returned by sentrux `test_gaps`.
//!
//! The MCP `tools/call` envelope wraps the tool's JSON text in
//! `result.content[0].text`. Unwrapping gives two distinct shapes:
//!
//! **Free tier (aggregate only)** — observed 2026-05-24 on sentrux 0.x:
//!
//! ```json
//! {
//!   "coverage_ratio": 264,
//!   "coverage_score": 0.0264,
//!   "source_files": 1173,
//!   "test_files": 59,
//!   "tested": 31,
//!   "untested": 1142
//! }
//! ```
//!
//! **Pro tier (with per-file ranking)** — projected shape (sentrux Pro adds
//! a `top_untested` array of repo-relative paths, ordered by risk score). We
//! also accept an alternate field name `untested_files` for forward-compat:
//!
//! ```json
//! {
//!   "coverage_score": 0.0264,
//!   "source_files": 1173,
//!   "tested": 31,
//!   "untested": 1142,
//!   "top_untested": [
//!     "crates/foo/src/lib.rs",
//!     "crates/bar/src/main.rs"
//!   ]
//! }
//! ```
//!
//! Either field name maps to [`TestGapsReport::untested_files`]. When neither
//! is present, the report carries `untested_files = []` and the bridge emits
//! zero per-file annotations (aggregate only — free tier).

use serde::{Deserialize, Serialize};

/// Top-level `test_gaps` payload (after MCP envelope unwrap).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TestGapsReport {
    /// Total source files scanned.
    #[serde(default)]
    pub source_files: u32,
    /// Total test files detected.
    #[serde(default)]
    pub test_files: u32,
    /// Source files with at least one test reference.
    #[serde(default)]
    pub tested: u32,
    /// Source files with zero detected test references.
    #[serde(default)]
    pub untested: u32,
    /// Coverage ratio (integer score sentrux exposes).
    #[serde(default)]
    pub coverage_ratio: u32,
    /// Coverage score, 0.0-1.0.
    #[serde(default)]
    pub coverage_score: f64,
    /// Per-file untested ranking. Pro tier only; empty on free tier.
    ///
    /// Accepts either `top_untested` (sentrux Pro canonical name) or
    /// `untested_files` (forward-compat alias) at parse time. Always
    /// serialized as `top_untested`.
    #[serde(default, alias = "untested_files", rename = "top_untested")]
    pub untested_files: Vec<String>,
}

impl TestGapsReport {
    /// True if the Pro-tier per-file array is present and non-empty.
    pub fn has_per_file(&self) -> bool {
        !self.untested_files.is_empty()
    }
}

/// Unwrap the MCP envelope `{ result: { content: [{ text: "<json>" }] } }`
/// and parse the contained JSON into a [`TestGapsReport`].
///
/// Pass the full `tools/call` response object as parsed JSON. If sentrux
/// returns a human-readable diagnostic instead of a JSON body (e.g.
/// `"No scan data. Call 'scan' first."`), the result is an empty report
/// — callers downstream see "0 untested files" rather than crashing.
pub fn parse_test_gaps_response(envelope: &serde_json::Value) -> Result<TestGapsReport, String> {
    let text = envelope
        .get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("text"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| "MCP envelope missing result.content[0].text".to_string())?;

    let trimmed = text.trim();
    if !trimmed.starts_with('{') {
        // Sentrux emitted a plaintext diagnostic. Treat as empty report.
        return Ok(TestGapsReport::default());
    }
    serde_json::from_str::<TestGapsReport>(trimmed)
        .map_err(|e| format!("test_gaps payload was not TestGapsReport JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(text_payload: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 99,
            "result": {
                "content": [{
                    "type": "text",
                    "text": text_payload.to_string()
                }]
            }
        })
    }

    #[test]
    fn parses_free_tier_aggregate_only() {
        // Captured from `sentrux.exe mcp` against ix worktree on 2026-05-24
        // (free tier — no top_untested array).
        let env = envelope(serde_json::json!({
            "coverage_ratio": 264,
            "coverage_score": 0.026427962489343565,
            "source_files": 1173,
            "test_files": 59,
            "tested": 31,
            "untested": 1142
        }));
        let report = parse_test_gaps_response(&env).expect("parses");
        assert_eq!(report.source_files, 1173);
        assert_eq!(report.tested, 31);
        assert_eq!(report.untested, 1142);
        assert!((report.coverage_score - 0.0264).abs() < 1e-3);
        assert!(!report.has_per_file());
    }

    #[test]
    fn parses_pro_tier_with_top_untested() {
        let env = envelope(serde_json::json!({
            "coverage_score": 0.07,
            "source_files": 100,
            "tested": 7,
            "untested": 93,
            "top_untested": [
                "crates/foo/src/lib.rs",
                "crates/bar/src/main.rs"
            ]
        }));
        let report = parse_test_gaps_response(&env).expect("parses");
        assert!(report.has_per_file());
        assert_eq!(report.untested_files.len(), 2);
        assert_eq!(report.untested_files[0], "crates/foo/src/lib.rs");
    }

    #[test]
    fn accepts_alias_untested_files() {
        // Forward-compat: a future sentrux release might rename the field.
        let env = envelope(serde_json::json!({
            "untested": 2,
            "untested_files": ["a.rs", "b.rs"]
        }));
        let report = parse_test_gaps_response(&env).expect("parses");
        assert_eq!(report.untested_files, vec!["a.rs", "b.rs"]);
    }

    #[test]
    fn plaintext_diagnostic_yields_empty_report() {
        // Sentrux returns plaintext (not JSON) when scan hasn't run yet.
        // We build the envelope by hand because the helper above
        // JSON-stringifies its argument — we want a literal string here.
        let env = serde_json::json!({
            "result": {
                "content": [{
                    "type": "text",
                    "text": "No scan data. Call 'scan' first."
                }]
            }
        });
        let report = parse_test_gaps_response(&env).expect("parses");
        assert_eq!(report.source_files, 0);
        assert_eq!(report.untested, 0);
        assert!(!report.has_per_file());
    }

    #[test]
    fn missing_envelope_text_is_error() {
        let env = serde_json::json!({"result": {"content": []}});
        let err = parse_test_gaps_response(&env).expect_err("must fail");
        assert!(err.contains("missing"), "got: {err}");
    }

    #[test]
    fn malformed_json_payload_is_error() {
        let env = serde_json::json!({
            "result": {"content": [{"text": "{not json"}]}
        });
        let err = parse_test_gaps_response(&env).expect_err("must fail");
        assert!(err.contains("TestGapsReport"), "got: {err}");
    }
}
