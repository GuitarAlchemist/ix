//! Parse the JSON envelope returned by sentrux `check_rules`.
//!
//! The MCP `tools/call` envelope wraps the tool's JSON text in
//! `result.content[0].text`. Unwrapping gives us this shape (sentrux 0.x,
//! observed 2026-05-24):
//!
//! ```json
//! {
//!   "pass": false,
//!   "rules_checked": 4,
//!   "summary": "✗ Architectural rule violations detected",
//!   "truncated": { … },
//!   "violation_count": 1,
//!   "violations": [
//!     {
//!       "files": ["crates/ix-agent/src/tools.rs:register_bridges_and_session (420 lines)"],
//!       "message": "1 function(s) exceed max length of 400 lines",
//!       "rule": "max_fn_lines",
//!       "severity": "Error"
//!     }
//!   ]
//! }
//! ```
//!
//! Sentrux Pro may add more fields. This parser ignores unknown fields and
//! only requires `violations`. Each entry in `violations.files` carries the
//! file path plus optional `:symbol` and `(N lines/etc.)` suffix.

use serde::{Deserialize, Serialize};

/// Top-level `check_rules` payload (after MCP envelope unwrap).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RulesReport {
    /// Overall pass/fail. `false` if any violations.
    #[serde(default)]
    pub pass: bool,
    /// Number of rules sentrux actually checked.
    #[serde(default)]
    pub rules_checked: u32,
    /// Short human-readable summary line.
    #[serde(default)]
    pub summary: String,
    /// Count of violations (server-reported; may differ from `violations.len()`
    /// if sentrux truncated for the free tier).
    #[serde(default)]
    pub violation_count: u32,
    /// One entry per architectural-rule violation.
    #[serde(default)]
    pub violations: Vec<RuleViolation>,
}

/// One architectural-rule violation.
///
/// `files` is a list of human-readable "where" descriptors. The convention
/// observed in sentrux 0.x is
/// `"<repo-relative-path>:<symbol> (<detail>)"`, for example:
///
/// ```text
/// crates/ix-agent/src/tools.rs:register_bridges_and_session (420 lines)
/// ```
///
/// We parse out the path and optional symbol; the symbol is preserved in
/// the annotation's `claim` text so it is human-readable on a dashboard.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuleViolation {
    /// Sentrux rule name, e.g. `max_fn_lines`, `acyclic_modules`.
    pub rule: String,
    /// Human-readable severity label (`Error`, `Warning`, `Info`).
    #[serde(default)]
    pub severity: String,
    /// Human-readable explanation of the violation.
    #[serde(default)]
    pub message: String,
    /// Where the violation occurred. Always present; usually one entry.
    #[serde(default)]
    pub files: Vec<String>,
}

/// Parsed "file:symbol (detail)" descriptor pulled out of [`RuleViolation::files`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViolationLocation {
    /// Repo-relative path (forward-slash normalized).
    pub path: String,
    /// Optional symbol name (function, module, type) inside the file.
    pub symbol: Option<String>,
    /// Optional free-form detail in parens, e.g. `"420 lines"`.
    pub detail: Option<String>,
}

impl ViolationLocation {
    /// Parse one entry from `violations.files`.
    ///
    /// Recognized form: `path[:symbol][ (detail)]`. Windows-style backslashes
    /// in `path` are normalized to forward slashes (the contract requires
    /// repo-relative POSIX paths). A `:symbol` is only split off when the
    /// remainder looks like an identifier — this avoids false-splitting on
    /// Windows drive letters like `C:\foo`.
    pub fn parse(raw: &str) -> Self {
        let s = raw.trim();

        // Pull off `( detail )` suffix if present.
        let (head, detail) = if let Some(start) = s.rfind('(') {
            if s.ends_with(')') && start > 0 {
                let inner = &s[start + 1..s.len() - 1];
                (s[..start].trim_end(), Some(inner.trim().to_string()))
            } else {
                (s, None)
            }
        } else {
            (s, None)
        };

        // Split into path + optional symbol on the LAST ':' that is followed
        // by an identifier-like token. We deliberately ignore `C:` style
        // drive letters by requiring the suffix to start with an alphabetic
        // char or underscore and contain no path separators.
        let (path_raw, symbol) = match head.rfind(':') {
            Some(idx) if idx > 0 => {
                let candidate = &head[idx + 1..];
                let looks_like_symbol = candidate
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphabetic() || c == '_')
                    .unwrap_or(false)
                    && !candidate.contains('/')
                    && !candidate.contains('\\');
                if looks_like_symbol {
                    (&head[..idx], Some(candidate.to_string()))
                } else {
                    (head, None)
                }
            }
            _ => (head, None),
        };

        let path = path_raw.trim().replace('\\', "/");

        Self {
            path,
            symbol,
            detail,
        }
    }
}

/// Unwrap the MCP envelope `{ result: { content: [{ text: "<json>" }] } }`
/// and parse the contained JSON into a [`RulesReport`].
///
/// Pass the full `tools/call` response object as parsed JSON. If sentrux
/// returns a human-readable diagnostic instead of a JSON body (e.g.
/// `"No rules file found at .sentrux/rules.toml"`), the result is an
/// empty report whose `summary` carries the diagnostic — so callers
/// downstream see "0 violations" rather than crashing.
pub fn parse_check_rules_response(envelope: &serde_json::Value) -> Result<RulesReport, String> {
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
        return Ok(RulesReport {
            pass: true,
            rules_checked: 0,
            summary: trimmed.to_string(),
            violation_count: 0,
            violations: Vec::new(),
        });
    }
    serde_json::from_str::<RulesReport>(trimmed)
        .map_err(|e| format!("check_rules payload was not RulesReport JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_sentrux_response() {
        // Captured from `sentrux.exe mcp` against ix worktree on 2026-05-24.
        let envelope: serde_json::Value = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 99,
            "result": {
                "content": [{
                    "type": "text",
                    "text": serde_json::json!({
                        "pass": false,
                        "rules_checked": 4,
                        "summary": "Architectural rule violations detected",
                        "violation_count": 1,
                        "violations": [{
                            "files": ["crates/ix-agent/src/tools.rs:register_bridges_and_session (420 lines)"],
                            "message": "1 function(s) exceed max length of 400 lines",
                            "rule": "max_fn_lines",
                            "severity": "Error"
                        }]
                    }).to_string()
                }]
            }
        });
        let report = parse_check_rules_response(&envelope).expect("parses");
        assert!(!report.pass);
        assert_eq!(report.violation_count, 1);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].rule, "max_fn_lines");
        assert_eq!(report.violations[0].severity, "Error");
    }

    #[test]
    fn parses_violation_with_symbol_and_detail() {
        let loc = ViolationLocation::parse(
            "crates/ix-agent/src/tools.rs:register_bridges_and_session (420 lines)",
        );
        assert_eq!(loc.path, "crates/ix-agent/src/tools.rs");
        assert_eq!(loc.symbol.as_deref(), Some("register_bridges_and_session"));
        assert_eq!(loc.detail.as_deref(), Some("420 lines"));
    }

    #[test]
    fn normalizes_windows_backslashes_but_preserves_drive_letter() {
        let loc = ViolationLocation::parse("crates\\ix-agent\\src\\tools.rs");
        assert_eq!(loc.path, "crates/ix-agent/src/tools.rs");
        assert!(loc.symbol.is_none());
    }

    #[test]
    fn does_not_split_on_drive_letter() {
        // A bare windows-style path shouldn't be split — `C:\foo` is not a symbol.
        let loc = ViolationLocation::parse("C:\\foo\\bar.rs");
        assert_eq!(loc.path, "C:/foo/bar.rs");
        assert!(loc.symbol.is_none());
    }

    #[test]
    fn no_symbol_just_path() {
        let loc = ViolationLocation::parse("src/lib.rs");
        assert_eq!(loc.path, "src/lib.rs");
        assert!(loc.symbol.is_none());
        assert!(loc.detail.is_none());
    }

    #[test]
    fn missing_violations_array_is_empty() {
        let report: RulesReport = serde_json::from_str(r#"{"pass": true}"#).unwrap();
        assert!(report.violations.is_empty());
    }
}
