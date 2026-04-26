//! Code smell detection — Layer 2½.
//!
//! Provides two layers of detection:
//! - **Lexical** (always available): TODO/FIXME comments, magic numbers, long lines.
//! - **AST-based** (`semantic` feature): deep nesting, long functions, too many parameters.
//!
//! Language-specific smells are documented on [`detect_smells`].

use serde::{Deserialize, Serialize};

use crate::analyze::Language;

/// Severity of a detected code smell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Stylistic suggestion — does not affect correctness.
    Low,
    /// Maintainability warning — technical debt accumulator.
    Medium,
    /// Likely bug or safety concern — should be addressed promptly.
    High,
}

/// A single detected code smell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSmell {
    /// Short machine-readable name, e.g. `"magic_number"`, `"deep_nesting"`.
    pub name: String,
    /// 1-based source line where the smell was detected, if known.
    pub line: Option<usize>,
    /// Severity classification.
    pub severity: Severity,
    /// Human-readable explanation.
    pub message: String,
}

impl CodeSmell {
    fn new(
        name: &str,
        line: Option<usize>,
        severity: Severity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            line,
            severity,
            message: message.into(),
        }
    }
}

/// Detect code smells in `source` for the given `language`.
///
/// Always performs lexical detection. When compiled with the `semantic`
/// feature, also runs AST-based checks (deep nesting, long functions, etc.).
pub fn detect_smells(source: &str, language: Language) -> Vec<CodeSmell> {
    let mut smells = Vec::new();
    lexical_smells(source, language, &mut smells);
    #[cfg(feature = "semantic")]
    ast_smells(source, language, &mut smells);
    smells
}

// ---------------------------------------------------------------------------
// Lexical layer (always compiled)
// ---------------------------------------------------------------------------

fn lexical_smells(source: &str, language: Language, out: &mut Vec<CodeSmell>) {
    for (i, line) in source.lines().enumerate() {
        let lineno = i + 1;
        let trimmed = line.trim();

        // TODO / FIXME markers
        if trimmed.contains("TODO") || trimmed.contains("FIXME") {
            out.push(CodeSmell::new(
                "todo_comment",
                Some(lineno),
                Severity::Low,
                format!("Unresolved TODO/FIXME at line {lineno}"),
            ));
        }

        // Lines longer than 120 characters
        if line.len() > 120 {
            out.push(CodeSmell::new(
                "long_line",
                Some(lineno),
                Severity::Low,
                format!("Line {lineno} is {} chars (limit 120)", line.len()),
            ));
        }

        // Magic numbers (standalone numeric literals that aren't 0, 1, -1)
        if has_magic_number(trimmed, language) {
            out.push(CodeSmell::new(
                "magic_number",
                Some(lineno),
                Severity::Low,
                format!("Magic numeric literal at line {lineno} — consider a named constant"),
            ));
        }
    }

    // Language-specific lexical smells
    match language {
        Language::Rust => rust_lexical_smells(source, out),
        Language::TypeScript | Language::JavaScript => ts_lexical_smells(source, out),
        Language::FSharp => fsharp_lexical_smells(source, out),
        _ => {}
    }
}

fn rust_lexical_smells(source: &str, out: &mut Vec<CodeSmell>) {
    let unwrap_count = source.matches(".unwrap()").count();
    if unwrap_count > 3 {
        out.push(CodeSmell::new(
            "excessive_unwrap",
            None,
            Severity::Medium,
            format!("{unwrap_count} `.unwrap()` calls — prefer `?` or explicit error handling"),
        ));
    }
}

fn ts_lexical_smells(source: &str, out: &mut Vec<CodeSmell>) {
    for (i, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains("@ts-ignore") || trimmed.contains("@ts-expect-error") {
            out.push(CodeSmell::new(
                "ts_suppress",
                Some(i + 1),
                Severity::Medium,
                format!(
                    "TypeScript error suppression at line {} — fix the type error instead",
                    i + 1
                ),
            ));
        }
        if trimmed.contains(": any") || trimmed.contains("<any>") || trimmed.contains("as any") {
            out.push(CodeSmell::new(
                "any_type",
                Some(i + 1),
                Severity::Medium,
                format!(
                    "`any` type usage at line {} — use a specific type or `unknown`",
                    i + 1
                ),
            ));
        }
    }
}

fn fsharp_lexical_smells(source: &str, out: &mut Vec<CodeSmell>) {
    let mutable_count = source.matches("mutable ").count();
    if mutable_count > 5 {
        out.push(CodeSmell::new(
            "excessive_mutable",
            None,
            Severity::Medium,
            format!("{mutable_count} `mutable` bindings — prefer immutable values and recursion"),
        ));
    }
}

// ---------------------------------------------------------------------------
// AST layer (semantic feature only)
// ---------------------------------------------------------------------------

#[cfg(feature = "semantic")]
fn ast_smells(source: &str, language: Language, out: &mut Vec<CodeSmell>) {
    use crate::semantic::extract_semantic_metrics_for;
    let metrics = extract_semantic_metrics_for(source, language);

    // Deep nesting
    if metrics.nesting_depth_max >= 5 {
        out.push(CodeSmell::new(
            "deep_nesting",
            None,
            Severity::High,
            format!(
                "Maximum nesting depth is {} (threshold: 5) — extract methods to reduce complexity",
                metrics.nesting_depth_max
            ),
        ));
    } else if metrics.nesting_depth_max >= 4 {
        out.push(CodeSmell::new(
            "deep_nesting",
            None,
            Severity::Medium,
            format!(
                "Nesting depth {} approaching limit of 5",
                metrics.nesting_depth_max
            ),
        ));
    }

    // Low parse quality (garbled / invalid source)
    if metrics.parse_quality < 0.5 && metrics.ast_node_count > 0 {
        out.push(CodeSmell::new(
            "parse_errors",
            None,
            Severity::High,
            format!(
                "Parse quality {:.0}% — source contains significant syntax errors",
                metrics.parse_quality * 100.0
            ),
        ));
    }

    // Safety-concern count (unsafe_blocks, @ts-ignore, mutable, etc.)
    if metrics.unsafe_blocks >= 3 {
        let label = match language {
            Language::Rust => "unsafe blocks",
            Language::CSharp => "unsafe statements",
            Language::TypeScript | Language::JavaScript => "type-safety suppressions",
            Language::FSharp => "mutable bindings (AST)",
            _ => "safety concerns",
        };
        out.push(CodeSmell::new(
            "safety_concerns",
            None,
            Severity::Medium,
            format!(
                "{} {} detected — review carefully",
                metrics.unsafe_blocks, label
            ),
        ));
    }
}

/// Returns true if `line` contains a numeric literal that isn't 0, 1, or -1,
/// and is not part of a comment.
fn has_magic_number(line: &str, lang: Language) -> bool {
    // Skip comment lines
    let comment_prefix = match lang {
        Language::FSharp => "//",
        Language::Python => "#",
        _ => "//",
    };
    let trimmed = line.trim();
    if trimmed.starts_with(comment_prefix) || trimmed.starts_with('#') {
        return false;
    }
    // Very cheap heuristic: look for digit sequences not preceded by letter/underscore
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            // back-track: if preceded by identifier char, skip
            let preceded_by_ident =
                i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if !preceded_by_ident {
                // read full number
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                let num_str = &line[start..i];
                if !matches!(num_str, "0" | "1" | "0.0" | "1.0") {
                    return true;
                }
                continue;
            }
        }
        i += 1;
    }
    false
}
