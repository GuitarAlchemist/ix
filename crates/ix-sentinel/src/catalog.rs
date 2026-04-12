//! Remediation catalog — maps observation patterns to fix commands.
//!
//! The catalog is the Sentinel's "knowledge base" of how to fix
//! things. Each entry says: "when you see an observation whose
//! claim_key contains this pattern, run this command."
//!
//! The catalog is loaded from a TOML file or falls back to a
//! built-in default set. The built-in set covers the adapter
//! types we've already built (clippy, cargo, submodule).
//!
//! Level 3 (the Remediation Amendment Protocol) will eventually
//! let the Sentinel propose new entries. For now, the catalog is
//! static and human-authored.

use std::path::{Path, PathBuf};

/// One entry in the remediation catalog.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CatalogEntry {
    /// Substring to match against `claim_key`. If a claim_key
    /// contains this pattern, this entry is a candidate fix.
    pub pattern: String,
    /// Shell command to run in the repo root. The Sentinel
    /// executes this via `bash -c`.
    pub command: String,
    /// Human-readable description for audit.
    pub description: String,
    /// Confidence level: P (probable fix, needs human review)
    /// or T (proven fix, auto-merge eligible in autonomous mode).
    pub confidence: char,
}

/// Load the catalog from a TOML file, or fall back to the
/// built-in defaults.
pub fn load(path: &Option<PathBuf>, _repo_root: &Path) -> Vec<CatalogEntry> {
    if let Some(p) = path {
        if p.exists() {
            // Future: parse TOML catalog. For now, fall through.
            eprintln!(
                "[sentinel] catalog file {} exists but TOML parsing not yet implemented, using defaults",
                p.display()
            );
        }
    }
    default_catalog()
}

/// Built-in remediation catalog covering the shipped adapters.
///
/// Each entry maps a claim_key pattern to a fix command. The
/// patterns are intentionally broad — they match any observation
/// from a given adapter type, not specific lints. More granular
/// entries can be added via the TOML catalog.
fn default_catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            pattern: "clippy:".to_string(),
            command: "cargo clippy --fix --workspace --tests --allow-dirty --allow-staged"
                .to_string(),
            description: "Auto-fix clippy warnings via cargo clippy --fix".to_string(),
            confidence: 'P',
        },
        CatalogEntry {
            pattern: "submodule:".to_string(),
            command: "git submodule update --remote".to_string(),
            description: "Update all submodules to their remote HEAD".to_string(),
            confidence: 'P',
        },
        // cargo test failures can't be auto-fixed — but we CAN
        // re-run them to confirm the failure is real.
        CatalogEntry {
            pattern: "cargo_suite::".to_string(),
            command: "cargo test --workspace --no-fail-fast".to_string(),
            description: "Re-run tests to confirm failure (not a fix, a verification)"
                .to_string(),
            confidence: 'P',
        },
    ]
}
