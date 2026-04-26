#![warn(missing_docs)]
//! Buf-style breaking-change detector for `capability-registry.json`.
//!
//! R3 of the IX roadmap per `examples/canonical-showcase/ix-roadmap-plan-v1.md`
//! §4.3. Compares two versions of the governance capability registry
//! and classifies each difference as compatible or breaking.
//!
//! # Rules
//!
//! Applied bottom-up on the `{repo → { tool_category → [tool_name] }}` tree:
//!
//! - Removing a repo entry → **breaking**
//! - Removing a tool category from a repo → **breaking**
//! - Removing a tool name from a category → **breaking**
//! - Adding repos, categories, or tools → **compatible** (additive)
//! - Changing the root-level `version` field → **informational**
//!
//! Renames surface as a removal of the old name plus an addition of
//! the new name; because we only classify the removal, that counts as
//! a breaking change. Callers who deliberately rename must either
//! keep the old alias or use the `--allowlist` flag on the binary.
//!
//! # Exit codes (binary)
//!
//! - `0` — no breaking changes (or all in allowlist)
//! - `1` — one or more breaking changes found
//! - `2` — usage or I/O error

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Parsed capability registry. Deliberately loose — we only extract
/// the fields the checker cares about.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Registry {
    /// Semver version of the registry schema.
    #[serde(default)]
    pub version: Option<String>,
    /// Map of repo name → repo description.
    #[serde(default)]
    pub repos: BTreeMap<String, RegistryRepo>,
}

/// A single repo entry in the registry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryRepo {
    /// Name of the MCP server this repo exposes.
    #[serde(default)]
    pub server: Option<String>,
    /// One-line description of the repo.
    #[serde(default)]
    pub description: Option<String>,
    /// Domain tags (math, signal, governance, ...).
    #[serde(default)]
    pub domains: Vec<String>,
    /// Tools grouped by category.
    #[serde(default)]
    pub tools: BTreeMap<String, Vec<String>>,
}

/// A classified difference between old and new registries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// Severity of the difference.
    pub severity: Severity,
    /// Dotted path identifying what changed: `repo.category.tool`.
    pub path: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Severity tiers the checker can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Schema change that would break downstream consumers.
    Breaking,
    /// Non-breaking change (additive).
    Compatible,
    /// Metadata-only change (version bump, description edit).
    Informational,
}

/// Compare two registries and return every difference, classified.
pub fn compare(old: &Registry, new: &Registry) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Version bump → informational only
    if old.version != new.version {
        findings.push(Finding {
            severity: Severity::Informational,
            path: "$.version".into(),
            message: format!(
                "version changed from {:?} to {:?}",
                old.version, new.version
            ),
        });
    }

    // Compare repos
    let old_repos: BTreeSet<&String> = old.repos.keys().collect();
    let new_repos: BTreeSet<&String> = new.repos.keys().collect();

    // Repos removed → breaking
    for repo in old_repos.difference(&new_repos) {
        findings.push(Finding {
            severity: Severity::Breaking,
            path: format!("repos.{repo}"),
            message: format!("repo '{repo}' was removed from the registry"),
        });
    }
    // Repos added → compatible
    for repo in new_repos.difference(&old_repos) {
        findings.push(Finding {
            severity: Severity::Compatible,
            path: format!("repos.{repo}"),
            message: format!("new repo '{repo}' added"),
        });
    }

    // Compare tools inside each shared repo
    for repo in old_repos.intersection(&new_repos) {
        let old_repo = &old.repos[*repo];
        let new_repo = &new.repos[*repo];
        let old_cats: BTreeSet<&String> = old_repo.tools.keys().collect();
        let new_cats: BTreeSet<&String> = new_repo.tools.keys().collect();

        for cat in old_cats.difference(&new_cats) {
            findings.push(Finding {
                severity: Severity::Breaking,
                path: format!("repos.{repo}.tools.{cat}"),
                message: format!("category '{cat}' was removed from repo '{repo}'"),
            });
        }
        for cat in new_cats.difference(&old_cats) {
            findings.push(Finding {
                severity: Severity::Compatible,
                path: format!("repos.{repo}.tools.{cat}"),
                message: format!("new category '{cat}' added to repo '{repo}'"),
            });
        }

        for cat in old_cats.intersection(&new_cats) {
            let old_tools: BTreeSet<&String> = old_repo.tools[*cat].iter().collect();
            let new_tools: BTreeSet<&String> = new_repo.tools[*cat].iter().collect();
            for tool in old_tools.difference(&new_tools) {
                findings.push(Finding {
                    severity: Severity::Breaking,
                    path: format!("repos.{repo}.tools.{cat}.{tool}"),
                    message: format!("tool '{tool}' was removed from '{repo}' category '{cat}'"),
                });
            }
            for tool in new_tools.difference(&old_tools) {
                findings.push(Finding {
                    severity: Severity::Compatible,
                    path: format!("repos.{repo}.tools.{cat}.{tool}"),
                    message: format!("new tool '{tool}' added to '{repo}' category '{cat}'"),
                });
            }
        }
    }

    findings
}

/// Convenience: `true` if any finding is classified as `Breaking`.
pub fn has_breaking(findings: &[Finding]) -> bool {
    findings
        .iter()
        .any(|f| matches!(f.severity, Severity::Breaking))
}

/// Load a registry JSON file from disk.
pub fn load(path: &std::path::Path) -> Result<Registry, CheckError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| CheckError::Io(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&raw).map_err(|e| CheckError::Parse(e.to_string()))
}

/// Error type for I/O and parse failures.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    /// Filesystem error.
    #[error("io: {0}")]
    Io(String),
    /// JSON parse failure.
    #[error("parse: {0}")]
    Parse(String),
}
