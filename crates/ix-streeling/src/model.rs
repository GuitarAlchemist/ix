//! The normalized learnings record — the cross-repo contract type.
//!
//! Schema: `docs/contracts/streeling-learning.schema.json` (contract v0.1).
//! Source-of-truth stays in each repo's `.md` files; this is the derived shape
//! that lands in `state/streeling/catalog.jsonl`.

use serde::{Deserialize, Serialize};

/// Contract schema version. Bump on any field change (coordinate cross-repo).
pub const SCHEMA_VERSION: &str = "0.1.0";

/// What kind of learning artifact a record came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// `docs/solutions/**` — a `/learnings` writeup (symptom + root_cause).
    Solution,
    /// `state/knowledge/**` — a Galactic-Protocol knowledge package.
    Knowledge,
    /// `docs/plans/**` — a planned change (one-way-door log).
    Plan,
    /// `docs/brainstorms/**` — a design exploration.
    Brainstorm,
}

impl Kind {
    /// Default faculty/category when the frontmatter doesn't name one.
    pub fn default_category(self) -> &'static str {
        match self {
            Kind::Solution => "uncategorized",
            Kind::Knowledge => "knowledge",
            Kind::Plan => "plan",
            Kind::Brainstorm => "brainstorm",
        }
    }
}

/// One normalized learning, ready for the catalog. A superset of the ix/ga
/// `docs/solutions` frontmatter so heterogeneous sources map cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRecord {
    pub schema_version: String,
    /// Stable id: `"{repo}:{path}"`.
    pub id: String,
    /// Originating repo (`ix`, `ga`, `tars`, `Demerzel`).
    pub repo: String,
    pub kind: Kind,
    /// Faculty — solution category, or the kind's default.
    pub category: String,
    pub title: String,
    /// ISO date string as written in the source, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symptom: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_cause: Option<String>,
    /// Repo-relative path to the source file (POSIX separators).
    pub path: String,
}

impl LearningRecord {
    /// Construct an id from repo + repo-relative path.
    pub fn make_id(repo: &str, path: &str) -> String {
        format!("{repo}:{path}")
    }
}

impl ix_registrar::Record for LearningRecord {
    // The id is already globally unique (`"{repo}:{path}"`), so it is both the
    // dedup key and the reported id.
    fn dedup_key(&self) -> String {
        self.id.clone()
    }
    fn report_id(&self) -> String {
        self.id.clone()
    }
    fn repo(&self) -> &str {
        &self.repo
    }
}
