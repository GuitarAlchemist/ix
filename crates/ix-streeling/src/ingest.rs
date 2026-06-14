//! Discover learning files across repos and parse their YAML frontmatter into
//! normalized [`LearningRecord`]s. Tolerant by design: a file with no/invalid
//! frontmatter is skipped-and-counted, never fatal; a missing repo root is
//! reported, not an error (cf. the `governance/demerzel` submodule pattern).

use crate::model::{Kind, LearningRecord, SCHEMA_VERSION};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// A repo to scan for learnings.
pub struct SourceRoot {
    pub repo: String,
    pub root: PathBuf,
}

impl SourceRoot {
    pub fn new(repo: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self { repo: repo.into(), root: root.into() }
    }
}

/// ix scans all four stores; sibling repos scan `docs/solutions` only in v1
/// (tars/Demerzel native stores need adapters — fast-follow, see the plan).
const IX_GROUPS: &[(&str, Kind)] = &[
    ("docs/solutions", Kind::Solution),
    ("state/knowledge", Kind::Knowledge),
    ("docs/plans", Kind::Plan),
    ("docs/brainstorms", Kind::Brainstorm),
];
const SOLUTION_ONLY: &[(&str, Kind)] = &[("docs/solutions", Kind::Solution)];

/// Which (subdir, kind) groups to scan for a given repo.
pub fn groups_for(repo: &str) -> &'static [(&'static str, Kind)] {
    if repo == "ix" {
        IX_GROUPS
    } else {
        SOLUTION_ONLY
    }
}

/// Outcome of an ingest pass.
#[derive(Debug, Default)]
pub struct IngestReport {
    pub records: Vec<LearningRecord>,
    /// Files scanned but skipped (no/invalid frontmatter).
    pub skipped: usize,
    pub roots_seen: Vec<String>,
    pub roots_missing: Vec<String>,
}

/// Raw frontmatter — every field optional so heterogeneous sources parse.
/// Unknown keys (e.g. a plan's `type`/`status`) are ignored.
#[derive(Debug, Default, Deserialize)]
struct Frontmatter {
    title: Option<String>,
    category: Option<String>,
    /// brainstorms use `topic` instead of `title`.
    topic: Option<String>,
    date: Option<serde_yaml::Value>,
    #[serde(default)]
    tags: Vec<String>,
    symptom: Option<String>,
    root_cause: Option<String>,
}

/// Extract the YAML frontmatter block delimited by leading `---` fences.
fn split_frontmatter(content: &str) -> Option<&str> {
    let rest = content.strip_prefix("---")?;
    let rest = rest.strip_prefix("\r\n").or_else(|| rest.strip_prefix('\n'))?;
    let end = rest.find("\n---").or_else(|| rest.find("\r\n---"))?;
    Some(&rest[..end])
}

fn date_to_string(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        other => serde_yaml::to_string(other).ok().map(|s| s.trim().to_string()),
    }
}

fn posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Parse one file into a record, or `None` if it lacks usable frontmatter.
fn parse_file(repo: &str, root: &Path, file: &Path, kind: Kind) -> Option<LearningRecord> {
    let content = std::fs::read_to_string(file).ok()?;
    let fm_str = split_frontmatter(&content)?;
    let fm: Frontmatter = serde_yaml::from_str(fm_str).ok()?;
    let rel = posix(file.strip_prefix(root).unwrap_or(file));
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let title = fm.title.or(fm.topic).unwrap_or(stem);
    let category = fm.category.unwrap_or_else(|| kind.default_category().to_string());
    Some(LearningRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        id: LearningRecord::make_id(repo, &rel),
        repo: repo.to_string(),
        kind,
        category,
        title,
        date: fm.date.as_ref().and_then(date_to_string),
        tags: fm.tags,
        symptom: fm.symptom,
        root_cause: fm.root_cause,
        path: rel,
    })
}

/// Ingest every configured root. Records are returned sorted by `id` for stable output.
pub fn ingest(roots: &[SourceRoot]) -> IngestReport {
    let mut report = IngestReport::default();
    for sr in roots {
        if !sr.root.exists() {
            report.roots_missing.push(sr.repo.clone());
            continue;
        }
        report.roots_seen.push(sr.repo.clone());
        for (subdir, kind) in groups_for(&sr.repo) {
            let dir = sr.root.join(subdir);
            if !dir.exists() {
                continue;
            }
            for entry in WalkDir::new(&dir).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path();
                if !p.is_file() || p.extension().and_then(|e| e.to_str()) != Some("md") {
                    continue;
                }
                match parse_file(&sr.repo, &sr.root, p, *kind) {
                    Some(rec) => report.records.push(rec),
                    None => report.skipped += 1,
                }
            }
        }
    }
    report.records.sort_by(|a, b| a.id.cmp(&b.id));
    report
}
