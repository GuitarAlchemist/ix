//! Hardcoded-path check — source that names one developer's home directory.
//!
//! A path under `C:\Users\<name>`, `/home/<name>` or `/Users/<name>` works on
//! exactly one machine. In production code it is a default that resolves to
//! nothing for every other contributor (ix#327: the sentrux bridge fell back to
//! one developer's `bin/`). In a test it is a guard that always fires elsewhere,
//! so the test passes having checked nothing (ix#328: a cross-repo render audit
//! that never ran on CI). In a doc comment it is an example nobody else can
//! paste. None of the three is caught by the compiler, clippy, or the test
//! suite: the code is valid, the path is just not yours.
//!
//! This check reads every `.rs` file under `crates/` and reports each line that
//! names a user's home directory. Portable spellings are not findings: a
//! placeholder segment (`<you>`, `$USER`, `%USERNAME%`, `{user}`), the shared
//! Windows accounts (`Public`, `Default`), a `~/` prefix, or a repo-relative
//! path. A test whose *input* is such a path (a sanitizer proving it rejects
//! them) belongs in the allowlist with a reason, keyed by file and a substring
//! of the line, so the exemption covers that line and not the file.
//!
//! # Deliberate under-reporting
//!
//! Like its neighbours, every ambiguity resolves toward *not* reporting. A
//! `/users/` segment in lower case is only a finding when a drive letter
//! precedes it (`c:/users/`), so an HTTP route like `/users/{id}` is never one.
//! `/home/` is matched exactly: a URL path containing it would be reported,
//! and the allowlist is where that case goes if it ever appears.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Path of the allowlist, relative to the repo root.
pub const ALLOWLIST_PATH: &str = "state/registry/hardcoded-paths.allow.json";

/// Directory, relative to the repo root, whose `.rs` files are scanned.
pub const SCAN_DIR: &str = "crates";

/// Windows accounts that belong to the machine, not to a person.
const SHARED_WINDOWS_ACCOUNTS: &[&str] = &["public", "default", "default user", "all users"];

/// One line that names a home directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// Repo-relative path with forward slashes, e.g. `crates/x/src/lib.rs`.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// The account segment that follows the home marker, e.g. `jane`.
    pub user: String,
    /// The line, trimmed and truncated for the report.
    pub excerpt: String,
}

/// One allowlist entry: a line that may keep its path, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowEntry {
    /// Repo-relative file path with forward slashes.
    pub file: String,
    /// A substring the allowed line must contain, so the exemption covers one
    /// line and not the whole file.
    pub contains: String,
    /// Why this path may stay. Required and non-empty.
    pub reason: String,
}

/// The committed allowlist.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Allowlist {
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub allow: Vec<AllowEntry>,
}

impl Allowlist {
    /// Read the allowlist from `root`. A missing file is an empty allowlist;
    /// a file that does not parse is an error, not an empty list.
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(ALLOWLIST_PATH);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

/// What the scan found.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Census {
    pub files_scanned: usize,
    /// Every line naming a home directory, allowlisted or not.
    pub findings: Vec<Finding>,
    /// Findings no allowlist entry covers. These fail the check.
    pub unlisted: Vec<Finding>,
    /// Findings an allowlist entry covers, as `file:line`.
    pub allowed: Vec<String>,
    /// Allowlist entries that matched no finding, as `file (contains)`.
    pub stale_allowlist: Vec<String>,
    /// Allowlist entries with an empty `reason`, as `file (contains)`.
    pub reasonless_allowlist: Vec<String>,
}

/// The account segment named by `line`, if the line contains a home path.
///
/// Returns the first segment that is not a placeholder and not a shared
/// Windows account.
pub fn home_user_in(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    let mut candidates: Vec<usize> = Vec::new();

    // Drive-letter forms, any case: `c:/users/`, `c:\users\`, `c:\\users\\`.
    for marker in [":/users/", ":\\users\\", ":\\\\users\\\\"] {
        let mut from = 0;
        while let Some(i) = lower[from..].find(marker) {
            let at = from + i;
            let drive_before = at > 0 && lower.as_bytes()[at - 1].is_ascii_alphabetic();
            if drive_before {
                candidates.push(at + marker.len());
            }
            from = at + marker.len();
        }
    }
    // POSIX forms, exact case: `/home/` (Linux) and `/Users/` (macOS).
    for marker in ["/home/", "/Users/"] {
        let mut from = 0;
        while let Some(i) = line[from..].find(marker) {
            let at = from + i;
            candidates.push(at + marker.len());
            from = at + marker.len();
        }
    }

    candidates.sort_unstable();
    candidates.into_iter().find_map(|start| {
        let rest = &line[start..];
        let end = rest
            .find(|c: char| {
                c.is_whitespace()
                    || matches!(
                        c,
                        '/' | '\\' | '"' | '\'' | '<' | '>' | ')' | ',' | ';' | '`'
                    )
            })
            .unwrap_or(rest.len());
        let segment = &rest[..end];
        if segment.is_empty() {
            return None;
        }
        let first = segment.chars().next().unwrap_or(' ');
        if matches!(first, '$' | '%' | '{' | '*' | '~' | '.' | '[' | '(') {
            return None;
        }
        if SHARED_WINDOWS_ACCOUNTS.contains(&segment.to_ascii_lowercase().as_str()) {
            return None;
        }
        Some(segment.to_string())
    })
}

/// Findings in one source file, labelled `file_label`.
pub fn findings_in(file_label: &str, src: &str) -> Vec<Finding> {
    src.lines()
        .enumerate()
        .filter_map(|(i, line)| {
            home_user_in(line).map(|user| Finding {
                file: file_label.to_string(),
                line: i + 1,
                user,
                excerpt: excerpt(line),
            })
        })
        .collect()
}

fn excerpt(line: &str) -> String {
    const MAX: usize = 120;
    let t = line.trim();
    if t.chars().count() <= MAX {
        t.to_string()
    } else {
        let cut: String = t.chars().take(MAX).collect();
        format!("{cut}…")
    }
}

/// Scan `root` against its committed allowlist.
pub fn scan(root: &Path) -> Result<Census, String> {
    let allowlist = Allowlist::load(root)?;
    scan_with_allowlist(root, &allowlist)
}

/// Scan `root` against `allowlist`.
pub fn scan_with_allowlist(root: &Path, allowlist: &Allowlist) -> Result<Census, String> {
    let mut files = Vec::new();
    collect_rs(&root.join(SCAN_DIR), &mut files);
    files.sort();

    let mut census = Census {
        files_scanned: files.len(),
        ..Census::default()
    };

    for path in &files {
        let src = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let label = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        census.findings.extend(findings_in(&label, &src));
    }

    let mut matched = vec![false; allowlist.allow.len()];
    for f in &census.findings {
        let hit = allowlist.allow.iter().position(|e| {
            e.file == f.file && !e.contains.is_empty() && f.excerpt.contains(&e.contains)
        });
        match hit {
            Some(i) => {
                matched[i] = true;
                census.allowed.push(format!("{}:{}", f.file, f.line));
            }
            None => census.unlisted.push(f.clone()),
        }
    }

    for (entry, was_matched) in allowlist.allow.iter().zip(matched) {
        let key = format!("{} ({})", entry.file, entry.contains);
        if entry.reason.trim().is_empty() {
            census.reasonless_allowlist.push(key.clone());
        }
        if !was_matched {
            census.stale_allowlist.push(key);
        }
    }

    Ok(census)
}

/// Every `.rs` file under `dir`, skipping build output.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The check as `ix doctor` reports it.
pub fn check(root: &Path) -> super::CheckResult {
    use super::{CheckResult, Status};
    const NAME: &str = "hardcoded-paths";

    let census = match scan(root) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, format!("could not scan sources: {e}"))
                .with_remedy(
                "this check reads crates/**/*.rs and state/registry/hardcoded-paths.allow.json \
                     directly — a read failure means the scan did not happen, so it is a failure \
                     rather than a pass",
            )
        }
    };

    let details = serde_json::to_value(&census).unwrap_or(serde_json::Value::Null);

    if !census.reasonless_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "allowlist entries without a reason: {}",
                census.reasonless_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "every entry in state/registry/hardcoded-paths.allow.json needs a non-empty \
             `reason` — an exemption nobody justified is a silencer, not a decision",
        )
        .with_details(details);
    }

    if !census.unlisted.is_empty() {
        let listed = census
            .unlisted
            .iter()
            .map(|f| format!("{}:{} (`{}`)", f.file, f.line, f.user))
            .collect::<Vec<_>>()
            .join(", ");
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "{} line(s) name one developer's home directory: {listed}",
                census.unlisted.len()
            ),
        )
        .with_remedy(
            "resolve the location from an environment variable, PATH, or the workspace root \
             (sibling repos live at `../<repo>`), and fail with a message naming those options \
             when none resolves; in a doc example write `../ga/...` or `<repo>/...`; a test \
             whose input must be such a path goes in state/registry/hardcoded-paths.allow.json \
             with `file`, `contains` and a `reason`",
        )
        .with_details(details);
    }

    if !census.stale_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Warn,
            format!(
                "allowlist entries that match no line: {}",
                census.stale_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "the line was fixed or moved — drop its entry from \
             state/registry/hardcoded-paths.allow.json so the list stays honest",
        )
        .with_details(details);
    }

    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} source file(s) scanned; {} allowlisted line(s), 0 unlisted home paths",
            census.files_scanned,
            census.allowed.len()
        ),
    )
    .with_details(details)
}
