//! Hardcoded-path check — source that names one developer's home directory.
//!
//! A path under `C:\Users\<name>`, `/home/<name>` or `/Users/<name>` works on
//! exactly one machine. In production code it is a default that resolves to
//! nothing for every other contributor (ix#327: the sentrux bridge fell back to
//! one developer's `bin/`). In a test it is a guard that always fires elsewhere,
//! so the test passes having checked nothing (ix#328: a cross-repo render audit
//! that never ran on CI). In a doc comment or a script it is an example nobody
//! else can paste. None of the three is caught by the compiler, clippy, or the
//! test suite: the code is valid, the path is just not yours.
//!
//! This check reads every source and script file under `crates/` (see
//! [`SCANNED_EXTENSIONS`], matched case-insensitively) and reports each line
//! that names a user's home directory. Portable spellings are not findings: a
//! placeholder segment (`<you>`, `$USER`, `%USERNAME%`, `{user}`), the shared
//! Windows accounts (`Public`, `Default`, `All Users`), a `~/` prefix, or a
//! repo-relative path. A line whose text must keep such a path (a sanitizer's
//! negative input, a fixture of this check, a URL whose path happens to contain
//! `/home/`) belongs in the allowlist with a reason, keyed by file and a
//! substring of the line, so the exemption covers those lines and not the file.
//!
//! # What counts as a home path
//!
//! A drive letter (a single letter, not the tail of a word such as `https`), `:`,
//! one or more separators, `users` in any case, one or more separators, then the
//! account (`C:\Users\<name>`, `c:/users/<name>`, `C:\\Users\\<name>` as escaped
//! in a string, `C:\\\\Users\\\\<name>` as JSON inside a string, `C:/Users\<name>`
//! mixed); a root-relative `\Users\<name>` at the start of a token; or one or
//! more `/`, then exactly `home` or `Users`, then one or more `/`, then the
//! account. A lower-case `/users/` with no drive letter is an HTTP route, not a
//! home.
//!
//! # Fail closed
//!
//! A directory that cannot be listed, an entry whose type cannot be read, a
//! file that cannot be read, or a `crates/` that is not there, is a failed scan,
//! not an empty one: the check reports `FAIL` rather than counting what it did
//! manage to read.
//!
//! # Deliberate under-reporting
//!
//! Like its neighbours, every remaining ambiguity resolves toward *not*
//! reporting.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Path of the allowlist, relative to the repo root.
pub const ALLOWLIST_PATH: &str = "state/registry/hardcoded-paths.allow.json";

/// Directory, relative to the repo root, whose files are scanned.
pub const SCAN_DIR: &str = "crates";

/// File extensions scanned under [`SCAN_DIR`] (compared case-insensitively):
/// Rust, SQL, the scripts that sit beside them, manifests and the crate-level docs.
pub const SCANNED_EXTENSIONS: &[&str] = &["rs", "sql", "ps1", "sh", "py", "toml", "md"];

/// Windows accounts that belong to the machine, not to a person. Compared
/// case-insensitively against the start of what follows the home marker, so
/// the two-word ones match before the segment is cut at the space.
const SHARED_WINDOWS_ACCOUNTS: &[&str] = &["all users", "default user", "public", "default"];

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
    /// The whole trimmed line; what the allowlist's `contains` is matched on.
    #[serde(skip)]
    pub text: String,
}

/// One allowlist entry: lines that may keep their path, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowEntry {
    /// Repo-relative file path with forward slashes.
    pub file: String,
    /// A substring the allowed lines must contain, so the exemption covers
    /// those lines and not the whole file.
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

fn is_sep(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

/// A character that can precede a path token: start of line, whitespace, a
/// quote, a bracket, or the `=` of an assignment.
fn can_start_token(before: Option<u8>) -> bool {
    match before {
        None => true,
        Some(b) => b.is_ascii_whitespace() || matches!(b, b'"' | b'\'' | b'`' | b'(' | b'[' | b'='),
    }
}

/// The account segment named by `line`, if the line contains a home path.
///
/// Returns the first segment that is not a placeholder and not a shared
/// Windows account.
pub fn home_user_in(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let lower = line.to_ascii_lowercase();
    let mut candidates: Vec<usize> = Vec::new();

    // `users` in any case (Windows), `home` and `Users` exactly (POSIX).
    for (needle, hay, exact) in [("users", lower.as_str(), false), ("home", line, true)] {
        let mut from = 0;
        while let Some(i) = hay[from..].find(needle) {
            let at = from + i;
            from = at + needle.len();

            // One or more separators before the keyword.
            let mut s = at;
            while s > 0 && is_sep(bytes[s - 1]) {
                s -= 1;
            }
            if s == at {
                continue;
            }
            let run = &bytes[s..at];
            let before_run = s.checked_sub(1).map(|i| bytes[i]);

            // `X:` where X is a single letter: not the `s:` of `https:`.
            let drive = s >= 2
                && bytes[s - 1] == b':'
                && bytes[s - 2].is_ascii_alphabetic()
                && !(s >= 3 && bytes[s - 3].is_ascii_alphanumeric());
            // `\Users\<name>` at the start of a token.
            let root_relative = run.iter().all(|&b| b == b'\\') && can_start_token(before_run);
            // `/home/<name>`, `/Users/<name>`.
            let posix = run.iter().all(|&b| b == b'/')
                && (exact || &line[at..at + needle.len()] == "Users");

            let ok = if exact {
                posix
            } else {
                drive || root_relative || posix
            };
            if !ok {
                continue;
            }

            // One or more separators after it, then the account.
            let mut e = at + needle.len();
            while e < bytes.len() && is_sep(bytes[e]) {
                e += 1;
            }
            if e == at + needle.len() {
                continue;
            }
            candidates.push(e);
        }
    }

    candidates.sort_unstable();
    candidates.dedup();
    candidates.into_iter().find_map(|start| {
        let rest = &line[start..];
        let rest_lower = rest.to_ascii_lowercase();
        if SHARED_WINDOWS_ACCOUNTS.iter().any(|acct| {
            rest_lower.starts_with(acct)
                && rest_lower[acct.len()..].chars().next().ends_an_account()
        }) {
            return None;
        }
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
        Some(segment.to_string())
    })
}

/// Whether an optional char ends an account name: nothing, a separator, a
/// quote or whitespace.
trait AccountEnd {
    fn ends_an_account(self) -> bool;
}

impl AccountEnd for Option<char> {
    fn ends_an_account(self) -> bool {
        match self {
            None => true,
            Some(c) => c.is_whitespace() || matches!(c, '/' | '\\' | '"' | '\'' | '`'),
        }
    }
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
                text: line.trim().to_string(),
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
    let dir = root.join(SCAN_DIR);
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let mut files = Vec::new();
    collect_sources(&dir, &mut files)?;
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

    // Every entry that covers a line counts as used, so two entries whose
    // substrings both appear on one line are both live.
    let mut matched = vec![false; allowlist.allow.len()];
    for f in &census.findings {
        let mut covered = false;
        for (i, e) in allowlist.allow.iter().enumerate() {
            if e.file == f.file && !e.contains.is_empty() && f.text.contains(&e.contains) {
                matched[i] = true;
                covered = true;
            }
        }
        if covered {
            census.allowed.push(format!("{}:{}", f.file, f.line));
        } else {
            census.unlisted.push(f.clone());
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

/// Every scanned file under `dir`, skipping build output. A directory that
/// cannot be listed, or an entry whose type cannot be read, fails the scan.
fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("listing {}: {e}", dir.display()))?;
        let path = entry.path();
        let kind = std::fs::metadata(&path)
            .map_err(|e| format!("reading the type of {}: {e}", path.display()))?;
        if kind.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_sources(&path, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| SCANNED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        {
            out.push(path);
        }
    }
    Ok(())
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
                    "this check reads crates/** and state/registry/hardcoded-paths.allow.json \
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
             when none resolves; in a doc example write `../ga/...` or `<repo>/...`; a line \
             whose text must keep such a path goes in state/registry/hardcoded-paths.allow.json \
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
