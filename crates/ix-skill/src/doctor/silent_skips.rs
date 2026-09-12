//! Silent-skip check — tests that pass without reaching an assertion.
//!
//! A `#[test]` function that probes its environment and returns early when the
//! probe fails is reported by `cargo test` as **passing**. It ran, it did not
//! fail, and it asserted nothing. Nothing in the repository's output
//! distinguishes it from a test that actually checked something, which is the
//! same invisibility class as [`super::dark_features`] (ix#315) reached by a
//! different route: there, the code is never compiled; here, it is compiled and
//! executed and simply steps over its own assertions.
//!
//! The motivating cases are ix#294. Five tests in `ix-duck` bound the DuckDB
//! SQL surfaces to their frozen goldens through the `duckdb` CLI, and each
//! handled a missing binary with `eprintln!(…); return;`. CI has no duckdb, so
//! every SQL surface in the repository was verified on a developer machine and
//! nowhere else, behind five green test results. Three more were found later in
//! `ix-sentrux-annotations`, `ix-voicings` and `ix-skill` — the last one inside
//! a test whose own comment says it exists to catch scans that silently return
//! nothing.
//!
//! # This is not an argument against skipping
//!
//! Some skips are correct. A test that needs a sibling repository's build
//! output cannot run on this repository's CI, and failing it there would make
//! the suite red for an environmental reason no contributor can fix. The defect
//! is not the skip — it is that **a skipping test is indistinguishable from an
//! asserting one in every signal the repository produces**. This check restores
//! the distinction and takes no position on whether a given skip should exist.
//! A skip that must stay belongs in the allowlist with a written reason.
//!
//! # Where the answer comes from
//!
//! Derived per run from the test sources, never a list to maintain. A finding
//! needs three things to be true at once:
//!
//! 1. a bare `return;` lexically inside a `#[test]` function's body,
//! 2. nested at least one block deeper than the body — an unconditional
//!    `return;` at body level ends a test, it does not skip one,
//! 3. an environment probe in the preceding [`PROBE_WINDOW`] lines: a path
//!    existence test, a spawned command, an environment variable, or a
//!    fallible operation whose failure arm is the block being returned from.
//!
//! # Reporting size, not just existence
//!
//! A guard that steps over nothing is noise; one that steps over eleven
//! assertions is the defect this check exists for. So each finding carries the
//! number of assertion sites it can bypass — `assert*`, `panic!`,
//! `unreachable!`, `todo!` after the guard — and only findings that bypass at
//! least one can fail the check.
//!
//! # Deliberate under-reporting
//!
//! Like its siblings, every ambiguity resolves toward *not* reporting:
//!
//! - only `crates/*/tests/**` is scanned, so `#[cfg(test)]` unit tests inside
//!   `src/` are missed entirely;
//! - a skip expressed as `return Ok(());` or by a helper that returns on the
//!   caller's behalf is not recognised — only a bare `return;`;
//! - a probe further back than [`PROBE_WINDOW`] lines is not associated with
//!   its guard;
//! - string literals, char literals and comments are blanked before scanning,
//!   so a brace or the word `return` inside them cannot create a finding, but
//!   real code inside a raw string would be skipped over.
//!
//! The check under-states how many tests can silently pass rather than crying
//! wolf.

use super::dark_features::Kind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Path of the allowlist, relative to the repo root.
pub const ALLOWLIST_PATH: &str = "state/registry/silent-skips.allow.json";

/// How far back from a `return;` an environment probe is looked for.
///
/// Wide enough to cover a `match Command::new(..).output() { .. Err(e) => {`
/// spread over a formatted block, narrow enough that an unrelated probe earlier
/// in a long test is not credited to this guard.
pub const PROBE_WINDOW: usize = 30;

/// What kind of environment probe guards a skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Probe {
    /// A filesystem path was tested for existence.
    MissingPath,
    /// An external program was spawned.
    MissingCommand,
    /// An environment variable was read.
    MissingEnvVar,
    /// A fallible operation's error arm.
    OperationFailed,
}

impl Probe {
    /// Human-readable, for the summary line.
    pub fn label(self) -> &'static str {
        match self {
            Probe::MissingPath => "a path that does not exist",
            Probe::MissingCommand => "an external command that is not installed",
            Probe::MissingEnvVar => "an unset environment variable",
            Probe::OperationFailed => "a fallible operation that failed",
        }
    }
}

/// One test that can return before asserting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// Repo-relative path, forward slashes.
    pub file: String,
    /// Name of the `#[test]` function.
    pub test: String,
    /// 1-based line of the `return;` that skips.
    pub line: usize,
    /// What the guard was probing for.
    pub probe: Probe,
    /// Assertion sites after the guard that this skip can bypass.
    pub assertions_skipped: usize,
    /// Whether the allowlist accounts for this one.
    pub allowed: bool,
}

/// An allowlist entry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AllowEntry {
    /// Repo-relative path, forward slashes.
    pub file: String,
    /// Test function name.
    pub test: String,
    /// Which sort of exemption this is. Shared with `dark-features`, so an
    /// unknown value fails parsing instead of silently counting as either.
    pub kind: Kind,
    /// Why this skip may stay. Must be non-empty.
    pub reason: String,
    /// Required when `kind` is `tracked`.
    #[serde(default)]
    pub issue: Option<String>,
}

/// The allowlist file.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Allowlist {
    /// Why the file exists; ignored by the check.
    #[serde(default)]
    pub note: String,
    /// The entries.
    #[serde(default)]
    pub allow: Vec<AllowEntry>,
}

impl Allowlist {
    /// Read the allowlist, treating absence as empty.
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(ALLOWLIST_PATH);
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("parsing {}: {e}", path.display()))
    }

    fn covers(&self, file: &str, test: &str) -> bool {
        self.allow.iter().any(|a| a.file == file && a.test == test)
    }
}

/// Everything one scan found.
#[derive(Debug, Clone, Serialize)]
pub struct Census {
    /// Test files read.
    pub files_scanned: usize,
    /// `#[test]` functions inspected.
    pub tests_scanned: usize,
    /// Every skip found, allowlisted or not.
    pub findings: Vec<Finding>,
    /// Allowlist entries whose test no longer skips, or no longer exists.
    pub stale_allowlist: Vec<String>,
    /// Allowlist entries that are malformed: empty reason, or `tracked`
    /// without an issue.
    pub unjustified_allowlist: Vec<String>,
}

impl Census {
    /// Findings the allowlist does not account for that bypass an assertion.
    pub fn blocking(&self) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| !f.allowed && f.assertions_skipped > 0)
            .collect()
    }

    /// Findings the allowlist does not account for that bypass nothing.
    pub fn advisory(&self) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| !f.allowed && f.assertions_skipped == 0)
            .collect()
    }
}

// ------------------------------------------------------------------- lexing

/// Replace comments, string literals and char literals with spaces, keeping
/// every byte offset and line break intact.
///
/// Brace counting and marker matching both run over the result, so a `{` in a
/// doc comment or the word `return` in an error message cannot manufacture a
/// finding. Offsets are preserved so a line number derived from the blanked
/// text is correct in the original.
pub fn blank_noise(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = vec![b' '; b.len()];
    let mut i = 0;
    while i < b.len() {
        // Keep newlines everywhere so line numbers survive.
        if b[i] == b'\n' {
            out[i] = b'\n';
            i += 1;
            continue;
        }
        // Line comment.
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment, nested per the Rust grammar.
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let mut depth = 1usize;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == b'\n' {
                    out[i] = b'\n';
                } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 1;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 1;
                }
                i += 1;
            }
            continue;
        }
        // Raw string: r"..", r#".."#, br#".."# and so on.
        if b[i] == b'r' || (b[i] == b'b' && i + 1 < b.len() && b[i + 1] == b'r') {
            let start = i;
            let mut j = if b[i] == b'b' { i + 2 } else { i + 1 };
            let hashes = {
                let h = j;
                while j < b.len() && b[j] == b'#' {
                    j += 1;
                }
                j - h
            };
            if j < b.len() && b[j] == b'"' {
                j += 1;
                // Scan to a closing quote followed by the same hash count.
                while j < b.len() {
                    if b[j] == b'\n' {
                        out[j] = b'\n';
                    }
                    if b[j] == b'"' {
                        let mut k = j + 1;
                        let mut seen = 0;
                        while k < b.len() && b[k] == b'#' && seen < hashes {
                            k += 1;
                            seen += 1;
                        }
                        if seen == hashes {
                            j = k;
                            break;
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
            // Not a raw string after all: keep the identifier byte.
            out[start] = b[start];
            i = start + 1;
            continue;
        }
        // Ordinary string.
        if b[i] == b'"' {
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    // A `\` line continuation -- which this workspace uses
                    // heavily to wrap long messages -- puts a real newline
                    // inside the literal. Skipping both bytes silently would
                    // drop it, and every line number after the first such
                    // string would be reported too low.
                    if i + 1 < b.len() && b[i + 1] == b'\n' {
                        out[i + 1] = b'\n';
                    }
                    i += 2;
                    continue;
                }
                if b[i] == b'\n' {
                    out[i] = b'\n';
                }
                if b[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        // Char literal, distinguished from a lifetime by the closing quote.
        if b[i] == b'\'' {
            let mut j = i + 1;
            if j < b.len() && b[j] == b'\\' {
                j += 2;
            } else if j < b.len() {
                j += 1;
            }
            if j < b.len() && b[j] == b'\'' {
                i = j + 1;
                continue;
            }
            out[i] = b[i];
            i += 1;
            continue;
        }
        out[i] = b[i];
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

// ------------------------------------------------------------------ scanning

/// Markers that mean "this block is reached because the environment was not
/// what the test needed".
fn classify_probe(window: &str) -> Option<Probe> {
    // Order matters: a command probe usually also contains an `Err(` arm, and
    // naming the command is the more useful diagnosis.
    if window.contains("Command::new")
        || window.contains(".output()")
        || window.contains("which::which")
    {
        return Some(Probe::MissingCommand);
    }
    if window.contains(".exists()") || window.contains("is_file()") || window.contains("is_dir()") {
        return Some(Probe::MissingPath);
    }
    if window.contains("env::var") || window.contains("var_os") {
        return Some(Probe::MissingEnvVar);
    }
    if window.contains("Err(")
        || window.contains("else {")
        || window.contains("is_err()")
        || window.contains(".ok()")
    {
        return Some(Probe::OperationFailed);
    }
    None
}

/// Count assertion sites in a slice of blanked source.
fn count_assertions(text: &str) -> usize {
    let mut n = 0;
    for marker in [
        "assert!",
        "assert_eq!",
        "assert_ne!",
        "panic!",
        "unreachable!",
        "todo!",
    ] {
        n += text.matches(marker).count();
    }
    n
}

/// Offset of the matching close brace for the `{` at `open`.
fn matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Scan one file's source for skipping tests. Returns the findings and how many
/// `#[test]` functions were inspected.
pub fn findings_in(file_label: &str, src: &str) -> (Vec<Finding>, usize) {
    let blanked = blank_noise(src);
    let bytes = blanked.as_bytes();
    let mut findings = Vec::new();
    let mut tests_seen = 0;

    // Every `#[test]` attribute, including `#[tokio::test]`.
    let mut search = 0usize;
    while let Some(rel) = blanked[search..].find("test]") {
        let at = search + rel;
        search = at + 5;
        // Confirm it is an attribute, not the tail of an identifier.
        let Some(hash) = blanked[..at].rfind('#') else {
            continue;
        };
        if !blanked[hash..at].starts_with("#[") {
            continue;
        }
        let between = &blanked[hash + 2..at];
        if !between.is_empty() && !between.ends_with("::") {
            continue;
        }

        // The function this attribute decorates.
        let Some(fn_rel) = blanked[at..].find("fn ") else {
            continue;
        };
        let fn_at = at + fn_rel + 3;
        let name: String = blanked[fn_at..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let Some(body_rel) = blanked[fn_at..].find('{') else {
            continue;
        };
        let body_open = fn_at + body_rel;
        let Some(body_close) = matching_brace(bytes, body_open) else {
            continue;
        };
        tests_seen += 1;

        // Bare `return;` inside the body, deeper than body level.
        let body = &blanked[body_open..body_close];
        let mut off = 0usize;
        while let Some(r) = body[off..].find("return;") {
            let ret_at = off + r;
            off = ret_at + 7;

            // Depth relative to the body's own brace: 1 is the body itself, so
            // a guarded return sits deeper.
            let mut depth = 0i32;
            for &c in &bytes[body_open..body_open + ret_at] {
                if c == b'{' {
                    depth += 1;
                } else if c == b'}' {
                    depth -= 1;
                }
            }
            if depth <= 1 {
                continue;
            }

            // Look back a bounded number of lines for the probe.
            let window: String = body[..ret_at]
                .lines()
                .rev()
                .take(PROBE_WINDOW)
                .collect::<Vec<_>>()
                .join("\n");
            let Some(probe) = classify_probe(&window) else {
                continue;
            };

            let line = 1 + blanked[..body_open + ret_at].matches('\n').count();
            findings.push(Finding {
                file: file_label.to_string(),
                test: name.clone(),
                line,
                probe,
                assertions_skipped: count_assertions(&body[ret_at..]),
                allowed: false,
            });
        }
    }
    (findings, tests_seen)
}

/// Walk `crates/*/tests/**` and report every test that can skip.
pub fn scan(root: &Path) -> Result<Census, String> {
    let allowlist = Allowlist::load(root)?;
    scan_with_allowlist(root, &allowlist)
}

/// [`scan`] against a supplied allowlist, so tests need no file on disk.
pub fn scan_with_allowlist(root: &Path, allowlist: &Allowlist) -> Result<Census, String> {
    let crates = root.join("crates");
    let mut files = Vec::new();
    if crates.is_dir() {
        let entries =
            std::fs::read_dir(&crates).map_err(|e| format!("reading {}: {e}", crates.display()))?;
        for entry in entries.flatten() {
            let tests = entry.path().join("tests");
            if tests.is_dir() {
                collect_rs(&tests, &mut files);
            }
        }
    }
    files.sort();

    let mut findings = Vec::new();
    let mut tests_scanned = 0usize;
    for file in &files {
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        let label = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        let (mut found, seen) = findings_in(&label, &src);
        tests_scanned += seen;
        for f in &mut found {
            f.allowed = allowlist.covers(&f.file, &f.test);
        }
        findings.append(&mut found);
    }
    findings.sort_by(|a, b| (&a.file, &a.test, a.line).cmp(&(&b.file, &b.test, b.line)));

    // Allowlist hygiene, mirroring dark_features: an entry that no longer
    // describes a real skip is stale and must go, and an entry without a reason
    // never justified anything.
    let present: BTreeSet<(String, String)> = findings
        .iter()
        .map(|f| (f.file.clone(), f.test.clone()))
        .collect();
    let mut stale_allowlist = Vec::new();
    let mut unjustified_allowlist = Vec::new();
    for entry in &allowlist.allow {
        let id = format!("{}::{}", entry.file, entry.test);
        if !present.contains(&(entry.file.clone(), entry.test.clone())) {
            stale_allowlist.push(id.clone());
        }
        if entry.reason.trim().is_empty() {
            unjustified_allowlist.push(format!("{id} (empty reason)"));
        } else if entry.kind == Kind::Tracked
            && entry.issue.as_deref().unwrap_or("").trim().is_empty()
        {
            unjustified_allowlist.push(format!("{id} (kind=tracked without an issue)"));
        }
    }

    Ok(Census {
        files_scanned: files.len(),
        tests_scanned,
        findings,
        stale_allowlist,
        unjustified_allowlist,
    })
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape this check exists for, copied from `ix-duck`'s golden tests:
    /// a `\` line continuation inside the message, then a guarded bare return.
    const DUCKDB_SHAPE: &str = r##"
#[test]
fn duckdb_cli_reproduces_the_frozen_golden_when_duckdb_is_installed() {
    let script = "crates/ix-duck/sql/pareto_frontier_golden.sql";
    let output = match Command::new("duckdb")
        .args(["-csv", "-c", &format!(".read {script}")])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!(
                "duckdb CLI not runnable ({error}); the SQL half of #294 was NOT checked by this \
                 run. Reproduce manually from the repo root:\n  \
                 duckdb -csv -c \".read {script}\""
            );
            return;
        }
    };
    assert!(output.status.success(), "duckdb exited badly");
    assert_eq!(produced, golden(), "the SQL surface and the golden disagree");
}
"##;

    #[test]
    fn blanking_preserves_length_and_every_newline() {
        // The regression that mattered: `\` continuations inside a string put a
        // real newline in the literal, and skipping the escape pair dropped it.
        // Every line number after the first such string was then too low -- a
        // finding in `ix-agent` was reported 40 lines before the test it was
        // in, which is how this was caught.
        let blanked = blank_noise(DUCKDB_SHAPE);
        assert_eq!(
            blanked.matches('\n').count(),
            DUCKDB_SHAPE.matches('\n').count(),
            "blanking lost a newline, so every later line number is wrong"
        );
        assert_eq!(blanked.len(), DUCKDB_SHAPE.len(), "offsets must be stable");
    }

    #[test]
    fn blanking_removes_code_shaped_text_from_comments_and_strings() {
        let src = "fn f() {\n    // return;\n    let s = \"return;\";\n}\n";
        let blanked = blank_noise(src);
        assert!(
            !blanked.contains("return;"),
            "a return in a comment or string must not be visible to the scanner"
        );
        assert_eq!(blanked.matches('\n').count(), src.matches('\n').count());
    }

    #[test]
    fn the_duckdb_shape_is_found_with_its_probe_and_line() {
        let (findings, tests) = findings_in("t.rs", DUCKDB_SHAPE);
        assert_eq!(tests, 1);
        assert_eq!(findings.len(), 1, "expected exactly one guarded return");
        let f = &findings[0];
        assert_eq!(
            f.test,
            "duckdb_cli_reproduces_the_frozen_golden_when_duckdb_is_installed"
        );
        assert_eq!(
            f.probe,
            Probe::MissingCommand,
            "a spawned command outranks the Err arm it arrives through"
        );
        assert_eq!(
            f.assertions_skipped, 2,
            "both asserts after the guard are bypassed"
        );
        // The `return;` sits on the line the source puts it on -- the property
        // the blanking regression broke.
        let expected = 1 + DUCKDB_SHAPE[..DUCKDB_SHAPE.find("return;").unwrap()]
            .matches('\n')
            .count();
        assert_eq!(f.line, expected, "line number must survive blanking");
    }

    #[test]
    fn an_unconditional_return_at_body_level_is_not_a_skip() {
        // Guard against the obvious false positive: a test that returns at the
        // end of its own body has not skipped anything.
        let src = "#[test]\nfn t() {\n    let p = q.exists();\n    assert!(p);\n    return;\n}\n";
        let (findings, tests) = findings_in("t.rs", src);
        assert_eq!(tests, 1);
        assert!(
            findings.is_empty(),
            "body-level return is how a test ends, not how it skips: {findings:?}"
        );
    }

    #[test]
    fn a_guarded_return_with_no_probe_is_not_reported() {
        // Deliberate under-reporting: an early return on ordinary logic is a
        // test's business, not this check's.
        let src =
            "#[test]\nfn t() {\n    if n > 3 {\n        return;\n    }\n    assert!(n < 4);\n}\n";
        let (findings, _) = findings_in("t.rs", src);
        assert!(
            findings.is_empty(),
            "no environment probe, no finding: {findings:?}"
        );
    }

    #[test]
    fn a_path_probe_that_skips_nothing_is_advisory_not_blocking() {
        let src = "#[test]\nfn t() {\n    if !p.exists() {\n        return;\n    }\n}\n";
        let (mut findings, _) = findings_in("t.rs", src);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].probe, Probe::MissingPath);
        assert_eq!(findings[0].assertions_skipped, 0);

        findings[0].allowed = false;
        let census = Census {
            files_scanned: 1,
            tests_scanned: 1,
            findings,
            stale_allowlist: vec![],
            unjustified_allowlist: vec![],
        };
        assert!(
            census.blocking().is_empty(),
            "nothing is bypassed, so nothing blocks"
        );
        assert_eq!(census.advisory().len(), 1);
    }

    #[test]
    fn allowlist_covers_live_skips_and_reports_stale_and_unjustified_entries() {
        let root = tempfile::tempdir().expect("temp dir");
        let tests = root.path().join("crates/demo/tests");
        std::fs::create_dir_all(&tests).expect("tests dir");
        std::fs::write(tests.join("golden.rs"), DUCKDB_SHAPE).expect("write test file");

        let entry = |test: &str, kind: Kind, reason: &str, issue: Option<&str>| AllowEntry {
            file: "crates/demo/tests/golden.rs".into(),
            test: test.into(),
            kind,
            reason: reason.into(),
            issue: issue.map(Into::into),
        };
        let live = "duckdb_cli_reproduces_the_frozen_golden_when_duckdb_is_installed";
        let allowlist = Allowlist {
            note: String::new(),
            allow: vec![
                entry(live, Kind::Tracked, "fail-closed CI job pending", None),
                entry(
                    "renamed_long_ago",
                    Kind::Environment,
                    "was a skip once",
                    None,
                ),
            ],
        };

        let census = scan_with_allowlist(root.path(), &allowlist).expect("scan");
        assert_eq!((census.files_scanned, census.tests_scanned), (1, 1));
        assert!(census.findings[0].allowed, "the live skip is covered");
        assert!(
            census.blocking().is_empty(),
            "a covered skip does not block"
        );
        assert_eq!(
            census.stale_allowlist,
            vec!["crates/demo/tests/golden.rs::renamed_long_ago".to_string()],
            "an entry naming no current skip is stale"
        );
        assert_eq!(
            census.unjustified_allowlist.len(),
            1,
            "tracked needs an issue"
        );
        assert!(census.unjustified_allowlist[0].contains(live));
    }

    #[test]
    fn an_unknown_kind_is_a_parse_error_not_a_silent_default() {
        let raw = r#"{"allow":[{"file":"a.rs","test":"t","kind":"trakced","reason":"r"}]}"#;
        assert!(serde_json::from_str::<Allowlist>(raw).is_err());
    }
}
