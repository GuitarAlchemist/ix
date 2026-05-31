//! Drift detection — does each `@ai:` claim still match the code it points at?
//!
//! The loop runner ([`crate::AssumptionGraph::revise`]) tracks the *temporal*
//! axis: whether a claim's **stated** verdict changed between runs. It trusts
//! the annotation text, so it cannot see the failure mode this module targets —
//! the code under a claim changing while the claim does not, or a claim that
//! advertises a test binding (`src:test_x`) whose test has been deleted.
//!
//! Two mechanisms, both deliberately executable rather than judgement-based
//! (an LLM panel confirms well but catches invalidity poorly, so it must never
//! be the drift oracle):
//!
//! 1. **Content identity + anchor hash.** A claim's identity is
//!    `sha256(path | kind | normalized-claim)` — independent of line number, so
//!    moving code is a `moved`, not a spurious remove+add. Alongside it we store
//!    a hash of the *code line the claim annotates*; when that hash changes the
//!    code under the claim drifted and the claim must be re-confirmed.
//! 2. **Binding verification.** If a claim cites a test (`src:…::tests::name`
//!    or `src:test_name`), that test function must still exist in the workspace.
//!    A `[T:test]` whose test vanished is no longer `T`.
//!
//! [`snapshot`] captures the current state; [`diff`] + [`verify_bindings`]
//! compare a later snapshot against it. `span_drifted` and broken bindings are
//! hard failures (the claim may now be lying); `moved` / `verdict_changed` /
//! `added` are informational (`verdict_changed` is healthy belief revision).

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::graph::{normalize_claim, production_annotations, BuildError};

/// One claim, captured for drift comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimRecord {
    /// Stable, line-independent identity: `sha256(path | kind | normalized claim)`.
    pub key: String,
    pub path: String,
    pub line: u32,
    pub kind: String,
    pub claim: String,
    pub verdict: String,
    pub certainty: String,
    pub evidence: Option<String>,
    /// `sha256` of the (whitespace-normalized) code line the claim annotates.
    /// Empty if no code line follows or the file is unreadable.
    pub anchor_hash: String,
    /// Test function the claim is bound to, parsed from its evidence, if any.
    pub test_binding: Option<String>,
}

/// A captured set of claims (the on-disk snapshot).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Snapshot {
    pub schema: u32,
    pub claims: Vec<ClaimRecord>,
}

/// A single drift finding.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DriftItem {
    pub key: String,
    pub path: String,
    pub line: u32,
    pub claim: String,
    pub detail: String,
}

/// The full comparison between two snapshots.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DriftReport {
    pub unchanged: usize,
    /// New claims not in the baseline.
    pub added: Vec<DriftItem>,
    /// Claims gone from the baseline (code/annotation removed).
    pub removed: Vec<DriftItem>,
    /// Same claim+anchor, different line — code shifted, claim still valid.
    pub moved: Vec<DriftItem>,
    /// **Hard fail:** the annotated code line changed — claim must be re-confirmed.
    pub span_drifted: Vec<DriftItem>,
    /// Healthy belief revision: verdict/certainty changed.
    pub verdict_changed: Vec<DriftItem>,
    /// **Hard fail:** a claim cites a test that no longer exists.
    pub broken_bindings: Vec<DriftItem>,
}

impl DriftReport {
    /// True when nothing requires human re-confirmation (no span drift, no
    /// broken bindings). `moved` / `added` / `verdict_changed` are allowed.
    pub fn is_clean(&self) -> bool {
        self.span_drifted.is_empty() && self.broken_bindings.is_empty()
    }
}

/// Capture the workspace's production `@ai:` claims for drift comparison.
pub fn snapshot(workspace: &Path) -> Result<Snapshot, BuildError> {
    let claims = production_annotations(workspace)?
        .into_iter()
        .map(|a| {
            let path = a.location.path.replace('\\', "/");
            let kind = crate::node::NodeKind::from(a.kind);
            let key = stable_key(&path, kind.as_str(), &a.claim);
            ClaimRecord {
                key,
                anchor_hash: anchor_hash(workspace, &path, a.location.line_end),
                test_binding: a.source.evidence.as_deref().and_then(extract_test_name),
                line: a.location.line_start,
                kind: kind.as_str().to_string(),
                claim: a.claim,
                verdict: a.truth_value.as_str().to_string(),
                certainty: format!("{:?}", a.certainty),
                evidence: a.source.evidence,
                path,
            }
        })
        .collect();
    Ok(Snapshot { schema: 1, claims })
}

/// Compare a fresh snapshot (`new`) against a baseline (`old`).
pub fn diff(old: &Snapshot, new: &Snapshot) -> DriftReport {
    let mut report = DriftReport::default();
    let new_by_key: std::collections::BTreeMap<&str, &ClaimRecord> =
        new.claims.iter().map(|c| (c.key.as_str(), c)).collect();
    let old_by_key: std::collections::BTreeMap<&str, &ClaimRecord> =
        old.claims.iter().map(|c| (c.key.as_str(), c)).collect();

    for c in &new.claims {
        match old_by_key.get(c.key.as_str()) {
            None => report.added.push(item(c, "new claim")),
            Some(o) => {
                let mut touched = false;
                if o.anchor_hash != c.anchor_hash {
                    report.span_drifted.push(item(
                        c,
                        &format!(
                            "annotated code changed (anchor {} -> {})",
                            short(&o.anchor_hash),
                            short(&c.anchor_hash)
                        ),
                    ));
                    touched = true;
                }
                if o.verdict != c.verdict || o.certainty != c.certainty {
                    report.verdict_changed.push(item(
                        c,
                        &format!(
                            "{}/{} -> {}/{}",
                            o.verdict, o.certainty, c.verdict, c.certainty
                        ),
                    ));
                    touched = true;
                }
                if o.line != c.line && o.anchor_hash == c.anchor_hash {
                    report
                        .moved
                        .push(item(c, &format!("line {} -> {}", o.line, c.line)));
                    touched = true;
                }
                if !touched {
                    report.unchanged += 1;
                }
            }
        }
    }
    for c in &old.claims {
        if !new_by_key.contains_key(c.key.as_str()) {
            report.removed.push(item(c, "claim no longer present"));
        }
    }
    report
}

/// Check every test-bound claim's test still exists in the workspace, appending
/// `broken_bindings` to `report`.
pub fn verify_bindings(workspace: &Path, snap: &Snapshot, report: &mut DriftReport) {
    let bound: Vec<&ClaimRecord> = snap
        .claims
        .iter()
        .filter(|c| c.test_binding.is_some())
        .collect();
    if bound.is_empty() {
        return;
    }
    let fns = collect_fn_names(workspace);
    for c in bound {
        let name = c.test_binding.as_deref().unwrap();
        if !fns.contains(name) {
            report.broken_bindings.push(item(
                c,
                &format!("cited test `{name}` not found in workspace"),
            ));
        }
    }
}

// --- helpers -------------------------------------------------------------

fn item(c: &ClaimRecord, detail: &str) -> DriftItem {
    DriftItem {
        key: c.key.clone(),
        path: c.path.clone(),
        line: c.line,
        claim: c.claim.clone(),
        detail: detail.to_string(),
    }
}

fn stable_key(path: &str, kind: &str, claim: &str) -> String {
    let norm = normalize_claim(claim);
    format!(
        "sha256:{:x}",
        Sha256::digest(format!("{path}|{kind}|{norm}").as_bytes())
    )
}

fn short(h: &str) -> String {
    h.strip_prefix("sha256:")
        .unwrap_or(h)
        .chars()
        .take(8)
        .collect()
}

/// Hash the first real code line after `after_line` (1-based) — the line the
/// claim annotates. Skips blank lines and comments (including chained `@ai:`).
fn anchor_hash(workspace: &Path, path: &str, after_line: u32) -> String {
    let Ok(text) = std::fs::read_to_string(workspace.join(path)) else {
        return String::new();
    };
    for line in text.lines().skip(after_line as usize) {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') {
            continue;
        }
        let norm = t.split_whitespace().collect::<Vec<_>>().join(" ");
        return format!("sha256:{:x}", Sha256::digest(norm.as_bytes()));
    }
    String::new()
}

/// Parse a test function name out of an evidence string:
/// `src:foo.rs::tests::my_test` -> `my_test`; `src:test_thing` -> `test_thing`.
/// Returns `None` for line refs like `src:lib.rs:328`.
fn extract_test_name(ev: &str) -> Option<String> {
    let ev = ev.trim();
    if ev.contains("::") {
        let tail = ev.rsplit("::").next().unwrap_or("");
        if is_ident(tail) {
            return Some(tail.to_string());
        }
    }
    ev.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|tok| tok.starts_with("test_") && tok.len() > 5)
        .map(|s| s.to_string())
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Collect every `fn <name>` defined under `root` (`.rs` files), recursively.
fn collect_fn_names(root: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if p.is_dir() {
                if name != "target" && name != ".git" {
                    stack.push(p);
                }
            } else if name.ends_with(".rs") {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    scan_fn_names(&text, &mut out);
                }
            }
        }
    }
    out
}

fn scan_fn_names(text: &str, out: &mut HashSet<String>) {
    for line in text.lines() {
        let mut rest = line;
        while let Some(pos) = rest.find("fn ") {
            // require a word boundary before `fn`
            let before_ok = pos == 0
                || !rest[..pos]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let after = &rest[pos + 3..];
            if before_ok {
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    out.insert(name);
                }
            }
            rest = &rest[pos + 3..];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(key: &str, anchor: &str, verdict: &str, line: u32) -> ClaimRecord {
        ClaimRecord {
            key: key.to_string(),
            path: "crates/x/src/lib.rs".to_string(),
            line,
            kind: "invariant".to_string(),
            claim: "c".to_string(),
            verdict: verdict.to_string(),
            certainty: "Test".to_string(),
            evidence: None,
            anchor_hash: anchor.to_string(),
            test_binding: None,
        }
    }

    #[test]
    fn extract_test_name_handles_both_forms() {
        assert_eq!(
            extract_test_name("opinion.rs::tests::masses_sum_to_one").as_deref(),
            Some("masses_sum_to_one")
        );
        assert_eq!(
            extract_test_name("test_schema_hash_mismatch").as_deref(),
            Some("test_schema_hash_mismatch")
        );
        assert_eq!(extract_test_name("lib.rs:328"), None);
    }

    #[test]
    fn scan_fn_names_finds_definitions_only() {
        let mut s = HashSet::new();
        scan_fn_names("fn alpha() {}\n    pub fn beta() {}\n// transform(x)\n", &mut s);
        assert!(s.contains("alpha"));
        assert!(s.contains("beta"));
        assert!(!s.contains("transform"));
    }

    #[test]
    fn diff_flags_span_drift_but_not_a_clean_move() {
        let old = Snapshot {
            schema: 1,
            claims: vec![rec("k1", "sha256:aaaa", "T", 10), rec("k2", "sha256:bbbb", "P", 20)],
        };
        let new = Snapshot {
            schema: 1,
            // k1: code under it changed (anchor differs) = span drift.
            // k2: only the line moved (anchor same) = harmless move.
            claims: vec![rec("k1", "sha256:cccc", "T", 10), rec("k2", "sha256:bbbb", "P", 25)],
        };
        let report = diff(&old, &new);
        assert_eq!(report.span_drifted.len(), 1);
        assert_eq!(report.span_drifted[0].key, "k1");
        assert_eq!(report.moved.len(), 1);
        assert_eq!(report.moved[0].key, "k2");
        assert!(!report.is_clean(), "span drift must fail the gate");
    }

    #[test]
    fn diff_treats_verdict_change_as_clean_revision() {
        let old = Snapshot { schema: 1, claims: vec![rec("k", "sha256:aa", "P", 1)] };
        let new = Snapshot { schema: 1, claims: vec![rec("k", "sha256:aa", "T", 1)] };
        let report = diff(&old, &new);
        assert_eq!(report.verdict_changed.len(), 1);
        assert!(report.is_clean(), "verdict revision alone is healthy, not drift");
    }

    #[test]
    fn diff_reports_added_and_removed() {
        let old = Snapshot { schema: 1, claims: vec![rec("gone", "sha256:aa", "T", 1)] };
        let new = Snapshot { schema: 1, claims: vec![rec("fresh", "sha256:bb", "T", 1)] };
        let report = diff(&old, &new);
        assert_eq!(report.added.len(), 1);
        assert_eq!(report.removed.len(), 1);
    }
}
