//! `ix check <noun>` — validation + governance + environment diagnostics.

use crate::exit;
use crate::output::{self, Format};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The registry inventory snapshot checked into git at `state/registry/skills.snapshot.json`
/// (path overridable via `IX_REGISTRY_SNAPSHOT`, mirroring `IX_GOVERNANCE_DIR`).
///
/// This replaces a hand-edited "expected minimum" count: the snapshot is
/// generated *from* the live registry (`--write-snapshot`), not typed by hand,
/// so a PR that adds/removes a skill shows up as a reviewable diff to this
/// file instead of a silently-drifting magic number.
#[derive(Debug, Serialize, Deserialize)]
struct RegistrySnapshot {
    count: usize,
    names: Vec<String>,
}

impl RegistrySnapshot {
    fn from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut names: Vec<String> = names.into_iter().map(str::to_string).collect();
        names.sort();
        RegistrySnapshot {
            count: names.len(),
            names,
        }
    }
}

fn registry_snapshot_path() -> PathBuf {
    std::env::var("IX_REGISTRY_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("state/registry/skills.snapshot.json"))
}

/// Difference between the live registry and the committed snapshot.
#[derive(Debug, Default, PartialEq, Eq)]
struct RegistryDiff {
    added: Vec<String>,
    removed: Vec<String>,
}

impl RegistryDiff {
    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Pure diff between the live skill names and a snapshot's names. Kept free of
/// I/O and the live `ix_registry` so it stays deterministic and fast to test
/// regardless of how many skills the workspace currently has.
fn diff_registry<'a>(current: impl IntoIterator<Item = &'a str>, snapshot: &RegistrySnapshot) -> RegistryDiff {
    let current: BTreeSet<&str> = current.into_iter().collect();
    let snapshot_names: BTreeSet<&str> = snapshot.names.iter().map(String::as_str).collect();
    RegistryDiff {
        added: current
            .difference(&snapshot_names)
            .map(|s| s.to_string())
            .collect(),
        removed: snapshot_names
            .difference(&current)
            .map(|s| s.to_string())
            .collect(),
    }
}

/// Write the current live registry to the snapshot path, creating parent
/// directories as needed. Used by `ix check doctor --write-snapshot` to
/// intentionally accept an inventory change.
fn write_snapshot(path: &Path, names: &[&str]) -> Result<RegistrySnapshot, String> {
    let snapshot = RegistrySnapshot::from_names(names.iter().copied());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(&snapshot).map_err(|e| format!("{e}"))?;
    std::fs::write(path, json + "\n").map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(snapshot)
}

/// One check result for the skill-inventory drift gate, built from the
/// snapshot file state and the live/snapshot diff. Separated from `doctor`
/// so the message-construction logic is unit-testable without a filesystem.
fn registry_check(
    path: &Path,
    snapshot_exists: bool,
    snapshot: Option<&RegistrySnapshot>,
    live_count: usize,
    diff: &RegistryDiff,
) -> Value {
    if !snapshot_exists {
        return json!({
            "check": "skill-inventory",
            "status": "fail",
            "skills": live_count,
            "message": format!(
                "no skill-inventory snapshot at {}; run `cargo run -p ix-skill -- check doctor --write-snapshot` to create one",
                path.display(),
            ),
        });
    }
    let Some(snapshot) = snapshot else {
        return json!({
            "check": "skill-inventory",
            "status": "fail",
            "skills": live_count,
            "message": format!("{} is not valid JSON — see stderr for the parse error", path.display()),
        });
    };
    if diff.is_empty() {
        return json!({
            "check": "skill-inventory",
            "status": "ok",
            "skills": live_count,
        });
    }
    json!({
        "check": "skill-inventory",
        "status": "fail",
        "skills": live_count,
        "snapshot_skills": snapshot.count,
        "added": diff.added,
        "removed": diff.removed,
        "message": format!(
            "skill inventory drifted from {} ({} added, {} removed) — if intentional, re-run with `--write-snapshot` and commit the updated snapshot",
            path.display(),
            diff.added.len(),
            diff.removed.len(),
        ),
    })
}

/// Environment self-diagnosis: rust toolchain, governance submodule, registry
/// health, key directories. Returns OK_TRUE on full green, PROBABLE on
/// non-fatal warnings, DOUBTFUL on missing optional pieces, FALSE on broken.
///
/// When `write_snapshot` is set, (re)writes `state/registry/skills.snapshot.json`
/// from the live registry instead of checking against it — the intentional
/// escape hatch for accepting a reviewed skill-inventory change.
pub fn doctor(format: Format, write_snapshot_flag: bool) -> Result<i32, String> {
    let mut checks: Vec<Value> = Vec::new();
    let mut any_fail = false;
    let mut any_warn = false;

    let live_names: Vec<&str> = ix_registry::all().map(|s| s.name).collect();
    let snapshot_path = registry_snapshot_path();

    if write_snapshot_flag {
        let snapshot = write_snapshot(&snapshot_path, &live_names)?;
        checks.push(json!({
            "check": "skill-inventory",
            "status": "written",
            "skills": snapshot.count,
            "path": snapshot_path.display().to_string(),
        }));
    } else {
        let snapshot_exists = snapshot_path.is_file();
        let snapshot: Option<RegistrySnapshot> = if snapshot_exists {
            let raw = std::fs::read_to_string(&snapshot_path)
                .map_err(|e| format!("reading {}: {e}", snapshot_path.display()))?;
            serde_json::from_str(&raw).ok()
        } else {
            None
        };
        let diff = snapshot
            .as_ref()
            .map(|s| diff_registry(live_names.iter().copied(), s))
            .unwrap_or_default();
        let check = registry_check(
            &snapshot_path,
            snapshot_exists,
            snapshot.as_ref(),
            live_names.len(),
            &diff,
        );
        if check["status"] != "ok" {
            any_fail = true;
        }
        checks.push(check);
    }

    // Governance submodule presence
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let gov_ok = std::path::Path::new(&gov_dir).is_dir();
    if !gov_ok {
        any_warn = true;
    }
    checks.push(json!({
        "check": "demerzel-governance",
        "status": if gov_ok { "ok" } else { "warn" },
        "path": gov_dir,
    }));

    // Constitution
    let constitution_path = format!("{gov_dir}/constitutions/default.constitution.md");
    let const_ok = std::path::Path::new(&constitution_path).is_file();
    if gov_ok && !const_ok {
        any_warn = true;
    }
    checks.push(json!({
        "check": "default-constitution",
        "status": if const_ok { "ok" } else if gov_ok { "warn" } else { "skip" },
        "path": constitution_path,
    }));

    // State directory (optional)
    let state_ok = std::path::Path::new("state").is_dir();
    checks.push(json!({
        "check": "state-directory",
        "status": if state_ok { "ok" } else { "absent" },
        "path": "state/",
    }));

    let verdict = if any_fail {
        "F"
    } else if any_warn {
        "P"
    } else {
        "T"
    };
    let exit_code = match verdict {
        "T" => exit::OK_TRUE,
        "P" => exit::PROBABLE,
        "F" => exit::FALSE,
        _ => exit::UNKNOWN,
    };

    let payload = json!({
        "verdict": verdict,
        "exit_code": exit_code,
        "checks": checks,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(exit_code)
}

/// Check a proposed action against the Demerzel constitution. Returns a
/// hexavalent-friendly exit code based on compliance.
pub fn action(action_text: &str, _context: Option<&str>, format: Format) -> Result<i32, String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");

    let constitution = ix_governance::Constitution::load(std::path::Path::new(&const_path))
        .map_err(|e| format!("loading {const_path}: {e}"))?;

    // Simple substring/keyword semantic scan over article texts.
    let action_lower = action_text.to_lowercase();
    let mut relevant: Vec<(u8, String)> = Vec::new();
    for art in &constitution.articles {
        // Relevance: any keyword from the article name appears in the action.
        for word in art.name.to_lowercase().split_whitespace() {
            if word.len() > 3 && action_lower.contains(word) {
                relevant.push((art.number, art.name.clone()));
                break;
            }
        }
    }

    // Heuristic verdict: no relevant articles hit → T (no constraint fired).
    // Relevant hits → P (probable compliance, review). Keywords like
    // "delete", "drop", "rm -rf", "force-push" → D (doubtful).
    let danger_words = [
        "delete",
        "drop table",
        "rm -rf",
        "force push",
        "--force",
        "truncate",
    ];
    let dangerous = danger_words.iter().any(|w| action_lower.contains(w));

    let verdict = if dangerous {
        "D"
    } else if !relevant.is_empty() {
        "P"
    } else {
        "T"
    };
    let exit_code = match verdict {
        "T" => exit::OK_TRUE,
        "P" => exit::PROBABLE,
        "D" => exit::DOUBTFUL,
        _ => exit::UNKNOWN,
    };

    let payload = json!({
        "verdict": verdict,
        "exit_code": exit_code,
        "action": action_text,
        "relevant_articles": relevant
            .iter()
            .map(|(n, name)| json!({ "number": n, "name": name }))
            .collect::<Vec<_>>(),
        "dangerous_keywords_matched": dangerous,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(exit_code)
}

#[cfg(test)]
mod registry_drift_tests {
    use super::*;

    #[test]
    fn diff_is_empty_when_names_match_regardless_of_order() {
        let snapshot = RegistrySnapshot::from_names(["b.skill", "a.skill"]);
        let diff = diff_registry(["a.skill", "b.skill"], &snapshot);
        assert!(diff.is_empty(), "{diff:?}");
    }

    #[test]
    fn diff_reports_added_skill() {
        let snapshot = RegistrySnapshot::from_names(["a.skill"]);
        let diff = diff_registry(["a.skill", "b.skill"], &snapshot);
        assert_eq!(diff.added, vec!["b.skill".to_string()]);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn diff_reports_removed_skill() {
        let snapshot = RegistrySnapshot::from_names(["a.skill", "b.skill"]);
        let diff = diff_registry(["a.skill"], &snapshot);
        assert_eq!(diff.removed, vec!["b.skill".to_string()]);
        assert!(diff.added.is_empty());
    }

    #[test]
    fn diff_reports_both_added_and_removed_in_a_rename() {
        let snapshot = RegistrySnapshot::from_names(["old.name"]);
        let diff = diff_registry(["new.name"], &snapshot);
        assert_eq!(diff.added, vec!["new.name".to_string()]);
        assert_eq!(diff.removed, vec!["old.name".to_string()]);
    }

    #[test]
    fn registry_check_fails_actionably_when_snapshot_missing() {
        let path = PathBuf::from("state/registry/skills.snapshot.json");
        let result = registry_check(&path, false, None, 5, &RegistryDiff::default());
        assert_eq!(result["status"], "fail");
        let message = result["message"].as_str().unwrap();
        assert!(
            message.contains("--write-snapshot"),
            "message should name the fix: {message}"
        );
    }

    #[test]
    fn registry_check_fails_actionably_on_drift() {
        let path = PathBuf::from("state/registry/skills.snapshot.json");
        let snapshot = RegistrySnapshot::from_names(["a.skill"]);
        let diff = RegistryDiff {
            added: vec!["b.skill".to_string()],
            removed: vec![],
        };
        let result = registry_check(&path, true, Some(&snapshot), 2, &diff);
        assert_eq!(result["status"], "fail");
        assert_eq!(result["added"], json!(["b.skill"]));
        let message = result["message"].as_str().unwrap();
        assert!(
            message.contains("1 added, 0 removed"),
            "message should quantify the drift: {message}"
        );
        assert!(
            message.contains("--write-snapshot"),
            "message should name the fix: {message}"
        );
    }

    #[test]
    fn registry_check_ok_when_no_drift() {
        let path = PathBuf::from("state/registry/skills.snapshot.json");
        let snapshot = RegistrySnapshot::from_names(["a.skill"]);
        let result = registry_check(&path, true, Some(&snapshot), 1, &RegistryDiff::default());
        assert_eq!(result["status"], "ok");
    }

    #[test]
    fn write_snapshot_round_trips_through_diff_registry() {
        let dir = std::env::temp_dir().join(format!(
            "ix-skill-registry-snapshot-test-{:?}",
            std::thread::current().id()
        ));
        let path = dir.join("skills.snapshot.json");
        let names = ["a.skill", "b.skill"];
        let snapshot = write_snapshot(&path, &names).expect("write succeeds");
        assert_eq!(snapshot.count, 2);

        let raw = std::fs::read_to_string(&path).expect("snapshot file exists");
        let reloaded: RegistrySnapshot = serde_json::from_str(&raw).expect("valid json");
        let diff = diff_registry(names, &reloaded);
        assert!(diff.is_empty(), "{diff:?}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
