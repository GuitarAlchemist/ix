//! Milestone promotion — copy a run dir into a tracked milestones tree.
//!
//! Pipeline:
//! 1. Validate slug against regex `^[a-z0-9][a-z0-9-]{0,63}$`.
//! 2. Validate run-id as UUIDv7 string and resolve its dir under
//!    `<runs_root>/<run-id>/`.
//! 3. Sanitization pass: scan every JSONL line for redaction patterns
//!    (API keys, absolute paths). Abort on match.
//! 4. Copy `<runs_root>/<run-id>/` → `<milestones_root>/<slug>.tmp/`.
//! 5. Atomic rename `.tmp/` → final `<slug>/`. (Windows: `MoveFileEx`
//!    via `std::fs::rename`.)
//! 6. Write `.complete` sentinel **last**. `list` skips dirs without it.
//!
//! Slug collision: if `<slug>/` already exists, returns
//! [`AutoresearchError::MilestoneSlugCollision`] unless `force = true`.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::AutoresearchError;

/// Maximum slug length excluding the leading char.
const MAX_SLUG_TAIL: usize = 63;

/// Patterns that indicate a leak in promote sanitization. Adding to this
/// list does NOT bump the schema; it strengthens the safety net.
const REDACTION_PATTERNS: &[&str] = &[
    "sk-ant-",          // Anthropic API key prefix
    "Bearer ",          // Generic OAuth bearer
    "AKIA",             // AWS access key prefix
    "ghp_",             // GitHub personal access token
    "ghs_",             // GitHub server token
    "github_pat_",      // GitHub fine-grained token prefix
    "AIza",             // Google API key prefix
];

/// Validate a milestone slug. Slugs must:
///
/// - Start with `[a-z0-9]`
/// - Followed by 0–63 chars from `[a-z0-9-]`
///
/// Anything else (dots, slashes, backslashes, leading hyphens, uppercase,
/// underscores) is rejected. This is the primary defense against the
/// SEV-HIGH path-traversal finding from the security review.
pub fn validate_slug(slug: &str) -> Result<(), AutoresearchError> {
    if slug.is_empty() || slug.len() > MAX_SLUG_TAIL + 1 {
        return Err(AutoresearchError::InvalidSlug(slug.to_string()));
    }
    let mut chars = slug.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(AutoresearchError::InvalidSlug(slug.to_string()));
    }
    for c in chars {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            return Err(AutoresearchError::InvalidSlug(slug.to_string()));
        }
    }
    Ok(())
}

/// Validate a run-id as a UUIDv7 hyphenated string. Returns the canonical
/// form on success.
pub fn validate_run_id(run_id: &str) -> Result<String, AutoresearchError> {
    let parsed = uuid::Uuid::parse_str(run_id)
        .map_err(|_| AutoresearchError::InvalidRunId(run_id.to_string()))?;
    // We accept v7 specifically, but also v4 for testing flexibility.
    // Production runs use v7.
    let version = parsed.get_version_num();
    if !(version == 7 || version == 4) {
        return Err(AutoresearchError::InvalidRunId(format!(
            "{run_id} (UUID version {version}, expected 7 or 4)"
        )));
    }
    Ok(parsed.hyphenated().to_string())
}

/// Scan the contents of every line for redaction patterns. Aborts on
/// match. Pattern detection is byte-substring; we deliberately don't do
/// regex-with-context (false-negative is the worse failure mode here).
pub fn sanitize_text(text: &str) -> Result<(), AutoresearchError> {
    for pat in REDACTION_PATTERNS {
        if text.contains(pat) {
            return Err(AutoresearchError::PromoteSanitizationFailed {
                detail: format!("redaction pattern matched: {pat}"),
            });
        }
    }
    // Absolute paths are also a leak — naive but useful check.
    // Windows drive letter paths.
    let bytes = text.as_bytes();
    for w in bytes.windows(3) {
        if w[0].is_ascii_uppercase() && w[1] == b':' && (w[2] == b'\\' || w[2] == b'/') {
            return Err(AutoresearchError::PromoteSanitizationFailed {
                detail: "absolute Windows path detected".to_string(),
            });
        }
    }
    // POSIX-ish absolute paths.
    if text.contains("/Users/") || text.contains("/home/") {
        return Err(AutoresearchError::PromoteSanitizationFailed {
            detail: "absolute Unix path detected (/Users/ or /home/)".to_string(),
        });
    }
    Ok(())
}

/// Promote a run to a milestone.
///
/// `runs_root`: parent dir of run subdirs (e.g. `state/autoresearch/runs/`).
/// `milestones_root`: parent dir of milestone subdirs.
/// `run_id`: source run identifier (validated).
/// `slug`: target milestone name (validated).
/// `force`: overwrite existing milestone with the same slug.
///
/// On success, returns the resolved milestone dir path. On failure, no
/// partial state remains under `<milestones_root>/` (the `.tmp/` dir is
/// cleaned up on error).
pub fn promote_run(
    runs_root: &Path,
    milestones_root: &Path,
    run_id: &str,
    slug: &str,
    force: bool,
) -> Result<PathBuf, AutoresearchError> {
    validate_slug(slug)?;
    let canonical_run_id = validate_run_id(run_id)?;

    let src = runs_root.join(&canonical_run_id);
    if !src.is_dir() {
        return Err(AutoresearchError::InvalidRunId(format!(
            "{canonical_run_id} (no dir at {})",
            src.display()
        )));
    }

    fs::create_dir_all(milestones_root)?;
    let final_dst = milestones_root.join(slug);
    if final_dst.exists() && !force {
        return Err(AutoresearchError::MilestoneSlugCollision {
            slug: slug.to_string(),
        });
    }

    let tmp_dst = milestones_root.join(format!("{slug}.tmp"));
    if tmp_dst.exists() {
        fs::remove_dir_all(&tmp_dst)?;
    }
    fs::create_dir_all(&tmp_dst)?;

    // Recursive copy with sanitization on every text file.
    if let Err(e) = copy_with_sanitization(&src, &tmp_dst) {
        let _ = fs::remove_dir_all(&tmp_dst);
        return Err(e);
    }

    // If forcing, remove the existing destination *just before* the
    // rename so the failure window stays small.
    if final_dst.exists() && force {
        fs::remove_dir_all(&final_dst)?;
    }

    // Atomic-ish rename. On Windows this calls MoveFileEx which fails on
    // non-empty existing dirs (which we already removed above) and on
    // cross-volume moves (caller's responsibility — we document the
    // same-volume invariant in the plan's State Lifecycle Risks).
    fs::rename(&tmp_dst, &final_dst)?;

    // Sentinel: list/promote use this to distinguish complete from
    // half-built dirs.
    fs::write(final_dst.join(".complete"), b"ok\n")?;
    Ok(final_dst)
}

/// Recursive directory copy that runs every text file's content through
/// [`sanitize_text`]. Binary files are copied bytewise without scanning.
/// This is an `O(content)` pass, acceptable on a per-run-dir basis.
fn copy_with_sanitization(src: &Path, dst: &Path) -> Result<(), AutoresearchError> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_name = entry.file_name();
        let target = dst.join(&file_name);
        if entry_path.is_dir() {
            copy_with_sanitization(&entry_path, &target)?;
        } else {
            // Treat .json, .jsonl, .md, .txt, .yaml, .toml as text.
            let is_text = matches!(
                entry_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase()),
                Some(ref e) if matches!(e.as_str(), "json" | "jsonl" | "md" | "txt" | "yaml" | "yml" | "toml")
            );
            if is_text {
                let content = fs::read_to_string(&entry_path)?;
                sanitize_text(&content)?;
                fs::write(&target, content)?;
            } else {
                fs::copy(&entry_path, &target)?;
            }
        }
    }
    Ok(())
}

/// Returns true iff a milestone directory exists AND has the `.complete`
/// sentinel. Used by `list` to skip half-built dirs.
pub fn is_complete_milestone(milestone_dir: &Path) -> bool {
    milestone_dir.is_dir() && milestone_dir.join(".complete").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_run(runs_root: &Path, run_id: &str, log_content: &str) {
        let dir = runs_root.join(run_id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("log.jsonl"), log_content).unwrap();
    }

    #[test]
    fn validate_slug_rejects_path_traversal() {
        for bad in &[
            "../etc",
            "..",
            ".",
            "/abs",
            "\\abs",
            "back\\slash",
            "with/slash",
            "UPPER",
            "with space",
            "-leading-hyphen",
            "good_with_underscore",
            "",
        ] {
            assert!(
                validate_slug(bad).is_err(),
                "should have rejected slug {bad:?}"
            );
        }
    }

    #[test]
    fn validate_slug_accepts_canonical() {
        for good in &[
            "ok",
            "1",
            "first-overnight-tune",
            "2026-04-26-grammar-smoke",
        ] {
            assert!(
                validate_slug(good).is_ok(),
                "should have accepted slug {good:?}"
            );
        }
    }

    #[test]
    fn validate_run_id_accepts_v7() {
        let v7 = uuid::Uuid::now_v7().hyphenated().to_string();
        assert!(validate_run_id(&v7).is_ok());

        assert!(validate_run_id("../../etc").is_err());
        assert!(validate_run_id("not-a-uuid").is_err());
    }

    #[test]
    fn sanitize_catches_anthropic_key() {
        let bad = r#"{"error": "auth: Bearer sk-ant-api03-xxx"}"#;
        assert!(sanitize_text(bad).is_err());
    }

    #[test]
    fn sanitize_catches_windows_path() {
        let bad = r#"{"path": "C:\\Users\\spare\\corpus.json"}"#;
        assert!(sanitize_text(bad).is_err());
    }

    #[test]
    fn sanitize_catches_unix_user_path() {
        let bad = r#"{"path": "/home/spare/corpus.json"}"#;
        assert!(sanitize_text(bad).is_err());
    }

    #[test]
    fn sanitize_passes_clean_log() {
        let ok = r#"{"event":"iteration","iteration":1,"reward":0.5}"#;
        assert!(sanitize_text(ok).is_ok());
    }

    #[test]
    fn promote_writes_complete_sentinel() {
        let workspace = TempDir::new().unwrap();
        let runs_root = workspace.path().join("runs");
        let milestones_root = workspace.path().join("milestones");
        fs::create_dir_all(&runs_root).unwrap();

        let v7 = uuid::Uuid::now_v7().hyphenated().to_string();
        make_run(&runs_root, &v7, r#"{"event":"run_start","ok":true}"#);

        let dst = promote_run(&runs_root, &milestones_root, &v7, "first-tune", false).unwrap();
        assert!(dst.exists());
        assert!(is_complete_milestone(&dst));
        assert!(dst.join("log.jsonl").exists());
    }

    #[test]
    fn promote_rejects_slug_collision_without_force() {
        let workspace = TempDir::new().unwrap();
        let runs_root = workspace.path().join("runs");
        let milestones_root = workspace.path().join("milestones");
        fs::create_dir_all(&runs_root).unwrap();

        let v7a = uuid::Uuid::now_v7().hyphenated().to_string();
        // sleep 1ms to ensure v7b has a different timestamp
        std::thread::sleep(std::time::Duration::from_millis(2));
        let v7b = uuid::Uuid::now_v7().hyphenated().to_string();
        make_run(&runs_root, &v7a, r#"{"a":1}"#);
        make_run(&runs_root, &v7b, r#"{"b":2}"#);

        let _ = promote_run(&runs_root, &milestones_root, &v7a, "shared-slug", false).unwrap();

        // Same source, same slug = idempotent error (collision; force not asked).
        let same = promote_run(&runs_root, &milestones_root, &v7a, "shared-slug", false);
        assert!(matches!(
            same,
            Err(AutoresearchError::MilestoneSlugCollision { .. })
        ));

        // Different source, same slug = collision (without force).
        let diff = promote_run(&runs_root, &milestones_root, &v7b, "shared-slug", false);
        assert!(matches!(
            diff,
            Err(AutoresearchError::MilestoneSlugCollision { .. })
        ));

        // With --force, the second source wins.
        let forced = promote_run(&runs_root, &milestones_root, &v7b, "shared-slug", true).unwrap();
        let log = fs::read_to_string(forced.join("log.jsonl")).unwrap();
        assert!(log.contains(r#"{"b":2}"#));
    }

    #[test]
    fn promote_aborts_on_poisoned_log_and_leaves_no_partial_state() {
        let workspace = TempDir::new().unwrap();
        let runs_root = workspace.path().join("runs");
        let milestones_root = workspace.path().join("milestones");
        fs::create_dir_all(&runs_root).unwrap();

        let v7 = uuid::Uuid::now_v7().hyphenated().to_string();
        // Poisoned: contains a Bearer token.
        make_run(
            &runs_root,
            &v7,
            r#"{"error":"call: Bearer sk-ant-api03-leak"}"#,
        );

        let res = promote_run(&runs_root, &milestones_root, &v7, "leaky", false);
        assert!(matches!(
            res,
            Err(AutoresearchError::PromoteSanitizationFailed { .. })
        ));
        // No partial state.
        assert!(!milestones_root.join("leaky.tmp").exists());
        assert!(!milestones_root.join("leaky").exists());
    }
}
