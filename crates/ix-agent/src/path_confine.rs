//! Confinement of caller-supplied filesystem paths.
//!
//! Tools classified `Read` (Tier 1) or `EditInProject` (Tier 2) in
//! `ix-approval` run without a prompt, so any path, directory or repo root a
//! caller names is resolved here before the tool reads, walks or writes it. A
//! path is accepted only when it lies under an allowed root:
//!
//! - the workspace root ([`workspace_root`]);
//! - each directory listed in `IX_EXTRA_ROOTS` (OS path-list syntax: `;` on
//!   Windows, `:` elsewhere), for operators whose MCP server must read another
//!   checkout such as a sibling repo;
//! - for the GA trace tools only, the trace locations the operator already
//!   chose ([`trace_roots`]).
//!
//! Defaults a tool picks for itself are not caller input and are not confined.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Environment variable naming extra directories auto-approved tools may use.
pub(crate) const EXTRA_ROOTS_ENV: &str = "IX_EXTRA_ROOTS";

/// The workspace root: `IX_ROOT`, else the repo root under `cargo run`/`cargo
/// test`, else the current directory.
pub(crate) fn workspace_root() -> PathBuf {
    if let Ok(root) = std::env::var("IX_ROOT") {
        return PathBuf::from(root);
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return Path::new(&manifest).join("../..");
    }
    PathBuf::from(".")
}

/// Directories from an `IX_EXTRA_ROOTS` value; empty entries are ignored.
fn extra_roots_from(value: Option<OsString>) -> Vec<PathBuf> {
    value
        .map(|v| {
            std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// `root` followed by the `IX_EXTRA_ROOTS` directories.
pub(crate) fn allowed_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![root.to_path_buf()];
    roots.extend(extra_roots_from(std::env::var_os(EXTRA_ROOTS_ENV)));
    roots
}

/// Trace locations chosen by the operator rather than the caller: the GA trace
/// directory (`~/.ga/traces`, the trace tools' default) and the `traces/`
/// directory beside the installed session log, where `ix_triage_session`
/// exports before it re-ingests.
pub(crate) fn trace_roots() -> Vec<PathBuf> {
    let mut roots = vec![ix_io::trace_bridge::default_trace_dir()];
    if let Some(dir) = crate::registry_bridge::current_session_log()
        .and_then(|log| log.path().parent().map(|p| p.join("traces")))
    {
        roots.push(dir);
    }
    roots
}

/// Resolve a caller-supplied path against `root` and refuse anything that is not
/// an existing path under `root` or an `IX_EXTRA_ROOTS` directory.
///
/// `..` is rejected lexically; everything else (absolute paths elsewhere,
/// symlinks or junctions leading out) is caught by canonicalizing and requiring
/// the result to sit under a canonical allowed root. A missing path and a path
/// outside get the same message, so the error does not reveal whether a file
/// exists elsewhere on disk, and nothing is read before the check. Returns the
/// joined (non-canonical) path so paths reported back stay readable.
pub(crate) fn confine(root: &Path, param: &str, raw: &str) -> Result<PathBuf, String> {
    confine_in(&allowed_roots(root), param, raw)
}

/// [`confine`] against an explicit root list. Relative paths resolve against
/// `roots[0]`, which must be accessible; other roots that do not exist are
/// skipped.
pub(crate) fn confine_in(roots: &[PathBuf], param: &str, raw: &str) -> Result<PathBuf, String> {
    let full = join_checked(roots, param, raw)?;
    let canonical_roots = canonical_roots(roots, param)?;
    match full.canonicalize() {
        Ok(resolved) if canonical_roots.iter().any(|r| resolved.starts_with(r)) => Ok(full),
        _ => Err(outside(param, raw)),
    }
}

/// Like [`confine_in`], for a directory the tool will create if it is missing:
/// the deepest existing ancestor is canonicalized (so a symlink or junction
/// leading out is still caught) and the missing tail, which holds no `..`, is
/// appended before the root check.
pub(crate) fn confine_dest_in(
    roots: &[PathBuf],
    param: &str,
    raw: &str,
) -> Result<PathBuf, String> {
    let full = join_checked(roots, param, raw)?;
    let canonical_roots = canonical_roots(roots, param)?;
    match resolve_existing_prefix(&full) {
        Some(resolved) if canonical_roots.iter().any(|r| resolved.starts_with(r)) => Ok(full),
        _ => Err(outside(param, raw)),
    }
}

/// serde's message can quote the offending value, which would echo file
/// contents back to the caller; report only where parsing failed.
pub(crate) fn parse_error(param: &str, raw: &str, e: &serde_json::Error) -> String {
    format!(
        "`{param}`: {raw} has the wrong shape ({:?} error at line {}, column {})",
        e.classify(),
        e.line(),
        e.column()
    )
}

fn join_checked(roots: &[PathBuf], param: &str, raw: &str) -> Result<PathBuf, String> {
    if Path::new(raw)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!("`{param}`: `..` is not allowed in {raw}"));
    }
    let base = roots
        .first()
        .ok_or_else(|| format!("`{param}`: no allowed root is configured"))?;
    Ok(base.join(raw))
}

fn canonical_roots(roots: &[PathBuf], param: &str) -> Result<Vec<PathBuf>, String> {
    let first = roots[0]
        .canonicalize()
        .map_err(|_| format!("`{param}`: the workspace root is not accessible"))?;
    let mut out = vec![first];
    out.extend(roots[1..].iter().filter_map(|r| resolve_existing_prefix(r)));
    Ok(out)
}

/// Canonicalize the deepest existing ancestor of `path` and re-append the
/// missing components. `None` if no ancestor exists or the tail has `..`.
fn resolve_existing_prefix(path: &Path) -> Option<PathBuf> {
    let mut existing = path;
    let mut tail = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut out = canonical;
            for part in tail.iter().rev() {
                out.push(part);
            }
            return Some(out);
        }
        let name = existing.file_name()?;
        tail.push(name.to_os_string());
        existing = existing.parent()?;
    }
}

fn outside(param: &str, raw: &str) -> String {
    format!("`{param}`: {raw} is not an existing path inside the workspace root")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    /// Symlinks need Developer Mode or elevation on Windows; a directory
    /// junction does not, and is the same escape.
    #[cfg(windows)]
    fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link).or_else(|_| {
            std::process::Command::new("cmd")
                .arg("/C")
                .arg("mklink")
                .arg("/J")
                .arg(link)
                .arg(target)
                .output()
                .and_then(|o| {
                    if o.status.success() {
                        Ok(())
                    } else {
                        Err(std::io::Error::other("mklink /J failed"))
                    }
                })
        })
    }

    fn one(root: &Path) -> Vec<PathBuf> {
        vec![root.to_path_buf()]
    }

    #[test]
    fn confine_accepts_existing_paths_inside_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("state")).unwrap();
        std::fs::write(root.path().join("state/log.jsonl"), "").unwrap();

        assert!(confine_in(&one(root.path()), "log", "state/log.jsonl").is_ok());
        assert!(confine_in(&one(root.path()), "workspace", "state").is_ok());
        let absolute = root.path().join("state/log.jsonl");
        assert!(confine_in(&one(root.path()), "log", absolute.to_str().unwrap()).is_ok());
    }

    #[test]
    fn confine_rejects_parent_dir_even_when_it_lands_inside() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("a")).unwrap();
        let err = confine_in(&one(root.path()), "workspace", "a/../a").unwrap_err();
        assert!(err.contains("`..` is not allowed"), "{err}");
        let err = confine_dest_in(&one(root.path()), "trace_dir", "a/../b").unwrap_err();
        assert!(err.contains("`..` is not allowed"), "{err}");
    }

    #[test]
    fn confine_gives_one_message_for_outside_and_missing() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let secret = elsewhere.path().join("id_ed25519");
        std::fs::write(&secret, "PRIVATE").unwrap();
        let missing = elsewhere.path().join("nope");

        let existing = secret.to_str().unwrap();
        let absent = missing.to_str().unwrap();
        let err_existing = confine_in(&one(root.path()), "log", existing).unwrap_err();
        let err_absent = confine_in(&one(root.path()), "log", absent).unwrap_err();
        assert_eq!(
            err_existing.replace(existing, "<p>"),
            err_absent.replace(absent, "<p>"),
            "the error must not reveal whether a file outside the root exists"
        );
        assert!(err_existing.contains("not an existing path inside the workspace root"));
    }

    #[test]
    fn confine_rejects_a_symlink_that_leads_out_of_the_root() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::write(elsewhere.path().join("secret.json"), "{}").unwrap();
        let link = root.path().join("link");
        // A symlink on Unix and a junction on Windows need no privileges, so a
        // failure here is a broken test environment, not a reason to skip.
        link_dir(elsewhere.path(), &link).expect("create a symlink or junction");
        assert!(
            link.join("secret.json").exists(),
            "precondition: link resolves"
        );
        for raw in ["link/secret.json", "link"] {
            let err = confine_in(&one(root.path()), "research", raw).unwrap_err();
            assert!(err.contains("not an existing path inside"), "{err}");
        }
        // A directory to be created under the link escapes the same way.
        let err = confine_dest_in(&one(root.path()), "trace_dir", "link/new/traces").unwrap_err();
        assert!(err.contains("not an existing path inside"), "{err}");
    }

    #[test]
    fn extra_roots_admit_their_own_tree_only() {
        let root = tempfile::tempdir().unwrap();
        let sibling = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(sibling.path().join("a.csv"), "x\n1\n").unwrap();
        std::fs::write(other.path().join("b.csv"), "x\n1\n").unwrap();
        let roots = vec![root.path().to_path_buf(), sibling.path().to_path_buf()];

        let allowed = sibling.path().join("a.csv");
        assert!(confine_in(&roots, "path", allowed.to_str().unwrap()).is_ok());
        let refused = other.path().join("b.csv");
        assert!(confine_in(&roots, "path", refused.to_str().unwrap()).is_err());
        // A missing extra root is skipped, not an error.
        let with_missing = vec![root.path().to_path_buf(), other.path().join("nope")];
        assert!(confine_in(&with_missing, "path", refused.to_str().unwrap()).is_err());
    }

    #[test]
    fn extra_roots_parse_the_os_path_list() {
        assert!(extra_roots_from(None).is_empty());
        let a = std::env::temp_dir().join("a");
        let b = std::env::temp_dir().join("b");
        let joined = std::env::join_paths([&a, &b]).unwrap();
        assert_eq!(extra_roots_from(Some(joined)), vec![a, b]);
    }

    #[test]
    fn dest_accepts_a_missing_directory_under_an_allowed_root() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        assert!(confine_dest_in(&one(root.path()), "trace_dir", "new/traces").is_ok());
        let out = elsewhere.path().join("new");
        assert!(confine_dest_in(&one(root.path()), "trace_dir", out.to_str().unwrap()).is_err());
        // A root that does not exist yet (e.g. ~/.ga/traces) still admits its tree.
        let future_root = elsewhere.path().join("ga/traces");
        let roots = vec![root.path().to_path_buf(), future_root.clone()];
        assert!(confine_dest_in(&roots, "trace_dir", future_root.to_str().unwrap()).is_ok());
    }

    #[test]
    fn parse_errors_do_not_echo_file_contents() {
        let e = serde_json::from_str::<Vec<u32>>(r#"["SECRET-VALUE"]"#).unwrap_err();
        assert!(
            e.to_string().contains("SECRET-VALUE"),
            "precondition: serde echoes it"
        );
        let msg = parse_error("research", "claims.json", &e);
        assert!(!msg.contains("SECRET-VALUE"), "{msg}");
        assert!(msg.contains("line 1"), "{msg}");
    }
}
