//! Confinement of caller-supplied trace destinations.
//!
//! `ix_session_flywheel_export` is classified `EditInProject` (Tier 2) in
//! `ix-approval` and runs without a prompt, so the directory it writes to is
//! resolved here first. A destination is accepted only inside the trace
//! locations the operator already chose ([`trace_roots`]).
//!
//! Nothing touches the filesystem until the raw argument passes
//! [`check_lexical`]: `..`, NUL, and on Windows anything but a relative path or
//! a drive-letter absolute path (UNC `\\server\share`, device `\\.\`, verbatim
//! `\\?\`, drive-relative `C:x`, rooted `\x`) are refused first, so the check
//! itself never opens a network share or a named pipe. The deepest existing
//! ancestor is then canonicalized and must sit under a canonical allowed root.
//!
//! This is the trace-destination subset of the wider path confinement in the
//! stacked confinement PRs; the helpers keep the same names and semantics so
//! that module replaces this one as is.
//!
//! Known limit, needing local write access inside a root: a symlink or
//! junction swapped into an already-checked path before the tool opens it is
//! followed.
//!
//! Defaults a tool picks for itself are not caller input and are not confined.

use std::path::{Component, Path, PathBuf};

/// Trace locations chosen by the operator rather than the caller: the GA trace
/// directory (`~/.ga/traces`, the trace tools' default) and the `traces/`
/// directory beside the installed session log, where `ix_triage_session`
/// exports before it re-ingests.
///
/// Without `HOME` or `USERPROFILE` the default trace directory would be
/// relative to the process's current directory, which no operator chose, so it
/// is left out.
pub(crate) fn trace_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Some(ix_io::trace_bridge::default_trace_dir())
        .filter(|dir| dir.is_absolute())
        .into_iter()
        .collect();
    if let Some(dir) = crate::registry_bridge::current_session_log()
        .and_then(|log| log.path().parent().map(|p| p.join("traces")))
    {
        roots.push(dir);
    }
    roots
}

/// Confine a caller-supplied directory the tool will create if it is missing.
/// Relative paths resolve against `roots[0]`. After [`check_lexical`], the
/// deepest existing ancestor is canonicalized (so a symlink or junction leading
/// out is still caught) and the missing tail, which holds no `..`, is appended
/// before the root check. An ancestor that exists but does not resolve, such as
/// a dangling link, is refused rather than skipped.
pub(crate) fn confine_dest_in(
    roots: &[PathBuf],
    param: &str,
    raw: &str,
) -> Result<PathBuf, String> {
    let full = join_checked(roots, param, raw)?;
    let canonical_roots = canonical_roots(roots, param)?;
    match resolve_existing_prefix(&full) {
        Some(resolved) if canonical_roots.iter().any(|r| resolved.starts_with(r)) => {
            Ok(simplify(resolved))
        }
        _ => Err(format!(
            "`{param}`: {raw} is not inside an allowed destination root"
        )),
    }
}

/// Checks that need no filesystem access: no NUL, no `..`, and on Windows only
/// a relative path or a drive-letter absolute path. Opening a UNC or WebDAV
/// path connects to the server with the user's credentials, and `\\.\pipe\x`
/// opens a pipe client, so those are refused before anything is resolved.
pub(crate) fn check_lexical(param: &str, raw: &str) -> Result<(), String> {
    if raw.contains('\0') {
        return Err(format!("`{param}`: NUL is not allowed in a path"));
    }
    let path = Path::new(raw);
    if !is_local_shape(path) {
        return Err(format!(
            "`{param}`: {raw} must be a relative path or an absolute path on a local drive \
             (UNC, device, verbatim, drive-relative and driveless rooted paths are refused)"
        ));
    }
    if path
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!("`{param}`: `..` is not allowed in {raw}"));
    }
    Ok(())
}

#[cfg(windows)]
fn is_local_shape(path: &Path) -> bool {
    use std::path::Prefix;
    let mut components = path.components();
    match components.next() {
        Some(Component::Prefix(prefix)) => {
            matches!(prefix.kind(), Prefix::Disk(_))
                && components.next() == Some(Component::RootDir)
        }
        // `\x` names no drive, and Win32 reads `\??\UNC\host\share` as an NT path.
        Some(Component::RootDir) => false,
        _ => true,
    }
}

#[cfg(not(windows))]
fn is_local_shape(_path: &Path) -> bool {
    true
}

fn join_checked(roots: &[PathBuf], param: &str, raw: &str) -> Result<PathBuf, String> {
    check_lexical(param, raw)?;
    let base = roots
        .first()
        .ok_or_else(|| format!("`{param}`: no allowed root is configured"))?;
    Ok(base.join(raw))
}

fn canonical_roots(roots: &[PathBuf], param: &str) -> Result<Vec<PathBuf>, String> {
    let out: Vec<PathBuf> = roots
        .iter()
        .filter_map(|r| resolve_existing_prefix(r))
        .collect();
    if out.is_empty() {
        return Err(format!("`{param}`: no allowed root is accessible"));
    }
    Ok(out)
}

/// Canonicalize the deepest existing ancestor of `path` and re-append the
/// missing components. `None` if no ancestor exists, or if an entry exists but
/// does not resolve (a dangling symlink or junction could point anywhere).
fn resolve_existing_prefix(path: &Path) -> Option<PathBuf> {
    let mut existing = path;
    let mut tail = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(canonical) => {
                let mut out = canonical;
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return Some(out);
            }
            Err(_) if std::fs::symlink_metadata(existing).is_ok() => return None,
            Err(_) => {}
        }
        let name = existing.file_name()?;
        tail.push(name.to_os_string());
        existing = existing.parent()?;
    }
}

/// `canonicalize` returns verbatim `\\?\C:\...` paths on Windows. Hand tools the
/// plain `C:\...` form when it names the same file, so paths they report stay
/// readable and can be passed back in (verbatim paths are refused as input).
#[cfg(windows)]
fn simplify(path: PathBuf) -> PathBuf {
    use std::path::Prefix;
    let plain = match path.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(rest) => PathBuf::from(rest),
        None => return path,
    };
    let same_file = plain.as_os_str().len() < 260
        && matches!(
            plain.components().next(),
            Some(Component::Prefix(p)) if matches!(p.kind(), Prefix::Disk(_))
        )
        && plain.components().all(|c| match c {
            Component::Normal(name) => name.to_str().is_some_and(|n| !n.ends_with(['.', ' '])),
            _ => true,
        });
    if same_file {
        plain
    } else {
        path
    }
}

#[cfg(not(windows))]
fn simplify(path: PathBuf) -> PathBuf {
    path
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
    fn confine_rejects_parent_dir_even_when_it_lands_inside() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("a")).unwrap();
        let err = confine_dest_in(&one(root.path()), "trace_dir", "a/../b").unwrap_err();
        assert!(err.contains("`..` is not allowed"), "{err}");
    }

    #[test]
    fn check_lexical_refuses_nul() {
        let err = check_lexical("path", "a\0b").unwrap_err();
        assert!(err.contains("NUL"), "{err}");
    }

    /// Component parsing only: none of these is ever resolved, so no network
    /// share, WebDAV server or pipe is contacted.
    #[cfg(windows)]
    #[test]
    fn check_lexical_refuses_unc_device_and_verbatim_paths() {
        for raw in [
            r"\\attacker.example\share\x.rs",
            "//attacker.example/share/x.rs",
            r"\\attacker.example@SSL@443\t",
            r"\\.\pipe\ix-confine-test",
            r"\\?\C:\x",
            r"\\?\UNC\attacker.example\share",
            r"\??\UNC\attacker.example\share",
            r"\x",
            "C:x",
        ] {
            let err = check_lexical("path", raw).unwrap_err();
            assert!(err.contains("absolute path on a local drive"), "{raw}: {err}");
        }
        for raw in [r"C:\x", "C:/x", "a/b", r"a\b", "a", "."] {
            assert!(check_lexical("path", raw).is_ok(), "{raw}");
        }
    }

    /// A verbatim path to an in-root directory is refused before it is
    /// resolved: the prefix check, not the root check, stops it.
    #[cfg(windows)]
    #[test]
    fn confine_refuses_a_verbatim_path_even_inside_the_root() {
        let root = tempfile::tempdir().unwrap();
        let verbatim = format!(r"\\?\{}", root.path().join("traces").display());
        let err = confine_dest_in(&one(root.path()), "trace_dir", &verbatim).unwrap_err();
        assert!(err.contains("absolute path on a local drive"), "{err}");
    }

    #[cfg(windows)]
    #[test]
    fn simplify_strips_the_verbatim_prefix_only_for_plain_disk_paths() {
        assert_eq!(
            simplify(PathBuf::from(r"\\?\C:\a\b")),
            PathBuf::from(r"C:\a\b")
        );
        for kept in [r"\\?\UNC\host\share\a", r"\\?\C:\a\trailing.", r"C:\a"] {
            assert_eq!(simplify(PathBuf::from(kept)), PathBuf::from(kept));
        }
    }

    #[test]
    fn confine_rejects_a_symlink_that_leads_out_of_the_root() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let link = root.path().join("link");
        // A symlink on Unix and a junction on Windows need no privileges, so a
        // failure here is a broken test environment, not a reason to skip.
        link_dir(elsewhere.path(), &link).expect("create a symlink or junction");
        // A directory to be created under the link escapes.
        let err = confine_dest_in(&one(root.path()), "trace_dir", "link/new/traces").unwrap_err();
        assert!(err.contains("not inside an allowed destination root"), "{err}");
        assert!(!elsewhere.path().join("new").exists());
    }

    #[test]
    fn dest_refuses_a_dangling_link_instead_of_skipping_past_it() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let gone = elsewhere.path().join("gone");
        std::fs::create_dir_all(&gone).unwrap();
        let link = root.path().join("dang");
        link_dir(&gone, &link).expect("create a symlink or junction");
        std::fs::remove_dir(&gone).unwrap();
        assert!(!link.exists(), "precondition: the link dangles");

        let err = confine_dest_in(&one(root.path()), "trace_dir", "dang/traces").unwrap_err();
        assert!(err.contains("not inside an allowed destination root"), "{err}");
        assert!(!gone.exists());
    }

    #[test]
    fn dest_accepts_a_missing_directory_under_an_allowed_root() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        assert!(confine_dest_in(&one(root.path()), "trace_dir", "new/traces").is_ok());
        let out = elsewhere.path().join("new");
        assert!(confine_dest_in(&one(root.path()), "trace_dir", out.to_str().unwrap()).is_err());
        // A root that does not exist yet (e.g. ~/.ga/traces) still admits its tree,
        // including when it is the first root relative paths resolve against.
        let future_root = elsewhere.path().join("ga/traces");
        let roots = vec![root.path().to_path_buf(), future_root.clone()];
        assert!(confine_dest_in(&roots, "trace_dir", future_root.to_str().unwrap()).is_ok());
        let only_future = vec![future_root];
        assert!(confine_dest_in(&only_future, "trace_dir", "sub").is_ok());
        assert!(confine_dest_in(&only_future, "trace_dir", out.to_str().unwrap()).is_err());
    }
}
