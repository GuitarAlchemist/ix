//! Confinement of caller-supplied filesystem paths.
//!
//! Tools classified `Read` (Tier 1) or `EditInProject` (Tier 2) in
//! `ix-approval` run without a prompt, so any path, directory or repo root a
//! caller names is resolved here before the tool reads, walks or writes it. A
//! path is accepted only when it lies under an allowed root:
//!
//! - the workspace root ([`workspace_root`]);
//! - each directory listed in `IX_EXTRA_ROOTS` (OS path-list syntax: `;` on
//!   Windows, `:` elsewhere; relative entries resolve against the workspace
//!   root), for operators whose MCP server must read another checkout such as a
//!   sibling repo;
//! - for the GA trace tools only, the trace locations the operator already
//!   chose ([`trace_roots`]). A trace *destination* is confined to those
//!   locations alone.
//!
//! Nothing touches the filesystem until the raw argument passes
//! [`check_lexical`]: `..`, NUL, and on Windows anything but a relative path or
//! a drive-letter absolute path (UNC `\\server\share`, device `\\.\`, verbatim
//! `\\?\`, drive-relative `C:x`, rooted `\x`) are refused first, so the check
//! itself never opens a network share or a named pipe. The joined path is then
//! canonicalized and must sit under a canonical allowed root, and the canonical
//! path is what the tool gets back.
//!
//! Known limits, both needing local write access inside a root: a hard link
//! inside a root to a file outside passes (the link has its own in-root path),
//! and a symlink or junction swapped into an already-checked path before the
//! tool opens it is followed.
//!
//! Defaults a tool picks for itself are not caller input and are not confined.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Environment variable naming extra directories auto-approved tools may use.
pub(crate) const EXTRA_ROOTS_ENV: &str = "IX_EXTRA_ROOTS";

/// Environment variable naming the workspace root.
pub(crate) const ROOT_ENV: &str = "IX_ROOT";

/// The workspace root confined paths resolve against:
///
/// 1. `IX_ROOT` when set;
/// 2. else the ix checkout holding the running executable (the server runs as
///    `<checkout>/target/<profile>/ix-mcp`);
/// 3. else the ix checkout holding the current directory.
///
/// A checkout is a directory whose `Cargo.toml` has `[workspace]` and which
/// contains `crates/ix-agent`. The bare current directory never becomes the
/// root, and neither does `CARGO_MANIFEST_DIR`, which a server spawned from
/// another cargo command would inherit. Fails closed, so every confined
/// argument is refused, when no root is found, it is not accessible, or it is
/// a volume root or the home directory.
pub(crate) fn workspace_root() -> Result<PathBuf, String> {
    let root = match std::env::var_os(ROOT_ENV) {
        Some(root) => PathBuf::from(root),
        None => std::env::current_exe()
            .ok()
            .and_then(|exe| find_ix_checkout(&exe))
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|cwd| find_ix_checkout(&cwd))
            })
            .ok_or_else(|| {
                format!(
                    "no workspace root: set {ROOT_ENV} to the ix checkout, or run the server from it"
                )
            })?,
    };
    check_root(&root)?;
    Ok(root)
}

/// The nearest ancestor of `start` (itself included) that is an ix checkout.
fn find_ix_checkout(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| {
            dir.join("crates/ix-agent/Cargo.toml").is_file()
                && std::fs::read_to_string(dir.join("Cargo.toml"))
                    .is_ok_and(|manifest| manifest.contains("[workspace]"))
        })
        .map(Path::to_path_buf)
}

/// Refuse a root that is inaccessible, a volume root or the home directory:
/// confining to those would admit nearly every file the user can read.
fn check_root(root: &Path) -> Result<(), String> {
    let canonical = root
        .canonicalize()
        .map_err(|_| format!("the workspace root {} is not accessible", root.display()))?;
    let is_home = ["HOME", "USERPROFILE"]
        .into_iter()
        .filter_map(std::env::var_os)
        .filter_map(|home| Path::new(&home).canonicalize().ok())
        .any(|home| home == canonical);
    if canonical.parent().is_none() || is_home {
        return Err(format!(
            "the workspace root {} is a volume root or the home directory; set {ROOT_ENV} to the ix checkout",
            root.display()
        ));
    }
    Ok(())
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
    roots_with_extras(root, std::env::var_os(EXTRA_ROOTS_ENV))
}

/// `root` followed by the directories in `extras`, relative entries resolved
/// against `root` rather than the process cwd.
fn roots_with_extras(root: &Path, extras: Option<OsString>) -> Vec<PathBuf> {
    let mut roots = vec![root.to_path_buf()];
    roots.extend(
        extra_roots_from(extras)
            .into_iter()
            .map(|extra| root.join(extra)),
    );
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
/// [`check_lexical`] runs first; everything else (absolute paths elsewhere,
/// symlinks or junctions leading out) is caught by canonicalizing and requiring
/// the result to sit under a canonical allowed root. A missing path and a path
/// outside get the same message, so the error does not reveal whether a file
/// exists elsewhere on disk, and nothing is read before the check. Returns the
/// canonical path, so the tool opens what was checked.
pub(crate) fn confine(root: &Path, param: &str, raw: &str) -> Result<PathBuf, String> {
    confine_in(&allowed_roots(root), param, raw)
}

/// [`confine`] against an explicit root list. Relative paths resolve against
/// `roots[0]`; roots that do not exist are skipped.
pub(crate) fn confine_in(roots: &[PathBuf], param: &str, raw: &str) -> Result<PathBuf, String> {
    let full = join_checked(roots, param, raw)?;
    let canonical_roots = canonical_roots(roots, param)?;
    match full.canonicalize() {
        Ok(resolved) if canonical_roots.iter().any(|r| resolved.starts_with(r)) => {
            Ok(simplify(resolved))
        }
        _ => Err(format!(
            "`{param}`: {raw} is not an existing path inside an allowed root"
        )),
    }
}

/// Like [`confine_in`], for a directory the tool will create if it is missing:
/// the deepest existing ancestor is canonicalized (so a symlink or junction
/// leading out is still caught) and the missing tail, which holds no `..`, is
/// appended before the root check. An ancestor that exists but does not
/// resolve, such as a dangling link, is refused rather than skipped.
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
    fn confine_accepts_existing_paths_inside_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("state")).unwrap();
        std::fs::write(root.path().join("state/log.jsonl"), "").unwrap();

        let got = confine_in(&one(root.path()), "log", "state/log.jsonl").unwrap();
        assert_eq!(
            got,
            simplify(root.path().join("state/log.jsonl").canonicalize().unwrap()),
            "the canonical path is returned"
        );
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

    /// A verbatim path to a real in-root file is refused before it is resolved:
    /// the prefix check, not the root check, stops it.
    #[cfg(windows)]
    #[test]
    fn confine_refuses_a_verbatim_path_even_inside_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("x.rs"), "").unwrap();
        let verbatim = format!(r"\\?\{}", root.path().join("x.rs").display());
        let err = confine_in(&one(root.path()), "path", &verbatim).unwrap_err();
        assert!(err.contains("absolute path on a local drive"), "{err}");
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
        assert!(err_existing.contains("not an existing path inside an allowed root"));
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
        assert!(err.contains("not inside an allowed destination root"), "{err}");
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
    fn relative_extra_roots_resolve_against_the_workspace_root() {
        let root = std::env::temp_dir().join("ws");
        let absolute = std::env::temp_dir().join("abs");
        let joined = std::env::join_paths([Path::new("../ga"), absolute.as_path()]).unwrap();
        assert_eq!(
            roots_with_extras(&root, Some(joined)),
            vec![root.clone(), root.join("../ga"), absolute]
        );
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

    #[test]
    fn find_ix_checkout_needs_a_workspace_manifest_and_ix_agent() {
        let dir = tempfile::tempdir().unwrap();
        let checkout = dir.path().join("ix");
        std::fs::create_dir_all(checkout.join("crates/ix-agent/src")).unwrap();
        std::fs::write(checkout.join("crates/ix-agent/Cargo.toml"), "[package]").unwrap();
        let deep = checkout.join("target/release");
        std::fs::create_dir_all(&deep).unwrap();
        assert_eq!(find_ix_checkout(&deep.join("ix-mcp.exe")), None, "no [workspace]");

        std::fs::write(checkout.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
        assert_eq!(find_ix_checkout(&deep.join("ix-mcp.exe")), Some(checkout.clone()));
        // Another Rust workspace is not an ix checkout.
        let other = dir.path().join("other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("Cargo.toml"), "[workspace]\n").unwrap();
        assert_eq!(find_ix_checkout(&other), None);
    }

    #[test]
    fn check_root_refuses_a_volume_root_and_the_home_directory() {
        let volume = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .ancestors()
            .last()
            .unwrap()
            .to_path_buf();
        let err = check_root(&volume).unwrap_err();
        assert!(err.contains("volume root or the home directory"), "{err}");
        let homes = ["HOME", "USERPROFILE"]
            .into_iter()
            .filter_map(std::env::var_os)
            .filter(|home| Path::new(home).is_dir());
        for home in homes {
            let err = check_root(Path::new(&home)).unwrap_err();
            assert!(err.contains("volume root or the home directory"), "{err}");
        }
        let root = tempfile::tempdir().unwrap();
        assert!(check_root(root.path()).is_ok());
        assert!(check_root(&root.path().join("missing")).is_err());
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
