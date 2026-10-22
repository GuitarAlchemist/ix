//! Locating the sentrux executable without a machine-specific default.
//!
//! Until ix#327 the fallback was one developer's `bin/` directory, so on every
//! other machine the bridge and the MCP tool tried a path that did not exist
//! and reported that path. The order is now: an explicit path from the caller,
//! then [`crate::SENTRUX_EXE_ENV`], then `sentrux` on `PATH`; when none
//! resolves the error names all three.

use crate::{Error, SENTRUX_EXE_ENV};
use std::ffi::OsString;
use std::path::PathBuf;

/// File names looked up on `PATH`, in order: the native spelling first, so a
/// stray `sentrux.exe` on Unix cannot mask a working `sentrux`.
#[cfg(windows)]
const CANDIDATES: &[&str] = &["sentrux.exe", "sentrux"];
#[cfg(not(windows))]
const CANDIDATES: &[&str] = &["sentrux", "sentrux.exe"];

/// Resolve the sentrux executable from `explicit`, `$SENTRUX_EXE`, then `PATH`.
pub fn resolve_sentrux_exe(explicit: Option<PathBuf>) -> Result<PathBuf, Error> {
    resolve_from(
        explicit,
        std::env::var_os(SENTRUX_EXE_ENV),
        std::env::var_os("PATH"),
    )
}

/// The pure form of [`resolve_sentrux_exe`]: `env` is the value of
/// `$SENTRUX_EXE`, `path` the value of `$PATH`. An empty value counts as unset.
pub fn resolve_from(
    explicit: Option<PathBuf>,
    env: Option<OsString>,
    path: Option<OsString>,
) -> Result<PathBuf, Error> {
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        return Ok(p);
    }
    if let Some(v) = env.filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(v));
    }
    let Some(path) = path else {
        return Err(Error::SentruxUnresolved);
    };
    for dir in std::env::split_paths(&path) {
        for name in CANDIDATES {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(Error::SentruxUnresolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn an_explicit_path_wins_without_being_checked() {
        let got = resolve_from(Some(PathBuf::from("x/sentrux")), os("y"), os("")).unwrap();
        assert_eq!(got, PathBuf::from("x/sentrux"));
    }

    #[test]
    fn the_env_var_beats_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sentrux.exe"), b"").unwrap();
        let got = resolve_from(
            None,
            os("from-env/sentrux"),
            Some(dir.path().as_os_str().to_owned()),
        )
        .unwrap();
        assert_eq!(got, PathBuf::from("from-env/sentrux"));
    }

    #[test]
    fn path_is_searched_for_both_spellings() {
        let dir = tempfile::tempdir().unwrap();
        let empty = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sentrux"), b"").unwrap();
        let path = std::env::join_paths([empty.path(), dir.path()]).unwrap();
        let got = resolve_from(None, os(""), Some(path)).unwrap();
        assert_eq!(got, dir.path().join("sentrux"));
    }

    #[test]
    fn nothing_resolving_names_every_option() {
        let empty = tempfile::tempdir().unwrap();
        let err = resolve_from(None, None, Some(empty.path().as_os_str().to_owned()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("SENTRUX_EXE"), "{err}");
        assert!(err.contains("PATH"), "{err}");
        assert!(err.contains("explicit"), "{err}");
    }
}
