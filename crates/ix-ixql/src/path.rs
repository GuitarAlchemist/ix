//! One normalizer for every pipeline-supplied path.
//!
//! A pipeline builds its write paths by interpolation, so a path is *data* —
//! and two components used to disagree about what a given string meant. The
//! schema gate matched prefixes on the raw text while [`FsHost`] handed the
//! same text to the OS, so `state/./quality/verdicts/x.json` slipped past a
//! rule registered on `state/quality/verdicts/` and still landed inside the
//! protected directory. Any normalization that is not shared is a bypass
//! waiting to be found, so both callers go through [`normalize`].
//!
//! [`FsHost`]: crate::FsHost

/// Why a path may not be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    #[error("`{0}` is absolute; pipeline paths must be relative to the run root")]
    Absolute(String),
    #[error("`{0}` contains `..` and would escape the run root")]
    Traversal(String),
    #[error("`{0}` names nothing after normalization")]
    Empty(String),
}

/// Reduce a path to the single canonical spelling both the gate and the host
/// will agree on: forward slashes, no empty segments, no `.` segments.
///
/// Rejects anything that could leave the run root — `..` at any position, and
/// absolute or drive-qualified paths — because a pipeline that computed such a
/// path computed it from data, and there is no legitimate case for it.
pub fn normalize(path: &str) -> Result<String, PathError> {
    let unified = path.replace('\\', "/");

    if unified.starts_with('/') || has_drive_prefix(&unified) {
        return Err(PathError::Absolute(path.to_string()));
    }

    let mut segments = Vec::new();
    for segment in unified.split('/') {
        match segment {
            // Collapses `a//b` and `a/./b` alike.
            "" | "." => continue,
            ".." => return Err(PathError::Traversal(path.to_string())),
            other => segments.push(other),
        }
    }

    if segments.is_empty() {
        return Err(PathError::Empty(path.to_string()));
    }
    Ok(segments.join("/"))
}

/// Best-effort form for prefix matching, where refusing outright is not an
/// option: an unnormalizable path still has to compare against *something*, and
/// comparing against the raw text is what let the bypass through. Callers that
/// can reject — the evaluator's `ix.io.write` — use [`normalize`] instead.
pub(crate) fn normalize_lossy(path: &str) -> String {
    normalize(path).unwrap_or_else(|_| path.replace('\\', "/"))
}

fn has_drive_prefix(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_segments_and_repeated_separators_collapse() {
        // The exact bypass: this must normalize onto the protected prefix.
        assert_eq!(
            "state/quality/verdicts/bad.json",
            normalize("state/./quality//verdicts/bad.json").unwrap()
        );
    }

    #[test]
    fn backslashes_become_forward_slashes() {
        assert_eq!(
            "state/quality/v.json",
            normalize("state\\quality\\v.json").unwrap()
        );
    }

    #[test]
    fn traversal_is_refused_wherever_it_appears() {
        assert!(matches!(
            normalize("../escape.json"),
            Err(PathError::Traversal(_))
        ));
        assert!(matches!(
            normalize("state/quality/../../escape.json"),
            Err(PathError::Traversal(_))
        ));
    }

    #[test]
    fn absolute_and_drive_qualified_paths_are_refused() {
        assert!(matches!(
            normalize("/etc/passwd"),
            Err(PathError::Absolute(_))
        ));
        assert!(matches!(
            normalize("C:\\Windows\\system32"),
            Err(PathError::Absolute(_))
        ));
    }

    #[test]
    fn a_path_that_names_nothing_is_refused() {
        assert!(matches!(normalize("./"), Err(PathError::Empty(_))));
        assert!(matches!(normalize(""), Err(PathError::Empty(_))));
    }

    #[test]
    fn already_canonical_paths_are_unchanged() {
        assert_eq!("state/a/b.json", normalize("state/a/b.json").unwrap());
    }

    #[test]
    fn lossy_normalization_never_fails_but_still_canonicalizes() {
        assert_eq!("state/a.json", normalize_lossy("state/./a.json"));
        assert_eq!("../x", normalize_lossy("..\\x"));
    }
}
