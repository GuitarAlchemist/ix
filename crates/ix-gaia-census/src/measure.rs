//! Measures files as raw bytes.
//!
//! Every read is a byte read. Nothing here opens a file in text mode, decodes
//! it, or normalises a line ending: a CRLF file hashes as the bytes it holds.

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::{CensusRefusal, EntryClass, NonFileEntry};

/// One measured file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Measured {
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// Everything the evidence root holds, in two parts.
///
/// A v0a census measures raw bytes, so a plain file is the only entry class it
/// can measure. Every other class is retained here rather than skipped: an
/// entry this census cannot read is a fact about the directory, not an absence.
pub(crate) struct RootListing {
    pub(crate) files: BTreeMap<String, Measured>,
    /// Every entry that is not a plain file, in ordinal name order, each with
    /// the class it belongs to.
    pub(crate) non_files: Vec<NonFileEntry>,
}

/// Measure every file directly inside `root`, and name every entry that is not
/// one.
///
/// The returned map is keyed by file name, and `BTreeMap` iterates its `String`
/// keys in raw byte order — the ordinal file-name order the manifest declares.
/// Directory entries are never ordered by `read_dir` yield order, which is
/// neither case-sensitive nor stable.
pub(crate) fn measure_root(root: &Path) -> Result<RootListing, CensusRefusal> {
    let entries = std::fs::read_dir(root).map_err(|e| CensusRefusal::EvidenceRootUnreadable {
        detail: format!("{}: {e}", root.display()),
    })?;

    let mut files = BTreeMap::new();
    // Keyed too, so the reported order is ordinal rather than `read_dir` order.
    let mut non_files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| CensusRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", root.display()),
        })?;
        let file_type = entry
            .file_type()
            .map_err(|e| CensusRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", entry.path().display()),
            })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !file_type.is_file() {
            non_files.insert(name, classify(&file_type));
            continue;
        }
        let bytes = std::fs::read(entry.path()).map_err(|e| {
            CensusRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", entry.path().display()),
            }
        })?;
        files.insert(
            name,
            Measured {
                bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
            },
        );
    }
    Ok(RootListing {
        files,
        non_files: non_files
            .into_iter()
            .map(|(name, entry_class)| NonFileEntry { name, entry_class })
            .collect(),
    })
}

/// The class of an entry a v0a census cannot measure.
///
/// `read_dir` does not follow links, so a link is a link here whatever it points
/// at, and is reported as one rather than as its target. Anything else — a
/// device, a socket, a FIFO — is `Other` rather than being dropped: an
/// unrecognised class is still reported, never silently skipped.
fn classify(file_type: &std::fs::FileType) -> EntryClass {
    if file_type.is_symlink() {
        EntryClass::Symlink
    } else if file_type.is_dir() {
        EntryClass::Directory
    } else {
        EntryClass::Other
    }
}

/// SHA-256 of `bytes`, as 64 lowercase hex characters.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}
