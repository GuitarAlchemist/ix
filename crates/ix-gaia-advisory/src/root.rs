//! Lists an evidence root without reading a single content byte.
//!
//! The structural rules of §9.1 read no content, so the listing they run over
//! must not read any either: an entry's length comes from its directory
//! metadata, never from a read. Content is read only when a rule asks for it,
//! and every such read is charged to that rule's byte account.
//!
//! Entry names are decoded strictly. `to_string_lossy` would map two distinct
//! names onto one string and merge them silently, which is exactly the defect
//! NC-6 exists to catch, so a name that is not valid UTF-8 is a refusal.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{AdvisoryRefusal, EntryClass, NonFileEntry};

/// One entry the listing could measure, by metadata alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Listed {
    /// unit: bytes — from directory metadata, not from a read.
    pub(crate) bytes: u64,
}

/// Everything the evidence root holds, in two parts.
///
/// `BTreeMap` iterates its `String` keys in raw UTF-8 byte order, which is the
/// ordinal order every emitted list uses. `read_dir` yield order is neither
/// ordinal nor stable and is never relied on.
pub(crate) struct RootListing {
    pub(crate) files: BTreeMap<String, Listed>,
    /// Every entry that is not a plain file, in ordinal name order.
    pub(crate) non_files: Vec<NonFileEntry>,
}

impl RootListing {
    /// Every name the root holds, plain files and other classes alike.
    ///
    /// A surplus directory is a surplus name: an entry class this crate cannot
    /// measure is still a fact about the root, never an absence.
    pub(crate) fn name_set(&self) -> std::collections::BTreeSet<String> {
        self.files
            .keys()
            .cloned()
            .chain(self.non_files.iter().map(|entry| entry.name.clone()))
            .collect()
    }
}

/// List every entry directly inside `root`, reading no content byte.
pub(crate) fn list(root: &Path) -> Result<RootListing, AdvisoryRefusal> {
    let entries = std::fs::read_dir(root).map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
        detail: format!("{}: {e}", root.display()),
    })?;

    let mut files = BTreeMap::new();
    let mut non_files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", root.display()),
        })?;
        let raw_name = entry.file_name();
        // Strict decode. A lossy decode is how two distinct names become one.
        let name = raw_name
            .to_str()
            .ok_or_else(|| AdvisoryRefusal::NonUtf8EntryName {
                lossy: raw_name.to_string_lossy().into_owned(),
            })?
            .to_string();

        let file_type = entry
            .file_type()
            .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", entry.path().display()),
            })?;
        if !file_type.is_file() {
            non_files.insert(name, classify(&file_type));
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", entry.path().display()),
            })?;
        files.insert(
            name,
            Listed {
                bytes: metadata.len(),
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

/// The class of an entry this crate cannot measure as raw bytes.
///
/// `read_dir` does not follow links, so a link is reported as a link whatever
/// it points at. Anything else is `Other` rather than dropped.
fn classify(file_type: &std::fs::FileType) -> EntryClass {
    if file_type.is_symlink() {
        EntryClass::Symlink
    } else if file_type.is_dir() {
        EntryClass::Directory
    } else {
        EntryClass::Other
    }
}
