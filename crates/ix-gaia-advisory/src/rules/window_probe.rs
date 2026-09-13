//! The Window Reference Table (§9.2) and the windowed rule.
//!
//! The WRT is a **disclosed derived constant**, not a declared one: the
//! manifest declares whole-file digests and says nothing about windows, so the
//! pristine head/tail digests are derived here, from the approved bytes, by a
//! recipe stated in full so any reviewer reproduces it independently.
//!
//! Head and tail **may overlap** when `len < 2w`. That is defined,
//! deterministic and disclosed rather than guarded against: at `w = 4096`
//! exactly one of the fourteen approved files is short enough for the two
//! windows to cover it completely, and at that width the rule degenerates to a
//! whole-file comparison for that file.
//!
//! This file's own text is what `rule_source_digest` hashes, so any change to
//! the rule changes the fingerprint every artifact carries.

use std::collections::BTreeMap;
use std::path::Path;

use crate::digest::sha256_hex;
use crate::AdvisoryRefusal;

/// One file's pristine window digests at one width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowRow {
    /// unit: bytes
    pub(crate) bytes: u64,
    pub(crate) head_sha256: String,
    pub(crate) tail_sha256: String,
}

/// The Window Reference Table at one width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowReferenceTable {
    pub(crate) rows: BTreeMap<String, WindowRow>,
    /// SHA-256 of the table text. A single-artifact digest, not a multi-file
    /// record aggregate, so §3.3's `*_agg_1` naming rule does not reach it.
    pub(crate) digest: String,
    /// unit: bytes — the one full pristine pass the table costs, accounted
    /// separately from any rule's own byte figure.
    pub(crate) construction_bytes: u64,
}

/// Build the table from a pristine reference root, or the empty table when no
/// reference root is bound.
///
/// The empty table is not a null: it is a table with no rows, and its digest is
/// the digest of the empty string. A rule that needs windows and has no
/// reference refuses in the binder before reaching here.
pub(crate) fn build(
    reference_root: Option<&Path>,
    window_bytes: u32,
) -> Result<WindowReferenceTable, AdvisoryRefusal> {
    let mut rows: BTreeMap<String, WindowRow> = BTreeMap::new();
    let mut construction_bytes = 0u64;

    if let Some(root) = reference_root {
        let entries =
            std::fs::read_dir(root).map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", root.display()),
            })?;
        for entry in entries {
            let entry = entry.map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", root.display()),
            })?;
            let file_type =
                entry
                    .file_type()
                    .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                        detail: format!("{}: {e}", entry.path().display()),
                    })?;
            if !file_type.is_file() {
                continue;
            }
            let raw_name = entry.file_name();
            let name = raw_name
                .to_str()
                .ok_or_else(|| AdvisoryRefusal::NonUtf8EntryName {
                    lossy: raw_name.to_string_lossy().into_owned(),
                })?
                .to_string();

            // At width zero the two windows are empty, so no content byte is
            // read and the pass costs nothing. Its digests are still defined.
            if window_bytes == 0 {
                let bytes = entry
                    .metadata()
                    .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                        detail: format!("{}: {e}", entry.path().display()),
                    })?
                    .len();
                rows.insert(
                    name,
                    WindowRow {
                        bytes,
                        head_sha256: sha256_hex(&[]),
                        tail_sha256: sha256_hex(&[]),
                    },
                );
                continue;
            }

            let content = std::fs::read(entry.path()).map_err(|e| {
                AdvisoryRefusal::EvidenceRootUnreadable {
                    detail: format!("{}: {e}", entry.path().display()),
                }
            })?;
            construction_bytes += content.len() as u64;
            let (head, tail) = windows(&content, window_bytes);
            rows.insert(
                name,
                WindowRow {
                    bytes: content.len() as u64,
                    head_sha256: sha256_hex(head),
                    tail_sha256: sha256_hex(tail),
                },
            );
        }
    }

    Ok(WindowReferenceTable {
        digest: sha256_hex(table_text(&rows).as_bytes()),
        rows,
        construction_bytes,
    })
}

/// `name | len | sha256(head) | sha256(tail)` per file, ordinal name order,
/// LF-joined, no trailing newline.
fn table_text(rows: &BTreeMap<String, WindowRow>) -> String {
    rows.iter()
        .map(|(name, row)| {
            format!(
                "{name} | {} | {} | {}",
                row.bytes, row.head_sha256, row.tail_sha256
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `head = bytes[0 .. min(w, len)]`, `tail = bytes[len.saturating_sub(w) .. len]`.
///
/// The two overlap when `len < 2w`. Reading the file once costs
/// `min(len, 2w)` content bytes, which is what the cost account charges.
pub(crate) fn windows(content: &[u8], window_bytes: u32) -> (&[u8], &[u8]) {
    let width = window_bytes as usize;
    let head_end = width.min(content.len());
    let tail_start = content.len().saturating_sub(width);
    (&content[..head_end], &content[tail_start..])
}

/// unit: bytes — what one windowed read of a file of `len` bytes costs.
///
/// The two windows overlap when `len < 2w`, so the file is read once and the
/// cost is `min(len, 2w)` rather than `2w`.
pub(crate) fn read_cost(len: u64, window_bytes: u32) -> u64 {
    len.min(2 * u64::from(window_bytes))
}
