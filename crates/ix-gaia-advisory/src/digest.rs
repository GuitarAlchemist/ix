//! The two named digest recipes of §3, and nothing else.
//!
//! They are different constructions and they are kept apart by name: every
//! aggregate this crate builds carries its recipe in the identifier that holds
//! it, so an implementer cannot bind the wrong construction. A recipe-free
//! aggregate identifier — one name over two incompatible recipes, which is the
//! defect B6 recorded — appears nowhere, and NC-19 fails the suite if one ever
//! does.

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::AdvisoryRefusal;

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

/// One regular file, measured as raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileMeasure {
    /// unit: bytes
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// Every regular file under a root, measured once, with both aggregates.
///
/// One pass. The bytes it costs are reported so they can be accounted against
/// the provenance account rather than hidden inside a rule's byte figure.
// `files` and `bytes_read` are consumed from slice S3 onward, when the
// artifact gains rows and a cost account.
#[allow(dead_code)]
pub(crate) struct MeasuredRoot {
    /// Keyed by slash-normalised path relative to the root; `BTreeMap` iterates
    /// in raw byte order, which is ordinal order.
    pub(crate) files: BTreeMap<String, FileMeasure>,
    pub(crate) ix_agg_1: String,
    pub(crate) gaia_agg_1: String,
    /// unit: bytes — what this pass cost.
    pub(crate) bytes_read: u64,
}

/// Measure every regular file under `root`, recursively.
pub(crate) fn measure_root(root: &Path) -> Result<MeasuredRoot, AdvisoryRefusal> {
    let mut files = BTreeMap::new();
    let mut bytes_read = 0u64;
    walk(root, root, &mut files, &mut bytes_read)?;

    Ok(MeasuredRoot {
        ix_agg_1: ix_agg_1(&files),
        gaia_agg_1: gaia_agg_1(&files),
        files,
        bytes_read,
    })
}

fn walk(
    root: &Path,
    dir: &Path,
    files: &mut BTreeMap<String, FileMeasure>,
    bytes_read: &mut u64,
) -> Result<(), AdvisoryRefusal> {
    let entries = std::fs::read_dir(dir).map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
        detail: format!("{}: {e}", dir.display()),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", dir.display()),
        })?;
        let file_type = entry
            .file_type()
            .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", entry.path().display()),
            })?;
        if file_type.is_dir() {
            walk(root, &entry.path(), files, bytes_read)?;
            continue;
        }
        if !file_type.is_file() {
            // Neither recipe measures an entry class that is not a regular
            // file. It is reported by the root listing, never digested here.
            continue;
        }
        let path = entry.path();
        let bytes = std::fs::read(&path).map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", path.display()),
        })?;
        let rel = relative_path(root, &path)?;
        *bytes_read += bytes.len() as u64;
        files.insert(
            rel,
            FileMeasure {
                bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
            },
        );
    }
    Ok(())
}

/// The path of `path` relative to `root`, slash-normalised, decoded strictly.
fn relative_path(root: &Path, path: &Path) -> Result<String, AdvisoryRefusal> {
    let rel = path
        .strip_prefix(root)
        .map_err(|_| AdvisoryRefusal::EvidenceRootUnreadable {
            detail: format!("{} is not under {}", path.display(), root.display()),
        })?;
    let text = rel
        .to_str()
        .ok_or_else(|| AdvisoryRefusal::NonUtf8EntryName {
            lossy: rel.to_string_lossy().into_owned(),
        })?;
    Ok(text.replace('\\', "/"))
}

/// `IX-AGG-1` (§3.1). One `path NUL bytes NUL file_sha256 LF` row per regular
/// file, ordinal by relative path, **trailing LF included**, SHA-256 over the
/// UTF-8 encoding of the concatenated rows, lowercase hex.
///
/// This is the only recipe M3 constructs a digest of its own with.
fn ix_agg_1(files: &BTreeMap<String, FileMeasure>) -> String {
    let mut rows = String::new();
    for (rel, measure) in files {
        rows.push_str(rel);
        rows.push('\0');
        rows.push_str(&measure.bytes.to_string());
        rows.push('\0');
        rows.push_str(&measure.sha256);
        rows.push('\n');
    }
    sha256_hex(rows.as_bytes())
}

/// `IX-AGG-1` (§3.1) over exactly one artifact, at a **fixed relative path**.
///
/// The recipe is [`ix_agg_1`] itself, reached with a one-row map rather than
/// forked: §11.3 pins `report_digest` to one canonical relative path precisely
/// because `IX-AGG-1` embeds each row's path, so a second construction that
/// happened to disagree by one byte would trip `ReplayDivergence` on every
/// replay and be indistinguishable from a real divergence.
// Reached by S12 when a holdout report is digested for a ledger receipt; no
// path in M3-as-selected calls it.
#[allow(dead_code)]
pub(crate) fn ix_agg_1_single(relative_path: &str, bytes: &[u8]) -> String {
    let mut files = BTreeMap::new();
    files.insert(
        relative_path.to_string(),
        FileMeasure {
            bytes: bytes.len() as u64,
            sha256: sha256_hex(bytes),
        },
    );
    ix_agg_1(&files)
}

/// `GAIA-AGG-1` (§3.2) — a **quoted foreign construction**, declared by the
/// approved evidence at bundle-manifest L18 and reproduced, never invented
/// here. One `name|bytes|sha256` record per file, ordinal filename order,
/// LF-joined, **with no trailing newline**, SHA-256 over the UTF-8 encoding.
///
/// Used at exactly the two sites §3.2 enumerates and derived from at none.
fn gaia_agg_1(files: &BTreeMap<String, FileMeasure>) -> String {
    let mut by_name: BTreeMap<&str, &FileMeasure> = BTreeMap::new();
    for (rel, measure) in files {
        let name = rel.rsplit('/').next().unwrap_or(rel);
        by_name.insert(name, measure);
    }
    let joined = by_name
        .into_iter()
        .map(|(name, measure)| format!("{name}|{}|{}", measure.bytes, measure.sha256))
        .collect::<Vec<_>>()
        .join("\n");
    sha256_hex(joined.as_bytes())
}
