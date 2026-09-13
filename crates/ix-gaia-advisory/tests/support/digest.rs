//! An independent Lane-W implementation of the two named digest recipes.
//!
//! Deliberately written from §3's normative text rather than shared with
//! `src/`. A digest test that calls the implementation under test proves only
//! that the implementation equals itself; this one can disagree with it.

use std::path::Path;

use sha2::{Digest, Sha256};

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// One regular file under a root, in ordinal order.
pub struct OrdinalFile {
    /// Path relative to the root, slash-normalised.
    pub rel: String,
    /// Bare file name.
    pub name: String,
    /// unit: bytes
    pub bytes: u64,
    pub sha256: String,
}

/// Every regular file under `root`, recursively, in ordinal order — ascending
/// raw-UTF-8-byte comparison of the relative path.
pub fn ordinal_files(root: &Path) -> Vec<OrdinalFile> {
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort_by(|a, b| a.rel.as_bytes().cmp(b.rel.as_bytes()));
    out
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<OrdinalFile>) {
    let entries = std::fs::read_dir(dir).expect("directory is readable");
    for entry in entries {
        let entry = entry.expect("directory entry is readable");
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out);
            continue;
        }
        let bytes = std::fs::read(&path).expect("file is readable");
        let rel = path
            .strip_prefix(root)
            .expect("path is under root")
            .to_str()
            .expect("path is UTF-8")
            .replace('\\', "/");
        out.push(OrdinalFile {
            rel,
            name: entry
                .file_name()
                .to_str()
                .expect("name is UTF-8")
                .to_string(),
            bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        });
    }
}

/// `IX-AGG-1` (§3.1) — `path NUL bytes NUL file_sha256 LF` per file, ordinal by
/// relative path, **trailing LF included**, SHA-256 over the UTF-8 encoding.
pub fn ix_agg_1(root: &Path) -> String {
    let mut rows = String::new();
    for file in ordinal_files(root) {
        rows.push_str(&file.rel);
        rows.push('\0');
        rows.push_str(&file.bytes.to_string());
        rows.push('\0');
        rows.push_str(&file.sha256);
        rows.push('\n');
    }
    sha256_hex(rows.as_bytes())
}

/// `GAIA-AGG-1` (§3.2) — a quoted foreign construction: `name|bytes|sha256` per
/// file, ordinal filename order, LF-joined, **no trailing newline**, SHA-256
/// over the UTF-8 encoding.
pub fn gaia_agg_1(root: &Path) -> String {
    let mut files = ordinal_files(root);
    files.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let joined = files
        .iter()
        .map(|file| format!("{}|{}|{}", file.name, file.bytes, file.sha256))
        .collect::<Vec<_>>()
        .join("\n");
    sha256_hex(joined.as_bytes())
}
