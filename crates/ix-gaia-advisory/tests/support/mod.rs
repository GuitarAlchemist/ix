//! Lane-W test support.
//!
//! Three things live here and nowhere else:
//!
//! * fixture and scratch paths (this file);
//! * the **label oracle** ([`oracle`]) — ground truth, computed from the staged
//!   bytes against the frozen original manifest, never from family membership,
//!   and sharing no module with any rule;
//! * the **corpus generator** ([`corpus`]) — a generator, never a labeller.
//!
//! None of it is part of the crate's public surface. `src/` holds the rules and
//! the two public verbs; a bug in a rule therefore cannot silently agree with
//! the oracle, because they are separate code paths that share nothing.
//!
//! Each integration test binary links the whole module, so items a given binary
//! does not use are legitimately unused there.
#![allow(dead_code)]

pub mod corpus;
pub mod digest;
pub mod oracle;

use std::path::{Path, PathBuf};

/// The 14 approved S1 fixtures — the only Gaia evidence a claim may be drawn
/// from. Read-only: no test writes here, and NC-13 re-measures both fixed
/// points after the suite.
pub fn approved_evidence_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("ix-gaia-census")
        .join("tests")
        .join("fixtures")
        .join("gaia-s1-r18")
}

/// This crate's own `src/` tree, for the controls that scan source text.
pub fn crate_src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

pub const MANIFEST_FILE_NAME: &str = "gaia-s1-r2-bundle-manifest.md";

/// A scratch directory outside the repository.
///
/// §19: the corpus is generated into a scratch directory and is never
/// committed. Nothing this suite writes lands inside the worktree.
pub fn scratch_root(slice: &str) -> PathBuf {
    let base = match std::env::var_os("IX_GAIA_M3_SCRATCH_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join("ix-gaia-m3-scratch"),
    };
    base.join(slice)
}

/// A freshly emptied scratch directory for one slice.
pub fn fresh_scratch(slice: &str) -> PathBuf {
    let dir = scratch_root(slice);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("scratch directory is removable");
    }
    std::fs::create_dir_all(&dir).expect("scratch directory is creatable");
    dir
}

/// Copy the 14 approved fixtures into `dest`, byte for byte.
pub fn stage_pristine(dest: &Path) {
    std::fs::create_dir_all(dest).expect("staged root is creatable");
    for entry in std::fs::read_dir(approved_evidence_root()).expect("approved evidence is readable")
    {
        let entry = entry.expect("directory entry is readable");
        let bytes = std::fs::read(entry.path()).expect("fixture is readable");
        std::fs::write(dest.join(entry.file_name()), bytes).expect("staged file is writable");
    }
}

/// Every file directly inside `root`, in ordinal (raw UTF-8 byte) name order.
pub fn ordinal_file_names(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .expect("root is readable")
        .map(|entry| {
            entry
                .expect("directory entry is readable")
                .file_name()
                .to_str()
                .expect("fixture names are UTF-8")
                .to_string()
        })
        .collect();
    names.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    names
}
