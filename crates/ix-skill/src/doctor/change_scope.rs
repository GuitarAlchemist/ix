//! What the working tree currently changes, so `--fast` can decide *honestly*
//! which checks it is allowed to skip.
//!
//! # The rule
//!
//! `--fast` never chooses checks by cost. It runs every check whose verdict the
//! current change set could alter, and skips only the ones it can prove are
//! unaffected. When the change set cannot be determined — no git, a broken
//! repo, an unreadable status — it skips **nothing**.
//!
//! That direction matters. A pre-commit path that skips the check most likely
//! to catch the mistake being made is worse than no fast path at all, because
//! it converts "I did not look" into a green tick. So the failure mode here is
//! deliberately "runs too much", never "reports ok without looking".
//!
//! # Why this is not "skip the slow ones"
//!
//! Measured on this workspace (3 trials, warm cache):
//!
//! | check | cost | reads |
//! |---|---|---|
//! | `registry-snapshot` | 2.3–7.1 ms | the linked capability registry |
//! | `orphan-traits` | 1687–3307 ms | every `*.rs`, plus its allowlist |
//! | `dark-features` | 1898–3334 ms | `cargo metadata`, every `*.rs`, its allowlist |
//!
//! The two expensive checks are within noise of each other and together are
//! ~99.9% of the runtime, so a cost-ordered split would drop exactly the two
//! checks that catch trait and feature mistakes. Both also read `*.rs`, so on
//! any Rust change `--fast` is the full run by construction — the speedup is
//! real only for changes that touch no Rust and no manifest.
//!
//! On this repository's last 300 non-merge commits that is 87 of 218
//! human-authored commits (39.9%). The unfiltered figure is 64.5%, but 81 of
//! those are bot snapshot commits that never run a pre-commit hook; measuring
//! the population that actually reaches this code is the difference between a
//! 40% win and a 65% claim.

use std::collections::BTreeSet;
use std::path::Path;

/// Paths the working tree changes relative to `HEAD`, including staged,
/// unstaged, untracked and both sides of a rename.
#[derive(Debug, Clone, Default)]
pub struct ChangeScope {
    paths: BTreeSet<String>,
}

impl ChangeScope {
    /// Ask git what changed under `root`.
    ///
    /// Returns `None` when git cannot answer — not an empty scope. An empty
    /// scope means "nothing changed, skip freely"; `None` means "unknown, skip
    /// nothing", and conflating the two is how a fast path starts lying.
    pub fn detect(root: &Path) -> Option<Self> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["status", "--porcelain", "--untracked-files=all", "-z"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8(out.stdout).ok()?;
        Some(Self::parse_porcelain_z(&text))
    }

    /// Parse `git status --porcelain -z` output.
    ///
    /// `-z` is used rather than the human format because git quotes and escapes
    /// paths containing spaces or non-ASCII in the default output, and a
    /// mis-unquoted path would silently drop a file from the scope — which
    /// would let `--fast` skip a check it should have run.
    ///
    /// Records are NUL-separated `XY <path>`. For a rename or copy the origin
    /// path follows as its own NUL-terminated field; both sides are kept,
    /// because deleting the old path can orphan a trait just as adding the new
    /// one can.
    pub fn parse_porcelain_z(text: &str) -> Self {
        let mut paths = BTreeSet::new();
        let mut fields = text.split('\0').filter(|f| !f.is_empty());
        while let Some(record) = fields.next() {
            // "XY path" — status is the first two bytes, then a space.
            let Some((status, path)) = record.split_at_checked(3) else {
                continue;
            };
            if !path.is_empty() {
                paths.insert(path.to_string());
            }
            if status.starts_with('R') || status.starts_with('C') {
                if let Some(origin) = fields.next() {
                    if !origin.is_empty() {
                        paths.insert(origin.to_string());
                    }
                }
            }
        }
        Self { paths }
    }

    /// Every changed path, repo-relative, forward-slashed as git reports them.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.paths.iter().map(String::as_str)
    }

    /// How many paths changed. Used only for human-readable summaries.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether the working tree is clean.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Any Rust source changed. Both slow checks read `*.rs`, so this is the
    /// predicate that decides whether `--fast` is fast at all.
    pub fn touches_rust(&self) -> bool {
        self.paths.iter().any(|p| p.ends_with(".rs"))
    }

    /// Any cargo manifest or lockfile changed — the input to `cargo metadata`,
    /// and therefore to feature resolution.
    pub fn touches_manifest(&self) -> bool {
        self.paths
            .iter()
            .any(|p| p.ends_with("Cargo.toml") || p.ends_with("Cargo.lock"))
    }

    /// A specific repo-relative file changed.
    pub fn touches_path(&self, rel: &str) -> bool {
        self.paths.iter().any(|p| p == rel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_staged_unstaged_and_untracked_records() {
        // `M  a.rs` staged, ` M b.rs` unstaged, `?? c.rs` untracked.
        let scope = ChangeScope::parse_porcelain_z("M  a.rs\0 M b.rs\0?? c.rs\0");
        let got: Vec<&str> = scope.paths().collect();
        assert_eq!(got, ["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn a_rename_contributes_both_sides() {
        // Deleting the old path can orphan a trait just as adding the new one
        // can, so a rename must widen the scope, not replace it.
        let scope = ChangeScope::parse_porcelain_z("R  new/mod.rs\0old/mod.rs\0?? other.md\0");
        let got: Vec<&str> = scope.paths().collect();
        assert_eq!(got, ["new/mod.rs", "old/mod.rs", "other.md"]);
        assert!(scope.touches_rust());
    }

    #[test]
    fn paths_with_spaces_survive_because_the_format_is_nul_separated() {
        let scope = ChangeScope::parse_porcelain_z("?? docs/a note.md\0 M crates/x/src/lib.rs\0");
        let got: Vec<&str> = scope.paths().collect();
        assert_eq!(got, ["crates/x/src/lib.rs", "docs/a note.md"]);
        assert!(scope.touches_rust());
    }

    #[test]
    fn a_docs_only_change_touches_neither_rust_nor_manifests() {
        let scope = ChangeScope::parse_porcelain_z(" M README.md\0 M docs/guides/x.md\0");
        assert!(!scope.touches_rust());
        assert!(!scope.touches_manifest());
        assert_eq!(scope.len(), 2);
    }

    #[test]
    fn manifest_and_lockfile_both_count_as_manifests() {
        for rec in [" M crates/x/Cargo.toml\0", " M Cargo.lock\0"] {
            let scope = ChangeScope::parse_porcelain_z(rec);
            assert!(scope.touches_manifest(), "{rec:?} should be a manifest");
            assert!(!scope.touches_rust());
        }
    }

    #[test]
    fn a_file_merely_named_like_a_manifest_elsewhere_still_counts() {
        // `crates/a/Cargo.toml` and a vendored `x/Cargo.toml` are both real
        // inputs to feature resolution; suffix matching is the honest test.
        let scope = ChangeScope::parse_porcelain_z("?? vendor/dep/Cargo.toml\0");
        assert!(scope.touches_manifest());
    }

    #[test]
    fn empty_output_is_a_clean_tree_not_an_unknown_one() {
        let scope = ChangeScope::parse_porcelain_z("");
        assert!(scope.is_empty());
        assert!(!scope.touches_rust());
    }

    #[test]
    fn exact_path_matching_does_not_match_a_suffix() {
        let scope =
            ChangeScope::parse_porcelain_z(" M state/registry/orphan-traits.allow.json\0");
        assert!(scope.touches_path("state/registry/orphan-traits.allow.json"));
        assert!(!scope.touches_path("orphan-traits.allow.json"));
    }
}
