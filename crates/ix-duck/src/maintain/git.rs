//! Ports & adapters for the two git questions the provenance check asks: *does this
//! commit object exist?* and *are tracked files dirty?* The [`Git`] trait is the port;
//! [`RealGit`] is the production adapter (shelling out to `git`), and a test-only
//! `FakeGit` lets the provenance decision be unit-tested without a real repo.
//!
//! The decision is kept PURE over the trait ([`commit_check`]), preserving the
//! three-way distinction that makes the gate fail *correctly*:
//! - **could-not-run** → [`CommitCheck::Unverifiable`] (a missing-git environment is
//!   never misreported as a forgery alarm),
//! - **ran-and-answered-no** → [`CommitCheck::NotACommit`] (forged/wrong sha),
//! - real-but-dirty-tracked → [`CommitCheck::DirtyTracked`],
//! - real-and-clean → [`CommitCheck::Verified`].
//!
//! Hex-validation comes first, so a `sha` like `--help` can never reach git as a flag.

use std::path::Path;

/// Outcome of externally verifying an iteration's `commit_sha` against real git state.
/// Distinguishes "git could not answer" from "git says forged" so the gate never reports
/// a missing-git environment as a forgery alarm, and ignores other agents' untracked WIP
/// so a shared tree doesn't spuriously fail a valid iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CommitCheck {
    /// git could not be run at all (not installed / repo error) — we cannot decide.
    Unverifiable,
    /// The sha is malformed, or git ran and it is not a real commit object (forged/wrong).
    NotACommit,
    /// The commit is real, but TRACKED files have uncommitted edits — the metric may
    /// reflect work not captured by the sha. (Untracked files are deliberately ignored:
    /// a multi-agent tree is full of other agents' untracked WIP, which is not ours.)
    DirtyTracked,
    /// The commit is real and no tracked files are modified.
    Verified,
}

/// The two git questions the provenance check needs, as a port. Each returns `Some`
/// when git *answered*, `None` when git *could not be run* — preserving the
/// could-not-run vs. answered-no distinction the verdict depends on.
pub(crate) trait Git {
    /// Does `sha^{commit}` resolve to a real commit object in `repo_dir`?
    /// `Some(true)` = yes, `Some(false)` = ran and no, `None` = git could not run.
    fn commit_exists(&self, repo_dir: &Path, sha: &str) -> Option<bool>;

    /// Are any TRACKED files modified in `repo_dir` (untracked WIP excluded)?
    /// `Some(true)` = dirty, `Some(false)` = clean, `None` = git could not run.
    fn tracked_dirty(&self, repo_dir: &Path) -> Option<bool>;
}

/// Production adapter: shells out to `git` (`cat-file -e` then
/// `status --porcelain --untracked-files=no`).
pub(crate) struct RealGit;

impl Git for RealGit {
    fn commit_exists(&self, repo_dir: &Path, sha: &str) -> Option<bool> {
        use std::process::Command;
        match Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .args(["cat-file", "-e", &format!("{sha}^{{commit}}")])
            .output()
        {
            Err(_) => None, // git couldn't run (≠ "forged")
            Ok(o) => Some(o.status.success()),
        }
    }

    fn tracked_dirty(&self, repo_dir: &Path) -> Option<bool> {
        use std::process::Command;
        // `--untracked-files=no` excludes other agents' untracked WIP, so a shared tree
        // doesn't spuriously fail a valid iteration.
        match Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .args(["status", "--porcelain", "--untracked-files=no"])
            .output()
        {
            Ok(o) if o.status.success() => Some(!o.stdout.is_empty()),
            _ => None,
        }
    }
}

/// Decide [`CommitCheck`] over a [`Git`] port — PURE given the port's answers.
/// Hex-validates first (so a `sha` like `--help` can't reach git as a flag), then asks
/// existence, then tracked-dirtiness. A git invocation that fails to *run* (vs answers
/// "no") is [`CommitCheck::Unverifiable`], not [`CommitCheck::NotACommit`].
pub(crate) fn commit_check(git: &dyn Git, repo_dir: &Path, sha: &str) -> CommitCheck {
    let hex = !sha.is_empty() && sha.len() <= 64 && sha.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex {
        return CommitCheck::NotACommit;
    }
    match git.commit_exists(repo_dir, sha) {
        None => return CommitCheck::Unverifiable, // git couldn't run
        Some(false) => return CommitCheck::NotACommit,
        Some(true) => {}
    }
    match git.tracked_dirty(repo_dir) {
        Some(false) => CommitCheck::Verified,
        Some(true) => CommitCheck::DirtyTracked,
        None => CommitCheck::Unverifiable,
    }
}

#[cfg(all(test, feature = "duck"))]
pub(crate) mod fake {
    use super::*;

    /// A scriptable [`Git`] for unit-testing the provenance decision without a real repo.
    /// `exists`/`dirty` are the canned answers (`None` models "git could not run").
    pub(crate) struct FakeGit {
        pub exists: Option<bool>,
        pub dirty: Option<bool>,
    }

    impl FakeGit {
        pub fn verified() -> Self {
            Self { exists: Some(true), dirty: Some(false) }
        }
        pub fn forged() -> Self {
            Self { exists: Some(false), dirty: Some(false) }
        }
        pub fn dirty() -> Self {
            Self { exists: Some(true), dirty: Some(true) }
        }
        /// git could not run at all (not installed / repo error).
        pub fn unverifiable() -> Self {
            Self { exists: None, dirty: None }
        }
    }

    impl Git for FakeGit {
        fn commit_exists(&self, _repo_dir: &Path, _sha: &str) -> Option<bool> {
            self.exists
        }
        fn tracked_dirty(&self, _repo_dir: &Path) -> Option<bool> {
            self.dirty
        }
    }
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::fake::FakeGit;
    use super::*;

    fn check(git: &dyn Git, sha: &str) -> CommitCheck {
        commit_check(git, Path::new("."), sha)
    }

    const GOOD_SHA: &str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    #[test]
    fn non_hex_sha_is_not_a_commit_without_touching_git() {
        // A flag-like sha must be rejected by hex-validation before reaching git.
        assert_eq!(check(&FakeGit::verified(), "--help"), CommitCheck::NotACommit);
        assert_eq!(check(&FakeGit::verified(), ""), CommitCheck::NotACommit);
    }

    #[test]
    fn three_way_distinction_is_preserved() {
        assert_eq!(check(&FakeGit::verified(), GOOD_SHA), CommitCheck::Verified);
        assert_eq!(check(&FakeGit::forged(), GOOD_SHA), CommitCheck::NotACommit);
        assert_eq!(check(&FakeGit::dirty(), GOOD_SHA), CommitCheck::DirtyTracked);
        // could-not-run is Unverifiable, never a forgery alarm.
        assert_eq!(check(&FakeGit::unverifiable(), GOOD_SHA), CommitCheck::Unverifiable);
    }

    #[test]
    fn exists_but_status_unrunnable_is_unverifiable_not_dirty() {
        let git = FakeGit { exists: Some(true), dirty: None };
        assert_eq!(check(&git, GOOD_SHA), CommitCheck::Unverifiable);
    }
}
