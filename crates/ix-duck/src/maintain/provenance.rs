//! Iteration provenance (Phase 3a) — the anti-forgery check on the correlation key.
//!
//! The `commit_sha` is minted GA-side by the loop being judged, so the gate trusts it
//! only after (a) externally verifying it against real git state, and (b) confirming a
//! recorded loop row exists for the exact `(loop_id, commit_sha)`. Verifying the commit
//! alone is insufficient: a clean but unrelated commit for a loop with earlier improving
//! rows would otherwise be scored as that loop (Codex P1). The decision is PURE over the
//! [`Git`](super::git::Git) port (via [`super::git::commit_check`]), so the forged / dirty
//! / unverifiable / verified truth table is unit-testable without a real repo.

use std::path::Path;

use duckdb::Connection;

use super::git::{commit_check, CommitCheck, RealGit};
use super::{IterationScope, MaintainError};

/// Verify `sha` against real git state in `repo_dir` over the production [`RealGit`]
/// adapter (the decision logic lives in [`super::git::commit_check`], pure over the port).
pub(crate) fn verify_commit(repo_dir: &Path, sha: &str) -> CommitCheck {
    commit_check(&RealGit, repo_dir, sha)
}

/// Correlation: does a REAL recorded `loop_iterations` row exist for `(loop_id, commit_sha)`?
/// `None` when there is no scope; `Some(false)` when there is no ledger to match against.
/// Parameterised query — `loop_id`/`commit_sha` are caller-supplied.
pub(crate) fn key_matched(
    conn: &Connection,
    iteration: Option<&IterationScope>,
    loops_built: bool,
) -> Result<Option<bool>, MaintainError> {
    Ok(match (iteration, loops_built) {
        (Some(scope), true) => {
            let n: i64 = conn.query_row(
                "SELECT count(*) FROM loop_iterations WHERE loop_id = ? AND commit_sha = ?",
                duckdb::params![scope.loop_id, scope.commit_sha],
                |r| r.get(0),
            )?;
            Some(n > 0)
        }
        (Some(_), false) => Some(false), // scope given but no ledger to match against
        (None, _) => None,
    })
}

/// Why (if at all) an iteration's correlation key can't be trusted — `None` means
/// trusted: the commit is real, clean of tracked WIP, and a recorded loop row matches the
/// exact `(loop_id, commit_sha)`. The key is minted by the judged loop, so it earns the
/// same external-derivation discipline as the metric.
pub(crate) fn provenance_failure(
    iteration: Option<&IterationScope>,
    trust: Option<CommitCheck>,
    key_matched: Option<bool>,
) -> Option<&'static str> {
    iteration?; // no scope → nothing to verify
    match trust {
        Some(CommitCheck::Unverifiable) => Some("could not verify commit_sha (git unavailable)"),
        Some(CommitCheck::NotACommit) => Some("commit_sha is not a real commit (untrusted/forged)"),
        Some(CommitCheck::DirtyTracked) => {
            Some("tracked files modified — uncommitted WIP not captured by commit_sha")
        }
        _ if key_matched != Some(true) => Some("no recorded loop row for this loop_id/commit_sha"),
        _ => None,
    }
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::super::git::fake::FakeGit;
    use super::super::git::commit_check;
    use super::*;

    const SHA: &str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    /// A `IterationScope` whose lifetimes outlive the call.
    fn scope<'a>(loop_id: &'a str, sha: &'a str, repo: &'a Path) -> IterationScope<'a> {
        IterationScope { loop_id, commit_sha: sha, repo_dir: repo }
    }

    /// The forged / dirty / unverifiable / verified truth table — the payoff of routing
    /// the decision through the `Git` port. `provenance_failure` is fed the `CommitCheck`
    /// a `FakeGit` produces, plus a key-match flag.
    #[test]
    fn provenance_truth_table_over_fake_git() {
        let repo = Path::new(".");
        let s = scope("l", SHA, repo);

        // No scope → always trusted (nothing to verify), regardless of trust/key.
        assert_eq!(provenance_failure(None, Some(CommitCheck::NotACommit), Some(false)), None);

        let trust = |g: &FakeGit| commit_check(g, repo, SHA);

        // Forged (commit object does not exist) → untrusted, names "forged".
        let why = provenance_failure(Some(&s), Some(trust(&FakeGit::forged())), Some(true)).unwrap();
        assert!(why.contains("forged"), "{why}");

        // Dirty tracked files → untrusted, names "tracked".
        let why = provenance_failure(Some(&s), Some(trust(&FakeGit::dirty())), Some(true)).unwrap();
        assert!(why.contains("tracked"), "{why}");

        // Git could not run → untrusted, names "git unavailable".
        let why =
            provenance_failure(Some(&s), Some(trust(&FakeGit::unverifiable())), Some(true)).unwrap();
        assert!(why.contains("git unavailable"), "{why}");

        // Verified commit + matched key → trusted (None).
        assert_eq!(
            provenance_failure(Some(&s), Some(trust(&FakeGit::verified())), Some(true)),
            None
        );

        // Verified commit but NO matching loop row → untrusted (Codex P1 guard).
        let why =
            provenance_failure(Some(&s), Some(trust(&FakeGit::verified())), Some(false)).unwrap();
        assert!(why.contains("no recorded loop row"), "{why}");
    }
}
