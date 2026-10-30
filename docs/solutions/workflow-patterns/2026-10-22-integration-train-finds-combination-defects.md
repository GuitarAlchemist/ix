---
title: "Twelve PRs green one by one, three defects once merged: rehearse the merge order on one branch before landing a stack"
category: workflow-patterns
date: 2026-10-22
tags: [merge, stacked-prs, integration, ci, ix-loop, ix-doctor, workflow-tests, semantic-conflict]
symptom: "Every open PR had its own green CI, yet merging them in dependency order needed five conflict rounds and left the combined tree red in three places: an ix-loop unit test, the CI-trigger test, and ix doctor's hardcoded-paths check (verdict F)"
root_cause: "Each PR was tested against the main it branched from, not against the other PRs. A fix in one PR (a per-run reset) did not cover state another PR added later; a test in one PR pinned a workflow shape another PR changed on purpose; a doctor check in one PR scanned code another PR introduced"
---

# Twelve PRs green one by one, three defects once merged

## Problem

On 2026-10-22 twelve ix PRs waited for merge (#425, #426, #424, #383, #366, #367, #368, #393,
#394, #422, #423, #427), five of them stacked. Each had green CI. Merged in dependency order on
one branch (#428), they conflicted five times, and after the textual conflicts were resolved the
tree was still red in three places that no single PR's CI could have shown.

## The three defects

1. **A fix that predates the state it should cover.** #383's Codex round added a reset at the top
   of `Host::run` so a reused host gives each root run its own proposal allowance. #393, on a
   branch older than that fix, added two more per-run counters (`says`, `gh_reads`). Merged, the
   reset covered the old counters only: a second run on one host could neither speak nor read
   GitHub. Git saw no conflict there; the conflict was semantic.
2. **A test that pins a shape another PR changes on purpose.** #424's workflow test asserted
   `branches: [main]` on both CI triggers; #423 dropped the branch filter on `pull_request` so
   stacked PRs get CI. The textual merge of `ci.yml` was easy; the test went red on it.
3. **A new repo-wide check meeting new code.** #425 added the hardcoded-paths doctor check; #394
   added a crate whose doc example read `C:/Users/me/...`. Each PR passed `ix doctor` alone.

## What an adversarial review of the workflows added

Three reviewers, each with one lens (security, triggers, release chain), read the train's
workflow diff and found what the green rehearsal could not, because a CI filter or a release
job only runs on events a PR does not produce:

- **A paths filter that skips the job reading the files.** #424 excluded `!.claude/**` and
  `!**.md` from CI, while `ix doctor` reads `.claude/settings.json` and the hooks (#424's own
  check) and scans `crates/**/*.md` (#425's check). A hook-only or crate-README-only PR would
  merge without CI and turn the next unrelated PR red. Re-included, and pinned by the test.
- **A release step green on a failed test.** In `ixctl-release.yml`, `go vet` and `go test`
  ran in one `pwsh` step before `go build`; GitHub's `pwsh` wrapper exits with the last native
  command's code, and `$ErrorActionPreference = 'Stop'` does not cover native commands. Each
  `go` command is now followed by `if ($LASTEXITCODE) { throw … }`, and the release is a draft.

## What worked

- One local branch from `origin/main`, `git merge --no-ff` of each PR head in dependency order,
  resolving each conflict once, then the full gate (`ix doctor`, the CI clippy on stable and
  nightly, `cargo test --workspace --no-fail-fast`, the Go suite). The defects showed up only at
  that last step.
- Red-first tests for the semantic defects, committed on the train between the merges.
- Rebuilding the merge commits with `git commit-tree` (same trees, same second parents, clean
  messages) once the rehearsal was green, so the branch could land as is.
- Landing through one PR merged with a **merge commit**: each PR head is a second parent, so
  GitHub marks all of them merged. Stacked PRs are retargeted to `main` first for that.

## Gotchas

- `cargo test --workspace` stops at the first failing test binary; use `--no-fail-fast` to see
  every red one in one run.
- A fresh worktree has no `governance/demerzel` checked out: the ix-agent governance tests fail
  with "Failed to load constitution" until `git submodule update --init governance/demerzel`.
- `cargo fmt -p <crate>` rewrites files main already carries unformatted (CI runs no fmt
  check); revert those rather than widen the diff.
- Give the train's worktree a `CARGO_TARGET_DIR` of its own. Shared with a worktree of `main`,
  cargo served one the other's rlibs (path crates hash to the same artifact in both), and
  `cargo build -p ix-skill` failed with E0432 on a function the train has, while `cargo check`
  passed.
