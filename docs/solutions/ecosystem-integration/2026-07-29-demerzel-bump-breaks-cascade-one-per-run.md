---
title: "A Demerzel submodule bump is a migration, not a chore — five ix-side breaks surfaced one per test run"
category: ecosystem-integration
date: 2026-07-29
tags: [demerzel, submodule, governance, contracts, ix-fuzzy, ix-governance, hex-merge, cargo-test, migration]
symptom: "Bumping governance/demerzel turns main red; fixing the failure reveals a different one on the next run, repeatedly"
root_cause: "The bot's PR title (`N commits behind`) frames the bump as a chore, but it carries breaking governance-contract changes. Failures are revealed one at a time twice over: `cargo test` halts at the first failing binary, and the hex-merge fixtures all run inside ONE #[test] loop that panics on the first mismatch, which `--no-fail-fast` cannot see past"
---

# A Demerzel bump is a migration, not a chore

## Problem

PR #263 bumped `governance/demerzel` by 554 commits. It changed exactly one
line — a submodule pointer — and failed `build-and-test` on all four platforms.

Fixing that failure produced a *different* failure. Fixing that one produced
another. Five in total, each invisible until the one before it was green.

## Root cause

Three things compounding:

1. **The bump carries breaking contract changes.** Demerzel is consumed by ix as
   fixtures, policies and schemas. A "544 commits behind" PR title reads like a
   chore; the content included a reworked hex-merge specification (hari#27/#28)
   and a re-shaped policy file format.
2. **`cargo test` stops at the first failing test binary.** So the failure list
   is never complete. Each fix is also a *discovery step* — you cannot estimate
   remaining work from the first red run. `--no-fail-fast` lifts this one.
3. **The hex-merge fixtures run inside a single `#[test]`.**
   `crates/ix-fuzzy/tests/hex_merge_conformance.rs::every_fixture_matches_expected`
   loops over every fixture and asserts inline, so the first mismatching fixture
   panics and ends the loop. Breaks 1–3 below all surface through that one test,
   which is why they arrived one per run *even with* `--no-fail-fast` — it runs
   every test, but this is one test.

## The five breaks

| # | Break | Where |
|---|---|---|
| 1 | `resolve_key_versions`: dedup was first-write-wins, must be a min-weight fold with divergent variants collapsing to `Contradictory` | `ix-fuzzy/src/observations.rs` |
| 2 | Pipeline step 0: incoming `demerzel-merge` observations must be dropped and re-derived | `ix-fuzzy/src/observations.rs` |
| 3 | Escalation v1.1 → v1.2: `C` measured against *informative* mass (Unknown excluded), not full mass | `ix-fuzzy/src/hexavalent.rs` |
| 4 | `ref:confidence#<key>` tokens unresolved in the typed loader | `ix-governance/src/policy.rs` |
| 5 | Same tokens unresolved in the *generic* loader behind `ix describe policy` | `ix-governance/src/policy.rs` |

Breaks 1–3 came from one upstream revision; 4–5 from an unrelated one. A single
bump spanned two independent migrations.

### Break 1 is the instructive one

The old dedup was:

```rust
by_key.entry(obs.dedup_key()).or_insert_with(|| obs.clone());  // first-write-wins
```

`or_insert_with` is **order-dependent**: `T@1.0` then `F@0.5` kept `T`, while the
swapped input kept `F`. Three lines above it sat
`// @ai:invariant merge is commutative ... [T:test conf:0.95]`. The annotation
claimed a property the code did not have, and the fixture corpus is what caught
it — not the local tests.

### Break 5 is the trap

`ix-skill`'s test asserted `proceed_autonomously.is_number()`, which now failed
because the value was the string `"ref:confidence#autonomous"`. The tempting fix
is to relax the assertion to accept a string. That turns CI green while shipping
a real defect: `ix describe policy alignment` would print the raw token to a
user. **The assertion was right; the code was wrong.**

## Solution

Resolve tokens at load, in *both* loaders, against
`<demerzel>/logic/confidence-thresholds.yaml`:

```rust
match self {
    ThresholdSpec::Literal(v) => Ok(*v),          // pre-change policies still load
    ThresholdSpec::Ref(token) => {
        let key = token.strip_prefix(CONFIDENCE_REF_PREFIX)...;
        ladder.thresholds.get(key).map(|r| r.value)
            .ok_or_else(|| /* dangling key is an ERROR, not a default */)
    }
}
```

Generic resolution walks the whole document rather than a fixed field list, so a
policy that grows a new token needs no matching code change.

Shipped as PR #268 (open at time of writing; `build-and-test` green on CI).
Locally: `cargo test --workspace` → 236 binaries, 0 failures;
`cargo clippy --workspace --all-targets -- -D warnings` → exit 0.

## Prevention

**Treat every `chore: update Demerzel submodule` PR as a migration until proven
otherwise, and drive it from a local `cargo test --workspace` loop rather than
from CI.** The bot's title measures distance, not risk. Run
`cargo test --workspace --no-fail-fast` so every binary reports, then re-run the
*whole* suite after every fix and keep going until a run is clean. Using CI as
the oracle costs ~10 minutes per discovered break; the local loop costs seconds.

`--no-fail-fast` does not help inside an aggregate fixture test: until
`every_fixture_matches_expected` collects all mismatches before failing (or is
split into one test per fixture), the count of remaining hex-merge breaks stays
unknowable from a single red run. Collecting the failures is the durable fix.

Corollary: the bump and the ix-side fixes **cannot be split into separate PRs**.
Fixtures `08`–`12` are new in the bump, so there is nothing to validate the
`ix-fuzzy` changes against without it.

## Related

- [ix-duck bundled build fails under a long CARGO_TARGET_DIR](../build-errors/duckdb-bundled-build-fails-under-long-target-path.md)
- [streeling catalog drops sibling repo records from a worktree](../workflow-patterns/streeling-catalog-drops-sibling-repo-records-from-a-worktree.md)
