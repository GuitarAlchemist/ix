# Change-type checklists

> **Before you open or merge a PR, run:**
>
> ```bash
> cargo run -p ix-cli --bin ix -- doctor
> ```
>
> Add `--full` to also run the clippy and test invocation CI uses. `ix doctor`
> exits `0` when everything is green, `1` on warnings only, `4` on a failure,
> and prints what to do about each one. French: [`docs/fr/CHECKLISTS.md`](fr/CHECKLISTS.md).

IX has a lot of seams: a link-time capability registry, an MCP tool surface,
DuckDB UDFs, stable-surface hashes, `@ai:` annotations, EN/FR docs, catalog
snapshots. Each is cheap on its own and expensive to *remember*. These lists
encode the seams so you do not rediscover them by breaking CI (ix#185).

Every list below is deliberately short. If a step can be mechanised, it belongs
in `ix doctor` instead of here.

---

## What `ix doctor` checks

| Check | What fails it | Fix |
| --- | --- | --- |
| `registry-snapshot` | The linked build's registry tools or MCP tools differ from `state/registry/tools.snapshot.json` | `ix doctor --write`, then review the **name** diff |
| `orphan-traits` | A `pub trait` has no implementor, no generic bound, and no allowlist entry | Delete it, implement it, or record it in `state/registry/orphan-traits.allow.json` with a `reason` |
| `dark-features` | A cargo feature no workspace member enables gates 100+ lines or at least one test, with no allowlist entry | Enable it from a workspace member, delete the code, or record it in `state/registry/dark-features.allow.json` with a `reason` |
| `silent-skips` | A `#[test]` returns early when its environment is missing and so can pass without asserting, with no allowlist entry | Make the skip impossible where the test must run (install the tool in CI, fail closed on a "required" variable), or record it in `state/registry/silent-skips.allow.json` with a `reason` |
| `hardcoded-paths` | A line of a source or script file under `crates/` (`.rs`, `.sql`, `.ps1`, `.sh`, `.py`, `.toml`, `.md`, any case) names one developer's home directory (`C:\Users\<name>`, `\Users\<name>`, `/home/<name>`, `/Users/<name>`) with no allowlist entry | Resolve from an env var, `PATH` or the workspace root (`../<repo>` for siblings) and fail naming those options; doc examples use `../ga/...`; a line whose text must keep such a path (a test's input, a URL containing `/home/`) goes in `state/registry/hardcoded-paths.allow.json` with `file`, `contains`, `reason` |
| `hooks` | A hook wired in `.claude/settings.json` is missing, a `.claude/hooks/*.sh` does not parse (`bash -n`), or a fixture `.claude/hooks/test-*.sh` exits non-zero; warns when no bash is found (Git bash on Windows) | Restore or drop the hook entry; `bash -n` the script; run the fixture by hand, its last lines are in `details.fixtures[].tail` |
| `rustfmt-touched` | A `.rs` file the change adds or modifies (committed, staged or untracked since the merge-base with `origin/main`, or with `IX_DOCTOR_BASE`, which CI sets) is not what `rustfmt` in the workspace edition gives back, and was formatted at the base; warns on drift the base already had, on a file rustfmt cannot parse, and when rustfmt or `origin/main` is missing (fails instead when `IX_DOCTOR_BASE` is set) | `rustfmt --edition 2021 <file>` on the named files, never `cargo fmt --all` (main carries drift in files other work is editing); a fixture that must stay unformatted takes `#![rustfmt::skip]` |
| `demerzel-governance` | `governance/demerzel` is absent | `git submodule update --init` |
| `default-constitution` | The submodule is present but incomplete | Re-sync the submodule |
| `state-directory` | `state/` is missing from the repo root | You are not at the workspace root |

`ix check doctor` is a retained alias for the same run.

### What it deliberately does not check

`ix doctor` is a convenience wrapper, not a new authority layer. It does not run
pipeline-mesh demos, Epistemic SQL, or the assumption-graph — per ix#185 those
stay opt-in and must never become mandatory gates for an ordinary PR.

---

## Adding an `ix doctor` check

Five sites, all in the same PR. Two PRs open at once each touched all five and
conflicted on every one of them; this list exists so the next one sees them
before opening the diff.

1. `crates/ix-cli/src/doctor/<name>.rs` exposing
   `pub fn check(root: &Path) -> super::CheckResult`. Every `Warn` or `Fail`
   result carries a `remedy`: `every_non_ok_check_carries_a_remedy` enforces it.
2. `pub mod <name>;` and the call inside `run()` in
   `crates/ix-cli/src/doctor/mod.rs`.
3. A row in the table above, in this file and in `docs/fr/CHECKLISTS.md`.
4. The `Doctor` doc comment in `crates/ix-cli/src/main.rs`: it is the
   `ix doctor --help` text, and it lists the checks by name.
5. If the check has an allowlist, `state/registry/<name>.allow.json`, every
   entry with a `reason`; a stale entry warns, a reasonless one fails.

Prove it before it lands: the check must be red on this workspace before its
fixes and green after (the fixture tests on scratch repos pin the behaviour; one
test scans the live workspace). In CI it runs twice: through
`doctor_is_green_on_this_workspace` (`cargo test -p ix-cli`) and as the
`ix doctor` step of `.github/workflows/ci.yml`, where it has a line of its own.

---

## Adding an MCP tool

1. Write the algorithm in its domain crate. Annotate the wrapper with
   `#[ix_tool(...)]` so the link-time registry picks it up.
2. If the crate is new, add it to `ix_cli::force_link` in
   `crates/ix-cli/src/lib.rs` — otherwise LTO strips the distributed-slice
   entry and the registry silently comes up short.
3. Add the tool name to `EXPECTED` in `crates/ix-agent/tests/parity.rs`. This
   list stays hand-maintained on purpose: it is the rate-limiter that forces
   every surface change through review.
4. Classify it in `crates/ix-approval/src/classify.rs` under the kind its
   effects warrant: pure computation or read → `READ_TOOLS` (Tier 1); writes to
   the workspace or to in-process state such as the global cache →
   `EDIT_IN_PROJECT_TOOLS` (Tier 2); shell, web fetch or out-of-project writes →
   `SHELL_COMMAND_TOOLS`, `WEB_FETCH_TOOLS` or `EDIT_OUT_OF_PROJECT_TOOLS`
   (Tier 3, refused). Tier 1 and Tier 2 tools run without a prompt, so a path,
   directory or repo root the caller names must be confined before it is read,
   walked or written: use `confine` (or `confine_in` / `confine_dest_in` for an
   explicit root list or a directory the tool creates) from
   `crates/ix-agent/src/path_confine.rs`. They admit the workspace root and the
   directories in `IX_EXTRA_ROOTS`; add a tool-specific root only when the
   operator, not the caller, chose it (the GA trace tools admit `~/.ga/traces`
   and the `traces/` directory beside the installed session log). Add the tool
   to `auto_approved_tools_refuse_paths_outside_the_workspace` in
   `crates/ix-agent/tests/parity.rs`. An unclassified
   registry-backed tool defaults to Tier 3 and every MCP call to it is refused;
   the parity test
   `every_registry_backed_tool_has_an_explicit_approval_classification` fails
   until it is classified.
5. Run `ix doctor --write` and commit the `tools.snapshot.json` diff. The diff
   should show exactly the names you intended, and nothing else.
6. If the tool is behind a non-default cargo feature, add its name to
   `feature_gated_tools` in the snapshot so both build configurations stay green.
7. Run `cargo test -p ix-agent --test parity`.
8. Document it: `docs/MANUAL.md`, plus the French translation under `docs/fr/`.

## Adding a DuckDB UDF

1. Implement it in `ix-duck` / `ix-duck-ext` and register it in that crate's
   registration function — a UDF that is written but not registered is invisible.
2. Add a test that executes the UDF through a real DuckDB connection, not just
   the Rust function underneath it. Registration is the part that breaks.
3. Keep the terse hand style used in `ix-duck`; do **not** run `cargo fmt` over
   it (see `CLAUDE.md` — repo-wide fmt skew is advisory, not a gate).
4. If the UDF is exposed as an MCP tool as well, follow the MCP list above too.
5. Note the feature flag. `maintain-gate` pulls bundled DuckDB and is off in the
   default/CI build; anything gated behind it must not change the default surface.
6. Document the signature in `docs/DUCKDB.md` and its French counterpart.

## Adding or changing a public crate API

1. Check the crate's tier in `crate-maturity.toml`. **Stable** crates are gated.
2. Run `cargo run -p ix-cli --bin ix -- stable-surface`. The gate hashes
   `pub `-prefixed lines, so adding a `pub fn` trips it while changing a
   function body does not.
3. If a Stable crate's hash changes, the PR needs an explicit version bump or a
   tier demotion — not a silent hash update.
4. New `pub trait`? It needs an implementor or a generic bound in-tree, or
   `ix doctor` will fail the `orphan-traits` check. That is the point: a
   declared contract nothing satisfies is either dead code or a promise the
   codebase does not keep. `ix-io`'s `DataSource` / `DataSink` were the
   motivating case (ix#299): the module doc claimed every backend implemented
   them while none did. The fix is worth copying — implementors *plus* a
   generic consumer (`protocol::pump`), and a written reason in each module
   that still does not implement the trait.

## Adding a feature-gated module

1. Ask first whether the gate is needed. It is the right tool for an optional
   heavy dependency — DuckDB, arrow, ONNX Runtime — and the root `Cargo.toml`
   says so. It is the wrong tool for "this is not ready yet": a `#[cfg]` is not
   a draft marker.
2. If you gate it, something must still compile it. A feature that **no
   workspace member enables** is never built by `cargo build --workspace`,
   `cargo clippy --workspace` or `cargo test --workspace` — the three
   invocations CI runs. Its tests are neither passed nor failed; they are
   absent, and a skipped crate adds zero to both counts, so nothing in the
   repository's output tells you (ix#315).
3. `ix doctor` fails the `dark-features` check on any such feature gating 100+
   lines or at least one test. It reports the crate, the feature, the modules,
   and how many lines and tests are behind them — a gate hiding two lines is
   noise, one hiding 95 tests is not.
4. The fix, in order of preference: enable the feature from a workspace member
   that wants it (a dev-dependency counts); delete the code; or add an entry to
   `state/registry/dark-features.allow.json`. An entry needs a non-empty
   `reason` and a `kind`:
   - `environment` — enabling it here is blocked or unreasonably costly for
     reasons outside the code (native toolchain, external binary, GPU, a
     toolchain newer than the workspace MSRV). Expected to stay.
   - `tracked` — nothing environmental stops it, it just is not wired up yet.
     **Requires an `issue`**, and is counted as outstanding debt in the summary
     so it does not become a parking space.
5. `ix-code`'s `topology` feature is the motivating case: 296 lines and five
   tests, merged and reviewed, never once compiled. The code was healthy —
   `cargo test -p ix-code --features full` passes — which is exactly why nobody
   noticed.

## Adding an `@ai:` invariant

1. Follow `docs/contracts/2026-05-24-ai-annotation.contract.md`. Format:
   `// @ai:invariant <claim> [T:test conf:0.95 src:path::to::test]`.
2. `certainty := strength of live binding`. Do not write `[T]` without a live
   binding (a test, the compiler, or sentrux); cap human-only claims at
   `P:assumed`; surface unenforced assumptions as `@ai:assumption [U:uncertain]`.
3. Keep `src:` a single clean token — the drift gate only binds `[T:test]` when
   `src:` is `::`-pathed or `test_`-prefixed.
4. Annotate one module at a time, and make each pass drive at least one real
   fix. Never mass-generate annotations.
5. Rust only for a `[T:test]` binding: the drift gate reads `@ai:` markers in every source file
   (Go included) but resolves the cited test among Rust functions only, so a `[T:test src:…::TestX]`
   in a Go file is a broken binding that fails the next `crates/**` PR (the gate's path filter does
   not run it on `go/**`; reproduce with `cargo run -p ix-assumption-graph --bin
   ix-assumption-graph-drift -- --check state/assumptions/annotations.snapshot.json`, exit 2). A
   marker without a test binding (`[P:assumed conf:0.7]`, `[U:uncertain]`) is safe in Go: the gate
   verifies bindings only. For a test-backed claim in Go, point the comment at the contract row
   instead.

## Adding a showcase or demo

1. Register the scenario so `ix demo list` finds it — an unregistered scenario
   is invisible regardless of how well it runs.
2. Seed every RNG. Demos are reproducibility surfaces; an unseeded demo produces
   a different transcript each run and cannot be diffed.
3. Keep it off the default test path if it is slow. Per ix#185, normal Rust
   iteration must not be blocked behind heavyweight demos.

## Changing docs that feed Streeling or a catalog

1. Terminology goes in `CONTEXT.md`. Architectural decisions go in `docs/adr/`.
   Session learnings go in `docs/solutions/` via `/learnings`. Do not leave
   loose unindexed markdown at the repo root.
2. Re-run the `ix-streeling` indexer so `state/streeling/catalog.jsonl` stays
   current.
3. Maintain the French translation alongside the English one, under `docs/fr/`.


## Changing a behavioural claim about a plane or a contract

A sentence that says what the plane does ("the list is exactly what `ctl` says up to the cursor",
"a consumer keys the series from the status") is a contract term, and it drifts when it is written
in several places: on the S2 slice-4 session two such claims lived in README EN, README FR, the
contract and a code comment, and two of the four copies went stale before Codex read them.

1. The contract under `docs/contracts/` is the single source of the claim; README paragraphs and
   code comments **point at it** (the row or the section) instead of restating it. A restatement
   that must stay (the one-line summary in a status field's doc comment) names the contract version.
2. Before the commit, grep every word of the claim (`exact`, `without having read`, `outside the
   lock`) across `docs/`, `go/ixctl/*.md` and the code comments, and update or remove every hit:
   the Order section of the contract still said a leaf's `probe.stopped` was published outside the
   lock one slice after that stopped being true.
3. A count or a total that a gate knows (the number of checks, the highest check index) is asserted
   by the gate, not typed into prose: `go/ixctl/e2e/e2e.py` E24 compares the totals in both READMEs,
   its own docstring and the gate comment of `.github/workflows/ci.yml` (every statement of each
   text) with what ran, so a check added without the texts following it fails.
4. The FR mirror changes in the same commit as the EN text, paragraph for paragraph.
5. A test added for the claim is one of two kinds, and the commit message says which: a **fix** has a
   red run against the previous tree (the failure text quoted); a **pin** protects a behaviour that
   already held and cannot go red by itself (a race the schedule decides, an order the loop owns).
   A pin declares its red "by the seam" and names what it protects, so a reader does not look for a
   failure that never existed (S2 slice 4: three of five tests in one commit were pins).
6. A test that depends on the kernel passes on one OS and fails on another: a test that waits for an
   overflow or a jump sets the server's `writeDelay` (or the link's `pipeDelay` / `ctlDelay`) and
   never counts on socket buffers filling; run `pwsh scripts/ixctl-test-linux.ps1` before the push
   (the first CI run of #423 lost two flood tests on ubuntu-latest that Windows passed).
7. A contract announcement (a version a consumer gates on, such as status `contract: "run/1"`) is
   set by the slice that completes the contract, never by the first one: a consumer that reads it
   relies on every promise behind it at once. Earlier slices test that it is absent. Slice A-1 of
   run/1 announced it while markers, recent exits, federation and apply results were still missing;
   an adversarial review caught it, and `TestStatusWithholdsTheRunContractUntilItShips` pins the
   absence until the last slice (N1, #444).

## Writing into a store that others read

A file that a person accepted, a run published or another process reads back (a library, a
receipt, a committed result) is shared state, even when one process writes it.

1. Create it with `create_new` (`OpenOptions::new().write(true).create_new(true)`), not
   `fs::write`: a taken name is then a visible conflict, never a silent replacement. Checking for
   the file first and writing after leaves a window for another writer. `ix-loop`'s
   `Store::accept` replaced an accepted loop that way until the ix#485 inventory found it.
2. Replacing on purpose is its own named operation, which the reviewer can see in the diff.
3. A refused write leaves its input where it was (the proposal stays pending), so nothing is lost.
