# Change-type checklists

> **Before you open or merge a PR, run:**
>
> ```bash
> cargo run -p ix-skill --bin ix -- doctor
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
| `registry-snapshot` | The linked build's skills or MCP tools differ from `state/registry/skills.snapshot.json` | `ix doctor --write`, then review the **name** diff |
| `orphan-traits` | A `pub trait` has no implementor, no generic bound, and no allowlist entry | Delete it, implement it, or record it in `state/registry/orphan-traits.allow.json` with a `reason` |
| `dark-features` | A cargo feature no workspace member enables gates 100+ lines or at least one test, with no allowlist entry | Enable it from a workspace member, delete the code, or record it in `state/registry/dark-features.allow.json` with a `reason` |
| `demerzel-governance` | `governance/demerzel` is absent | `git submodule update --init` |
| `default-constitution` | The submodule is present but incomplete | Re-sync the submodule |
| `state-directory` | `state/` is missing from the repo root | You are not at the workspace root |

`ix check doctor` is a retained alias for the same run.

### What it deliberately does not check

`ix doctor` is a convenience wrapper, not a new authority layer. It does not run
pipeline-mesh demos, Epistemic SQL, or the assumption-graph — per ix#185 those
stay opt-in and must never become mandatory gates for an ordinary PR.

---

## Adding an MCP skill / tool

1. Write the algorithm in its domain crate. Annotate the wrapper with
   `#[ix_skill(...)]` so the link-time registry picks it up.
2. If the crate is new, add it to `ix_skill::force_link` in
   `crates/ix-skill/src/lib.rs` — otherwise LTO strips the distributed-slice
   entry and the registry silently comes up short.
3. Add the tool name to `EXPECTED` in `crates/ix-agent/tests/parity.rs`. This
   list stays hand-maintained on purpose: it is the rate-limiter that forces
   every surface change through review.
4. Run `ix doctor --write` and commit the `skills.snapshot.json` diff. The diff
   should show exactly the names you intended, and nothing else.
5. If the tool is behind a non-default cargo feature, add its name to
   `feature_gated_tools` in the snapshot so both build configurations stay green.
6. Run `cargo test -p ix-agent --test parity`.
7. Document it: `docs/MANUAL.md`, plus the French translation under `docs/fr/`.

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
2. Run `cargo run -p ix-skill --bin ix -- stable-surface`. The gate hashes
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
