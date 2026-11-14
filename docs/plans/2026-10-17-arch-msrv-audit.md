---
title: MSRV audit — what each IX package's dependency closure declares
type: arch
status: draft
date: 2026-10-17
issue: F4 of GuitarAlchemist/ix#386 (Astra review of exp/ix-recover at 3d59043)
tags: [msrv, rust-version, cargo, dependencies]
reversibility: two-way (no IX crate is published to crates.io); decision pending
revisit-trigger: a lock refresh, wgpu past 28, or a consumer's build failure on an older toolchain → rerun the audit
---

# MSRV audit — what each IX package's dependency closure declares

- **Date**: 2026-10-17
- **Status**: proposed. The measurement is done; the decision in §6 is the
  operator's.
- **Reversibility**: this document is a plan and changes no code. Each option
  in §5 states its own door; all of them are **two-way doors**, because no IX
  crate is published to crates.io (no workflow runs `cargo publish`).
  **Revisit / delete-by trigger**: see §7.
- **Origin**: finding F4 of the Astra review of `exp/ix-recover` at `3d59043`,
  GuitarAlchemist/ix#386. The review reclassified F4 as "a separate
  compatibility follow-up" and asked for "an explicit compatibility decision
  using the affected package/feature/target dependency closures". This
  document is that follow-up.
- **One-way-door check**: no option touches a schema hash, a Galactic Protocol
  contract or a stable-tier API signature. A `rust-version` is a support
  statement, not an interface.

## 1. The problem

Every workspace member inherits `rust-version = "1.80"` (`Cargo.toml:98`).
`CLAUDE.md:33` and `README.md:475` say "MSRV: Rust 1.80+ (due to wgpu 28)".
But wgpu 28.0.0 declares Rust 1.92, so that sentence contradicts itself, and
the declaration is false for the CLI (`ix-cli`) and the MCP server
(`ix-agent`).

Two groups are hurt:

- **Someone building IX with Rust 1.80 to 1.91.** By Cargo's documented
  behaviour, the build stops early with a "requires rustc 1.92" error, so the
  failure is loud, not silent. The harm is a false promise and wasted setup.
  *Assumed*: Cargo's behaviour is documented, but was not run here, because no
  old toolchain is installed.
- **A reviewer asking whether a change raised the minimum.** F4 could not
  answer that, because no baseline had ever been measured. This audit is that
  baseline.

## 2. Method, and what it does not measure

- **Workspace and tools.** origin/main at `008f07c`, with Cargo 1.94.0.
- **The lock.** The local `Cargo.lock` has SHA-256
  `8b896cbbb499eb311dde8899145cd5c10c64900d500dd973d4ccd3336a9658c6`.
  `Cargo.lock` is gitignored (`.gitignore:3`), so these numbers come from one
  resolution on one machine. They are not a property of the repository.
- **The runs.** All 83 members were measured, each on three targets
  (`x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`,
  `aarch64-apple-darwin`) and two feature sets (default and
  `--all-features`). Each run is:

  ```text
  cargo tree --offline --locked -p <member>@<version> -e normal,build \
    --target <target> --prefix none --format "{p}" [--all-features]
  ```

  Each closure package's `rust_version` comes from
  `cargo metadata --offline --locked --format-version 1`. dev-dependencies
  are left out, because they never build for a consumer.

What this does **not** establish:

1. **It reads declared bounds only.** A crate may need more than it declares,
   or build with less. Nothing was compiled with an old toolchain. In F4's own
   words, "stable/nightly success is not an MSRV check", and neither is this.
   The result is a lower bound on what Cargo will accept, not a proven
   minimum.
2. **165 distinct crates declare no `rust-version`.** They are counted as
   unknown, not as compatible.
3. **The resolver is not MSRV-aware.** The workspace uses
   `resolver = "2"`, so a fresh `cargo update`, or a CI checkout without a
   lock, may resolve newer versions that declare more. The numbers move with
   the lock.
4. **One feature cell is missing.** `ix-cli --all-features` did not resolve
   offline on any target, because a feature-only crate is not in the local
   cache.
5. **Only the two feature extremes were measured.** Single features in
   between were not enumerated.

## 3. Findings

- 81 of the 83 members declare 1.80 through the workspace. `ix-optick` and
  `ix-optick-invariants` declare nothing.
- **35 of the 83 members** have a closure whose highest declaration is above
  their own.
- **The target never changed a maximum.** For every member and feature set,
  all three targets gave the same highest declaration. Closure sizes do
  differ: `ix-cli` with default features has 342, 328 and 320 packages on
  Windows, Linux and macOS.
- **The worst case**, over targets and feature sets, sorts the members like
  this:

  | Highest declared | Members |
  |---|---|
  | 1.92 | 6 |
  | 1.90 | 3 |
  | 1.89 | 7 |
  | 1.88 | 7 |
  | 1.85 | 12 |
  | ≤ 1.80, or none | 48 |

- **Four members change maximum with features:**
  - `memristive-markov`: 1.77 → 1.92
  - `ix-code`: 1.80 → 1.90
  - `ix-duck`: none → 1.90
  - `ix-baml`: 1.77 → 1.89
- **The drivers:**
  - 1.92: wgpu 28.0.0, reached through `ix-gpu`.
  - 1.90: tree-sitter-language 0.1.9 and ordered-float 5.5.0.
  - 1.89: uuid 1.28.0.
  - 1.88: icu\_\* 2.3.0, encoding_rs 0.8.42, ignore 0.4.33 / globset 0.4.20,
    darling 0.24.1.
  - 1.85: clap 4.6.7, indexmap 2.14.2, hashbrown 0.17.1, rmp 0.8.15.

The 35 members, with the same maximum on all three targets. "none declared"
means that no package in that closure declares a `rust-version`.

| Member | Declares | Closure max, default features | Closure max, all features | Highest declarations above the member's own |
|---|---|---|---|---|
| memristive-markov | 1.80 | 1.77.0 | 1.92.0 | wgpu 28.0.0 (1.92), ordered-float 5.5.0 (1.90), indexmap 2.14.2 (1.85) … |
| ix-cli | 1.80 | 1.92.0 | not resolved offline | wgpu 28.0.0 (1.92), ordered-float 5.5.0 (1.90), tree-sitter-language 0.1.9 (1.90) … |
| ix-nn | 1.80 | 1.92.0 | 1.92.0 | wgpu 28.0.0 (1.92), ordered-float 5.5.0 (1.90), indexmap 2.14.2 (1.85) … |
| ix-gpu | 1.80 | 1.92.0 | 1.92.0 | wgpu 28.0.0 (1.92), ordered-float 5.5.0 (1.90), indexmap 2.14.2 (1.85) … |
| ix-demo | 1.80 | 1.92.0 | 1.92.0 | wgpu 28.0.0 (1.92), tree-sitter-language 0.1.9 (1.90), ordered-float 5.5.0 (1.90) … |
| ix-agent | 1.80 | 1.92.0 | 1.92.0 | wgpu 28.0.0 (1.92), ordered-float 5.5.0 (1.90), tree-sitter-language 0.1.9 (1.90) … |
| ix-duck | 1.80 | none declared | 1.90.0 | tree-sitter-language 0.1.9 (1.90), icu_collections 2.3.0 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-context | 1.80 | 1.90.0 | 1.90.0 | tree-sitter-language 0.1.9 (1.90), uuid 1.28.0 (1.89.0), icu_collections 2.3.0 (1.88) … |
| ix-code | 1.80 | 1.80.0 | 1.90.0 | tree-sitter-language 0.1.9 (1.90), uuid 1.28.0 (1.89.0), icu_collections 2.3.0 (1.88) … |
| ix-quality-validate | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), icu_collections 2.3.0 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-quality-trend | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), cpufeatures 0.3.1 (1.85), getrandom 0.4.3 (1.85) … |
| ix-optick-sae | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), clap_lex 1.1.1 (1.85), clap 4.6.7 (1.85) … |
| ix-ixql | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), icu_collections 2.3.0 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-governance | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), cpufeatures 0.3.1 (1.85), constant_time_eq 0.4.2 (1.85.0) … |
| ix-baml | 1.80 | 1.77.0 | 1.89.0 | uuid 1.28.0 (1.89.0), icu_normalizer_data 2.3.0 (1.88), icu_collections 2.3.0 (1.88) … |
| ix-autoresearch | 1.80 | 1.89.0 | 1.89.0 | uuid 1.28.0 (1.89.0), cpufeatures 0.3.1 (1.85), constant_time_eq 0.4.2 (1.85.0) … |
| ix-voice | 1.80 | 1.88.0 | 1.88.0 | icu_normalizer_data 2.3.0 (1.88), icu_collections 2.3.0 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-sentrux-annotations | 1.80 | 1.88.0 | 1.88.0 | ignore 0.4.33 (1.88), globset 0.4.20 (1.88), clap_lex 1.1.1 (1.85) … |
| ix-io | 1.80 | 1.88.0 | 1.88.0 | icu_collections 2.3.0 (1.88), encoding_rs 0.8.42 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-dashboard | 1.80 | 1.88.0 | 1.88.0 | darling_macro 0.24.1 (1.88.0), instability 0.3.14 (1.88), darling_core 0.24.1 (1.88.0) … |
| ix-assumption-graph | 1.80 | 1.88.0 | 1.88.0 | ignore 0.4.33 (1.88), globset 0.4.20 (1.88), clap_lex 1.1.1 (1.85) … |
| ix-ai-annotations | 1.80 | 1.88.0 | 1.88.0 | ignore 0.4.33 (1.88), globset 0.4.20 (1.88), clap_lex 1.1.1 (1.85) … |
| ga-chatbot | 1.80 | 1.88.0 | 1.88.0 | icu_collections 2.3.0 (1.88), encoding_rs 0.8.42 (1.88), icu_locale_core 2.3.0 (1.88) … |
| ix-voicings | 1.80 | 1.85.0 | 1.85.0 | clap_lex 1.1.1 (1.85), clap 4.6.7 (1.85), rmp-serde 1.3.1 (1.85) … |
| ix-value | 1.80 | 1.85.0 | 1.85.0 | clap_derive 4.6.7 (1.85), clap_builder 4.6.7 (1.85), clap_lex 1.1.1 (1.85) … |
| ix-streeling | 1.80 | 1.85.0 | 1.85.0 | clap_lex 1.1.1 (1.85), clap 4.6.7 (1.85), indexmap 2.14.2 (1.85) … |
| ix-pipeline | 1.80 | 1.85.0 | 1.85.0 | hashbrown 0.17.1 (1.85.0), indexmap 2.14.2 (1.85) |
| ix-optick-invariants | none | 1.85.0 | 1.85.0 | clap_lex 1.1.1 (1.85), rmp 0.8.15 (1.85), clap 4.6.7 (1.85) … |
| ix-optick | none | 1.85.0 | 1.85.0 | rmp-serde 1.3.1 (1.85), rmp 0.8.15 (1.85), thiserror 2.0.21 (1.77) … |
| ix-invariant-coverage | 1.80 | 1.85.0 | 1.85.0 | clap_derive 4.6.7 (1.85), clap_builder 4.6.7 (1.85), clap_lex 1.1.1 (1.85) … |
| ix-harness-signing | 1.80 | 1.85.0 | 1.85.0 | zeroize 1.9.1 (1.85), ed25519-dalek 2.2.0 (1.81) |
| ix-harness-github-actions | 1.80 | 1.85.0 | 1.85.0 | hashbrown 0.17.1 (1.85.0), indexmap 2.14.2 (1.85) |
| ix-embedding-diagnostics | 1.80 | 1.85.0 | 1.85.0 | clap_lex 1.1.1 (1.85), clap 4.6.7 (1.85), rmp-serde 1.3.1 (1.85) … |
| ix-blast-radius | 1.80 | 1.85.0 | 1.85.0 | clap_derive 4.6.7 (1.85), clap_builder 4.6.7 (1.85), clap_lex 1.1.1 (1.85) … |
| ix-autograd | 1.80 | 1.85.0 | 1.85.0 | cpufeatures 0.3.1 (1.85), constant_time_eq 0.4.2 (1.85.0) |

## 4. The answer to F4's narrow question

F4 asked whether `ix-recover`'s 1.89 raises the CLI's minimum. This was
measured with the same method on `exp/ix-recover` at `3f2e935`, with default
features, on Windows and Linux:

- **ix-recover** declares 1.89, for its own use of `File::try_lock`, which
  is stable since 1.89. Its closure has 21 packages, itself included, and
  declares at most 1.71 (proc-macro2 1.0.107, quote 1.0.47, serde_json
  1.0.154, zmij 1.0.23).
- **ix-cli on that branch** has 344 packages on Windows and 330 on Linux,
  itself included (§3 counts exclude the member).
  Its highest declarations are wgpu 28.0.0 (1.92), tree-sitter-language
  0.1.9 (1.90), ordered-float 5.5.0 (1.90) and uuid 1.28.0 (1.89.0).
- **ix-cli on main**, without ix-recover, already declares 1.92 (§3).

So, by declared bounds and with this lock, adding ix-recover does **not**
raise the CLI's highest declaration: its 1.89 is below the 1.92 already
there. This agrees with the review's own correction. It does not prove that
the CLI builds on 1.92 (limit 1 in §2).

## 5. Options

**A. Truthful declarations, member by member.**
- **The change.** Set each member's `rust-version` to the higher of its own
  need and its default-feature closure's highest declaration. Members where
  1.80 is true keep 1.80. That changes 35 manifests; the two `ix-optick`
  crates gain a declaration.
- **Feature-only raises.** `memristive-markov`, `ix-code`, `ix-duck` and
  `ix-baml` need more only with some features. A `rust-version` is set per
  package, not per feature, so these get a comment next to the feature, as
  `crates/ix-duck/Cargo.toml:19-22` already does for duckdb.
- **Gain.** Small crates such as `ix-math` and `ix-types` keep a true 1.80.
  This follows the review: "do not raise every independently consumable crate
  to 1.92 merely because one workspace member needs it".
- **Cost.** 35 values that drift whenever the lock changes, unless something
  checks them (option C).

**B. One project policy: the workspace declares 1.92.**
- **The change.** One line in `Cargo.toml`, and the docs sentence.
- **Gain.** Simple, and true for the binaries.
- **Cost.** It over-declares the 48 members whose closures sit at 1.80 or
  below. The review allows it only as a decision: "a unified 1.92 project
  policy would be a deliberate support decision, not a deduction that every
  package technically needs 1.92".

**C. Enforcement**, combined with A or B:
- **C1. A CI job at the declared toolchain.** It would run, for example,
  `cargo +<declared> check -p <member>` on a chosen set, with the lock
  resolved in the job. Only this turns a declaration into a measured minimum.
  It costs one more toolchain and CI minutes on every run.
- **C2. A closure check in `ix doctor`.** It would fail, or first warn, when
  a member's closure declares more than the member. It needs no old toolchain
  and catches the drift option A creates. It is still declared bounds only.
- **C3. MSRV-aware resolution.** That is `resolver = "3"`, or
  `resolver.incompatible-rust-versions = "fallback"` in `.cargo/config.toml`;
  Cargo's documentation places both from Cargo 1.84. It changes which
  versions every build resolves, so it is a behaviour change, not only a
  check. By the same documentation, Cargo before 1.84 does not accept
  `resolver = "3"`, which conflicts with a 1.80 promise. Not tried here.

**D. Only correct the sentence.** Replace "MSRV: Rust 1.80+ (due to wgpu 28)"
with a measured statement that points to this document. It is the smallest
change, and it leaves the manifests false.

## 6. Recommendation

The decision is the operator's. I recommend a tracer bullet, as one PR:

1. **D, wherever the sentence appears:** `CLAUDE.md:33`, `README.md:475` and
   `compound-engineering.local.md:13`.
2. **A, limited to the 1.92 group.** Raise `rust-version` to 1.92 for the five
   members whose *default* closure declares it: `ix-gpu`, `ix-nn`,
   `ix-agent`, `ix-demo` and `ix-cli`. `memristive-markov` gets the feature
   comment instead.
3. **C2 as a warning in `ix doctor`**, with a test. It reports what §3
   reports.

Then decide, with C2's output in hand, whether to extend A to the other 30
members or adopt B.

- **Why not all 35 now.** The lock is not committed, so the values below 1.92
  are the most likely to move at the next refresh. The check (C2) should
  exist before 30 more values that can drift.
- **Why not C1 first.** It is worth its CI cost once someone depends on an
  old toolchain. Whether anyone does is *unknown* (question 1 below).

Open questions for the operator:

1. Does any consumer build IX with a toolchain older than current stable,
   such as a sibling repo, a packager or a pinned CI image? If none does,
   option B, or even a "latest stable" policy, is the honest and simple
   answer.
2. Is `Cargo.lock` gitignored on purpose? A committed lock would make §3
   reproducible and C1 meaningful. That is a separate decision, recorded here
   only because it changes how far these numbers can be trusted.

## 7. Revisit and delete-by triggers

Rerun the audit when any of these happens:

- the lock is refreshed (`cargo update`), or a member of the 1.92 group gains
  a dependency;
- wgpu moves past 28;
- a consumer reports a build failure on an older toolchain.

**Delete-by:** once C2 or C1 exists, its output supersedes the table in §3.
Keep this document as the decision record, and drop the table from it.
