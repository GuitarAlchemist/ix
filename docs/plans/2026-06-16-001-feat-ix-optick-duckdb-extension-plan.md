---
title: "Tier 3 — `ix_optick` DuckDB loadable extension"
type: feat
status: completed
date: 2026-06-16
relates: docs/DUCKDB.md (Tier 3), docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md
door: one-way
---

# Tier 3 — `ix_optick` DuckDB loadable extension

> **Sign-off:** operator approved the build (2026-06-16). **Status: BUILT + VALIDATED.**
>
> Reality found during the build (correcting this plan's first draft): the loadable
> C-API **harness already existed** (`crates/ix-duck-ext`, prior session) — a
> `LOAD`-able `ix.duckdb_extension` exposing `ix_cosine`/`ix_euclidean` with the
> 512-byte metadata footer (`append_extension_metadata.py`). So the hard, irreversible
> part of the one-way door was already paid. This work added the **differentiated
> Tier-3 capability**: the OPTIC-K production mmap as a SQL table function.
>
> **Built:** `ix_optick_scan(index_path) → TABLE(voicing, instrument, embedding DOUBLE[])`
> (`crates/ix-duck-ext/src/optick.rs`, wraps `ix_optick::OptickIndex`). Validated by
> `LOAD` into the real DuckDB v1.5.3 CLI against GA's production
> `optick.index`: **313,047 voicings, dim 124**; `ix_euclidean` composes over two
> scanned embeddings (voicing 0↔1 = 1.1168). The rebuild also refreshed the artifact
> to the full current UDF surface (now incl. `ix_kdist`/`ix_dbscan`).
>
> **`ix_voicing_distance` was deliberately NOT built** — it's redundant: voicing
> distance = `ix_euclidean(a.embedding, b.embedding)` over `ix_optick_scan` rows.
>
> Build/validate: `pwsh crates/ix-duck-ext/build.ps1 -SmokeTest`. The crate stays
> **excluded from the workspace** (root `Cargo.toml`), so default/CI `cargo build
> --workspace` never compiles DuckDB — Tier 3 is out-of-band by design.

## What it would be

A real DuckDB **extension** (`.duckdb_extension`) exposing OPTIC-K natively in SQL:

- `ix_optick_scan(partition)` — table function over the `optick.index` mmap (voicing rows + embeddings) **without** the Tier-2 Parquet export.
- `ix_voicing_distance(a_id, b_id)` — voicing-distance scalar UDF using the index's own metric.

Versus what we already have: Tiers 0–2 read **derived files** (JSONL/Parquet) with the IX UDFs (`ix_cosine`/`ix_kdist`/`ix_pca_project`/`ix_silhouette`). Tier 3's only new capability is querying the **live production mmap directly** — no export step, always current.

## Why it's a one-way door

1. **Loadable artifact + signing.** DuckDB loads extensions by name; community extensions need per-platform builds (linux/macos/windows × amd64/arm64) and signing, plus a distribution channel. Reversing a published, downloaded-by-others extension is not clean.
2. **Public API surface.** `ix_optick_scan` / `ix_voicing_distance` become a contract others script against — breaking them is a breaking change (stable-surface discipline).
3. **Experimental build path.** DuckDB's Rust-extension story is still immature; the supported path is the C/C++ extension template (`duckdb/extension-template`). Either means a new non-Rust build toolchain in IX (contradicts "pure Rust except wgpu") **or** an FFI bridge from the C template into `ix-optick`.
4. **mmap UB surface.** Exposing the raw mmap through a long-lived extension widens the unsafe surface (lifetime/locking vs. GA's `GaApi`/`GaMcpServer` which also mmap the index — see the rebuild runbook's "stop those first" lock note).

## Approaches (if signed off)

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | **C/C++ extension** (official template) + FFI to a `cdylib` `ix-optick` | supported, signable, canonical | new C toolchain + FFI; cross-platform CI matrix |
| B | **Rust community extension** (`duckdb-extension-framework`) | stays in Rust | experimental, thinner DuckDB-version support, signing still required |
| C | **Don't — keep Tier 2.** Export Parquet, query with existing UDFs | zero new door, already 90% built | requires the export step; not "live" |

## Recommendation

**Default to C (stay at Tier 2) unless a concrete need forces Tier 3.** The only thing Tier 3 buys over Tier 2 is querying the live mmap without an export — and we have no use case that needs sub-rebuild freshness for *analytics* (the production *search* path already uses the mmap directly via GA). Build Tier 3 only when: (a) an analytics consumer demonstrably needs always-current embeddings AND (b) the Parquet export proves too stale/expensive — neither true today.

## Decision (resolved 2026-06-16)

- [x] **Signed off to build.** Approach **B** (Rust loadable extension via duckdb-rs) —
      the harness already used it, so no new C toolchain. `ix_optick_scan` shipped and
      validated against the production index.

Remaining (not blocking; out-of-band, demand-gated):
- **Signed distribution** to other machines needs a version-matched `duckdb.exe` +
  DuckDB's signing infra (the footer is appended locally for `-unsigned` LOAD today).
  Cross-platform builds (linux/macos/arm) are a CI-matrix follow-up if we ever publish.
- These are the only parts of the one-way door still open; the local capability is done.
