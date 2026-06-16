---
title: "Tier 3 — published `ix_optick` DuckDB extension"
type: feat
status: needs-sign-off
date: 2026-06-16
relates: docs/DUCKDB.md (Tier 3), docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md
door: one-way
---

# Tier 3 — published `ix_optick` DuckDB extension

> **STOP / sign-off gate.** This is a **one-way door** (CLAUDE.md "log one-way doors";
> `docs/DUCKDB.md` one-way-door log). It is a **plan only** — no code is to be written
> until the operator signs off. Tiers 0–2 are two-way doors and need no sign-off; this
> one does, because it ships a loadable, signed, versioned public artifact.

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

## Decision needed

- [ ] **Sign off to build** (pick A or B, accept the one-way-door cost + CI matrix), **or**
- [ ] **Decline / defer** — stay at Tier 2 (recommended), revisit on the trigger above.

Until a box is checked, this stays `status: needs-sign-off` and no extension code is written.
