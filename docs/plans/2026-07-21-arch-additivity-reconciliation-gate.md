---
title: Additivity reconciliation gate — fail-closed integrity check for sharded warehouse aggregates
type: arch
status: draft
date: 2026-07-21
issue: W1 of docs/research/2026-07-21-ktheory-duckdb-supercharge.md; motivating bug GuitarAlchemist/ix#248
reversibility: two-way (a SQL query + a CI assertion; no schema, no public API)
revisit-trigger: a legitimate non-additive aggregate appears (e.g. dedup-across-shards) → relax to a per-metric allowlist
---

# Additivity reconciliation gate — a fail-closed data-integrity check for sharded warehouse aggregates

- **Issue:** W1 from `docs/research/2026-07-21-ktheory-duckdb-supercharge.md`; motivating bug #248.
- **Date:** 2026-07-21
- **Status:** proposal (tracer-bullet)
- **Reversibility:** two-way door (a query + a CI assertion; no schema, no public API). Revisit trigger: if a *legitimate* non-additive aggregate appears (e.g. a dedup-across-shards metric), relax the gate to a per-metric allowlist.

## Who is in pain

Whoever trusts a sharded DuckDB aggregate over the optick-sae / voicing artifacts. Today a split that silently drops rows (bug #248: `feature_activations.parquet` keyed on `optick_row = train_idx` only → 297,395 of 313,047 corpus rows, a ~5% gap) produces a green-looking aggregate that is quietly wrong. Nothing catches it.

## The idea (K-theory's one useful axiom, used as a gate)

The collapse theorem says valid warehouse invariants are **valuations**: additive over disjoint shards, `v(A) + v(B/A) = v(A∪B)`. Turn that axiom around: **if a sharded aggregate is *not* additive, the shard boundary is lossy or mis-joined.** That is a fail-closed integrity oracle, not a computed feature — the honest, zero-Rust payoff of the whole K-theory investigation.

## Tracer-bullet slice (end-to-end, every layer, smallest)

A single DuckDB SQL file `crates/ix-duck/sql/reconciliation.sql` (or a `reconcile` subcommand on the existing ix-duck CLI) that materialises one `reconciliation` table with one boolean column per assertion, and a CI step that fails if any is false. Three assertions, each mapping directly to a #248 failure mode:

```sql
-- Reconciliation gate over the optick-sae activation artifact vs the corpus.
-- Paths are parameters; corpus_n is the authoritative optick.index row count.
WITH acts AS (
  SELECT * FROM read_parquet($activations)          -- feature_activations.parquet
),
recon AS (
  SELECT
    -- A1: the join key has no duplicates (each optick_row appears once)
    (SELECT count(*) = count(DISTINCT optick_row) FROM acts)            AS optick_row_no_dupes,
    -- A2: every id is a legal corpus position [0, corpus_n)
    (SELECT bool_and(optick_row >= 0 AND optick_row < $corpus_n) FROM acts) AS rows_in_corpus_range,
    -- A3: ADDITIVITY — the artifact covers the whole corpus, not just one split.
    -- v(train) + v(val) must equal v(corpus). A train-split-only key fails here.
    (SELECT count(*) FROM acts) = $corpus_n                              AS split_additivity
)
SELECT * FROM recon;
```

`$corpus_n = 313047` (from `optick.index`); `$activations` is the snapshot parquet. A3 is the additivity assertion — it is exactly `v(A) + v(B/A) = v(whole)` specialised to counts, and it is the one that catches #248 (297,395 ≠ 313,047 → `split_additivity = false` → gate red).

## Falsifiers (how we prove it works, not just runs)

1. **Positive control:** point the gate at the complete 2026-06-14 snapshot → all three columns `true`.
2. **Negative control (the bug):** point it at 2026-07-20 → `split_additivity = false`. If it goes green on #248's data, the gate is green-but-dead and must be rejected.
3. **Dup injection:** duplicate one `optick_row` → `optick_row_no_dupes = false`.

The negative control is mandatory: a reconciliation gate that passes on known-broken data is worse than none (per `feedback_green_but_dead`).

## Why not a UDF / why not `ix-ktheory`

Per the collapse theorem (Leg 1), Mayer–Vietoris additivity over IX's data *is* inclusion–exclusion on counts — plain `COUNT`/`bool_and`. Wrapping it in a Rust UDF or routing through `ix-ktheory::mayer_vietoris` adds a dependency and a maturity-gate surface for zero capability. Keep it SQL.

## Out of scope

Generalising to weighted valuations (SUM of activation mass, not just row counts), and to the voicing-partition shards. Log as follow-ups once the count-level gate is green in CI.
