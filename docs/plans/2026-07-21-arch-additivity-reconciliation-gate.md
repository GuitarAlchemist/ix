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

---

## Research Insights (deepened 2026-07-21)

_Parallel research + architecture review (Fable-5 agents; the reviewer counted real rows) + learnings scan. **This plan needs revision before implementation** — its positive control is empirically false and its CI placement recreates a known trap in this repo._

### Enhancement summary — required revisions
1. **A3 is permanently red as written.** Both SAE snapshots are 297,395 rows (train-only), never 313,047 — so `count(*) = $corpus_n` fails the declared *positive* control too. Re-point A3 at **declared-vs-observed**.
2. **Add the prefix-set check** — it's the *actual* #248 detector; the current three assertions all pass on the bug.
3. **Add A0 NULL-key check** — the current SQL is green on an all-NULL key column.
4. **Move execution to produce time** in the `ix-optick-sae` pipeline (there is no ix-duck CLI; the parquet is gitignored ⇒ absent on CI ⇒ the #238/#244 absence trap).
5. `read_parquet($activations)` **won't bind** — use `getvariable()` + `SET VARIABLE`.
6. **Generated-in-test fixtures**, not live gitignored snapshots, so the negative control actually runs in CI.

### HEADLINE — the positive control is factually false (agent: architecture review)
Row counts (parquet metadata):
- `ga/state/quality/optick-sae/2026-06-14/feature_activations.parquet` → **297,395**
- `ga/state/quality/optick-sae/2026-07-20/…` → **297,395**

The "complete 2026-06-14 snapshot" (Falsifiers §1) is **also train-only**. A3 (`count(*) = 313047`) goes red on **every artifact that has ever existed**, and PR #250 (`optick_coverage.py`) codifies train-only as the *intended* semantics. The plan's own revisit-trigger ("a legitimate non-additive aggregate appears") has already fired, before implementation. A permanently-red gate is the dual of green-but-dead — it trains everyone to ignore it.

### Re-point A3 + add the real #248 detector (agents: architecture + research)
- **A3 → declared-vs-observed:** `parquet row count == artifact.json activations_coverage.n_train` (the field PR #250 now emits), and `activations_coverage.corpus_n == live optick.index row count` (catches a stale declaration after a ~140s index rebuild). This also **replaces the hardcoded `$corpus_n = 313047`** — a literal that goes stale on the next corpus rebuild and, if "fixed" by copying the new artifact count, makes A3 tautological (self-referential-oracle failure).
- **New assertion — prefix-set check (the actual #248 discriminator):** the train-split-only key bug (`optick_row = train_idx`) yields values that are unique, in-range, and count == n_train — so it **passes all three current assertions**. The discriminator is: a seeded random train split must satisfy `max(optick_row) >= n_train` (some rows with `optick_row >= n_train` must exist). `optick_row == {0..n_train-1}` exactly means the column holds *split positions*, not *corpus positions*. Without this, the re-pointed gate loses its mandatory negative control.
- **A0 — NULL key (research agent):** `count(DISTINCT)` and `bool_and` both ignore NULLs, so a NULL `optick_row` slips through A1 *and* A2. Add `optick_row IS NOT NULL` (or fold `count(*) = count(optick_row)` into A1). Single most actionable line — the current SQL is green on an all-NULL key.

### DuckDB specifics (agent: research)
- **No SQL ASSERT.** Two idioms: (1) SELECT booleans, caller exits nonzero — matches the existing `crates/ix-duck/examples/ix_maintain_gate.rs` gate (exit 0/1/2 + JSONL ledger); reports all verdicts. (2) `error('msg')` scalar throws so even a bare `duckdb -c` exits nonzero — good belt-and-suspenders, but fails on the first violation. Recommend idiom 1; keep the `.sql` `error()`-free so it stays composable.
- **Paths can't be prepared-statement params** — `read_parquet($x)` won't bind ("read_parquet only accepts constant parameters", duckdb #13750). Use `SET VARIABLE activations = ?; … read_parquet(getvariable('activations'))` (VARIABLE works for *reads*, #14490). **Scalars like `$corpus_n` → genuine bound params** via duckdb-rs `params![]`; never interpolate.
- Row counts are **metadata-cheap** (parquet footer); A3 is near-free. A1/A2 scan one projected column — cheap at 313k rows.

### Placement, absence trap & coupling (agent: architecture)
- **ix-duck has NO CLI** — only examples run by CI. The plan's "`reconcile` subcommand on the existing ix-duck CLI" names a seam that doesn't exist. A `crates/ix-duck/sql/` file that no ix-duck code compiles/executes is a dead asset by construction.
- **Right seam: colocate with the producer** (`crates/ix-optick-sae/python/`, run via the `duckdb` pip package as the last produce step) — beside `test_activations_coverage.py` / `test_partition_contract.py`, the contract-check home PR #250 just established.
- **Absence trap is real:** `feature_activations.parquet` is gitignored (`*.parquet`, 56 MB) ⇒ structurally absent on any fresh CI checkout ⇒ a naively-wired gate publishes "cannot evaluate here" as **red** — byte-for-byte the #238 maintain-gate disease (`docs/solutions/workflow-patterns/2026-07-20-green-ci-degraded-dashboard-missing-input.md`). **Run only at produce time**, where the just-written parquet being absent genuinely *is* an error (fail-closed on absence is then correct, and the skip/fail tri-state is unnecessary). Any scheduled re-check must use the #238 idiom: absent → skip-with-reason, exit 0, publish nothing.
- **Don't read ga from ix CI** — the sibling checkout doesn't exist on ix runners, and it inverts the federation direction (ga *pulls* ix outputs; `docs/solutions/ecosystem-integration/2026-06-21-federate-ix-snapshot-into-ga-state-quality.md`). Gate the artifact **at birth**; a snapshot that fails reconciliation never gets published.

### Compose with the producer fix, not redundantly (agent: architecture)
PR #250's produce-time assert is **self-referential** (in-memory counts vs each other). The gate's residual, independent value: declared-vs-observed (writer dropped/duplicated rows *after* counts computed), declared-vs-world (stale corpus_n after rebuild), A0/A1/A2 (producer asserts none), and the prefix-set check (the real keying bug). It's a **declare/verify pair** — #250 supplies the machine-readable baseline, the gate audits it against the physical parquet + live index at the artifact boundary.

### Pattern mapping & framing (agent: research)
- Assertions map 1:1 onto dbt/GE primitives: A1 = `unique`/`expect_column_values_to_be_unique`; A2 = `dbt_utils.accepted_range`; A3 = `dbt_utils.equal_rowcount` / `expect_table_row_count_to_equal_other_table`. Canonical shape: verdict row **plus** a `LIMIT 20` sample of failing rows for the CI log (dbt/Soda "verdict + sample" convention).
- Additivity = **distributive** in Gray's data-cube taxonomy: generalizes to SUM/COUNT/MIN/MAX and per-`GROUP BY shard` reconciliation (localizes *which* shard is lossy); breaks for holistic aggregates (`COUNT(DISTINCT)`, medians) and non-disjoint shards → then inclusion–exclusion `v(A)+v(B)−v(A∩B)=v(A∪B)` (the Mayer–Vietoris point made real) or a per-metric allowlist. Sharpen the framing: **valuations require disjoint shards**, so the gate really asserts two things — the decomposition is a partition (A1) *and* it reconciles (A3); comment this in the SQL so a future editor doesn't drop A1 thinking A3 subsumes it.

### Fixtures & CI (agents: research + learnings)
- **Generate fixtures in-test** (`COPY (SELECT …) TO 'tmp.parquet'`), not committed binaries and **not** the live gitignored snapshots (the plan's Falsifiers §1–2 point at machine-local artifacts CI can't see — the exact 175 MB-index green-history failure this repo already hit). Minimal seeded-bad set (`corpus_n = 100`): `good` (0..99), `dropped` (0..94 → A3-repoint red only), `dup` (0..98+dup42 → A1 red), `oob` (0..98+100 → A2 red), `nullkey` (one NULL → A0 red). One mutant per assertion, each killing exactly one — the non-vacuous minimum (dbt unit-test / mutation-testing discipline). Real-snapshot controls become a one-time manual PR validation, not the CI test.
- Ride the existing `cargo test`/produce pipeline; **note main has no required checks** — "required" is aspirational until branch protection is set (say so, so the gate isn't assumed blocking).

### References
dbt-utils (`equal_rowcount`, `accepted_range`), dbt data tests + unit tests (1.8+), Great Expectations, Gray et al. data-cube (distributive/algebraic/holistic), DuckDB Parquet tips + discussions #13750/#14490, `docs/solutions/workflow-patterns/2026-07-20-green-ci-degraded-dashboard-missing-input.md`.
