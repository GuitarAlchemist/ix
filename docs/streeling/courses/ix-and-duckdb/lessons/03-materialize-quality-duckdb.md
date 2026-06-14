# Lesson 3 — Materializing the portable `quality.duckdb` (`union_by_name` + `CREATE TABLE AS`)

**The one idea.** Lessons 1–2 queried files *live* — every run re-reads and re-infers the JSON.
That's great for ad-hoc, but for a durable analytics layer you want two more things:
**drift tolerance** (files change shape over time) and **materialization** (freeze the result
into a real table so it's portable and fast). That's what `build-views.sql` does to produce
`quality.duckdb` — and it's the whole quality-analytics layer (PR #101).

## Part A — `union_by_name=true` (schema-drift tolerance)
When a glob reads many files and an older one is missing a column (or a newer one adds
`degraded`/`degraded_reason`), positional matching breaks. `union_by_name=true` merges files
by **column name**, NULL-filling whatever's absent:
```sql
FROM read_json_auto('chatbot-qa/*.json', filename = true, union_by_name = true)
```
This is why ix's snapshots can evolve without breaking the pipeline. (Pair it with `TRY_CAST`
so a stray string in a numeric column yields NULL instead of an error.)

## Part B — `CREATE OR REPLACE TABLE … AS SELECT` (materialize)
A *view* re-runs its query every time (and keeps the glob-path dependency). A **materialized
table** runs the query once and stores the rows in the `.duckdb` file:
```sql
CREATE OR REPLACE TABLE ix_harness AS
SELECT domain, emitted_at, metric_name, TRY_CAST(metric_value AS DOUBLE) AS metric_value, …
FROM read_json_auto('ix-harness/last.json', filename = true, union_by_name = true);
```
Now `quality.duckdb` is **portable** — anything can open it (the Rust lens, the duckdb CLI,
GA's tools) with no knowledge of the original JSON paths. Re-run the script to refresh.

## Worked example (real — just built it)
```bash
# from state/quality/ :
duckdb analytics/quality.duckdb < analytics/build-views.sql
```
```sql
SHOW TABLES;
-- ix_harness · optick_sae · quality_health · router_eval · quality_latest
SELECT * FROM quality_latest;
```
```
source          day         metric              metric_name
optick_sae      2026-06-14  0.999559            reconstruction_r2
ix_harness      2026-05-16  1.0                 harness_ready
quality_health  2026-04-25  25.0                total_metrics
router_eval     NULL        0.8193629899512…    test_macro_f1
```
- The first four are **materialized tables** (real rows, no glob); `quality_latest` is a
  **view** over them — always current, zero path dependency, safe to query from anywhere.
- **`optick_sae` has a real row** — the script reads GA's *consumed* SAE artifacts cross-repo
  (`../../../ga/state/quality/optick-sae/*/optick-sae-artifact.json`). So **ix's analytics DB
  queries the contract output it produces for GA**: SQL on both sides of one contract
  (producer/consumer parity).

## Why this matters for steering
This is the full shape of an ix analytics layer: *glob the on-disk state → tolerate drift →
materialize into a portable `.duckdb` → expose a `_latest` rollup view*. When you review a
proposal, you can now check it against this pattern — and the `.duckdb` is a **regenerable,
gitignored binary** (the SQL + README are the tracked source of truth).

## Self-check
1. Why materialize into tables instead of just keeping views over the JSON globs?
2. What does `union_by_name=true` buy you, and what pairs well with it for numeric drift?
3. `optick_sae` reads a file under `../../../ga/…` — what cross-repo idea is that demonstrating?

<details><summary>Answers</summary>

1. **Portability + speed + decoupling** — the `.duckdb` carries the rows with no dependency on
   the original JSON paths, so any tool can open it; views would re-read the globs every time.
2. Tolerance of **schema drift** (merge by name, NULL-fill missing columns); pair with
   **`TRY_CAST`** so an off-type value becomes NULL instead of erroring.
3. **Producer/consumer SQL parity** — ix produces the SAE artifact GA consumes; reading GA's
   copy lets ix query the same contract as SQL, so both sides inspect identical fields.
</details>

← [Lesson 2](./02-glob-filename-timeseries.md) · **Course capstone reached** — you can now read
any `build-views.sql` in the repo and explain every table, view, and the producer/consumer story.
