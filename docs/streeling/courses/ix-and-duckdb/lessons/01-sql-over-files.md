# Lesson 1 — DuckDB is SQL over your files (and why ix leans on it)

**The one idea.** You know SQL as something you run *against a database* — load data in, then
query. DuckDB flips that: it's an **in-process analytics engine** ("SQLite for analytics") that
runs SQL **directly against files on disk** — JSON, JSONL, CSV, Parquet — with **no import step
and no server**. You point SQL at a file path and it's a table.

That single property is *why ix uses it*. ix already produces piles of on-disk state
(`*.json`, `*.jsonl`) — learnings, quality snapshots, telemetry, catalogs. Before DuckDB,
answering "how many learnings per repo?" meant writing a Rust/JS loader. With DuckDB it's one
query. The three ix DuckDB surfaces are all the same move — *point SQL at our state*:
- **`ix-duck`** — in-process bench + ix algorithms as SQL functions (over `hits.jsonl`)
- **Streeling catalog** — learnings as a queryable table
- **quality-analytics** — daily quality snapshots → `quality.duckdb` (PR #101)

## Worked example (real, against `state/streeling/catalog.jsonl`)

```sql
SELECT repo, kind, count(*) AS n
FROM read_json_auto('state/streeling/catalog.jsonl')   -- ← the whole trick is right here
GROUP BY repo, kind ORDER BY n DESC;
```
```
ga  solution    27
ix  solution    17
ix  plan        17
ix  brainstorm  11
```

Three DuckDB-specific things a SQL-knower should notice:
1. **`read_json_auto('…file…')` in the `FROM`** — the file *is* the table. No `CREATE TABLE`,
   no load, no schema declaration; DuckDB infers columns. (Also `read_csv_auto`,
   `read_parquet`, and glob paths like `'snapshots/*.json'`.)
2. **It's still just SQL** — `GROUP BY`, `JOIN`, window functions all work unchanged.
3. **"Friendly SQL"** — e.g. `GROUP BY ALL` groups by every non-aggregated column automatically:
   ```sql
   SELECT repo, count(*) AS learnings
   FROM read_json_auto('state/streeling/catalog.jsonl')
   GROUP BY ALL ORDER BY learnings DESC;   -- → ix 45, ga 27
   ```

**Why this matters for steering:** when someone proposes "let's add analytics over X," the
reflex question becomes — *is X already on disk as JSON/JSONL?* If yes, the cost is a `.sql`
file, not a new service. That's the lens behind every DuckDB decision in ix.

## Self-check
1. What has to happen to a JSON file before you can query it with DuckDB?
2. ix has `state/quality-snapshots/embeddings/*.json` (one file per day). In one phrase, how
   would you point DuckDB at *all* of them at once?
3. Why is DuckDB a better fit than Postgres for ix's `state/` artifacts specifically?

<details><summary>Answers</summary>

1. **Nothing** — no import/load/schema. `read_json_auto('path')` reads it in place.
2. A **glob**: `read_json_auto('state/quality-snapshots/embeddings/*.json')` (optionally
   `filename=true` to recover which file each row came from → that's how the date is parsed).
3. The data is **already files on disk, regenerable, queried in-process** — no server, no ETL.
   Postgres would mean standing up a server and importing.
</details>

→ Next: [Lesson 2 — glob + filename → time-series](./02-glob-filename-timeseries.md)
