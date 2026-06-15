# Glossary — IX & DuckDB

- **DuckDB** — in-process OLAP (analytics) engine; "SQLite for analytics." Runs SQL directly
  against files (JSON/CSV/Parquet) with no server and no import step.
- **In-process** — runs inside your program/CLI, not a separate server (contrast: Postgres).
- **`read_json_auto('path')`** — table function reading a JSON/JSONL file (or glob) as a table,
  inferring columns. Used in `FROM`. Siblings: `read_csv_auto`, `read_parquet`.
- **Glob** — wildcard path like `'dir/*.json'`; DuckDB reads every match as one combined table.
- **`filename=true`** — adds the source file path as a column, so you can recover per-file info
  (e.g. the date) not present in the JSON body.
- **`regexp_extract(s, pattern, group)`** — pulls a substring via regex; used to parse
  `YYYY-MM-DD` out of a filename into a `day` column.
- **Friendly SQL** — DuckDB's ergonomic extensions (`GROUP BY ALL`, `SELECT * EXCLUDE`, `QUALIFY`).
- **`GROUP BY ALL`** — group by every non-aggregated column automatically.
- **`CREATE OR REPLACE TABLE x AS SELECT …`** — materialize a query into a stored table (vs a
  view); makes the `.duckdb` portable — re-run the script to refresh.
- **`union_by_name=true`** — when reading many files, merge by column *name* (NULL-fill missing)
  instead of by position — tolerates schema drift between snapshots.
- **`ix-duck`** — ix crate wrapping in-process DuckDB + ix algorithms as SQL scalar UDFs;
  `duckdb` crate with the `bundled` feature (compiles DuckDB from source → no system lib).
- **Streeling catalog** — `state/streeling/catalog.jsonl`, the federated learnings index;
  queryable as a DuckDB table.
- **quality-analytics layer** — `state/quality/analytics/build-views.sql` → `quality.duckdb`;
  materializes daily quality snapshots into tables + a `quality_latest` rollup (PR #101).
- **Producer/consumer SQL parity** — ix *produces* artifacts (SAE, optick index) that ga
  *consumes*; DuckDB lets both sides query the same contract as SQL.
