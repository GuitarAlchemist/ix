# Resources — IX & DuckDB

> Only real, verified sources: official docs + in-repo files you can open right now.

## DuckDB (official)
- **DuckDB docs (root):** https://duckdb.org/docs/ — start here.
- **Friendly SQL** — ergonomic extensions (`GROUP BY ALL`, `SELECT * EXCLUDE`, `QUALIFY`):
  search "Friendly SQL" at duckdb.org/docs.
- **Reading files directly** (`read_json_auto`, `read_csv_auto`, `read_parquet`, globs,
  `filename=`): the "Data Import" section of duckdb.org/docs.
- **No-browser doc search:** the `duckdb-skills:duckdb-docs` skill (cached docs index).
  `duckdb-skills:query` runs SQL.

## In-repo (ground truth — open in Zed)
- `docs/DUCKDB.md` — ix's DuckDB tier model + decisions.
- `crates/ix-duck/` — in-process bench + ix algorithms as SQL UDFs (`src/udf.rs`,
  `examples/yield_analysis.rs`, `examples/ix_quality_lens.rs`).
- `state/streeling/catalog.jsonl` + `docs/streeling/queries.sql` — learnings catalog + 5
  example registrar queries (Lesson 1).
- `state/quality-snapshots/{chatbot-qa,embeddings,voicing-analysis}/*.json` — daily snapshot
  folders (Lesson 2).
- `state/quality/analytics/build-views.sql` — the quality-analytics materialization (Lesson 3;
  shipped in PR #101). Mirrors `../ga/state/quality/analytics/build-views.sql`.

## CLI
- `duckdb` is on PATH (v1.5.3). Quick check: `duckdb -c "SELECT 42"`.
- Query a file in place: `duckdb -c "SELECT * FROM read_json_auto('state/streeling/catalog.jsonl') LIMIT 5"`.
- Ready session with catalogs loaded: `duckdb -init docs/streeling/courses/ix-and-duckdb/playground.sql` (run from repo root).
