# Course: IX & DuckDB

A Streeling University course — learn how DuckDB is used across IX (and GA), by querying
ix's **real** on-disk catalogs. Built with `/teach`; committed here as institutional material
(the "Learn" in Learn→Ship→Steer, `docs/LEARNING.md`).

- **Audience:** know SQL, new to DuckDB; want to *steer & understand* (light on coding).
- **Style:** worked examples on real ix data.
- **Mission / success criteria:** [`MISSION.md`](./MISSION.md)
- **Resources (verified):** [`RESOURCES.md`](./RESOURCES.md)
- **Glossary:** [`reference/glossary.md`](./reference/glossary.md)

## Lessons
1. [SQL over files — and why ix leans on DuckDB](./lessons/01-sql-over-files.md)
2. [A folder of daily files → one time-series table (`glob` + `filename`)](./lessons/02-glob-filename-timeseries.md)
3. _next:_ `union_by_name` + `CREATE TABLE AS` — materializing the portable `quality.duckdb`

## Play along — a DuckDB session in a separate terminal

Open a terminal **at the repo root** (e.g. a Zed terminal tab) and launch a session with
ix's catalogs pre-loaded as views:

```bash
duckdb -init docs/streeling/courses/ix-and-duckdb/playground.sql
```

You'll get views `learnings`, `chatbot_qa`, `embeddings`, `voicing_analysis` ready to query:

```sql
SELECT repo, count(*) FROM learnings GROUP BY ALL;
SELECT day, round(pass_pct,1) FROM chatbot_qa ORDER BY day;
```

`duckdb` is on PATH (v1.5.3). The `.duckdb` analytics DB is regenerable — see
`state/quality/analytics/` (gitignored binary; rebuild with its `build-views.sql`).

## Progress
- 2026-06-14: Lessons 1–2 delivered. Next: Lesson 3 (materialization / `quality.duckdb`).
