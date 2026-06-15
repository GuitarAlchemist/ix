# Lesson 2 — A folder of daily files → one time-series table (`glob` + `filename`)

**The one idea.** Lesson 1 pointed SQL at *one* file. Real quality data is *one file per day*
in a folder. DuckDB reads the whole folder with a **glob** (`*.json`), and `filename=true`
adds a hidden column telling you which file each row came from — so you can **recover the date
from the path** and get a time series. No loop, no per-file code.

## Worked example (real, against `state/quality-snapshots/chatbot-qa/`)

```sql
SELECT regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day,  -- date from path
       round(pass_pct, 1) AS pass_pct
FROM read_json_auto('state/quality-snapshots/chatbot-qa/*.json', filename=true)  -- glob + filename
ORDER BY day;
```
```
2026-04-12  19.5
2026-04-13  19.5
   …          …      ← 14 days, 2026-04-12 → 2026-04-25
2026-04-25  19.5
```

The three moving parts:
1. **`'…/*.json'`** — a glob; DuckDB reads every matching file as one table (14 days here).
2. **`filename=true`** — adds the source path as a column. The files carry no `date` field
   *inside*; the date lives in the **filename**, the canonical key across ix's quality pipeline.
3. **`regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1)`** — pulls `YYYY-MM-DD` out
   of the path into a real `day` column.

**This is exactly the pattern in `state/quality/analytics/build-views.sql`** (PR #101):
`CREATE TABLE … AS SELECT regexp_extract(filename,…) AS day … FROM read_json_auto('…/*.json',
filename=true)`. You can now read that file and know what every line does.

**A steering insight, free from the data:** `pass_pct` is **flat at 19.5 for all 14 days**.
That's not a frozen chatbot — these are `deterministic-fixture-ci` snapshots (same fixtures in,
same number out). A director's reflex: *"is this metric measuring reality or replaying a
fixture?"* A flat line in a quality trend is a smell worth questioning — the "green-but-dead"
trap ix's discipline warns about.

## Self-check
1. Two files lack the `pass_pct` field (older format). What happens, and which option makes it
   tolerant?
2. Where does the `day` value come from — inside the JSON, or somewhere else?
3. You see a quality metric perfectly flat for weeks. What's your first question?

<details><summary>Answers</summary>

1. It can error on the missing column. **`union_by_name=true`** merges files by column *name*,
   NULL-filling missing ones — how `build-views.sql` tolerates drift (Lesson 3).
2. **The filename/path**, not the JSON body — parsed with `regexp_extract`.
3. *"Is this real measurement or a deterministic fixture/cached value?"*
</details>

← [Lesson 1](./01-sql-over-files.md) · → _next: Lesson 3 — `union_by_name` + `CREATE TABLE AS` (materializing `quality.duckdb`)_
