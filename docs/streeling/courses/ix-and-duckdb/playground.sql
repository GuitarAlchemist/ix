-- ix DuckDB playground — a ready-to-query session over ix's on-disk catalogs.
--
-- Launch a persistent session in a separate terminal, FROM THE REPO ROOT:
--   duckdb -init docs/streeling/courses/ix-and-duckdb/playground.sql
--
-- (The relative paths below resolve from the repo root, so launch there.)
-- Then just type SQL. Examples:
--   SELECT repo, count(*) FROM learnings GROUP BY ALL;
--   SELECT day, round(pass_pct,1) FROM chatbot_qa ORDER BY day;

-- Streeling learnings catalog (Lesson 1).
CREATE OR REPLACE VIEW learnings AS
  SELECT * FROM read_json_auto('state/streeling/catalog.jsonl');

-- Daily quality snapshots as time-series, date parsed from the filename (Lesson 2).
CREATE OR REPLACE VIEW chatbot_qa AS
  SELECT regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day, *
  FROM read_json_auto('state/quality-snapshots/chatbot-qa/*.json', filename = true);

CREATE OR REPLACE VIEW embeddings AS
  SELECT regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day, *
  FROM read_json_auto('state/quality-snapshots/embeddings/*.json', filename = true);

CREATE OR REPLACE VIEW voicing_analysis AS
  SELECT regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day, *
  FROM read_json_auto('state/quality-snapshots/voicing-analysis/*.json', filename = true);

-- The materialized quality-analytics DB (tables + quality_latest rollup), PR #101.
-- It's a regenerable, gitignored binary; rebuild first if missing:
--   duckdb state/quality/analytics/quality.duckdb < state/quality/analytics/build-views.sql
-- Then uncomment to attach it read-only as schema `quality`:
-- ATTACH 'state/quality/analytics/quality.duckdb' AS quality (READ_ONLY);

.tables
SELECT 'ix DuckDB playground ready — views: learnings, chatbot_qa, embeddings, voicing_analysis' AS hint;
