-- Streeling University — Registrar queries (DuckDB over the learnings catalog).
--
-- Run with the duckdb CLI / duckdb-skills, or via ix-duck:
--   duckdb -c ".read docs/streeling/queries.sql"
--   (or paste a single query into /duckdb-skills:query)
--
-- The catalog is state/streeling/catalog.jsonl (regenerate: `cargo run -p ix-streeling -- catalog`).

-- 1) Enrollment: how many learnings per repo and kind (cross-repo overview).
SELECT repo, kind, count(*) AS n
FROM read_json_auto('state/streeling/catalog.jsonl')
GROUP BY repo, kind
ORDER BY n DESC;

-- 2) Root-cause search: every CI / build learning across all repos.
SELECT repo, title, root_cause, path
FROM read_json_auto('state/streeling/catalog.jsonl')
WHERE kind = 'solution'
  AND (category ILIKE '%build%' OR category ILIKE '%ci%'
       OR lower(coalesce(symptom, '')) LIKE '%ci%'
       OR list_contains(tags, 'cargo'))
ORDER BY repo, title;

-- 3) Faculties by size (which categories hold the most institutional knowledge).
SELECT category, count(*) AS n, count(DISTINCT repo) AS repos
FROM read_json_auto('state/streeling/catalog.jsonl')
GROUP BY category
ORDER BY n DESC;

-- 4) Recent learnings (most recently dated, cross-repo) — what the team learned lately.
SELECT date, repo, kind, title
FROM read_json_auto('state/streeling/catalog.jsonl')
WHERE date IS NOT NULL
ORDER BY date DESC
LIMIT 20;

-- 5) Free-text lookup: "what have we learned about <topic>?" (edit the term).
SELECT repo, kind, category, title, path
FROM read_json_auto('state/streeling/catalog.jsonl')
WHERE lower(title) LIKE '%voicing%'
   OR lower(coalesce(symptom, '')) LIKE '%voicing%'
   OR list_contains(tags, 'voicings')
ORDER BY repo, category;
