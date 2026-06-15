-- Business-Value Scorecard — DuckDB queries over the federated catalog.
-- The catalog is plain JSONL; DuckDB reads it directly (no import). Run from repo root:
--   duckdb -c ".read docs/value/queries.sql"
-- See docs/contracts/business-value.contract.md and crates/ix-value.

-- 1. Top demos across the ecosystem by continuous score (excludes repo rollups).
SELECT repo, id, title, stars, round(score01, 3) AS score
FROM read_json_auto('state/value/catalog.jsonl')
WHERE kind = 'demo'
ORDER BY score01 DESC
LIMIT 10;

-- 2. Repo leaderboard — the rollup row per repo.
SELECT repo, stars, round(score01, 3) AS score, rationale
FROM read_json_auto('state/value/catalog.jsonl')
WHERE kind = 'repo'
ORDER BY score01 DESC;

-- 3. Low-confidence, high-impact surfaces — where measuring usage would move the score most
--    (impact high but confidence is the binding that caps it).
SELECT repo, id, title, impact, confidence, stars
FROM read_json_auto('state/value/catalog.jsonl')
WHERE kind = 'demo' AND impact >= 4 AND confidence <= 3
ORDER BY impact DESC, confidence ASC;
