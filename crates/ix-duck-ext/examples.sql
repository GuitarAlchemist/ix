-- examples.sql — ready-made IX-extension demos for the DuckDB UI.
--
-- Open this file in the DuckDB UI (http://localhost:4213) and run the cells
-- top to bottom, or one section at a time. If you launched via ix-duck.ps1 the
-- extension is already loaded; otherwise run the LOAD line first (adjust path).
--
-- Note on scalar vs table functions:
--   * ix_cosine / ix_euclidean are SCALAR — they take two DOUBLE[] columns and
--     work naturally per-row over a table (sections 2-3).
--   * ix_pca_project / ix_silhouette are TABLE functions — they take a whole set
--     as a constant JSON arg. DuckDB forbids subqueries in table-function args,
--     so we stage the set into a session variable with SET VARIABLE and read it
--     back with getvariable() (sections 4-5). That's the table→JSON→table bridge.

-- 0) Load the extension (skip if launched via ix-duck.ps1 / build already loaded it).
LOAD 'C:/Users/spare/source/repos/ix/crates/ix-duck-ext/ix.duckdb_extension';

-- 1) A toy table of labelled 4-D embedding vectors — two rough families:
--    "bright" (major-ish) and "tense" (dominant-ish).
CREATE OR REPLACE TABLE voicings AS
SELECT * FROM (VALUES
  ('Cmaj7',  'bright', [0.00, 1.00, 0.00, 1.00]::DOUBLE[]),
  ('Am7',    'bright', [0.10, 0.90, 0.10, 0.80]::DOUBLE[]),
  ('Fmaj7',  'bright', [0.00, 1.00, 0.20, 0.90]::DOUBLE[]),
  ('Em7',    'bright', [0.05, 0.95, 0.05, 0.85]::DOUBLE[]),
  ('G7',     'tense',  [1.00, 0.00, 1.00, 0.00]::DOUBLE[]),
  ('B7',     'tense',  [0.90, 0.10, 0.90, 0.20]::DOUBLE[]),
  ('D7',     'tense',  [0.95, 0.05, 0.85, 0.10]::DOUBLE[]),
  ('E7',     'tense',  [0.85, 0.15, 0.95, 0.05]::DOUBLE[])
) AS t(name, family, vec);

SELECT * FROM voicings;

-- 2) kNN — the 4 nearest neighbours to Cmaj7 by L2 distance.
--    ix_euclidean is the primitive: ORDER BY it, LIMIT k.
SELECT name, family,
       round(ix_euclidean((SELECT vec FROM voicings WHERE name = 'Cmaj7'), vec), 3) AS dist
FROM voicings
WHERE name <> 'Cmaj7'
ORDER BY dist
LIMIT 4;

-- 3) Cosine similarity ranking against a query vector.
SELECT name, family,
       round(ix_cosine([0.0, 1.0, 0.0, 1.0]::DOUBLE[], vec), 3) AS cos_sim
FROM voicings
ORDER BY cos_sim DESC;

-- 4) PCA → 2-D coordinates per voicing (good for a scatter plot in the UI).
--    Stage the set + labels into session variables, then feed the table function.
SET VARIABLE vecs  = (SELECT to_json(list(vec    ORDER BY name)) FROM voicings)::VARCHAR;
SET VARIABLE names = (SELECT list(name   ORDER BY name) FROM voicings);
SET VARIABLE fams  = (SELECT list(family ORDER BY name) FROM voicings);

SELECT getvariable('names')[row + 1] AS name,     -- list is 1-based, row is 0-based
       getvariable('fams')[row + 1]  AS family,
       round(coords[1], 3) AS pc1,
       round(coords[2], 3) AS pc2
FROM ix_pca_project(getvariable('vecs'), 2)
ORDER BY name;
-- Tip: in the UI, switch this result to a scatter chart with x=pc1, y=pc2,
-- colour=family — the two families separate cleanly along PC1.

-- 5) Silhouette — how well does the "bright vs tense" labelling cluster?
--    (Reuses 'vecs' and 'names' from section 4; sets the integer labels here.)
SET VARIABLE labels =
  (SELECT to_json(list(CASE family WHEN 'bright' THEN 0 ELSE 1 END ORDER BY name)) FROM voicings)::VARCHAR;

-- Per-point coefficients (1 = deep in its cluster, < 0 = probably mislabeled):
SELECT getvariable('names')[row + 1] AS name,
       label,
       round(silhouette, 3) AS silhouette
FROM ix_silhouette(getvariable('vecs'), getvariable('labels'))
ORDER BY silhouette DESC;

-- ...and the single overall score:
SELECT round(avg(silhouette), 3) AS mean_silhouette
FROM ix_silhouette(getvariable('vecs'), getvariable('labels'));
