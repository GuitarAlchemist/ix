-- IX analytics pipelines, composed in SQL over the ix-duck UDFs.
--
-- DuckDB is IX's analyst's bench (docs/DUCKDB.md): SQL is the *composition* layer
-- for the algorithm primitives. This is NOT the governed `ix-pipeline` DAG engine
-- (that runs in the CLI / agent with StageGate + provenance). Two composition
-- patterns, because of one DuckDB rule:
--
--   * SCALAR UDFs (ix_cosine, ix_ndcg, ix_forte_number, …) chain freely over
--     columns in a single query — true SQL pipelines.
--   * TABLE UDFs (ix_kmeans, ix_silhouette, ix_pca_project, ix_optick_scan, …)
--     take only *constant* params (DuckDB forbids lateral-join column args), so
--     one table-fn's output can't directly feed another's input in the same query.
--     Bridge them with `SET VARIABLE` + `getvariable()` (materialize → constant).
--
-- Run (extension built via crates/ix-duck-ext/build.ps1):
--   duckdb -unsigned -c ".read docs/duck/pipelines.sql"

-- ── Pipeline 1: retrieval / intent-routing quality (scalar chain, one query) ──
-- The GA ga-retrieval + routing-eval oracle pattern. `rels` is each query's
-- ranked relevance list (rel > 0 = relevant). All metrics are scalars over the
-- DOUBLE[] column, so this is a single declarative pipeline. avg(RR) = MRR.
WITH eval(query, rels) AS (
    VALUES ('q1', [3.0, 2.0, 0.0, 1.0]),
           ('q2', [0.0, 0.0, 1.0]),
           ('q3', [1.0, 0.0])
)
SELECT round(avg(ix_ndcg(rels, 10)), 4)          AS mean_ndcg,
       round(avg(ix_reciprocal_rank(rels)), 4)   AS mrr,
       round(avg(ix_precision_at_k(rels, 3)), 4) AS mean_p_at_3
FROM eval;

-- ── Pipeline 2: cluster → score (table-fn bridge) ────────────────────────────
-- ix_kmeans (labels) → ix_silhouette (quality). Two table functions, so bridge
-- via getvariable: materialize the labels as a JSON int list, pass as a constant.
SET VARIABLE vecs = '[[0,0],[0,1],[1,0],[10,10],[10,11],[11,10]]';
SET VARIABLE labels = (
    SELECT list(cluster ORDER BY row)::VARCHAR FROM ix_kmeans(getvariable('vecs'), 2)
);
SELECT round(avg(silhouette), 4) AS mean_silhouette  -- ~1.0 for the two tight blobs
FROM ix_silhouette(getvariable('vecs'), getvariable('labels'));

-- ── Pipeline 3: reduce → cluster → score (three table fns, bridged) ──────────
-- ix_pca_project (2-D coords) → ix_kmeans (labels on the projection) → silhouette.
SET VARIABLE proj = (
    SELECT list(coords ORDER BY row)::VARCHAR FROM ix_pca_project(getvariable('vecs'), 2)
);
SET VARIABLE proj_labels = (
    SELECT list(cluster ORDER BY row)::VARCHAR FROM ix_kmeans(getvariable('proj'), 2)
);
SELECT round(avg(silhouette), 4) AS mean_silhouette_on_pca
FROM ix_silhouette(getvariable('proj'), getvariable('proj_labels'));

-- ── Pipeline 4: music set-class histogram of voicings (scalar over JSONL) ─────
-- Scalars chain over a column; needs the corpus file. ix_forte_number(midiNotes)
-- groups the voicing corpus by pitch-class set-class — impossible in any other DB.
-- (Inline demo; swap the VALUES for read_json_auto('state/voicings/raw/guitar.jsonl').)
WITH voicings(midiNotes) AS (
    VALUES ([60, 64, 67]), ([62, 65, 69]), ([60, 63, 67]), ([60, 64, 67, 70])
)
SELECT ix_forte_number(midiNotes) AS set_class,
       ix_classify_triad(midiNotes) AS triad,
       count(*) AS n
FROM voicings
GROUP BY set_class, triad
ORDER BY n DESC;

-- ── Pipeline 5: embedding QA (Tier-3 production index) ────────────────────────
-- ix_optick_scan exposes the OPTIC-K mmap; here a quick distribution check.
-- (Needs a real index path; uncomment and point at your optick.index.)
-- SELECT instrument, count(*) AS voicings, any_value(len(embedding)) AS dim
-- FROM ix_optick_scan('C:/Users/you/source/repos/ga/state/voicings/optick.index')
-- GROUP BY instrument;
