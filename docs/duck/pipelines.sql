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

-- ── Pipeline 6: probabilistic sketches (build → blob → probe / merge) ─────────
-- Portable sketches as column values — things DuckDB exposes natively for none of:
-- set membership (Bloom), cardinality (HLL), frequency (Count-Min), delete (Cuckoo).
-- Build is a scalar over list(); the JSON blob then probes/merges cheaply.

-- 6a. Bloom: which candidates are NOT already in the seen-set (one filter, scanned).
WITH seen(id) AS (VALUES (1),(2),(3),(5),(8)),
     cand(id) AS (VALUES (2),(4),(8),(9)),
     filt AS (SELECT ix_bloom_build(list(id), 1000, 0.01) AS bf FROM seen)
SELECT cand.id,
       ix_bloom_contains(filt.bf, cand.id) AS probably_seen
FROM cand, filt
ORDER BY cand.id;

-- 6b. HyperLogLog: approx distinct count, and merge two partitions' sketches into
-- one cardinality estimate — without re-scanning either partition (~p=14, <1% err).
SET VARIABLE hll_a = (SELECT ix_hll_build(list(r), 14) FROM range(0, 700)   t(r));
SET VARIABLE hll_b = (SELECT ix_hll_build(list(r), 14) FROM range(300, 1000) t(r));
SELECT ix_hll_count(getvariable('hll_a'))                                AS distinct_a,   -- ~700
       ix_hll_count(ix_hll_merge(getvariable('hll_a'), getvariable('hll_b'))) AS distinct_union; -- ~1000 (overlap 300..700 counted once)

-- ── Pipeline 7: SQL over a codebase (ix-code; Tier B needs code-semantic) ─────
-- read_text() yields (filename, content) rows; the scalar ix_code_* / ix_ast_query
-- UDFs then analyse each file. Nothing else does code analysis in SQL.

-- 7a. Complexity hot-spots across the crate (Tier A — no tree-sitter).
SELECT filename, round(ix_code_complexity(content, filename), 0) AS cyclomatic
FROM read_text('crates/ix-duck/src/*.rs')
ORDER BY cyclomatic DESC
LIMIT 5;

-- 7b. Tree-sitter AST query: count function definitions per file (Tier B).
SELECT regexp_replace(filename, '.*[\\/]', '') AS file,
       len(from_json(ix_ast_query(content, 'rust',
            '(function_item name:(identifier) @fn)'), '["json"]')) AS fn_defs
FROM read_text('crates/ix-duck/src/*.rs')
ORDER BY fn_defs DESC
LIMIT 5;
