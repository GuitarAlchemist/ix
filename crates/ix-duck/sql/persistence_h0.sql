-- Persistent homology (H0) over a DuckDB column -- gap-matrix row B1b.
--
-- This is the DuckDB half of the pipeline. The Rust half is the general
-- persistence engine `ix_topo::simplex::rips_complex` +
-- `ix_topo::persistence::compute_persistence`, and the two are pinned to the
-- same frozen golden file
-- (`crates/ix-duck/tests/fixtures/persistence/golden-h0.csv`), so neither is
-- the source of truth for the other -- the golden is, and drift in either side
-- is a test failure.
--
-- It is plain SQL, not an ix-duck Rust UDF, on purpose: the `duck` / `udf`
-- features of ix-duck are never compiled by `cargo build --workspace` or by CI,
-- so logic placed in a UDF would be invisible to every CI job. These macros run
-- on the stock `duckdb` CLI with no build step. Same reasoning, and the same
-- two-surface shape, as `pareto_frontier.sql` (issue #294).
--
-- SCOPE -- H0 only, and that is a property of the input, not a shortcut. One
-- DuckDB column is a point cloud in R^1. Over R^1 a Vietoris-Rips complex
-- capped at dimension 1 does report H1 classes, but every one of them is an
-- artefact of the cap -- there are no triangles in the filtration to fill the
-- loops that the complete edge set creates. H1 and above only mean something
-- with >= 2 coordinates, which is a different input shape (a LIST column or a
-- wide table) and needs the general engine, not a window function. Anything
-- claiming Hk>0 from a single column would be measuring its own truncation.
--
-- WHY IT IS CLOSED-FORM -- for points on a line, the H0 deaths of the Rips
-- filtration are exactly the edge weights of the minimum spanning tree, and the
-- MST of a 1-D point set is the path through the sorted points. So the finite
-- H0 pairs are `(0, gap)` for each consecutive gap of the sorted DISTINCT
-- values, plus one essential `(0, inf)` class per series. That is a `lag()`
-- window -- O(n log n) instead of the engine's boundary-matrix reduction. The
-- golden is what proves the two agree; the argument above is only why they
-- should.
--
-- USAGE -- note the order: DuckDB binds a table macro's body when the macro is
-- created, so `ix_topo_input` has to exist before this file is read.
--   CREATE OR REPLACE TABLE ix_topo_input AS SELECT * FROM read_csv(...);
--   .read crates/ix-duck/sql/persistence_h0.sql
--   SELECT * FROM ix_persistence_h0();
--
-- INPUT  ix_topo_input(series VARCHAR, value DOUBLE)
-- OUTPUT ix_persistence_h0() -> (series, ordinal, dim, birth, death)
--        one row per H0 class; `birth`/`death` are rendered with six fixed
--        decimals (the essential class renders as `inf`).
--        ix_betti_0_at(radius) -> (series, betti_0)
--
-- Output is advisory and read-only. Nothing here writes state or gates a merge.

--------------------------------------------------------------------------------
-- Validation. Fail closed: every violation is collected, and the diagram macro
-- refuses to emit anything at all if there is one.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_topo_violations() AS TABLE

-- 1. empty identifiers
  SELECT 1 AS ord, 'EmptyField' AS code, 'series' AS detail
  FROM ix_topo_input
  WHERE series IS NULL OR trim(series) = ''

-- 2. finite values only (inf / -inf / nan / NULL are all rejected)
UNION ALL
  SELECT 2, 'NonFiniteValue', coalesce(series, 'NULL')
  FROM ix_topo_input
  WHERE value IS NULL OR NOT isfinite(value)

-- 3. Distinct values closer together than 1e-9. Two surfaces have to agree on
--    what counts as "the same point": `compute_persistence` drops any pair
--    whose lifetime is <= 1e-15, and `printf('%.6f')` renders anything below
--    5e-7 as 0.000000. Rather than pick one of those two thresholds and hope
--    the other never bites, reject the whole band. A caller that genuinely has
--    points this close should rescale before asking for a diagram.
UNION ALL
  SELECT 3, 'NearDuplicateValue',
         series || ' gap=' || printf('%.3e', gap)
  FROM (
    SELECT series,
           value - lag(value) OVER (PARTITION BY series ORDER BY value) AS gap
    FROM (SELECT DISTINCT series, value FROM ix_topo_input)
  )
  WHERE gap IS NOT NULL AND gap < 1e-9

-- 4. an empty table clears every check above, so it is caught last
UNION ALL
  SELECT 4, 'EmptyInput', ''
  WHERE (SELECT count(*) FROM ix_topo_input) = 0
;

--------------------------------------------------------------------------------
-- The H0 persistence diagram.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_persistence_h0() AS TABLE
WITH
-- The guard is a one-row aggregate, so it is always evaluated; a violation
-- raises before a single diagram row is produced. It is cross-joined on the
-- LEFT of the data so an empty (and therefore invalid) input still trips it.
guard AS (
  SELECT CASE
           WHEN count(*) = 0 THEN true
           ELSE error(
             'ix_persistence_h0: input rejected (' || count(*) || ' violation(s)); first: '
             || min(ord::VARCHAR || ' ' || code || ' ' || detail)
           )
         END AS ok
  FROM ix_topo_violations()
),

-- DISTINCT is the merge-at-radius-0 step: co-located points are one component
-- from the start, and the engine drops their zero-lifetime pairs.
pts AS (SELECT DISTINCT series, value FROM ix_topo_input),

-- Finite classes: consecutive gaps of the sorted values = MST edge weights.
finite AS (
  SELECT series,
         value - lag(value) OVER (PARTITION BY series ORDER BY value) AS death
  FROM pts
),

-- Essential class: exactly one per series -- the component that never merges.
essential AS (
  SELECT DISTINCT series, 'Infinity'::DOUBLE AS death FROM pts
),

classes AS (
  SELECT series, death FROM finite WHERE death IS NOT NULL
  UNION ALL
  SELECT series, death FROM essential
)

SELECT c.series,
       -- Ordinal enumerates the multiset in death order. Equal deaths render to
       -- byte-identical rows, so which of them gets which ordinal cannot change
       -- the output bytes -- and (series, ordinal) is still a key over rows.
       row_number() OVER (PARTITION BY c.series ORDER BY c.death) AS ordinal,
       0 AS dim,
       '0.000000' AS birth,
       CASE WHEN isfinite(c.death) THEN printf('%.6f', c.death) ELSE 'inf' END AS death
FROM guard g, classes c
WHERE g.ok
ORDER BY c.series, c.death
;

--------------------------------------------------------------------------------
-- Betti_0 at a filtration radius: the number of connected components once every
-- pair within `radius` has been joined. One component, plus one more for each
-- gap the radius does not bridge.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_betti_0_at(radius) AS TABLE
SELECT series,
       count(*) FILTER (WHERE death = 'inf' OR death::DOUBLE > radius) AS betti_0
FROM ix_persistence_h0()
GROUP BY series
ORDER BY series
;
