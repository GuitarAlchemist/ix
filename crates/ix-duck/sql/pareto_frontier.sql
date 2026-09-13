-- Deterministic Pareto frontier over a long-form objective table (issue #294).
--
-- This is the DuckDB half of the pipeline. The Rust half is
-- `ix_evolution::frontier` and the two are pinned to the same frozen golden
-- file (`crates/ix-duck/tests/fixtures/pareto/golden-frontier.csv`), so neither
-- is the source of truth for the other -- the golden is, and drift in either
-- side is a test failure.
--
-- It is plain SQL, not an ix-duck Rust UDF, on purpose: the `duck` / `udf`
-- features of ix-duck are never compiled by `cargo build --workspace` or by CI,
-- so determinism logic placed in a UDF would be invisible to every CI job.
-- These macros run on the stock `duckdb` CLI with no build step.
--
-- USAGE -- note the order: DuckDB binds a table macro's body when the macro is
-- created, so `ix_pareto_input` has to exist before this file is read.
--   CREATE OR REPLACE TABLE ix_pareto_input AS SELECT * FROM read_csv(...);
--   .read crates/ix-duck/sql/pareto_frontier.sql
--   SELECT * FROM ix_pareto_frontier();
--
-- INPUT  ix_pareto_input(subject_revision, task_class, candidate_id,
--                        metric, direction VARCHAR in ('MIN','MAX'), value DOUBLE)
-- OUTPUT (subject_revision, task_class, candidate_id, objectives)
--        one row per non-dominated candidate; `objectives` is
--        `metric:DIRECTION=%.6f` joined by ';' in ascending metric order.
--
-- Output is advisory and read-only. Nothing here writes state or gates a merge.

--------------------------------------------------------------------------------
-- Validation. Fail closed: every violation is collected, and the frontier macro
-- refuses to emit anything at all if there is one. `ord` reproduces the fixed
-- check sequence of `ix_evolution::frontier`, so both surfaces report the same
-- *class* of violation first. The `detail` strings are formatted per surface
-- and are not promised to match character for character; what is promised is
-- that neither surface ever emits a frontier for an input the other rejects.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_pareto_violations() AS TABLE

-- 1. empty identifiers
  SELECT 1 AS ord, 'EmptyField' AS code, 'subject_revision' AS detail
  FROM ix_pareto_input
  WHERE subject_revision IS NULL OR trim(subject_revision) = ''
UNION ALL
  SELECT 1, 'EmptyField', 'task_class'
  FROM ix_pareto_input
  WHERE task_class IS NULL OR trim(task_class) = ''
UNION ALL
  SELECT 1, 'EmptyField', 'candidate_id'
  FROM ix_pareto_input
  WHERE candidate_id IS NULL OR trim(candidate_id) = ''
UNION ALL
  SELECT 1, 'EmptyField', 'metric'
  FROM ix_pareto_input
  WHERE metric IS NULL OR trim(metric) = ''

-- 2. direction spelling: MIN or MAX, never guessed
UNION ALL
  SELECT 2, 'UnknownDirection',
         candidate_id || '/' || metric || '=' || coalesce(direction, 'NULL')
  FROM ix_pareto_input
  WHERE direction IS NULL OR direction NOT IN ('MIN', 'MAX')

-- 3. duplicate (revision, class, candidate, metric). Rejecting these is what
--    makes (subject_revision, task_class, candidate_id) a key over output rows,
--    which is the whole basis of the total output order.
UNION ALL
  SELECT 3, 'DuplicateMetric',
         subject_revision || '/' || task_class || '/' || candidate_id || '/' || metric
  FROM ix_pareto_input
  GROUP BY subject_revision, task_class, candidate_id, metric
  HAVING count(*) > 1

-- 4. one direction per (revision, class, metric)
UNION ALL
  SELECT 4, 'MixedDirection',
         subject_revision || '/' || task_class || '/' || metric
  FROM ix_pareto_input
  GROUP BY subject_revision, task_class, metric
  HAVING count(DISTINCT direction) > 1

-- 5. every candidate exposes the whole metric set its task class declares
UNION ALL
  SELECT 5, 'MissingMetric',
         c.subject_revision || '/' || c.task_class || '/' || c.candidate_id || '/' || d.metric
  FROM (SELECT DISTINCT subject_revision, task_class, candidate_id FROM ix_pareto_input) c
  JOIN (SELECT DISTINCT subject_revision, task_class, metric FROM ix_pareto_input) d
    ON c.subject_revision = d.subject_revision AND c.task_class = d.task_class
  WHERE NOT EXISTS (
    SELECT 1 FROM ix_pareto_input s
    WHERE s.subject_revision = c.subject_revision
      AND s.task_class = c.task_class
      AND s.candidate_id = c.candidate_id
      AND s.metric = d.metric
  )

-- 6. finite values only (inf / -inf / nan / NULL are all rejected)
UNION ALL
  SELECT 6, 'NonFiniteValue', candidate_id || '/' || metric
  FROM ix_pareto_input
  WHERE value IS NULL OR NOT isfinite(value)

-- 7. an empty table clears every check above, so it is caught last
UNION ALL
  SELECT 7, 'EmptyInput', ''
  WHERE (SELECT count(*) FROM ix_pareto_input) = 0
;

--------------------------------------------------------------------------------
-- Frontier.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_pareto_frontier() AS TABLE
WITH
-- The guard is a one-row aggregate, so it is always evaluated; a violation
-- raises before a single frontier row is produced. It is cross-joined on the
-- LEFT of the data so an empty (and therefore invalid) input still trips it.
guard AS (
  SELECT CASE
           WHEN count(*) = 0 THEN true
           -- min() over 'ord code detail' is a deterministic pick, and putting
           -- `ord` first makes it the check-sequence order rather than the
           -- alphabetical order of the violation names.
           ELSE error(
             'ix_pareto_frontier: input rejected (' || count(*) || ' violation(s)); first: '
             || min(ord::VARCHAR || ' ' || code || ' ' || detail)
           )
         END AS ok
  FROM ix_pareto_violations()
),

-- Pairwise dominance. `a` dominates `b` iff `a` is no worse on every metric and
-- strictly better on at least one -- the strictness is the mechanism: drop it
-- and two candidates with identical vectors would eliminate each other.
-- bool_and / bool_or are commutative and associative over booleans, so neither
-- scan order nor DuckDB's parallelism can change the verdict.
pairs AS (
  SELECT a.subject_revision,
         a.task_class,
         a.candidate_id AS winner,
         b.candidate_id AS loser,
         bool_and(CASE WHEN a.direction = 'MIN' THEN a.value <= b.value
                                                ELSE a.value >= b.value END) AS no_worse,
         bool_or (CASE WHEN a.direction = 'MIN' THEN a.value <  b.value
                                                ELSE a.value >  b.value END) AS strictly_better
  FROM ix_pareto_input a
  JOIN ix_pareto_input b
    ON a.subject_revision = b.subject_revision   -- revision isolation
   AND a.task_class       = b.task_class         -- task-class isolation
   AND a.metric           = b.metric
  WHERE a.candidate_id <> b.candidate_id
  GROUP BY a.subject_revision, a.task_class, a.candidate_id, b.candidate_id
),
dominated AS (
  SELECT DISTINCT subject_revision, task_class, loser
  FROM pairs
  WHERE no_worse AND strictly_better
),

-- Canonical objective rendering: metrics ascending, six fixed decimals (which
-- C's %.6f and Rust's {:.6} agree on). Dominance above used the full DOUBLE.
wide AS (
  SELECT subject_revision,
         task_class,
         candidate_id,
         string_agg(metric || ':' || direction || '=' || printf('%.6f', value), ';'
                    ORDER BY metric) AS objectives
  FROM ix_pareto_input
  GROUP BY subject_revision, task_class, candidate_id
)

SELECT w.subject_revision, w.task_class, w.candidate_id, w.objectives
FROM guard g, wide w
WHERE g.ok
  AND NOT EXISTS (
    SELECT 1 FROM dominated d
    WHERE d.subject_revision = w.subject_revision
      AND d.task_class       = w.task_class
      AND d.loser            = w.candidate_id
  )
-- Total order. Validation rule 3 makes this triple a key over output rows, so
-- distinct rows always differ in some component and byte comparison never
-- returns "equal" -- there is no residual tie left for a further rule to break.
-- Ties in objective *values* do not collapse rows: mutually non-dominated
-- candidates both survive and are separated here by candidate_id.
ORDER BY w.subject_revision, w.task_class, w.candidate_id
;
