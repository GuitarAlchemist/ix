-- Invoke a compiled IXQL pipeline plan from DuckDB (issue #185 follow-on; refs #281).
--
-- The Rust half is `ix_ixql::compile`, which turns an IXQL program into a stage
-- DAG. It emits the *structure* only -- ordinal, stage id, kind, callee, and the
-- upstream stage ids. It deliberately does NOT emit the execution schedule.
-- This file re-derives that schedule from the structure alone, so the two
-- surfaces compute the same answer independently and are pinned to the same
-- frozen golden (`crates/ix-duck/tests/fixtures/ixql/golden-schedule.csv`).
-- Neither side is the source of truth for the other -- the golden is.
--
-- It is plain SQL, not an ix-duck Rust UDF, on purpose. The `duck` / `udf`
-- features of ix-duck are never compiled by `cargo build --workspace` or by any
-- CI job, and `ix-duck-ext` is excluded from the workspace outright, so logic
-- placed in a UDF would be invisible to every CI job. These macros run on the
-- stock `duckdb` CLI with no build step. (Same reasoning as
-- `pareto_frontier.sql`, which established it.)
--
-- USAGE -- note the order: DuckDB binds a table macro's body when the macro is
-- created, so `ix_ixql_plan_input` has to exist before this file is read.
--   CREATE OR REPLACE TABLE ix_ixql_plan_input AS SELECT * FROM read_csv(...);
--   .read crates/ix-duck/sql/ixql_plan.sql
--   SELECT * FROM ix_ixql_schedule();
--
-- INPUT  ix_ixql_plan_input(ordinal BIGINT, stage_id VARCHAR, kind VARCHAR,
--                           op VARCHAR, deps VARCHAR)
--        `deps` is ';'-joined upstream stage ids, empty for a root.
-- OUTPUT ix_ixql_schedule() -> (level, stage_count, stages)
--        one row per parallel level; `stages` is ';'-joined in ascending id
--        order. Level N contains every stage all of whose dependencies sit in
--        levels < N, which is the same rule `ix_pipeline::dag::parallel_levels`
--        applies (level = 1 + max(level of predecessors)).
--
-- Output is advisory and read-only. Nothing here writes state or gates a merge.

--------------------------------------------------------------------------------
-- Edges. One row per dependency. `deps` is a ';'-joined list and DuckDB's
-- string_split returns [''] for the empty string, so the empty parts are
-- filtered rather than becoming an edge from a stage named ''.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_ixql_plan_edges() AS TABLE
  SELECT p.stage_id AS child,
         trim(dep)  AS parent
  FROM ix_ixql_plan_input p,
       UNNEST(string_split(p.deps, ';')) AS t(dep)
  WHERE trim(dep) <> ''
;

--------------------------------------------------------------------------------
-- Validation. Fail closed: every violation is collected, and the schedule macro
-- refuses to emit anything at all if there is one. `ord` fixes the check
-- sequence so the first reported violation is stable.
--
-- Rule 5 (no forward dependency) is what makes the recursion below terminating:
-- every edge points at a strictly lower ordinal, so the graph is acyclic by
-- construction and the recursive CTE cannot loop.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_ixql_plan_violations() AS TABLE

-- 1. empty identifiers
  SELECT 1 AS ord, 'EmptyStageId' AS code, '' AS detail
  FROM ix_ixql_plan_input
  WHERE stage_id IS NULL OR trim(stage_id) = ''

-- 2. kind must be one of the six the compiler emits, never guessed
UNION ALL
  SELECT 2, 'UnknownKind', stage_id || '=' || coalesce(kind, 'NULL')
  FROM ix_ixql_plan_input
  WHERE kind IS NULL
     OR kind NOT IN ('bind', 'source', 'pipe', 'compound', 'effect', 'when')

-- 3. stage_id is a key. Without this the edge join below would fan out.
UNION ALL
  SELECT 3, 'DuplicateStageId', stage_id
  FROM ix_ixql_plan_input
  GROUP BY stage_id
  HAVING count(*) > 1

-- 4. every dependency names a stage that exists
UNION ALL
  SELECT 4, 'DanglingDependency', e.child || '<-' || e.parent
  FROM ix_ixql_plan_edges() e
  WHERE NOT EXISTS (
    SELECT 1 FROM ix_ixql_plan_input p WHERE p.stage_id = e.parent
  )

-- 5. dependencies point backwards in compiled order (acyclicity witness)
UNION ALL
  SELECT 5, 'ForwardDependency', e.child || '<-' || e.parent
  FROM ix_ixql_plan_edges() e
  JOIN ix_ixql_plan_input c ON c.stage_id = e.child
  JOIN ix_ixql_plan_input p ON p.stage_id = e.parent
  WHERE p.ordinal >= c.ordinal

-- 6. ordinals are a dense 0-based range, so "compiled order" is total
UNION ALL
  SELECT 6, 'OrdinalGap', 'expected 0..' || ((SELECT count(*) FROM ix_ixql_plan_input) - 1)
  WHERE (SELECT count(DISTINCT ordinal) FROM ix_ixql_plan_input)
        <> (SELECT count(*) FROM ix_ixql_plan_input)
     OR (SELECT coalesce(min(ordinal), 0) FROM ix_ixql_plan_input) <> 0
     OR (SELECT coalesce(max(ordinal), -1) FROM ix_ixql_plan_input)
        <> (SELECT count(*) - 1 FROM ix_ixql_plan_input)

-- 7. an empty plan clears every check above, so it is caught last
UNION ALL
  SELECT 7, 'EmptyInput', ''
  WHERE (SELECT count(*) FROM ix_ixql_plan_input) = 0
;

--------------------------------------------------------------------------------
-- Levels. A stage's level is the length of the longest path reaching it, which
-- is exactly `1 + max(level of predecessors)` -- the rule
-- `ix_pipeline::dag::parallel_levels` uses. The recursion enumerates paths and
-- `max()` picks the longest; a stage reachable by both a short and a long path
-- must wait for the long one, so taking the max (not the min, not the first) is
-- the whole correctness argument.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_ixql_plan_levels() AS TABLE
WITH RECURSIVE
reached(stage_id, level) AS (
    SELECT p.stage_id, 0
    FROM ix_ixql_plan_input p
    WHERE NOT EXISTS (
      SELECT 1 FROM ix_ixql_plan_edges() e WHERE e.child = p.stage_id
    )
  UNION ALL
    SELECT e.child, r.level + 1
    FROM reached r
    JOIN ix_ixql_plan_edges() e ON e.parent = r.stage_id
    -- Belt and braces. Rule 5 already makes a cycle impossible, but an
    -- unbounded recursion is a bad way to find that out.
    WHERE r.level < 10000
)
SELECT stage_id, max(level) AS level
FROM reached
GROUP BY stage_id
;

--------------------------------------------------------------------------------
-- Schedule. One row per parallel level.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_ixql_schedule() AS TABLE
WITH
-- The guard is a one-row aggregate, so it is always evaluated; a violation
-- raises before a single schedule row is produced. Cross-joined on the LEFT of
-- the data so an empty (and therefore invalid) plan still trips it.
guard AS (
  SELECT CASE
           WHEN count(*) = 0 THEN true
           ELSE error(
             'ix_ixql_schedule: plan rejected (' || count(*) || ' violation(s)); first: '
             || min(ord::VARCHAR || ' ' || code || ' ' || detail)
           )
         END AS ok
  FROM ix_ixql_plan_violations()
),
levelled AS (
  SELECT level,
         count(*) AS stage_count,
         string_agg(stage_id, ';' ORDER BY stage_id) AS stages
  FROM ix_ixql_plan_levels()
  GROUP BY level
)
SELECT l.level, l.stage_count, l.stages
FROM guard g, levelled l
WHERE g.ok
-- Total order: `level` is a key over output rows (one row per level), so byte
-- comparison never returns "equal" and no residual tie is left.
ORDER BY l.level
;

--------------------------------------------------------------------------------
-- Convenience: stages a given stage transitively depends on. This is the query
-- the plan exists to make answerable and the interpreter cannot answer at all.
--------------------------------------------------------------------------------
CREATE OR REPLACE MACRO ix_ixql_plan_closure(target) AS TABLE
WITH RECURSIVE
upstream(stage_id, depth) AS (
    SELECT e.parent, 1
    FROM ix_ixql_plan_edges() e
    WHERE e.child = target
  UNION ALL
    SELECT e.parent, u.depth + 1
    FROM upstream u
    JOIN ix_ixql_plan_edges() e ON e.child = u.stage_id
    WHERE u.depth < 10000
)
SELECT stage_id, max(depth) AS depth
FROM upstream
GROUP BY stage_id
ORDER BY max(depth), stage_id
;
