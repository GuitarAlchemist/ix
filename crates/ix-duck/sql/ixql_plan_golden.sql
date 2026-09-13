-- Golden driver for the compiled-IXQL schedule (refs #281).
--
-- Reproduce the frozen golden result with ONE DuckDB CLI command, run from the
-- repository root:
--
--   duckdb -csv -c ".read crates/ix-duck/sql/ixql_plan_golden.sql"
--
-- Its stdout must equal
-- `crates/ix-duck/tests/fixtures/ixql/golden-schedule.csv` byte for byte.
--
-- The input is the compiled plan of Demerzel's real `qa-architect-cycle.ixql`,
-- produced by `ix_ixql::compile` and regenerable with:
--
--   cargo run -q -p ix-ixql --bin ixql-plan -- \
--     crates/ix-ixql/tests/fixtures/qa-architect-cycle.ixql \
--     > crates/ix-duck/tests/fixtures/ixql/qa-architect-cycle-plan.csv
--
-- The plan carries no scheduling column, so reproducing the golden here means
-- DuckDB derived the parallel levels from the dependency edges on its own. The
-- plan rows are deliberately NOT in level order -- `verdict_id` sits at ordinal
-- 6 but level 2, and the five level-0 stages are spread across ordinals 0-5 --
-- so a query that leaked compiled order could not reproduce this output.
--
-- No network, no extensions, no credentials: read_csv over one local file.

CREATE OR REPLACE TABLE ix_ixql_plan_input AS
SELECT * FROM read_csv(
  'crates/ix-duck/tests/fixtures/ixql/qa-architect-cycle-plan.csv',
  header = true,
  columns = {
    'ordinal':  'BIGINT',
    'stage_id': 'VARCHAR',
    'kind':     'VARCHAR',
    'op':       'VARCHAR',
    'deps':     'VARCHAR'
  }
);

-- read_csv turns an empty trailing field into NULL; the macros treat `deps` as
-- a possibly-empty string. Normalising here keeps the macro bodies free of
-- coalesce noise.
UPDATE ix_ixql_plan_input SET deps = '' WHERE deps IS NULL;
UPDATE ix_ixql_plan_input SET op = '' WHERE op IS NULL;

-- The macros bind their input table at CREATE time, so the table is created
-- first and the macro file is read second.
.read crates/ix-duck/sql/ixql_plan.sql

SELECT * FROM ix_ixql_schedule();
