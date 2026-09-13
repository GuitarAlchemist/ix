-- Golden driver for the deterministic Pareto frontier pipeline (issue #294).
--
-- Reproduce the frozen golden result with ONE DuckDB CLI command, run from the
-- repository root:
--
--   duckdb -csv -c ".read crates/ix-duck/sql/pareto_frontier_golden.sql"
--
-- Its stdout must equal
-- `crates/ix-duck/tests/fixtures/pareto/golden-frontier.csv` byte for byte.
-- The fixture rows are deliberately shuffled -- not grouped, not sorted, with
-- dominated rows interleaved -- so a pipeline that leaked input order could not
-- reproduce this output.
--
-- No network, no extensions, no credentials: read_csv over one local file.

CREATE OR REPLACE TABLE ix_pareto_input AS
SELECT * FROM read_csv(
  'crates/ix-duck/tests/fixtures/pareto/objectives.csv',
  header = true,
  columns = {
    'subject_revision': 'VARCHAR',
    'task_class':       'VARCHAR',
    'candidate_id':     'VARCHAR',
    'metric':           'VARCHAR',
    'direction':        'VARCHAR',
    'value':            'DOUBLE'
  }
);

-- The macros bind their input table at CREATE time, so the table is created
-- first and the macro file is read second.
.read crates/ix-duck/sql/pareto_frontier.sql

SELECT * FROM ix_pareto_frontier();
