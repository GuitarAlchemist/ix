-- Golden driver for the H0 persistence pipeline (gap-matrix row B1b).
--
-- Reproduce the frozen golden result with ONE DuckDB CLI command, run from the
-- repository root:
--
--   duckdb -csv -c ".read crates/ix-duck/sql/persistence_h0_golden.sql"
--
-- Its stdout must equal
-- `crates/ix-duck/tests/fixtures/persistence/golden-h0.csv` byte for byte.
-- The fixture rows are deliberately shuffled -- values interleaved across
-- series and out of numeric order -- so a pipeline that leaked input order
-- could not reproduce this output.
--
-- No network, no extensions, no credentials: read_csv over one local file.

CREATE OR REPLACE TABLE ix_topo_input AS
SELECT * FROM read_csv(
  'crates/ix-duck/tests/fixtures/persistence/series.csv',
  header = true,
  columns = {
    'series': 'VARCHAR',
    'value':  'DOUBLE'
  }
);

-- The macros bind their input table at CREATE time, so the table is created
-- first and the macro file is read second.
.read crates/ix-duck/sql/persistence_h0.sql

SELECT * FROM ix_persistence_h0();
