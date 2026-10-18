-- The probe samples as DuckDB views, for agents and people alike:
--   duckdb -c ".read probes.sql" -c "FROM probes_latest"
-- It reads the host's output under %LOCALAPPDATA%/ix-probes, where the long-running host writes
-- (host.exe -out "%LOCALAPPDATA%\ix-probes"). The host's ts is UTC and now() is local, so the
-- windows compare against now() taken in UTC.
CREATE OR REPLACE VIEW probes AS
    SELECT ts::TIMESTAMP AS ts, probe, build, metric, value, unit
    FROM read_json_auto(getenv('LOCALAPPDATA') || '/ix-probes/probes.jsonl');

-- One row per metric: its latest value in the last minute, and when it was taken.
CREATE OR REPLACE VIEW probes_latest AS
    SELECT metric, arg_max(value, ts) AS value, any_value(unit) AS unit, max(ts) AS at
    FROM probes
    WHERE ts > (now() AT TIME ZONE 'UTC') - INTERVAL 1 MINUTE
    GROUP BY metric ORDER BY metric;
