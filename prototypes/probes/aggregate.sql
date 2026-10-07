-- Aggregates what the probes wrote. Run from prototypes/probes:
--   duckdb -c ".read aggregate.sql"

-- One row per metric: how many samples, and their range.
SELECT probe, metric, unit,
       count(*)              AS n,
       round(min(value), 1)  AS min,
       round(avg(value), 1)  AS avg,
       round(max(value), 1)  AS max,
       max(ts)               AS last_ts
FROM read_json_auto('out/probes.jsonl')
GROUP BY ALL
ORDER BY probe, metric;

-- One row per probe build: the hot reloads, in order.
SELECT probe, build,
       min(ts)                AS first_ts,
       max(ts)                AS last_ts,
       count(*)               AS n,
       list(DISTINCT metric ORDER BY metric) AS metrics
FROM read_json_auto('out/probes.jsonl')
GROUP BY ALL
ORDER BY first_ts;
