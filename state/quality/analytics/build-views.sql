-- build-views.sql — DuckDB quality analytics layer for ix (the producer side)
--
-- Materializes ix's file-based quality / eval / metrics artifacts under
-- state/quality/ (plus the cross-repo SAE contract output and the router-spike
-- eval) into a self-contained quality.duckdb so trends are queryable across
-- sessions and from Rust (crates/ix-duck, example `ix_quality_lens`) without
-- depending on bespoke JSON-glob loaders that silently skip off-pattern files.
--
-- Mirrors the GA design at ga/state/quality/analytics/build-views.sql.
--
-- Refresh (run from the artifact root, state/quality/):
--   duckdb analytics/quality.duckdb < analytics/build-views.sql
--
-- Design notes:
--   * read_json_auto(glob, filename=true, union_by_name=true) tolerates schema
--     drift between snapshots and lets us recover the date key from the path.
--   * Tables are materialized (CREATE OR REPLACE TABLE AS) so the .duckdb is
--     portable; re-run this script to pick up new artifacts.
--   * The day is parsed from the filename / dir via regexp_extract on the
--     canonical YYYY-MM-DD stem convention.

-- optick-sae: the cross-repo CONTRACT OUTPUT ix produces for GA's consumer side
-- (state/quality/optick-sae/<date>/optick-sae-artifact.json, per
-- ga/docs/contracts/2026-05-02-optick-sae-artifact.contract.md). Columns are the
-- flattened contract shape so producer (ix) and consumer (GA) have SQL parity
-- over the same fields. No artifact is emitted on disk yet, so — exactly like
-- GA's pr_grades — this starts as an explicit empty table with the contract
-- schema. The moment a state/quality/optick-sae/<date>/optick-sae-artifact.json
-- lands, swap the body for the commented read below and re-run:
--
--   CREATE OR REPLACE TABLE optick_sae AS
--   SELECT
--       regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day,
--       artifact_id, schema_version, trained_at, trainer, trainer_version,
--       input.optick_index_path        AS input_optick_index_path,
--       input.optick_index_sha         AS input_optick_index_sha,
--       input.optick_dim               AS input_optick_dim,
--       input.compact_training_dim     AS input_compact_training_dim,
--       input.schema_version           AS input_schema_version,
--       input.corpus_size              AS input_corpus_size,
--       input.partitions_used          AS input_partitions_used,
--       model.kind                     AS model_kind,
--       model.dict_size                AS model_dict_size,
--       model.k_sparse                 AS model_k_sparse,
--       model.training.epochs          AS model_epochs,
--       model.training.batch_size      AS model_batch_size,
--       model.training.lr              AS model_lr,
--       model.training.seed            AS model_seed,
--       model.training.loss_final      AS model_loss_final,
--       model.training.sparsity_actual_mean AS model_sparsity_actual_mean,
--       metrics.reconstruction_mse     AS reconstruction_mse,
--       metrics.reconstruction_r2      AS reconstruction_r2,
--       metrics.active_features_per_voicing_p50 AS active_features_p50,
--       metrics.active_features_per_voicing_p95 AS active_features_p95,
--       metrics.dead_features_pct      AS dead_features_pct,
--       metrics.feature_partition_purity_mean AS purity_mean,
--       metrics.feature_partition_purity_p10  AS purity_p10,
--       features_summary.total         AS features_total,
--       features_summary.alive         AS features_alive,
--       features_summary.high_frequency_count AS features_high_freq,
--       features_summary.low_frequency_count  AS features_low_freq,
--       links.supersedes               AS links_supersedes,
--       narrative
--   FROM read_json_auto('optick-sae/*/optick-sae-artifact.json',
--                       filename = true, union_by_name = true)
--   ORDER BY day;
CREATE OR REPLACE TABLE optick_sae (
    day                          VARCHAR,
    artifact_id                  VARCHAR,
    schema_version               BIGINT,
    trained_at                   VARCHAR,
    trainer                      VARCHAR,
    trainer_version              VARCHAR,
    input_optick_index_path      VARCHAR,
    input_optick_index_sha       VARCHAR,
    input_optick_dim             BIGINT,
    input_compact_training_dim   BIGINT,
    input_schema_version         VARCHAR,
    input_corpus_size            BIGINT,
    input_partitions_used        VARCHAR[],
    model_kind                   VARCHAR,
    model_dict_size              BIGINT,
    model_k_sparse               BIGINT,
    model_epochs                 BIGINT,
    model_batch_size             BIGINT,
    model_lr                     DOUBLE,
    model_seed                   BIGINT,
    model_loss_final             DOUBLE,
    model_sparsity_actual_mean   DOUBLE,
    reconstruction_mse           DOUBLE,
    reconstruction_r2            DOUBLE,
    active_features_p50          BIGINT,
    active_features_p95          BIGINT,
    dead_features_pct            DOUBLE,
    purity_mean                  DOUBLE,
    purity_p10                   DOUBLE,
    features_total               BIGINT,
    features_alive               BIGINT,
    features_high_freq           BIGINT,
    features_low_freq            BIGINT,
    links_supersedes             VARCHAR,
    narrative                    VARCHAR
);

-- ix-harness: repo-level Agent Blackbox readiness (last.json is the current run;
-- emitted by scripts/verify.ps1). Single-file source, no date in the name — the
-- emitted_at timestamp carries the day.
CREATE OR REPLACE TABLE ix_harness AS
SELECT
    domain,
    emitted_at,
    regexp_extract(CAST(emitted_at AS VARCHAR), '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1) AS day,
    metric_name,
    TRY_CAST(metric_value AS DOUBLE)                                  AS metric_value,
    oracle_status,
    oracle_command,
    summary
FROM read_json_auto('ix-harness/last.json', filename = true, union_by_name = true);

-- quality-health: daily roll-up of the regression scan (quality-health-<date>.json).
CREATE OR REPLACE TABLE quality_health AS
SELECT
    COALESCE(
        CAST(generated_on AS VARCHAR),
        regexp_extract(filename, '([0-9]{4}-[0-9]{2}-[0-9]{2})', 1)
    )                                                                 AS day,
    status,
    total_metrics,
    regressions,
    drifts,
    TRY_CAST(regression_threshold_pct AS DOUBLE)                      AS regression_threshold_pct
FROM read_json_auto('quality-health-*.json', filename = true, union_by_name = true)
ORDER BY day;

-- router-eval: the router-spike head-eval (semantic intent router accuracy /
-- macro-F1 / OOS decline over the held-out test set). Lives one level up under
-- state/router-spike/, read via a relative path from state/quality/.
CREATE OR REPLACE TABLE router_eval AS
SELECT
    generated_for,
    dim_used,
    TRY_CAST(test_inscope_argmax_acc AS DOUBLE)                       AS test_inscope_argmax_acc,
    TRY_CAST(test_inscope_acc_with_tau AS DOUBLE)                     AS test_inscope_acc_with_tau,
    TRY_CAST(test_macro_f1 AS DOUBLE)                                 AS test_macro_f1,
    TRY_CAST(test_oos_decline_rate AS DOUBLE)                         AS test_oos_decline_rate,
    TRY_CAST(delta_argmax_pp AS DOUBLE)                               AS delta_argmax_pp,
    verdict
FROM read_json_auto('../router-spike/head-eval.json', filename = true, union_by_name = true);

-- Unified latest-value-per-source rollup. A view over the materialized tables,
-- so it carries no JSON path dependency and is safe to query from anywhere.
-- (optick_sae is omitted from the rollup body when empty would yield no row; it
-- is included so it lights up automatically once an artifact lands.)
CREATE OR REPLACE VIEW quality_latest AS
SELECT 'optick_sae'     AS source, day, reconstruction_r2     AS metric, 'reconstruction_r2' AS metric_name FROM optick_sae     QUALIFY row_number() OVER (ORDER BY day DESC) = 1
UNION ALL
SELECT 'ix_harness'     AS source, day, metric_value          AS metric, metric_name          AS metric_name FROM ix_harness     QUALIFY row_number() OVER (ORDER BY day DESC) = 1
UNION ALL
SELECT 'quality_health' AS source, day, total_metrics::DOUBLE AS metric, 'total_metrics'     AS metric_name FROM quality_health QUALIFY row_number() OVER (ORDER BY day DESC) = 1
UNION ALL
SELECT 'router_eval'    AS source, NULL AS day, test_macro_f1 AS metric, 'test_macro_f1'     AS metric_name FROM router_eval    QUALIFY row_number() OVER (ORDER BY test_macro_f1 DESC) = 1;
