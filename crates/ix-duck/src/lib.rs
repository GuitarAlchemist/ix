//! `ix-duck` — an in-process DuckDB "analyst bench" over IX telemetry, with IX
//! algorithms exposed as SQL UDFs.
//!
//! DuckDB here is the analyst's bench — **not** a production engine and **not** a
//! source of truth (see `docs/DUCKDB.md`). All DuckDB code lives behind the optional
//! `duck` feature, which pulls a bundled (C++) DuckDB build; the default workspace /
//! CI build never compiles it.
//!
//! ```no_run
//! # #[cfg(feature = "duck")]
//! # fn demo() -> duckdb::Result<()> {
//! let conn = ix_duck::open_bench()?;
//! let yield_: Option<f64> = conn.query_row(
//!     "SELECT avg(coverage_max) FROM read_json_auto('state/thinking-machine/hits.jsonl')",
//!     [], |r| r.get(0))?;
//! println!("blended yield = {yield_:?}");
//! # Ok(())
//! # }
//! ```

/// IX algorithms as DuckDB UDFs. Public + gated on `udf` (a subset of `duck`)
/// so the loadable C-API extension crate can call [`udf::register_all`] without
/// pulling a bundled engine.
#[cfg(feature = "udf")]
pub mod udf;

/// IX algorithms as DuckDB *table* functions (ix_pca_project, ix_silhouette),
/// registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod tablefn;

/// Music set-theory scalar UDFs over pitch-class sets (ix_forte_number, ix_icv,
/// ix_prime_form, ix_classify_triad), registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod bracelet;

/// Twelve-tone serialism scalar UDFs over ordered rows (ix_row_retrograde,
/// ix_row_invert, ix_row_transpose, ix_row_matrix, ix_row_combinatoriality),
/// registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod serial;

/// Graph (ix_pagerank, ix_shortest_path) + signal (ix_rfft, ix_autocorrelation)
/// table functions over JSON edge-lists / series, registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod graphsig;

/// Grothendieck-style operations: ICV group delta / nearby / harmonic path over
/// PC-sets (ix_grothendieck_delta, ix_icv_l1, ix_z_related, ix_grothendieck_nearby,
/// ix_grothendieck_path) + graph K-theory (ix_k0, ix_k1). Registered by
/// [`udf::register_all`]. Pure wraps of ix-bracelet + ix-ktheory.
#[cfg(feature = "udf")]
mod grothendieck;

/// ML-evaluation UDFs — ranking metrics (ix_ndcg, …), ix_classification_report,
/// ix_knn_leakage — registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod eval;

/// Probabilistic data-structure UDFs — Bloom / HyperLogLog / Count-Min / Cuckoo
/// sketches as portable blobs (ix_bloom_*, ix_hll_*, ix_cms_*, ix_cuckoo_*),
/// registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod sketch;

/// Code-analysis UDFs over `ix-code` — complexity/metrics/smells (Tier A) and,
/// under the `code-semantic` feature, tree-sitter AST queries + semantic metrics
/// (ix_code_*, ix_ast_query, ix_semantic_metrics). Registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod code;

/// Statistical-inference UDFs over `ix-math::inference` — distribution-shape
/// scalars (ix_skewness/ix_kurtosis/ix_mad/ix_quantile/ix_entropy/ix_kl/ix_js)
/// and the two-sample test table fn ix_two_sample. Registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod inference;

/// Supervised fit/predict UDFs over `ix-supervised` — `ix_linreg_fit`/`_predict`
/// and `ix_logistic_fit`/`_predict`, models round-tripped as JSON `VARCHAR` blobs.
/// Registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod supervised;

/// Epistemic SQL — hexavalent-logic scalar UDFs over `ix-types::Hexavalent`
/// (`ix_hex_consensus`). Registered by [`udf::register_all`].
#[cfg(feature = "udf")]
mod hexavalent;

/// Artifact source — the deep module every lens reads through: safe materialization of
/// a GA-emitted JSON artifact set into a bench table (file selection + read_json_auto
/// flags + the json_extract projection + empty-fallback). See `CONTEXT.md` → "Artifact source".
#[cfg(feature = "duck")]
pub mod source;

/// Chatbot flight recorder — GA golden-trace warehouse (Slice A) + canonical-diff
/// regression gate (Slice B). See `docs/plans/2026-06-14-004-…-flight-recorder-plan.md`.
#[cfg(feature = "duck")]
pub mod chatbot;

/// Routing-quality trend lens over GA's `routing-eval-*.json` — per-intent F1
/// trend, weakest intents, and run-over-run regressions.
#[cfg(feature = "duck")]
pub mod routing;

/// AFK / self-improvement loop-iteration lens over GA's
/// `state/quality/loops/*.iterations.jsonl` — failure-signature clustering
/// (recurring worst-items, oscillating loops, artifact churn, convergence).
#[cfg(feature = "duck")]
pub mod loops;

/// Out-of-domain query lens over GA chatbot query embeddings (Contract B, proposed):
/// mean top-k cosine to nearest neighbours flags out-of-domain queries.
#[cfg(feature = "duck")]
pub mod ood;

/// `maintain-gate` — fail-closed RSI evaluation oracle fusing the maintain lenses
/// into one hexavalent verdict (metric↑ ∧ guardrail held ∧ …), never an average.
#[cfg(feature = "duck")]
pub mod maintain;

/// Pipeline mesh — compose N IX pipelines over many streams and correlate them
/// (smooth → ix_pearson → ix_connected_components → ix_centrality). The substrate
/// decided in `docs/adr/0004-duckdb-sql-pipeline-mesh.md`.
#[cfg(feature = "duck")]
pub mod mesh;

#[cfg(feature = "duck")]
pub use duckdb::{Connection, Result};

/// Open an in-memory DuckDB connection with every IX UDF registered.
///
/// In-memory only — the bench holds no state across calls. Point queries at the
/// on-disk telemetry via `read_json_auto('…')` / `read_parquet('…')`.
// @ai:invariant open_bench returns an in-memory DuckDB connection on which every IX UDF is registered and callable from SQL [T:test conf:0.9 src:ix_duck::tests::open_bench_reads_jsonl_and_registers_udfs]
#[cfg(feature = "duck")]
pub fn open_bench() -> duckdb::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    udf::register_all(&conn)?;
    Ok(conn)
}

/// Open an existing on-disk DuckDB **read-only** (e.g. the materialized
/// `state/quality/analytics/quality.duckdb`). The analyst lens never mutates the
/// analytics DB — `build-views.sql` owns all writes.
// @ai:invariant open_readonly opens an existing DuckDB file with no write access [T:manual conf:0.8]
#[cfg(feature = "duck")]
pub fn open_readonly(path: &str) -> duckdb::Result<Connection> {
    use duckdb::{AccessMode, Config};
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    Connection::open_with_flags(path, config)
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a tiny JSONL fixture and return its DuckDB-safe path string.
    fn fixture(dir: &std::path::Path) -> String {
        let p = dir.join("hits.jsonl");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, r#"{{"outcome":"compiled","coverage_max":0.5}}"#).unwrap();
        writeln!(f, r#"{{"outcome":"compiled","coverage_max":0.7}}"#).unwrap();
        writeln!(f, r#"{{"outcome":"out_of_domain","coverage_max":0.1}}"#).unwrap();
        p.to_string_lossy().replace('\\', "/")
    }

    #[test]
    fn open_bench_reads_jsonl_and_registers_udfs() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path());
        let conn = open_bench().unwrap();

        let n: i64 = conn
            .query_row(
                &format!("SELECT count(*) FROM read_json_auto('{path}')"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 3, "read_json_auto should see all 3 rows");

        let avg: f64 = conn
            .query_row(
                &format!("SELECT avg(coverage_max) FROM read_json_auto('{path}') WHERE outcome='compiled'"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            (avg - 0.6).abs() < 1e-9,
            "compiled yield should be 0.6, got {avg}"
        );

        // UDFs are registered + callable from SQL.
        let sim: f64 = conn
            .query_row(
                "SELECT ix_cosine([1.0, 0.0]::DOUBLE[], [1.0, 0.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "ix_cosine should be registered, got {sim}"
        );
    }

    #[test]
    fn ix_cosine_matches_ix_math() {
        let conn = open_bench().unwrap();
        let same: f64 = conn
            .query_row(
                "SELECT ix_cosine([1.0,2.0,3.0]::DOUBLE[], [1.0,2.0,3.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((same - 1.0).abs() < 1e-12, "identical → 1.0, got {same}");

        let orth: f64 = conn
            .query_row(
                "SELECT ix_cosine([1.0,0.0]::DOUBLE[], [0.0,1.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(orth.abs() < 1e-12, "orthogonal → 0.0, got {orth}");

        // Dimension mismatch surfaces as a SQL error, not a panic.
        let err = conn.query_row(
            "SELECT ix_cosine([1.0,0.0]::DOUBLE[], [1.0]::DOUBLE[])",
            [],
            |r| r.get::<_, f64>(0),
        );
        assert!(err.is_err(), "dimension mismatch should be a SQL error");
    }

    #[test]
    fn ix_euclidean_matches_ix_math() {
        let conn = open_bench().unwrap();
        let d: f64 = conn
            .query_row(
                "SELECT ix_euclidean([0.0,0.0]::DOUBLE[], [3.0,4.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((d - 5.0).abs() < 1e-12, "3-4-5 triangle → 5.0, got {d}");

        let z: f64 = conn
            .query_row(
                "SELECT ix_euclidean([1.0,2.0]::DOUBLE[], [1.0,2.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(z.abs() < 1e-12, "equal vectors → 0.0, got {z}");
    }

    #[test]
    fn ix_manhattan_matches_ix_math() {
        let conn = open_bench().unwrap();
        let d: f64 = conn
            .query_row(
                "SELECT ix_manhattan([0.0,0.0]::DOUBLE[], [3.0,4.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((d - 7.0).abs() < 1e-12, "|3|+|4| → 7.0, got {d}");

        // Dimension mismatch surfaces as a SQL error, not a panic.
        let err = conn.query_row(
            "SELECT ix_manhattan([1.0,0.0]::DOUBLE[], [1.0]::DOUBLE[])",
            [],
            |r| r.get::<_, f64>(0),
        );
        assert!(err.is_err(), "dimension mismatch should be a SQL error");
    }

    #[test]
    fn ix_chebyshev_matches_ix_math() {
        let conn = open_bench().unwrap();
        let d: f64 = conn
            .query_row(
                "SELECT ix_chebyshev([0.0,0.0]::DOUBLE[], [3.0,4.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((d - 4.0).abs() < 1e-12, "max(|3|,|4|) → 4.0, got {d}");
    }

    #[test]
    fn ix_cosine_distance_matches_ix_math() {
        let conn = open_bench().unwrap();
        // identical → 0.0
        let same: f64 = conn
            .query_row(
                "SELECT ix_cosine_distance([1.0,2.0,3.0]::DOUBLE[], [1.0,2.0,3.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(same.abs() < 1e-12, "identical → 0.0, got {same}");

        // orthogonal → 1.0
        let orth: f64 = conn
            .query_row(
                "SELECT ix_cosine_distance([1.0,0.0]::DOUBLE[], [0.0,1.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((orth - 1.0).abs() < 1e-12, "orthogonal → 1.0, got {orth}");
    }

    #[test]
    fn ix_minkowski_matches_ix_math() {
        let conn = open_bench().unwrap();
        // p=1 → manhattan (|3|+|4| = 7)
        let m1: f64 = conn
            .query_row(
                "SELECT ix_minkowski([0.0,0.0]::DOUBLE[], [3.0,4.0]::DOUBLE[], 1.0)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((m1 - 7.0).abs() < 1e-12, "p=1 → manhattan 7.0, got {m1}");

        // p=2 → euclidean (3-4-5 → 5)
        let m2: f64 = conn
            .query_row(
                "SELECT ix_minkowski([0.0,0.0]::DOUBLE[], [3.0,4.0]::DOUBLE[], 2.0)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((m2 - 5.0).abs() < 1e-12, "p=2 → euclidean 5.0, got {m2}");

        // Dimension mismatch surfaces as a SQL error, not a panic.
        let err = conn.query_row(
            "SELECT ix_minkowski([1.0,0.0]::DOUBLE[], [1.0]::DOUBLE[], 2.0)",
            [],
            |r| r.get::<_, f64>(0),
        );
        assert!(err.is_err(), "dimension mismatch should be a SQL error");
    }

    #[test]
    fn ix_cosine_passes_null_through() {
        let conn = open_bench().unwrap();
        // A NULL list argument yields SQL NULL, not a spurious number.
        let n: Option<f64> = conn
            .query_row(
                "SELECT ix_cosine(NULL::DOUBLE[], [1.0,2.0]::DOUBLE[])",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, None, "NULL arg → SQL NULL");

        // In a mixed chunk, non-NULL rows still compute while the NULL row stays NULL.
        let mut stmt = conn
            .prepare(
                "SELECT ix_cosine(a, b) FROM (VALUES \
                   (1, [1.0,2.0,3.0]::DOUBLE[], [1.0,2.0,3.0]::DOUBLE[]), \
                   (2, NULL::DOUBLE[],          [1.0,2.0,3.0]::DOUBLE[]), \
                   (3, [1.0,0.0]::DOUBLE[],     [0.0,1.0]::DOUBLE[]) \
                 ) t(i,a,b) ORDER BY i",
            )
            .unwrap();
        let rows: Vec<Option<f64>> = stmt
            .query_map([], |r| r.get::<_, Option<f64>>(0))
            .unwrap()
            .map(|x| x.unwrap())
            .collect();
        assert_eq!(rows.len(), 3);
        assert!((rows[0].unwrap() - 1.0).abs() < 1e-12, "identical → 1.0");
        assert_eq!(rows[1], None, "NULL row stays NULL in a mixed chunk");
        assert!(rows[2].unwrap().abs() < 1e-12, "orthogonal → 0.0");
    }
}
