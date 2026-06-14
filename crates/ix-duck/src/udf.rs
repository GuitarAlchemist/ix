//! IX algorithms exposed as DuckDB scalar UDFs (via the `VScalar` trait).
//!
//! Phase 2 registers `ix_cosine` and `ix_euclidean` here, each wrapping the real
//! `ix-math::distance` functions (no reimplementation). Set-relative operations
//! (`ix_pca_project`, `ix_silhouette`) are table functions, deferred to a follow-up
//! plan — see docs/plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md.

use duckdb::Connection;

/// Register every IX UDF on `conn`.
pub fn register_all(_conn: &Connection) -> duckdb::Result<()> {
    // Phase 2: register_scalar_function::<IxCosine>("ix_cosine") + ix_euclidean.
    Ok(())
}
