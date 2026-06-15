//! `ix-duck-ext` — loadable DuckDB C-API extension exposing IX algorithms as SQL.
//!
//! This cdylib is the `LOAD`-able sibling of the in-process `ix-duck` bench: it
//! reuses the exact same UDF registration (`ix_duck::udf::register_all`) but ships
//! as a standalone `ix.duckdb_extension` any `duckdb.exe` can load — no embedding
//! host required.
//!
//! Built via the C-API (`C_STRUCT` ABI): DuckDB passes a struct of function
//! pointers at LOAD time rather than the extension linking a specific engine, so
//! one artifact is tolerant of engine versions >= the declared `min_duckdb_version`.
//! A 512-byte metadata footer (appended by `build.ps1` /
//! `append_extension_metadata.py`) is required before `LOAD` will accept the file.
//!
//! Usage:
//! ```text
//! duckdb -unsigned -c "LOAD 'ix.duckdb_extension';
//!   SELECT ix_cosine([1,0]::DOUBLE[], [1,0]::DOUBLE[]);   -- 1.0
//!   SELECT ix_euclidean([0,0]::DOUBLE[], [3,4]::DOUBLE[]); -- 5.0"
//! ```

use duckdb::{duckdb_entrypoint_c_api, Connection, Result};
use std::error::Error;

/// Extension entrypoint. The macro generates the C-API `ix_init_c_api` symbol
/// DuckDB calls on `LOAD`; `ext_name` must match the metadata footer's name and
/// the `ix.duckdb_extension` filename stem.
#[duckdb_entrypoint_c_api(ext_name = "ix", min_duckdb_version = "v1.0.0")]
pub fn ix_extension_init(con: Connection) -> Result<(), Box<dyn Error>> {
    // Same registration the in-process bench uses — single source of truth.
    ix_duck::udf::register_all(&con)?;
    Ok(())
}
