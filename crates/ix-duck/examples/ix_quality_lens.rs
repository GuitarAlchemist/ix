//! `ix_quality_lens` — the ix-native analog of GA's Tools/QualityLens. Opens the
//! materialized quality.duckdb **read-only** and runs a query, printing rows.
//!
//! Build the DB first (from the artifact root):
//!   duckdb analytics/quality.duckdb < analytics/build-views.sql   # in state/quality/
//!
//! Run (default DB path = state/quality/analytics/quality.duckdb, default query =
//! the quality_latest rollup):
//!   cargo run -p ix-duck --features duck --example ix_quality_lens
//!   cargo run -p ix-duck --features duck --example ix_quality_lens -- "SELECT * FROM router_eval"
//!   cargo run -p ix-duck --features duck --example ix_quality_lens -- "SELECT * FROM ix_harness" path/to/quality.duckdb

fn main() -> duckdb::Result<()> {
    let mut args = std::env::args().skip(1);
    let query = args
        .next()
        .unwrap_or_else(|| "SELECT * FROM quality_latest".to_string());
    let db_path = args
        .next()
        .unwrap_or_else(|| "state/quality/analytics/quality.duckdb".to_string());

    // Read-only: the lens never mutates the analytics DB (build-views.sql owns
    // writes). Routed through ix_duck so the crate's build-script link directives
    // (rstrtmgr on Windows) reach this example binary.
    let conn = ix_duck::open_readonly(&db_path)?;

    println!("{db_path}  «{query}»\n");

    // Wrap the user query and CAST every column to VARCHAR via DuckDB's COLUMNS(*)
    // expression, so the lens stays schema-agnostic — numbers, lists and NULLs all
    // come back as text and print uniformly regardless of which table is queried.
    let wrapped = format!("SELECT CAST(COLUMNS(*) AS VARCHAR) FROM ({query}) AS _q");
    let mut stmt = conn.prepare(&wrapped)?;

    // Execute first — this duckdb-rs binds column metadata at execution time.
    let mut rows = stmt.query([])?;
    let col_names: Vec<String> = rows
        .as_ref()
        .map(|s| s.column_names())
        .unwrap_or_default();
    let col_count = col_names.len();
    let header = col_names.join(" | ");
    println!("{header}");
    println!("{}", "-".repeat(header.len().max(8)));

    let mut n = 0usize;
    while let Some(row) = rows.next()? {
        let cells: Vec<String> = (0..col_count)
            .map(|i| row.get::<_, Option<String>>(i).ok().flatten().unwrap_or_else(|| "NULL".into()))
            .collect();
        println!("{}", cells.join(" | "));
        n += 1;
    }
    println!("\n{n} row(s)");
    Ok(())
}
