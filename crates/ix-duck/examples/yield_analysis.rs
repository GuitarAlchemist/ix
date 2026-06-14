//! Reproduce the thinking-machine yield-split over `hits.jsonl` through ix-duck's
//! in-process DuckDB — the MVP value proof. Needs **no** custom UDF (pure SQL).
//!
//! Kills the documented yield-split footgun (a blended cumulative mean hides the
//! pre/post split). See docs/DUCKDB.md.
//!
//! Run:
//!   cargo run -p ix-duck --features duck --example yield_analysis
//!   cargo run -p ix-duck --features duck --example yield_analysis -- path/to/hits.jsonl

use ix_duck::open_bench;

fn main() -> duckdb::Result<()> {
    let hits = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "state/thinking-machine/hits.jsonl".to_string());

    let conn = open_bench()?;

    println!("yield by outcome ({hits}):");
    let mut stmt = conn.prepare(&format!(
        "SELECT outcome, count(*) AS n, round(avg(coverage_max), 4) AS yield
         FROM read_json_auto('{hits}')
         GROUP BY outcome
         ORDER BY n DESC"
    ))?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, Option<String>>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, Option<f64>>(2)?,
        ))
    })?;
    for row in rows {
        let (outcome, n, yld) = row?;
        println!(
            "  {:<20} n={:<4} yield={}",
            outcome.unwrap_or_else(|| "<null>".into()),
            n,
            yld.map(|v| format!("{v:.4}")).unwrap_or_else(|| "<null>".into())
        );
    }

    let blended: Option<f64> = conn.query_row(
        &format!("SELECT round(avg(coverage_max), 4) FROM read_json_auto('{hits}') WHERE ts_ms > 0"),
        [],
        |r| r.get(0),
    )?;
    println!("\nblended yield (the footgun number) = {blended:?}");

    Ok(())
}
