//! `ix_voicing_lens` — Tier-2 analyst's bench over GA's voicing artifacts.
//!
//! Two halves, matching the `ga-voicing-analysis` contract
//! (`docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md`):
//!
//!   1. SEARCH TELEMETRY (available today) — GA already emits per-query JSONL at
//!      `state/telemetry/voicing-search/*.jsonl`. This lens runs coverage-gap and
//!      latency queries over it directly (no export needed).
//!   2. EMBEDDINGS + METADATA (Tier-2, pending GA export) — the production
//!      `optick.index` is a binary mmap, NOT DuckDB-readable. The contract asks GA
//!      to export an analyzable Parquet slice; when that file exists this lens reads
//!      it and the IX vector UDFs (`ix_pca_project`, `ix_kdist`) compose over it.
//!      Until then this half degrades to a one-line hint — never an error.
//!
//! Run (resolves GA via `GA_ROOT` env or the sibling `../ga`):
//!   cargo run -p ix-duck --features duck --example ix_voicing_lens
//!
//! Hermetic by design: absent GA artifacts → graceful skip, exit 0.

use std::path::PathBuf;

fn ga_root() -> PathBuf {
    std::env::var("GA_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("../ga"))
}

/// Run a query and print rows schema-agnostically (every column cast to VARCHAR).
fn print_query(conn: &duckdb::Connection, label: &str, query: &str) -> duckdb::Result<()> {
    println!("\n── {label} ──");
    let wrapped = format!("SELECT CAST(COLUMNS(*) AS VARCHAR) FROM ({query}) AS _q");
    let mut stmt = conn.prepare(&wrapped)?;
    let mut rows = stmt.query([])?;
    let cols: Vec<String> = rows.as_ref().map(|s| s.column_names()).unwrap_or_default();
    println!("{}", cols.join(" | "));
    let mut n = 0usize;
    while let Some(row) = rows.next()? {
        let cells: Vec<String> = (0..cols.len())
            .map(|i| row.get::<_, Option<String>>(i).ok().flatten().unwrap_or_else(|| "NULL".into()))
            .collect();
        println!("{}", cells.join(" | "));
        n += 1;
    }
    println!("({n} row(s))");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ga = ga_root();
    let conn = ix_duck::open_bench()?; // in-memory, ix UDFs registered

    // ── 1. Search telemetry (runs on real data today) ──────────────────────────
    let tele_glob = ga.join("state/telemetry/voicing-search").join("*.jsonl");
    let tele = tele_glob.to_string_lossy().replace('\\', "/");
    let has_tele = glob_has_match(&ga.join("state/telemetry/voicing-search"), "jsonl");
    if has_tele {
        println!("Voicing search telemetry: {tele}");
        let src = format!("read_json_auto('{tele}', union_by_name=true)");
        print_query(
            &conn,
            "Coverage gaps — empty-result rate by instrument",
            &format!(
                "SELECT instr, count(*) AS queries, sum(CAST(empty AS INT)) AS empties, \
                 round(100.0*sum(CAST(empty AS INT))/count(*), 1) AS empty_pct \
                 FROM {src} GROUP BY instr ORDER BY empty_pct DESC"
            ),
        )?;
        print_query(
            &conn,
            "Latency (non-empty queries) p50 / p95 ms by instrument",
            &format!(
                "SELECT instr, round(quantile_cont(ms, 0.5), 1) AS p50_ms, \
                 round(quantile_cont(ms, 0.95), 1) AS p95_ms FROM {src} \
                 WHERE NOT empty GROUP BY instr ORDER BY p95_ms DESC"
            ),
        )?;
        print_query(
            &conn,
            "Top coverage gaps — most-repeated zero-result queries",
            &format!(
                "SELECT q, any_value(chord) AS chord, count(*) AS times FROM {src} \
                 WHERE empty GROUP BY q ORDER BY times DESC LIMIT 10"
            ),
        )?;
    } else {
        println!("No voicing-search telemetry under {} — skipping telemetry lens.", ga.display());
    }

    // ── 2. Embeddings + metadata (Tier-2 Parquet, pending GA export) ────────────
    let parquet = ga.join("state/voicings/analysis/voicings.parquet");
    if parquet.exists() {
        let p = parquet.to_string_lossy().replace('\\', "/");
        println!("\nVoicing embeddings slice: {p}");
        print_query(
            &conn,
            "Embeddings slice — row count + dim",
            &format!("SELECT count(*) AS voicings, any_value(len(embedding)) AS dim FROM read_parquet('{p}')"),
        )?;
        println!(
            "  → compose vector UDFs over `embedding`: ix_pca_project / ix_kdist \
             (feed a JSON 2-D array of embeddings)."
        );
    } else {
        println!(
            "\nEmbeddings slice not found at {} — pending GA export per \
             docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md \
             (optick.index is a binary mmap, not DuckDB-readable).",
            parquet.display()
        );
    }

    Ok(())
}

/// Does `dir` contain at least one file with the given extension?
fn glob_has_match(dir: &std::path::Path, ext: &str) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == ext))
        })
        .unwrap_or(false)
}
