//! `ix_grothendieck_lens` — set-theory pattern discovery over GA's voicing corpus
//! using the Grothendieck UDFs (PR: ix_grothendieck_delta / ix_icv_l1 / ix_forte_number).
//!
//! The tracer-bullet for "should we crunch a massive all-instruments/tunings corpus?".
//! It runs the *realization-map* analysis — which set-classes are realized, how densely,
//! and where IX and GA disagree — on whatever voicing JSONL is available: the full export
//! `GA_ROOT/state/voicings/analysis/voicings.jsonl`, else the `GA_ROOT/voicing_index.jsonl`
//! sample. Absent both → graceful one-line skip, exit 0 (the optick.index is a binary mmap,
//! not DuckDB-readable; see docs/contracts/2026-06-16-ga-voicing-analysis-parquet).
//!
//! Run:  cargo run -p ix-duck --features duck --example ix_grothendieck_lens

use std::path::PathBuf;

fn ga_root() -> PathBuf {
    std::env::var("GA_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("../ga"))
}

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
    let full = ga.join("state/voicings/analysis/voicings.jsonl");
    let sample = ga.join("voicing_index.jsonl");
    let corpus = if full.exists() {
        full
    } else if sample.exists() {
        sample
    } else {
        println!(
            "No DuckDB-readable voicing corpus under {} (optick.index is a binary mmap). \
             Skipping — pending GA export per docs/contracts/2026-06-16-ga-voicing-analysis-parquet.",
            ga.display()
        );
        return Ok(());
    };
    let p = corpus.to_string_lossy().replace('\\', "/");
    println!("Voicing corpus: {p}");

    let conn = ix_duck::open_bench()?;
    // Materialize once; cast MidiNotes to BIGINT[] so the set-theory UDFs bind regardless
    // of how read_json_auto infers the integer array.
    conn.execute_batch(&format!(
        "CREATE TABLE v AS SELECT ChordName, Diagram, TuningId, ForteCode, \
           IntervalClassVector AS ga_icv, CAST(MidiNotes AS BIGINT[]) AS notes \
         FROM read_json_auto('{p}', union_by_name=true, sample_size=-1) \
         WHERE MidiNotes IS NOT NULL;"
    ))?;

    print_query(
        &conn,
        "Corpus shape",
        "SELECT count(*) AS voicings, count(DISTINCT TuningId) AS tunings, \
         count(DISTINCT ChordName) AS chords, min(len(notes)) AS min_card, max(len(notes)) AS max_card FROM v",
    )?;

    print_query(
        &conn,
        "Set-class realization density — most-realized set classes (IX canonical Forte)",
        "SELECT ix_forte_number(notes) AS set_class, ix_icv(notes) AS icv, count(*) AS voicings \
         FROM v GROUP BY 1, 2 ORDER BY voicings DESC LIMIT 12",
    )?;

    // The headline IX↔GA cross-check, quantified on real data: ICV agrees, Forte diverges.
    // Normalize GA's space-separated ICV "<0 0 0 0 1 0>" to IX's comma form "<0,0,0,0,1,0>".
    print_query(
        &conn,
        "IX↔GA agreement on real voicings — ICV vs Forte number",
        "SELECT \
           round(100.0*avg(CASE WHEN ix_icv(notes) = replace(ga_icv,' ',',') THEN 1 ELSE 0 END), 1) AS icv_agree_pct, \
           round(100.0*avg(CASE WHEN ix_forte_number(notes) = ForteCode THEN 1 ELSE 0 END), 1) AS forte_agree_pct \
         FROM v WHERE ga_icv IS NOT NULL AND ForteCode IS NOT NULL",
    )?;

    print_query(
        &conn,
        "Sample IX↔GA Forte divergences (same chord, different numbering)",
        "SELECT DISTINCT ChordName, ix_icv(notes) AS icv, ForteCode AS ga_forte, \
           ix_forte_number(notes) AS ix_forte \
         FROM v WHERE ForteCode IS NOT NULL AND ix_forte_number(notes) IS DISTINCT FROM ForteCode \
         ORDER BY ChordName LIMIT 8",
    )?;

    print_query(
        &conn,
        "Voice-leading: voicings closest to Cmaj7 {0,4,7,11} by Grothendieck L1 cost",
        "SELECT ChordName, Diagram, ix_grothendieck_delta(notes, [0,4,7,11]) AS delta, \
           ix_icv_l1(notes, [0,4,7,11]) AS cost \
         FROM v WHERE len(notes) >= 3 ORDER BY cost, ChordName LIMIT 10",
    )?;

    println!(
        "\nNote: this corpus has {} — the *type* space is closed (224 set-classes); a bigger \
         all-tunings corpus grows physical realizations, not chord types. The blocker to \
         crunching the full 313k is the pending GA parquet/jsonl export, not IX.",
        if p.ends_with("voicing_index.jsonl") { "1000 sampled rows" } else { "the exported slice" }
    );
    Ok(())
}
