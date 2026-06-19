//! `ix_ood_lens` — out-of-domain query lens over GA's `query-embeddings/*.jsonl`
//! (Contract B). Thin shell over `ix_duck::ood`: build the query-embedding table, score
//! each query by mean top-k cosine to its nearest neighbours (leave-one-out), and surface
//! the most out-of-domain queries. Also **times the O(n²) all-pairs scoring** so we can
//! decide empirically (not by guessing) whether brute force is a problem at real scale.
//!
//!   cargo run -p ix-duck --features duck --example ix_ood_lens            # live ../ga
//!   cargo run -p ix-duck --features duck --example ix_ood_lens -- <dir>   # explicit query-embeddings dir

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use ix_duck::ood;

fn default_dir() -> PathBuf {
    if let Ok(root) = std::env::var("GA_ROOT") {
        return PathBuf::from(root).join("state/quality/query-embeddings");
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ix_root = cwd.canonicalize().unwrap_or(cwd);
    ix_root
        .parent()
        .map(|p| p.join("ga/state/quality/query-embeddings"))
        .unwrap_or_else(|| PathBuf::from("../ga/state/quality/query-embeddings"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(default_dir);
    if dir.to_string_lossy().contains("://") {
        eprintln!("refusing non-local path: {}", dir.display());
        std::process::exit(2);
    }

    let conn = ix_duck::open_bench()?;
    println!("query-embeddings dir: {}\n", dir.display());

    let n = ood::build_query_embeddings(&conn, &dir)?;
    if n < 2 {
        println!("(only {n} query row(s) — need ≥2 to score against neighbours)");
        return Ok(());
    }

    // query_id → query_text, for readable output.
    let mut text: HashMap<String, String> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT query_id, query_text FROM query_embeddings")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, t) = row?;
            text.insert(id, t);
        }
    }

    // Time the exact all-pairs cosine scoring on the REAL corpus. `ood_scores` dedups by
    // embedding first, so the work is over DISTINCT embeddings (= scores.len()), not raw n.
    let k = 3;
    let t0 = Instant::now();
    let scores = ood::ood_scores(&conn, k)?;
    let elapsed = t0.elapsed();

    let distinct = scores.len();
    let pairs = distinct.saturating_mul(distinct.saturating_sub(1));
    println!(
        "rows = {n} raw  |  {distinct} distinct embeddings ({:.1}× replay)  |  dim 1024, k = {k}",
        n as f64 / distinct.max(1) as f64
    );
    println!(
        "exact all-pairs cosine over the distinct set: {pairs} pairs in {:.1} ms\n",
        elapsed.as_secs_f64() * 1e3
    );

    println!("most out-of-domain (lowest mean top-{k} cosine):");
    println!("  {:>6}  {:<26}  query", "score", "intent");
    for (id, intent, score) in scores.iter().take(15) {
        let q = text.get(id).map(|s| s.as_str()).unwrap_or("");
        let q = if q.chars().count() > 60 {
            format!("{}…", q.chars().take(59).collect::<String>())
        } else {
            q.to_string()
        };
        println!("  {score:>6.3}  {intent:<26}  {q}");
    }

    for thr in [0.3_f64, 0.5] {
        let flagged = ood::flag_ood(&conn, k, thr)?;
        println!(
            "\nflagged OOD at threshold {thr}: {} / {distinct} distinct",
            flagged.len()
        );
    }
    Ok(())
}
