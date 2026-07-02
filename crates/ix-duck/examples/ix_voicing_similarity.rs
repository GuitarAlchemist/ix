//! `ix_voicing_similarity` — harmonic nearest-neighbours over the GA voicing corpus,
//! with a built-in correctness oracle.
//!
//! Question: *given a chord's set-class, what are its harmonically-nearest neighbours by
//! interval content — and which set-classes in the real repertoire are harmonically
//! isolated (no close neighbour)?*
//!
//! The metric is `ix_icv_l1(a, b)` — the L1 distance between two PC-sets' **interval-class
//! vectors** (the Grothendieck harmonic cost). Each set-class present in the corpus is one
//! point; a self-join computes the full N×N distance table on the DuckDB bench.
//!
//! **Validation (the analog of the mesh's null model, matched to a *retrieval* demo).**
//! By music theory, two *distinct* set-classes share an identical ICV iff they are
//! **Z-related** — so every pair at harmonic distance 0 must satisfy `ix_z_related`. The
//! demo asserts exactly this against the corpus: if any distance-0 pair were *not*
//! Z-related, the metric would be unfaithful. The Z-relation is a ground-truth oracle,
//! not a significance test — the right validation for a similarity/retrieval claim.
//!
//! Run: `cargo run -p ix-duck --example ix_voicing_similarity --features duck`

use std::collections::BTreeMap;

use ix_duck::open_bench;

const CORPUS_FULL: &str = "state/voicings/raw/guitar.jsonl";
const CORPUS_SAMPLE: &str = "state/voicings/guitar-corpus.json";

/// Set-classes to showcase nearest-neighbours for (skipped if absent from the corpus).
/// 4-Z15 is included on purpose: its nearest neighbour is its Z-partner 4-Z29 at
/// distance 0 — the Z-relation made visible.
const QUERIES: [&str; 4] = ["3-11", "4-27", "4-Z15", "5-35"];
const K: usize = 5;

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;
    let (corpus, full) = if std::path::Path::new(CORPUS_FULL).exists() {
        (CORPUS_FULL, true)
    } else {
        (CORPUS_SAMPLE, false)
    };
    if !full {
        println!("note: full corpus absent (gitignored, 110 MB) — using the tracked sample.\n");
    }
    // Rank isolation only among set-classes common enough to be "really used" — scaled
    // to the corpus so the sample still surfaces something.
    let isolation_min_freq: i64 = if full { 500 } else { 5 };

    // Each set-class present in the corpus → a representative PC-set (any voicing's
    // midiNotes; ICV is a set-class invariant) + its voicing frequency.
    // The self-join then computes the full harmonic-distance table + Z-relation flags.
    let sql = format!(
        "WITH sc AS (
            SELECT ix_forte_number(midiNotes) AS forte,
                   any_value(midiNotes)        AS rep,
                   count(*)                     AS freq
            FROM read_json_auto('{corpus}')
            WHERE ix_forte_number(midiNotes) IS NOT NULL
            GROUP BY ix_forte_number(midiNotes)
         )
         SELECT a.forte, a.freq, b.forte, ix_icv_l1(a.rep, b.rep), ix_z_related(a.rep, b.rep)
         FROM sc a JOIN sc b ON a.forte <> b.forte"
    );

    let mut freq: BTreeMap<String, i64> = BTreeMap::new();
    // dist[(a,b)] for a<b (symmetric); pairs with dist==0 carry their z_related flag.
    let mut neighbours: BTreeMap<String, Vec<(i64, String)>> = BTreeMap::new();
    let mut zero_pairs: Vec<(String, String, bool)> = Vec::new();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, bool>(4)?,
        ))
    })?;
    for row in rows {
        let (a, fa, b, dist, zrel) = row?;
        freq.insert(a.clone(), fa);
        neighbours.entry(a.clone()).or_default().push((dist, b.clone()));
        if dist == 0 && a < b {
            zero_pairs.push((a, b, zrel));
        }
    }
    for v in neighbours.values_mut() {
        v.sort();
    }

    println!("corpus: {corpus} → {} distinct set-classes\n", freq.len());

    // ── harmonic nearest-neighbours for the showcase queries ─────────────────────
    println!("harmonic nearest-neighbours (ix_icv_l1 = L1 on the interval-class vector):");
    for q in QUERIES {
        let Some(nbrs) = neighbours.get(q) else {
            println!("   {q:>6}: (absent from this corpus)");
            continue;
        };
        let shown: Vec<String> = nbrs
            .iter()
            .take(K)
            .map(|(d, name)| format!("{name} (d={d})"))
            .collect();
        println!("   {q:>6} → {}", shown.join(", "));
    }

    // ── validation oracle: every distance-0 pair must be Z-related ────────────────
    let violations: Vec<&(String, String, bool)> = zero_pairs.iter().filter(|(_, _, z)| !z).collect();
    println!(
        "\nvalidation — {} distinct-set-class pairs at harmonic distance 0:",
        zero_pairs.len()
    );
    for (a, b, _) in zero_pairs.iter().take(6) {
        println!("   {a} ≡ {b}  (identical ICV — Z-related)");
    }
    if violations.is_empty() {
        println!("   ✅ ORACLE PASS: every distance-0 pair is ix_z_related (the metric respects the Z-relation ground truth)");
    } else {
        println!("   ❌ ORACLE FAIL: {} distance-0 pair(s) are NOT Z-related — the metric is unfaithful:", violations.len());
        for (a, b, _) in &violations {
            println!("      {a} ≡ {b} but ix_z_related = false");
        }
    }

    // ── harmonic isolation: set-classes whose nearest neighbour is farthest ───────
    let mut isolation: Vec<(String, i64, i64)> = neighbours
        .iter()
        .filter(|(name, _)| freq.get(*name).copied().unwrap_or(0) >= isolation_min_freq)
        .map(|(name, nbrs)| {
            let nn = nbrs.first().map(|(d, _)| *d).unwrap_or(i64::MAX);
            (name.clone(), nn, freq[name])
        })
        .collect();
    isolation.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
    println!(
        "\nmost harmonically-isolated set-classes (≥{isolation_min_freq} voicings; nn = nearest-neighbour distance):"
    );
    for (name, nn, f) in isolation.iter().take(8) {
        println!("   {name:>6}: nn = {nn}  ({f} voicings)");
    }

    Ok(())
}
