//! `ix_voicing_mesh_nullcheck` — the reproducible **null model** behind the validation
//! section of `docs/walkthroughs/voicing-mesh.md`.
//!
//! A betweenness "hub" from [`ix_voicing_mesh`] is only a claim if the real data's
//! hub-concentration exceeds what the pipeline manufactures from structureless input.
//! This builds the real per-axis mesh, then **K null meshes** by shuffling each bin's
//! counts across set-classes — a permutation that *keeps* the common-mode trend (low
//! frets / wide spans crowded) but *destroys* per-set-class co-variation. The same
//! pipeline (normalize → common-mode removal → `ix_pearson` → τ → betweenness) runs on
//! real and null alike; `p` is the fraction of nulls whose top betweenness ≥ the real's.
//!
//! Faithful to the demo: betweenness is `ix_graph::Graph::betweenness_centrality` (what
//! `ix_centrality` calls) and correlation is `ix_math::inference::pearson` (what
//! `ix_pearson` wraps). The only omission is the wavelet-smoothing stage — a denoiser
//! applied equally to real and null, so the real-vs-null comparison is unaffected.
//!
//! Run: `cargo run -p ix-duck --example ix_voicing_mesh_nullcheck --features duck`

use std::collections::BTreeMap;

use ix_duck::open_bench;
use ix_graph::graph::Graph;
use ix_math::inference::pearson;

const TAU: f64 = 0.8;
const NULLS: usize = 500;
/// Fixed RNG seed so the null distribution (and `p`) is reproducible run to run.
const SEED: u64 = 0x5715_3D2A_91C7_0E4B;

const CORPUS_FULL: &str = "state/voicings/raw/guitar.jsonl";
const CORPUS_SAMPLE: &str = "state/voicings/guitar-corpus.json";

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;
    let (corpus, min_support, top_k, full) = if std::path::Path::new(CORPUS_FULL).exists() {
        (CORPUS_FULL, 1000.0_f64, 120usize, true)
    } else {
        (CORPUS_SAMPLE, 2.0, 24, false)
    };
    if !full {
        println!("note: full corpus absent — using the tracked sample; verdicts need the full corpus.\n");
    }
    println!("null model: {NULLS} per-bin shuffles, τ = {TAU}, betweenness = ix_graph (same as the demo)\n");

    for (axis, col, bins) in [("position", "minFret", 18usize), ("stretch", "fretSpan", 5usize)] {
        let mat = load_matrix(&conn, corpus, col, bins, min_support, top_k)?;
        let (names, counts) = mat;
        let real = top_betweenness(&counts);

        // Null distribution: per-column shuffle of the count matrix.
        let mut rng = Xorshift::new(SEED ^ (axis.len() as u64));
        let mut nulls: Vec<f64> = Vec::with_capacity(NULLS);
        for _ in 0..NULLS {
            nulls.push(top_betweenness(&shuffle_columns(&counts, &mut rng)).score);
        }
        nulls.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ge = nulls.iter().filter(|&&x| x >= real.score).count();
        let p = (1 + ge) as f64 / (NULLS + 1) as f64;
        let mean = nulls.iter().sum::<f64>() / nulls.len() as f64;
        let p95 = nulls[(0.95 * nulls.len() as f64) as usize];
        let verdict = if p < 0.05 { "SIGNAL (real > null)" } else { "ARTIFACT (within null range)" };

        println!("=== {axis} (N = {} streams, {bins} bins) ===", names.len());
        println!("  real: leader = {}  top betweenness = {:.1}", names[real.node], real.score);
        println!("  null: mean {mean:.1}  95th {p95:.1}  max {:.1}", nulls.last().copied().unwrap_or(0.0));
        println!("  p(null ≥ real) = {p:.4}  →  {verdict}\n");
    }
    Ok(())
}

struct Top {
    node: usize,
    score: f64,
}

/// The full pipeline on a count matrix: normalize → common-mode removal → |Pearson|
/// edges at τ → `ix_graph` betweenness. Returns the top (node, score).
fn top_betweenness(counts: &[Vec<f64>]) -> Top {
    let n = counts.len();
    let bins = counts[0].len();
    // normalize each row to a distribution
    let dists: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| {
            let s: f64 = row.iter().sum();
            row.iter().map(|x| if s > 0.0 { x / s } else { 0.0 }).collect()
        })
        .collect();
    // subtract the per-bin cross-set-class mean → residuals
    let mut mean = vec![0.0f64; bins];
    for d in &dists {
        for (k, p) in d.iter().enumerate() {
            mean[k] += p / n as f64;
        }
    }
    let resid: Vec<Vec<f64>> = dists
        .iter()
        .map(|d| d.iter().zip(&mean).map(|(p, m)| p - m).collect())
        .collect();
    // |Pearson| ≥ τ edges (undefined / constant residual → no edge, like the demo)
    let mut g = Graph::with_nodes(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if let Ok(r) = pearson(&resid[i], &resid[j]) {
                if r.abs() >= TAU {
                    g.add_edge(i, j, 1.0);
                }
            }
        }
    }
    let bc = g.betweenness_centrality();
    let (node, &score) = bc
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((0, &0.0));
    Top { node, score }
}

/// Per-column shuffle: independently permute each bin's values across set-classes.
/// Preserves every bin's marginal (the common-mode) while destroying row coherence.
#[allow(clippy::needless_range_loop)] // col indexes a column across rows of a row-major matrix
fn shuffle_columns(counts: &[Vec<f64>], rng: &mut Xorshift) -> Vec<Vec<f64>> {
    let (n, bins) = (counts.len(), counts[0].len());
    let mut out = counts.to_vec();
    for col in 0..bins {
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            let (a, b) = (out[i][col], out[j][col]);
            out[i][col] = b;
            out[j][col] = a;
        }
    }
    out
}

/// Load the (set-class → per-bin counts) matrix for one axis: top-`top_k` set-classes
/// with ≥ `min_support` voicings. Uses `ix_forte_number` to annotate the corpus.
fn load_matrix(
    conn: &ix_duck::Connection,
    corpus: &str,
    col: &str,
    bins: usize,
    min_support: f64,
    top_k: usize,
) -> ix_duck::Result<(Vec<String>, Vec<Vec<f64>>)> {
    let sql = format!(
        "WITH a AS (SELECT ix_forte_number(midiNotes) AS sc, {col} AS bin FROM read_json_auto('{corpus}'))
         SELECT sc, bin, count(*) FROM a WHERE sc IS NOT NULL AND bin >= 0 AND bin < {bins} GROUP BY sc, bin"
    );
    let mut profiles: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))?;
    for row in rows {
        let (sc, bin, n) = row?;
        profiles.entry(sc).or_insert_with(|| vec![0.0; bins])[bin as usize] = n as f64;
    }
    let mut ranked: Vec<(String, Vec<f64>, f64)> = profiles
        .into_iter()
        .map(|(sc, v)| { let s: f64 = v.iter().sum(); (sc, v, s) })
        .filter(|(_, _, s)| *s >= min_support)
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    ranked.truncate(top_k);
    Ok((ranked.iter().map(|r| r.0.clone()).collect(), ranked.into_iter().map(|r| r.1).collect()))
}

/// Deterministic xorshift64 — reproducible nulls without a `rand` dependency.
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Xorshift(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}
