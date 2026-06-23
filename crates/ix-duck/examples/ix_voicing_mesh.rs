//! `ix_voicing_mesh` — a **real-world** pipeline-mesh (ADR-0004) over the GA guitar
//! voicing corpus (`state/voicings/raw/guitar.jsonl`, ~667 k fingerings).
//!
//! Analysis question (run for **two axes**):
//!   *Which chord set-classes behave alike along the neck, and which set-class is the
//!    structural hub of that geometry?* — once by **position** (`minFret`: where on the
//!    neck a set-class lives) and once by **stretch** (`fretSpan`: how ergonomically
//!    wide its shapes are).
//!
//! Each **stream** is one Forte set-class; the aligned axis is a histogram over the
//! chosen bin column, so a stream is that set-class's *position profile* or *stretch
//! profile*. One load-bearing pipeline stage — **common-mode removal** — makes the mesh
//! informative: every set-class crowds the low frets / wide spans, so raw profiles all
//! correlate ~1 (one clique, no hub). Normalizing each profile to a distribution and
//! subtracting the cross-set-class mean leaves each set-class's *distinctive* preference,
//! which is what the mesh correlates.
//!
//! **Both layered** (ADR-0004): the demo is a graph of ~460 *operators* (a 4-stage
//! pipeline per stream, joined through a common-mode barrier, plus a shared head and
//! mesh tail — reified as an `ix_pipeline::dag::Dag`) whose ~120 *stream* outputs feed
//! the N×N correlation mesh. Every layer is an IX UDF on the DuckDB bench:
//! ```text
//!   read_json_auto ─▶ ix_forte_number (annotate) ─▶ GROUP BY <axis> (bin)
//!     ─▶ per-set-class profile ─▶ normalize ─▶ subtract common-mode
//!     ─▶ ix_wavelet_denoise (condition) ─▶ ix_pearson (N×N)
//!     ─▶ ix_connected_components (regions) ─▶ ix_centrality betweenness (hub)
//! ```
//!
//! The `|r|` threshold τ is the operating lever (ADR-0004): a τ-sweep shows the single
//! web at τ = 0.8 fracturing into named fret-band regions as τ rises.
//!
//! **The betweenness leaders this prints are raw mesh output, not validated claims.** A
//! null model (per-bin shuffle, 1 000 nulls — see `docs/walkthroughs/voicing-mesh.md` →
//! *Validation*) shows the **position** structure is real (p = 0.001) but the **stretch**
//! leader is an artifact (real betweenness sits *below* the null median). Treat the
//! stretch axis as a negative result and any single "hub" as advisory.
//!
//! Run: `cargo run -p ix-duck --example ix_voicing_mesh --features duck`

use std::collections::BTreeMap;

use ix_duck::mesh::{correlate, MeshConfig, MeshResult};
use ix_duck::open_bench;
use ix_pipeline::dag::Dag;

/// Default `|r|` edge threshold (the mesh's headline operating point).
const THRESHOLD: f64 = 0.8;
/// τ values swept to show region structure emerging as the threshold tightens.
const TAU_SWEEP: [f64; 4] = [0.8, 0.9, 0.95, 0.98];
/// Print the full N×N matrix only for a small (tracer-bullet) mesh.
const FULL_MATRIX_MAX: usize = 12;

/// The full GA CLI dump (~667 k fingerings) — gitignored (110 MB), so present only on
/// a machine that has run `ix-voicings`. Yields the headline 100+ stream run.
const CORPUS_FULL: &str = "state/voicings/raw/guitar.jsonl";
/// A tracked 500-voicing sample (same schema) so the demo runs on a fresh checkout.
/// Smaller N → the structure is real but the exact hub differs from the full corpus.
const CORPUS_SAMPLE: &str = "state/voicings/guitar-corpus.json";

/// Per-corpus parameters: a set-class needs `min_support` voicings to be a stream
/// (so every bin is populated and a sparse profile can't fake a bridge), and the mesh
/// keeps the top `top_k` by support. Scaled to the corpus so the sample still runs.
struct Params {
    min_support: f64,
    top_k: usize,
}

/// The aligned axis a stream's profile is binned over.
#[derive(Clone, Copy)]
enum Axis {
    /// `minFret` — *where* on the neck a set-class's voicings sit (0..17).
    Position,
    /// `fretSpan` — *how wide* its shapes stretch (0..4); an ergonomic-difficulty lens.
    Stretch,
}

impl Axis {
    /// The voicing column binned over, and the inclusive bin count.
    fn column(self) -> &'static str {
        match self {
            Axis::Position => "minFret",
            Axis::Stretch => "fretSpan",
        }
    }
    fn bins(self) -> usize {
        match self {
            Axis::Position => 18,
            Axis::Stretch => 5,
        }
    }
    fn title(self) -> &'static str {
        match self {
            Axis::Position => "POSITION (minFret — where on the neck)",
            Axis::Stretch => "STRETCH (fretSpan — ergonomic width)",
        }
    }
    /// Name the fret-band a peak bin falls in (used to label regions).
    fn band(self, bin: usize) -> &'static str {
        match self {
            Axis::Position if bin <= 4 => "open",
            Axis::Position if bin <= 11 => "mid-neck",
            Axis::Position => "upper",
            Axis::Stretch if bin == 0 => "no-stretch",
            Axis::Stretch if bin <= 2 => "moderate",
            Axis::Stretch => "wide",
        }
    }
}

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;

    // Prefer the full corpus; fall back to the tracked sample on a fresh checkout.
    let (corpus, params, full) = if std::path::Path::new(CORPUS_FULL).exists() {
        (CORPUS_FULL, Params { min_support: 1000.0, top_k: 120 }, true)
    } else {
        (CORPUS_SAMPLE, Params { min_support: 2.0, top_k: 24 }, false)
    };
    if !full {
        println!(
            "note: full corpus '{CORPUS_FULL}' not found (gitignored, 110 MB) — using the \
             tracked {CORPUS_SAMPLE} sample.\n      Run `cargo run -p ix-voicings` to dump the \
             full corpus and reproduce the 100+ stream result.\n"
        );
    }

    for axis in [Axis::Position, Axis::Stretch] {
        run_axis(&conn, corpus, &params, axis)?;
    }
    Ok(())
}

/// Run the full mesh for one axis: annotate → bin → common-mode → streams → DAG →
/// mesh → τ-sweep → named regions → hub.
fn run_axis(conn: &ix_duck::Connection, corpus: &str, params: &Params, axis: Axis) -> ix_duck::Result<()> {
    let (bins, col) = (axis.bins(), axis.column());
    println!("\n══════════════════════════════════════════════════════════════════════");
    println!("AXIS: {}", axis.title());
    println!("══════════════════════════════════════════════════════════════════════\n");

    // ── annotate + bin: one pass → (set-class, bin) → count ──────────────────────
    let sql = format!(
        "WITH annotated AS (
            SELECT ix_forte_number(midiNotes) AS sc, {col} AS bin
            FROM read_json_auto('{corpus}')
         )
         SELECT sc, bin, count(*) AS n
         FROM annotated
         WHERE sc IS NOT NULL AND bin >= 0 AND bin < {bins}
         GROUP BY sc, bin"
    );
    let mut profiles: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))?;
    for row in rows {
        let (sc, bin, n) = row?;
        profiles.entry(sc).or_insert_with(|| vec![0.0; bins])[bin as usize] = n as f64;
    }

    let total: f64 = profiles.values().flatten().sum();
    println!("corpus: {corpus} → {} set-classes, {total:.0} classifiable voicings over {col} 0..{}\n", profiles.len(), bins - 1);

    // ── qualifying population, then common-mode removal ──────────────────────────
    // Raw profiles share one dominant trend (every set-class crowds the low frets /
    // wide spans), so raw Pearson sees them as identical. We want each set-class's
    // *distinctive* preference: normalize to a distribution, subtract the per-bin
    // cross-set-class mean, and correlate the residual (the anomaly vs the typical).
    let qualifying: Vec<(String, Vec<f64>, f64)> = profiles
        .into_iter()
        .map(|(sc, v)| { let s: f64 = v.iter().sum(); (sc, v, s) })
        .filter(|(_, _, s)| *s >= params.min_support)
        .collect();

    let dists: Vec<Vec<f64>> = qualifying.iter().map(|(_, v, s)| v.iter().map(|n| n / s).collect()).collect();
    let mut mean = vec![0.0f64; bins];
    for d in &dists {
        for (k, p) in d.iter().enumerate() {
            mean[k] += p / dists.len() as f64;
        }
    }

    // ── select streams: top-K by support, as residual profiles; keep the raw peak
    //    bin for region naming. ──────────────────────────────────────────────────
    let mut ranked: Vec<Stream> = qualifying
        .iter()
        .zip(&dists)
        .map(|((sc, raw, support), dist)| Stream {
            name: sc.clone(),
            residual: dist.iter().zip(&mean).map(|(p, m)| p - m).collect(),
            support: *support,
            peak_bin: argmax(raw),
        })
        .collect();
    ranked.sort_by(|a, b| b.support.partial_cmp(&a.support).unwrap());
    ranked.truncate(params.top_k);

    let streams: Vec<(String, Vec<f64>)> = ranked.iter().map(|s| (s.name.clone(), s.residual.clone())).collect();
    println!("{} streams (top set-classes by support, ≥{:.0} voicings), common-mode removed.", streams.len(), params.min_support);

    // ── reify the operator DAG (ADR-0004 "both layered") ─────────────────────────
    let dag = build_operator_dag(&streams, col);
    let levels = dag.parallel_levels();
    let (crit, crit_len) = dag.critical_path(|_, _| 1.0);
    println!(
        "operator DAG (ix_pipeline::dag): {} operators, {} edges, depth {}, fan-out {}, critical path {} ops",
        dag.node_count(), dag.edge_count(), levels.len(),
        levels.iter().map(Vec::len).max().unwrap_or(0), crit_len as usize
    );
    println!("   critical chain: {}\n", crit.iter().take(7).map(|s| s.as_str()).collect::<Vec<_>>().join(" → "));

    // ── run the mesh at the headline τ ───────────────────────────────────────────
    let cfg = MeshConfig { threshold: THRESHOLD, ..MeshConfig::default() };
    let mesh = correlate(conn, &streams, &cfg)?;

    if streams.len() <= FULL_MATRIX_MAX {
        print_matrix(&mesh, &streams);
    }

    // ── τ-sweep: region structure as the threshold tightens ──────────────────────
    // Recount components in-process from the already-computed correlation matrix —
    // no need to re-run the mesh per τ.
    println!("τ-sweep (regions = connected components of the |r| ≥ τ graph):");
    for tau in TAU_SWEEP {
        let comps = components_at(&mesh.correlation, tau);
        let non_singleton = comps.iter().filter(|c| c.len() > 1).count();
        println!("   τ = {tau:.2} → {} regions ({} multi-member)", comps.len(), non_singleton);
    }

    // Name the regions at the τ that fractures the web into the most multi-member
    // regions (on ties `max_by_key` keeps the tighter τ — the sharper split).
    let split_tau = TAU_SWEEP
        .iter()
        .copied()
        .max_by_key(|&t| components_at(&mesh.correlation, t).iter().filter(|c| c.len() > 1).count())
        .unwrap_or(THRESHOLD);
    println!("\nnamed fret-band regions at τ = {split_tau:.2}:");
    let mut regions = components_at(&mesh.correlation, split_tau);
    regions.sort_by_key(|c| std::cmp::Reverse(c.len()));
    for region in regions.iter().filter(|c| c.len() >= 2).take(6) {
        let band = dominant_band(region, &ranked, axis);
        let mut members: Vec<&str> = region.iter().map(|&i| ranked[i].name.as_str()).collect();
        members.truncate(10);
        println!("   {band:>9}-band region ({} set-classes): {{ {} }}", region.len(), members.join(", "));
    }

    // ── betweenness leaders at the headline τ (raw output — see the null-model
    //    validation in docs/walkthroughs/voicing-mesh.md before treating as a claim) ─
    println!("\nix_centrality ({:?}) — top betweenness set-classes at τ = {THRESHOLD}:", cfg.centrality);
    for &(i, score) in mesh.centrality.iter().take(6) {
        let s = &ranked[i];
        println!("   {:>6}: betweenness {score:>8.1}  ({:.0} voicings, peaks in {}-band)", s.name, s.support, axis.band(s.peak_bin));
    }
    let caveat = match axis {
        Axis::Position => "validated as real structure (null-model p = 0.001), though the single-leader identity is fragile",
        Axis::Stretch => "NOT significant — within the null-model range (p = 0.98); treat as a negative result",
    };
    println!("▶ betweenness leader on the {col} axis: {}  [{caveat}]", mesh.lead_name().unwrap_or("?"));
    Ok(())
}

/// A mesh stream: a set-class, its common-mode-removed profile, total support, and the
/// raw peak bin (for region naming).
struct Stream {
    name: String,
    residual: Vec<f64>,
    support: f64,
    peak_bin: usize,
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

/// Connected components of the graph `{ (i,j) : |correlation[i][j]| ≥ τ }`, via
/// union-find. Used for the in-process τ-sweep (no mesh re-run).
#[allow(clippy::needless_range_loop)] // triangular i<j scan over the matrix; indices feed union-find
fn components_at(corr: &[Vec<f64>], tau: f64) -> Vec<Vec<usize>> {
    let n = corr.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r { r = p[r]; }
        let mut c = x;
        while p[c] != c { let nx = p[c]; p[c] = r; c = nx; }
        r
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if corr[i][j].abs() >= tau {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b { parent[a] = b; }
            }
        }
    }
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        by_root.entry(r).or_default().push(i);
    }
    by_root.into_values().collect()
}

/// The band a region is *distinctively* drawn to: the bin where the region's members
/// most over-index versus the typical set-class (argmax of their mean residual). This
/// is what the mesh actually clustered on — more telling than the raw peak, which is
/// "open" for almost every set-class.
fn dominant_band(region: &[usize], ranked: &[Stream], axis: Axis) -> &'static str {
    let bins = ranked.first().map(|s| s.residual.len()).unwrap_or(0);
    let mut mean = vec![0.0f64; bins];
    for &i in region {
        for (k, r) in ranked[i].residual.iter().enumerate() {
            mean[k] += r / region.len() as f64;
        }
    }
    axis.band(argmax(&mean))
}

fn print_matrix(mesh: &MeshResult, streams: &[(String, Vec<f64>)]) {
    print!("ix_pearson residual-profile correlation\n        ");
    for (sc, _) in streams { print!("{sc:>7}"); }
    println!();
    for (i, (sc, _)) in streams.iter().enumerate() {
        print!("{sc:>6}  ");
        for j in 0..streams.len() { print!("{:>7.2}", mesh.correlation[i][j]); }
        println!();
    }
    println!();
}

/// Build the operator graph behind the mesh: shared head (read → annotate → bin), a
/// 4-stage pipeline per stream (profile → normalize → residual → smooth) joined through
/// a common-mode barrier, and the shared mesh tail. Acyclic by construction.
fn build_operator_dag(streams: &[(String, Vec<f64>)], bin_col: &str) -> Dag<&'static str> {
    let mut dag: Dag<&'static str> = Dag::new();
    let node = |d: &mut Dag<&'static str>, id: &str, kind: &'static str| { d.add_node(id.to_string(), kind).ok(); };
    let edge = |d: &mut Dag<&'static str>, a: &str, b: &str| { d.add_edge(a.to_string(), b.to_string()).expect("acyclic by construction"); };

    let bin_label: &'static str = if bin_col == "fretSpan" { "GROUP BY fretSpan" } else { "GROUP BY minFret" };
    for (id, kind) in [
        ("src:read_json", "read_json_auto"),
        ("op:annotate", "ix_forte_number"),
        ("op:bin", bin_label),
        ("op:common_mode", "cross-set-class mean"),
        ("mesh:pearson", "ix_pearson N×N"),
        ("mesh:threshold", "|r| ≥ τ"),
        ("mesh:components", "ix_connected_components"),
        ("mesh:centrality", "ix_centrality"),
    ] {
        node(&mut dag, id, kind);
    }
    edge(&mut dag, "src:read_json", "op:annotate");
    edge(&mut dag, "op:annotate", "op:bin");
    edge(&mut dag, "mesh:pearson", "mesh:threshold");
    edge(&mut dag, "mesh:threshold", "mesh:components");
    edge(&mut dag, "mesh:components", "mesh:centrality");

    for (i, _) in streams.iter().enumerate() {
        let (profile, norm, resid, smooth) =
            (format!("profile:{i}"), format!("norm:{i}"), format!("resid:{i}"), format!("smooth:{i}"));
        node(&mut dag, &profile, "profile");
        node(&mut dag, &norm, "normalize");
        node(&mut dag, &resid, "subtract common-mode");
        node(&mut dag, &smooth, "ix_wavelet_denoise");
        edge(&mut dag, "op:bin", &profile);
        edge(&mut dag, &profile, &norm);
        edge(&mut dag, &norm, "op:common_mode");
        edge(&mut dag, &norm, &resid);
        edge(&mut dag, "op:common_mode", &resid);
        edge(&mut dag, &resid, &smooth);
        edge(&mut dag, &smooth, "mesh:pearson");
    }
    dag
}
