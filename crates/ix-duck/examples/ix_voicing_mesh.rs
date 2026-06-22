//! `ix_voicing_mesh` — a **real-world** pipeline-mesh (ADR-0004) over the GA guitar
//! voicing corpus (`state/voicings/raw/guitar.jsonl`, ~667 k fingerings).
//!
//! Analysis question:
//!   *Which chord set-classes occupy the same fretboard regions, and which
//!    set-class is the structural hub of the guitar's voicing geometry?*
//!
//! Each **stream** is one Forte set-class; the aligned axis is **fret position**
//! (`minFret`), so a stream is that set-class's *fretboard-position profile*
//! (#voicings per fret). One load-bearing pipeline stage — **common-mode removal** —
//! makes the mesh informative: every set-class crowds the low frets, so raw profiles
//! all correlate ~1 (one clique, no hub). Normalizing each profile to a distribution
//! and subtracting the cross-set-class mean leaves each set-class's *distinctive*
//! fret preference, which is what the mesh correlates.
//!
//! **Both layered** (ADR-0004): the demo is a graph of ~460 *operators* (a 4-stage
//! pipeline per stream, joined through a common-mode barrier, plus a shared head and
//! mesh tail — reified as an `ix_pipeline::dag::Dag`) whose ~120 *stream* outputs feed
//! the N×N correlation mesh. Every layer is an IX UDF on the DuckDB bench:
//! ```text
//!   read_json_auto ─▶ ix_forte_number (annotate) ─▶ GROUP BY minFret (bin)
//!     ─▶ per-set-class profile ─▶ normalize ─▶ subtract common-mode
//!     ─▶ ix_wavelet_denoise (condition) ─▶ ix_pearson (N×N)
//!     ─▶ ix_connected_components (regions) ─▶ ix_centrality betweenness (hub)
//! ```
//!
//! Finding (τ = 0.8, 114 well-supported set-classes): guitar voicing set-classes form
//! one connected fretboard-geometry web, with **5-29** its structural hub (betweenness
//! ~112, 3.5× the runner-up). The lever is the `|r|` threshold τ (ADR-0004).
//!
//! Run: `cargo run -p ix-duck --example ix_voicing_mesh --features duck`

use std::collections::BTreeMap;

use ix_duck::mesh::{correlate, MeshConfig, MeshResult};
use ix_duck::open_bench;
use ix_pipeline::dag::Dag;

/// Fret-position axis: `minFret` 0..FRETS-1 (the aligned vector index).
const FRETS: usize = 18;
/// `|r|` edge threshold. At 0.4 the de-trended mesh is one giant clique; a stricter
/// operating point carves out distinct fretboard regions (ADR-0004: the lever is τ).
const THRESHOLD: f64 = 0.8;
/// Print the full N×N matrix only for a small (tracer-bullet) mesh.
const FULL_MATRIX_MAX: usize = 12;

/// The full GA CLI dump (~667 k fingerings) — gitignored (110 MB), so present only on
/// a machine that has run `ix-voicings`. Yields the headline 100+ stream / 5-29-hub run.
const CORPUS_FULL: &str = "state/voicings/raw/guitar.jsonl";
/// A tracked 500-voicing sample (same schema) so the demo runs on a fresh checkout.
/// Smaller N → the structure is real but the exact hub differs from the full corpus.
const CORPUS_SAMPLE: &str = "state/voicings/guitar-corpus.json";

/// Per-corpus parameters: a set-class needs `min_support` voicings to be a stream
/// (so every fret bin is populated and a sparse profile can't fake a bridge), and the
/// mesh keeps the top `top_k` by support. Scaled to the corpus so the sample still runs.
struct Params {
    min_support: f64,
    top_k: usize,
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
             full corpus and reproduce the 100+ stream / 5-29-hub result.\n"
        );
    }

    // ── annotate + bin: one pass over the corpus → (set-class, fret) → count ──────
    // ix_forte_number(midiNotes) is the per-row annotate operator; GROUP BY is the bin.
    let sql = format!(
        "WITH annotated AS (
            SELECT ix_forte_number(midiNotes) AS sc, minFret AS fret
            FROM read_json_auto('{corpus}')
         )
         SELECT sc, fret, count(*) AS n
         FROM annotated
         WHERE sc IS NOT NULL AND fret >= 0 AND fret < {FRETS}
         GROUP BY sc, fret"
    );

    let mut profiles: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    for row in rows {
        let (sc, fret, n) = row?;
        let v = profiles.entry(sc).or_insert_with(|| vec![0.0; FRETS]);
        v[fret as usize] = n as f64;
    }

    // ── data-shape summary (calibration) ─────────────────────────────────────────
    let total_voicings: f64 = profiles.values().flat_map(|v| v.iter()).sum();
    println!(
        "corpus: {} → {} distinct set-classes, {:.0} classifiable voicings over frets 0..{}\n",
        corpus,
        profiles.len(),
        total_voicings,
        FRETS - 1
    );

    // ── qualifying population: set-classes with enough support ───────────────────
    let qualifying: Vec<(String, Vec<f64>, f64)> = profiles
        .into_iter()
        .map(|(sc, v)| {
            let support: f64 = v.iter().sum();
            (sc, v, support)
        })
        .filter(|(_, _, support)| *support >= params.min_support)
        .collect();

    // ── common-mode removal (the load-bearing pipeline stage) ────────────────────
    // Raw fret profiles all share one dominant trend — *every* set-class crowds the
    // low frets — so raw Pearson sees them as identical (one clique, no hub). We want
    // each set-class's *distinctive* fret preference, so:
    //   1. normalize each profile to a positional distribution (removes the
    //      frequency confound: a common set-class shouldn't outweigh a rare one), then
    //   2. subtract the cross-set-class mean distribution per fret, leaving the
    //      *anomaly* — where this set-class over/under-indexes vs the typical one.
    // Pearson on these residuals correlates distinctive co-location, not the shared trend.
    let mut mean_dist = vec![0.0f64; FRETS];
    let dists: Vec<Vec<f64>> = qualifying
        .iter()
        .map(|(_, raw, support)| raw.iter().map(|n| n / support).collect::<Vec<f64>>())
        .collect();
    for dist in &dists {
        for (k, p) in dist.iter().enumerate() {
            mean_dist[k] += p / dists.len() as f64;
        }
    }
    let residual = |dist: &[f64]| -> Vec<f64> {
        dist.iter().zip(&mean_dist).map(|(p, m)| p - m).collect()
    };

    // ── select streams: top-K set-classes by total support, as residual profiles ──
    let mut ranked: Vec<(String, Vec<f64>, f64)> = qualifying
        .iter()
        .zip(&dists)
        .map(|((sc, _, support), dist)| (sc.clone(), residual(dist), *support))
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    ranked.truncate(params.top_k);

    let streams: Vec<(String, Vec<f64>)> = ranked
        .iter()
        .map(|(sc, v, _)| (sc.clone(), v.clone()))
        .collect();

    println!(
        "{} streams (top set-classes by support, ≥{:.0} voicings) over a {FRETS}-fret \
         axis, common-mode removed:",
        streams.len(),
        params.min_support
    );
    for (sc, _, support) in &ranked {
        println!("   {sc:>6}  ({support:.0} voicings)");
    }
    println!();

    // ── reify the operator DAG (ADR-0004 "both layered") ─────────────────────────
    // The mesh is *two layered*: a graph of 100+ operators (the per-stream pipelines
    // + the shared head/tail) whose stream outputs feed the N×N correlation mesh.
    let dag = build_operator_dag(&streams);
    let levels = dag.parallel_levels();
    let (crit, crit_len) = dag.critical_path(|_, _| 1.0);
    println!("operator DAG (ix_pipeline::dag) — the \"both layered\" pipeline graph:");
    println!(
        "   {} operators, {} edges, depth {} (parallel levels), critical path {} ops",
        dag.node_count(),
        dag.edge_count(),
        levels.len(),
        crit_len as usize
    );
    println!(
        "   widest level fans out to {} concurrent operators (one per stream)",
        levels.iter().map(Vec::len).max().unwrap_or(0)
    );
    println!("   critical chain: {}\n", crit.iter().take(7).cloned().cloned().collect::<Vec<_>>().join(" → "));

    // ── run the mesh ─────────────────────────────────────────────────────────────
    let cfg = MeshConfig { threshold: THRESHOLD, ..MeshConfig::default() };
    let mesh = correlate(&conn, &streams, &cfg)?;

    if streams.len() <= FULL_MATRIX_MAX {
        print!("ix_pearson fretboard-profile correlation\n        ");
        for (sc, _) in &streams {
            print!("{sc:>7}");
        }
        println!();
        for (i, (sc, _)) in streams.iter().enumerate() {
            print!("{sc:>6}  ");
            for j in 0..streams.len() {
                print!("{:>7.2}", mesh.correlation[i][j]);
            }
            println!();
        }
        println!();
    }

    report_mesh(&mesh, &cfg, &ranked);
    Ok(())
}

/// Build the operator graph behind the mesh: shared head (read → annotate → bin),
/// a 4-stage pipeline per stream (profile → normalize → residual → smooth) joined
/// through a common-mode barrier, and the shared mesh tail (pearson → threshold →
/// components → centrality). Acyclic by construction.
fn build_operator_dag(streams: &[(String, Vec<f64>)]) -> Dag<&'static str> {
    let mut dag: Dag<&'static str> = Dag::new();
    let node = |d: &mut Dag<&'static str>, id: &str, kind: &'static str| {
        d.add_node(id.to_string(), kind).ok();
    };
    let edge = |d: &mut Dag<&'static str>, a: &str, b: &str| {
        d.add_edge(a.to_string(), b.to_string()).expect("acyclic by construction");
    };

    // Shared head + common-mode barrier + shared tail.
    for (id, kind) in [
        ("src:read_json", "read_json_auto"),
        ("op:annotate", "ix_forte_number"),
        ("op:bin", "GROUP BY minFret"),
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

    // Per-stream 4-stage pipeline, joined through the common-mode barrier.
    for (i, _) in streams.iter().enumerate() {
        let (profile, norm, resid, smooth) =
            (format!("profile:{i}"), format!("norm:{i}"), format!("resid:{i}"), format!("smooth:{i}"));
        node(&mut dag, &profile, "fret profile");
        node(&mut dag, &norm, "normalize");
        node(&mut dag, &resid, "subtract common-mode");
        node(&mut dag, &smooth, "ix_wavelet_denoise");
        edge(&mut dag, "op:bin", &profile);
        edge(&mut dag, &profile, &norm);
        edge(&mut dag, &norm, "op:common_mode"); // fan-in to the mean
        edge(&mut dag, &norm, &resid);
        edge(&mut dag, "op:common_mode", &resid); // fan-out back to each residual
        edge(&mut dag, &resid, &smooth);
        edge(&mut dag, &smooth, "mesh:pearson"); // fan-in to the correlation barrier
    }
    dag
}

/// Print the mesh verdict — region sizes (largest first) + the top hub ranking.
fn report_mesh(mesh: &MeshResult, cfg: &MeshConfig, ranked: &[(String, Vec<f64>, f64)]) {
    let mut sizes: Vec<&Vec<usize>> = mesh.clusters.iter().collect();
    sizes.sort_by_key(|c| std::cmp::Reverse(c.len()));
    println!(
        "ix_connected_components (|r| ≥ {}) → {} fretboard region(s); sizes: {:?}",
        cfg.threshold,
        mesh.clusters.len(),
        sizes.iter().map(|c| c.len()).collect::<Vec<_>>()
    );
    if let Some(largest) = sizes.first() {
        let mut labels: Vec<&str> = largest.iter().map(|&i| mesh.names[i].as_str()).collect();
        labels.truncate(16);
        println!("   largest region (first {} of {}): {{ {} }}", labels.len(), sizes[0].len(), labels.join(", "));
    }

    println!("\nix_centrality ({:?}) — top hub set-classes:", cfg.centrality);
    let support = |name: &str| ranked.iter().find(|(sc, _, _)| sc == name).map(|(_, _, s)| *s).unwrap_or(0.0);
    for &(i, score) in mesh.centrality.iter().take(10) {
        let name = mesh.names[i].as_str();
        println!("   {name:>6}: betweenness {score:>10.1}  ({:.0} voicings)", support(name));
    }
    println!(
        "\n▶ structural hub of the guitar's voicing geometry: {} \
         (the set-class that most bridges distinct fretboard regions)",
        mesh.lead_name().unwrap_or("?")
    );
}
