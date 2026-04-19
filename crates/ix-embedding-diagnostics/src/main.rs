//! # baseline-diagnostics
//!
//! BASELINE diagnostic tool for OPTIC-K voicing embeddings.
//!
//! Runs four measurements over the current embedding space and writes a JSON
//! report. These baselines will be compared against post-refactor measurements
//! to prove (or refute) that the chord-recognition architecture refactor
//! actually improved embedding quality.
//!
//! ## Diagnostics
//!
//! 1. **Leak detection** — train a classifier to predict instrument from the
//!    embedding, sliced by partition (STRUCTURE / MORPHOLOGY / CONTEXT /
//!    SYMBOLIC / MODAL / FULL). If accuracy on the STRUCTURE partition is
//!    meaningfully above chance (~33% for 3 instruments), pitch-class
//!    structure is leaking instrument identity.
//!
//! 2. **Retrieval consistency** — for random query voicings, synthesize a
//!    STRUCTURE-only query (other partitions zeroed) and check the fraction
//!    of top-10 results sharing the query's pitch-class set.
//!
//! 3. **K-means cluster baseline** — run K-means (k=50) over the full corpus
//!    and save cluster assignments. Post-refactor, the same clustering will be
//!    compared via Adjusted Rand Index.
//!
//! 4. **Per-instrument topology** — compute β₀ and β₁ via Vietoris-Rips
//!    persistent homology on a random sample per instrument. Divergent Betti
//!    numbers across instruments indicate instrument-specific topology in the
//!    embedding space.
//!
//! ## CLI
//!
//! ```bash
//! cargo run -p ix-embedding-diagnostics --release -- \
//!     --index "C:/Users/spare/source/repos/ga/state/voicings/optick.index" \
//!     --out-dir "C:/Users/spare/source/repos/ga/state/baseline"
//! ```

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use ix_ensemble::random_forest::RandomForest;
use ix_ensemble::traits::EnsembleClassifier;
use ix_optick::OptickIndex;
use ix_supervised::metrics::accuracy;
use ix_supervised::validation::StratifiedKFold;
use ix_topo::pointcloud::betti_at_radius;
use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::traits::Clusterer;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;

// -----------------------------------------------------------------------------
// Partition layout — must match ix-optick SCHEMA_SEED.
//   STRUCTURE  0..24
//   MORPHOLOGY 24..48
//   CONTEXT    48..60
//   SYMBOLIC   60..72
//   MODAL      72..112
//   ROOT       112..124   (v1.8 — schema v4-pp-r, 2026-04-19)
// -----------------------------------------------------------------------------

const DIM: usize = 124;

#[derive(Clone, Copy)]
struct Partition {
    name: &'static str,
    start: usize,
    end: usize,
    /// Hypothesis tag used in the report (not algorithmic).
    expected_leak: &'static str,
}

const PARTITIONS: &[Partition] = &[
    Partition { name: "STRUCTURE",  start: 0,   end: 24,  expected_leak: "likely_leak" },
    Partition { name: "MORPHOLOGY", start: 24,  end: 48,  expected_leak: "by_design"   },
    Partition { name: "CONTEXT",    start: 48,  end: 60,  expected_leak: "possible"    },
    Partition { name: "SYMBOLIC",   start: 60,  end: 72,  expected_leak: "likely_leak" },
    Partition { name: "MODAL",      start: 72,  end: 112, expected_leak: "possible"    },
    Partition { name: "ROOT",       start: 112, end: 124, expected_leak: "by_design"   },
];

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "baseline-diagnostics",
    about = "Baseline OPTIC-K embedding diagnostics (leak, retrieval, clustering, topology)"
)]
struct Cli {
    /// Path to the optick.index v4 file.
    #[arg(long)]
    index: PathBuf,

    /// Directory to write the report + cluster labels.
    #[arg(long, default_value = "state/baseline")]
    out_dir: PathBuf,

    /// Max samples PER INSTRUMENT for classification (diagnostic 1).
    /// Balanced sampling counteracts the ~96/3/1 guitar/bass/ukulele imbalance.
    #[arg(long, default_value_t = 4000)]
    class_samples_per_instrument: usize,

    /// Number of random-forest trees (deeper = slower but more stable).
    #[arg(long, default_value_t = 30)]
    n_trees: usize,

    /// Max tree depth.
    #[arg(long, default_value_t = 10)]
    tree_depth: usize,

    /// Number of queries for retrieval consistency (diagnostic 2).
    #[arg(long, default_value_t = 50)]
    retrieval_queries: usize,

    /// K for the K-means baseline (diagnostic 3).
    #[arg(long, default_value_t = 50)]
    kmeans_k: usize,

    /// Max K-means iterations (lower = faster; default is enough to settle).
    #[arg(long, default_value_t = 30)]
    kmeans_iter: usize,

    /// Sample size per instrument for persistent homology (diagnostic 4).
    /// Rips is O(n²) in edges — keep modest.
    #[arg(long, default_value_t = 1000)]
    topo_sample: usize,

    /// Deterministic seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// -----------------------------------------------------------------------------
// Report struct
// -----------------------------------------------------------------------------

#[derive(Serialize)]
struct CorpusInfo {
    file: String,
    count: u64,
    dims: u32,
    instruments: BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct PartitionResult {
    partition: String,
    start: usize,
    end: usize,
    dims: usize,
    accuracy_mean: f64,
    accuracy_std: f64,
    per_fold: Vec<f64>,
    leak_hypothesis: String,
    leak_confirmed: bool,
}

#[derive(Serialize)]
struct LeakDetection {
    sample_size: usize,
    per_class: BTreeMap<String, usize>,
    n_folds: usize,
    classifier: String,
    full_classifier_accuracy: f64,
    baseline_random: f64,
    by_partition: Vec<PartitionResult>,
}

#[derive(Serialize)]
struct RetrievalConsistency {
    queries_tested: usize,
    top_k: usize,
    avg_pc_set_match_pct: f64,
    per_query_match_pct: Vec<f64>,
    note: String,
}

#[derive(Serialize)]
struct ClusterBaseline {
    k: usize,
    n_voicings: usize,
    iterations_ran: usize,
    file: String,
}

#[derive(Serialize)]
struct TopologyEntry {
    sample_size: usize,
    beta_0: usize,
    beta_1: usize,
    filtration_radius: f64,
    avg_pairwise_distance: f64,
}

#[derive(Serialize)]
struct TopologyReport {
    guitar: Option<TopologyEntry>,
    bass: Option<TopologyEntry>,
    ukulele: Option<TopologyEntry>,
    cross_instrument_match: String,
}

#[derive(Serialize)]
struct Report {
    timestamp: String,
    tool: String,
    seed: u64,
    corpus: CorpusInfo,
    leak_detection: LeakDetection,
    retrieval_consistency: RetrievalConsistency,
    cluster_baseline: ClusterBaseline,
    topology: TopologyReport,
    notes: Vec<String>,
    runtime_seconds: f64,
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Return `(label_per_voicing, indices_per_class)` from the instrument slices.
fn collect_labels(index: &OptickIndex) -> (Vec<u8>, [Vec<usize>; 3]) {
    let count = index.count() as usize;
    let mut labels = vec![0u8; count];
    let mut per_class: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, lbl_slot) in labels.iter_mut().enumerate().take(count) {
        let lbl = match index.instrument_of(i) {
            Some("guitar") => 0u8,
            Some("bass") => 1u8,
            Some("ukulele") => 2u8,
            _ => 0u8,
        };
        *lbl_slot = lbl;
        per_class[lbl as usize].push(i);
    }
    (labels, per_class)
}

/// Stratified sample: take up to `n_per_class` from each class.
fn stratified_sample(
    per_class: &[Vec<usize>; 3],
    n_per_class: usize,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<u8>) {
    let mut ids = Vec::new();
    let mut labels = Vec::new();
    for (class, pool) in per_class.iter().enumerate() {
        let take = n_per_class.min(pool.len());
        let mut shuffled: Vec<usize> = pool.clone();
        shuffled.shuffle(rng);
        for &v in shuffled.iter().take(take) {
            ids.push(v);
            labels.push(class as u8);
        }
    }
    (ids, labels)
}

/// Build an `Array2<f64>` of selected voicings × slice dims.
fn gather_matrix(
    index: &OptickIndex,
    voicing_ids: &[usize],
    dim_range: std::ops::Range<usize>,
) -> Array2<f64> {
    let d = dim_range.end - dim_range.start;
    let vectors = index.vectors();
    let mut arr = Array2::<f64>::zeros((voicing_ids.len(), d));
    for (row, &vi) in voicing_ids.iter().enumerate() {
        let base = vi * DIM;
        for (col, src) in dim_range.clone().enumerate() {
            arr[[row, col]] = vectors[base + src] as f64;
        }
    }
    arr
}

fn select_rows(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), x.ncols()), |(r, c)| x[[idx[r], c]])
}

fn select_elems(y: &Array1<usize>, idx: &[usize]) -> Array1<usize> {
    Array1::from_iter(idx.iter().map(|&i| y[i]))
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Midi note → pitch class (0..12).
fn pc_set(midi: &[i32]) -> HashSet<u8> {
    midi.iter().map(|&m| (m.rem_euclid(12)) as u8).collect()
}

/// Build a query vector that preserves only a given partition slice.
fn partition_only_query(src: &[f32], part: Partition) -> Vec<f32> {
    let mut q = vec![0.0f32; DIM];
    q[part.start..part.end].copy_from_slice(&src[part.start..part.end]);
    q
}

// -----------------------------------------------------------------------------
// Diagnostic 1: leak detection
// -----------------------------------------------------------------------------

fn run_leak_detection(
    index: &OptickIndex,
    per_class: &[Vec<usize>; 3],
    cli: &Cli,
    rng: &mut StdRng,
) -> LeakDetection {
    let (voicing_ids, labels_u8) = stratified_sample(per_class, cli.class_samples_per_instrument, rng);
    let n = voicing_ids.len();
    let y: Array1<usize> = Array1::from_iter(labels_u8.iter().map(|&b| b as usize));

    let mut per_class_count: BTreeMap<String, usize> = BTreeMap::new();
    for (cls, pool) in per_class.iter().enumerate() {
        let name = match cls {
            0 => "guitar",
            1 => "bass",
            _ => "ukulele",
        };
        per_class_count.insert(
            name.to_string(),
            cli.class_samples_per_instrument.min(pool.len()),
        );
    }

    // Full-dim classifier (baseline)
    let full_x = gather_matrix(index, &voicing_ids, 0..DIM);
    let full_folds_acc = kfold_eval(&full_x, &y, cli.n_trees, cli.tree_depth, cli.seed, 5);
    let (full_mean, _) = mean_std(&full_folds_acc);

    // Per-partition classifiers
    let mut by_partition = Vec::new();
    for part in PARTITIONS {
        let x = gather_matrix(index, &voicing_ids, part.start..part.end);
        let fold_scores = kfold_eval(&x, &y, cli.n_trees, cli.tree_depth, cli.seed, 5);
        let (mean, std) = mean_std(&fold_scores);

        // "Leak confirmed" = accuracy materially above random for 3 balanced classes.
        // Random baseline = 1/3 = 0.333. We call leak confirmed if mean > 0.40
        // AND the partition is not MORPHOLOGY (which is leak-by-design).
        let leak_confirmed = mean > 0.40 && part.name != "MORPHOLOGY";

        by_partition.push(PartitionResult {
            partition: part.name.to_string(),
            start: part.start,
            end: part.end,
            dims: part.end - part.start,
            accuracy_mean: mean,
            accuracy_std: std,
            per_fold: fold_scores,
            leak_hypothesis: part.expected_leak.to_string(),
            leak_confirmed,
        });
    }

    LeakDetection {
        sample_size: n,
        per_class: per_class_count,
        n_folds: 5,
        classifier: format!(
            "RandomForest(n_trees={}, max_depth={})",
            cli.n_trees, cli.tree_depth
        ),
        full_classifier_accuracy: full_mean,
        baseline_random: 1.0 / 3.0,
        by_partition,
    }
}

fn kfold_eval(
    x: &Array2<f64>,
    y: &Array1<usize>,
    n_trees: usize,
    depth: usize,
    seed: u64,
    k: usize,
) -> Vec<f64> {
    let skf = StratifiedKFold::new(k).with_seed(seed);
    let folds = skf.split(y);
    folds
        .iter()
        .map(|(train_idx, test_idx)| {
            let x_tr = select_rows(x, train_idx);
            let y_tr = select_elems(y, train_idx);
            let x_te = select_rows(x, test_idx);
            let y_te = select_elems(y, test_idx);
            let mut rf = RandomForest::new(n_trees, depth).with_seed(seed);
            rf.fit(&x_tr, &y_tr);
            let preds = rf.predict(&x_te);
            accuracy(&y_te, &preds)
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Diagnostic 2: retrieval consistency
// -----------------------------------------------------------------------------

fn run_retrieval_consistency(
    index: &OptickIndex,
    cli: &Cli,
    rng: &mut StdRng,
) -> RetrievalConsistency {
    let count = index.count() as usize;
    let top_k = 10;
    let mut per_query = Vec::with_capacity(cli.retrieval_queries);

    for _ in 0..cli.retrieval_queries {
        let qi = rng.random_range(0..count);
        let q_vec = match index.vector(qi) {
            Some(v) => v.to_vec(),
            None => continue,
        };
        let q_meta = match index.metadata(qi) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let q_pcs = pc_set(&q_meta.midi_notes);

        // STRUCTURE-only query (zero out everything else).
        let structure = PARTITIONS[0];
        let query = partition_only_query(&q_vec, structure);
        let results = match index.search(&query, None, top_k) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let mut match_count = 0usize;
        for r in &results {
            let r_pcs = pc_set(&r.metadata.midi_notes);
            if r_pcs == q_pcs {
                match_count += 1;
            }
        }
        let frac = match_count as f64 / results.len().max(1) as f64;
        per_query.push(frac);
    }

    let avg = if per_query.is_empty() {
        0.0
    } else {
        per_query.iter().sum::<f64>() / per_query.len() as f64
    };

    RetrievalConsistency {
        queries_tested: per_query.len(),
        top_k,
        avg_pc_set_match_pct: avg,
        per_query_match_pct: per_query,
        note: "A STRUCTURE-only query should return voicings sharing the query's pitch-class set. \
               A low match rate indicates the STRUCTURE partition is not cleanly encoding PC-set."
            .to_string(),
    }
}

// -----------------------------------------------------------------------------
// Diagnostic 3: K-means baseline
// -----------------------------------------------------------------------------

#[derive(Serialize)]
struct ClusterLabel {
    voicing_index: usize,
    cluster: usize,
    instrument: String,
}

fn run_cluster_baseline(
    index: &OptickIndex,
    cli: &Cli,
    out_dir: &std::path::Path,
) -> (ClusterBaseline, std::path::PathBuf) {
    let count = index.count() as usize;
    let vectors = index.vectors();

    // Build full 313k × 112 matrix. Peak memory: ~280 MB f64.
    let mut x = Array2::<f64>::zeros((count, DIM));
    for i in 0..count {
        for d in 0..DIM {
            x[[i, d]] = vectors[i * DIM + d] as f64;
        }
    }

    let mut km = KMeans::new(cli.kmeans_k).with_seed(cli.seed);
    km.max_iterations = cli.kmeans_iter;
    km.fit(&x);
    let labels = km.predict(&x);

    let cluster_file = out_dir.join(format!("embedding-clusters-k{}.json", cli.kmeans_k));
    let rows: Vec<ClusterLabel> = (0..count)
        .map(|i| ClusterLabel {
            voicing_index: i,
            cluster: labels[i],
            instrument: index.instrument_of(i).unwrap_or("unknown").to_string(),
        })
        .collect();
    let json = serde_json::to_string(&rows).expect("serialize cluster labels");
    std::fs::write(&cluster_file, json).expect("write cluster labels");

    (
        ClusterBaseline {
            k: cli.kmeans_k,
            n_voicings: count,
            iterations_ran: cli.kmeans_iter,
            file: cluster_file.to_string_lossy().to_string(),
        },
        cluster_file,
    )
}

// -----------------------------------------------------------------------------
// Diagnostic 4: per-instrument topology
// -----------------------------------------------------------------------------

fn run_topology(
    index: &OptickIndex,
    per_class: &[Vec<usize>; 3],
    cli: &Cli,
    rng: &mut StdRng,
) -> TopologyReport {
    let guitar = topo_one(index, &per_class[0], cli.topo_sample, rng);
    let bass = topo_one(index, &per_class[1], cli.topo_sample, rng);
    let ukulele = topo_one(index, &per_class[2], cli.topo_sample, rng);

    // Compare β₀ and β₁ across instruments.
    let all = [&guitar, &bass, &ukulele];
    let finite: Vec<&TopologyEntry> = all.iter().filter_map(|o| o.as_ref()).collect();
    let match_str = if finite.len() < 2 {
        "insufficient_data".to_string()
    } else {
        let b0s: Vec<usize> = finite.iter().map(|e| e.beta_0).collect();
        let b1s: Vec<usize> = finite.iter().map(|e| e.beta_1).collect();
        let b0_diverge = b0s.iter().max().copied().unwrap_or(0)
            .saturating_sub(b0s.iter().min().copied().unwrap_or(0));
        let b1_diverge = b1s.iter().max().copied().unwrap_or(0)
            .saturating_sub(b1s.iter().min().copied().unwrap_or(0));
        format!(
            "beta0_range={}, beta1_range={} (larger = more instrument-specific topology)",
            b0_diverge, b1_diverge
        )
    };

    TopologyReport {
        guitar,
        bass,
        ukulele,
        cross_instrument_match: match_str,
    }
}

fn topo_one(
    index: &OptickIndex,
    pool: &[usize],
    sample_size: usize,
    rng: &mut StdRng,
) -> Option<TopologyEntry> {
    if pool.is_empty() {
        return None;
    }
    let take = sample_size.min(pool.len());
    let mut shuffled: Vec<usize> = pool.to_vec();
    shuffled.shuffle(rng);
    let sample = &shuffled[..take];

    // Convert to Vec<Vec<f64>>.
    let vectors = index.vectors();
    let points: Vec<Vec<f64>> = sample
        .iter()
        .map(|&vi| {
            let base = vi * DIM;
            (0..DIM).map(|d| vectors[base + d] as f64).collect()
        })
        .collect();

    // Estimate avg pairwise distance via sampled pairs (avoids O(n²) on 1000 points).
    let n_pairs = 500;
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for _ in 0..n_pairs {
        let i = rng.random_range(0..points.len());
        let j = rng.random_range(0..points.len());
        if i == j {
            continue;
        }
        let d: f64 = points[i]
            .iter()
            .zip(points[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        sum += d;
        cnt += 1;
    }
    let avg_pairwise = if cnt > 0 { sum / cnt as f64 } else { 0.0 };

    // Use ~40% of avg pairwise as filtration radius — captures connected
    // components while keeping the Rips complex tractable on 1000 points.
    let radius = avg_pairwise * 0.4;

    // Compute Betti numbers at this single radius (max_dim=1 → β₀, β₁).
    let betti = betti_at_radius(&points, 1, radius);
    let b0 = betti.first().copied().unwrap_or(0);
    let b1 = betti.get(1).copied().unwrap_or(0);

    Some(TopologyEntry {
        sample_size: take,
        beta_0: b0,
        beta_1: b1,
        filtration_radius: radius,
        avg_pairwise_distance: avg_pairwise,
    })
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let t0 = Instant::now();

    eprintln!("[1/5] Opening OPTIC-K index: {}", cli.index.display());
    let index = OptickIndex::open(&cli.index)?;
    let count = index.count();
    let dims = index.dimension();
    eprintln!("      count={count} dims={dims}");
    assert_eq!(dims as usize, DIM, "expected {DIM}-dim v4-pp-r index");

    std::fs::create_dir_all(&cli.out_dir)?;

    let (labels_u8, per_class) = collect_labels(&index);
    let mut instrument_counts = BTreeMap::new();
    instrument_counts.insert("guitar".to_string(), per_class[0].len() as u64);
    instrument_counts.insert("bass".to_string(), per_class[1].len() as u64);
    instrument_counts.insert("ukulele".to_string(), per_class[2].len() as u64);

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut notes: Vec<String> = Vec::new();
    let _ = labels_u8; // retained for potential future use

    // Diagnostic 1
    eprintln!("[2/5] Diagnostic 1: leak detection (per-partition classification)");
    let leak = run_leak_detection(&index, &per_class, &cli, &mut rng);
    eprintln!(
        "      full-dim acc={:.3}  (baseline random={:.3})",
        leak.full_classifier_accuracy, leak.baseline_random
    );
    for p in &leak.by_partition {
        eprintln!(
            "      {:<10} acc={:.3}±{:.3}  dims={}  leak_confirmed={}",
            p.partition, p.accuracy_mean, p.accuracy_std, p.dims, p.leak_confirmed
        );
    }

    // Diagnostic 2
    eprintln!("[3/5] Diagnostic 2: retrieval consistency (STRUCTURE-only queries)");
    let retrieval = run_retrieval_consistency(&index, &cli, &mut rng);
    eprintln!(
        "      queries={} avg_pc_set_match={:.3}",
        retrieval.queries_tested, retrieval.avg_pc_set_match_pct
    );

    // Diagnostic 3
    eprintln!("[4/5] Diagnostic 3: K-means baseline (k={})", cli.kmeans_k);
    let (clusters, _cluster_file) = run_cluster_baseline(&index, &cli, &cli.out_dir);
    eprintln!(
        "      wrote cluster labels for {} voicings → {}",
        clusters.n_voicings, clusters.file
    );

    // Diagnostic 4
    eprintln!(
        "[5/5] Diagnostic 4: persistent homology per instrument (sample={})",
        cli.topo_sample
    );
    let topo = run_topology(&index, &per_class, &cli, &mut rng);
    for (name, entry) in [("guitar", &topo.guitar), ("bass", &topo.bass), ("ukulele", &topo.ukulele)] {
        match entry {
            Some(e) => eprintln!(
                "      {:<8} n={} β₀={} β₁={} radius={:.4} avg_d={:.4}",
                name, e.sample_size, e.beta_0, e.beta_1, e.filtration_radius, e.avg_pairwise_distance
            ),
            None => eprintln!("      {:<8} no data", name),
        }
    }
    eprintln!("      {}", topo.cross_instrument_match);

    // Known API gaps — documented so the post-refactor run can close them.
    notes.push(
        "LogisticRegression in ix-supervised is binary-only; RandomForest used instead for 3-class"
            .into(),
    );
    notes.push(
        "ix-topo has no native Adjusted Rand Index; post-refactor comparison will add one"
            .into(),
    );
    notes.push(
        "Rips complex is O(n²) in edges — topology sampled at 1000 points per instrument"
            .into(),
    );
    notes.push(format!(
        "Leak confirmed criterion: mean accuracy > 0.40 on balanced 3-class problem (baseline={:.3})",
        1.0 / 3.0
    ));

    let elapsed = t0.elapsed().as_secs_f64();

    let report = Report {
        timestamp: chrono::Utc::now().to_rfc3339(),
        tool: format!("ix-embedding-diagnostics {}", env!("CARGO_PKG_VERSION")),
        seed: cli.seed,
        corpus: CorpusInfo {
            file: cli.index.to_string_lossy().to_string(),
            count,
            dims,
            instruments: instrument_counts,
        },
        leak_detection: leak,
        retrieval_consistency: retrieval,
        cluster_baseline: clusters,
        topology: topo,
        notes,
        runtime_seconds: elapsed,
    };

    // Write JSON report.
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let report_path = cli
        .out_dir
        .join(format!("embedding-diagnostics-{date}.json"));
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_path, json)?;

    eprintln!();
    eprintln!("─────────────────────────────────────────────────────────────");
    eprintln!(" BASELINE SUMMARY");
    eprintln!("─────────────────────────────────────────────────────────────");
    eprintln!(" corpus            {} voicings × {} dims", report.corpus.count, report.corpus.dims);
    eprintln!(" full-dim acc      {:.3}  (vs random {:.3})",
        report.leak_detection.full_classifier_accuracy, report.leak_detection.baseline_random);
    for p in &report.leak_detection.by_partition {
        let flag = if p.leak_confirmed { "LEAK " } else if p.partition == "MORPHOLOGY" { "BYDSG" } else { "  ok " };
        eprintln!(" {:<10} [{}]  acc={:.3} ± {:.3}", p.partition, flag, p.accuracy_mean, p.accuracy_std);
    }
    eprintln!(" retrieval         avg PC-set match = {:.1}%", 100.0 * report.retrieval_consistency.avg_pc_set_match_pct);
    eprintln!(" clusters          k={} saved to {}", report.cluster_baseline.k, report.cluster_baseline.file);
    if let (Some(g), Some(b), Some(u)) =
        (&report.topology.guitar, &report.topology.bass, &report.topology.ukulele)
    {
        eprintln!(" topology          guitar β₀={} β₁={}   bass β₀={} β₁={}   ukulele β₀={} β₁={}",
            g.beta_0, g.beta_1, b.beta_0, b.beta_1, u.beta_0, u.beta_1);
    }
    eprintln!(" runtime           {:.1}s", report.runtime_seconds);
    eprintln!(" report            {}", report_path.display());
    eprintln!("─────────────────────────────────────────────────────────────");

    Ok(())
}
