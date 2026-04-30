//! `tsne-voicings` — project an OPTIC-K voicing index to 2D via t-SNE.
//!
//! Loads `optick.index`, samples N voicings (default 2000 — t-SNE is
//! O(n²) so the full 313K is impractical without Barnes-Hut), runs
//! `ix-manifold::Tsne`, and emits a JSON array suitable for plotting.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p ix-voicings --bin tsne-voicings -- \
//!     --index ../ga/state/voicings/optick.index \
//!     --output state/viz/voicings-tsne.json \
//!     --sample 2000 \
//!     --perplexity 30 \
//!     --iterations 1000 \
//!     --seed 42
//! ```
//!
//! ## Output schema
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "perplexity": 30.0,
//!   "iterations": 1000,
//!   "seed": 42,
//!   "n_sampled": 2000,
//!   "n_total": 313487,
//!   "dim": 124,
//!   "points": [
//!     {"id": 0, "instrument": "guitar", "x": 0.123, "y": -0.456},
//!     ...
//!   ]
//! }
//! ```

use std::path::PathBuf;

use clap::Parser;
use ix_manifold::Tsne;
use ix_optick::OptickIndex;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Project OPTIC-K voicings to 2D via t-SNE")]
struct Cli {
    /// Path to optick.index (v4-pp-r format).
    #[arg(long)]
    index: PathBuf,

    /// Output JSON path.
    #[arg(long, default_value = "state/viz/voicings-tsne.json")]
    output: PathBuf,

    /// Number of voicings to sample (t-SNE is O(n²); 2000 finishes in
    /// under a minute on a laptop, 5000 in ~5 minutes).
    #[arg(long, default_value_t = 2000)]
    sample: usize,

    /// Perplexity. Default 30 matches scikit-learn.
    #[arg(long, default_value_t = 30.0)]
    perplexity: f64,

    /// Number of optimization iterations.
    #[arg(long, default_value_t = 1000)]
    iterations: usize,

    /// Deterministic seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Serialize)]
struct Point {
    id: usize,
    instrument: String,
    x: f64,
    y: f64,
}

#[derive(Serialize)]
struct Output {
    schema_version: u32,
    perplexity: f64,
    iterations: usize,
    seed: u64,
    n_sampled: usize,
    n_total: u64,
    dim: u32,
    points: Vec<Point>,
}

fn main() {
    let cli = Cli::parse();
    let started = std::time::Instant::now();

    let index = OptickIndex::open(&cli.index).unwrap_or_else(|e| {
        eprintln!("error: cannot open index {:?}: {e}", cli.index);
        std::process::exit(2);
    });
    let total = index.count() as usize;
    let dim = index.dimension() as usize;
    eprintln!(
        "loaded index: count={total}, dim={dim}, sample={}",
        cli.sample
    );

    if total < 2 {
        eprintln!("error: index has < 2 voicings, nothing to project");
        std::process::exit(2);
    }
    let sample_n = cli.sample.min(total);

    // Deterministic random sample of voicing indices.
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let mut all: Vec<usize> = (0..total).collect();
    all.shuffle(&mut rng);
    all.truncate(sample_n);
    all.sort_unstable(); // stable order for output

    // Build the (sample_n, dim) f64 matrix.
    let vectors = index.vectors();
    let mut x = Array2::<f64>::zeros((sample_n, dim));
    for (row, &voicing_idx) in all.iter().enumerate() {
        let base = voicing_idx * dim;
        for col in 0..dim {
            x[[row, col]] = vectors[base + col] as f64;
        }
    }

    // Project.
    eprintln!(
        "running t-SNE: perplexity={}, iters={}, seed={}",
        cli.perplexity, cli.iterations, cli.seed
    );
    let y = Tsne::new()
        .with_perplexity(cli.perplexity)
        .with_n_iter(cli.iterations)
        .with_seed(cli.seed)
        .fit_transform(x.view());

    // Build output.
    let points: Vec<Point> = all
        .iter()
        .enumerate()
        .map(|(row, &voicing_idx)| Point {
            id: voicing_idx,
            instrument: index
                .instrument_of(voicing_idx)
                .unwrap_or("unknown")
                .to_string(),
            x: y[[row, 0]],
            y: y[[row, 1]],
        })
        .collect();

    let out = Output {
        schema_version: 1,
        perplexity: cli.perplexity,
        iterations: cli.iterations,
        seed: cli.seed,
        n_sampled: sample_n,
        n_total: total as u64,
        dim: dim as u32,
        points,
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("error: cannot create output directory: {e}");
            std::process::exit(2);
        });
    }
    let json = serde_json::to_string_pretty(&out).expect("serialize");
    std::fs::write(&cli.output, json).unwrap_or_else(|e| {
        eprintln!("error: cannot write {:?}: {e}", cli.output);
        std::process::exit(2);
    });

    eprintln!(
        "wrote {} points to {} in {:.1}s",
        out.n_sampled,
        cli.output.display(),
        started.elapsed().as_secs_f64()
    );
}
