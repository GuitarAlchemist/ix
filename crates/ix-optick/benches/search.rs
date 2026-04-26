//! Criterion benchmark suite for `ix-optick` search performance.
//!
//! These benches measure brute-force cosine search latency across the full
//! OPTIC-K v4 corpus at different instrument filters. They are gated on the
//! `OPTICK_INDEX_PATH` environment variable just like the conformance tests —
//! if unset (CI, fresh clone), every bench exits cleanly without touching
//! Criterion.
//!
//! ## Purpose
//!
//! Answer the question: "when do we need ANN?" by measuring brute-force scan
//! time at production corpus sizes. Each bench uses a query vector with a
//! single spike at dim 0 (STRUCTURE partition) and requests top-10 results.
//!
//! ## Running
//!
//! ```bash
//! # Windows (PowerShell)
//! $env:OPTICK_INDEX_PATH = "C:/Users/spare/source/repos/ga/state/voicings/optick.index"
//! cargo bench -p ix-optick
//!
//! # Unix
//! OPTICK_INDEX_PATH=/path/to/ga/state/voicings/optick.index \
//!   cargo bench -p ix-optick
//! ```
//!
//! ## Benches
//!
//! - `bench_open` — `OptickIndex::open` (mmap + header parse), sub-ms expected
//! - `bench_search_unfiltered_top10` — worst case, scans all ~688k voicings
//! - `bench_search_guitar_top10` — largest partition (~96% of corpus)
//! - `bench_search_bass_top10` — small partition (~2%), shows filter benefit
//! - `bench_search_ukulele_top10` — smallest partition

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ix_optick::OptickIndex;

/// Return the index path if the env var is set AND the file exists.
fn resolve_index_path() -> Option<PathBuf> {
    let raw = std::env::var("OPTICK_INDEX_PATH").ok()?;
    let path = PathBuf::from(raw);
    if !path.exists() {
        eprintln!(
            "ix-optick benches: OPTICK_INDEX_PATH points to missing file: {}",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// Build a 112-dim query with a single spike at dim 0 (STRUCTURE partition).
/// Allocated once outside the bench loop.
fn make_spike_query(dim: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    q
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

fn bench_open(c: &mut Criterion) {
    let Some(path) = resolve_index_path() else {
        eprintln!("ix-optick bench_open: OPTICK_INDEX_PATH not set — skipping");
        return;
    };

    c.bench_function("optick_open", |b| {
        b.iter(|| {
            let idx = OptickIndex::open(black_box(&path)).expect("open index");
            black_box(idx);
        });
    });
}

fn bench_search_unfiltered_top10(c: &mut Criterion) {
    let Some(path) = resolve_index_path() else {
        eprintln!("ix-optick bench_search_unfiltered_top10: OPTICK_INDEX_PATH not set — skipping");
        return;
    };

    let index = OptickIndex::open(&path).expect("open index");
    let dim = index.dimension() as usize;
    let query = make_spike_query(dim);
    let count = index.count();

    let mut group = c.benchmark_group("optick_search_unfiltered");
    group.sample_size(10);
    group.bench_function(format!("top10_n{count}"), |b| {
        b.iter(|| {
            let results = index
                .search(black_box(&query), black_box(None), black_box(10))
                .expect("search");
            black_box(results);
        });
    });
    group.finish();
}

fn bench_search_guitar_top10(c: &mut Criterion) {
    let Some(path) = resolve_index_path() else {
        eprintln!("ix-optick bench_search_guitar_top10: OPTICK_INDEX_PATH not set — skipping");
        return;
    };

    let index = OptickIndex::open(&path).expect("open index");
    let dim = index.dimension() as usize;
    let query = make_spike_query(dim);
    let guitar_count = index.header().instrument_slices[0].count;

    if guitar_count == 0 {
        eprintln!("ix-optick bench_search_guitar_top10: guitar partition empty — skipping");
        return;
    }

    let mut group = c.benchmark_group("optick_search_guitar");
    group.sample_size(10);
    group.bench_function(format!("top10_n{guitar_count}"), |b| {
        b.iter(|| {
            let results = index
                .search(black_box(&query), black_box(Some("guitar")), black_box(10))
                .expect("search");
            black_box(results);
        });
    });
    group.finish();
}

fn bench_search_bass_top10(c: &mut Criterion) {
    let Some(path) = resolve_index_path() else {
        eprintln!("ix-optick bench_search_bass_top10: OPTICK_INDEX_PATH not set — skipping");
        return;
    };

    let index = OptickIndex::open(&path).expect("open index");
    let dim = index.dimension() as usize;
    let query = make_spike_query(dim);
    let bass_count = index.header().instrument_slices[1].count;

    if bass_count == 0 {
        eprintln!("ix-optick bench_search_bass_top10: bass partition empty — skipping");
        return;
    }

    let mut group = c.benchmark_group("optick_search_bass");
    group.sample_size(20);
    group.bench_function(format!("top10_n{bass_count}"), |b| {
        b.iter(|| {
            let results = index
                .search(black_box(&query), black_box(Some("bass")), black_box(10))
                .expect("search");
            black_box(results);
        });
    });
    group.finish();
}

fn bench_search_ukulele_top10(c: &mut Criterion) {
    let Some(path) = resolve_index_path() else {
        eprintln!("ix-optick bench_search_ukulele_top10: OPTICK_INDEX_PATH not set — skipping");
        return;
    };

    let index = OptickIndex::open(&path).expect("open index");
    let dim = index.dimension() as usize;
    let query = make_spike_query(dim);
    let uke_count = index.header().instrument_slices[2].count;

    if uke_count == 0 {
        eprintln!("ix-optick bench_search_ukulele_top10: ukulele partition empty — skipping");
        return;
    }

    let mut group = c.benchmark_group("optick_search_ukulele");
    group.sample_size(20);
    group.bench_function(format!("top10_n{uke_count}"), |b| {
        b.iter(|| {
            let results = index
                .search(black_box(&query), black_box(Some("ukulele")), black_box(10))
                .expect("search");
            black_box(results);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_open,
    bench_search_unfiltered_top10,
    bench_search_guitar_top10,
    bench_search_bass_top10,
    bench_search_ukulele_top10,
);
criterion_main!(benches);
