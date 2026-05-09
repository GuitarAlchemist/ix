---
date: 2026-05-09
reversibility: two-way door (Phase B re-run + Phase C re-run; no schema published externally)
revisit-trigger: Phase B is re-run on a sample corpus AND Phase C is subsequently run on a larger corpus, OR Phase 1.5 swaps in UMAP/MDS layout (which eliminates the index-based lookup entirely)
status: diagnosed and fixed in PR; Phase B must be re-run on instruments with full corpora to populate normalization in the cluster artifact
---

# Cluster degeneracy: 99.94% of voicings in C0

## Problem

`ix-voicings viz-precompute` (Phase C) produced wildly degenerate cluster assignments in `state/viz/voicing-layout.json`. Observed counts in `state/viz/cluster-assignments.json` (derived by `serve_viz` on 2026-05-02):

| Instrument | Total voicings | In C0 | % in C0 |
|------------|---------------|-------|---------|
| guitar | 667,125 | 666,712 | 99.94% |
| bass | 12,614 | 12,184 | 96.59% |
| ukulele | 8,612 | 8,197 | 95.18% |

The other 12 clusters together held ~1,500 voicings. All downstream visualization was broken: the 3D viewer's "color by cluster" showed one color for 99% of points, and the 2D click-region panel was useless. A runtime "spread" slider was added as a workaround in commit 209c9f4 (2026-05-02) — that is a cosmetic mitigation, not a fix.

## Reproduction

```python
import json

for inst in ['guitar', 'bass', 'ukulele']:
    with open(f'state/voicings/{inst}-clusters.json') as f:
        cl = json.load(f)
    with open(f'state/voicings/{inst}-corpus.json') as f:
        corpus = json.load(f)
    assignments_len = len(cl['assignments'])
    corpus_len = len(corpus)
    out_of_bounds = corpus_len - assignments_len
    c0_from_sample = sum(1 for a in cl['assignments'] if a == 0)
    predicted_c0 = c0_from_sample + out_of_bounds
    print(f"{inst}: corpus={corpus_len}, assignments={assignments_len}, "
          f"OOB→C0={out_of_bounds}, predicted C0={predicted_c0} "
          f"({100*predicted_c0/corpus_len:.2f}%)")
```

With current on-disk artifacts this reproduces the exact counts from the bug report for bass and ukulele (guitar currently has corpus=500 matching assignments=500).

## Root cause

**Index-based fallback to cluster 0 for out-of-bounds corpus rows.**

Phase B's `cluster()` (lib.rs) calls `featurize()` then runs K-Means on the resulting feature matrix. Phase B was invoked with `--export-max 500`, so `{instrument}-features.json` had 500 rows, and `{instrument}-clusters.json` stores exactly 500 assignments.

However, `{instrument}-corpus.json` is the **full enumeration** for bass (12,614 voicings) and ukulele (8,612 voicings). Phase A for these instruments was run without the `--export-max` cap.

Phase C (`viz_precompute.rs:load_and_layout_instrument`, line ~545) assigns clusters by index:

```rust
let local_cluster = cluster_art
    .assignments
    .get(idx)       // None for idx >= 500
    .copied()
    .unwrap_or(0)   // silently defaults to C0
    .min(cluster_art.k.saturating_sub(1));
```

For all corpus voicings with `idx >= 500`, `get(idx)` returns `None` and `unwrap_or(0)` assigns cluster 0. The result exactly matches the observed numbers:

- bass: 70 (sample C0) + 12,114 (OOB) = 12,184 ✓
- ukulele: 85 (sample C0) + 8,112 (OOB) = 8,197 ✓
- guitar: 500 corpus matches 500 assignments → no OOB, balanced result ✓

### Ruled-out candidates

| Candidate | Verdict | Evidence |
|-----------|---------|----------|
| Feature scaling / normalization | **Not the cause** | Numeric columns are z-scored in `zscore_numeric_columns`; one-hot columns are intentionally left raw at 0/1 |
| K-Means++ initialization | **Not the cause** | `KMeans.init_centroids` implements proper distance-weighted centroid seeding |
| Distance metric mismatch | **Not the cause** | Euclidean on z-scored features is appropriate for this mixed numeric+one-hot space |
| Empty-cluster handling | **Not the cause** | With only 500 training points and k=5 the clustering is balanced (14-32% per cluster) |
| Dead features | **Not the cause** | The z-scoring confirms columns have nonzero variance |
| Index-based OOB fallback | **ROOT CAUSE** | Proven by arithmetic: assignments.len()=500 < corpus.len() for bass and ukulele |

## Fix (shipped in this PR)

Two-part change, under 50 lines:

### 1. Store normalization in `ClusterArtifacts` (lib.rs)

Added `#[serde(default)] pub normalization: HashMap<String, Normalization>` to `ClusterArtifacts`. Phase B's `cluster()` now saves `fm.normalization` into the artifact so Phase C has the z-score parameters needed to classify new voicings using the same feature space as the centroids.

### 2. Centroid-based prediction for OOB corpus rows (viz_precompute.rs)

Replaced the `unwrap_or(0)` fallback with a call to the new `predict_cluster()` function when the corpus index is out of bounds and normalization is available:

```rust
let local_cluster = if idx < cluster_art.assignments.len() {
    cluster_art.assignments[idx].min(cluster_art.k.saturating_sub(1))
} else if !cluster_art.normalization.is_empty() {
    crate::predict_cluster(row, &cluster_art.centroids, &cluster_art.normalization)
} else {
    // old artifact, no normalization stored — re-run Phase B to fix
    0
};
```

`predict_cluster()` applies the same z-score transform as Phase B, then finds the nearest centroid by squared Euclidean distance. This is equivalent to what K-Means would assign if it had seen this voicing during training.

### 3. Regression guard tests

Added two tests to `lib.rs`:

- `predict_cluster_lands_in_valid_range`: verifies the function returns a valid cluster index
- `cluster_balance_invariant`: asserts that no cluster holds > 70% of voicings in a well-separated synthetic dataset — this is the invariant the bug violated

## Backward compatibility

- `ClusterArtifacts` uses `#[serde(default)]` so existing `*-clusters.json` files without the `normalization` field still deserialize. The old broken behavior (C0 fallback) triggers only when normalization is absent AND the corpus is larger than the assignments array — the code emits a warning in that case.
- **Action required**: re-run `ix-voicings phase-b --instruments bass,ukulele` to regenerate `*-clusters.json` with normalization populated. Then re-run `ix-voicings viz-precompute` to regenerate `state/viz/`. The guitar corpus currently has 500 voicings matching 500 assignments, so it is not affected until the guitar corpus is regenerated at full scale.

## What would have caught this earlier

The `cluster_balance_invariant` test now in lib.rs. A pre-flight check in `run_viz_precompute` would also work:

```rust
let max_cluster_count = counts.iter().copied().max().unwrap_or(0);
let total = counts.iter().sum::<usize>();
assert!(max_cluster_count * 100 / total.max(1) <= 70,
    "degenerate cluster: {max_cluster_count}/{total} in one bin");
```

Such an assert would have surfaced the bug the first time Phase C ran on a full bass/ukulele corpus.

## Workaround shipped on 2026-05-02 (not a fix)

The "spread" slider in `crates/ix-voicings/web/3d.html` (commit 209c9f4, `<input type="range" id="spread">`) adds triangular noise per axis to make the dense sub-pixel knots readable at runtime. It does not change the underlying cluster assignments. The 3D Prime Radiant viewer also carries the same `default_spread: 1.5` parameter in the `voicings.payload.v1` schema. These workarounds can be kept for aesthetic reasons but are no longer necessary once the cluster assignments are correct.

## Open questions

- Once Phase 1.5 replaces the index-based lookup with UMAP/MDS layout (per `viz_precompute.rs` Phase 1.5 comment), the `assignments` field of `ClusterArtifacts` becomes unused in Phase C entirely — Phase C will compute positions from the embedding directly. At that point the `predict_cluster` path also becomes dead code and can be removed.
- The guitar corpus (500 voicings from export-max) is too small for meaningful clustering. The silhouette score of 0.199 for 500 guitar voicings will likely be different when the full 667K corpus is clustered. K-Means on 667K points is slow; consider a mini-batch or reservoir-sampled approach before locking in the Phase 1.5 design.
