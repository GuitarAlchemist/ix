# ix-unsupervised Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `Clusterer::predict` MUST be called after `fit`; calling first panics (`expect("Model not fitted")` in DBSCAN; uninitialized centroids in KMeans).
- `DBSCAN` labels: `0` = NOISE, clusters start at `1`. Do NOT collapse to "cluster 0 is first cluster" — this convention is load-bearing for callers using `label == 0` as a noise check.
- `DBSCAN::predict` on new points uses NEAREST-CORE-POINT assignment from the fitted data, not re-clustering. New points may receive `NOISE` if no core point is within `eps`.
- `KMeans::fit` uses K-Means++ initialization (NOT random init). Centroid count = `self.k`; the result has exactly `k` clusters or panics if `n_samples < k`.
- `KMeans::save_state` returns `None` until `fit` has been called. `load_state` reconstructs a fitted model whose `predict` works without re-running `fit`.
- `KMeans::load_state` panics (`expect("KMeansState centroids dimensions mismatch")`) if the saved state's `centroids[i].len()` is inconsistent — defensive but not Result-typed.
- `PCA::transform` projects onto the top-k components by eigenvalue magnitude (descending). Sign of components is NOT canonicalized — sign flips across runs are valid even with the same input.
- `tsne` is Barnes-Hut (theta-approximated); output coordinates are NOT comparable across runs even with the same seed unless `theta = 0.0` (exact).
- `GMM::fit` is EM with k-means init; converges when log-likelihood delta < `tol` or `max_iterations` reached. `predict_proba` rows sum to 1.0 within f64 rounding.

## Concurrency contracts
- Fitted models are `Send + Sync`. `predict` is `&self` and thread-safe.
- `fit` requires `&mut self`. No internal parallelism in DBSCAN (O(n^2) region queries) — for large n, callers should batch externally.

## Failure contracts
- All `Clusterer` / `DimensionReducer` methods are infallible by signature; failure paths panic via `expect`.
- `DBSCAN` with `min_points = 0` produces a single cluster (every point qualifies). With `min_points > n` produces all noise.
- `KMeans` with `k = 0` panics at array construction. With `k > n_samples` produces empty clusters at the tail; predict still returns indices in `0..k`.

## Determinism contracts
- `KMeans` is deterministic given `seed`. Default seed = 42. K-Means++ probability-weighted picks use `StdRng` seeded once.
- `DBSCAN` is fully deterministic (no RNG). Cluster ID assignment depends on row-iteration order — reordering rows in `x` changes which cluster gets ID 1 vs 2.
- `GMM`, `tsne` use seeded RNG; identical seeds + input = identical output.
- DBSCAN border points: a point reachable from multiple clusters gets assigned to the FIRST cluster that reaches it (row-iteration order). This is a stable but not order-independent contract.

## Memory contracts
- `DBSCAN::fit` stores a clone of the training data (`fitted_data: Some(x.to_owned())`) for predict; memory = O(n * p) post-fit.
- `KMeans` stores only the `(k, p)` centroid matrix — no training-set retention.
- `PCA` stores `(n_components, n_features)` components matrix and `(n_features,)` mean.
- `tsne` allocates a `(n, n)` affinity matrix during fit — memory scales QUADRATICALLY in input size. Not suitable for n > ~10k.
