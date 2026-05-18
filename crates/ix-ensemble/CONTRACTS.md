# ix-ensemble Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `RandomForest::fit` infers `n_classes` from `*y.iter().max() + 1`. Missing classes in training data still claim probability columns. Empty `y` panics.
- `RandomForest::fit` default `max_features` is `ceil(sqrt(n_features))` clamped to `n_features`. Override via `with_max_features` is silently clamped to `n_features` — does not error.
- `RandomForest::predict_proba` returns rows summing to 1.0 within f64 rounding; each entry is the fraction of trees voting that class.
- `RandomForest::predict` breaks ties (equal vote counts) by LOWEST class index — deterministic but not configurable.
- Bootstrap samples are drawn WITH replacement (`rng.random_range(0..n)` per row), n samples per tree. Out-of-bag indices are NOT exposed; OOB error is not computed.
- `GradientBoostedClassifier::fit` initializes with class-prior log-odds (not zeros); the first tree fits residuals against that prior.
- `GradientBoostedClassifier` uses softmax over accumulated scores. Multiclass treats it as one-vs-rest with shared residual update per round.
- `EnsembleClassifier` trait (this crate's own trait) is NOT interchangeable with `ix_supervised::traits::Classifier` even though method signatures match — they are distinct trait identities.

## Concurrency contracts
- Trees are fitted SEQUENTIALLY inside `RandomForest::fit` — no internal rayon. Parallelizing must happen at the caller and requires per-tree RNG split.
- Fitted ensembles are `Send + Sync`. `predict` is read-only and safe to share `&self`.

## Failure contracts
- `fit` panics on empty input, label-domain gaps that cause `Array1::from_iter` shape mismatches, or `max_features = 0`.
- `predict` before `fit` panics — the trees `Vec` is empty and `unwrap` fires in `partial_cmp`.
- No `Result` returns anywhere on this surface.

## Determinism contracts
- `RandomForest` is fully deterministic given `seed`. Default seed = 42. Identical seed + identical (x, y) produces bit-identical trees and predictions.
- Feature subset selection uses Fisher-Yates partial shuffle; order of selected features is RNG-dependent but reproducible.
- Bootstrap sample order is RNG-dependent but reproducible; trees built from `sample_indices` in row order.
- `GradientBoostedClassifier` is deterministic (no random sub-sampling); pure function of (x, y, n_rounds, learning_rate, max_depth).

## Memory contracts
- Each tree stores `(DecisionTree, Vec<usize>)` — feature_indices vector is heap-allocated per tree. Memory = O(n_trees * max_features).
- `fit` materializes a fresh `sub_x: Array2` of shape `(n, max_features)` per tree (no zero-copy view) — peak memory during fit is `O(n * max_features)` on top of the original `x`.
- `predict_proba` allocates a fresh `Array2<f64>` per call.
