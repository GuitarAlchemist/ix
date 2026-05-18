# ix-supervised Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `Regressor::fit` / `Classifier::fit` return `()`; failure modes (singular matrix, empty input) typically PANIC rather than return Err. Callers MUST pre-validate `x.nrows() > 0` and `x.nrows() == y.len()`.
- `Classifier::predict` and `predict_proba` MUST be called only after `fit`. Calling before `fit` panics (uninitialized state). This is the discipline; consider it a precondition contract.
- `predict_proba` returns probabilities that sum to 1.0 per row within f64 rounding. Number of columns = `*y.iter().max() + 1` (i.e. classes are `0..=max_label`; missing labels in training still occupy a column).
- `metrics::accuracy(y_true, y_pred)` panics on length mismatch (no Result wrapper). Callers must ensure same length.
- `metrics::r_squared` returns `0.0` (not NaN) when `ss_tot < 1e-12` — degenerate constant-target case.
- `metrics::roc_curve` requires binary labels `{0, 1}`. Multi-class labels produce undefined results; no validation.
- `DecisionTree::fit` with `max_depth = 0` produces a single-leaf tree predicting the majority class.
- `KNN::predict` ties broken by lowest-numbered class (deterministic but not configurable).
- `linear_regression::LinearRegression` uses normal-equations (via `ndarray-linalg`-free pseudo-inverse). Singular X^T X panics; callers should pre-check feature rank.

## Concurrency contracts
- Fitted models (`DecisionTree`, `RandomForest` via ix-ensemble, `KNN`, `LinearRegression`) are `Send + Sync`. Safe to share `&model` across threads for `predict`.
- `fit` requires `&mut self`; serialize fits or clone per thread.

## Failure contracts
- This crate panics on contract violation (length mismatch, predict-before-fit, NaN labels). It does NOT use `Result`-typed `Regressor::fit` / `Classifier::fit`.
- `metrics::*` functions are `fn` returning bare scalars; no Result. Empty input or length mismatch panics inside ndarray `mean()`.
- `text::*` (TF-IDF) treats unseen vocabulary at predict time as zero-weight, not panic.
- `resampling::SMOTE` with `k_neighbors >= minority_class_size` panics.

## Determinism contracts
- All seeded algorithms (`KNN` with random tie-break, `RandomForest` via ix-ensemble, SMOTE) use `StdRng::seed_from_u64`. Identical seed + identical input = identical model.
- `DecisionTree` split selection iterates features in `0..n_features` order and thresholds in sorted order; ties broken by lowest feature index.
- `metrics::confusion_matrix` rows = true class, columns = predicted class — opposite convention from some libraries. Diagonal = correct predictions.

## Memory contracts
- `DecisionTree` stores nodes via `Box<Node>` recursion (heap, one allocation per node). Deep trees risk stack overflow on serialization; depth bound = `max_depth` (default unbounded in test harnesses, set explicitly in production).
- `predict_proba` allocates a fresh `Array2<f64>` per call (no buffer reuse).
- `KNN::fit` stores the full training set (no copy elision); memory = O(n * p).
- `text::TfidfVectorizer` builds a `HashMap<String, usize>` vocabulary; vocabulary order is INSERTION-ORDERED (not sorted) — see resampling contract.
