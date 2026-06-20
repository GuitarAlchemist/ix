//! Batch 1 — canary migration of 6 MCP tools to the capability registry.
//!
//! These wrappers delegate to the pre-existing `handlers::*` functions to
//! preserve exact MCP behavior; only the *registration surface* changes.
//! Each wrapper carries its hand-written JSON schema via `schema_fn = ...`.

use crate::handlers;
use crate::schema::{object, output, Prop};
use ix_skill_macros::ix_skill;
use serde_json::Value;

// --- ix_stats --------------------------------------------------------------

fn stats_schema() -> Value {
    object(
        vec![(
            "data",
            Prop::num_array().desc("List of numbers to compute statistics on"),
        )],
        &["data"],
    )
}

/// Compute statistics (mean, std, min, max, median) on a list of numbers.
#[ix_skill(
    domain = "math",
    name = "stats",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::stats_schema"
)]
pub fn stats(params: Value) -> Result<Value, String> {
    handlers::stats(params)
}

// --- ix_distance -----------------------------------------------------------

fn distance_schema() -> Value {
    object(
        vec![
            ("a", Prop::num_array().desc("First vector")),
            ("b", Prop::num_array().desc("Second vector")),
            (
                "metric",
                Prop::string()
                    .enum_of(&["euclidean", "cosine", "manhattan"])
                    .desc("Distance metric"),
            ),
        ],
        &["a", "b", "metric"],
    )
}

/// Compute distance between two vectors (euclidean, cosine, or manhattan).
#[ix_skill(
    domain = "math",
    name = "distance",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::distance_schema"
)]
pub fn distance(params: Value) -> Result<Value, String> {
    handlers::distance(params)
}

// --- ix_fft ----------------------------------------------------------------

fn fft_schema() -> Value {
    object(
        vec![
            (
                "signal",
                Prop::num_array().desc("Real-valued input signal (length must be power of 2)"),
            ),
            (
                "inverse",
                Prop::boolean()
                    .default(false)
                    .desc("If true, compute the inverse FFT"),
            ),
        ],
        &["signal"],
    )
}

/// Compute the Fast Fourier Transform of a real-valued signal.
#[ix_skill(
    domain = "signal",
    name = "fft",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::fft_schema"
)]
pub fn fft(params: Value) -> Result<Value, String> {
    handlers::fft(params)
}

// --- ix_kmeans -------------------------------------------------------------

fn kmeans_schema() -> Value {
    object(
        vec![
            (
                "data",
                Prop::num_matrix().desc("Matrix where each row is a data point"),
            ),
            ("k", Prop::integer().minimum(1).desc("Number of clusters")),
            (
                "max_iter",
                Prop::integer().default(100).desc("Max iterations"),
            ),
            ("seed", Prop::integer().default(42).desc("RNG seed")),
        ],
        &["data", "k"],
    )
}

fn kmeans_output_schema() -> Value {
    output(vec![
        (
            "labels",
            Prop::int_array().desc("Cluster index assigned to each input row"),
        ),
        (
            "centroids",
            Prop::num_matrix().desc("The k cluster-center vectors"),
        ),
        (
            "inertia",
            Prop::number().desc("Sum of squared distances to centroids"),
        ),
        ("k", Prop::integer().desc("Number of clusters")),
    ])
}

/// Cluster points using K-Means.
#[ix_skill(
    domain = "unsupervised",
    name = "kmeans",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::kmeans_schema",
    output_schema_fn = "crate::skills::batch1::kmeans_output_schema"
)]
pub fn kmeans(params: Value) -> Result<Value, String> {
    handlers::kmeans(params)
}

// --- ix_pca ----------------------------------------------------------------

fn pca_schema() -> Value {
    object(
        vec![
            (
                "data",
                Prop::num_matrix()
                    .desc("Matrix where each row is an observation and each column a feature"),
            ),
            (
                "n_components",
                Prop::integer()
                    .minimum(1)
                    .desc("Number of principal components to keep"),
            ),
        ],
        &["data", "n_components"],
    )
}

fn pca_output_schema() -> Value {
    output(vec![
        (
            "transformed",
            Prop::num_matrix()
                .desc("Input projected onto the top n_components axes (one row per observation)"),
        ),
        (
            "explained_variance_ratio",
            Prop::num_array().desc("Fraction of total variance carried by each component"),
        ),
        (
            "components",
            Prop::num_matrix().desc("The principal-axis vectors (n_components rows)"),
        ),
        (
            "n_components",
            Prop::integer().desc("Number of components kept"),
        ),
    ])
}

/// Reduce dimensionality with Principal Component Analysis: project the data
/// onto its top `n_components` principal axes and report the explained-variance
/// ratio per component.
#[ix_skill(
    domain = "unsupervised",
    name = "pca",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::pca_schema",
    output_schema_fn = "crate::skills::batch1::pca_output_schema"
)]
pub fn pca(params: Value) -> Result<Value, String> {
    handlers::pca(params)
}

// --- ix_dbscan -------------------------------------------------------------

fn dbscan_schema() -> Value {
    object(
        vec![
            ("data", Prop::num_matrix().desc("Matrix where each row is a data point")),
            (
                "eps",
                Prop::number().exclusive_min(0).desc(
                    "Neighborhood radius: two points are neighbors when within this Euclidean distance",
                ),
            ),
            (
                "min_points",
                Prop::integer().minimum(1).default(4).desc(
                    "Minimum neighbors (including the point itself) for a core point; defaults to 4 when omitted",
                ),
            ),
        ],
        &["data", "eps"],
    )
}

fn dbscan_output_schema() -> Value {
    output(vec![
        (
            "labels",
            Prop::int_array().desc("Cluster id per input row; 0 = noise, clusters numbered from 1"),
        ),
        (
            "n_clusters",
            Prop::integer().desc("Number of clusters found (noise excluded)"),
        ),
        (
            "n_noise",
            Prop::integer().desc("Number of points labeled noise (label 0)"),
        ),
        ("eps", Prop::number().desc("Neighborhood radius used")),
        (
            "min_points",
            Prop::integer().desc("Core-point threshold used"),
        ),
    ])
}

/// Cluster points by density with DBSCAN. Unlike k-means there is no preset
/// cluster count: dense regions become clusters (numbered from 1) and points in
/// sparse regions are labeled noise (0).
#[ix_skill(
    domain = "unsupervised",
    name = "dbscan",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::dbscan_schema",
    output_schema_fn = "crate::skills::batch1::dbscan_output_schema"
)]
pub fn dbscan(params: Value) -> Result<Value, String> {
    handlers::dbscan(params)
}

// --- ix_eigen --------------------------------------------------------------

fn eigen_schema() -> Value {
    object(
        vec![(
            "matrix",
            Prop::num_matrix().desc("A real SYMMETRIC square matrix (row-major)"),
        )],
        &["matrix"],
    )
}

fn eigen_output_schema() -> Value {
    output(vec![
        (
            "eigenvalues",
            Prop::num_array().desc("Eigenvalues in descending order"),
        ),
        (
            "eigenvectors",
            Prop::num_matrix()
                .desc("Unit eigenvectors; eigenvectors[k] corresponds to eigenvalues[k]"),
        ),
        ("n", Prop::integer().desc("Matrix dimension")),
    ])
}

/// Eigendecomposition of a real symmetric matrix via cyclic Jacobi rotations:
/// returns eigenvalues in descending order and the matching unit eigenvectors.
#[ix_skill(
    domain = "math",
    name = "eigen",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::eigen_schema",
    output_schema_fn = "crate::skills::batch1::eigen_output_schema"
)]
pub fn eigen(params: Value) -> Result<Value, String> {
    handlers::eigen(params)
}

// --- ix_silhouette ---------------------------------------------------------

fn silhouette_schema() -> Value {
    object(
        vec![
            (
                "data",
                Prop::num_matrix().desc("Matrix where each row is the point that was clustered"),
            ),
            (
                "labels",
                Prop::int_array().item_minimum(0).desc(
                    "Cluster id per row — e.g. wire the `labels` output of kmeans/dbscan via {from: \"<stage>.labels\"}",
                ),
            ),
        ],
        &["data", "labels"],
    )
}

fn silhouette_output_schema() -> Value {
    output(vec![
        (
            "score",
            Prop::number().desc("Mean silhouette coefficient in [-1,1]; higher = denser, better-separated clusters. 0.0 when undefined (<2 samples or <2 clusters)"),
        ),
        ("n_clusters", Prop::integer().desc("Number of distinct cluster ids")),
        ("n_samples", Prop::integer().desc("Number of rows scored")),
    ])
}

/// Silhouette score: evaluate a clustering's quality from the data and its
/// cluster labels (e.g. the output of kmeans/dbscan). Mean of (b−a)/max(a,b)
/// over all points; higher means denser, better-separated clusters.
#[ix_skill(
    domain = "unsupervised",
    name = "silhouette",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::silhouette_schema",
    output_schema_fn = "crate::skills::batch1::silhouette_output_schema"
)]
pub fn silhouette(params: Value) -> Result<Value, String> {
    handlers::silhouette(params)
}

// --- ix_feature_importances ------------------------------------------------

fn feature_importances_schema() -> Value {
    object(
        vec![
            (
                "data",
                Prop::num_matrix().desc("Feature matrix (each row an observation)"),
            ),
            (
                "labels",
                Prop::int_array()
                    .item_minimum(0)
                    .desc("Class label per row"),
            ),
            (
                "n_trees",
                Prop::integer()
                    .minimum(1)
                    .default(100)
                    .desc("Trees in the random forest fitted to score importances"),
            ),
            (
                "max_depth",
                Prop::integer()
                    .minimum(1)
                    .default(8)
                    .desc("Maximum tree depth"),
            ),
            (
                "n_repeats",
                Prop::integer()
                    .minimum(1)
                    .default(5)
                    .desc("Permutation repeats per feature (averaged)"),
            ),
            (
                "seed",
                Prop::integer()
                    .minimum(0)
                    .default(42)
                    .desc("RNG seed — makes the importances reproducible"),
            ),
        ],
        &["data", "labels"],
    )
}

fn feature_importances_output_schema() -> Value {
    output(vec![
        (
            "importances",
            Prop::num_array().desc("Permutation importance per feature: the mean drop in accuracy when that feature's column is scrambled"),
        ),
        ("ranking", Prop::int_array().desc("Feature indices ordered most→least important")),
        ("method", Prop::string().desc("Always 'permutation' (model-agnostic; not Gini/MDI)")),
        ("n_features", Prop::integer().desc("Number of feature columns")),
    ])
}

/// Feature importances: fit a random forest to (data, labels) and report each
/// feature's permutation importance — the mean drop in accuracy when that
/// feature's column is randomly scrambled. Model-agnostic; reproducible given
/// `seed`.
#[ix_skill(
    domain = "ensemble",
    name = "feature_importances",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::feature_importances_schema",
    output_schema_fn = "crate::skills::batch1::feature_importances_output_schema"
)]
pub fn feature_importances(params: Value) -> Result<Value, String> {
    handlers::feature_importances(params)
}

// --- ix_svd ----------------------------------------------------------------

fn svd_schema() -> Value {
    object(
        vec![(
            "matrix",
            Prop::num_matrix().desc("A real m×n matrix (row-major); m and n may differ"),
        )],
        &["matrix"],
    )
}

fn svd_output_schema() -> Value {
    output(vec![
        (
            "u",
            Prop::num_matrix().desc("Left singular vectors as columns (m×k)"),
        ),
        (
            "singular_values",
            Prop::num_array().desc("Singular values in descending order (length k)"),
        ),
        (
            "v",
            Prop::num_matrix().desc("Right singular vectors as columns (n×k)"),
        ),
        (
            "rank",
            Prop::integer()
                .desc("Numerical rank (singular values above a magnitude-relative tolerance)"),
        ),
    ])
}

/// Singular Value Decomposition: factor a real matrix into U·diag(s)·Vᵀ,
/// returning the singular values (descending) and the left/right singular
/// vectors. Use for low-rank structure, pseudo-inverse, or rank estimation.
#[ix_skill(
    domain = "math",
    name = "svd",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::svd_schema",
    output_schema_fn = "crate::skills::batch1::svd_output_schema"
)]
pub fn svd(params: Value) -> Result<Value, String> {
    handlers::svd(params)
}

// --- ix_gmm ----------------------------------------------------------------

fn gmm_schema() -> Value {
    object(
        vec![
            (
                "data",
                Prop::num_matrix().desc("Matrix where each row is a data point"),
            ),
            (
                "k",
                Prop::integer()
                    .minimum(1)
                    .desc("Number of mixture components"),
            ),
            (
                "max_iter",
                Prop::integer()
                    .minimum(1)
                    .default(100)
                    .desc("Max EM iterations"),
            ),
            (
                "seed",
                Prop::integer()
                    .minimum(0)
                    .default(42)
                    .desc("RNG seed (reproducible init)"),
            ),
        ],
        &["data", "k"],
    )
}

fn gmm_output_schema() -> Value {
    output(vec![
        (
            "labels",
            Prop::int_array().desc("Hard cluster id per row (argmax responsibility)"),
        ),
        ("means", Prop::num_matrix().desc("Component means (k×d)")),
        ("weights", Prop::num_array().desc("Mixture weights (k)")),
        (
            "covariances",
            Prop::num_matrix().desc("Diagonal covariances per component (k×d)"),
        ),
        (
            "responsibilities",
            Prop::num_matrix().desc("Soft assignment P(component|point) (n×k); each row sums to 1"),
        ),
        ("k", Prop::integer().desc("Number of components")),
    ])
}

/// Gaussian Mixture Model (diagonal covariance, EM): soft clustering that, unlike
/// k-means, returns per-point membership probabilities plus per-component means,
/// weights, and covariances.
#[ix_skill(
    domain = "unsupervised",
    name = "gmm",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::gmm_schema",
    output_schema_fn = "crate::skills::batch1::gmm_output_schema"
)]
pub fn gmm(params: Value) -> Result<Value, String> {
    handlers::gmm(params)
}

// --- ix_wavelet_denoise ----------------------------------------------------

fn wavelet_denoise_schema() -> Value {
    object(
        vec![
            (
                "signal",
                Prop::num_array().desc("Real-valued input signal; length must be divisible by 2^levels (Haar DWT round-trips exactly)"),
            ),
            ("levels", Prop::integer().minimum(1).default(3).desc("Number of Haar DWT decomposition levels")),
            (
                "threshold",
                Prop::number().minimum(0).default(0.1).desc("Soft-threshold applied to detail coefficients (in signal units)"),
            ),
        ],
        &["signal"],
    )
}

fn wavelet_denoise_output_schema() -> Value {
    output(vec![
        (
            "denoised",
            Prop::num_array().desc("Denoised signal, same length as the input"),
        ),
        ("levels", Prop::integer()),
        ("threshold", Prop::number()),
    ])
}

/// Wavelet denoising: Haar DWT → soft-threshold the detail coefficients →
/// inverse DWT. Removes high-frequency noise while preserving signal structure.
#[ix_skill(
    domain = "signal",
    name = "wavelet_denoise",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::wavelet_denoise_schema",
    output_schema_fn = "crate::skills::batch1::wavelet_denoise_output_schema"
)]
pub fn wavelet_denoise(params: Value) -> Result<Value, String> {
    handlers::wavelet_denoise(params)
}

// --- ix_fir_filter ---------------------------------------------------------

fn fir_filter_schema() -> Value {
    object(
        vec![
            ("signal", Prop::num_array().desc("Real-valued input signal")),
            (
                "kind",
                Prop::string()
                    .enum_of(&["lowpass", "highpass", "bandpass"])
                    .desc("Filter type"),
            ),
            (
                "cutoff",
                Prop::number().exclusive_min(0).exclusive_max(0.5).desc(
                    "Normalized cutoff frequency (fraction of sample rate) for lowpass/highpass",
                ),
            ),
            (
                "low_cutoff",
                Prop::number()
                    .exclusive_min(0)
                    .exclusive_max(0.5)
                    .desc("Lower normalized cutoff (bandpass)"),
            ),
            (
                "high_cutoff",
                Prop::number()
                    .exclusive_min(0)
                    .exclusive_max(0.5)
                    .desc("Upper normalized cutoff (bandpass)"),
            ),
            (
                "order",
                Prop::integer()
                    .minimum(1)
                    .default(32)
                    .desc("Filter order (number of taps ≈ order+1)"),
            ),
        ],
        &["signal", "kind"],
    )
}

fn fir_filter_output_schema() -> Value {
    output(vec![
        ("filtered", Prop::num_array().desc("Filtered signal")),
        ("kind", Prop::string()),
        ("order", Prop::integer()),
    ])
}

/// FIR filter (windowed-sinc): low-pass, high-pass, or band-pass a signal at a
/// normalized cutoff frequency.
#[ix_skill(
    domain = "signal",
    name = "fir_filter",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::fir_filter_schema",
    output_schema_fn = "crate::skills::batch1::fir_filter_output_schema"
)]
pub fn fir_filter(params: Value) -> Result<Value, String> {
    handlers::fir_filter(params)
}

// --- ix_spectrogram --------------------------------------------------------

fn spectrogram_schema() -> Value {
    object(
        vec![
            ("signal", Prop::num_array().desc("Real-valued input signal")),
            (
                "window_size",
                Prop::integer()
                    .minimum(2)
                    .desc("STFT window length (samples); MUST be a power of two (radix-2 FFT)"),
            ),
            (
                "hop_size",
                Prop::integer()
                    .minimum(1)
                    .desc("Step between frames (samples); defaults to window_size/2"),
            ),
            (
                "db",
                Prop::boolean()
                    .default(false)
                    .desc("Return magnitudes in decibels"),
            ),
        ],
        &["signal", "window_size"],
    )
}

fn spectrogram_output_schema() -> Value {
    output(vec![
        (
            "spectrogram",
            Prop::num_matrix().desc("Magnitude matrix: n_frames rows × n_bins cols"),
        ),
        ("n_frames", Prop::integer().desc("Number of time frames")),
        (
            "n_bins",
            Prop::integer().desc("Frequency bins = window_size/2 + 1"),
        ),
        ("window_size", Prop::integer()),
        ("hop_size", Prop::integer()),
        ("db", Prop::boolean()),
    ])
}

/// Spectrogram (STFT magnitude): time–frequency representation of a signal via a
/// sliding Hann-windowed FFT. Shows how spectral content evolves over time.
#[ix_skill(
    domain = "signal",
    name = "spectrogram",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::spectrogram_schema",
    output_schema_fn = "crate::skills::batch1::spectrogram_output_schema"
)]
pub fn spectrogram(params: Value) -> Result<Value, String> {
    handlers::spectrogram(params)
}

// --- ix_autocorrelation ----------------------------------------------------

fn autocorrelation_schema() -> Value {
    object(
        vec![("signal", Prop::num_array().desc("Real-valued input series"))],
        &["signal"],
    )
}

fn autocorrelation_output_schema() -> Value {
    output(vec![
        (
            "autocorrelation",
            Prop::num_array().desc("One-sided normalized ACF, lag 0..n-1; acf[0]=1.0. A peak at lag L indicates periodicity with period L"),
        ),
        ("n_lags", Prop::integer().desc("Number of lags returned (= signal length)")),
    ])
}

/// Autocorrelation: normalized self-similarity of a series at each lag. A peak at
/// lag L reveals periodicity/seasonality with period L.
#[ix_skill(
    domain = "signal",
    name = "autocorrelation",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::autocorrelation_schema",
    output_schema_fn = "crate::skills::batch1::autocorrelation_output_schema"
)]
pub fn autocorrelation(params: Value) -> Result<Value, String> {
    handlers::autocorrelation(params)
}

// --- ix_linear_regression --------------------------------------------------

fn linear_regression_schema() -> Value {
    object(
        vec![
            (
                "X",
                Prop::num_matrix().desc("Feature matrix (each row an observation)"),
            ),
            ("y", Prop::num_array().desc("Target vector")),
        ],
        &["X", "y"],
    )
}

/// Fit an ordinary least-squares linear regression model.
#[ix_skill(
    domain = "supervised",
    name = "linear_regression",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::linear_regression_schema"
)]
pub fn linear_regression(params: Value) -> Result<Value, String> {
    handlers::linear_regression(params)
}

// --- ix_governance_belief --------------------------------------------------

fn governance_belief_schema() -> Value {
    object(
        vec![
            (
                "proposition",
                Prop::string().desc("Belief proposition text"),
            ),
            (
                "truth_value",
                Prop::string().enum_of(&["T", "F", "U", "C"]).desc(
                    "Tetravalent truth (legacy). Hexavalent values P/D supported when present.",
                ),
            ),
            ("confidence", Prop::number().minimum(0.0).maximum(1.0)),
            ("supporting", Prop::obj_array()),
            ("contradicting", Prop::obj_array()),
        ],
        &["proposition", "truth_value", "confidence"],
    )
}

/// Query the Demerzel belief engine with a belief state and receive a
/// resolved action recommendation.
#[ix_skill(
    domain = "governance",
    name = "governance.belief",
    governance = "safety,reversible",
    schema_fn = "crate::skills::batch1::governance_belief_schema"
)]
pub fn governance_belief(params: Value) -> Result<Value, String> {
    handlers::governance_belief(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Executes-test for the new `pca` skill: the catalog entry must EXECUTE,
    // not merely register (no green-but-dead). Classic Lindsay-Smith 2D PCA
    // example — its first principal component carries ~96% of the variance.
    #[test]
    fn pca_skill_executes_and_reduces_dimensions() {
        let out = pca(json!({
            "data": [
                [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]
            ],
            "n_components": 1
        }))
        .expect("pca skill runs");

        let transformed = out["transformed"].as_array().expect("transformed array");
        assert_eq!(
            transformed.len(),
            10,
            "one projected row per input observation"
        );
        assert_eq!(
            transformed[0].as_array().unwrap().len(),
            1,
            "reduced from 2 features to 1 component"
        );

        let evr = out["explained_variance_ratio"].as_array().unwrap();
        assert_eq!(evr.len(), 1);
        assert!(
            evr[0].as_f64().unwrap() > 0.9,
            "the single retained PC should explain most of the variance, got {}",
            evr[0].as_f64().unwrap()
        );
    }

    // The skill must be registered AND pipeline-callable (arity-1), so the
    // NL→pipeline proposer can actually see it (the dogfood gap this closes).
    #[test]
    fn pca_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "pca")
            .expect("pca skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // Honest boundary: more components than features is rejected, not silently
    // clamped — the proposer/caller gets a clear error instead of a wrong shape.
    #[test]
    fn pca_rejects_too_many_components() {
        let err = pca(json!({ "data": [[1.0, 2.0], [3.0, 4.0]], "n_components": 5 })).unwrap_err();
        assert!(err.contains("exceeds the feature count"), "got: {err}");
    }

    // Executes-test for the new `dbscan` skill: two well-separated dense blobs
    // plus a far outlier → exactly two clusters and one noise point. The catalog
    // entry must EXECUTE, not merely register (no green-but-dead).
    #[test]
    fn dbscan_skill_executes_and_separates_clusters_from_noise() {
        let out = dbscan(json!({
            "data": [
                [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
                [5.0, 5.0], [5.1, 5.1], [5.2, 5.0], [5.0, 5.2],
                [50.0, 50.0]
            ],
            "eps": 0.5,
            "min_points": 2
        }))
        .expect("dbscan skill runs");

        assert_eq!(out["n_clusters"].as_u64().unwrap(), 2, "two dense blobs");
        assert_eq!(
            out["n_noise"].as_u64().unwrap(),
            1,
            "the far point is noise"
        );
        let labels = out["labels"].as_array().unwrap();
        assert_eq!(labels.len(), 9, "one label per input row");
        assert_eq!(
            labels[8].as_u64().unwrap(),
            0,
            "the outlier is labeled noise"
        );
        assert_eq!(labels[0], labels[1], "the first blob shares a cluster");
        assert_ne!(labels[0], labels[4], "the two blobs are different clusters");
    }

    // The skill must be registered AND pipeline-callable (arity-1), so the
    // NL→pipeline proposer can actually see it.
    #[test]
    fn dbscan_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "dbscan")
            .expect("dbscan skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // Honest boundary: `eps` is required — there is no data-blind default radius,
    // so an omitted `eps` returns a clear error instead of a guessed clustering.
    #[test]
    fn dbscan_requires_eps() {
        let err = dbscan(json!({ "data": [[0.0, 0.0], [1.0, 1.0]] })).unwrap_err();
        assert!(err.contains("eps"), "got: {err}");
    }

    // The canonical-label fix must propagate through the SKILL, not just the
    // crate: a point density-unreachable from any core (within eps of a border
    // point only) is reported as noise, and n_noise counts it. With the default
    // Clusterer fit_predict (fit+predict) this point would be absorbed and
    // n_noise would read 0.
    #[test]
    fn dbscan_skill_labels_border_adjacent_point_as_noise() {
        let out = dbscan(json!({
            "data": [
                [0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.2, 0.2],
                [1.3, 0.0],
                [2.2, 0.0]
            ],
            "eps": 1.0,
            "min_points": 4
        }))
        .expect("dbscan skill runs");
        let labels = out["labels"].as_array().unwrap();
        assert_eq!(
            labels[5].as_u64().unwrap(),
            0,
            "unreachable point must be noise"
        );
        assert_eq!(out["n_noise"].as_u64().unwrap(), 1);
        assert_eq!(out["n_clusters"].as_u64().unwrap(), 1);
    }

    // Honest boundary: min_points < 1 is rejected — under it nothing can be noise,
    // contradicting the advertised contract.
    #[test]
    fn dbscan_rejects_min_points_below_one() {
        let err = dbscan(json!({ "data": [[0.0, 0.0]], "eps": 1.0, "min_points": 0 })).unwrap_err();
        assert!(err.contains("min_points"), "got: {err}");
    }

    // Executes-test for `eigen`. Uses a 3x3 with DISTINCT eigenvalues, whose
    // eigenvector matrix is NOT component-symmetric — so this test discriminates
    // a wrong row/column transpose (a 2x2 like [[2,1],[1,2]] has V == V^T, so a
    // missing transpose would pass identically — the over-claim a review caught).
    // Asserts A·v_k = λ_k·v_k for EVERY k (binds both the per-index alignment and
    // the transpose), plus descending order and unit norm.
    #[test]
    fn eigen_skill_executes_on_symmetric_matrix() {
        let m = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]];
        let out = eigen(json!({ "matrix": m })).expect("eigen runs");

        let vals: Vec<f64> = out["eigenvalues"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(vals.len(), 3);
        assert!(
            vals[0] >= vals[1] && vals[1] >= vals[2],
            "eigenvalues must be descending: {vals:?}"
        );

        let vecs: Vec<Vec<f64>> = out["eigenvectors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(vecs.len(), 3, "one eigenvector per dimension");

        for (k, vk) in vecs.iter().enumerate() {
            let norm: f64 = vk.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "eigenvector {k} must be unit norm, got {norm}"
            );
            // A·v_k = λ_k·v_k. The rows of the eigenvector matrix are NOT
            // eigenvectors of A, so a transposed/misaligned output fails here.
            for (i, mrow) in m.iter().enumerate() {
                let avi: f64 = (0..3).map(|j| mrow[j] * vk[j]).sum();
                assert!(
                    (avi - vals[k] * vk[i]).abs() < 1e-9,
                    "A·v[{k}] = λ[{k}]·v[{k}] failed at row {i}"
                );
            }
        }
    }

    #[test]
    fn eigen_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "eigen")
            .expect("eigen skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // Honest boundary: a non-square matrix is rejected with a clear error.
    #[test]
    fn eigen_rejects_non_square() {
        let err = eigen(json!({ "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] })).unwrap_err();
        assert!(err.contains("square"), "got: {err}");
    }

    // Honest boundary: a non-symmetric matrix is rejected, not silently solved
    // as if symmetric — the Jacobi solver would otherwise return wrong results.
    #[test]
    fn eigen_rejects_non_symmetric() {
        let err = eigen(json!({ "matrix": [[1.0, 2.0], [3.0, 4.0]] })).unwrap_err();
        assert!(err.contains("symmetric"), "got: {err}");
    }

    // --- silhouette ---------------------------------------------------------

    // Executes-test: two tight, far-apart clusters score near 1.
    #[test]
    fn silhouette_skill_scores_tight_clusters_near_one() {
        let out = silhouette(json!({
            "data": [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
            "labels": [0, 0, 0, 1, 1, 1]
        }))
        .expect("silhouette skill runs");
        let score = out["score"].as_f64().expect("score is a number");
        assert!(score > 0.9, "well-separated clusters → ~1, got {score}");
        assert_eq!(out["n_clusters"].as_u64().unwrap(), 2);
        assert_eq!(out["n_samples"].as_u64().unwrap(), 6);
    }

    // The skill must measure separation, not just run: the correct labelling
    // scores strictly higher than one interleaving the two physical groups.
    #[test]
    fn silhouette_skill_penalizes_mislabeled_clustering() {
        let data = json!([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1]
        ]);
        let good = silhouette(json!({ "data": data, "labels": [0, 0, 0, 1, 1, 1] })).unwrap()
            ["score"]
            .as_f64()
            .unwrap();
        let bad = silhouette(json!({ "data": data, "labels": [0, 1, 0, 1, 0, 1] })).unwrap()
            ["score"]
            .as_f64()
            .unwrap();
        assert!(
            good > bad,
            "correct labels must outscore interleaved: {good} vs {bad}"
        );
    }

    // Honest boundary: a labels/rows length mismatch is rejected, not scored.
    #[test]
    fn silhouette_rejects_label_row_mismatch() {
        let err =
            silhouette(json!({ "data": [[0.0, 0.0], [1.0, 1.0]], "labels": [0] })).unwrap_err();
        assert!(
            err.contains("must equal the number of data rows"),
            "got: {err}"
        );
    }

    #[test]
    fn silhouette_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "silhouette")
            .expect("silhouette skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // --- feature_importances ------------------------------------------------

    // Executes-test AND the @ai:invariant binding: feature 0 alone determines the
    // class (±2 separation); features 1,2 are noise. The fitted RF must rely on
    // feature 0, so scrambling it drops accuracy most → ranking[0] == 0.
    #[test]
    fn feature_importances_skill_ranks_informative_feature_first() {
        let out = feature_importances(json!({
            "data": [
                [2.0, 0.3, 9.0], [2.1, 0.9, 1.0], [1.9, 0.1, 4.0], [2.2, 0.7, 7.0],
                [2.0, 0.5, 2.0], [2.05, 0.2, 8.0],
                [-2.0, 0.4, 3.0], [-2.1, 0.8, 6.0], [-1.9, 0.2, 5.0], [-2.2, 0.6, 1.0],
                [-2.0, 0.5, 9.0], [-2.05, 0.1, 2.0]
            ],
            "labels": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "n_trees": 40,
            "seed": 7
        }))
        .expect("feature_importances skill runs");

        let imps: Vec<f64> = out["importances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(imps.len(), 3, "one importance per feature");
        let ranking = out["ranking"].as_array().unwrap();
        assert_eq!(
            ranking[0].as_u64().unwrap(),
            0,
            "feature 0 (label-determining) must rank first, importances={imps:?}"
        );
        assert_eq!(out["method"].as_str().unwrap(), "permutation");
        assert_eq!(out["n_features"].as_u64().unwrap(), 3);
    }

    // Honest boundary: a single-class label vector can't train a classifier.
    #[test]
    fn feature_importances_rejects_single_class() {
        let err = feature_importances(json!({
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "labels": [0, 0]
        }))
        .unwrap_err();
        assert!(
            err.contains("at least 2 distinct class labels"),
            "got: {err}"
        );
    }

    #[test]
    fn feature_importances_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "feature_importances")
            .expect("feature_importances skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // --- svd ----------------------------------------------------------------

    // Executes-test + @ai binding: U·diag(s)·Vᵀ must reconstruct the input. A
    // wrong/transposed U or V fails the reconstruction.
    #[test]
    fn svd_skill_reconstructs_input() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let out = svd(json!({ "matrix": m })).expect("svd runs");
        let u: Vec<Vec<f64>> = serde_json::from_value(out["u"].clone()).unwrap();
        let v: Vec<Vec<f64>> = serde_json::from_value(out["v"].clone()).unwrap();
        let s: Vec<f64> = serde_json::from_value(out["singular_values"].clone()).unwrap();
        for w in s.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-9 && w[1] >= -1e-9,
                "s not descending/non-neg: {s:?}"
            );
        }
        let k = s.len();
        for (i, mrow) in m.iter().enumerate() {
            for (j, &mij) in mrow.iter().enumerate() {
                let recon: f64 = (0..k).map(|r| u[i][r] * s[r] * v[j][r]).sum();
                assert!(
                    (recon - mij).abs() < 1e-6,
                    "reconstruction[{i}][{j}]={recon} != {mij}"
                );
            }
        }
    }

    // --- gmm ----------------------------------------------------------------

    // Executes-test + @ai binding: on two well-separated blobs GMM recovers two
    // components near the blob centers, soft responsibilities sum to 1 per row,
    // and the hard labels split the blobs.
    #[test]
    fn gmm_skill_recovers_separated_blobs() {
        let out = gmm(json!({
            "data": [
                [0.0, 0.0], [0.2, 0.1], [0.1, 0.2], [0.0, 0.1],
                [10.0, 10.0], [10.2, 10.1], [10.1, 10.2], [10.0, 10.1]
            ],
            "k": 2,
            "seed": 42
        }))
        .expect("gmm runs");

        let means: Vec<Vec<f64>> = serde_json::from_value(out["means"].clone()).unwrap();
        assert_eq!(means.len(), 2, "two components");
        for center in [[0.0, 0.0], [10.0, 10.0]] {
            let matched = means
                .iter()
                .any(|m| (m[0] - center[0]).powi(2) + (m[1] - center[1]).powi(2) < 1.0);
            assert!(matched, "no fitted mean near {center:?}; means={means:?}");
        }

        let resp: Vec<Vec<f64>> = serde_json::from_value(out["responsibilities"].clone()).unwrap();
        assert_eq!(resp.len(), 8);
        for row in &resp {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "responsibility row must sum to 1, got {sum}"
            );
        }

        let labels: Vec<usize> = serde_json::from_value(out["labels"].clone()).unwrap();
        assert!(
            labels[0] == labels[1] && labels[4] == labels[5] && labels[0] != labels[4],
            "labels must separate the two blobs: {labels:?}"
        );
    }

    #[test]
    fn gmm_rejects_k_exceeding_points() {
        let err = gmm(json!({ "data": [[0.0], [1.0]], "k": 5 })).unwrap_err();
        assert!(
            err.contains("exceeds the number of data points"),
            "got: {err}"
        );
    }

    // --- wavelet_denoise ----------------------------------------------------

    // Executes-test + @ai binding: with a threshold above the noise, the denoised
    // signal is closer to the clean signal than the noisy input, length preserved.
    #[test]
    fn wavelet_denoise_skill_reduces_noise() {
        let clean: Vec<f64> = (0..16).map(|i| i as f64 * 0.3).collect();
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let out = wavelet_denoise(json!({ "signal": noisy, "levels": 1, "threshold": 0.4 }))
            .expect("wavelet_denoise runs");
        let denoised: Vec<f64> = serde_json::from_value(out["denoised"].clone()).unwrap();
        assert_eq!(denoised.len(), 16, "length preserved");

        let mse = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
        };
        assert!(
            mse(&denoised, &clean) < mse(&noisy, &clean),
            "denoising must reduce MSE to the clean signal"
        );
    }

    // --- fir_filter ---------------------------------------------------------

    // Executes-test + @ai binding: a lowpass FIR reduces high-frequency content —
    // measured as squared consecutive-difference energy (high freq -> large diffs).
    #[test]
    fn fir_lowpass_attenuates_high_frequency() {
        use std::f64::consts::PI;
        let signal: Vec<f64> = (0..128)
            .map(|t| {
                let t = t as f64;
                (2.0 * PI * 0.03 * t).sin() + (2.0 * PI * 0.40 * t).sin()
            })
            .collect();
        let out = fir_filter(json!({
            "signal": signal, "kind": "lowpass", "cutoff": 0.15, "order": 32
        }))
        .expect("fir_filter runs");
        let filtered: Vec<f64> = serde_json::from_value(out["filtered"].clone()).unwrap();

        let diff_energy = |s: &[f64]| -> f64 { s.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum() };
        assert!(
            diff_energy(&filtered) < diff_energy(&signal),
            "lowpass must reduce high-frequency (diff) energy"
        );
    }

    #[test]
    fn fir_rejects_unknown_kind() {
        let err = fir_filter(json!({ "signal": [1.0, 2.0, 3.0], "kind": "notch" })).unwrap_err();
        assert!(err.contains("unknown filter kind"), "got: {err}");
    }

    // --- spectrogram --------------------------------------------------------

    // Executes-test + @ai binding: n_bins = window/2+1 and a pure tone's energy
    // localizes to the expected frequency bin.
    #[test]
    fn spectrogram_skill_localizes_a_tone() {
        use std::f64::consts::PI;
        let signal: Vec<f64> = (0..64)
            .map(|t| (2.0 * PI * 0.25 * t as f64).sin())
            .collect();
        let out = spectrogram(json!({ "signal": signal, "window_size": 16, "hop_size": 8 }))
            .expect("spectrogram runs");
        assert_eq!(out["n_bins"].as_u64().unwrap(), 9, "window/2 + 1 bins");
        let spec: Vec<Vec<f64>> = serde_json::from_value(out["spectrogram"].clone()).unwrap();
        assert!(!spec.is_empty(), "at least one frame");
        let frame = &spec[0];
        let peak = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak as i64 - 4).abs() <= 1,
            "tone should peak near bin 4, got {peak}"
        );
    }

    // --- autocorrelation ----------------------------------------------------

    // Executes-test + @ai binding: acf[0]=1.0 is the max, and a period-4 signal
    // shows a strong positive peak at lag 4 (vs a trough at the half-period lag 2).
    #[test]
    fn autocorrelation_skill_peaks_at_period() {
        use std::f64::consts::PI;
        let signal: Vec<f64> = (0..16).map(|t| (2.0 * PI * t as f64 / 4.0).sin()).collect();
        let out = autocorrelation(json!({ "signal": signal })).expect("autocorrelation runs");
        let acf: Vec<f64> = serde_json::from_value(out["autocorrelation"].clone()).unwrap();
        assert_eq!(acf.len(), 16, "one lag per sample");
        assert!(
            (acf[0] - 1.0).abs() < 1e-9,
            "zero-lag must normalize to 1.0, got {}",
            acf[0]
        );
        let max = acf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (acf[0] - max).abs() < 1e-9,
            "acf[0] must be the global maximum"
        );
        assert!(
            acf[4] > acf[2],
            "period-4 signal: lag-4 ({}) must exceed half-period lag-2 ({})",
            acf[4],
            acf[2]
        );
    }

    // --- batch coverage -----------------------------------------------------

    #[test]
    fn catalog_breadth_batch2_skills_registered_and_pipeline_callable() {
        for name in [
            "svd",
            "gmm",
            "wavelet_denoise",
            "fir_filter",
            "spectrogram",
            "autocorrelation",
        ] {
            let desc = ix_registry::all()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("{name} skill not registered"));
            assert_eq!(
                desc.inputs.len(),
                1,
                "{name} must be arity-1 to be pipeline-callable"
            );
        }
    }

    // Review P1: a non-power-of-two window would make rfft zero-pad and
    // miscalibrate the reported bins — reject it rather than mislead.
    #[test]
    fn spectrogram_rejects_non_power_of_two_window() {
        let signal: Vec<f64> = (0..64).map(|t| t as f64).collect();
        let err = spectrogram(json!({ "signal": signal, "window_size": 24 })).unwrap_err();
        assert!(err.contains("power of two"), "got: {err}");
    }

    // Review P1: the Haar DWT silently drops samples when the length is not
    // divisible by 2^levels — reject so the "preserves length" contract holds.
    #[test]
    fn wavelet_denoise_rejects_indivisible_length() {
        // length 15, levels 1 -> 15 % 2 != 0 -> reject (would otherwise return 14).
        let signal: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let err = wavelet_denoise(json!({ "signal": signal, "levels": 1, "threshold": 0.1 }))
            .unwrap_err();
        assert!(err.contains("divisible by 2^levels"), "got: {err}");
    }

    // Review P1: an extreme outlier underflows every component pdf to 0; the
    // responsibility row must still be a valid distribution (uniform fallback).
    #[test]
    fn gmm_responsibilities_sum_to_one_even_for_outliers() {
        let out = gmm(json!({
            "data": [
                [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],
                [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
                [1.0e9, 1.0e9]
            ],
            "k": 2,
            "seed": 42
        }))
        .expect("gmm runs");
        let resp: Vec<Vec<f64>> = serde_json::from_value(out["responsibilities"].clone()).unwrap();
        for (i, row) in resp.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "row {i} must sum to 1 (incl. the far outlier), got {sum}"
            );
        }
    }
}
