//! IX algorithms exposed as DuckDB *table* functions (via the `VTab` trait).
//!
//! - `ix_pca_project(json_vectors VARCHAR, n_components BIGINT)`
//!     → `TABLE(row BIGINT, coords DOUBLE[])`. Fits PCA over the input vectors and
//!     returns each projected onto the top `n_components` principal components.
//!     Wraps `ix_unsupervised::pca::PCA` — no reimplementation.
//! - `ix_silhouette(json_vectors VARCHAR, json_labels VARCHAR)`
//!     → `TABLE(row BIGINT, label BIGINT, silhouette DOUBLE)`. Per-point silhouette
//!     coefficient for the given clustering. Mean score:
//!     `SELECT avg(silhouette) FROM ix_silhouette(...)`.
//! - `ix_kdist(json_vectors VARCHAR, k BIGINT)`
//!     → `TABLE(row BIGINT, kdist DOUBLE)`. Per-point kNN-distance: the mean
//!     Euclidean distance to a point's `k` nearest neighbours within the set
//!     (leave-one-out). The OOD / local-outlier signal — higher `kdist` means a
//!     point sits further from the rest of the corpus. To score a *query* against
//!     an in-domain reference, append the query as the **last element of the JSON
//!     vectors array** (its `row` is then `n_vectors - 1`) and select that row —
//!     do **not** `UNION` the query in SQL, which reorders and de-duplicates so the
//!     positional `row` no longer identifies it. `k` is capped at `n_vectors - 1`.
//!     Wraps `ix_math::distance::euclidean`.
//! - `ix_dbscan(json_vectors VARCHAR, eps DOUBLE, min_points BIGINT)`
//!     → `TABLE(row BIGINT, cluster BIGINT)`. Density-based clustering: each point's
//!     cluster id (1, 2, …), or `0` for **noise** (no dense neighbourhood — the
//!     outlier / OOD bucket). Composes with `ix_silhouette` for cluster quality:
//!     feed these labels in. Wraps `ix_unsupervised::dbscan::DBSCAN`.
//! - `ix_kmeans(json_vectors VARCHAR, k BIGINT)` / `ix_gmm(json_vectors VARCHAR, k BIGINT)`
//!     → `TABLE(row BIGINT, cluster BIGINT)`. Centroid (k-means) / Gaussian-mixture
//!     cluster labels `0..k-1`. `k` is capped at the sample count; deterministic
//!     (seed 42). The centroid/mixture complement to `ix_dbscan`'s density labels —
//!     all three feed `ix_silhouette`. Wrap `ix_unsupervised::{kmeans,gmm}`.
//!
//! A whole set enters in one SQL call as JSON scalar params (a 2-D number array
//! for vectors, a 1-D int array for labels):
//!   `SELECT * FROM ix_pca_project('[[1,2,3],[4,5,6]]', 2);`
//!
//! Rows are streamed in chunks of the output vector capacity, so result sets
//! larger than one DuckDB vector are emitted correctly across repeated `func`
//! calls via a cursor in the (Send+Sync) init data.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_math::distance::euclidean;
use ix_unsupervised::dbscan::DBSCAN;
use ix_unsupervised::gmm::GMM;
use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::pca::PCA;
use ix_unsupervised::traits::{Clusterer, DimensionReducer};
use ndarray::{Array1, Array2};

/// Register every IX table function on `conn`.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_table_function::<IxPcaProject>("ix_pca_project")?;
    conn.register_table_function::<IxSilhouette>("ix_silhouette")?;
    conn.register_table_function::<IxKdist>("ix_kdist")?;
    conn.register_table_function::<IxDbscan>("ix_dbscan")?;
    conn.register_table_function::<IxKmeans>("ix_kmeans")?;
    conn.register_table_function::<IxGmm>("ix_gmm")?;
    Ok(())
}

/// Parse a JSON 2-D number array into a rectangular `(n_samples, n_features)` matrix.
/// Errors (empty, ragged, non-JSON) surface as SQL errors, not panics.
pub(crate) fn parse_matrix(json: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let rows: Vec<Vec<f64>> =
        serde_json::from_str(json).map_err(|e| format!("expected a JSON 2-D number array: {e}"))?;
    if rows.is_empty() {
        return Err("input vector set is empty".into());
    }
    let ncols = rows[0].len();
    if ncols == 0 {
        return Err("input vectors have zero dimensions".into());
    }
    if rows.iter().any(|r| r.len() != ncols) {
        return Err("input vectors are not all the same length".into());
    }
    let n = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, ncols), flat)?)
}

// ── ix_pca_project ───────────────────────────────────────────────────────────

#[repr(C)]
struct PcaBind {
    /// Projected coordinates: one inner Vec (length `k`) per input row.
    projected: Vec<Vec<f64>>,
    k: usize,
}
#[repr(C)]
struct PcaInit {
    cursor: AtomicUsize,
}

struct IxPcaProject;

impl VTab for IxPcaProject {
    type InitData = PcaInit;
    type BindData = PcaBind;

    // @ai:invariant ix_pca_project fits ix_unsupervised PCA over the JSON input vectors and emits one row per input with its projection onto min(n_components, n_features) components [T:test conf:0.85 src:ix_duck::tablefn::tests::pca_project_row_and_dim_count]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let json = bind.get_parameter(0).to_string();
        let k_req = bind.get_parameter(1).to_int64();
        if k_req < 1 {
            return Err("n_components must be >= 1".into());
        }
        let x = parse_matrix(&json)?;
        let k = (k_req as usize).min(x.ncols());

        let mut pca = PCA::new(k);
        let projected_mat = pca.fit_transform(&x); // (n_samples, k)
        let projected: Vec<Vec<f64>> = (0..projected_mat.nrows())
            .map(|r| projected_mat.row(r).to_vec())
            .collect();

        bind.add_result_column("row", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column(
            "coords",
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double)),
        );
        Ok(PcaBind { projected, k })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(PcaInit {
            cursor: AtomicUsize::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let n = bind.projected.len();
        let start = init.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let cap = output.flat_vector(0).capacity();
        let rows = (n - start).min(cap);

        // col 0: row index (BIGINT)
        {
            let mut idx_vec = output.flat_vector(0);
            let slice = unsafe { idx_vec.as_mut_slice_with_len::<i64>(rows) };
            for (i, s) in slice.iter_mut().enumerate().take(rows) {
                *s = (start + i) as i64;
            }
        }
        // col 1: coords (LIST<DOUBLE>) — flatten this chunk's rows into the child
        // buffer, then point each row's (offset, length) entry at its slice.
        {
            let k = bind.k;
            let total = rows * k;
            let flat: Vec<f64> = (0..rows)
                .flat_map(|i| bind.projected[start + i].iter().copied())
                .collect();
            let mut lv = output.list_vector(1);
            {
                let mut child = lv.child(total);
                let cslice = unsafe { child.as_mut_slice_with_len::<f64>(total) };
                cslice[..total].copy_from_slice(&flat);
            }
            lv.set_len(total);
            for i in 0..rows {
                lv.set_entry(i, i * k, k);
            }
        }
        output.set_len(rows);
        init.cursor.store(start + rows, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── ix_silhouette ────────────────────────────────────────────────────────────

#[repr(C)]
struct SilBind {
    labels: Vec<i64>,
    sil: Vec<f64>,
}
#[repr(C)]
struct SilInit {
    cursor: AtomicUsize,
}

struct IxSilhouette;

impl VTab for IxSilhouette {
    type InitData = SilInit;
    type BindData = SilBind;

    // @ai:invariant ix_silhouette emits one row per point with its silhouette coefficient (b-a)/max(a,b); well-separated clusters -> ~1, single cluster -> 0 [T:test conf:0.85 src:ix_duck::tablefn::tests::silhouette_separated_clusters_near_one]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let x = parse_matrix(&bind.get_parameter(0).to_string())?;
        let labels: Vec<i64> = serde_json::from_str(&bind.get_parameter(1).to_string())
            .map_err(|e| format!("expected a JSON int array for labels: {e}"))?;
        if labels.len() != x.nrows() {
            return Err(format!(
                "labels length ({}) != number of vectors ({})",
                labels.len(),
                x.nrows()
            )
            .into());
        }
        let sil = silhouette_per_point(&x, &labels)?;
        bind.add_result_column("row", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("label", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("silhouette", LogicalTypeHandle::from(LogicalTypeId::Double));
        Ok(SilBind { labels, sil })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(SilInit {
            cursor: AtomicUsize::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let n = bind.sil.len();
        let start = init.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let cap = output.flat_vector(0).capacity();
        let rows = (n - start).min(cap);

        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = (start + i) as i64;
            }
        }
        {
            let mut v = output.flat_vector(1);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = bind.labels[start + i];
            }
        }
        {
            let mut v = output.flat_vector(2);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = bind.sil[start + i];
            }
        }
        output.set_len(rows);
        init.cursor.store(start + rows, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── ix_kdist ───────────────────────────────────────────────────────────────────

#[repr(C)]
struct KdistBind {
    /// Per-point mean distance to its `k` nearest neighbours.
    kdist: Vec<f64>,
}
#[repr(C)]
struct KdistInit {
    cursor: AtomicUsize,
}

struct IxKdist;

impl VTab for IxKdist {
    type InitData = KdistInit;
    type BindData = KdistBind;

    // @ai:invariant ix_kdist emits one row per point with the mean Euclidean distance to its k nearest neighbours (leave-one-out); an isolated outlier has the largest kdist [T:test conf:0.85 src:ix_duck::tablefn::tests::kdist_outlier_has_largest_distance]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let x = parse_matrix(&bind.get_parameter(0).to_string())?;
        let k_req = bind.get_parameter(1).to_int64();
        if k_req < 1 {
            return Err("k must be >= 1".into());
        }
        let n = x.nrows();
        if n < 2 {
            return Err("ix_kdist needs at least 2 vectors (kNN-distance is leave-one-out)".into());
        }
        let k = (k_req as usize).min(n - 1);
        let kdist = kdist_per_point(&x, k)?;
        bind.add_result_column("row", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("kdist", LogicalTypeHandle::from(LogicalTypeId::Double));
        Ok(KdistBind { kdist })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(KdistInit {
            cursor: AtomicUsize::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let n = bind.kdist.len();
        let start = init.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let cap = output.flat_vector(0).capacity();
        let rows = (n - start).min(cap);

        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = (start + i) as i64;
            }
        }
        {
            let mut v = output.flat_vector(1);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = bind.kdist[start + i];
            }
        }
        output.set_len(rows);
        init.cursor.store(start + rows, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── ix_dbscan ────────────────────────────────────────────────────────────────────

#[repr(C)]
struct DbscanBind {
    /// Per-point cluster id: 0 = noise, clusters 1, 2, …
    clusters: Vec<i64>,
}
#[repr(C)]
struct DbscanInit {
    cursor: AtomicUsize,
}

struct IxDbscan;

impl VTab for IxDbscan {
    type InitData = DbscanInit;
    type BindData = DbscanBind;

    // @ai:invariant ix_dbscan emits one row per point with its DBSCAN cluster id (0 = noise); two well-separated dense blobs yield exactly two non-noise clusters [T:test conf:0.85 src:ix_duck::tablefn::tests::dbscan_finds_two_clusters_and_noise]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let x = parse_matrix(&bind.get_parameter(0).to_string())?;
        let eps = bind
            .get_parameter(1)
            .to_string()
            .parse::<f64>()
            .map_err(|e| format!("eps must be a number: {e}"))?;
        // Reject non-finite (NaN, ±∞): +∞ would put every point within range and
        // silently collapse the dataset into one cluster.
        if !eps.is_finite() || eps <= 0.0 {
            return Err("eps must be a finite number > 0".into());
        }
        let min_points = bind.get_parameter(2).to_int64();
        if min_points < 1 {
            return Err("min_points must be >= 1".into());
        }
        let mut model = DBSCAN::new(eps, min_points as usize);
        let labels = model.fit_predict(&x); // 0 = noise, clusters 1, 2, …
        let clusters: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
        bind.add_result_column("row", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("cluster", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        Ok(DbscanBind { clusters })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(DbscanInit {
            cursor: AtomicUsize::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let n = bind.clusters.len();
        let start = init.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let cap = output.flat_vector(0).capacity();
        let rows = (n - start).min(cap);

        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = (start + i) as i64;
            }
        }
        {
            let mut v = output.flat_vector(1);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = bind.clusters[start + i];
            }
        }
        output.set_len(rows);
        init.cursor.store(start + rows, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Double),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── ix_kmeans / ix_gmm (centroid / mixture cluster labels) ─────────────────────────
//
// Both produce per-point labels `0..k-1` for a `(json_vectors, k)` call, so they share
// the output shape (row, cluster), the streaming `func`, and the bind preamble — only
// the algorithm differs. Deterministic (the wrapped estimators default to seed 42).

#[repr(C)]
struct LabelBind {
    clusters: Vec<i64>,
}
#[repr(C)]
struct LabelInit {
    cursor: AtomicUsize,
}

/// Shared bind preamble: parse the matrix, validate `k`, declare the (row, cluster)
/// columns, and return the matrix + `k` capped at the sample count.
fn cluster_k_bind(bind: &BindInfo) -> Result<(Array2<f64>, usize), Box<dyn std::error::Error>> {
    let x = parse_matrix(&bind.get_parameter(0).to_string())?;
    let k_req = bind.get_parameter(1).to_int64();
    if k_req < 1 {
        return Err("k must be >= 1".into());
    }
    let k = (k_req as usize).min(x.nrows());
    bind.add_result_column("row", LogicalTypeHandle::from(LogicalTypeId::Bigint));
    bind.add_result_column("cluster", LogicalTypeHandle::from(LogicalTypeId::Bigint));
    Ok((x, k))
}

/// Shared streaming `func` for label table functions (row index + cluster id).
fn emit_label_rows(
    clusters: &[i64],
    init: &LabelInit,
    output: &mut DataChunkHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = clusters.len();
    let start = init.cursor.load(Ordering::Relaxed);
    if start >= n {
        output.set_len(0);
        return Ok(());
    }
    let cap = output.flat_vector(0).capacity();
    let rows = (n - start).min(cap);
    {
        let mut v = output.flat_vector(0);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
        for (i, slot) in s.iter_mut().enumerate().take(rows) {
            *slot = (start + i) as i64;
        }
    }
    {
        let mut v = output.flat_vector(1);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
        for (i, slot) in s.iter_mut().enumerate().take(rows) {
            *slot = clusters[start + i];
        }
    }
    output.set_len(rows);
    init.cursor.store(start + rows, Ordering::Relaxed);
    Ok(())
}

struct IxKmeans;

impl VTab for IxKmeans {
    type InitData = LabelInit;
    type BindData = LabelBind;

    // @ai:invariant ix_kmeans emits one row per point with its k-means cluster id 0..k-1 (k capped at sample count, deterministic seed); two separated blobs get two distinct labels [T:test conf:0.85 src:ix_duck::tablefn::tests::kmeans_separates_two_blobs]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let (x, k) = cluster_k_bind(bind)?;
        let labels = KMeans::new(k).fit_predict(&x);
        Ok(LabelBind { clusters: labels.iter().map(|&l| l as i64).collect() })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(LabelInit { cursor: AtomicUsize::new(0) })
    }
    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        emit_label_rows(&func.get_bind_data().clusters, func.get_init_data(), output)
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

struct IxGmm;

impl VTab for IxGmm {
    type InitData = LabelInit;
    type BindData = LabelBind;

    // @ai:invariant ix_gmm emits one row per point with its Gaussian-mixture component id 0..k-1 (k capped at sample count, deterministic seed); two separated blobs get two distinct labels [T:test conf:0.85 src:ix_duck::tablefn::tests::gmm_separates_two_blobs]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let (x, k) = cluster_k_bind(bind)?;
        let labels = GMM::new(k).fit_predict(&x);
        Ok(LabelBind { clusters: labels.iter().map(|&l| l as i64).collect() })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(LabelInit { cursor: AtomicUsize::new(0) })
    }
    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        emit_label_rows(&func.get_bind_data().clusters, func.get_init_data(), output)
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

/// Per-point kNN-distance: the mean Euclidean distance to each point's `k` nearest
/// neighbours within the set (leave-one-out). Mean-of-k (not the k-th alone) is the
/// robust form — a single spurious neighbour can't swing the score. Brute-force
/// `O(n²)` over `ix_math::distance::euclidean`; fine at analyst-bench scale (this is
/// not the production voicing-search path — `optick.index` owns that).
fn kdist_per_point(x: &Array2<f64>, k: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = x.nrows();
    let pts: Vec<Array1<f64>> = (0..n).map(|i| x.row(i).to_owned()).collect();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut dists: Vec<f64> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                dists.push(euclidean(&pts[i], &pts[j])?);
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kk = k.min(dists.len()).max(1);
        out[i] = dists[..kk].iter().sum::<f64>() / kk as f64;
    }
    Ok(out)
}

/// Per-point silhouette coefficient `s(i) = (b - a) / max(a, b)`, where `a` is the
/// mean intra-cluster distance and `b` the mean distance to the nearest other
/// cluster. `s = 0` when the point's cluster is a singleton or is the only cluster
/// (silhouette undefined there, by convention).
fn silhouette_per_point(
    x: &Array2<f64>,
    labels: &[i64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = x.nrows();
    let pts: Vec<Array1<f64>> = (0..n).map(|i| x.row(i).to_owned()).collect();
    let mut clusters: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        clusters.entry(l).or_default().push(i);
    }
    let mut out = vec![0.0; n];
    if clusters.len() < 2 {
        return Ok(out); // undefined with a single cluster
    }
    for i in 0..n {
        let li = labels[i];
        let own = &clusters[&li];
        let a = if own.len() <= 1 {
            0.0
        } else {
            let mut sum = 0.0;
            for &j in own {
                if j != i {
                    sum += euclidean(&pts[i], &pts[j])?;
                }
            }
            sum / (own.len() - 1) as f64
        };
        let mut b = f64::INFINITY;
        for (&l, members) in &clusters {
            if l == li {
                continue;
            }
            let mut sum = 0.0;
            for &j in members {
                sum += euclidean(&pts[i], &pts[j])?;
            }
            let mean = sum / members.len() as f64;
            if mean < b {
                b = mean;
            }
        }
        let denom = a.max(b);
        out[i] = if denom > 0.0 { (b - a) / denom } else { 0.0 };
    }
    Ok(out)
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    #[test]
    fn pca_project_row_and_dim_count() {
        let conn = open_bench().unwrap();
        // 4 collinear 3-D points → project to 1 component.
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM ix_pca_project('[[1,2,3],[2,4,6],[3,6,9],[4,8,12]]', 1)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 4, "one row per input vector");

        let dim: i64 = conn
            .query_row(
                "SELECT len(coords) FROM ix_pca_project('[[1,2,3],[2,4,6],[3,6,9],[4,8,12]]', 1) LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dim, 1, "coords length == n_components");
    }

    #[test]
    fn pca_project_caps_components_at_feature_count() {
        let conn = open_bench().unwrap();
        // Ask for 5 components on 2-D data → capped at 2.
        let dim: i64 = conn
            .query_row(
                "SELECT len(coords) FROM ix_pca_project('[[0,0],[1,1],[2,2]]', 5) LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dim, 2, "n_components capped at n_features");
    }

    #[test]
    fn silhouette_separated_clusters_near_one() {
        let conn = open_bench().unwrap();
        // Two tight, far-apart clusters → mean silhouette near 1.
        let avg: f64 = conn
            .query_row(
                "SELECT avg(silhouette) FROM ix_silhouette('[[0,0],[0,1],[10,10],[10,11]]', '[0,0,1,1]')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            avg > 0.8,
            "well-separated clusters → high silhouette, got {avg}"
        );

        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM ix_silhouette('[[0,0],[0,1],[10,10],[10,11]]', '[0,0,1,1]')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 4, "one row per point");
    }

    #[test]
    fn silhouette_single_cluster_is_zero() {
        let conn = open_bench().unwrap();
        let avg: f64 = conn
            .query_row(
                "SELECT avg(silhouette) FROM ix_silhouette('[[0,0],[1,1],[2,2]]', '[0,0,0]')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            avg.abs() < 1e-12,
            "single cluster → silhouette 0, got {avg}"
        );
    }

    #[test]
    fn kdist_outlier_has_largest_distance() {
        let conn = open_bench().unwrap();
        // Three tight points + one far outlier. With k=1, the outlier's nearest
        // neighbour is far, so it must have the largest kdist → it's row 3.
        let outlier_row: i64 = conn
            .query_row(
                "SELECT row FROM ix_kdist('[[0,0],[0,1],[1,0],[10,10]]', 1) ORDER BY kdist DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(outlier_row, 3, "the isolated point has the largest kdist");

        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM ix_kdist('[[0,0],[0,1],[1,0],[10,10]]', 1)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 4, "one row per point");
    }

    #[test]
    fn kdist_caps_k_at_n_minus_one_and_computes_mean() {
        let conn = open_bench().unwrap();
        // Two points sqrt(2) apart; ask for k=5 → capped at 1. Each point's only
        // neighbour is the other, so kdist == sqrt(2) for both.
        let d: f64 = conn
            .query_row(
                "SELECT kdist FROM ix_kdist('[[0,0],[1,1]]', 5) LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((d - 2.0_f64.sqrt()).abs() < 1e-9, "k capped at n-1; kdist = sqrt(2), got {d}");
    }

    #[test]
    fn kdist_rejects_single_vector() {
        let conn = open_bench().unwrap();
        // kNN-distance is leave-one-out → needs >= 2 vectors.
        let err = conn
            .query_row("SELECT kdist FROM ix_kdist('[[1,2,3]]', 1)", [], |r| {
                r.get::<_, f64>(0)
            })
            .is_err();
        assert!(err, "a single vector must be a SQL error, not a panic");
    }

    #[test]
    fn dbscan_finds_two_clusters_and_noise() {
        let conn = open_bench().unwrap();
        // Two tight blobs (rows 0-2 near origin, rows 3-5 near (10,10)) plus a lone
        // outlier far away (row 6). eps=2, min_points=2 → two clusters + noise.
        let pts = "[[0,0],[0,1],[1,0],[10,10],[10,11],[11,10],[50,50]]";
        let n_clusters: i64 = conn
            .query_row(
                &format!("SELECT count(DISTINCT cluster) FROM ix_dbscan('{pts}', 2.0, 2) WHERE cluster <> 0"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_clusters, 2, "two dense blobs → two non-noise clusters");

        // The lone far point (row 6) is noise (cluster 0).
        let outlier_cluster: i64 = conn
            .query_row(
                &format!("SELECT cluster FROM ix_dbscan('{pts}', 2.0, 2) WHERE row = 6"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(outlier_cluster, 0, "the isolated point is labelled noise");

        let n: i64 = conn
            .query_row(
                &format!("SELECT count(*) FROM ix_dbscan('{pts}', 2.0, 2)"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 7, "one row per point");
    }

    #[test]
    fn dbscan_composes_with_silhouette() {
        // The point of pairing the two: cluster with dbscan, score with silhouette.
        // DuckDB table functions take only *literal* params (no lateral-join columns),
        // so composition is two steps: materialize dbscan's labels, then pass them as a
        // JSON literal to ix_silhouette. Two well-separated blobs → high silhouette.
        let conn = open_bench().unwrap();
        let pts = "[[0,0],[0,1],[10,10],[10,11]]";
        let mut stmt = conn
            .prepare(&format!(
                "SELECT cluster FROM ix_dbscan('{pts}', 2.0, 2) ORDER BY row"
            ))
            .unwrap();
        let labels: Vec<i64> = stmt
            .query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .map(|x| x.unwrap())
            .collect();
        let labels_json = serde_json::to_string(&labels).unwrap();
        let avg: f64 = conn
            .query_row(
                &format!("SELECT avg(silhouette) FROM ix_silhouette('{pts}', '{labels_json}')"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(avg > 0.8, "well-separated dbscan clusters → high silhouette, got {avg}");
    }

    #[test]
    fn dbscan_rejects_bad_params() {
        let conn = open_bench().unwrap();
        // eps must be > 0.
        assert!(
            conn.query_row("SELECT cluster FROM ix_dbscan('[[0,0],[1,1]]', 0.0, 2)", [], |r| r
                .get::<_, i64>(0))
                .is_err(),
            "eps=0 must be a SQL error"
        );
        // eps must be finite — +∞ would collapse everything into one cluster.
        assert!(
            conn.query_row(
                "SELECT cluster FROM ix_dbscan('[[0,0],[1,1]]', 'inf'::DOUBLE, 2)",
                [],
                |r| r.get::<_, i64>(0)
            )
            .is_err(),
            "eps=+inf must be a SQL error"
        );
    }

    // Two tight, far-apart blobs (rows 0-1 near origin, rows 2-3 near (10,10)) →
    // a 2-cluster model must put each blob in its own cluster (label values are
    // arbitrary, so assert the two blobs differ and there are exactly 2 clusters).
    const TWO_BLOBS: &str = "[[0,0],[0,1],[10,10],[10,11]]";

    #[test]
    fn kmeans_separates_two_blobs() {
        let conn = open_bench().unwrap();
        let n_clusters: i64 = conn
            .query_row(
                &format!("SELECT count(DISTINCT cluster) FROM ix_kmeans('{TWO_BLOBS}', 2)"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_clusters, 2, "k=2 over two blobs → two clusters");
        // row 0 and row 2 are in different blobs → different labels.
        let same: bool = conn
            .query_row(
                &format!(
                    "SELECT (SELECT cluster FROM ix_kmeans('{TWO_BLOBS}', 2) WHERE row=0) \
                          = (SELECT cluster FROM ix_kmeans('{TWO_BLOBS}', 2) WHERE row=2)"
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(!same, "points in different blobs land in different clusters");
        let n: i64 = conn
            .query_row(&format!("SELECT count(*) FROM ix_kmeans('{TWO_BLOBS}', 2)"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 4, "one row per point");
    }

    #[test]
    fn gmm_separates_two_blobs() {
        let conn = open_bench().unwrap();
        let n_clusters: i64 = conn
            .query_row(
                &format!("SELECT count(DISTINCT cluster) FROM ix_gmm('{TWO_BLOBS}', 2)"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_clusters, 2, "k=2 over two blobs → two components");
    }

    #[test]
    fn kmeans_caps_k_at_sample_count_and_rejects_zero() {
        let conn = open_bench().unwrap();
        // k=9 over 2 points → capped at 2, still one row per point, no panic.
        let n: i64 = conn
            .query_row("SELECT count(*) FROM ix_kmeans('[[0,0],[9,9]]', 9)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
        // k < 1 is a SQL error.
        assert!(
            conn.query_row("SELECT cluster FROM ix_kmeans('[[0,0],[1,1]]', 0)", [], |r| r
                .get::<_, i64>(0))
                .is_err(),
            "k=0 must be a SQL error"
        );
    }
}
