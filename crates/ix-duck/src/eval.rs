//! ML-evaluation UDFs — the metrics GA's retrieval / intent-routing / embedding
//! diagnostics need, but DuckDB has no native form of.
//!
//! A. Ranking metrics (scalar over a `DOUBLE[]` of relevances in rank order;
//!    `rel > 0` = relevant). Call per-query, aggregate with SQL `avg`:
//!    - `ix_ndcg(rels, k)`            — nDCG@k ∈ [0,1].
//!    - `ix_reciprocal_rank(rels)`    — 1/(rank of first relevant); `avg` ⇒ MRR.
//!    - `ix_precision_at_k(rels, k)`  — relevant in top-k / k.
//!    - `ix_recall_at_k(rels, k, total_relevant)` — relevant in top-k / total.
//!    Wraps `ix_supervised::ranking`.
//! B. `ix_classification_report(predicted_json, actual_json)` → `TABLE(label,
//!    precision, recall, f1, support)` — per-class one-vs-rest. Wraps
//!    `ix_supervised::metrics`. (Routing eval per-intent P/R/F1.)
//! C. `ix_knn_leakage(vectors_json, labels_json, k)` → `TABLE(leakage,
//!    random_baseline)` — mean fraction of each point's k nearest neighbours that
//!    share its label (embedding separability / leakage). `leakage ≫ baseline`
//!    means the embedding encodes the label. Composes the brute-force kNN.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_math::distance::euclidean;
use ix_supervised::metrics::{f1_score, precision, recall};
use ix_supervised::ranking::{ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank};
use ndarray::Array1;

use crate::tablefn::parse_matrix;
use crate::udf::read_list_col;

fn list_double() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double))
}
fn bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Bigint)
}
fn double() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Double)
}

/// Read an i64 scalar column as owned values.
fn read_i64_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<i64> {
    let v = input.flat_vector(col);
    unsafe { v.as_slice_with_len::<i64>(n) }.to_vec()
}

/// Write a per-row `f64` result computed from row `i`.
fn write_f64<F: Fn(usize) -> f64>(output: &mut dyn WritableVector, n: usize, f: F) {
    let mut out = output.flat_vector();
    let s = unsafe { out.as_mut_slice_with_len::<f64>(n) };
    for (i, slot) in s.iter_mut().enumerate().take(n) {
        *slot = f(i);
    }
}

// ── A. ranking-metric scalars ──────────────────────────────────────────────────

struct IxNdcg;
impl VScalar for IxNdcg {
    type State = ();
    // @ai:invariant ix_ndcg(rels,k) = ix_supervised::ranking ndcg_at_k; a relevance list already in ideal order scores 1.0 [T:test conf:0.9 src:ix_duck::eval::tests::ndcg_scalar]
    unsafe fn invoke(_: &Self::State, input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let rels = read_list_col(input, 0, n);
        let ks = read_i64_col(input, 1, n);
        write_f64(output, n, |i| ndcg_at_k(&rels[i], ks[i].max(0) as usize));
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![list_double(), bigint()], double())]
    }
}

struct IxReciprocalRank;
impl VScalar for IxReciprocalRank {
    type State = ();
    // @ai:invariant ix_reciprocal_rank(rels) = ix_supervised::ranking reciprocal_rank (1/rank of first relevant); avg over rows is MRR [T:test conf:0.9 src:ix_duck::eval::tests::reciprocal_rank_scalar]
    unsafe fn invoke(_: &Self::State, input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let rels = read_list_col(input, 0, n);
        write_f64(output, n, |i| reciprocal_rank(&rels[i]));
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![list_double()], double())]
    }
}

struct IxPrecisionAtK;
impl VScalar for IxPrecisionAtK {
    type State = ();
    // @ai:invariant ix_precision_at_k(rels,k) = relevant in top-k / k [T:test conf:0.9 src:ix_duck::eval::tests::precision_recall_scalars]
    unsafe fn invoke(_: &Self::State, input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let rels = read_list_col(input, 0, n);
        let ks = read_i64_col(input, 1, n);
        write_f64(output, n, |i| precision_at_k(&rels[i], ks[i].max(0) as usize));
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![list_double(), bigint()], double())]
    }
}

struct IxRecallAtK;
impl VScalar for IxRecallAtK {
    type State = ();
    // @ai:invariant ix_recall_at_k(rels,k,total) = relevant in top-k / total_relevant [T:test conf:0.9 src:ix_duck::eval::tests::precision_recall_scalars]
    unsafe fn invoke(_: &Self::State, input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let rels = read_list_col(input, 0, n);
        let ks = read_i64_col(input, 1, n);
        let totals = read_i64_col(input, 2, n);
        write_f64(output, n, |i| recall_at_k(&rels[i], ks[i].max(0) as usize, totals[i].max(0) as usize));
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![list_double(), bigint(), bigint()], double())]
    }
}

// ── B. ix_classification_report ──────────────────────────────────────────────────

#[repr(C)]
struct ReportBind {
    rows: Vec<(i64, f64, f64, f64, i64)>, // label, precision, recall, f1, support
}
#[repr(C)]
struct CursorInit {
    cursor: AtomicUsize,
}

fn parse_labels(json: &str, what: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let v: Vec<i64> = serde_json::from_str(json).map_err(|e| format!("{what}: expected a JSON int array: {e}"))?;
    if v.iter().any(|&x| x < 0) {
        return Err(format!("{what}: labels must be non-negative").into());
    }
    Ok(v)
}

struct IxClassificationReport;
impl VTab for IxClassificationReport {
    type InitData = CursorInit;
    type BindData = ReportBind;

    // @ai:invariant ix_classification_report emits per-class (precision,recall,f1,support) via ix_supervised::metrics; a perfect prediction gives precision=recall=f1=1 for every class [T:test conf:0.85 src:ix_duck::eval::tests::classification_report_perfect]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let pred = parse_labels(&bind.get_parameter(0).to_string(), "predicted")?;
        let actual = parse_labels(&bind.get_parameter(1).to_string(), "actual")?;
        if pred.len() != actual.len() {
            return Err(format!("predicted ({}) and actual ({}) lengths differ", pred.len(), actual.len()).into());
        }
        if pred.is_empty() {
            return Err("empty label arrays".into());
        }
        let yp: Array1<usize> = Array1::from_iter(pred.iter().map(|&x| x as usize));
        let yt: Array1<usize> = Array1::from_iter(actual.iter().map(|&x| x as usize));
        let classes: BTreeSet<usize> = yt.iter().chain(yp.iter()).copied().collect();
        let rows: Vec<(i64, f64, f64, f64, i64)> = classes
            .iter()
            .map(|&c| {
                let support = yt.iter().filter(|&&v| v == c).count() as i64;
                (c as i64, precision(&yt, &yp, c), recall(&yt, &yp, c), f1_score(&yt, &yp, c), support)
            })
            .collect();
        bind.add_result_column("label", bigint());
        bind.add_result_column("precision", double());
        bind.add_result_column("recall", double());
        bind.add_result_column("f1", double());
        bind.add_result_column("support", bigint());
        Ok(ReportBind { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(CursorInit { cursor: AtomicUsize::new(0) })
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let rows = &func.get_bind_data().rows;
        let cur = func.get_init_data();
        let n = rows.len();
        let start = cur.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let take = (n - start).min(output.flat_vector(0).capacity());
        let put_i64 = |output: &mut DataChunkHandle, col: usize, f: &dyn Fn(usize) -> i64| {
            let mut v = output.flat_vector(col);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(take) };
            for (i, slot) in s.iter_mut().enumerate().take(take) {
                *slot = f(start + i);
            }
        };
        let put_f64 = |output: &mut DataChunkHandle, col: usize, f: &dyn Fn(usize) -> f64| {
            let mut v = output.flat_vector(col);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(take) };
            for (i, slot) in s.iter_mut().enumerate().take(take) {
                *slot = f(start + i);
            }
        };
        put_i64(output, 0, &|j| rows[j].0);
        put_f64(output, 1, &|j| rows[j].1);
        put_f64(output, 2, &|j| rows[j].2);
        put_f64(output, 3, &|j| rows[j].3);
        put_i64(output, 4, &|j| rows[j].4);
        output.set_len(take);
        cur.cursor.store(start + take, Ordering::Relaxed);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar), LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ── C. ix_knn_leakage ────────────────────────────────────────────────────────────

#[repr(C)]
struct LeakBind {
    leakage: f64,
    baseline: f64,
}

struct IxKnnLeakage;
impl VTab for IxKnnLeakage {
    type InitData = CursorInit;
    type BindData = LeakBind;

    // @ai:invariant ix_knn_leakage = mean fraction of each point's k nearest neighbours sharing its label; well-separated labels -> leakage ~1 >> random_baseline (1/n_classes) [T:test conf:0.8 src:ix_duck::eval::tests::knn_leakage_separated_labels]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let x = parse_matrix(&bind.get_parameter(0).to_string())?;
        let labels = parse_labels(&bind.get_parameter(1).to_string(), "labels")?;
        if labels.len() != x.nrows() {
            return Err(format!("labels ({}) != vectors ({})", labels.len(), x.nrows()).into());
        }
        let n = x.nrows();
        if n < 2 {
            return Err("ix_knn_leakage needs at least 2 vectors".into());
        }
        let k_req = bind.get_parameter(2).to_int64();
        if k_req < 1 {
            return Err("k must be >= 1".into());
        }
        let k = (k_req as usize).min(n - 1);
        let pts: Vec<Array1<f64>> = (0..n).map(|i| x.row(i).to_owned()).collect();
        let mut agree = 0.0f64;
        for i in 0..n {
            let mut d: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (euclidean(&pts[i], &pts[j]).unwrap_or(f64::INFINITY), j))
                .collect();
            d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let same = d.iter().take(k).filter(|&&(_, j)| labels[j] == labels[i]).count();
            agree += same as f64 / k as f64;
        }
        let n_classes = labels.iter().collect::<BTreeSet<_>>().len().max(1);
        bind.add_result_column("leakage", double());
        bind.add_result_column("random_baseline", double());
        Ok(LeakBind { leakage: agree / n as f64, baseline: 1.0 / n_classes as f64 })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(CursorInit { cursor: AtomicUsize::new(0) })
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let cur = func.get_init_data();
        if cur.cursor.swap(1, Ordering::Relaxed) != 0 {
            output.set_len(0);
            return Ok(());
        }
        let b = func.get_bind_data();
        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(1) };
            s[0] = b.leakage;
        }
        {
            let mut v = output.flat_vector(1);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(1) };
            s[0] = b.baseline;
        }
        output.set_len(1);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            bigint(),
        ])
    }
}

/// Register the ML-evaluation UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxNdcg>("ix_ndcg")?;
    conn.register_scalar_function::<IxReciprocalRank>("ix_reciprocal_rank")?;
    conn.register_scalar_function::<IxPrecisionAtK>("ix_precision_at_k")?;
    conn.register_scalar_function::<IxRecallAtK>("ix_recall_at_k")?;
    conn.register_table_function::<IxClassificationReport>("ix_classification_report")?;
    conn.register_table_function::<IxKnnLeakage>("ix_knn_leakage")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    #[test]
    fn ndcg_scalar() {
        let conn = open_bench().unwrap();
        // already-ideal order → 1.0; reversed → < 1.
        let perfect: f64 = conn
            .query_row("SELECT ix_ndcg([3.0,2.0,1.0,0.0], 4)", [], |r| r.get(0))
            .unwrap();
        assert!((perfect - 1.0).abs() < 1e-9);
        let worse: f64 = conn.query_row("SELECT ix_ndcg([0.0,1.0,2.0,3.0], 4)", [], |r| r.get(0)).unwrap();
        assert!(worse < perfect);
    }

    #[test]
    fn reciprocal_rank_scalar() {
        let conn = open_bench().unwrap();
        let rr: f64 = conn.query_row("SELECT ix_reciprocal_rank([0.0,0.0,1.0])", [], |r| r.get(0)).unwrap();
        assert!((rr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn precision_recall_scalars() {
        let conn = open_bench().unwrap();
        let p: f64 = conn.query_row("SELECT ix_precision_at_k([1.0,0.0,1.0,0.0], 2)", [], |r| r.get(0)).unwrap();
        assert!((p - 0.5).abs() < 1e-9);
        let rc: f64 = conn.query_row("SELECT ix_recall_at_k([1.0,0.0,1.0], 3, 4)", [], |r| r.get(0)).unwrap();
        assert!((rc - 0.5).abs() < 1e-9);
    }

    #[test]
    fn classification_report_perfect() {
        let conn = open_bench().unwrap();
        // perfect prediction → every class precision=recall=f1=1.
        let min_f1: f64 = conn
            .query_row(
                "SELECT min(f1) FROM ix_classification_report('[0,1,2,0]', '[0,1,2,0]')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((min_f1 - 1.0).abs() < 1e-9);
        let n: i64 = conn
            .query_row("SELECT count(*) FROM ix_classification_report('[0,1,2,0]', '[0,1,2,0]')", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 3, "one row per class");
    }

    #[test]
    fn knn_leakage_separated_labels() {
        let conn = open_bench().unwrap();
        // two tight, far-apart, single-label blobs → leakage ~1, well above baseline 0.5.
        let (leak, base): (f64, f64) = conn
            .query_row(
                "SELECT leakage, random_baseline FROM ix_knn_leakage('[[0,0],[0,1],[10,10],[10,11]]', '[0,0,1,1]', 1)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(leak > 0.9, "separated labels → high leakage, got {leak}");
        assert!((base - 0.5).abs() < 1e-9, "2 classes → baseline 0.5");
    }
}
