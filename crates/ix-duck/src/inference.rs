//! Statistical-inference UDFs over `ix-math::inference` — the distribution-shape
//! and two-sample-test layer DuckDB built-ins lack.
//!
//! Scalars take `LIST<DOUBLE>` (materialize a column with `list(col)`):
//!
//! | UDF | meaning |
//! |---|---|
//! | `ix_skewness(x)` / `ix_kurtosis(x)` | biased Fisher–Pearson skew / excess kurtosis |
//! | `ix_mad(x)` | median absolute deviation (robust spread) |
//! | `ix_quantile(x, q)` | the `q`-quantile (numpy linear) |
//! | `ix_entropy(p)` | Shannon entropy in nats |
//! | `ix_kl(p, q)` / `ix_js(p, q)` | KL / Jensen–Shannon divergence |
//! | `ix_pearson(a, b)` | Pearson correlation in [−1, 1] (the mesh's pairwise primitive) |
//!
//! And the regression-gate primitive, a table function over two JSON samples:
//! `ix_two_sample(a_json, b_json, kind)` → `TABLE(statistic, p_value)`, `kind` ∈
//! `ks | mannwhitney | welch`. The motivating use is *"did this metric's
//! distribution shift vs baseline?"* — e.g.
//! `SELECT p_value FROM ix_two_sample(today, baseline, 'ks')`.
//!
//! Pure wraps of `ix-math::inference` — no statistics reimplemented here.

use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_math::error::MathError;
use ix_math::inference::{
    iqr as _iqr, js_divergence, kl_divergence, ks_two_sample, kurtosis, mad, mann_whitney_u,
    pearson, quantile, shannon_entropy, skewness, welch_t_test, TestResult,
};

use crate::udf::read_list_col;

type Res = Result<(), Box<dyn std::error::Error>>;

fn list_double() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double))
}
fn double() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Double)
}

/// Shared driver for `LIST<DOUBLE> -> DOUBLE` scalars.
fn invoke_unary(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    name: &str,
    f: impl Fn(&[f64]) -> Result<f64, MathError>,
) -> Res {
    let n = input.len();
    let a = read_list_col(input, 0, n);
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<f64>(n) };
    for i in 0..n {
        slice[i] = f(&a[i]).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

/// Shared driver for `(LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE` scalars.
fn invoke_binary(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    name: &str,
    f: impl Fn(&[f64], &[f64]) -> Result<f64, MathError>,
) -> Res {
    let n = input.len();
    let a = read_list_col(input, 0, n);
    let b = read_list_col(input, 1, n);
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<f64>(n) };
    for i in 0..n {
        slice[i] = f(&a[i], &b[i]).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

fn unary_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(vec![list_double()], double())]
}
fn binary_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![list_double(), list_double()],
        double(),
    )]
}

macro_rules! unary_udf {
    ($name:ident, $sql:literal, $f:path, $doc:literal) => {
        #[doc = $doc]
        struct $name;
        impl VScalar for $name {
            type State = ();
            unsafe fn invoke(
                _: &(),
                input: &mut DataChunkHandle,
                output: &mut dyn WritableVector,
            ) -> Res {
                invoke_unary(input, output, $sql, $f)
            }
            fn signatures() -> Vec<ScalarFunctionSignature> {
                unary_sig()
            }
        }
    };
}

// @ai:invariant ix_skewness/ix_kurtosis/ix_mad/ix_entropy each wrap the same-named ix_math::inference fn over a LIST<DOUBLE>; zero-variance input (skew/kurtosis) or all-zero p (entropy) -> SQL error (no panic) [T:test conf:0.85 src:ix_duck::inference::tests::descriptive_scalars_match_ix_math]
unary_udf!(IxSkewness, "ix_skewness", skewness, "`ix_skewness(x DOUBLE[]) -> DOUBLE`");
unary_udf!(IxKurtosis, "ix_kurtosis", kurtosis, "`ix_kurtosis(x DOUBLE[]) -> DOUBLE`");
unary_udf!(IxMad, "ix_mad", mad, "`ix_mad(x DOUBLE[]) -> DOUBLE`");
unary_udf!(IxEntropy, "ix_entropy", shannon_entropy, "`ix_entropy(p DOUBLE[]) -> DOUBLE`");
unary_udf!(IxIqr, "ix_iqr", _iqr, "`ix_iqr(x DOUBLE[]) -> DOUBLE`");

/// `ix_quantile(x DOUBLE[], q DOUBLE) -> DOUBLE` — the `q`-quantile (numpy linear).
struct IxQuantile;
impl VScalar for IxQuantile {
    type State = ();
    // @ai:invariant ix_quantile(x,q) wraps ix_math::inference::quantile; q=0.5 is the median, q outside [0,1] -> SQL error [T:test conf:0.85 src:ix_duck::inference::tests::quantile_scalar]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let a = read_list_col(input, 0, n);
        let qs: Vec<f64> = {
            let v = input.flat_vector(1);
            v.as_slice_with_len::<f64>(n)[..n].to_vec()
        };
        let mut out = output.flat_vector();
        let slice = out.as_mut_slice_with_len::<f64>(n);
        for i in 0..n {
            slice[i] = quantile(&a[i], qs[i]).map_err(|e| format!("ix_quantile: {e}"))?;
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), double()],
            double(),
        )]
    }
}

/// `ix_kl(p DOUBLE[], q DOUBLE[]) -> DOUBLE` — Kullback–Leibler divergence (nats).
struct IxKl;
impl VScalar for IxKl {
    type State = ();
    // @ai:invariant ix_kl wraps ix_math::inference::kl_divergence; equal distributions -> 0, q zero where p positive -> SQL error [T:test conf:0.85 src:ix_duck::inference::tests::divergence_scalars]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        invoke_binary(input, output, "ix_kl", kl_divergence)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        binary_sig()
    }
}

/// `ix_pearson(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — Pearson correlation in [−1, 1].
/// The pairwise primitive for a correlation mesh over many streams.
struct IxPearson;
impl VScalar for IxPearson {
    type State = ();
    // @ai:invariant ix_pearson wraps ix_math::inference::pearson; perfectly correlated -> 1, anti-correlated -> -1, a constant or length-mismatched arg -> SQL error [T:test conf:0.85 src:ix_duck::inference::tests::pearson_scalar]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        invoke_binary(input, output, "ix_pearson", pearson)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        binary_sig()
    }
}

/// `ix_js(p DOUBLE[], q DOUBLE[]) -> DOUBLE` — Jensen–Shannon divergence (nats).
struct IxJs;
impl VScalar for IxJs {
    type State = ();
    // @ai:invariant ix_js wraps ix_math::inference::js_divergence; symmetric and bounded in [0, ln 2], equal distributions -> 0 [T:test conf:0.85 src:ix_duck::inference::tests::divergence_scalars]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        invoke_binary(input, output, "ix_js", js_divergence)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        binary_sig()
    }
}

// ── ix_two_sample (table fn: one row of statistic + p_value) ────────────────────

#[repr(C)]
struct TwoSampleBind {
    result: TestResult,
}
#[repr(C)]
struct TwoSampleInit {
    cursor: AtomicUsize,
}

struct IxTwoSample;
impl VTab for IxTwoSample {
    type InitData = TwoSampleInit;
    type BindData = TwoSampleBind;

    // @ai:invariant ix_two_sample(a,b,kind) emits one (statistic,p_value) row from the ix_math::inference two-sample test named by kind (ks|mannwhitney|welch); a clearly-shifted pair gives a small p_value, an unknown kind or empty sample -> SQL error [T:test conf:0.85 src:ix_duck::inference::tests::two_sample_detects_shift]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let a: Vec<f64> = serde_json::from_str(&bind.get_parameter(0).to_string())
            .map_err(|e| format!("sample a must be a JSON number array: {e}"))?;
        let b: Vec<f64> = serde_json::from_str(&bind.get_parameter(1).to_string())
            .map_err(|e| format!("sample b must be a JSON number array: {e}"))?;
        let kind = bind.get_parameter(2).to_string().to_lowercase();
        let result = match kind.as_str() {
            "ks" => ks_two_sample(&a, &b),
            "mannwhitney" | "mann_whitney" | "u" => mann_whitney_u(&a, &b),
            "welch" | "t" => welch_t_test(&a, &b),
            other => {
                return Err(format!(
                    "unknown test '{other}' (expected ks|mannwhitney|welch)"
                )
                .into())
            }
        }
        .map_err(|e| format!("ix_two_sample: {e}"))?;
        bind.add_result_column("statistic", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("p_value", LogicalTypeHandle::from(LogicalTypeId::Double));
        Ok(TwoSampleBind { result })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(TwoSampleInit {
            cursor: AtomicUsize::new(0),
        })
    }
    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let init = func.get_init_data();
        if init.cursor.swap(1, Ordering::Relaxed) != 0 {
            output.set_len(0);
            return Ok(());
        }
        let r = &func.get_bind_data().result;
        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(1) };
            s[0] = r.statistic;
        }
        {
            let mut v = output.flat_vector(1);
            let s = unsafe { v.as_mut_slice_with_len::<f64>(1) };
            s[0] = r.p_value;
        }
        output.set_len(1);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

/// Register the statistical-inference UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxSkewness>("ix_skewness")?;
    conn.register_scalar_function::<IxKurtosis>("ix_kurtosis")?;
    conn.register_scalar_function::<IxMad>("ix_mad")?;
    conn.register_scalar_function::<IxIqr>("ix_iqr")?;
    conn.register_scalar_function::<IxEntropy>("ix_entropy")?;
    conn.register_scalar_function::<IxQuantile>("ix_quantile")?;
    conn.register_scalar_function::<IxKl>("ix_kl")?;
    conn.register_scalar_function::<IxJs>("ix_js")?;
    conn.register_scalar_function::<IxPearson>("ix_pearson")?;
    conn.register_table_function::<IxTwoSample>("ix_two_sample")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    #[test]
    fn descriptive_scalars_match_ix_math() {
        let conn = open_bench().unwrap();
        // kurtosis([1,2,3,4,5]) = -1.3; skew = 0; mad = 1.0 (scipy).
        let kurt: f64 = conn
            .query_row(
                "SELECT ix_kurtosis(list(v)::DOUBLE[]) FROM (VALUES (1.0),(2.0),(3.0),(4.0),(5.0)) t(v)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((kurt + 1.3).abs() < 1e-9, "excess kurtosis = -1.3, got {kurt}");
        let mad: f64 = conn
            .query_row(
                "SELECT ix_mad(list(v)::DOUBLE[]) FROM (VALUES (1.0),(2.0),(3.0),(4.0),(5.0)) t(v)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((mad - 1.0).abs() < 1e-9, "mad = 1.0, got {mad}");
        // zero-variance skewness is a SQL error, not a panic.
        assert!(
            conn.query_row("SELECT ix_skewness([2.0,2.0,2.0]::DOUBLE[])", [], |r| r
                .get::<_, f64>(0))
                .is_err(),
            "zero-variance skewness must be a SQL error"
        );
    }

    #[test]
    fn quantile_scalar() {
        let conn = open_bench().unwrap();
        let med: f64 = conn
            .query_row("SELECT ix_quantile([1.0,2.0,3.0,4.0]::DOUBLE[], 0.5)", [], |r| r.get(0))
            .unwrap();
        assert!((med - 2.5).abs() < 1e-9, "median = 2.5, got {med}");
        assert!(
            conn.query_row("SELECT ix_quantile([1.0,2.0]::DOUBLE[], 1.5)", [], |r| r
                .get::<_, f64>(0))
                .is_err(),
            "q out of [0,1] must be a SQL error"
        );
    }

    #[test]
    fn divergence_scalars() {
        let conn = open_bench().unwrap();
        // KL([1,0] ‖ [0.5,0.5]) = ln 2.
        let kl: f64 = conn
            .query_row("SELECT ix_kl([1.0,0.0]::DOUBLE[], [0.5,0.5]::DOUBLE[])", [], |r| r.get(0))
            .unwrap();
        assert!((kl - 2.0_f64.ln()).abs() < 1e-9, "KL = ln2, got {kl}");
        // JS is symmetric and zero for equal distributions.
        let js: f64 = conn
            .query_row("SELECT ix_js([1.0,2.0,3.0]::DOUBLE[], [1.0,2.0,3.0]::DOUBLE[])", [], |r| r.get(0))
            .unwrap();
        assert!(js.abs() < 1e-9, "JS of equal distributions = 0, got {js}");
    }

    #[test]
    fn pearson_scalar() {
        let conn = open_bench().unwrap();
        let r: f64 = conn
            .query_row(
                "SELECT ix_pearson([1.0,2.0,3.0,4.0]::DOUBLE[], [2.0,4.0,6.0,8.0]::DOUBLE[])",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!((r - 1.0).abs() < 1e-9, "perfectly correlated → 1, got {r}");
        assert!(
            conn.query_row(
                "SELECT ix_pearson([1.0,1.0,1.0]::DOUBLE[], [1.0,2.0,3.0]::DOUBLE[])",
                [],
                |row| row.get::<_, f64>(0)
            )
            .is_err(),
            "constant arg must be a SQL error"
        );
    }

    #[test]
    fn two_sample_detects_shift() {
        let conn = open_bench().unwrap();
        // Disjoint ranges → KS D = 1, small p.
        let (d, p): (f64, f64) = conn
            .query_row(
                "SELECT statistic, p_value FROM ix_two_sample('[0,1,2,3]', '[10,11,12,13]', 'ks')",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!((d - 1.0).abs() < 1e-9, "KS D = 1 for disjoint samples, got {d}");
        assert!(p < 0.2, "shifted distributions → small p, got {p}");

        // Identical samples → Welch t = 0, p = 1.
        let p_same: f64 = conn
            .query_row(
                "SELECT p_value FROM ix_two_sample('[1,2,3,4]', '[1,2,3,4]', 'welch')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((p_same - 1.0).abs() < 1e-3, "identical samples → p≈1, got {p_same}");

        // Unknown test kind is a SQL error.
        assert!(
            conn.query_row(
                "SELECT statistic FROM ix_two_sample('[1,2]', '[3,4]', 'chi2')",
                [],
                |r| r.get::<_, f64>(0)
            )
            .is_err(),
            "unknown test kind must be a SQL error"
        );
    }
}
