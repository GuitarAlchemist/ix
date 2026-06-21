//! Supervised fit/predict UDFs over `ix-supervised`.
//!
//! DuckDB has `regr_slope`/`regr_intercept` (simple OLS aggregates) but nothing
//! that fits a *multivariate* model you can persist and re-apply. These UDFs do,
//! following the same **fit → blob → predict** shape as the `sketch` module: fit
//! returns the trained model as a JSON `VARCHAR` you store in a column, then
//! `predict` re-applies it to a feature vector.
//!
//! | model | fit | predict |
//! |---|---|---|
//! | linear   | `ix_linreg_fit(x_json, y_json)`   | `ix_linreg_predict(model, x_json)` → DOUBLE |
//! | logistic | `ix_logistic_fit(x_json, y_json)` | `ix_logistic_predict(model, x_json)` → BIGINT (class) |
//!
//! `x_json` for **fit** is a JSON 2-D array `[[…features…], …]` (one row per
//! sample); `y_json` is a 1-D array (`DOUBLE`s for linear, integer class ids for
//! logistic). `x_json` for **predict** is a single 1-D feature vector. The whole
//! training set enters as one scalar string, so a typical call is
//! `SELECT ix_linreg_fit('[[1],[2],[3]]', '[2,4,6]')`.
//!
//! Pure wraps of `ix-supervised` — linear round-trips through its own
//! `LinearRegressionState`; logistic serializes its public `(weights, bias)`.
//! No model math here.

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::ffi::duckdb_string_t;
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_supervised::linear_regression::{LinearRegression, LinearRegressionState};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::{Classifier, Regressor};
use ndarray::{Array1, Array2};
use std::error::Error;
use std::ffi::CString;

use crate::tablefn::parse_matrix;

type Res = Result<(), Box<dyn Error>>;

fn ty(id: LogicalTypeId) -> LogicalTypeHandle {
    LogicalTypeHandle::from(id)
}

/// Read a `VARCHAR` column into owned `String`s (one per row).
fn read_varchar(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<String> {
    let v = input.flat_vector(col);
    let slice = unsafe { v.as_slice_with_len::<duckdb_string_t>(n) };
    slice
        .iter()
        .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
        .collect()
}

fn write_varchar(output: &mut dyn WritableVector, vals: &[String]) -> Res {
    let out = output.flat_vector();
    for (i, v) in vals.iter().enumerate() {
        out.insert(i, CString::new(v.as_str())?);
    }
    Ok(())
}

fn write_f64(output: &mut dyn WritableVector, vals: &[f64]) {
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<f64>(vals.len()) };
    slice.copy_from_slice(vals);
}

fn write_i64(output: &mut dyn WritableVector, vals: &[i64]) {
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<i64>(vals.len()) };
    slice.copy_from_slice(vals);
}

/// Parse a single 1-D JSON feature vector into a `(1, n_features)` matrix (the
/// `predict` input shape).
fn parse_feature_row(json: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let feat: Vec<f64> =
        serde_json::from_str(json).map_err(|e| format!("expected a JSON number array: {e}"))?;
    if feat.is_empty() {
        return Err("feature vector is empty".into());
    }
    let n = feat.len();
    Ok(Array2::from_shape_vec((1, n), feat)?)
}

// ── linear regression ──────────────────────────────────────────────────────────

struct IxLinregFit;
impl VScalar for IxLinregFit {
    type State = ();
    // @ai:invariant ix_linreg_fit fits ix_supervised LinearRegression over the JSON (x 2-D, y 1-D) and returns its LinearRegressionState as a JSON blob; mismatched row counts -> SQL error (no panic) [T:test conf:0.85 src:ix_duck::supervised::tests::linreg_fit_predict_recovers_slope]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let xs = read_varchar(input, 0, n);
        let ys = read_varchar(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = parse_matrix(&xs[i])?;
            let y: Vec<f64> = serde_json::from_str(&ys[i])
                .map_err(|e| format!("ix_linreg_fit: y must be a JSON number array: {e}"))?;
            if y.len() != x.nrows() {
                return Err(format!(
                    "ix_linreg_fit: y length ({}) != number of samples ({})",
                    y.len(),
                    x.nrows()
                )
                .into());
            }
            let mut model = LinearRegression::new();
            model.fit(&x, &Array1::from_vec(y));
            let state = model
                .save_state()
                .ok_or("ix_linreg_fit: model did not fit")?;
            out.push(serde_json::to_string(&state)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxLinregPredict;
impl VScalar for IxLinregPredict {
    type State = ();
    // @ai:invariant ix_linreg_predict reloads a LinearRegressionState blob and returns the model's prediction for the JSON feature vector; a model fit on y=2x predicts ~2*x [T:test conf:0.85 src:ix_duck::supervised::tests::linreg_fit_predict_recovers_slope]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let models = read_varchar(input, 0, n);
        let xs = read_varchar(input, 1, n);
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let state: LinearRegressionState = serde_json::from_str(&models[i])
                .map_err(|e| format!("ix_linreg_predict: invalid model blob: {e}"))?;
            let model = LinearRegression::load_state(&state);
            let x = parse_feature_row(&xs[i])?;
            out[i] = model.predict(&x)[0];
        }
        write_f64(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
            ty(LogicalTypeId::Double),
        )]
    }
}

// ── logistic regression ────────────────────────────────────────────────────────

struct IxLogisticFit;
impl VScalar for IxLogisticFit {
    type State = ();
    // @ai:invariant ix_logistic_fit fits ix_supervised LogisticRegression over the JSON (x 2-D, y 1-D class ids) and returns its (weights, bias) as a JSON blob; mismatched row counts -> SQL error (no panic) [T:test conf:0.85 src:ix_duck::supervised::tests::logistic_fit_predict_separates_classes]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let xs = read_varchar(input, 0, n);
        let ys = read_varchar(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = parse_matrix(&xs[i])?;
            let y: Vec<usize> = serde_json::from_str(&ys[i])
                .map_err(|e| format!("ix_logistic_fit: y must be a JSON int array: {e}"))?;
            if y.len() != x.nrows() {
                return Err(format!(
                    "ix_logistic_fit: y length ({}) != number of samples ({})",
                    y.len(),
                    x.nrows()
                )
                .into());
            }
            let mut model = LogisticRegression::new();
            model.fit(&x, &Array1::from_vec(y));
            let weights = model
                .weights
                .as_ref()
                .ok_or("ix_logistic_fit: model did not fit")?
                .to_vec();
            // LogisticRegression has no save_state; its (weights, bias) fully
            // determine predictions, so serialize that tuple directly.
            out.push(serde_json::to_string(&(weights, model.bias))?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxLogisticPredict;
impl VScalar for IxLogisticPredict {
    type State = ();
    // @ai:invariant ix_logistic_predict reloads a (weights, bias) blob and returns the predicted class id for the JSON feature vector; points on opposite sides of a fitted boundary get different classes [T:test conf:0.85 src:ix_duck::supervised::tests::logistic_fit_predict_separates_classes]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let models = read_varchar(input, 0, n);
        let xs = read_varchar(input, 1, n);
        let mut out = vec![0i64; n];
        for i in 0..n {
            let (weights, bias): (Vec<f64>, f64) = serde_json::from_str(&models[i])
                .map_err(|e| format!("ix_logistic_predict: invalid model blob: {e}"))?;
            let mut model = LogisticRegression::new();
            model.weights = Some(Array1::from_vec(weights));
            model.bias = bias;
            let x = parse_feature_row(&xs[i])?;
            out[i] = model.predict(&x)[0] as i64;
        }
        write_i64(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
            ty(LogicalTypeId::Bigint),
        )]
    }
}

/// Register the supervised fit/predict UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxLinregFit>("ix_linreg_fit")?;
    conn.register_scalar_function::<IxLinregPredict>("ix_linreg_predict")?;
    conn.register_scalar_function::<IxLogisticFit>("ix_logistic_fit")?;
    conn.register_scalar_function::<IxLogisticPredict>("ix_logistic_predict")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    #[test]
    fn linreg_fit_predict_recovers_slope() {
        let conn = open_bench().unwrap();
        // Fit y = 2x on a perfect line, then predict at x=6 → ~12.
        let pred: f64 = conn
            .query_row(
                "SELECT ix_linreg_predict(\
                    ix_linreg_fit('[[1],[2],[3],[4],[5]]', '[2,4,6,8,10]'), '[6]')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((pred - 12.0).abs() < 1e-6, "y=2x model predicts ~12 at x=6, got {pred}");
    }

    #[test]
    fn linreg_fit_rejects_mismatched_lengths() {
        let conn = open_bench().unwrap();
        assert!(
            conn.query_row(
                "SELECT ix_linreg_fit('[[1],[2],[3]]', '[2,4]')",
                [],
                |r| r.get::<_, String>(0)
            )
            .is_err(),
            "y shorter than x must be a SQL error"
        );
    }

    #[test]
    fn logistic_fit_predict_separates_classes() {
        let conn = open_bench().unwrap();
        // 1-D, linearly separable: class 0 below ~2.5, class 1 above. A point at
        // x=0 should predict class 0; x=5 should predict class 1.
        let model = "ix_logistic_fit('[[0],[1],[2],[3],[4],[5]]', '[0,0,0,1,1,1]')";
        let lo: i64 = conn
            .query_row(&format!("SELECT ix_logistic_predict({model}, '[0]')"), [], |r| r.get(0))
            .unwrap();
        let hi: i64 = conn
            .query_row(&format!("SELECT ix_logistic_predict({model}, '[5]')"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(lo, 0, "x=0 is in the low class");
        assert_eq!(hi, 1, "x=5 is in the high class");
    }
}
