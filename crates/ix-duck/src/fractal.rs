//! Fractal-curve UDFs over `ix-fractal` + `ix-chaos` (ix#203).
//!
//! Minimum vertical slice per `docs/plans/2026-07-20-ix-fractal-takagi-derham-exposure.md`
//! §"Research Insights" (recommended-minimum path): `ix_takagi` is the one Takagi function
//! that maps cleanly onto a scalar UDF (`takagi_series` is already covered by MCP `ix_fractal`
//! and is strictly dominated in SQL — see the plan §2); `ix_hurst` closes the loop by making
//! the roughness a generated de Rham curve exhibits *measurable* from SQL. De Rham generation
//! is exposed on MCP only (`ix_fractal { operation: "de_rham_1d" }`, added by this same slice),
//! not as a DuckDB table function — see the module note below and SKILL.md.
//! Pure wraps — no fractal math reimplemented here.
//!
//! | UDF | wraps |
//! |---|---|
//! | `ix_takagi(t DOUBLE, terms BIGINT) -> DOUBLE` | `ix_fractal::takagi::takagi` |
//! | `ix_hurst(x DOUBLE[]) -> DOUBLE` | `ix_chaos::fractal::hurst_exponent` |
//!
//! `ix_hurst` is order-sensitive (R/S over windows) — callers must materialize with
//! `list(value ORDER BY i)`, not bare `list(value)`. It yields SQL NULL rather than a
//! number for anything R/S cannot score: fewer than 8 samples (the callee's `0.5` sentinel
//! range), an empty list, a non-finite sample, or a list containing a NULL element.

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_chaos::fractal::hurst_exponent;
use ix_fractal::takagi::takagi;

use crate::udf::{null_mask, read_list_col_checked};

type Res = Result<(), Box<dyn std::error::Error>>;

fn double() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Double)
}
fn bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Bigint)
}
fn list_double() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double))
}

/// `ix_takagi(t DOUBLE, terms BIGINT) -> DOUBLE` — the Blancmange (Takagi) function.
struct IxTakagi;
impl VScalar for IxTakagi {
    type State = ();
    // @ai:invariant ix_takagi(t,terms) wraps ix_fractal::takagi::takagi; NULL on either arg -> SQL NULL [T:test conf:0.85 src:ix_duck::fractal::tests::takagi_known_values]
    // @ai:invariant ix_takagi rejects terms < 0 with a SQL error, and terms > 53 is silently capped by the callee (not re-checked here), so terms=54.. is equivalent to terms=53 [T:test conf:0.9 src:ix_duck::fractal::tests::takagi_terms_boundaries]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let t_null = null_mask(input, 0, n);
        let terms_null = null_mask(input, 1, n);
        let ts = { let v = input.flat_vector(0); v.as_slice_with_len::<f64>(n)[..n].to_vec() };
        let terms = { let v = input.flat_vector(1); v.as_slice_with_len::<i64>(n)[..n].to_vec() };
        let mut out = output.flat_vector();
        {
            let slice = out.as_mut_slice_with_len::<f64>(n);
            for i in 0..n {
                if t_null[i] || terms_null[i] {
                    slice[i] = 0.0; // placeholder; flagged NULL below
                    continue;
                }
                if terms[i] < 0 {
                    return Err("ix_takagi: terms must be >= 0".into());
                }
                slice[i] = takagi(ts[i], terms[i] as usize);
            }
        }
        for i in 0..n {
            if t_null[i] || terms_null[i] {
                out.set_null(i);
            }
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![double(), bigint()], double())]
    }
}

/// Minimum sample count R/S analysis can actually score.
///
/// Mirrors the `n < 8` early return in `ix_chaos::fractal::hurst_exponent`, which returns a
/// hard-coded `0.5` below it. That sentinel is indistinguishable from a genuine "Brownian"
/// estimate, so `ix_hurst` refuses the input rather than passing the number on.
const HURST_MIN_SAMPLES: usize = 8;

/// Estimate, or `None` for an input R/S analysis cannot honestly score.
///
/// Two rejections, both of which the callee would otherwise answer with a number: too few
/// samples (its `0.5` sentinel) and any non-finite sample (which poisons the log/regression
/// sums into a `NaN` — a `NaN` DOUBLE in SQL, not a NULL).
fn hurst_or_null(xs: &[f64]) -> Option<f64> {
    if xs.len() < HURST_MIN_SAMPLES || !xs.iter().all(|v| v.is_finite()) {
        return None;
    }
    let h = hurst_exponent(xs);
    h.is_finite().then_some(h)
}

/// `ix_hurst(x DOUBLE[]) -> DOUBLE` — Hurst exponent (uncorrected R/S) estimating
/// long-range dependence / roughness. H<0.5 rough (anti-persistent), H~0.5 Brownian,
/// H>0.5 smooth (persistent). Order-sensitive: materialize with `list(x ORDER BY i)`.
///
/// Returns SQL NULL — never a number — when the input cannot be scored: a NULL list, a list
/// containing a NULL element, fewer than [`HURST_MIN_SAMPLES`] samples (including an empty
/// list), or any non-finite sample. NULL rather than a SQL error, matching the module's NULL
/// convention and so that one unusable group does not abort the whole query.
struct IxHurst;
impl VScalar for IxHurst {
    type State = ();
    // @ai:invariant ix_hurst(x) wraps ix_chaos::fractal::hurst_exponent over a LIST<DOUBLE>; NULL -> SQL NULL; a lower-roughness de Rham curve (small roughness param) scores a higher Hurst estimate than a higher-roughness one on the same depth/seed [T:test conf:0.7 src:ix_duck::fractal::tests::hurst_orders_roughness]
    // @ai:invariant ix_hurst returns SQL NULL, never a number, for an unscorable input: < 8 samples (incl. empty), any non-finite sample, or any NULL element — element nullity is read from the child validity mask, not inferred from the raw child buffer [T:test conf:0.9 src:ix_duck::fractal::tests::hurst_rejects_inner_nulls]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let a_null = null_mask(input, 0, n);
        let a = read_list_col_checked(input, 0, n);
        let mut null_out = vec![false; n];
        let mut out = output.flat_vector();
        {
            let slice = out.as_mut_slice_with_len::<f64>(n);
            for i in 0..n {
                let h = if a_null[i] {
                    None
                } else {
                    a[i].as_deref().and_then(hurst_or_null)
                };
                match h {
                    Some(v) => slice[i] = v,
                    // placeholder; flagged NULL below
                    None => {
                        slice[i] = 0.0;
                        null_out[i] = true;
                    }
                }
            }
        }
        for (i, &is_null) in null_out.iter().enumerate() {
            if is_null {
                out.set_null(i);
            }
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(vec![list_double()], double())]
    }
}

/// Register the fractal-curve scalar UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxTakagi>("ix_takagi")?;
    conn.register_scalar_function::<IxHurst>("ix_hurst")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;
    use ix_fractal::de_rham::de_rham_curve_1d;
    use rand::SeedableRng;

    fn f(sql: &str) -> f64 {
        open_bench().unwrap().query_row(sql, [], |r| r.get::<_, f64>(0)).unwrap()
    }

    /// Same as [`f`], but keeps SQL NULL distinguishable from a value.
    fn opt(sql: &str) -> Option<f64> {
        open_bench().unwrap().query_row(sql, [], |r| r.get::<_, Option<f64>>(0)).unwrap()
    }

    // S1 — mirrors the crate's own doc-tested values (ix_fractal::takagi doctest).
    #[test]
    fn takagi_known_values() {
        assert!((f("SELECT ix_takagi(0.0, 20)") - 0.0).abs() < 1e-10);
        assert!((f("SELECT ix_takagi(0.5, 20)") - 0.5).abs() < 1e-10);
        assert!((f("SELECT ix_takagi(1.0, 20)") - 0.0).abs() < 1e-10);
    }

    // Boundaries the invariant above actually claims: the wrap's own `terms < 0` guard,
    // and the callee's silent cap at MAX_TERMS = 53 (so 54+ must neither error nor differ).
    #[test]
    fn takagi_terms_boundaries() {
        let err = open_bench()
            .unwrap()
            .query_row("SELECT ix_takagi(0.5, -1)", [], |r| r.get::<_, f64>(0));
        assert!(err.is_err(), "terms < 0 must be a SQL error, not a value");

        assert!((f("SELECT ix_takagi(0.37, 0)") - 0.0).abs() < 1e-12, "terms = 0 is the empty sum");
        let at_cap = f("SELECT ix_takagi(0.37, 53)");
        assert!((f("SELECT ix_takagi(0.37, 54)") - at_cap).abs() < 1e-12, "54 is capped to 53");
        assert!((f("SELECT ix_takagi(0.37, 9999)") - at_cap).abs() < 1e-12, "cap never errors");
    }

    // S3 — the inference.rs NULL convention: NULL in, NULL out, never an error.
    #[test]
    fn takagi_null_propagates() {
        let is_null = open_bench()
            .unwrap()
            .query_row("SELECT ix_takagi(NULL, 20)", [], |r| {
                r.get::<_, Option<f64>>(0)
            })
            .unwrap();
        assert!(is_null.is_none());
    }

    // S4 — vectorized invoke over a real chunk (the bug class a single-row test misses),
    // asserted against the crate's own known Blancmange bound [0, 2/3].
    #[test]
    fn takagi_over_a_column() {
        let n = open_bench()
            .unwrap()
            .query_row(
                "SELECT count(*) FROM (SELECT ix_takagi(i/100.0, 20) v FROM range(101) r(i)) \
                 WHERE v BETWEEN 0 AND 0.6667",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(n, 101);
    }

    // S13 (Rust-side, per the plan's simplicity review — the tracer bullet doesn't need
    // the generator in SQL): generate → ix_hurst, assert direction of roughness agrees.
    // Probes first differences (increments), not the raw curve: feeding the de Rham
    // *values* (an fBm-like profile) pushes H toward 1 and saturates the comparison
    // (plan §"Hurst-estimator caveats", profile-vs-increments gotcha).
    #[test]
    fn hurst_orders_roughness() {
        let mut rng_smooth = rand::rngs::StdRng::seed_from_u64(42);
        let smooth = de_rham_curve_1d(10, 0.05, &mut rng_smooth);
        let mut rng_rough = rand::rngs::StdRng::seed_from_u64(42);
        let rough = de_rham_curve_1d(10, 0.9, &mut rng_rough);

        let diffs = |xs: &ndarray::Array1<f64>| -> Vec<f64> {
            xs.windows(2).into_iter().map(|w| w[1] - w[0]).collect()
        };
        let list_literal = |xs: &[f64]| -> String {
            let parts: Vec<String> = xs.iter().map(|v| v.to_string()).collect();
            format!("[{}]", parts.join(","))
        };
        let h_smooth = f(&format!("SELECT ix_hurst({})", list_literal(&diffs(&smooth))));
        let h_rough = f(&format!("SELECT ix_hurst({})", list_literal(&diffs(&rough))));
        assert!(
            h_smooth > h_rough,
            "expected smoother curve (roughness=0.05) to score a higher Hurst estimate \
             than the rougher one (roughness=0.9) on first differences: \
             h_smooth={h_smooth}, h_rough={h_rough}"
        );
    }

    // S14 — NULL propagates the same way as every other LIST<DOUBLE> scalar.
    #[test]
    fn hurst_null_propagates() {
        assert!(opt("SELECT ix_hurst(NULL)").is_none());
    }

    /// R/S analysis needs at least 8 samples; below that the callee returns a hard-coded
    /// 0.5 — a value indistinguishable from a genuine "Brownian" estimate. SQL NULL is the
    /// only honest answer, so an empty list and a 7-sample list must both be NULL while the
    /// first scorable size (8) is not.
    #[test]
    fn hurst_rejects_undersized_input() {
        assert!(opt("SELECT ix_hurst([]::DOUBLE[])").is_none(), "empty list must be NULL");
        assert!(
            opt("SELECT ix_hurst([1.0,3.0,2.0,5.0,4.0,7.0,6.0])").is_none(),
            "7 samples is below the R/S minimum and must be NULL, not the 0.5 sentinel"
        );
        assert!(
            opt("SELECT ix_hurst([1.0,3.0,2.0,5.0,4.0,7.0,6.0,9.0])").is_some(),
            "8 samples is the first scorable size and must not be NULL"
        );
    }

    /// A non-finite sample poisons the R/S sums and comes back out as NaN, which DuckDB
    /// would surface as a `nan` DOUBLE rather than a NULL. Reject the input instead.
    #[test]
    fn hurst_rejects_non_finite_samples() {
        for bad in ["'nan'::DOUBLE", "'inf'::DOUBLE", "'-inf'::DOUBLE"] {
            let sql =
                format!("SELECT ix_hurst([1.0,3.0,2.0,5.0,4.0,7.0,6.0,{bad}])");
            assert!(opt(&sql).is_none(), "{bad} in the list must yield SQL NULL");
        }
    }

    /// An *inner* NULL is not the same as a NULL list: the row is valid, but the child
    /// buffer holds an unspecified double at that slot. Reading it raw silently invents a
    /// sample; the UDF must consult child validity and return NULL for the whole row.
    #[test]
    fn hurst_rejects_inner_nulls() {
        assert!(
            opt("SELECT ix_hurst([1.0,3.0,2.0,5.0,4.0,7.0,6.0,NULL])").is_none(),
            "a NULL element must yield SQL NULL, never a value read from the raw child buffer"
        );
        assert!(
            opt("SELECT ix_hurst([1.0,NULL,2.0,5.0,4.0,7.0,6.0,9.0])").is_none(),
            "an interior NULL element must yield SQL NULL"
        );
    }

    /// Vectorized path: a bad row must NULL only itself, not poison its neighbours or
    /// abort the query (the failure mode a single-row test cannot see).
    #[test]
    fn hurst_rejects_per_row_not_per_chunk() {
        let (nulls, values) = open_bench()
            .unwrap()
            .query_row(
                "SELECT count(*) FILTER (WHERE h IS NULL), count(h) FROM ( \
                   SELECT ix_hurst(xs) h FROM (VALUES \
                     ([1.0,3.0,2.0,5.0,4.0,7.0,6.0,9.0]), \
                     ([1.0,3.0,2.0]), \
                     ([1.0,3.0,2.0,5.0,4.0,7.0,6.0,NULL]), \
                     ([2.0,4.0,3.0,6.0,5.0,8.0,7.0,10.0]) \
                   ) v(xs) \
                 )",
                [],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
            )
            .unwrap();
        assert_eq!((nulls, values), (2, 2), "exactly the two bad rows must be NULL");
    }
}
