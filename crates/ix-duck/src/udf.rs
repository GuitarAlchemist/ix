//! IX algorithms exposed as DuckDB scalar UDFs (via the `VScalar` trait).
//!
//! The pairwise distance UDFs (`ix_cosine`, `ix_euclidean`, `ix_manhattan`,
//! `ix_chebyshev`, `ix_cosine_distance`) each take two `DOUBLE[]` (LIST<DOUBLE>)
//! and return a `DOUBLE`, wrapping the real `ix_math::distance` functions (no
//! reimplementation). `ix_euclidean` is the primitive for the kNN-distance / OOD
//! SQL recipe (`ORDER BY ix_euclidean(q, r) LIMIT k`); `ix_cosine_distance`
//! (= 1 − cosine_similarity) is the metric-space companion for the same recipe.
//! `ix_minkowski(a, b, p)` adds a third scalar exponent (p=1 → manhattan, p=2 →
//! euclidean) and so has its own invoke path.
//!
//! Set-relative operations (`ix_pca_project`, `ix_silhouette`) are table functions
//! in the sibling `tablefn` module; [`register_all`] registers both surfaces.

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_math::distance::{
    chebyshev, cosine_distance, cosine_similarity, euclidean, manhattan, minkowski,
};
use ix_math::error::MathError;
use ndarray::Array1;

/// The `LIST<DOUBLE>` logical type used for both vector arguments.
fn list_double() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double))
}

/// Extract every row of a `LIST<DOUBLE>` column as owned `Vec<f64>`.
///
/// Reads the per-row `(offset, length)` entries *before* borrowing the child
/// buffer so the child slice and the entry lookups never alias the same vector.
/// `cap` is `max(offset + length)`, so every `all[o..o + l]` slice is in bounds
/// by construction — a malformed entry cannot index past the child buffer.
// Reads the raw (offset,length) slice for EVERY row, including NULL list rows (whose
// entry is typically (0,0) → an empty Vec). NULL handling is the caller's job: the
// scalar UDFs read per-row validity separately and emit SQL NULL (see `invoke_pairwise`),
// so a NULL row's harmless raw read is overridden downstream.
pub(crate) fn read_list_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<Vec<f64>> {
    let lv = input.list_vector(col);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<f64>(cap) };
    entries
        .iter()
        .map(|&(o, l)| all[o..o + l].to_vec())
        .collect()
}

/// Like [`read_list_col`], but consults the child vector's validity mask and yields `None`
/// for any row containing a SQL NULL *element*.
///
/// A NULL element is invisible to [`read_list_col`]: the child buffer still holds a double at
/// that slot (whatever was last there, or zero), so reading it raw invents a sample out of
/// nothing. Row validity — the mask [`null_mask`] reads — only covers a NULL *list*, not a
/// list of partly-NULL values, so the two checks are independent and a caller that cares
/// about element-level nullity needs this reader.
pub(crate) fn read_list_col_checked(
    input: &mut DataChunkHandle,
    col: usize,
    n: usize,
) -> Vec<Option<Vec<f64>>> {
    let lv = input.list_vector(col);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let child_null: Vec<bool> = (0..cap).map(|i| child.row_is_null(i as u64)).collect();
    let all = unsafe { child.as_slice_with_len::<f64>(cap) };
    entries
        .iter()
        .map(|&(o, l)| (!child_null[o..o + l].iter().any(|&b| b)).then(|| all[o..o + l].to_vec()))
        .collect()
}

/// Shared row-wise driver: read two `LIST<DOUBLE>` columns, apply `metric`
/// pairwise, write a `DOUBLE` per row. A metric error (e.g. dimension mismatch)
/// becomes a DuckDB SQL error rather than a panic.
// @ai:invariant a NULL list argument (either side) yields SQL NULL, never a spurious number: per-row validity is read via FlatVector::row_is_null and the output row flagged via FlatVector::set_null [T:test conf:0.9 src:ix_duck::tests::ix_cosine_passes_null_through]
fn invoke_pairwise(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    name: &str,
    metric: impl Fn(&Array1<f64>, &Array1<f64>) -> Result<f64, MathError>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = input.len();
    // SQL NULL passthrough: capture each argument column's per-row validity *before*
    // borrowing it as a list. A list vector's NULL mask lives on the vector itself, so a
    // `FlatVector` view of the same column pointer reads list-row nullity correctly.
    let a_null = null_mask(input, 0, n);
    let b_null = null_mask(input, 1, n);
    let a = read_list_col(input, 0, n);
    let b = read_list_col(input, 1, n);
    let mut out = output.flat_vector();
    {
        let out_slice = unsafe { out.as_mut_slice_with_len::<f64>(n) };
        for i in 0..n {
            if a_null[i] || b_null[i] {
                out_slice[i] = 0.0; // placeholder; the row is flagged NULL below
                continue;
            }
            let av = Array1::from_vec(a[i].clone());
            let bv = Array1::from_vec(b[i].clone());
            out_slice[i] = metric(&av, &bv).map_err(|e| format!("{name}: {e}"))?;
        }
    }
    // Flag NULL rows after the value slice is dropped (set_null re-borrows the vector).
    for (i, (&an, &bn)) in a_null.iter().zip(&b_null).enumerate() {
        if an || bn {
            out.set_null(i);
        }
    }
    Ok(())
}

/// Per-row NULL mask of a `LIST` column. A list vector's validity mask is on the
/// vector itself, so a `FlatVector` view of the same column pointer reads it.
pub(crate) fn null_mask(input: &DataChunkHandle, col: usize, n: usize) -> Vec<bool> {
    let v = input.flat_vector(col);
    (0..n).map(|i| v.row_is_null(i as u64)).collect()
}

/// `ix_cosine(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — cosine similarity in [-1, 1].
struct IxCosine;

impl VScalar for IxCosine {
    type State = ();

    // @ai:invariant ix_cosine returns ix_math cosine_similarity of its two DOUBLE[] args; identical vectors -> 1.0, orthogonal -> 0.0, dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_cosine_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pairwise(input, output, "ix_cosine", cosine_similarity)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), list_double()],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// `ix_euclidean(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — L2 distance. Primitive for
/// the kNN-distance / OOD SQL recipe.
struct IxEuclidean;

impl VScalar for IxEuclidean {
    type State = ();

    // @ai:invariant ix_euclidean returns ix_math euclidean L2 distance of its two DOUBLE[] args; equal vectors -> 0.0, dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_euclidean_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pairwise(input, output, "ix_euclidean", euclidean)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), list_double()],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// `ix_manhattan(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — L1 (city-block) distance.
struct IxManhattan;

impl VScalar for IxManhattan {
    type State = ();

    // @ai:invariant ix_manhattan returns ix_math manhattan L1 distance of its two DOUBLE[] args; equal vectors -> 0.0, dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_manhattan_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pairwise(input, output, "ix_manhattan", manhattan)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), list_double()],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// `ix_chebyshev(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — L∞ (max-coordinate) distance.
struct IxChebyshev;

impl VScalar for IxChebyshev {
    type State = ();

    // @ai:invariant ix_chebyshev returns ix_math chebyshev L∞ distance of its two DOUBLE[] args; equal vectors -> 0.0, dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_chebyshev_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pairwise(input, output, "ix_chebyshev", chebyshev)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), list_double()],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// `ix_cosine_distance(a DOUBLE[], b DOUBLE[]) -> DOUBLE` — `1 − cosine_similarity`,
/// in `[0, 2]`. The metric-space companion to `ix_cosine` for kNN-distance / OOD.
struct IxCosineDistance;

impl VScalar for IxCosineDistance {
    type State = ();

    // @ai:invariant ix_cosine_distance returns ix_math cosine_distance (1 − cosine_similarity) of its two DOUBLE[] args; identical vectors -> 0.0, orthogonal -> 1.0, dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_cosine_distance_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pairwise(input, output, "ix_cosine_distance", cosine_distance)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_double(), list_double()],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// `ix_minkowski(a DOUBLE[], b DOUBLE[], p DOUBLE) -> DOUBLE` — the Lᵖ family.
/// `p = 1` reduces to manhattan, `p = 2` to euclidean. Unlike the 2-arg metrics
/// this reads a third (scalar `DOUBLE`) argument, so it has its own invoke path.
struct IxMinkowski;

impl VScalar for IxMinkowski {
    type State = ();

    // @ai:invariant ix_minkowski(a,b,p) returns ix_math minkowski Lᵖ distance; p=1 equals ix_manhattan, p=2 equals ix_euclidean; a NULL list or NULL p yields SQL NULL; dimension mismatch -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::tests::ix_minkowski_matches_ix_math]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let a_null = null_mask(input, 0, n);
        let b_null = null_mask(input, 1, n);
        let p_null = null_mask(input, 2, n);
        let a = read_list_col(input, 0, n);
        let b = read_list_col(input, 1, n);
        // The `p` exponent is a flat DOUBLE column (one value per row).
        let p_vals: Vec<f64> = {
            let pv = input.flat_vector(2);
            let s = pv.as_slice_with_len::<f64>(n);
            s[..n].to_vec()
        };
        let mut out = output.flat_vector();
        {
            let out_slice = out.as_mut_slice_with_len::<f64>(n);
            for i in 0..n {
                if a_null[i] || b_null[i] || p_null[i] {
                    out_slice[i] = 0.0; // placeholder; row flagged NULL below
                    continue;
                }
                let av = Array1::from_vec(a[i].clone());
                let bv = Array1::from_vec(b[i].clone());
                out_slice[i] =
                    minkowski(&av, &bv, p_vals[i]).map_err(|e| format!("ix_minkowski: {e}"))?;
            }
        }
        for i in 0..n {
            if a_null[i] || b_null[i] || p_null[i] {
                out.set_null(i);
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                list_double(),
                list_double(),
                LogicalTypeHandle::from(LogicalTypeId::Double),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// Register every IX UDF on `conn` — scalar (`ix_cosine`, `ix_euclidean`,
/// `ix_manhattan`, `ix_chebyshev`, `ix_cosine_distance`) and table
/// (`ix_pca_project`, `ix_silhouette`) functions.
pub fn register_all(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxCosine>("ix_cosine")?;
    conn.register_scalar_function::<IxEuclidean>("ix_euclidean")?;
    conn.register_scalar_function::<IxManhattan>("ix_manhattan")?;
    conn.register_scalar_function::<IxChebyshev>("ix_chebyshev")?;
    conn.register_scalar_function::<IxCosineDistance>("ix_cosine_distance")?;
    conn.register_scalar_function::<IxMinkowski>("ix_minkowski")?;
    crate::tablefn::register(conn)?;
    crate::bracelet::register(conn)?;
    crate::serial::register(conn)?;
    crate::graphsig::register(conn)?;
    crate::grothendieck::register(conn)?;
    crate::eval::register(conn)?;
    crate::sketch::register(conn)?;
    crate::code::register(conn)?;
    crate::inference::register(conn)?;
    crate::supervised::register(conn)?;
    crate::hexavalent::register(conn)?;
    crate::fractal::register(conn)?;
    crate::petri::register(conn)?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    fn poisoned_null_list() -> DataChunkHandle {
        let input = DataChunkHandle::new(&[list_double()]);
        input.set_len(3);
        {
            let mut list = input.list_vector(0);
            // All child elements used by valid rows are initialized.
            unsafe { list.set_child(&[1.0_f64, 2.0, 3.0, 4.0]) };
            list.set_entry(0, 0, 2);
            list.set_entry(1, usize::MAX, 1);
            list.set_null(1);
            list.set_entry(2, 2, 2);
        }
        input
    }

    #[test]
    fn list_reader_ignores_null_row_descriptor() {
        let mut input = poisoned_null_list();
        assert_eq!(
            read_list_col(&mut input, 0, 3),
            vec![vec![1.0, 2.0], vec![], vec![3.0, 4.0]],
        );
    }

    #[test]
    fn checked_list_reader_ignores_null_row_descriptor() {
        let mut input = poisoned_null_list();
        assert_eq!(
            read_list_col_checked(&mut input, 0, 3),
            vec![Some(vec![1.0, 2.0]), None, Some(vec![3.0, 4.0])],
        );
    }

    struct PoisonedNullCosine;

    impl VScalar for PoisonedNullCosine {
        type State = ();

        unsafe fn invoke(
            _: &Self::State,
            input: &mut DataChunkHandle,
            output: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let n = input.len();
            for col in 0..2 {
                let nulls = null_mask(input, col, n);
                let mut list = input.list_vector(col);
                for (row, is_null) in nulls.iter().enumerate() {
                    if *is_null {
                        // A NULL row's payload is unusable; only its validity
                        // bit is meaningful. Make accidental reads deterministic.
                        list.set_entry(row, usize::MAX, 1);
                    }
                }
            }
            unsafe { IxCosine::invoke(&(), input, output) }
        }

        fn signatures() -> Vec<ScalarFunctionSignature> {
            IxCosine::signatures()
        }
    }

    #[test]
    fn cosine_null_descriptors_do_not_reach_child_buffer() {
        let conn = Connection::open_in_memory().unwrap();
        conn.register_scalar_function::<PoisonedNullCosine>("poisoned_null_cosine")
            .unwrap();
        let mut stmt = conn.prepare(
            "SELECT poisoned_null_cosine(a, b) FROM (VALUES \
                (1, [1.0,0.0]::DOUBLE[], [1.0,0.0]::DOUBLE[]), \
                (2, NULL::DOUBLE[], [1.0,0.0]::DOUBLE[]), \
                (3, [1.0,0.0]::DOUBLE[], NULL::DOUBLE[]), \
                (4, NULL::DOUBLE[], NULL::DOUBLE[]), \
                (5, [1.0,0.0]::DOUBLE[], [0.0,1.0]::DOUBLE[]) \
             ) t(i,a,b) ORDER BY i",
        ).unwrap();
        let rows: Vec<Option<f64>> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(rows.len(), 5);
        assert!((rows[0].unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(&rows[1..4], &[None, None, None]);
        assert!(rows[4].unwrap().abs() < 1e-12);
    }

    #[test]
    fn checked_reader_distinguishes_null_elements_empty_lists_and_null_lists() {
        let mut input = DataChunkHandle::new(&[list_double()]);
        input.set_len(4);
        {
            let mut list = input.list_vector(0);
            unsafe { list.set_child(&[1.0_f64, 2.0, 4.0]) };
            list.child(3).set_null(1);
            list.set_entry(0, 0, 2);
            list.set_entry(1, 2, 0);
            list.set_entry(2, usize::MAX, 1);
            list.set_null(2);
            list.set_entry(3, 2, 1);
        }
        assert_eq!(
            read_list_col_checked(&mut input, 0, 4),
            vec![None, Some(vec![]), None, Some(vec![4.0])],
        );
    }

    #[test]
    fn empty_and_null_lists_do_not_read_child_storage() {
        let mut input = DataChunkHandle::new(&[list_double()]);
        input.set_len(2);
        {
            let mut list = input.list_vector(0);
            list.set_entry(0, 0, 0);
            list.set_entry(1, usize::MAX, 1);
            list.set_null(1);
        }
        assert_eq!(read_list_col(&mut input, 0, 2), vec![Vec::<f64>::new(), Vec::<f64>::new()]);
        assert_eq!(
            read_list_col_checked(&mut input, 0, 2),
            vec![Some(vec![]), None],
        );
    }
}
