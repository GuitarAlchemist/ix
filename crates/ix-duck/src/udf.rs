//! IX algorithms exposed as DuckDB scalar UDFs (via the `VScalar` trait).
//!
//! `ix_cosine(a, b)` and `ix_euclidean(a, b)` take two `DOUBLE[]` (LIST<DOUBLE>)
//! and return a `DOUBLE`, wrapping the real `ix_math::distance` functions (no
//! reimplementation). `ix_euclidean` is the primitive for the kNN-distance / OOD
//! SQL recipe (`ORDER BY ix_euclidean(q, r) LIMIT k`).
//!
//! Set-relative operations (`ix_pca_project`, `ix_silhouette`) are table functions
//! in the sibling `tablefn` module; [`register_all`] registers both surfaces.

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_math::distance::{cosine_similarity, euclidean};
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
fn null_mask(input: &DataChunkHandle, col: usize, n: usize) -> Vec<bool> {
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

/// Register every IX UDF on `conn` — scalar (`ix_cosine`, `ix_euclidean`) and
/// table (`ix_pca_project`, `ix_silhouette`) functions.
pub fn register_all(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxCosine>("ix_cosine")?;
    conn.register_scalar_function::<IxEuclidean>("ix_euclidean")?;
    crate::tablefn::register(conn)?;
    crate::bracelet::register(conn)?;
    crate::serial::register(conn)?;
    crate::graphsig::register(conn)?;
    crate::eval::register(conn)?;
    crate::sketch::register(conn)?;
    crate::code::register(conn)?;
    Ok(())
}
