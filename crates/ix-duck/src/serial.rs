//! Twelve-tone serialism scalar UDFs over **ordered** rows (`ix-bracelet::serial`).
//!
//! Each takes a `BIGINT[]` of 12 pitch values (MIDI notes or pitch classes — reduced
//! mod 12) **in row order** and returns a VARCHAR. Unlike the [`crate::bracelet`] UDFs,
//! which read the input as an unordered pitch-class *set*, these preserve order: the
//! input must be a valid tone row (a permutation of all 12 pitch classes) or the result
//! is SQL NULL.
//!
//! - `ix_row_retrograde(notes)`      → the row reversed, e.g. `"[11,10,…,0]"`.
//! - `ix_row_invert(notes)`          → the I₀ inversion, e.g. `"[0,11,10,…]"`.
//! - `ix_row_matrix(notes)`          → the 12×12 row matrix, rows `;`-joined.
//! - `ix_row_combinatoriality(notes)`→ combinatorial levels, e.g. `"P:[6] I:[]"`.
//!
//! Pure wraps of `ix_bracelet::serial` — no theory here. Use to analyse serial-music
//! corpora in SQL the way [`crate::bracelet`] analyses set-class corpora.

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_bracelet::serial::ToneRow;
use std::ffi::CString;

fn list_bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Bigint))
}

fn varchar_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![list_bigint()],
        LogicalTypeHandle::from(LogicalTypeId::Varchar),
    )]
}

/// Read a `LIST<BIGINT>` column into one ordered pitch-class vector per row (values
/// reduced mod 12, order preserved). Mirrors `bracelet::read_pcsets` but keeps order so
/// the values can form a [`ToneRow`]. Reads entries before borrowing the child buffer.
fn read_rows(input: &mut DataChunkHandle, n: usize) -> Vec<Vec<u8>> {
    let lv = input.list_vector(0);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<i64>(cap) };
    entries
        .iter()
        .map(|&(o, l)| all[o..o + l].iter().map(|&v| v.rem_euclid(12) as u8).collect())
        .collect()
}

/// Shared driver: parse each row's values into a [`ToneRow`] and map through `f`.
/// A non-row input (wrong length or not a permutation) → SQL NULL.
fn invoke_row_str(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    f: impl Fn(ToneRow) -> Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = input.len();
    let rows = read_rows(input, n);
    let mut out = output.flat_vector();
    for (i, r) in rows.iter().enumerate() {
        match ToneRow::from_slice(r).ok().and_then(&f) {
            Some(v) => out.insert(i, CString::new(v)?),
            None => out.set_null(i),
        }
    }
    Ok(())
}

/// `[0,1,2,…]` rendering of a row (no spaces, matching the `bracelet` UDFs' style).
fn fmt_row(row: &ToneRow) -> String {
    let pcs: Vec<String> = row.pcs().iter().map(|p| p.to_string()).collect();
    format!("[{}]", pcs.join(","))
}

struct IxRowRetrograde;
impl VScalar for IxRowRetrograde {
    type State = ();
    // @ai:invariant ix_row_retrograde reverses a valid 12-tone row via ix_bracelet serial retrograde; [0,1,..,11] -> "[11,10,..,0]", non-row -> SQL NULL [T:test conf:0.9 src:ix_duck::serial::tests::retrograde_chromatic]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_row_str(input, output, |r| Some(fmt_row(&r.retrograde())))
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxRowInvert;
impl VScalar for IxRowInvert {
    type State = ();
    // @ai:invariant ix_row_invert applies ix_bracelet serial invert (I0) to a valid 12-tone row; [0,1,..,11] -> "[0,11,10,..,1]", non-row -> SQL NULL [T:test conf:0.9 src:ix_duck::serial::tests::invert_chromatic]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_row_str(input, output, |r| Some(fmt_row(&r.invert())))
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxRowMatrix;
impl VScalar for IxRowMatrix {
    type State = ();
    // @ai:invariant ix_row_matrix renders ix_bracelet serial 12x12 matrix of a valid row as 12 bracketed rows joined by ';'; the first row equals the input row [T:test conf:0.9 src:ix_duck::serial::tests::matrix_first_row_is_input]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_row_str(input, output, |r| {
            let rows: Vec<String> = r
                .matrix()
                .iter()
                .map(|mr| {
                    let cells: Vec<String> = mr.iter().map(|c| c.to_string()).collect();
                    format!("[{}]", cells.join(","))
                })
                .collect();
            Some(rows.join(";"))
        })
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxRowCombinatoriality;
impl VScalar for IxRowCombinatoriality {
    type State = ();
    // @ai:invariant ix_row_combinatoriality renders ix_bracelet serial prime+inversion combinatorial levels as "P:[..] I:[..]"; the chromatic row [0..11] is T6-combinatorial so P contains 6 [T:test conf:0.9 src:ix_duck::serial::tests::combinatoriality_chromatic_has_p6]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_row_str(input, output, |r| {
            let p: Vec<String> = r
                .combinatorial_prime_levels()
                .iter()
                .map(|n| n.to_string())
                .collect();
            let inv: Vec<String> = r
                .combinatorial_inversion_levels()
                .iter()
                .map(|n| n.to_string())
                .collect();
            Some(format!("P:[{}] I:[{}]", p.join(","), inv.join(",")))
        })
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

/// Register the twelve-tone serialism scalar UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxRowRetrograde>("ix_row_retrograde")?;
    conn.register_scalar_function::<IxRowInvert>("ix_row_invert")?;
    conn.register_scalar_function::<IxRowMatrix>("ix_row_matrix")?;
    conn.register_scalar_function::<IxRowCombinatoriality>("ix_row_combinatoriality")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    // The chromatic row 0..11 — exact, recognisable transforms.
    const CHROMATIC: &str = "[0,1,2,3,4,5,6,7,8,9,10,11]";

    fn scalar(sql: &str) -> Option<String> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<String>>(0))
            .unwrap()
    }

    #[test]
    fn retrograde_chromatic() {
        assert_eq!(
            scalar(&format!("SELECT ix_row_retrograde({CHROMATIC})")).as_deref(),
            Some("[11,10,9,8,7,6,5,4,3,2,1,0]")
        );
    }

    #[test]
    fn invert_chromatic() {
        assert_eq!(
            scalar(&format!("SELECT ix_row_invert({CHROMATIC})")).as_deref(),
            Some("[0,11,10,9,8,7,6,5,4,3,2,1]")
        );
    }

    #[test]
    fn matrix_first_row_is_input() {
        let m = scalar(&format!("SELECT ix_row_matrix({CHROMATIC})")).unwrap();
        // 12 rows joined by ';' → 11 separators.
        assert_eq!(m.matches(';').count(), 11);
        assert!(
            m.starts_with("[0,1,2,3,4,5,6,7,8,9,10,11];"),
            "matrix first row must be the input row, got {m}"
        );
    }

    #[test]
    fn combinatoriality_chromatic_has_p6() {
        // Chromatic hexachord {0..5} maps to {6..11} under T6 → P-combinatorial at 6.
        let c = scalar(&format!("SELECT ix_row_combinatoriality({CHROMATIC})")).unwrap();
        assert!(c.starts_with("P:["), "got {c}");
        assert!(c.contains('6'), "chromatic row is T6-combinatorial, got {c}");
    }

    #[test]
    fn non_row_input_is_null() {
        // Too short, and not a permutation → SQL NULL (not an error).
        assert_eq!(scalar("SELECT ix_row_retrograde([0,1,2])"), None);
        // 12 values but with a duplicate (two 0s, no 11) → not a row → NULL.
        assert_eq!(
            scalar("SELECT ix_row_invert([0,0,2,3,4,5,6,7,8,9,10,1])"),
            None
        );
    }
}
