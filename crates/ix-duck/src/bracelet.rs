//! Music set-theory scalar UDFs over pitch-class sets (`ix-bracelet`).
//!
//! Each takes a `BIGINT[]` of pitch values (MIDI notes or pitch classes — values
//! are reduced mod 12, so a voicing's `midiNotes` work directly) and returns a
//! VARCHAR characterising the pitch-class *set*:
//!
//! - `ix_forte_number(notes)` → Forte set-class id, e.g. `"3-11"` (NULL if none).
//! - `ix_icv(notes)`          → interval-class vector, e.g. `"<0,0,1,1,1,0>"`.
//! - `ix_prime_form(notes)`   → bracelet prime form, e.g. `"[0,3,7]"`.
//! - `ix_classify_triad(notes)` → `"<root> major|minor"` (NULL if not a triad).
//!
//! The headline use: annotate / group the OPTIC-K voicing corpus by music theory —
//! `SELECT ix_forte_number(midiNotes) AS sc, count(*) FROM read_json_auto(
//! 'state/voicings/raw/guitar.jsonl') GROUP BY sc ORDER BY count(*) DESC`. Nothing
//! else does set-class analysis in SQL. Pure wraps of `ix_bracelet` — no math here.

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_bracelet::forte::forte_number;
use ix_bracelet::grothendieck::icv;
use ix_bracelet::neo_riemannian::{classify_triad, TriadKind};
use ix_bracelet::pc_set::PcSet;
use ix_bracelet::prime_form::bracelet_prime_form;
use std::ffi::CString;

fn list_bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Bigint))
}

/// Read a `LIST<BIGINT>` column into one [`PcSet`] per row (values reduced mod 12;
/// `rem_euclid` so negative ints map correctly). Mirrors `udf::read_list_col` but
/// for integers; reads entries before borrowing the child buffer (no aliasing).
fn read_pcsets(input: &mut DataChunkHandle, n: usize) -> Vec<PcSet> {
    let lv = input.list_vector(0);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<i64>(cap) };
    entries
        .iter()
        .map(|&(o, l)| PcSet::from_pcs(all[o..o + l].iter().map(|&v| v.rem_euclid(12) as u8)))
        .collect()
}

/// Shared driver: map each row's PcSet through `f`; `None` → SQL NULL.
fn invoke_pcset_str(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    f: impl Fn(PcSet) -> Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = input.len();
    let sets = read_pcsets(input, n);
    let mut out = output.flat_vector();
    for (i, &s) in sets.iter().enumerate() {
        match f(s) {
            Some(v) => out.insert(i, CString::new(v)?),
            None => out.set_null(i),
        }
    }
    Ok(())
}

fn varchar_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![list_bigint()],
        LogicalTypeHandle::from(LogicalTypeId::Varchar),
    )]
}

struct IxForteNumber;
impl VScalar for IxForteNumber {
    type State = ();
    // @ai:invariant ix_forte_number renders ix_bracelet forte_number of the PC-set (notes mod 12); {0,4,7} -> "3-11", non-classifiable -> SQL NULL [T:test conf:0.9 src:ix_duck::bracelet::tests::forte_number_major_triad]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pcset_str(input, output, |s| forte_number(s).map(|f| f.to_string()))
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxIcv;
impl VScalar for IxIcv {
    type State = ();
    // @ai:invariant ix_icv renders ix_bracelet icv (6 interval-class counts) as "<a,b,c,d,e,f>"; major triad {0,4,7} -> "<0,0,1,1,1,0>" [T:test conf:0.9 src:ix_duck::bracelet::tests::icv_major_triad]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pcset_str(input, output, |s| {
            let d = icv(s).data;
            Some(format!("<{},{},{},{},{},{}>", d[0], d[1], d[2], d[3], d[4], d[5]))
        })
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxPrimeForm;
impl VScalar for IxPrimeForm {
    type State = ();
    // @ai:invariant ix_prime_form renders ix_bracelet bracelet_prime_form pitch classes as "[p,...]"; major triad {0,4,7} -> "[0,3,7]" [T:test conf:0.9 src:ix_duck::bracelet::tests::prime_form_major_triad]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pcset_str(input, output, |s| {
            let pcs: Vec<String> = bracelet_prime_form(s).iter_pcs().map(|p| p.to_string()).collect();
            Some(format!("[{}]", pcs.join(",")))
        })
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

struct IxClassifyTriad;
impl VScalar for IxClassifyTriad {
    type State = ();
    // @ai:invariant ix_classify_triad renders ix_bracelet classify_triad as "<root> major|minor"; {0,4,7} -> "0 major", non-triad -> SQL NULL [T:test conf:0.9 src:ix_duck::bracelet::tests::classify_triad_major_and_nontriad]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_pcset_str(input, output, |s| {
            classify_triad(s).map(|(root, kind)| {
                let k = match kind {
                    TriadKind::Major => "major",
                    TriadKind::Minor => "minor",
                };
                format!("{root} {k}")
            })
        })
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        varchar_sig()
    }
}

/// Register the music set-theory scalar UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxForteNumber>("ix_forte_number")?;
    conn.register_scalar_function::<IxIcv>("ix_icv")?;
    conn.register_scalar_function::<IxPrimeForm>("ix_prime_form")?;
    conn.register_scalar_function::<IxClassifyTriad>("ix_classify_triad")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    // C major triad as MIDI notes (60,64,67) → pitch classes {0,4,7}.
    const CMAJ: &str = "[60, 64, 67]";

    fn scalar(sql: &str) -> Option<String> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<String>>(0)).unwrap()
    }

    #[test]
    fn forte_number_major_triad() {
        assert_eq!(scalar(&format!("SELECT ix_forte_number({CMAJ})")).as_deref(), Some("3-11"));
    }

    #[test]
    fn icv_major_triad() {
        // Major triad interval-class vector is <0,0,1,1,1,0>.
        assert_eq!(scalar(&format!("SELECT ix_icv({CMAJ})")).as_deref(), Some("<0,0,1,1,1,0>"));
    }

    #[test]
    fn prime_form_major_triad() {
        assert_eq!(scalar(&format!("SELECT ix_prime_form({CMAJ})")).as_deref(), Some("[0,3,7]"));
    }

    #[test]
    fn classify_triad_major_and_nontriad() {
        assert_eq!(scalar(&format!("SELECT ix_classify_triad({CMAJ})")).as_deref(), Some("0 major"));
        // A single dyad is not a triad → NULL.
        assert_eq!(scalar("SELECT ix_classify_triad([0, 7])"), None);
    }
}
