//! Petri-net analysis UDF over `ix-petri` — deadlock, boundedness and liveness
//! of a net described as JSON, from SQL.
//!
//! ```sql
//! SELECT json_extract_string(a, '$.deadlock_free.verdict') AS deadlock_free,
//!        json_extract(a, '$.deadlock_free.detail[0].marking') AS wedged_at
//! FROM (SELECT ix_petri_analyze(net, 50000) AS a FROM read_text('nets/*.json'));
//! ```
//!
//! `ix_petri_analyze(net VARCHAR, max_states BIGINT) -> VARCHAR` returns the full
//! `ix_petri::Analysis` as JSON (net shape: `ix_petri::json`). The state budget
//! is a required argument, not a default, so every result names the bound it was
//! computed under. A net the builder refuses, JSON of the wrong shape, net JSON
//! over `ix_petri::json::MAX_NET_JSON_BYTES`, a budget outside
//! `1..=ix_petri::json::MAX_STATES_CEILING`, or a budget whose worst-case heap
//! for this net (`ix_petri::json::heap_bound`) is over
//! `ix_petri::json::HEAP_BUDGET_BYTES` is a SQL error naming the reason; NULL
//! in either argument is NULL out. One refused row fails the whole statement,
//! as any SQL error does. Pure wrap of `ix_petri::analyze_json` — no Petri
//! logic here.
//!
//! The heap bound covers what this wrap does with a row: the copy of the text,
//! the analysis, its JSON and the `CString` DuckDB copies. It is per row, and
//! DuckDB keeps a chunk's result strings together, so a statement over many
//! large nets holds their outputs at once; the bound does not cap that.
//!
//! The refusal text quotes ids and field names verbatim, and JSON can spell a
//! NUL (`\u0000`) inside either. duckdb-rs hands an error to DuckDB through
//! `CString::new(..).unwrap()` inside an `extern "C"` callback, where a NUL is
//! a panic that cannot unwind and aborts the host process. So every error
//! leaving `invoke` goes through [`sql_error`], which escapes NUL as `\0`.

use crate::udf::null_mask;
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::ffi::duckdb_string_t;
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use std::error::Error;
use std::ffi::CString;

/// The only way an error leaves this UDF for DuckDB: its text with NUL escaped,
/// so no message — today's or a future one — can abort the host.
fn sql_error(e: Box<dyn Error>) -> Box<dyn Error> {
    e.to_string().replace('\0', "\\0").into()
}

struct IxPetriAnalyze;
impl VScalar for IxPetriAnalyze {
    type State = ();
    // @ai:invariant ix_petri_analyze(net, max_states) returns serde_json of ix_petri::analyze_json(net, max_states) byte-for-byte; a refused net/JSON/budget is a SQL error whose text holds no NUL; NULL arg -> NULL. Only duck-feature tests exercise it (sql_equals_rust_wire_bytes, nul_in_refusal_text_is_a_sql_error_not_an_abort), CI compiles neither and the drift snapshot lists neither [P:assumed conf:0.7]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn Error>> {
        analyze_rows(input, output).map_err(sql_error)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Varchar), LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        )]
    }
}

/// # Safety
/// As [`VScalar::invoke`]: `input` is a live DuckDB chunk of (VARCHAR, BIGINT).
unsafe fn analyze_rows(input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), Box<dyn Error>> {
    let n = input.len();
    let (net_null, max_null) = (null_mask(input, 0, n), null_mask(input, 1, n));
    let nets = input.flat_vector(0);
    let nets = nets.as_slice_with_len::<duckdb_string_t>(n);
    let maxes = input.flat_vector(1);
    let maxes = maxes.as_slice_with_len::<i64>(n);
    let mut out = output.flat_vector();
    for i in 0..n {
        if net_null[i] || max_null[i] {
            out.set_null(i);
            continue;
        }
        let net = DuckString::new(&mut { nets[i] }).as_str().to_string();
        let analysis = ix_petri::analyze_json(&net, maxes[i])
            .map_err(|e| format!("ix_petri_analyze: {e}").replace('\0', "\\0"))?;
        // serde_json escapes control characters, so the output cannot hold a NUL.
        out.insert(i, CString::new(serde_json::to_string(&analysis)?)?);
    }
    Ok(())
}

pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxPetriAnalyze>("ix_petri_analyze")
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    const LOCK: &str = r#"{"name":"leaked-lock","places":[{"id":"lock","tokens":1},{"id":"working"}],"transitions":[{"id":"acquire"}],"arcs":[{"from":"lock","to":"acquire"},{"from":"acquire","to":"working"}]}"#;

    fn analyze(sql: &str, params: &[&dyn duckdb::ToSql]) -> duckdb::Result<Option<String>> {
        open_bench()?.query_row(sql, params, |r| r.get::<_, Option<String>>(0))
    }

    #[test]
    fn sql_equals_rust_wire_bytes() {
        let want = serde_json::to_string(&ix_petri::analyze_json(LOCK, 50_000).unwrap()).unwrap();
        assert_eq!(analyze("SELECT ix_petri_analyze(?, 50000)", &[&LOCK]).unwrap(), Some(want));
    }

    #[test]
    fn runs_row_wise_over_a_column() {
        let sql = "SELECT string_agg(json_extract_string(ix_petri_analyze(net, 100), '$.deadlock_free.verdict'), ',' ORDER BY k) \
                   FROM (VALUES (1, ?), (2, ?)) t(k, net)";
        let live = r#"{"places":[{"id":"a","tokens":1},{"id":"b"}],"transitions":[{"id":"go"},{"id":"back"}],"arcs":[{"from":"a","to":"go"},{"from":"go","to":"b"},{"from":"b","to":"back"},{"from":"back","to":"a"}]}"#;
        assert_eq!(analyze(sql, &[&LOCK, &live]).unwrap().as_deref(), Some("fails,holds"));
    }

    #[test]
    fn null_in_null_out() {
        assert_eq!(analyze("SELECT ix_petri_analyze(NULL::VARCHAR, 10)", &[]).unwrap(), None);
        assert_eq!(analyze("SELECT ix_petri_analyze(?, NULL::BIGINT)", &[&LOCK]).unwrap(), None);
    }

    #[test]
    fn refusals_are_sql_errors() {
        let too_long = format!("{}{LOCK}", " ".repeat(ix_petri::json::MAX_NET_JSON_BYTES));
        for (net, max) in [(r#"{"places":[{"id":"p","initial_marking":1}],"transitions":[],"arcs":[]}"#, 10), (r#"{"places":[{"id":"p"},{"id":"p"}],"transitions":[],"arcs":[]}"#, 10), (LOCK, 0), (LOCK, ix_petri::json::MAX_STATES_CEILING + 1), (LOCK, ix_petri::json::MAX_STATES_CEILING), (too_long.as_str(), 10)] {
            let err = analyze("SELECT ix_petri_analyze(?, ?)", &[&net, &max]).unwrap_err().to_string();
            assert!(err.contains("ix_petri_analyze"), "{err}");
        }
        let over_heap = analyze("SELECT ix_petri_analyze(?, ?)", &[&LOCK, &ix_petri::json::MAX_STATES_CEILING]).unwrap_err().to_string();
        assert!(over_heap.contains("bytes of heap") && over_heap.contains("largest admissible max_states"), "{over_heap}");
    }

    /// Before the escape these aborted the test process (a panic inside duckdb-rs's
    /// `extern "C"` callback), so reaching the assertions at all is half the test.
    #[test]
    fn nul_in_refusal_text_is_a_sql_error_not_an_abort() {
        let dup_id = r#"{"places":[{"id":"p\u0000"},{"id":"p\u0000"}],"transitions":[],"arcs":[]}"#;
        let unknown_field = r#"{"places":[],"transitions":[],"arcs":[],"x\u0000":1}"#;
        for (net, quoted) in [(dup_id, "duplicate place id `p\\0`"), (unknown_field, "x\\0")] {
            let err = analyze("SELECT ix_petri_analyze(?, 10)", &[&net]).unwrap_err().to_string();
            assert!(err.contains(quoted) && !err.contains('\0'), "{err}");
        }
    }

    #[test]
    fn sql_error_escapes_nul_from_any_message() {
        assert_eq!(super::sql_error("a\0b".into()).to_string(), "a\\0b");
    }
}
