//! Epistemic SQL: hexavalent-logic scalar UDFs over `ix-types::Hexavalent`.
//!
//! The tracer bullet for treating *uncertain claims* as first-class SQL values
//! instead of ground-truth strings. `ix_hex_consensus(tags)` folds a
//! `LIST<VARCHAR>` of hexavalent tags — the verdicts of a multi-model judge
//! panel, say — into a single consensus tag using the canonical algebra.
//!
//! Tags are the wire symbols `T`/`P`/`U`/`D`/`F`/`C` (case-insensitive; the full
//! words `True`..`Contradictory` are also accepted). The fold is
//! [`Hexavalent::and`] — the *meet*, i.e. the most-pessimistic consensus, where
//! `F` (False) is absorbing and any `Contradictory` survives unless an `F`
//! beats it. Headline use:
//!
//! ```sql
//! SELECT claim, ix_hex_consensus(list(verdict)) AS consensus
//! FROM   read_json_auto('state/judge-panel.jsonl')
//! GROUP  BY claim;
//! ```
//!
//! Nothing else collapses a panel of uncertain verdicts in SQL. Pure wrap of
//! `ix-types` — no logic here beyond parsing + the fold.
//!
//! Surface:
//! - `ix_hex_consensus(tags)` — and-fold (the pessimistic meet; panel consensus).
//! - `ix_hex_or(tags)`        — or-fold (the optimistic join; the dual).
//! - `ix_hex_not(tag)`        — negation (`T↔F`, `P↔D`, `U→U`, `C→C`).
//! - `ix_hex_from_evidence(support, refute)` — build a belief from counts.
//!
//! A native `HEXAVALENT` DuckDB type (vs the `VARCHAR` tag) is the next slice.

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::ffi::duckdb_string_t;
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_types::Hexavalent;
use std::ffi::CString;

fn list_varchar() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Varchar))
}

/// Parse a hexavalent tag — the wire symbol or the full word. `None` → not a tag.
fn parse_tag(s: &str) -> Option<Hexavalent> {
    use Hexavalent::*;
    match s.trim().to_ascii_uppercase().as_str() {
        "T" | "TRUE" => Some(True),
        "P" | "PROBABLE" => Some(Probable),
        "U" | "UNKNOWN" => Some(Unknown),
        "D" | "DOUBTFUL" => Some(Doubtful),
        "F" | "FALSE" => Some(False),
        "C" | "CONTRADICTORY" => Some(Contradictory),
        _ => None,
    }
}

/// Read a `LIST<VARCHAR>` column into one `Vec<String>` per row. Composes the
/// list driver (bracelet's `read_pcsets`) with the varchar reader (`code`):
/// entries are read before the child buffer is borrowed (no aliasing).
fn read_str_lists(input: &mut DataChunkHandle, n: usize) -> Vec<Vec<String>> {
    let lv = input.list_vector(0);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<duckdb_string_t>(cap) };
    entries
        .iter()
        .map(|&(o, l)| {
            all[o..o + l]
                .iter()
                .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
                .collect()
        })
        .collect()
}

/// Fold a row's tags with a binary hexavalent op. `Ok(None)` for an empty list
/// (no evidence → no value); `Err` names the first unrecognised tag.
fn fold_tags(
    tags: &[String],
    op: fn(Hexavalent, Hexavalent) -> Hexavalent,
) -> Result<Option<Hexavalent>, String> {
    let mut acc: Option<Hexavalent> = None;
    for t in tags {
        let h = parse_tag(t)
            .ok_or_else(|| format!("unrecognised hexavalent tag {t:?} (expected T/P/U/D/F/C)"))?;
        acc = Some(acc.map_or(h, |a| op(a, h)));
    }
    Ok(acc)
}

/// Shared driver for the `LIST<VARCHAR>` folds: each row's tags reduced through
/// `op`; empty list → SQL NULL; unrecognised tag → SQL error (no panic).
unsafe fn invoke_fold(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    fname: &str,
    op: fn(Hexavalent, Hexavalent) -> Hexavalent,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = input.len();
    let lists = read_str_lists(input, n);
    let mut out = output.flat_vector();
    for (i, tags) in lists.iter().enumerate() {
        match fold_tags(tags, op).map_err(|e| format!("{fname}: {e}"))? {
            Some(h) => out.insert(i, CString::new(h.symbol().to_string())?),
            None => out.set_null(i),
        }
    }
    Ok(())
}

/// Map a (support, refute) evidence count to a hexavalent belief: no evidence →
/// `U`; support only → `T`; refute only → `F`; both → `C` (genuine conflict).
/// The P/D gradations emerge from *aggregating* such values, not raw counts.
fn evidence_to_hex(support: i64, refute: i64) -> Hexavalent {
    use Hexavalent::*;
    match (support > 0, refute > 0) {
        (false, false) => Unknown,
        (true, true) => Contradictory,
        (true, false) => True,
        (false, true) => False,
    }
}

fn list_fold_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![list_varchar()],
        LogicalTypeHandle::from(LogicalTypeId::Varchar),
    )]
}

struct IxHexConsensus;
impl VScalar for IxHexConsensus {
    type State = ();
    // @ai:invariant ix_hex_consensus folds a LIST<VARCHAR> of hexavalent tags with ix_types::Hexavalent::and (the meet; F absorbing): [T,T]->T, [T,P]->P, [P,D]->D, [T,F]->F, [T,C]->C, [C,F]->F; empty list -> SQL NULL; unrecognised tag -> SQL error (no panic) [T:test conf:0.9 src:ix_duck::hexavalent::tests::consensus_meet_fold]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_fold(input, output, "ix_hex_consensus", Hexavalent::and)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        list_fold_sig()
    }
}

struct IxHexOr;
impl VScalar for IxHexOr {
    type State = ();
    // @ai:invariant ix_hex_or folds a LIST<VARCHAR> with ix_types::Hexavalent::or (the join; the optimistic dual of consensus): [F,F]->F, [F,P]->P, [T,_]->T; empty list -> SQL NULL; unrecognised tag -> SQL error [T:test conf:0.9 src:ix_duck::hexavalent::tests::or_join_fold]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        invoke_fold(input, output, "ix_hex_or", Hexavalent::or)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        list_fold_sig()
    }
}

struct IxHexNot;
impl VScalar for IxHexNot {
    type State = ();
    // @ai:invariant ix_hex_not(VARCHAR) is ix_types::Hexavalent::not: T<->F, P<->D, U->U, C->C; NULL in -> NULL out; unrecognised tag -> SQL error [T:test conf:0.9 src:ix_duck::hexavalent::tests::not_unary]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let null = crate::udf::null_mask(input, 0, n);
        let v = input.flat_vector(0);
        let slice = v.as_slice_with_len::<duckdb_string_t>(n);
        let tags: Vec<String> = slice
            .iter()
            .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
            .collect();
        let mut out = output.flat_vector();
        for (i, t) in tags.iter().enumerate() {
            if null[i] {
                out.set_null(i);
                continue;
            }
            let h = parse_tag(t).ok_or_else(|| {
                format!("ix_hex_not: unrecognised hexavalent tag {t:?} (expected T/P/U/D/F/C)")
            })?;
            out.insert(i, CString::new(h.not().symbol().to_string())?);
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)],
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        )]
    }
}

struct IxHexFromEvidence;
impl VScalar for IxHexFromEvidence {
    type State = ();
    // @ai:invariant ix_hex_from_evidence(support BIGINT, refute BIGINT) -> tag: (0,0)->U, (+,0)->T, (0,+)->F, (+,+)->C; NULL in either arg -> NULL out [T:test conf:0.9 src:ix_duck::hexavalent::tests::from_evidence_map]
    unsafe fn invoke(
        _: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = input.len();
        let s_null = crate::udf::null_mask(input, 0, n);
        let r_null = crate::udf::null_mask(input, 1, n);
        let support = input.flat_vector(0).as_slice_with_len::<i64>(n)[..n].to_vec();
        let refute = input.flat_vector(1).as_slice_with_len::<i64>(n)[..n].to_vec();
        let mut out = output.flat_vector();
        for i in 0..n {
            if s_null[i] || r_null[i] {
                out.set_null(i);
                continue;
            }
            let h = evidence_to_hex(support[i], refute[i]);
            out.insert(i, CString::new(h.symbol().to_string())?);
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        )]
    }
}

/// Register the epistemic-logic scalar UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxHexConsensus>("ix_hex_consensus")?;
    conn.register_scalar_function::<IxHexOr>("ix_hex_or")?;
    conn.register_scalar_function::<IxHexNot>("ix_hex_not")?;
    conn.register_scalar_function::<IxHexFromEvidence>("ix_hex_from_evidence")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    fn scalar(sql: &str) -> Option<String> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<String>>(0)).unwrap()
    }

    #[test]
    fn consensus_meet_fold() {
        // The meet: most-pessimistic consensus, F absorbing (matches Hexavalent::and).
        assert_eq!(scalar("SELECT ix_hex_consensus(['T','T','T'])").as_deref(), Some("T"));
        assert_eq!(scalar("SELECT ix_hex_consensus(['T','P'])").as_deref(), Some("P"));
        assert_eq!(scalar("SELECT ix_hex_consensus(['P','D'])").as_deref(), Some("D"));
        assert_eq!(scalar("SELECT ix_hex_consensus(['T','F'])").as_deref(), Some("F"));
        assert_eq!(scalar("SELECT ix_hex_consensus(['T','C'])").as_deref(), Some("C"));
        // F dominates even Contradictory.
        assert_eq!(scalar("SELECT ix_hex_consensus(['C','F'])").as_deref(), Some("F"));
        // Full words are accepted too.
        assert_eq!(scalar("SELECT ix_hex_consensus(['True','Doubtful'])").as_deref(), Some("D"));
    }

    #[test]
    fn empty_list_is_null() {
        assert_eq!(scalar("SELECT ix_hex_consensus([]::VARCHAR[])"), None);
    }

    #[test]
    fn unrecognised_tag_errors() {
        let conn = open_bench().unwrap();
        let err = conn.query_row("SELECT ix_hex_consensus(['T','X'])", [], |r| {
            r.get::<_, Option<String>>(0)
        });
        assert!(err.is_err(), "an unrecognised tag must surface as a SQL error, not a panic or NULL");
    }

    #[test]
    fn or_join_fold() {
        // The join: optimistic dual of consensus. F is identity for OR.
        assert_eq!(scalar("SELECT ix_hex_or(['F','F'])").as_deref(), Some("F"));
        assert_eq!(scalar("SELECT ix_hex_or(['F','P'])").as_deref(), Some("P"));
        assert_eq!(scalar("SELECT ix_hex_or(['T','F'])").as_deref(), Some("T"));
        assert_eq!(scalar("SELECT ix_hex_or([]::VARCHAR[])"), None);
    }

    #[test]
    fn not_unary() {
        assert_eq!(scalar("SELECT ix_hex_not('T')").as_deref(), Some("F"));
        assert_eq!(scalar("SELECT ix_hex_not('P')").as_deref(), Some("D"));
        assert_eq!(scalar("SELECT ix_hex_not('U')").as_deref(), Some("U"));
        assert_eq!(scalar("SELECT ix_hex_not('C')").as_deref(), Some("C"));
        // Double negation is identity on directed values.
        assert_eq!(scalar("SELECT ix_hex_not(ix_hex_not('D'))").as_deref(), Some("D"));
        assert_eq!(scalar("SELECT ix_hex_not(NULL)"), None);
    }

    #[test]
    fn from_evidence_map() {
        assert_eq!(scalar("SELECT ix_hex_from_evidence(0, 0)").as_deref(), Some("U"));
        assert_eq!(scalar("SELECT ix_hex_from_evidence(3, 0)").as_deref(), Some("T"));
        assert_eq!(scalar("SELECT ix_hex_from_evidence(0, 2)").as_deref(), Some("F"));
        assert_eq!(scalar("SELECT ix_hex_from_evidence(2, 1)").as_deref(), Some("C"));
        assert_eq!(scalar("SELECT ix_hex_from_evidence(NULL, 1)"), None);
    }

    // A panel pipeline end-to-end: per-judge evidence → belief → consensus.
    #[test]
    fn evidence_then_consensus_pipeline() {
        // Two judges agree (T,T), one is unsure (U) → meet = U.
        let sql = "SELECT ix_hex_consensus([
            ix_hex_from_evidence(2, 0),
            ix_hex_from_evidence(1, 0),
            ix_hex_from_evidence(0, 0)
        ])";
        assert_eq!(scalar(sql).as_deref(), Some("U"));
        // One judge sees conflict (C) → consensus carries it (unless an F absorbs).
        let sql2 = "SELECT ix_hex_consensus([
            ix_hex_from_evidence(2, 0),
            ix_hex_from_evidence(1, 1)
        ])";
        assert_eq!(scalar(sql2).as_deref(), Some("C"));
    }
}
