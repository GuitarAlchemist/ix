//! Grothendieck-style operations over pitch-class sets and graphs, as DuckDB
//! UDFs — pure wraps of `ix-bracelet` (set-theory K₀ on interval content) and
//! `ix-ktheory` (K₀/K₁ of a graph). No new math lives here.
//!
//! **Tier 1 — Grothendieck group on the interval-class vector (ICV).** The ICV
//! sends a PC-set into the monoid ℕ⁶; lifting to its Grothendieck group ℤ⁶ lets
//! us *subtract* — "what interval content is gained/lost A→B?". Over `BIGINT[]`
//! PC-sets (values reduced mod 12, so a voicing's `midiNotes` work directly):
//! - `ix_grothendieck_delta(a, b)` → signed ℤ⁶ delta, e.g. `"<+0,+0,-1,+2,-1,+0>"`.
//! - `ix_icv_l1(a, b)`             → L1 harmonic cost (= `|delta|₁`), a BIGINT.
//! - `ix_z_related(a, b)`          → BOOLEAN: same ICV + same cardinality but a
//!   *different* set-class (the classical Z-relation — ICV can't tell them apart).
//! - `ix_grothendieck_nearby(set, max_l1)` → TABLE(pc_set, delta, l1): PC-sets
//!   within an L1 budget of `set` (orbit-aware; `set` as a JSON array `'[0,4,7]'`).
//! - `ix_grothendieck_path(src, dst, max_steps)` → TABLE(step, pc_set): A*
//!   shortest harmonic path between equal-cardinality PC-sets.
//!
//! Headline use: voice-lead the whole OPTIC-K corpus toward a target chord in one
//! query — `SELECT diagram, ix_grothendieck_delta(midiNotes, [0,4,7,11]) AS delta,
//! ix_icv_l1(midiNotes, [0,4,7,11]) AS cost FROM read_json_auto('…voicings…')
//! ORDER BY cost LIMIT 20`.
//!
//! **Tier 3 — K-theory of a graph.** Over a JSON edge-list `'[[from,to],…]'`
//! (node count = max id + 1), matching `graphsig`'s graph convention:
//! - `ix_k0(edges)` → TABLE(rank, torsion): K₀ = coker(I − Aᵀ), resource balance.
//! - `ix_k1(edges)` → TABLE(rank):          K₁ = ker(I − Aᵀ), feedback cycles.
//!
//! IX↔GA note: `ix_grothendieck_delta` is built on the ICV, which agrees with GA's
//! `ga_chord_to_set` (cross-checked 2026-06-19). GA's *Forte number* uses a
//! divergent numbering (maj=3-2 vs IX/Forte-catalog 3-11), so bridge IX↔GA on
//! ICV / delta — never on Forte number.

use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_bracelet::grothendieck::{find_nearby, find_shortest_path, grothendieck_delta, icv, Delta};
use ix_bracelet::pc_set::PcSet;
use ix_bracelet::prime_form::bracelet_prime_form;
use ix_ktheory::graph_k::{k0_from_adjacency, k1_from_adjacency};
use ndarray::Array2;

type Res = Result<(), Box<dyn std::error::Error>>;

// ── shared helpers ─────────────────────────────────────────────────────────────

fn list_bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Bigint))
}

/// Read a `LIST<BIGINT>` column into one [`PcSet`] per row (values reduced mod 12;
/// `rem_euclid` so negatives map correctly). Mirrors `bracelet::read_pcsets` but
/// for an arbitrary column. Reads entries before borrowing the child buffer.
fn read_pcset_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<PcSet> {
    let lv = input.list_vector(col);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<i64>(cap) };
    entries
        .iter()
        .map(|&(o, l)| PcSet::from_pcs(all[o..o + l].iter().map(|&v| v.rem_euclid(12) as u8)))
        .collect()
}

/// Per-row NULL mask of a `LIST` column (validity lives on the vector itself, so a
/// `FlatVector` view of the same column pointer reads it — same trick as `udf.rs`).
fn null_mask(input: &DataChunkHandle, col: usize, n: usize) -> Vec<bool> {
    let v = input.flat_vector(col);
    (0..n).map(|i| v.row_is_null(i as u64)).collect()
}

/// Render a signed ℤ⁶ delta as `"<+a,+b,…>"` (explicit signs distinguish it from
/// the unsigned ICV string `ix_icv` emits).
fn fmt_delta(d: Delta) -> String {
    let x = d.data;
    format!("<{:+},{:+},{:+},{:+},{:+},{:+}>", x[0], x[1], x[2], x[3], x[4], x[5])
}

/// Render a PC-set as `"[p,…]"` of its pitch classes (ascending).
fn fmt_pcs(s: PcSet) -> String {
    let pcs: Vec<String> = s.iter_pcs().map(|p| p.to_string()).collect();
    format!("[{}]", pcs.join(","))
}

/// Parse a JSON int array (`'[0,4,7]'`) into a [`PcSet`] (values mod 12).
fn parse_pcset_json(json: &str) -> Result<PcSet, Box<dyn std::error::Error>> {
    let v: Vec<i64> = serde_json::from_str(json)
        .map_err(|e| format!("expected a JSON int array like [0,4,7]: {e}"))?;
    Ok(PcSet::from_pcs(v.into_iter().map(|x| x.rem_euclid(12) as u8)))
}

/// Build a square `f64` adjacency matrix from a JSON edge list `[[from,to(,w)],…]`
/// (node count = max id + 1). Mirrors `graphsig::parse_graph` validation.
fn parse_adjacency(json: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let edges: Vec<Vec<f64>> = serde_json::from_str(json)
        .map_err(|e| format!("expected a JSON edge list [[from,to(,w)],…]: {e}"))?;
    let mut max_id = 0usize;
    for e in &edges {
        if e.len() < 2 {
            return Err("each edge needs at least [from, to]".into());
        }
        if e[0] < 0.0 || e[1] < 0.0 {
            return Err("node ids must be non-negative".into());
        }
        max_id = max_id.max(e[0] as usize).max(e[1] as usize);
    }
    let n = max_id + 1;
    let mut adj = Array2::<f64>::zeros((n, n));
    for e in &edges {
        let w = if e.len() >= 3 { e[2] } else { 1.0 };
        adj[[e[0] as usize, e[1] as usize]] = w;
    }
    Ok(adj)
}

/// True iff `a` and `b` are Z-related: equal cardinality and ICV but a different
/// set-class (D₁₂ orbit). The cardinality guard excludes the empty-vs-singleton
/// false positive (both have an all-zero ICV).
fn is_z_related(a: PcSet, b: PcSet) -> bool {
    a.cardinality() == b.cardinality()
        && icv(a) == icv(b)
        && bracelet_prime_form(a) != bracelet_prime_form(b)
}

// ── Tier 1: scalar UDFs over BIGINT[] ──────────────────────────────────────────

fn pcset_pair_sig(ret: LogicalTypeId) -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![list_bigint(), list_bigint()],
        LogicalTypeHandle::from(ret),
    )]
}

struct IxGrothendieckDelta;
impl VScalar for IxGrothendieckDelta {
    type State = ();
    // @ai:invariant ix_grothendieck_delta renders ix_bracelet grothendieck_delta(a,b) (target−source ICV diff in ℤ⁶) as "<±,…>"; {0,4,7}→{0,4,8} = "<+0,+0,-1,+2,-1,+0>"; NULL arg → SQL NULL [T:test conf:0.9 src:ix_duck::grothendieck::tests::delta_major_to_aug]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let an = null_mask(input, 0, n);
        let bn = null_mask(input, 1, n);
        let a = read_pcset_col(input, 0, n);
        let b = read_pcset_col(input, 1, n);
        let mut out = output.flat_vector();
        for i in 0..n {
            if an[i] || bn[i] {
                out.set_null(i);
            } else {
                out.insert(i, CString::new(fmt_delta(grothendieck_delta(a[i], b[i])))?);
            }
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        pcset_pair_sig(LogicalTypeId::Varchar)
    }
}

struct IxIcvL1;
impl VScalar for IxIcvL1 {
    type State = ();
    // @ai:invariant ix_icv_l1 returns |grothendieck_delta(a,b)|₁ as BIGINT (total interval-class steps gained+lost); {0,4,7}→{0,4,8} = 4; NULL arg → SQL NULL [T:test conf:0.9 src:ix_duck::grothendieck::tests::icv_l1_major_to_aug]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let an = null_mask(input, 0, n);
        let bn = null_mask(input, 1, n);
        let a = read_pcset_col(input, 0, n);
        let b = read_pcset_col(input, 1, n);
        let mut out = output.flat_vector();
        {
            let s = unsafe { out.as_mut_slice_with_len::<i64>(n) };
            for (i, slot) in s.iter_mut().enumerate().take(n) {
                *slot = if an[i] || bn[i] {
                    0 // placeholder; flagged NULL below
                } else {
                    grothendieck_delta(a[i], b[i]).l1_norm() as i64
                };
            }
        }
        for (i, (&x, &y)) in an.iter().zip(&bn).enumerate() {
            if x || y {
                out.set_null(i);
            }
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        pcset_pair_sig(LogicalTypeId::Bigint)
    }
}

struct IxZRelated;
impl VScalar for IxZRelated {
    type State = ();
    // @ai:invariant ix_z_related is true iff a,b share cardinality+ICV but differ in bracelet prime form (the Z-relation); 4-Z15{0,1,4,6}↔4-Z29{0,1,3,7} = true; maj↔min (same set-class) = false; NULL arg → SQL NULL [T:test conf:0.9 src:ix_duck::grothendieck::tests::z_related_canonical_pair]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let an = null_mask(input, 0, n);
        let bn = null_mask(input, 1, n);
        let a = read_pcset_col(input, 0, n);
        let b = read_pcset_col(input, 1, n);
        let mut out = output.flat_vector();
        {
            let s = unsafe { out.as_mut_slice_with_len::<bool>(n) };
            for (i, slot) in s.iter_mut().enumerate().take(n) {
                *slot = !(an[i] || bn[i]) && is_z_related(a[i], b[i]);
            }
        }
        for (i, (&x, &y)) in an.iter().zip(&bn).enumerate() {
            if x || y {
                out.set_null(i);
            }
        }
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        pcset_pair_sig(LogicalTypeId::Boolean)
    }
}

// ── streaming plumbing for the table functions ─────────────────────────────────

#[repr(C)]
struct Cursor {
    at: AtomicUsize,
}
fn new_cursor() -> Cursor {
    Cursor { at: AtomicUsize::new(0) }
}

#[repr(C)]
struct RowsNearby {
    rows: Vec<(String, String, i64)>,
}
#[repr(C)]
struct RowsStep {
    rows: Vec<(i64, String)>,
}

/// Emit `(VARCHAR, VARCHAR, BIGINT)` rows in output-vector-sized chunks.
fn emit_str_str_i64(rows: &[(String, String, i64)], cur: &Cursor, output: &mut DataChunkHandle) -> Res {
    let n = rows.len();
    let start = cur.at.load(Ordering::Relaxed);
    if start >= n {
        output.set_len(0);
        return Ok(());
    }
    let cap = output.flat_vector(0).capacity();
    let take = (n - start).min(cap);
    {
        let v = output.flat_vector(0);
        for i in 0..take {
            v.insert(i, CString::new(rows[start + i].0.as_str())?);
        }
    }
    {
        let v = output.flat_vector(1);
        for i in 0..take {
            v.insert(i, CString::new(rows[start + i].1.as_str())?);
        }
    }
    {
        let mut v = output.flat_vector(2);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(take) };
        for (i, slot) in s.iter_mut().enumerate().take(take) {
            *slot = rows[start + i].2;
        }
    }
    output.set_len(take);
    cur.at.store(start + take, Ordering::Relaxed);
    Ok(())
}

/// Emit `(BIGINT, VARCHAR)` rows in output-vector-sized chunks.
fn emit_i64_str(rows: &[(i64, String)], cur: &Cursor, output: &mut DataChunkHandle) -> Res {
    let n = rows.len();
    let start = cur.at.load(Ordering::Relaxed);
    if start >= n {
        output.set_len(0);
        return Ok(());
    }
    let cap = output.flat_vector(0).capacity();
    let take = (n - start).min(cap);
    {
        let mut v = output.flat_vector(0);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(take) };
        for (i, slot) in s.iter_mut().enumerate().take(take) {
            *slot = rows[start + i].0;
        }
    }
    {
        let v = output.flat_vector(1);
        for i in 0..take {
            v.insert(i, CString::new(rows[start + i].1.as_str())?);
        }
    }
    output.set_len(take);
    cur.at.store(start + take, Ordering::Relaxed);
    Ok(())
}

// ── Tier 1: ix_grothendieck_nearby ─────────────────────────────────────────────

struct IxGrothendieckNearby;
impl VTab for IxGrothendieckNearby {
    type InitData = Cursor;
    type BindData = RowsNearby;

    // @ai:invariant ix_grothendieck_nearby emits (pc_set,delta,l1) for every PcSet within Grothendieck L1 ≤ max_l1 of the source via ix_bracelet find_nearby; the source's orbit appears at l1=0; max_l1<0 → SQL error [T:test conf:0.8 src:ix_duck::grothendieck::tests::nearby_source_at_zero]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let set = parse_pcset_json(&bind.get_parameter(0).to_string())?;
        let max_l1 = bind.get_parameter(1).to_int64();
        if max_l1 < 0 {
            return Err("max_l1 must be >= 0".into());
        }
        let budget: u32 = max_l1.try_into().unwrap_or(u32::MAX);
        let rows: Vec<(String, String, i64)> = find_nearby(set, budget)
            .into_iter()
            .map(|(s, d, cost)| (fmt_pcs(s), fmt_delta(d), cost as i64))
            .collect();
        bind.add_result_column("pc_set", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("delta", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("l1", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        Ok(RowsNearby { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Res {
        emit_str_str_i64(&func.get_bind_data().rows, func.get_init_data(), output)
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── Tier 1: ix_grothendieck_path ───────────────────────────────────────────────

struct IxGrothendieckPath;
impl VTab for IxGrothendieckPath {
    type InitData = Cursor;
    type BindData = RowsStep;

    // @ai:invariant ix_grothendieck_path emits the A* harmonic path src→dst as (step,pc_set) in order via ix_bracelet find_shortest_path; src==dst → single row; different cardinality or no path within max_steps → no rows [T:test conf:0.8 src:ix_duck::grothendieck::tests::path_maj_to_min]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let src = parse_pcset_json(&bind.get_parameter(0).to_string())?;
        let dst = parse_pcset_json(&bind.get_parameter(1).to_string())?;
        let max_steps = bind.get_parameter(2).to_int64();
        if max_steps < 0 {
            return Err("max_steps must be >= 0".into());
        }
        let steps: usize = max_steps.try_into().unwrap_or(usize::MAX);
        let rows: Vec<(i64, String)> = find_shortest_path(src, dst, steps)
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i as i64, fmt_pcs(s)))
            .collect();
        bind.add_result_column("step", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("pc_set", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        Ok(RowsStep { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Res {
        emit_i64_str(&func.get_bind_data().rows, func.get_init_data(), output)
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── Tier 3: K-theory of a graph ────────────────────────────────────────────────

struct IxK0;
impl VTab for IxK0 {
    type InitData = Cursor;
    type BindData = RowsStep;

    // @ai:invariant ix_k0 emits one (rank,torsion) row: K₀ = coker(I−Aᵀ) of the edge-list graph via ix_ktheory k0_from_adjacency; rank = free part, torsion = JSON of invariant factors >1; non-JSON edges → SQL error [T:test conf:0.8 src:ix_duck::grothendieck::tests::k0_cycle_has_free_rank]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let adj = parse_adjacency(&bind.get_parameter(0).to_string())?;
        let k = k0_from_adjacency(&adj).map_err(|e| format!("ix_k0: {e}"))?;
        let torsion = serde_json::to_string(&k.torsion)?;
        bind.add_result_column("rank", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("torsion", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        Ok(RowsStep { rows: vec![(k.rank as i64, torsion)] })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Res {
        emit_i64_str(&func.get_bind_data().rows, func.get_init_data(), output)
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

#[repr(C)]
struct RowsScalar {
    rows: Vec<i64>,
}

/// Emit single-column `BIGINT` rows in output-vector-sized chunks.
fn emit_i64_one(rows: &[i64], cur: &Cursor, output: &mut DataChunkHandle) {
    let n = rows.len();
    let start = cur.at.load(Ordering::Relaxed);
    if start >= n {
        output.set_len(0);
        return;
    }
    let cap = output.flat_vector(0).capacity();
    let take = (n - start).min(cap);
    {
        let mut v = output.flat_vector(0);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(take) };
        for (i, slot) in s.iter_mut().enumerate().take(take) {
            *slot = rows[start + i];
        }
    }
    output.set_len(take);
    cur.at.store(start + take, Ordering::Relaxed);
}

struct IxK1;
impl VTab for IxK1 {
    type InitData = Cursor;
    type BindData = RowsScalar;

    // @ai:invariant ix_k1 emits one (rank) row: K₁ = ker(I−Aᵀ) of the edge-list graph via ix_ktheory k1_from_adjacency; rank>0 iff feedback cycles exist (a DAG → 0); non-JSON edges → SQL error [T:test conf:0.8 src:ix_duck::grothendieck::tests::k1_cycle_vs_dag]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let adj = parse_adjacency(&bind.get_parameter(0).to_string())?;
        let k = k1_from_adjacency(&adj).map_err(|e| format!("ix_k1: {e}"))?;
        bind.add_result_column("rank", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        Ok(RowsScalar { rows: vec![k.rank as i64] })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Res {
        emit_i64_one(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

/// Register the Grothendieck (Tier 1) + K-theory (Tier 3) UDFs.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxGrothendieckDelta>("ix_grothendieck_delta")?;
    conn.register_scalar_function::<IxIcvL1>("ix_icv_l1")?;
    conn.register_scalar_function::<IxZRelated>("ix_z_related")?;
    conn.register_table_function::<IxGrothendieckNearby>("ix_grothendieck_nearby")?;
    conn.register_table_function::<IxGrothendieckPath>("ix_grothendieck_path")?;
    conn.register_table_function::<IxK0>("ix_k0")?;
    conn.register_table_function::<IxK1>("ix_k1")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    fn scalar_str(sql: &str) -> Option<String> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<String>>(0)).unwrap()
    }
    fn scalar_i64(sql: &str) -> Option<i64> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<i64>>(0)).unwrap()
    }
    fn scalar_bool(sql: &str) -> Option<bool> {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<bool>>(0)).unwrap()
    }
    fn count(sql: &str) -> i64 {
        let conn = open_bench().unwrap();
        conn.query_row(sql, [], |r| r.get(0)).unwrap()
    }

    // ── Tier 1: delta / l1 ─────────────────────────────────────────────────────

    #[test]
    fn delta_major_to_aug() {
        // C major {0,4,7} → C aug {0,4,8}: ICV <0,0,1,1,1,0> → <0,0,0,3,0,0>,
        // delta = <0,0,-1,+2,-1,0>. CROSS-CHECKED IX↔GA 2026-06-19: both ICVs match
        // ga_chord_to_set (C=<0 0 1 1 1 0>, Caug=<0 0 0 3 0 0>) → this delta is the
        // GA-agreed value. Guards against IX-side drift away from GA on the ICV.
        assert_eq!(
            scalar_str("SELECT ix_grothendieck_delta([0,4,7], [0,4,8])").as_deref(),
            Some("<+0,+0,-1,+2,-1,+0>")
        );
    }

    #[test]
    fn icv_l1_major_to_aug() {
        // |<0,0,-1,2,-1,0>|₁ = 4.
        assert_eq!(scalar_i64("SELECT ix_icv_l1([0,4,7], [0,4,8])"), Some(4));
        // Transposed major triad has identical ICV → zero distance.
        assert_eq!(scalar_i64("SELECT ix_icv_l1([0,4,7], [2,6,9])"), Some(0));
    }

    #[test]
    fn delta_and_l1_pass_null_through() {
        assert_eq!(scalar_str("SELECT ix_grothendieck_delta(NULL::BIGINT[], [0,4,7])"), None);
        assert_eq!(scalar_i64("SELECT ix_icv_l1([0,4,7], NULL::BIGINT[])"), None);
    }

    // ── Tier 1: Z-relation ─────────────────────────────────────────────────────

    #[test]
    fn z_related_canonical_pair() {
        // 4-Z15 {0,1,4,6} ↔ 4-Z29 {0,1,3,7}: the all-interval tetrachords, same ICV
        // <1,1,1,1,1,1>, different set classes → Z-related.
        assert_eq!(scalar_bool("SELECT ix_z_related([0,1,4,6], [0,1,3,7])"), Some(true));
        // Major vs minor triad: same set class (both 3-11) → NOT Z-related.
        assert_eq!(scalar_bool("SELECT ix_z_related([0,4,7], [0,3,7])"), Some(false));
        // Major vs augmented: different ICV → NOT Z-related.
        assert_eq!(scalar_bool("SELECT ix_z_related([0,4,7], [0,4,8])"), Some(false));
        // NULL passthrough.
        assert_eq!(scalar_bool("SELECT ix_z_related(NULL::BIGINT[], [0,1,4,6])"), None);
    }

    // ── Tier 1: nearby / path table functions ──────────────────────────────────

    #[test]
    fn nearby_source_at_zero() {
        // At budget 0 only the source's own orbit (l1 = 0) appears, and it's non-empty.
        let conn = open_bench().unwrap();
        let (n, allzero): (i64, bool) = conn
            .query_row(
                "SELECT count(*), bool_and(l1 = 0) FROM ix_grothendieck_nearby('[0,4,7]', 0)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(n > 0, "source orbit must appear at l1=0");
        assert!(allzero, "budget 0 ⇒ every row at l1=0");
        // The source's prime form [0,3,7] is among the l1=0 sets.
        let hit: i64 = conn
            .query_row(
                "SELECT count(*) FROM ix_grothendieck_nearby('[0,4,7]', 0) WHERE pc_set = '[0,3,7]'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(hit >= 1, "minor triad [0,3,7] shares the major-triad orbit (3-11)");
        // Negative budget is a SQL error, not a panic.
        assert!(conn
            .query_row("SELECT count(*) FROM ix_grothendieck_nearby('[0,4,7]', -1)", [], |r| r
                .get::<_, i64>(0))
            .is_err());
    }

    #[test]
    fn path_maj_to_min() {
        // C major → C minor triad share orbit (3-11) → reachable; path includes both ends.
        let conn = open_bench().unwrap();
        let mut stmt = conn
            .prepare("SELECT pc_set FROM ix_grothendieck_path('[0,4,7]', '[0,3,7]', 5) ORDER BY step")
            .unwrap();
        let path: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().map(|x| x.unwrap()).collect();
        assert!(path.len() >= 2, "path spans at least source and target");
        assert_eq!(path.first().map(String::as_str), Some("[0,4,7]"));
        assert_eq!(path.last().map(String::as_str), Some("[0,3,7]"));
        // src == dst → single-row path.
        assert_eq!(count("SELECT count(*) FROM ix_grothendieck_path('[0,4,7]', '[0,4,7]', 5)"), 1);
        // Different cardinality (triad → seventh) → no path.
        assert_eq!(count("SELECT count(*) FROM ix_grothendieck_path('[0,4,7]', '[0,4,7,11]', 5)"), 0);
    }

    // ── Tier 3: K-theory ───────────────────────────────────────────────────────

    #[test]
    fn k0_cycle_has_free_rank() {
        // 3-cycle 0→1→2→0 has eigenvalue 1 in Aᵀ → K₀ = coker(I−Aᵀ) has a free generator.
        let conn = open_bench().unwrap();
        let (rank, torsion): (i64, String) = conn
            .query_row("SELECT rank, torsion FROM ix_k0('[[0,1],[1,2],[2,0]]')", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert!(rank >= 1, "a cycle's K₀ has a free generator, got rank {rank}");
        assert_eq!(torsion, "[]", "this cycle has no torsion");
        // Malformed edges → SQL error.
        assert!(conn
            .query_row("SELECT rank FROM ix_k0('not json')", [], |r| r.get::<_, i64>(0))
            .is_err());
    }

    #[test]
    fn k1_cycle_vs_dag() {
        // K₁ = ker(I−Aᵀ): a cycle has feedback (rank ≥ 1); a pure DAG has none (0).
        assert_eq!(
            scalar_i64("SELECT rank FROM ix_k1('[[0,1],[1,2],[2,0]]')").map(|r| r >= 1),
            Some(true)
        );
        assert_eq!(scalar_i64("SELECT rank FROM ix_k1('[[0,1],[1,2]]')"), Some(0));
    }
}
