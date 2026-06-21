//! Graph + signal IX algorithms as DuckDB table functions — capability classes
//! DuckDB has nothing native for.
//!
//! Graph (`ix-graph`), input = a JSON edge list `[[from,to],…]` or `[[from,to,w],…]`
//! (node ids are non-negative ints; node count = max id + 1):
//! - `ix_pagerank(edges, damping DOUBLE, iterations BIGINT)` → `TABLE(node, rank)`.
//! - `ix_shortest_path(edges, src BIGINT, dst BIGINT)` → `TABLE(step, node)` (the
//!   Dijkstra path src→dst; empty if unreachable).
//! - `ix_connected_components(edges)` → `TABLE(node, component)` — weakly-connected
//!   component id per node (edge direction ignored).
//! - `ix_viterbi(initial, transition, emission, observations)` → `TABLE(step, state)` —
//!   most-likely hidden-state path of an HMM (initial 1-D, transition/emission 2-D
//!   JSON matrices, observations a 1-D int array).
//!
//! Signal (`ix-signal`), input = a JSON number array (the series):
//! - `ix_rfft(series)` → `TABLE(bin, magnitude)` — real FFT magnitude spectrum.
//! - `ix_autocorrelation(series)` → `TABLE(lag, value)`.
//!
//! All wrap the real IX crates — no DSP/graph math reimplemented here.

use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_graph::graph::Graph;
use ix_graph::hmm::HiddenMarkovModel;
use ix_signal::correlation::autocorrelation;
use ix_signal::fft::rfft;
use ndarray::Array1;

use crate::tablefn::parse_matrix;

// ── shared output plumbing ────────────────────────────────────────────────────

#[repr(C)]
struct RowsF64 {
    rows: Vec<(i64, f64)>,
}
#[repr(C)]
struct RowsI64 {
    rows: Vec<(i64, i64)>,
}
#[repr(C)]
struct Cursor {
    at: AtomicUsize,
}

fn new_cursor() -> Cursor {
    Cursor { at: AtomicUsize::new(0) }
}

/// Stream `(BIGINT, DOUBLE)` rows in output-vector-sized chunks.
fn emit_f64(rows: &[(i64, f64)], cur: &Cursor, output: &mut DataChunkHandle) {
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
            *slot = rows[start + i].0;
        }
    }
    {
        let mut v = output.flat_vector(1);
        let s = unsafe { v.as_mut_slice_with_len::<f64>(take) };
        for (i, slot) in s.iter_mut().enumerate().take(take) {
            *slot = rows[start + i].1;
        }
    }
    output.set_len(take);
    cur.at.store(start + take, Ordering::Relaxed);
}

/// Stream `(BIGINT, BIGINT)` rows in output-vector-sized chunks.
fn emit_i64(rows: &[(i64, i64)], cur: &Cursor, output: &mut DataChunkHandle) {
    let n = rows.len();
    let start = cur.at.load(Ordering::Relaxed);
    if start >= n {
        output.set_len(0);
        return;
    }
    let cap = output.flat_vector(0).capacity();
    let take = (n - start).min(cap);
    for col in 0..2 {
        let mut v = output.flat_vector(col);
        let s = unsafe { v.as_mut_slice_with_len::<i64>(take) };
        for (i, slot) in s.iter_mut().enumerate().take(take) {
            *slot = if col == 0 { rows[start + i].0 } else { rows[start + i].1 };
        }
    }
    output.set_len(take);
    cur.at.store(start + take, Ordering::Relaxed);
}

/// Parse a JSON series (`[..numbers..]`) into `Vec<f64>`.
fn parse_series(json: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let v: Vec<f64> =
        serde_json::from_str(json).map_err(|e| format!("expected a JSON number array: {e}"))?;
    if v.is_empty() {
        return Err("series is empty".into());
    }
    Ok(v)
}

/// Build a directed graph from a JSON edge list; returns the graph and node count.
fn parse_graph(json: &str) -> Result<(Graph, usize), Box<dyn std::error::Error>> {
    let edges: Vec<Vec<f64>> =
        serde_json::from_str(json).map_err(|e| format!("expected a JSON edge list [[from,to(,w)],…]: {e}"))?;
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
    let mut g = Graph::with_nodes(max_id + 1);
    for e in &edges {
        let w = if e.len() >= 3 { e[2] } else { 1.0 };
        // Dijkstra (ix_shortest_path) and PageRank require non-negative weights; a
        // negative weight gives wrong paths and a negative cycle never settles.
        if w < 0.0 || w.is_nan() {
            return Err("edge weights must be finite and non-negative".into());
        }
        g.add_edge(e[0] as usize, e[1] as usize, w);
    }
    Ok((g, max_id + 1))
}

fn two_cols(bind: &BindInfo, a: &str, a_ty: LogicalTypeId, b: &str, b_ty: LogicalTypeId) {
    bind.add_result_column(a, LogicalTypeHandle::from(a_ty));
    bind.add_result_column(b, LogicalTypeHandle::from(b_ty));
}

// ── ix_pagerank ────────────────────────────────────────────────────────────────

struct IxPagerank;
impl VTab for IxPagerank {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_pagerank emits one (node,rank) per graph node from ix_graph pagerank, normalized to sum to 1 (so dangling-node mass leakage doesn't break the probability-distribution contract) [T:test conf:0.8 src:ix_duck::graphsig::tests::pagerank_normalized_even_with_sink]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let (g, n) = parse_graph(&bind.get_parameter(0).to_string())?;
        let damping = bind.get_parameter(1).to_string().parse::<f64>().unwrap_or(0.85);
        if !(0.0..=1.0).contains(&damping) {
            return Err("damping must be in [0, 1]".into());
        }
        let iters = bind.get_parameter(2).to_int64();
        if iters < 1 {
            return Err("iterations must be >= 1".into());
        }
        let pr = g.pagerank(damping, iters as usize);
        let mut rows: Vec<(i64, f64)> = (0..n).map(|i| (i as i64, *pr.get(&i).unwrap_or(&0.0))).collect();
        rows.sort_by_key(|r| r.0);
        // ix-graph's pagerank doesn't redistribute dangling-node (sink) mass, so the
        // raw vector can sum below 1. Normalize to a proper distribution at the UDF
        // boundary so downstream SQL comparing/aggregating ranks stays sound.
        let total: f64 = rows.iter().map(|r| r.1).sum();
        if total > 0.0 {
            for r in &mut rows {
                r.1 /= total;
            }
        }
        two_cols(bind, "node", LogicalTypeId::Bigint, "rank", LogicalTypeId::Double);
        Ok(RowsF64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_f64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Double),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── ix_shortest_path ─────────────────────────────────────────────────────────────

struct IxShortestPath;
impl VTab for IxShortestPath {
    type InitData = Cursor;
    type BindData = RowsI64;

    // @ai:invariant ix_shortest_path emits the Dijkstra path src->dst as (step,node) in order from ix_graph shortest_path; unreachable -> no rows [T:test conf:0.8 src:ix_duck::graphsig::tests::shortest_path_orders_nodes]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let (g, n) = parse_graph(&bind.get_parameter(0).to_string())?;
        let src = bind.get_parameter(1).to_int64();
        let dst = bind.get_parameter(2).to_int64();
        if src < 0 || dst < 0 || src as usize >= n || dst as usize >= n {
            return Err(format!("src/dst out of range 0..{n}").into());
        }
        let path = g.shortest_path(src as usize, dst as usize).unwrap_or_default();
        let rows: Vec<(i64, i64)> = path.iter().enumerate().map(|(i, &node)| (i as i64, node as i64)).collect();
        two_cols(bind, "step", LogicalTypeId::Bigint, "node", LogicalTypeId::Bigint);
        Ok(RowsI64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_i64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        ])
    }
}

// ── ix_rfft ──────────────────────────────────────────────────────────────────────

struct IxRfft;
impl VTab for IxRfft {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_rfft emits (bin,magnitude) for the real-FFT spectrum of the series via ix_signal rfft; a pure tone has its energy at the matching bin [T:test conf:0.8 src:ix_duck::graphsig::tests::rfft_peaks_at_tone_bin]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let series = parse_series(&bind.get_parameter(0).to_string())?;
        let rows: Vec<(i64, f64)> =
            rfft(&series).iter().enumerate().map(|(i, c)| (i as i64, c.magnitude())).collect();
        two_cols(bind, "bin", LogicalTypeId::Bigint, "magnitude", LogicalTypeId::Double);
        Ok(RowsF64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_f64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ── ix_autocorrelation ────────────────────────────────────────────────────────────

struct IxAutocorrelation;
impl VTab for IxAutocorrelation {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_autocorrelation emits (lag,value) from ix_signal autocorrelation, normalized so lag 0 = 1.0 and is the maximum; lags run -(n-1)..=(n-1) (two-sided, zero-lag centered) [T:test conf:0.8 src:ix_duck::graphsig::tests::autocorrelation_peaks_at_lag_zero]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let series = parse_series(&bind.get_parameter(0).to_string())?;
        // ix_signal returns a two-sided result (len 2n-1) with zero-lag at the
        // centre (index n-1); map the index to the true lag so lag 0 = peak (1.0).
        let center = series.len() as i64 - 1;
        let rows: Vec<(i64, f64)> = autocorrelation(&series)
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as i64 - center, v))
            .collect();
        two_cols(bind, "lag", LogicalTypeId::Bigint, "value", LogicalTypeId::Double);
        Ok(RowsF64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_f64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ── ix_connected_components ──────────────────────────────────────────────────────

struct IxConnectedComponents;
impl VTab for IxConnectedComponents {
    type InitData = Cursor;
    type BindData = RowsI64;

    // @ai:invariant ix_connected_components emits (node,component) for the weakly-connected components of the edge-list graph via ix_graph connected_components; two disjoint subgraphs get distinct component ids, an isolated node is its own component [T:test conf:0.8 src:ix_duck::graphsig::tests::connected_components_finds_islands]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let (g, n) = parse_graph(&bind.get_parameter(0).to_string())?;
        let comp = g.connected_components();
        let rows: Vec<(i64, i64)> = (0..n).map(|i| (i as i64, comp[i] as i64)).collect();
        two_cols(bind, "node", LogicalTypeId::Bigint, "component", LogicalTypeId::Bigint);
        Ok(RowsI64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_i64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ── ix_viterbi ───────────────────────────────────────────────────────────────────

struct IxViterbi;
impl VTab for IxViterbi {
    type InitData = Cursor;
    type BindData = RowsI64;

    // @ai:invariant ix_viterbi emits the most-likely hidden-state path as (step,state) from ix_graph HMM viterbi given the initial/transition/emission/observations JSON; a near-deterministic HMM recovers the generating state sequence; a non-stochastic matrix or out-of-range observation -> SQL error (no panic) [T:test conf:0.8 src:ix_duck::graphsig::tests::viterbi_decodes_biased_hmm]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let initial: Vec<f64> = serde_json::from_str(&bind.get_parameter(0).to_string())
            .map_err(|e| format!("initial must be a JSON number array: {e}"))?;
        let transition = parse_matrix(&bind.get_parameter(1).to_string())?;
        let emission = parse_matrix(&bind.get_parameter(2).to_string())?;
        let obs: Vec<i64> = serde_json::from_str(&bind.get_parameter(3).to_string())
            .map_err(|e| format!("observations must be a JSON int array: {e}"))?;
        if obs.iter().any(|&o| o < 0) {
            return Err("observation symbols must be non-negative".into());
        }
        let hmm = HiddenMarkovModel::new(Array1::from_vec(initial), transition, emission)
            .map_err(|e| format!("ix_viterbi: {e}"))?;
        let m = hmm.n_observations();
        if obs.iter().any(|&o| o as usize >= m) {
            return Err(format!("ix_viterbi: observation symbol out of range 0..{m}").into());
        }
        let observations: Vec<usize> = obs.iter().map(|&o| o as usize).collect();
        let (path, _log_prob) = hmm.viterbi(&observations);
        let rows: Vec<(i64, i64)> =
            path.iter().enumerate().map(|(i, &s)| (i as i64, s as i64)).collect();
        two_cols(bind, "step", LogicalTypeId::Bigint, "state", LogicalTypeId::Bigint);
        Ok(RowsI64 { rows })
    }
    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(new_cursor())
    }
    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        emit_i64(&func.get_bind_data().rows, func.get_init_data(), output);
        Ok(())
    }
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

/// Register the graph + signal table functions.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_table_function::<IxPagerank>("ix_pagerank")?;
    conn.register_table_function::<IxShortestPath>("ix_shortest_path")?;
    conn.register_table_function::<IxConnectedComponents>("ix_connected_components")?;
    conn.register_table_function::<IxViterbi>("ix_viterbi")?;
    conn.register_table_function::<IxRfft>("ix_rfft")?;
    conn.register_table_function::<IxAutocorrelation>("ix_autocorrelation")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    #[test]
    fn pagerank_normalized_even_with_sink() {
        let conn = open_bench().unwrap();
        // 3-cycle → ranks sum to ~1.
        let (n, total): (i64, f64) = conn
            .query_row(
                "SELECT count(*), sum(rank) FROM ix_pagerank('[[0,1],[1,2],[2,0]]', 0.85, 100)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(n, 3, "one row per node");
        assert!((total - 1.0).abs() < 1e-6, "pageranks sum to ~1, got {total}");
        // Sink graph 0->1 (node 1 dangling): ix-graph leaks mass, but the UDF
        // normalizes → still a valid distribution summing to 1.
        let sink_total: f64 = conn
            .query_row("SELECT sum(rank) FROM ix_pagerank('[[0,1]]', 0.85, 100)", [], |r| r.get(0))
            .unwrap();
        assert!((sink_total - 1.0).abs() < 1e-6, "normalized despite the sink, got {sink_total}");
    }

    #[test]
    fn graph_rejects_negative_weights() {
        let conn = open_bench().unwrap();
        // Negative weights break Dijkstra/PageRank — must be a SQL error, not a hang.
        assert!(
            conn.query_row("SELECT node FROM ix_pagerank('[[0,1,-1.0]]', 0.85, 10)", [], |r| r
                .get::<_, i64>(0))
                .is_err(),
            "negative edge weight must be a SQL error"
        );
    }

    #[test]
    fn connected_components_finds_islands() {
        let conn = open_bench().unwrap();
        // Two disjoint edges: {0–1} and {2–3} → two components (4 nodes).
        let edges = "[[0,1],[2,3]]";
        let n_comp: i64 = conn
            .query_row(
                &format!("SELECT count(DISTINCT component) FROM ix_connected_components('{edges}')"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_comp, 2, "two disjoint edges → two components");

        // Nodes 0 and 1 share a component; 0 and 2 do not.
        let same: bool = conn
            .query_row(
                &format!(
                    "SELECT (SELECT component FROM ix_connected_components('{edges}') WHERE node=0) \
                          = (SELECT component FROM ix_connected_components('{edges}') WHERE node=1)"
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(same, "0 and 1 are connected");
        let cross: bool = conn
            .query_row(
                &format!(
                    "SELECT (SELECT component FROM ix_connected_components('{edges}') WHERE node=0) \
                          = (SELECT component FROM ix_connected_components('{edges}') WHERE node=2)"
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(!cross, "0 and 2 are in different components");
    }

    #[test]
    fn viterbi_decodes_biased_hmm() {
        let conn = open_bench().unwrap();
        // 2-state HMM whose emissions strongly couple state i to symbol i, so the
        // most-likely path tracks the observed symbols: obs [0,0,1,1] → states [0,0,1,1].
        let init = "[0.5,0.5]";
        let trans = "[[0.7,0.3],[0.3,0.7]]";
        let emis = "[[0.9,0.1],[0.1,0.9]]";
        let obs = "[0,0,1,1]";
        let path: Vec<i64> = {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT state FROM ix_viterbi('{init}','{trans}','{emis}','{obs}') ORDER BY step"
                ))
                .unwrap();
            stmt.query_map([], |r| r.get::<_, i64>(0)).unwrap().map(|x| x.unwrap()).collect()
        };
        assert_eq!(path, vec![0, 0, 1, 1], "biased HMM decodes states matching the symbols");
    }

    #[test]
    fn viterbi_rejects_non_stochastic() {
        let conn = open_bench().unwrap();
        // Emission row 0 = [0.5,0.1] sums to 0.6 ≠ 1 → HMM construction must fail as a
        // SQL error, not a panic.
        assert!(
            conn.query_row(
                "SELECT state FROM ix_viterbi('[0.5,0.5]','[[0.7,0.3],[0.3,0.7]]','[[0.5,0.1],[0.1,0.9]]','[0]')",
                [],
                |r| r.get::<_, i64>(0)
            )
            .is_err(),
            "non-stochastic emission matrix must be a SQL error"
        );
    }

    #[test]
    fn shortest_path_orders_nodes() {
        let conn = open_bench().unwrap();
        // 0->1->2->3 chain; path 0..3 visits 0,1,2,3 in order.
        let path: Vec<i64> = {
            let conn2 = open_bench().unwrap();
            let mut stmt = conn2
                .prepare("SELECT node FROM ix_shortest_path('[[0,1],[1,2],[2,3]]', 0, 3) ORDER BY step")
                .unwrap();
            stmt.query_map([], |r| r.get::<_, i64>(0)).unwrap().map(|x| x.unwrap()).collect()
        };
        assert_eq!(path, vec![0, 1, 2, 3]);
        let _ = conn;
    }

    #[test]
    fn rfft_peaks_at_tone_bin() {
        let conn = open_bench().unwrap();
        // 8-sample signal with 1 full cycle → spectral peak at bin 1.
        let series = "[0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707]";
        let peak_bin: i64 = conn
            .query_row(
                &format!("SELECT bin FROM ix_rfft('{series}') ORDER BY magnitude DESC LIMIT 1"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(peak_bin, 1, "one-cycle tone peaks at bin 1");
    }

    #[test]
    fn autocorrelation_peaks_at_lag_zero() {
        let conn = open_bench().unwrap();
        let max_lag: i64 = conn
            .query_row(
                "SELECT lag FROM ix_autocorrelation('[1,2,3,4,3,2,1]') ORDER BY value DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(max_lag, 0, "autocorrelation is maximal at lag 0");
    }
}
