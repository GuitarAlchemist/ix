//! Graph + signal IX algorithms as DuckDB table functions — capability classes
//! DuckDB has nothing native for.
//!
//! Graph (`ix-graph`), input = a JSON edge list `[[from,to],…]` or `[[from,to,w],…]`
//! (node ids are non-negative ints; node count = max id + 1):
//! - `ix_pagerank(edges, damping DOUBLE, iterations BIGINT)` → `TABLE(node, rank)`.
//! - `ix_shortest_path(edges, src BIGINT, dst BIGINT)` → `TABLE(step, node)` (the
//!   Dijkstra path src→dst; empty if unreachable).
//!
//! Signal (`ix-signal`), input = a JSON number array (the series):
//! - `ix_rfft(series)` → `TABLE(bin, magnitude)` — real FFT magnitude spectrum.
//! - `ix_autocorrelation(series)` → `TABLE(lag, value)`.
//! - `ix_wavelet_denoise(series, levels BIGINT, threshold DOUBLE)` → `TABLE(i, value)` —
//!   soft-thresholded series (same length as input).
//! - `ix_kalman_smooth(series, process_noise DOUBLE, measurement_noise DOUBLE)` →
//!   `TABLE(i, value)` — constant-velocity 1-D Kalman position estimate (dt=1).
//! - `ix_dct(series)` → `TABLE(k, coefficient)` — DCT-II coefficients.
//!
//! All wrap the real IX crates — no DSP/graph math reimplemented here.

use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;
use ix_graph::graph::Graph;
use ix_signal::correlation::autocorrelation;
use ix_signal::dct::dct2;
use ix_signal::fft::rfft;
use ix_signal::kalman::constant_velocity_1d;
use ix_signal::wavelet::wavelet_denoise;
use ndarray::Array1;

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

// ── ix_wavelet_denoise ─────────────────────────────────────────────────────────

struct IxWaveletDenoise;
impl VTab for IxWaveletDenoise {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_wavelet_denoise emits (i,value) for the wavelet soft-thresholded series via ix_signal wavelet_denoise (same length as input); a larger threshold removes more high-frequency energy, so output variance does not exceed the input's [T:test conf:0.8 src:ix_duck::graphsig::tests::wavelet_denoise_reduces_variance]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let series = parse_series(&bind.get_parameter(0).to_string())?;
        let levels_req = bind.get_parameter(1).to_int64();
        if levels_req < 1 {
            return Err("levels must be >= 1".into());
        }
        let threshold = bind
            .get_parameter(2)
            .to_string()
            .parse::<f64>()
            .map_err(|e| format!("threshold must be a number: {e}"))?;
        if !threshold.is_finite() || threshold < 0.0 {
            return Err("threshold must be finite and >= 0".into());
        }
        // Cap the decomposition depth at floor(log2(n)) so a short series is never
        // over-decomposed (each level halves the signal).
        let max_levels = ((series.len() as f64).log2().floor() as usize).max(1);
        let levels = (levels_req as usize).min(max_levels);
        let rows: Vec<(i64, f64)> = wavelet_denoise(&series, levels, threshold)
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as i64, v))
            .collect();
        two_cols(bind, "i", LogicalTypeId::Bigint, "value", LogicalTypeId::Double);
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
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
            LogicalTypeHandle::from(LogicalTypeId::Double),
        ])
    }
}

// ── ix_kalman_smooth ───────────────────────────────────────────────────────────

struct IxKalmanSmooth;
impl VTab for IxKalmanSmooth {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_kalman_smooth emits (i,value) where value is the filtered position of a constant-velocity 1-D Kalman filter (ix_signal, dt=1) over the series; fed a constant-velocity ramp it locks onto the trajectory, so the final estimate converges to the final measurement [T:test conf:0.8 src:ix_duck::graphsig::tests::kalman_smooth_tracks_ramp]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let series = parse_series(&bind.get_parameter(0).to_string())?;
        let process_noise = parse_pos_finite(&bind.get_parameter(1).to_string(), "process_noise")?;
        let measurement_noise =
            parse_pos_finite(&bind.get_parameter(2).to_string(), "measurement_noise")?;
        // dt = 1: telemetry samples are an evenly-spaced unit-step sequence.
        let mut kf = constant_velocity_1d(process_noise, measurement_noise, 1.0);
        let measurements: Vec<Array1<f64>> =
            series.iter().map(|&z| Array1::from_vec(vec![z])).collect();
        // state = [position, velocity]; the smoothed signal is the position estimate.
        let rows: Vec<(i64, f64)> = kf
            .filter(&measurements)
            .iter()
            .enumerate()
            .map(|(i, s)| (i as i64, s[0]))
            .collect();
        two_cols(bind, "i", LogicalTypeId::Bigint, "value", LogicalTypeId::Double);
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
            LogicalTypeHandle::from(LogicalTypeId::Double),
        ])
    }
}

/// Parse a positive, finite `f64` parameter (Kalman noise covariances must be > 0).
fn parse_pos_finite(s: &str, name: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let v: f64 = s.parse().map_err(|e| format!("{name} must be a number: {e}"))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("{name} must be a finite number > 0").into());
    }
    Ok(v)
}

// ── ix_dct ─────────────────────────────────────────────────────────────────────

struct IxDct;
impl VTab for IxDct {
    type InitData = Cursor;
    type BindData = RowsF64;

    // @ai:invariant ix_dct emits (k,coefficient) for the DCT-II of the series via ix_signal dct2 (same length as input); a constant signal concentrates all energy in the k=0 (DC) coefficient [T:test conf:0.8 src:ix_duck::graphsig::tests::dct_constant_concentrates_at_dc]
    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let series = parse_series(&bind.get_parameter(0).to_string())?;
        let rows: Vec<(i64, f64)> =
            dct2(&series).iter().enumerate().map(|(i, &v)| (i as i64, v)).collect();
        two_cols(bind, "k", LogicalTypeId::Bigint, "coefficient", LogicalTypeId::Double);
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

/// Register the graph + signal table functions.
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_table_function::<IxPagerank>("ix_pagerank")?;
    conn.register_table_function::<IxShortestPath>("ix_shortest_path")?;
    conn.register_table_function::<IxRfft>("ix_rfft")?;
    conn.register_table_function::<IxAutocorrelation>("ix_autocorrelation")?;
    conn.register_table_function::<IxWaveletDenoise>("ix_wavelet_denoise")?;
    conn.register_table_function::<IxKalmanSmooth>("ix_kalman_smooth")?;
    conn.register_table_function::<IxDct>("ix_dct")?;
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
    fn wavelet_denoise_reduces_variance() {
        let conn = open_bench().unwrap();
        // High-frequency square wave around mean 5 (raw var_pop = 25). A large
        // threshold kills the detail coefficients → the denoised series collapses
        // toward the mean, so its variance must not exceed the raw series'.
        let noisy = "[10,0,10,0,10,0,10,0]";
        let (n, var): (i64, f64) = conn
            .query_row(
                &format!("SELECT count(*), var_pop(value) FROM ix_wavelet_denoise('{noisy}', 3, 50.0)"),
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(n, 8, "denoised series keeps the input length");
        assert!(var < 25.0, "large threshold reduces variance below raw 25, got {var}");
    }

    #[test]
    fn kalman_smooth_tracks_ramp() {
        let conn = open_bench().unwrap();
        // A clean constant-velocity ramp 0..7 is exactly the filter's motion model;
        // from a cold start [0,0] it locks on, so the final estimate lands near the
        // final measurement (7). One estimate per measurement.
        let ramp = "[0,1,2,3,4,5,6,7]";
        let n: i64 = conn
            .query_row(
                &format!("SELECT count(*) FROM ix_kalman_smooth('{ramp}', 0.01, 0.5)"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 8, "one filtered estimate per measurement");
        let last: f64 = conn
            .query_row(
                &format!("SELECT value FROM ix_kalman_smooth('{ramp}', 0.01, 0.5) WHERE i=7"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            (5.0..=9.0).contains(&last),
            "filter tracks the ramp to ~7, got {last}"
        );
    }

    #[test]
    fn kalman_smooth_rejects_nonpositive_noise() {
        let conn = open_bench().unwrap();
        assert!(
            conn.query_row(
                "SELECT value FROM ix_kalman_smooth('[1,2,3]', 0.0, 1.0)",
                [],
                |r| r.get::<_, f64>(0)
            )
            .is_err(),
            "process_noise=0 must be a SQL error"
        );
    }

    #[test]
    fn dct_constant_concentrates_at_dc() {
        let conn = open_bench().unwrap();
        // A constant signal has all its energy in the DC (k=0) coefficient.
        let dc_bin: i64 = conn
            .query_row(
                "SELECT k FROM ix_dct('[1,1,1,1,1,1,1,1]') ORDER BY abs(coefficient) DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dc_bin, 0, "constant signal → energy at the k=0 DC coefficient");
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
