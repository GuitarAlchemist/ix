//! Pipeline mesh — compose N IX "pipelines" over many streams and correlate them
//! on the DuckDB analyst bench. The substrate decided in
//! [ADR-0004](../../docs/adr/0004-duckdb-sql-pipeline-mesh.md).
//!
//! A *pipeline* is a named SQL view/macro over a JSON-on-disk stream composing IX
//! UDFs; the *mesh* is the N×N correlation across their outputs. [`correlate`] runs
//! the canonical shape end-to-end:
//!
//! ```text
//!   N streams → ix_wavelet_denoise (condition) → ix_pearson (pairwise)
//!             → threshold → ix_connected_components (clusters)
//!             → ix_centrality (lead / hub indicator)
//! ```
//!
//! and returns a structured [`MeshResult`]. [`install_pipeline_catalog`] registers
//! a few reusable table macros as the SQL-native declaration layer.
//!
//! Advisory analysis only — never a binding gate or source of truth (`docs/DUCKDB.md`).

use std::collections::BTreeMap;

use duckdb::Connection;

/// Which `ix_centrality` measure ranks the mesh's lead/hub stream.
///
/// **Note:** a hub-and-spoke correlation graph is *bipartite*, on which
/// [`Eigenvector`](CentralityKind::Eigenvector) oscillates and mislabels every node
/// equal; [`Betweenness`](CentralityKind::Betweenness) (the default) and
/// [`Degree`](CentralityKind::Degree) are the correct hub lenses. Eigenvector suits
/// dense, mutually-correlated clusters. See ADR-0004.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralityKind {
    Degree,
    Closeness,
    Eigenvector,
    Betweenness,
}

impl CentralityKind {
    fn as_sql(self) -> &'static str {
        match self {
            CentralityKind::Degree => "degree",
            CentralityKind::Closeness => "closeness",
            CentralityKind::Eigenvector => "eigenvector",
            CentralityKind::Betweenness => "betweenness",
        }
    }
}

/// How a mesh is built: smoothing, the correlation-edge threshold, and the
/// lead-indicator centrality.
#[derive(Debug, Clone, Copy)]
pub struct MeshConfig {
    /// `|r|` threshold above which two streams get a correlation edge.
    pub threshold: f64,
    /// Optional per-stream wavelet smoothing `(levels, threshold)`; `None` = raw.
    pub smooth: Option<(i64, f64)>,
    /// Centrality measure used to pick the lead/hub stream.
    pub centrality: CentralityKind,
}

impl Default for MeshConfig {
    fn default() -> Self {
        // Betweenness is the robust hub lens (bipartite-safe); 0.4 separates
        // orthogonal pairs (≈0) from a shared latent driver (≈0.58). See ADR-0004.
        Self {
            threshold: 0.4,
            smooth: Some((2, 0.05)),
            centrality: CentralityKind::Betweenness,
        }
    }
}

/// The correlated structure of a stream mesh.
#[derive(Debug, Clone)]
pub struct MeshResult {
    /// Stream names, indexed as everything else is.
    pub names: Vec<String>,
    /// `correlation[i][j]` = Pearson r between streams i and j (1.0 on the diagonal;
    /// 0.0 for pairs where r is undefined, e.g. a constant stream).
    pub correlation: Vec<Vec<f64>>,
    /// Incident clusters (weakly-connected components of the `|r| ≥ threshold` graph),
    /// each a sorted list of stream indices.
    pub clusters: Vec<Vec<usize>>,
    /// `(stream index, centrality score)` sorted by score descending.
    pub centrality: Vec<(usize, f64)>,
    /// The lead/hub stream index (highest centrality), or `None` for an empty mesh.
    pub lead: Option<usize>,
}

impl MeshResult {
    /// The name of the lead/hub stream, if any.
    pub fn lead_name(&self) -> Option<&str> {
        self.lead.map(|i| self.names[i].as_str())
    }
}

fn arr(v: &[f64]) -> String {
    let body: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", body.join(","))
}

/// Run the canonical correlation mesh over `streams` (each `(name, series)`).
///
/// Returns the correlation matrix, incident clusters, centrality ranking, and the
/// lead stream. Streams shorter than 2 samples, or whose Pearson pair is undefined
/// (a constant stream), simply yield no edge rather than aborting.
// @ai:invariant mesh::correlate composes ix_wavelet_denoise → ix_pearson → ix_connected_components → ix_centrality; a hub-and-spoke input (one stream correlated with several mutually-uncorrelated others) yields one cluster containing all of them and ranks the hub highest under betweenness [T:test conf:0.85 src:ix_duck::mesh::tests::hub_and_spoke_mesh]
pub fn correlate(
    conn: &Connection,
    streams: &[(String, Vec<f64>)],
    cfg: &MeshConfig,
) -> duckdb::Result<MeshResult> {
    let n = streams.len();
    let names: Vec<String> = streams.iter().map(|(name, _)| name.clone()).collect();

    // 1. Condition each stream (optional wavelet smoothing).
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (_, raw) in streams {
        if let Some((levels, threshold)) = cfg.smooth {
            let sql = format!(
                "SELECT value FROM ix_wavelet_denoise('{}', {levels}, {threshold}) ORDER BY i",
                arr(raw)
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map([], |row| row.get::<_, f64>(0))?;
            series.push(rows.collect::<Result<Vec<f64>, _>>()?);
        } else {
            series.push(raw.clone());
        }
    }

    // 2. Pairwise Pearson correlation matrix (upper triangle, mirrored).
    let mut correlation = vec![vec![0.0f64; n]; n];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        correlation[i][i] = 1.0;
        for j in (i + 1)..n {
            // A constant stream makes Pearson undefined → SQL error → treat as r = 0
            // (no edge) so one degenerate stream can't sink the whole mesh.
            let rij: f64 = conn
                .query_row(
                    &format!(
                        "SELECT ix_pearson({}::DOUBLE[], {}::DOUBLE[])",
                        arr(&series[i]),
                        arr(&series[j])
                    ),
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            correlation[i][j] = rij;
            correlation[j][i] = rij;
            if rij.abs() >= cfg.threshold {
                edges.push((i, j));
            }
        }
    }

    // 3. Edge list (self-loops keep every stream a node even with no strong edge).
    let mut edge_json: Vec<String> = (0..n).map(|i| format!("[{i},{i}]")).collect();
    edge_json.extend(edges.iter().map(|(i, j)| format!("[{i},{j}]")));
    let edge_list = format!("[{}]", edge_json.join(","));

    // 4. Incident clusters via ix_connected_components.
    let mut by_comp: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    if n > 0 {
        let mut stmt = conn.prepare(&format!(
            "SELECT node, component FROM ix_connected_components('{edge_list}') ORDER BY node"
        ))?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for row in rows {
            let (node, comp) = row?;
            by_comp.entry(comp).or_default().push(node as usize);
        }
    }
    let clusters: Vec<Vec<usize>> = by_comp.into_values().collect();

    // 5. Lead/hub via ix_centrality.
    let mut centrality: Vec<(usize, f64)> = Vec::with_capacity(n);
    if n > 0 {
        let mut stmt = conn.prepare(&format!(
            "SELECT node, score FROM ix_centrality('{edge_list}', '{}') ORDER BY score DESC, node",
            cfg.centrality.as_sql()
        ))?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)))?;
        for row in rows {
            let (node, score) = row?;
            centrality.push((node as usize, score));
        }
    }
    let lead = centrality.first().map(|&(i, _)| i);

    Ok(MeshResult {
        names,
        correlation,
        clusters,
        centrality,
        lead,
    })
}

/// Install the reusable **pipeline catalog** — table macros that name a conditioning
/// pipeline so callers can `SELECT … FROM ix_smooth('[…]', 2, 0.05)` declaratively.
/// This is the SQL-native declaration layer of ADR-0004; the 100+ mesh is a catalog
/// of such macros over different sources.
pub fn install_pipeline_catalog(conn: &Connection) -> duckdb::Result<()> {
    conn.execute_batch(
        "CREATE OR REPLACE MACRO ix_smooth(series, levels, threshold) AS TABLE
            SELECT i, value FROM ix_wavelet_denoise(series, levels, threshold);
         CREATE OR REPLACE MACRO ix_kalman_pipeline(series, process_noise, measurement_noise) AS TABLE
            SELECT i, value FROM ix_kalman_smooth(series, process_noise, measurement_noise);",
    )
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;
    use crate::open_bench;

    /// Deterministic ±0.05 pseudo-noise (no RNG → reproducible).
    fn noise(t: usize, seed: u64) -> f64 {
        let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x as f64 / u64::MAX as f64 - 0.5) * 0.1
    }
    fn tone(f: f64, seed: u64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * f * t as f64 / n as f64).sin() + noise(t, seed))
            .collect()
    }

    #[test]
    fn hub_and_spoke_mesh() {
        let conn = open_bench().unwrap();
        let n = 64;
        let p = tone(1.0, 11, n);
        let q = tone(2.0, 22, n);
        let r = tone(3.0, 33, n);
        let h: Vec<f64> = (0..n).map(|t| (p[t] + q[t] + r[t]) / 3.0 + noise(t, 99)).collect();
        let d1 = tone(5.0, 44, n);
        let d2 = tone(6.0, 55, n);
        let streams = vec![
            ("H".into(), h),
            ("P".into(), p),
            ("Q".into(), q),
            ("R".into(), r),
            ("D1".into(), d1),
            ("D2".into(), d2),
        ];

        let res = correlate(&conn, &streams, &MeshConfig::default()).unwrap();

        // H correlates with each spoke; spokes are mutually orthogonal.
        assert!(res.correlation[0][1] > 0.4, "H↔P correlated, got {}", res.correlation[0][1]);
        assert!(res.correlation[1][2].abs() < 0.3, "P↔Q orthogonal, got {}", res.correlation[1][2]);

        // One cluster holds {H,P,Q,R}; D1 and D2 are isolated → 3 clusters total.
        assert_eq!(res.clusters.len(), 3, "expected {{H,P,Q,R}}, {{D1}}, {{D2}}");
        let hub_cluster = res.clusters.iter().find(|c| c.contains(&0)).unwrap();
        assert_eq!(hub_cluster.len(), 4, "H,P,Q,R cluster together");
        assert!(hub_cluster.contains(&1) && hub_cluster.contains(&2) && hub_cluster.contains(&3));

        // Betweenness ranks the hub H (index 0) as the lead.
        assert_eq!(res.lead, Some(0), "H is the lead/hub");
        assert_eq!(res.lead_name(), Some("H"));
    }

    #[test]
    fn constant_stream_yields_no_edge_not_error() {
        let conn = open_bench().unwrap();
        // A flat (constant) stream makes Pearson undefined; the mesh must treat it as
        // uncorrelated rather than erroring out.
        let streams = vec![
            ("flat".into(), vec![5.0; 16]),
            ("ramp".into(), (0..16).map(|t| t as f64).collect()),
        ];
        let res = correlate(&conn, &streams, &MeshConfig { smooth: None, ..Default::default() })
            .unwrap();
        assert!(res.correlation[0][1].abs() < 1e-9, "constant↔ramp → r treated as 0");
        assert_eq!(res.clusters.len(), 2, "two isolated streams");
    }

    #[test]
    fn pipeline_catalog_macro_runs() {
        let conn = open_bench().unwrap();
        install_pipeline_catalog(&conn).unwrap();
        // The declared ix_smooth pipeline macro composes like any table function.
        let n: i64 = conn
            .query_row("SELECT count(*) FROM ix_smooth('[1,2,3,4,5,6,7,8]', 2, 0.05)", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n, 8, "ix_smooth macro returns the conditioned series");
    }
}
