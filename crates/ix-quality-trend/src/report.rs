//! Markdown report emission.
//!
//! Given a [`SnapshotSet`] and a regression threshold, builds metric series,
//! computes trends, and formats the final markdown document.

use std::fmt::Write as _;
use std::path::Path;

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::snapshot::{
    ChatbotQaSnapshot, EmbeddingsSnapshot, SnapshotSet, VoicingAnalysisSnapshot,
};
use crate::trend::{compute_trend, MetricSeries, MetricTrend, TrendDirection};

/// Top-level metric catalogue — the rows that appear in the headline table.
/// Order here is the order shown in the report.
const PARTITIONS: &[&str] = &["STRUCTURE", "MORPHOLOGY", "CONTEXT", "SYMBOLIC", "MODAL"];

/// Structured quality summary that can be consumed by governance or telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendSummary {
    pub embedding_trends: Vec<MetricTrend>,
    pub voicing_trends: Vec<MetricTrend>,
    pub chatbot_trends: Vec<MetricTrend>,
}

impl QualityTrendSummary {
    pub fn all_trends(&self) -> impl Iterator<Item = &MetricTrend> {
        self.embedding_trends
            .iter()
            .chain(self.voicing_trends.iter())
            .chain(self.chatbot_trends.iter())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityHealthStatus {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    pub metric: String,
    pub status: QualityHealthStatus,
    pub regression: Option<String>,
    pub drift: Option<String>,
    pub drift_since: Option<NaiveDate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityHealthArtifact {
    pub generated_on: NaiveDate,
    pub regression_threshold_pct: f64,
    pub status: QualityHealthStatus,
    pub total_metrics: usize,
    pub regressions: usize,
    pub drifts: usize,
    pub key_metric_alerts: Vec<QualityAlert>,
    pub all_alerts: Vec<QualityAlert>,
}

pub fn summarize(set: &SnapshotSet, regression_threshold_pct: f64) -> QualityTrendSummary {
    QualityTrendSummary {
        embedding_trends: embedding_metrics(&set.embeddings, regression_threshold_pct),
        voicing_trends: voicing_metrics(&set.voicing, regression_threshold_pct),
        chatbot_trends: chatbot_metrics(&set.chatbot, regression_threshold_pct),
    }
}

pub fn build_health_artifact(
    summary: &QualityTrendSummary,
    regression_threshold_pct: f64,
) -> QualityHealthArtifact {
    let all_alerts: Vec<QualityAlert> = summary
        .all_trends()
        .filter_map(|trend| {
            if trend.regression.is_none() && trend.drift.is_none() {
                return None;
            }

            let status = if is_key_metric_name(&trend.name) && trend.drift.is_some() {
                QualityHealthStatus::Critical
            } else {
                QualityHealthStatus::Warning
            };

            Some(QualityAlert {
                metric: trend.name.clone(),
                status,
                regression: trend.regression.as_ref().map(|r| r.description.clone()),
                drift: trend.drift.as_ref().map(|d| d.description.clone()),
                drift_since: trend.drift.as_ref().map(|d| d.since),
            })
        })
        .collect();
    let key_metric_alerts: Vec<QualityAlert> = all_alerts
        .iter()
        .filter(|alert| is_key_metric_name(&alert.metric))
        .cloned()
        .collect();

    let status = if key_metric_alerts
        .iter()
        .any(|alert| alert.status == QualityHealthStatus::Critical)
    {
        QualityHealthStatus::Critical
    } else if !all_alerts.is_empty() {
        QualityHealthStatus::Warning
    } else {
        QualityHealthStatus::Healthy
    };

    QualityHealthArtifact {
        generated_on: Utc::now().date_naive(),
        regression_threshold_pct,
        status,
        total_metrics: summary.all_trends().count(),
        regressions: summary
            .all_trends()
            .filter(|trend| trend.regression.is_some())
            .count(),
        drifts: summary
            .all_trends()
            .filter(|trend| trend.drift.is_some())
            .count(),
        key_metric_alerts,
        all_alerts,
    }
}

pub fn is_key_metric_name(name: &str) -> bool {
    matches!(
        name,
        "Voicing · cross-instrument consistency"
            | "Voicing · ChordName-Unknown rate"
            | "Embeddings · STRUCTURE leak accuracy"
            | "Embeddings · retrieval PC-set match (top-10)"
            | "Chatbot · overall pass rate"
            | "Chatbot · avg response time"
    )
}

/// Render a complete markdown report.
pub fn render(set: &SnapshotSet, snapshots_dir: &Path, regression_threshold_pct: f64) -> String {
    let today = Utc::now().date_naive();
    let mut out = String::new();

    writeln!(out, "# Quality Trend Report — {today}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "_Generated by `ix-quality-trend` on {today}. Regression threshold: {regression_threshold_pct:.1}% vs 7-day average._"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Build all trends up front so we can emit the headline and the
    // regression section before the detail sections.
    let summary = summarize(set, regression_threshold_pct);
    let embedding_trends = &summary.embedding_trends;
    let voicing_trends = &summary.voicing_trends;
    let chatbot_trends = &summary.chatbot_trends;

    render_headline(&mut out, embedding_trends, voicing_trends, chatbot_trends);
    render_regressions(&mut out, embedding_trends, voicing_trends, chatbot_trends);
    render_drift(&mut out, embedding_trends, voicing_trends, chatbot_trends);

    render_section(
        &mut out,
        "Embeddings detail",
        embedding_trends,
        set.embeddings.len(),
    );
    render_section(
        &mut out,
        "Voicing analysis detail",
        voicing_trends,
        set.voicing.len(),
    );
    render_section(
        &mut out,
        "Chatbot QA detail",
        chatbot_trends,
        set.chatbot.len(),
    );

    render_methodology(&mut out, snapshots_dir, set, regression_threshold_pct);

    out
}

// ---------------------------------------------------------------------------
// Metric construction
// ---------------------------------------------------------------------------

fn embedding_metrics(
    series: &[crate::snapshot::DatedSnapshot<EmbeddingsSnapshot>],
    thresh: f64,
) -> Vec<MetricTrend> {
    let mut metrics: Vec<MetricSeries> = Vec::new();

    for p in PARTITIONS {
        let mut ms = MetricSeries::new(
            format!("Embeddings · {p} leak accuracy"),
            "%",
            TrendDirection::LowerIsBetter,
        );
        for snap in series {
            if let Some(v) = snap.data.partition_accuracy(p) {
                ms.push(snap.date, v * 100.0);
            }
        }
        metrics.push(ms);
    }

    let mut retr = MetricSeries::new(
        "Embeddings · retrieval PC-set match (top-10)",
        "%",
        TrendDirection::HigherIsBetter,
    );
    for snap in series {
        if let Some(rc) = &snap.data.retrieval_consistency {
            if let Some(v) = rc.match_pct() {
                retr.push(snap.date, v * 100.0);
            }
        }
    }
    metrics.push(retr);

    for (inst, get) in [
        ("guitar", topology_getter(|t| t.guitar.as_ref())),
        ("bass", topology_getter(|t| t.bass.as_ref())),
        ("ukulele", topology_getter(|t| t.ukulele.as_ref())),
    ] {
        let mut b0 =
            MetricSeries::new(format!("Topology · {inst} β₀"), "", TrendDirection::Neutral);
        let mut b1 =
            MetricSeries::new(format!("Topology · {inst} β₁"), "", TrendDirection::Neutral);
        for snap in series {
            if let Some(topo) = &snap.data.topology {
                if let Some(b) = get(topo) {
                    if let Some(v) = b.beta_0 {
                        b0.push(snap.date, v as f64);
                    }
                    if let Some(v) = b.beta_1 {
                        b1.push(snap.date, v as f64);
                    }
                }
            }
        }
        metrics.push(b0);
        metrics.push(b1);
    }

    metrics
        .into_iter()
        .map(|s| compute_trend(&s, thresh))
        .collect()
}

/// Returns a closure that extracts a given instrument's BettiNumbers from a
/// `Topology`. Factoring this out keeps the series-builder readable.
fn topology_getter(
    f: fn(&crate::snapshot::Topology) -> Option<&crate::snapshot::BettiNumbers>,
) -> fn(&crate::snapshot::Topology) -> Option<&crate::snapshot::BettiNumbers> {
    f
}

fn voicing_metrics(
    series: &[crate::snapshot::DatedSnapshot<VoicingAnalysisSnapshot>],
    thresh: f64,
) -> Vec<MetricTrend> {
    let mut metrics = Vec::new();

    let mut unknown = MetricSeries::new(
        "Voicing · ChordName-Unknown rate",
        "%",
        TrendDirection::LowerIsBetter,
    );
    let mut consistency = MetricSeries::new(
        "Voicing · cross-instrument consistency",
        "%",
        TrendDirection::HigherIsBetter,
    );
    let mut forte = MetricSeries::new(
        "Voicing · Forte coverage",
        "%",
        TrendDirection::HigherIsBetter,
    );
    let mut invariants = MetricSeries::new(
        "Voicing · invariant failures (total)",
        "",
        TrendDirection::LowerIsBetter,
    );
    let mut total = MetricSeries::new("Voicing · corpus size", "", TrendDirection::Neutral);

    for snap in series {
        if let Some(cr) = &snap.data.chord_recognition {
            if let Some(u) = &cr.unknown_chord_name {
                if let Some(p) = u.pct {
                    unknown.push(snap.date, p);
                }
            }
        }
        if let Some(c) = &snap.data.cross_instrument_consistency {
            if let Some(p) = c.consistency_pct() {
                consistency.push(snap.date, p);
            }
        }
        if let Some(f) = &snap.data.forte_coverage {
            if let Some(p) = f.pct {
                forte.push(snap.date, p);
            }
        }
        if let Some(i) = &snap.data.invariant_failures {
            invariants.push(snap.date, i.total() as f64);
        }
        if let Some(c) = &snap.data.corpus {
            if let Some(t) = c.total {
                total.push(snap.date, t as f64);
            }
        }
    }

    metrics.push(unknown);
    metrics.push(consistency);
    metrics.push(forte);
    metrics.push(invariants);
    metrics.push(total);

    metrics
        .into_iter()
        .map(|s| compute_trend(&s, thresh))
        .collect()
}

fn chatbot_metrics(
    series: &[crate::snapshot::DatedSnapshot<ChatbotQaSnapshot>],
    thresh: f64,
) -> Vec<MetricTrend> {
    let mut metrics = Vec::new();

    let mut overall = MetricSeries::new(
        "Chatbot · overall pass rate",
        "%",
        TrendDirection::HigherIsBetter,
    );
    let mut latency = MetricSeries::new(
        "Chatbot · avg response time",
        "ms",
        TrendDirection::LowerIsBetter,
    );

    for snap in series {
        if let Some(p) = snap.data.pass_pct {
            overall.push(snap.date, p);
        }
        if let Some(ms) = snap.data.avg_response_ms {
            latency.push(snap.date, ms as f64);
        }
    }
    metrics.push(overall);
    metrics.push(latency);

    // Per-level metrics — discovered dynamically so new levels don't require a
    // code change. Ordered L1..L9 alphabetically, so sorting the keys is enough.
    let mut level_keys: std::collections::BTreeSet<String> = Default::default();
    for snap in series {
        if let Some(cat) = &snap.data.by_category {
            for k in cat.keys() {
                level_keys.insert(k.clone());
            }
        }
    }
    for key in level_keys {
        let mut ms = MetricSeries::new(
            format!("Chatbot · {key} pass rate"),
            "%",
            TrendDirection::HigherIsBetter,
        );
        for snap in series {
            if let Some(cat) = &snap.data.by_category {
                if let Some(stats) = cat.get(&key) {
                    if let Some(p) = stats.pass_pct {
                        ms.push(snap.date, p);
                    }
                }
            }
        }
        metrics.push(ms);
    }

    metrics
        .into_iter()
        .map(|s| compute_trend(&s, thresh))
        .collect()
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_headline(
    out: &mut String,
    emb: &[MetricTrend],
    voi: &[MetricTrend],
    cha: &[MetricTrend],
) {
    writeln!(out, "## Headline").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Metric | Latest | 7d avg | 30d avg | Trend |").unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();

    for t in headline_picks(emb, voi, cha) {
        writeln!(
            out,
            "| {} | {} | {} | {} | {} {} |",
            t.name,
            fmt_value(t.latest, &t.unit),
            fmt_value(t.avg_7d, &t.unit),
            fmt_value(t.avg_30d, &t.unit),
            if t.sparkline.is_empty() {
                "—".into()
            } else {
                t.sparkline.clone()
            },
            t.marker()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

fn headline_picks<'a>(
    emb: &'a [MetricTrend],
    voi: &'a [MetricTrend],
    cha: &'a [MetricTrend],
) -> Vec<&'a MetricTrend> {
    let wanted: &[&str] = &[
        "Voicing · cross-instrument consistency",
        "Voicing · ChordName-Unknown rate",
        "Voicing · Forte coverage",
        "Voicing · invariant failures (total)",
        "Embeddings · STRUCTURE leak accuracy",
        "Embeddings · retrieval PC-set match (top-10)",
        "Chatbot · overall pass rate",
    ];
    let mut picks: Vec<&MetricTrend> = Vec::new();
    for name in wanted {
        if let Some(t) = find(emb, name)
            .or_else(|| find(voi, name))
            .or_else(|| find(cha, name))
        {
            if t.latest.is_some() {
                picks.push(t);
            }
        }
    }
    picks
}

fn find<'a>(trends: &'a [MetricTrend], name: &str) -> Option<&'a MetricTrend> {
    trends.iter().find(|t| t.name == name)
}

fn render_regressions(
    out: &mut String,
    emb: &[MetricTrend],
    voi: &[MetricTrend],
    cha: &[MetricTrend],
) {
    let all: Vec<&MetricTrend> = emb.iter().chain(voi.iter()).chain(cha.iter()).collect();
    let flags: Vec<&MetricTrend> = all
        .iter()
        .filter(|t| t.regression.is_some())
        .copied()
        .collect();

    writeln!(out, "## Regressions flagged").unwrap();
    writeln!(out).unwrap();
    if flags.is_empty() {
        writeln!(
            out,
            "_None. All tracked metrics are within threshold vs their 7-day average._"
        )
        .unwrap();
        writeln!(out).unwrap();
        return;
    }
    for t in flags {
        if let Some(r) = &t.regression {
            writeln!(out, "- **{}** — {}", t.name, r.description).unwrap();
        }
    }
    writeln!(out).unwrap();
}

fn render_drift(out: &mut String, emb: &[MetricTrend], voi: &[MetricTrend], cha: &[MetricTrend]) {
    let all: Vec<&MetricTrend> = emb.iter().chain(voi.iter()).chain(cha.iter()).collect();
    let flags: Vec<&MetricTrend> = all.iter().filter(|t| t.drift.is_some()).copied().collect();

    writeln!(out, "## Drift detected").unwrap();
    writeln!(out).unwrap();
    if flags.is_empty() {
        writeln!(
            out,
            "_None. No persistent regime shifts were detected in the bad direction._"
        )
        .unwrap();
        writeln!(out).unwrap();
        return;
    }

    for t in flags {
        if let Some(d) = &t.drift {
            writeln!(out, "- **{}** — {}", t.name, d.description).unwrap();
        }
    }
    writeln!(out).unwrap();
}

fn render_section(out: &mut String, title: &str, trends: &[MetricTrend], snapshot_count: usize) {
    writeln!(out, "## {title}").unwrap();
    writeln!(out).unwrap();
    if snapshot_count == 0 {
        writeln!(out, "_No snapshots in this category yet._").unwrap();
        writeln!(out).unwrap();
        return;
    }
    writeln!(
        out,
        "_{snapshot_count} snapshot(s) loaded. Sparklines show up to 24 most recent points._"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Metric | Latest | Δ vs prev | Δ vs 7d | Drift | 30d avg | n | Sparkline |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|").unwrap();

    for t in trends {
        writeln!(
            out,
            "| {} {} | {} | {} | {} | {} | {} | {} | {} |",
            t.marker(),
            t.name,
            fmt_value(t.latest, &t.unit),
            fmt_delta(t.delta_vs_previous_pct),
            fmt_delta(t.delta_vs_7d_pct),
            fmt_drift(t),
            fmt_value(t.avg_30d, &t.unit),
            t.n_points,
            if t.sparkline.is_empty() {
                "—".into()
            } else {
                t.sparkline.clone()
            },
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

fn render_methodology(out: &mut String, snapshots_dir: &Path, set: &SnapshotSet, thresh: f64) {
    writeln!(out, "## Methodology").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- **Snapshots root:** `{}`", snapshots_dir.display()).unwrap();
    writeln!(
        out,
        "- **Snapshots loaded:** {} embeddings / {} voicing-analysis / {} chatbot-qa",
        set.embeddings.len(),
        set.voicing.len(),
        set.chatbot.len()
    )
    .unwrap();
    writeln!(
        out,
        "- **Regression threshold:** {:.1}% absolute Δ vs 7-day average, in the \"bad\" direction",
        thresh
    )
    .unwrap();
    writeln!(
        out,
        "- **Drift detection:** Page-Hinkley over direction-adjusted metric levels; only higher-is-better and lower-is-better metrics participate"
    )
    .unwrap();
    writeln!(
        out,
        "- **Direction semantics:** higher-is-better (pass rates, coverage, consistency), lower-is-better (leak accuracy, unknown-chord rate, invariant failures), neutral (corpus counts, Betti numbers — reported for context only)"
    )
    .unwrap();
    writeln!(
        out,
        "- **Date source:** filename stem parsed as `YYYY-MM-DD`; files without a parseable date are skipped"
    )
    .unwrap();
    writeln!(
        out,
        "- **Producers:** `ix-embedding-diagnostics` (Rust), `Demos/VoicingAnalysisAudit` (.NET), `ga-chatbot qa --benchmark` (Rust)"
    )
    .unwrap();
    if let Some((first, last)) = date_range(set) {
        writeln!(out, "- **Date range:** {first} … {last}").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "Methodology reference: [`docs/methodology/invariants-catalog.md`](../methodology/invariants-catalog.md)."
    )
    .unwrap();
}

fn date_range(set: &SnapshotSet) -> Option<(NaiveDate, NaiveDate)> {
    let mut min = None;
    let mut max = None;
    for d in set
        .embeddings
        .iter()
        .map(|s| s.date)
        .chain(set.voicing.iter().map(|s| s.date))
        .chain(set.chatbot.iter().map(|s| s.date))
    {
        min = Some(min.map_or(d, |m: NaiveDate| m.min(d)));
        max = Some(max.map_or(d, |m: NaiveDate| m.max(d)));
    }
    match (min, max) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn fmt_value(v: Option<f64>, unit: &str) -> String {
    match v {
        None => "—".into(),
        Some(x) if unit == "%" => format!("{x:.2}%"),
        Some(x) if unit == "ms" => format!("{x:.0} ms"),
        Some(x) if unit.is_empty() => {
            if x.fract().abs() < 1e-9 && x.abs() < 1e12 {
                format!("{}", x as i64)
            } else {
                format!("{x:.2}")
            }
        }
        Some(x) => format!("{x:.2} {unit}"),
    }
}

fn fmt_delta(v: Option<f64>) -> String {
    match v {
        None => "—".into(),
        Some(x) if x.abs() < 0.05 => "0.0%".into(),
        Some(x) if x > 0.0 => format!("+{x:.1}%"),
        Some(x) => format!("{x:.1}%"),
    }
}

fn fmt_drift(t: &MetricTrend) -> String {
    match &t.drift {
        Some(d) => format!("since {}", d.since),
        None => "—".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{DatedSnapshot, EmbeddingsSnapshot, SnapshotSet};
    use chrono::NaiveDate;

    #[test]
    fn empty_set_renders_no_data_sections() {
        let set = SnapshotSet::default();
        let md = render(&set, Path::new("/nowhere"), 5.0);
        assert!(md.contains("# Quality Trend Report"));
        assert!(md.contains("_No snapshots in this category yet._"));
        assert!(md.contains("_None. All tracked metrics"));
        assert!(md.contains("## Drift detected"));
    }

    #[test]
    fn render_with_single_embedding_snapshot() {
        let mut set = SnapshotSet::default();
        let snap: EmbeddingsSnapshot = serde_json::from_str(
            r#"{"leak_detection":{"by_partition":[{"partition":"STRUCTURE","accuracy_mean":0.56}]},"retrieval_consistency":{"avg_pc_set_match_pct":0.87}}"#,
        )
        .unwrap();
        set.embeddings.push(DatedSnapshot {
            date: NaiveDate::from_ymd_opt(2026, 4, 17).unwrap(),
            path: "/tmp/x.json".into(),
            data: snap,
        });
        let md = render(&set, Path::new("/tmp"), 5.0);
        assert!(md.contains("STRUCTURE leak accuracy"));
        assert!(md.contains("56.00%") || md.contains("56.0%"));
    }

    #[test]
    fn render_reports_drift_for_persistent_regime_shift() {
        let mut set = SnapshotSet::default();
        for day in 1..=10 {
            let accuracy = if day <= 5 { 0.40 } else { 0.78 };
            let snap: EmbeddingsSnapshot = serde_json::from_str(&format!(
                r#"{{"leak_detection":{{"by_partition":[{{"partition":"STRUCTURE","accuracy_mean":{accuracy}}}]}}}}"#
            ))
            .unwrap();
            set.embeddings.push(DatedSnapshot {
                date: NaiveDate::from_ymd_opt(2026, 4, day).unwrap(),
                path: format!("/tmp/{day}.json").into(),
                data: snap,
            });
        }

        let md = render(&set, Path::new("/tmp"), 5.0);
        assert!(md.contains("## Drift detected"));
        assert!(md.contains("Embeddings · STRUCTURE leak accuracy"));
        assert!(md.contains("worse regime"));
    }

    #[test]
    fn summarize_exposes_structured_drift_and_regression_flags() {
        let mut set = SnapshotSet::default();
        for day in 1..=10 {
            let accuracy = if day <= 5 { 0.40 } else { 0.78 };
            let snap: EmbeddingsSnapshot = serde_json::from_str(&format!(
                r#"{{"leak_detection":{{"by_partition":[{{"partition":"STRUCTURE","accuracy_mean":{accuracy}}}]}}}}"#
            ))
            .unwrap();
            set.embeddings.push(DatedSnapshot {
                date: NaiveDate::from_ymd_opt(2026, 4, day).unwrap(),
                path: format!("/tmp/{day}.json").into(),
                data: snap,
            });
        }

        let summary = summarize(&set, 5.0);
        assert!(summary
            .all_trends()
            .any(|trend| trend.regression.is_some() || trend.drift.is_some()));
    }

    #[test]
    fn health_artifact_marks_key_metric_drift_as_critical() {
        let mut set = SnapshotSet::default();
        for day in 1..=10 {
            let accuracy = if day <= 5 { 0.40 } else { 0.78 };
            let snap: EmbeddingsSnapshot = serde_json::from_str(&format!(
                r#"{{"leak_detection":{{"by_partition":[{{"partition":"STRUCTURE","accuracy_mean":{accuracy}}}]}}}}"#
            ))
            .unwrap();
            set.embeddings.push(DatedSnapshot {
                date: NaiveDate::from_ymd_opt(2026, 4, day).unwrap(),
                path: format!("/tmp/{day}.json").into(),
                data: snap,
            });
        }

        let summary = summarize(&set, 5.0);
        let artifact = build_health_artifact(&summary, 5.0);
        assert_eq!(artifact.status, QualityHealthStatus::Critical);
        assert!(artifact
            .key_metric_alerts
            .iter()
            .any(|alert| alert.metric == "Embeddings · STRUCTURE leak accuracy"));
    }
}
