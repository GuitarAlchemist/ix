//! Trend analysis over sorted date-indexed metric series.
//!
//! A [`MetricSeries`] is a dated series of `(date, value)` points. From it we
//! compute a [`MetricTrend`] with headline numbers (latest, 7-day average,
//! 30-day average) and a sparkline. Regressions are flagged when the absolute
//! percentage delta between `latest` and the 7-day average exceeds a
//! configurable threshold.
//!
//! # Direction semantics
//!
//! Some metrics are "higher is better" (cross-instrument consistency, Forte
//! coverage, pass rates) and some are "lower is better" (leak accuracy,
//! ChordName-Unknown rate, invariant failures). The [`TrendDirection`]
//! encodes this so the report can render a coherent arrow/checkmark.

use chrono::NaiveDate;
use ix_signal::timeseries::{page_hinkley_detect, DriftState, PageHinkleyConfig};
use serde::{Deserialize, Serialize};

/// Whether a metric is better when the value goes up, down, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    HigherIsBetter,
    LowerIsBetter,
    Neutral,
}

/// One observation in a time-indexed series.
#[derive(Debug, Clone)]
pub struct Point {
    pub date: NaiveDate,
    pub value: f64,
}

/// A named, dated time series of scalar measurements, with semantic direction
/// and a display unit suffix.
#[derive(Debug, Clone)]
pub struct MetricSeries {
    pub name: String,
    pub unit: String,
    pub direction: TrendDirection,
    pub points: Vec<Point>,
}

impl MetricSeries {
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        direction: TrendDirection,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            direction,
            points: Vec::new(),
        }
    }

    pub fn push(&mut self, date: NaiveDate, value: f64) {
        self.points.push(Point { date, value });
    }

    pub fn with(mut self, date: NaiveDate, value: Option<f64>) -> Self {
        if let Some(v) = value {
            self.push(date, v);
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn latest(&self) -> Option<&Point> {
        self.points.last()
    }

    pub fn previous(&self) -> Option<&Point> {
        if self.points.len() < 2 {
            None
        } else {
            self.points.get(self.points.len() - 2)
        }
    }

    /// Average over the last `window_days` ending at the most recent point's
    /// date (inclusive). Returns `None` if the series is empty.
    pub fn rolling_average(&self, window_days: i64) -> Option<f64> {
        let last = self.latest()?;
        let cutoff = last.date - chrono::Duration::days(window_days - 1);
        let vals: Vec<f64> = self
            .points
            .iter()
            .filter(|p| p.date >= cutoff)
            .map(|p| p.value)
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    }
}

/// Sparkline rendered as unicode block characters, scaled to the series'
/// observed min/max.
pub fn sparkline(points: &[Point]) -> String {
    if points.is_empty() {
        return String::new();
    }
    const BLOCKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = points.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
    let max = points
        .iter()
        .map(|p| p.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;

    // Cap at the last 24 points so the sparkline stays narrow in reports.
    let tail: &[Point] = if points.len() > 24 {
        &points[points.len() - 24..]
    } else {
        points
    };

    tail.iter()
        .map(|p| {
            let idx = if span.abs() < f64::EPSILON {
                3
            } else {
                let t = (p.value - min) / span;
                ((t * (BLOCKS.len() - 1) as f64).round() as usize).min(BLOCKS.len() - 1)
            };
            BLOCKS[idx]
        })
        .collect()
}

/// Computed trend for one metric: headline values, deltas, and regression flag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub name: String,
    pub unit: String,
    pub direction: TrendDirection,
    pub latest: Option<f64>,
    pub latest_date: Option<NaiveDate>,
    pub previous: Option<f64>,
    pub avg_7d: Option<f64>,
    pub avg_30d: Option<f64>,
    pub delta_vs_previous_pct: Option<f64>,
    pub delta_vs_7d_pct: Option<f64>,
    pub sparkline: String,
    pub regression: Option<RegressionFlag>,
    pub drift: Option<DriftFlag>,
    pub n_points: usize,
}

/// Indicator that a metric has moved in the "wrong" direction by more than the
/// configured threshold vs the 7-day average.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionFlag {
    pub metric: String,
    pub delta_pct: f64,
    pub description: String,
}

/// Indicator that a metric has shifted into a worse regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftFlag {
    pub metric: String,
    pub since: NaiveDate,
    pub description: String,
}

/// Compute the trend summary for a series against a percentage-delta threshold
/// (e.g. `5.0` for 5%). The regression check uses `latest` vs `avg_7d`.
pub fn compute_trend(series: &MetricSeries, regression_threshold_pct: f64) -> MetricTrend {
    let latest_point = series.latest().cloned();
    let previous_point = series.previous().cloned();
    let latest = latest_point.as_ref().map(|p| p.value);
    let previous = previous_point.as_ref().map(|p| p.value);

    let avg_7d = series.rolling_average(7);
    let avg_30d = series.rolling_average(30);

    let delta_vs_previous_pct = match (latest, previous) {
        (Some(l), Some(p)) if p.abs() > f64::EPSILON => Some((l - p) / p.abs() * 100.0),
        _ => None,
    };
    let delta_vs_7d_pct = match (latest, avg_7d) {
        (Some(l), Some(a)) if a.abs() > f64::EPSILON => Some((l - a) / a.abs() * 100.0),
        _ => None,
    };

    // Prefer the 7-day-avg delta when it's meaningful (≥3 points in window);
    // fall back to day-over-day delta when the series is still young. This
    // ensures big regressions get flagged from the second snapshot onwards
    // without waiting for a full week of history to accumulate.
    let use_7d = series.points.len() >= 3 && delta_vs_7d_pct.is_some();
    let comparison_delta = if use_7d {
        delta_vs_7d_pct
    } else {
        delta_vs_previous_pct
    };
    let comparison_label = if use_7d {
        "7-day average"
    } else {
        "previous snapshot"
    };
    let regression = regression_flag(
        &series.name,
        series.direction,
        comparison_delta,
        comparison_label,
        regression_threshold_pct,
    );
    let drift = drift_flag(series);

    MetricTrend {
        name: series.name.clone(),
        unit: series.unit.clone(),
        direction: series.direction,
        latest,
        latest_date: latest_point.map(|p| p.date),
        previous,
        avg_7d,
        avg_30d,
        delta_vs_previous_pct,
        delta_vs_7d_pct,
        sparkline: sparkline(&series.points),
        regression,
        drift,
        n_points: series.points.len(),
    }
}

fn regression_flag(
    name: &str,
    dir: TrendDirection,
    delta_pct: Option<f64>,
    comparison_label: &str,
    threshold: f64,
) -> Option<RegressionFlag> {
    let d = delta_pct?;
    if d.abs() < threshold {
        return None;
    }
    let bad = match dir {
        TrendDirection::HigherIsBetter => d < 0.0,
        TrendDirection::LowerIsBetter => d > 0.0,
        TrendDirection::Neutral => return None,
    };
    if !bad {
        return None;
    }
    let verb = match dir {
        TrendDirection::HigherIsBetter => "dropped",
        TrendDirection::LowerIsBetter => "rose",
        TrendDirection::Neutral => "changed",
    };
    Some(RegressionFlag {
        metric: name.to_string(),
        delta_pct: d,
        description: format!(
            "{name} {verb} {:.1}% vs {comparison_label} — investigate",
            d.abs()
        ),
    })
}

fn drift_flag(series: &MetricSeries) -> Option<DriftFlag> {
    if series.points.len() < 5 {
        return None;
    }

    let oriented: Vec<f64> = match series.direction {
        TrendDirection::HigherIsBetter => series.points.iter().map(|p| -p.value).collect(),
        TrendDirection::LowerIsBetter => series.points.iter().map(|p| p.value).collect(),
        TrendDirection::Neutral => return None,
    };

    let baseline_len = oriented.len().min(7);
    let baseline = &oriented[..baseline_len];
    let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
    let baseline_var = baseline
        .iter()
        .map(|value| (value - baseline_mean).powi(2))
        .sum::<f64>()
        / baseline.len() as f64;
    let baseline_scale = baseline_var.sqrt().max(baseline_mean.abs() * 0.01).max(1.0);
    let normalized: Vec<f64> = oriented
        .iter()
        .map(|value| (value - baseline_mean) / baseline_scale)
        .collect();

    let snapshots = page_hinkley_detect(
        &normalized,
        PageHinkleyConfig {
            min_samples: 5,
            delta: 0.01,
            lambda: 5.0,
            alpha: 1.0,
        },
    );
    let first_drift_index = snapshots
        .iter()
        .position(|snapshot| snapshot.state == DriftState::Drift)?;

    let bad_direction = match series.direction {
        TrendDirection::HigherIsBetter => "downward",
        TrendDirection::LowerIsBetter => "upward",
        TrendDirection::Neutral => unreachable!(),
    };
    let since = series.points[first_drift_index].date;

    Some(DriftFlag {
        metric: series.name.clone(),
        since,
        description: format!(
            "{} shifted {} into a worse regime around {}",
            series.name, bad_direction, since
        ),
    })
}

impl MetricTrend {
    /// Short human-readable marker: ✓ (improving), ⚠ (regression), → (flat).
    pub fn marker(&self) -> &'static str {
        if self.regression.is_some() {
            return "⚠";
        }
        if self.drift.is_some() {
            return "Δ";
        }
        // Use whichever delta is available — prefer 7d when we have real
        // window history, fall back to day-over-day otherwise.
        let delta = self.delta_vs_7d_pct.or(self.delta_vs_previous_pct);
        match (delta, self.direction) {
            (Some(d), TrendDirection::HigherIsBetter) if d > 1.0 => "✓",
            (Some(d), TrendDirection::LowerIsBetter) if d < -1.0 => "✓",
            (None, _) => "·",
            _ => "→",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn rolling_average_windowed() {
        let mut s = MetricSeries::new("m", "%", TrendDirection::HigherIsBetter);
        s.push(d("2026-04-01"), 10.0);
        s.push(d("2026-04-08"), 20.0); // outside 7-day window from 2026-04-15
        s.push(d("2026-04-10"), 30.0);
        s.push(d("2026-04-15"), 50.0);
        // Window: 2026-04-09..2026-04-15 inclusive → 30.0 and 50.0
        let a = s.rolling_average(7).unwrap();
        assert!((a - 40.0).abs() < 1e-9);
    }

    #[test]
    fn regression_triggers_on_drop_for_higher_is_better() {
        let mut s = MetricSeries::new("pass_rate", "%", TrendDirection::HigherIsBetter);
        s.push(d("2026-04-10"), 85.0);
        s.push(d("2026-04-11"), 85.0);
        s.push(d("2026-04-12"), 85.0);
        s.push(d("2026-04-15"), 78.0);
        let t = compute_trend(&s, 5.0);
        assert!(t.regression.is_some(), "expected regression flag");
    }

    #[test]
    fn regression_does_not_trigger_on_improvement() {
        let mut s = MetricSeries::new("unknown_rate", "%", TrendDirection::LowerIsBetter);
        s.push(d("2026-04-10"), 30.0);
        s.push(d("2026-04-11"), 25.0);
        s.push(d("2026-04-12"), 20.0);
        s.push(d("2026-04-15"), 5.0);
        let t = compute_trend(&s, 5.0);
        assert!(t.regression.is_none(), "improvement should not flag");
    }

    #[test]
    fn sparkline_renders_blocks() {
        let mut s = MetricSeries::new("m", "", TrendDirection::Neutral);
        for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            s.push(NaiveDate::from_ymd_opt(2026, 4, 10 + i as u32).unwrap(), *v);
        }
        let sp = sparkline(&s.points);
        assert_eq!(sp.chars().count(), 5);
        // Increasing sequence → first char should be low, last high.
        assert_eq!(sp.chars().next(), Some('▁'));
        assert_eq!(sp.chars().last(), Some('█'));
    }

    #[test]
    fn regression_falls_back_to_day_over_day_when_series_is_young() {
        // Only 2 points — 7-day avg equals the latest and Δ-vs-7d is 0%.
        // Without the fallback a huge day-over-day drop would go unflagged.
        let mut s = MetricSeries::new("pass_rate", "%", TrendDirection::HigherIsBetter);
        s.push(d("2026-04-16"), 85.0);
        s.push(d("2026-04-17"), 70.0);
        let t = compute_trend(&s, 5.0);
        assert!(
            t.regression.is_some(),
            "day-over-day regression should flag even with only 2 points"
        );
        assert!(
            t.regression
                .as_ref()
                .unwrap()
                .description
                .contains("previous snapshot"),
            "young-series regression should cite the previous-snapshot comparison"
        );
    }

    #[test]
    fn empty_series_is_handled() {
        let s = MetricSeries::new("m", "", TrendDirection::Neutral);
        let t = compute_trend(&s, 5.0);
        assert!(t.latest.is_none());
        assert!(t.avg_7d.is_none());
        assert!(t.regression.is_none());
        assert!(t.drift.is_none());
    }

    #[test]
    fn drift_flags_worse_regime_for_higher_is_better_metric() {
        let mut s = MetricSeries::new("pass_rate", "%", TrendDirection::HigherIsBetter);
        for day in 1..=6 {
            s.push(d(&format!("2026-04-{day:02}")), 95.0);
        }
        for day in 7..=12 {
            s.push(d(&format!("2026-04-{day:02}")), 70.0);
        }

        let t = compute_trend(&s, 5.0);
        let drift = t.drift.expect("expected drift flag");
        assert!(drift.since >= d("2026-04-07"));
        assert!(drift.since <= d("2026-04-12"));
        assert!(drift.description.contains("worse regime"));
    }

    #[test]
    fn drift_ignores_neutral_metrics() {
        let mut s = MetricSeries::new("corpus", "", TrendDirection::Neutral);
        for day in 1..=10 {
            s.push(d(&format!("2026-04-{day:02}")), day as f64);
        }

        let t = compute_trend(&s, 5.0);
        assert!(t.drift.is_none());
    }
}
