//! Jarvis J4 tracer bullet — GBDT vs persistence on ONE ga `state/quality/` series.
//!
//! Smallest end-to-end slice of the Jarvis J4 "learned predictive world model"
//! epic (GuitarAlchemist/ix#221). It answers exactly one question with real
//! evidence: **can a gradient-boosted decision tree over hand-crafted history
//! features beat the persistence baseline (last value carries forward) at
//! predicting the next step of a ga quality metric?**
//!
//! Framing follows the reference technique in the issue (Facebook predictive
//! test selection, ICSE-SEIP 2019): a *classifier* over history features, not a
//! neural world model. The target is the **direction of the next step**
//! (`Down` / `Flat` / `Up`). Persistence ("last value carries forward") is the
//! image, in this decision space, of always predicting `Flat` (no change) — so
//! the two models are directly comparable on held-out snapshots.
//!
//! **Honest-pause rule (Karpathy R4).** If a series is too thin to train and
//! evaluate meaningfully, [`evaluate`] returns [`Verdict::PausedInsufficientData`]
//! with the exact counts rather than fabricating a number. A paused verdict
//! committed as evidence IS a valid outcome — beating persistence is not
//! assumed.
//!
//! GBDT is reused from the workspace-internal `ix-ensemble` crate; no new
//! external dependency is introduced. The gradient-boosting fit is an
//! exhaustive threshold search (no RNG), so results are reproducible.

use chrono::NaiveDate;
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;
use ndarray::{Array1, Array2};
use serde::Serialize;
use serde_json::Value;

/// Which ga `state/quality/` sub-series to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Category {
    ChatbotQa,
    Embeddings,
    VoicingAnalysis,
}

impl Category {
    /// Directory name under `state/quality/`.
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::ChatbotQa => "chatbot-qa",
            Self::Embeddings => "embeddings",
            Self::VoicingAnalysis => "voicing-analysis",
        }
    }

    /// Human label for the scalar metric this category exposes.
    pub fn metric_label(self) -> &'static str {
        match self {
            Self::ChatbotQa => "pass_pct",
            Self::Embeddings => "leak_detection_full_classifier_accuracy",
            Self::VoicingAnalysis => "voicing_analysis_avg_pass_rate",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "chatbot-qa" => Some(Self::ChatbotQa),
            "embeddings" => Some(Self::Embeddings),
            "voicing-analysis" => Some(Self::VoicingAnalysis),
            _ => None,
        }
    }

    pub const ALL: [Category; 3] = [Self::ChatbotQa, Self::Embeddings, Self::VoicingAnalysis];
}

/// One usable observation of the target series.
#[derive(Debug, Clone, PartialEq)]
pub struct Obs {
    pub date: NaiveDate,
    pub value: f64,
    /// `true` = a fresh measurement; `false` = degraded / carried-forward
    /// (the producer literally copied the previous value).
    pub real: bool,
}

/// Direction of a step from the previous value. The classification target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Down = 0,
    Flat = 1,
    Up = 2,
}

impl Direction {
    fn from_delta(delta: f64, eps: f64) -> Self {
        if delta > eps {
            Self::Up
        } else if delta < -eps {
            Self::Down
        } else {
            Self::Flat
        }
    }
    fn label(self) -> usize {
        self as usize
    }
}

/// The six hand-crafted, recency-style history features, in matrix-column order.
pub const FEATURE_NAMES: [&str; 6] = [
    "last_value",           // level at t-1
    "last_delta",           // momentum: v[t-1] - v[t-2]
    "flat_run_len",         // consecutive flat steps ending at t-1
    "steps_since_change",   // recency: steps since the last non-flat step
    "last_direction_sign",  // sign of last_delta in {-1, 0, 1}
    "last_was_real",        // 1.0 fresh measurement, 0.0 carried/degraded
];

/// Knobs for the eval. `Default` mirrors the values used in the committed run.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub min_samples_leaf: usize,
    /// Fraction of samples (chronological head) used for training.
    pub train_frac: f64,
    /// `eps = eps_frac * (max - min)` of the real series, floored at 1e-9.
    pub eps_frac: f64,
    /// Guardrail: minimum fresh measurements to attempt an eval.
    pub min_real: usize,
    /// Guardrail: minimum non-flat steps (transition events) to learn from.
    pub min_transitions: usize,
    /// Guardrail: minimum held-out samples.
    pub min_test: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 0.1,
            min_samples_leaf: 1,
            train_frac: 0.7,
            eps_frac: 0.01,
            min_real: 8,
            min_transitions: 3,
            min_test: 3,
        }
    }
}

/// Whether the eval ran or paused per the honest-pause rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Enough data: GBDT and persistence were both evaluated on held-out steps.
    Ran,
    /// Too thin to train/evaluate meaningfully — epic pauses (Karpathy R4).
    PausedInsufficientData,
}

/// The committed evidence for one series.
#[derive(Debug, Clone, Serialize)]
pub struct ForecastEval {
    pub category: Category,
    pub metric: String,
    pub verdict: Verdict,
    pub n_snapshots_seen: usize,
    pub n_real: usize,
    pub n_carried_or_degraded: usize,
    pub n_distinct_values: usize,
    pub n_steps: usize,
    pub n_transition_events: usize,
    pub eps: f64,
    pub feature_names: Vec<String>,
    pub train_len: usize,
    pub test_len: usize,
    /// Accuracy of "always predict Flat" == persistence, on the held-out tail.
    pub persistence_accuracy: Option<f64>,
    pub gbdt_accuracy: Option<f64>,
    pub persistence_correct: Option<usize>,
    pub gbdt_correct: Option<usize>,
    /// `Some(true)` only if GBDT strictly beat persistence on held-out accuracy.
    pub gbdt_beats_persistence: Option<bool>,
    pub notes: Vec<String>,
}

fn count_distinct(values: &[f64]) -> usize {
    let mut seen: Vec<f64> = Vec::new();
    for &v in values {
        if !seen.iter().any(|&s| (s - v).abs() < 1e-12) {
            seen.push(v);
        }
    }
    seen.len()
}

/// Build the feature matrix and direction labels from a chronological series.
///
/// Sample `t` (for `t >= 2`) predicts the direction of `v[t]` vs `v[t-1]`
/// using only information available at `t-1`.
fn build_samples(obs: &[Obs], eps: f64) -> (Array2<f64>, Array1<usize>) {
    let n = obs.len();
    let mut rows: Vec<[f64; 6]> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for t in 2..n {
        let last = obs[t - 1].value;
        let prev = obs[t - 2].value;
        let last_delta = last - prev;

        // Flat-run length and steps-since-change over history up to t-1.
        let mut flat_run = 0.0f64;
        for k in (1..t).rev() {
            if (obs[k].value - obs[k - 1].value).abs() <= eps {
                flat_run += 1.0;
            } else {
                break;
            }
        }
        let mut steps_since_change = (t - 1) as f64;
        for k in (1..t).rev() {
            if (obs[k].value - obs[k - 1].value).abs() > eps {
                steps_since_change = (t - 1 - k) as f64;
                break;
            }
        }

        let sign = if last_delta > eps {
            1.0
        } else if last_delta < -eps {
            -1.0
        } else {
            0.0
        };

        rows.push([
            last,
            last_delta,
            flat_run,
            steps_since_change,
            sign,
            if obs[t - 1].real { 1.0 } else { 0.0 },
        ]);
        labels.push(Direction::from_delta(obs[t].value - obs[t - 1].value, eps).label());
    }

    let m = rows.len();
    let mut x = Array2::<f64>::zeros((m, 6));
    for (i, r) in rows.iter().enumerate() {
        for (j, &val) in r.iter().enumerate() {
            x[[i, j]] = val;
        }
    }
    (x, Array1::from(labels))
}

/// Run the GBDT-vs-persistence comparison on one chronological series.
///
/// `obs` MUST be sorted by date ascending. Returns committed evidence either
/// way; consult [`ForecastEval::verdict`].
pub fn evaluate(category: Category, obs: &[Obs], cfg: &EvalConfig) -> ForecastEval {
    let metric = category.metric_label().to_string();
    let feature_names = FEATURE_NAMES.iter().map(|s| s.to_string()).collect();

    let n_snapshots_seen = obs.len();
    let n_real = obs.iter().filter(|o| o.real).count();
    let n_carried_or_degraded = n_snapshots_seen - n_real;
    let values: Vec<f64> = obs.iter().map(|o| o.value).collect();
    let n_distinct_values = count_distinct(&values);

    let (vmin, vmax) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let eps = if vmax > vmin {
        (cfg.eps_frac * (vmax - vmin)).max(1e-9)
    } else {
        1e-9
    };

    let n_steps = n_snapshots_seen.saturating_sub(1);
    let n_transition_events = (1..obs.len())
        .filter(|&t| (obs[t].value - obs[t - 1].value).abs() > eps)
        .count();

    let mut notes: Vec<String> = Vec::new();

    let (x, y) = build_samples(obs, eps);
    let m = x.nrows();
    let train_len = ((m as f64) * cfg.train_frac).floor() as usize;
    let test_len = m.saturating_sub(train_len);

    // Honest-pause guardrail — collect only the conditions that actually fail.
    let mut fails: Vec<String> = Vec::new();
    if n_real < cfg.min_real {
        fails.push(format!("n_real={n_real} < {}", cfg.min_real));
    }
    if n_transition_events < cfg.min_transitions {
        fails.push(format!(
            "transitions={n_transition_events} < {}",
            cfg.min_transitions
        ));
    }
    if test_len < cfg.min_test {
        fails.push(format!("held_out={test_len} < {}", cfg.min_test));
    }
    if train_len == 0 {
        fails.push("train_len=0".to_string());
    }
    if !fails.is_empty() {
        notes.push(format!(
            "PAUSED ({}). Series too thin/degenerate to train/evaluate a learned transition \
             model against persistence.",
            fails.join("; ")
        ));
        return ForecastEval {
            category,
            metric,
            verdict: Verdict::PausedInsufficientData,
            n_snapshots_seen,
            n_real,
            n_carried_or_degraded,
            n_distinct_values,
            n_steps,
            n_transition_events,
            eps,
            feature_names,
            train_len,
            test_len,
            persistence_accuracy: None,
            gbdt_accuracy: None,
            persistence_correct: None,
            gbdt_correct: None,
            gbdt_beats_persistence: None,
            notes,
        };
    }

    let x_train = x.slice(ndarray::s![..train_len, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_len]).to_owned();
    let x_test = x.slice(ndarray::s![train_len.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![train_len..]).to_owned();

    let mut gbdt = GradientBoostedClassifier::new(cfg.n_estimators, cfg.learning_rate)
        .with_min_samples_leaf(cfg.min_samples_leaf);
    gbdt.fit(&x_train, &y_train);
    let gbdt_pred = gbdt.predict(&x_test);

    let flat = Direction::Flat.label();
    let mut gbdt_correct = 0usize;
    let mut persistence_correct = 0usize;
    for i in 0..test_len {
        if gbdt_pred[i] == y_test[i] {
            gbdt_correct += 1;
        }
        if flat == y_test[i] {
            persistence_correct += 1;
        }
    }
    let gbdt_acc = gbdt_correct as f64 / test_len as f64;
    let persistence_acc = persistence_correct as f64 / test_len as f64;

    notes.push(format!(
        "eps={eps:.6} ({}% of observed range). Persistence == 'always predict Flat' (last value carries forward).",
        (cfg.eps_frac * 100.0) as u64
    ));
    if gbdt_acc <= persistence_acc {
        notes.push(
            "GBDT did NOT beat persistence on held-out snapshots. Honest-pause: J4 stays parked \
             until a series with more real transitions (or per-change commit features) is available."
                .to_string(),
        );
    }

    ForecastEval {
        category,
        metric,
        verdict: Verdict::Ran,
        n_snapshots_seen,
        n_real,
        n_carried_or_degraded,
        n_distinct_values,
        n_steps,
        n_transition_events,
        eps,
        feature_names,
        train_len,
        test_len,
        persistence_accuracy: Some(persistence_acc),
        gbdt_accuracy: Some(gbdt_acc),
        persistence_correct: Some(persistence_correct),
        gbdt_correct: Some(gbdt_correct),
        gbdt_beats_persistence: Some(gbdt_acc > persistence_acc),
        notes,
    }
}

// ─────────────────────── ga snapshot ingestion ──────────────────────────────

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn get_bool(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}

/// Extract the scalar metric and a `real` (fresh-measurement) flag from one
/// snapshot JSON, per the category's producer schema. Returns `None` when the
/// snapshot carries no usable value (truly missing data — skipped).
pub fn extract_value(category: Category, v: &Value) -> Option<(f64, bool)> {
    match category {
        Category::ChatbotQa => {
            if let Some(p) = get_f64(v, "pass_pct") {
                return Some((p, true));
            }
            // Degraded envelope: carried last-known-good is a continuity hint,
            // not a fresh measurement.
            if get_bool(v, "degraded").unwrap_or(false) {
                if let Some(carried) = get_f64(v, "last_known_good_pass_pct") {
                    return Some((carried, false));
                }
            }
            None
        }
        Category::Embeddings => {
            let value = get_f64(v, "metric_value").or_else(|| {
                v.get("leak_detection")
                    .and_then(|ld| ld.get("full_classifier_accuracy"))
                    .and_then(Value::as_f64)
            })?;
            let carried = get_bool(v, "carried_forward").unwrap_or(false);
            let degraded = get_bool(v, "degraded").unwrap_or(false);
            Some((value, !(carried || degraded)))
        }
        Category::VoicingAnalysis => {
            // Newer snapshots carry a top-level `metric_value`; older raw
            // analyzer dumps expose ForteCoverage.Pct (0..100).
            if let Some(mv) = get_f64(v, "metric_value") {
                return Some((mv, true));
            }
            v.get("ForteCoverage")
                .and_then(|fc| fc.get("Pct"))
                .and_then(Value::as_f64)
                .map(|pct| (pct / 100.0, true))
        }
    }
}

/// Parse a `YYYY-MM-DD` prefix from a snapshot filename stem.
pub fn date_from_filename(stem: &str) -> Option<NaiveDate> {
    if stem.len() < 10 {
        return None;
    }
    NaiveDate::parse_from_str(&stem[..10], "%Y-%m-%d").ok()
}

/// Load one category's chronological series from `<root>/<category>/*.json`.
///
/// `root` is the ga `state/quality/` directory. Files whose name does not start
/// with `YYYY-MM-DD` are skipped. The returned series is sorted by date.
pub fn load_series(root: &std::path::Path, category: Category) -> std::io::Result<Vec<Obs>> {
    let dir = root.join(category.dir_name());
    let mut obs: Vec<Obs> = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        let date = match date_from_filename(stem) {
            Some(d) => d,
            None => continue,
        };
        let text = std::fs::read_to_string(&path)?;
        let json: Value = match serde_json::from_str(&text) {
            Ok(j) => j,
            Err(_) => continue,
        };
        if let Some((value, real)) = extract_value(category, &json) {
            obs.push(Obs { date, value, real });
        }
    }
    obs.sort_by_key(|o| o.date);
    Ok(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(n: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap() + chrono::Duration::days(n as i64)
    }

    fn obs_from(values: &[(f64, bool)]) -> Vec<Obs> {
        values
            .iter()
            .enumerate()
            .map(|(i, &(value, real))| Obs {
                date: day(i as u32),
                value,
                real,
            })
            .collect()
    }

    #[test]
    fn thin_series_triggers_honest_pause() {
        // Mirrors the real chatbot-qa reality: only two fresh measurements.
        let obs = obs_from(&[(75.0, true), (7.69, true)]);
        let eval = evaluate(Category::ChatbotQa, &obs, &EvalConfig::default());
        assert_eq!(eval.verdict, Verdict::PausedInsufficientData);
        assert_eq!(eval.n_real, 2);
        assert!(eval.gbdt_accuracy.is_none());
        assert!(eval.notes.iter().any(|n| n.contains("PAUSED")));
    }

    #[test]
    fn piecewise_constant_series_runs_but_persistence_holds() {
        // Long flat run with a couple of jumps — the embeddings shape. GBDT
        // should not strictly beat "always predict Flat".
        let mut v: Vec<(f64, bool)> = Vec::new();
        for _ in 0..12 {
            v.push((0.75, true));
        }
        v.push((0.83, true));
        for _ in 0..12 {
            v.push((0.83, false));
        }
        v.push((0.90, true));
        for _ in 0..6 {
            v.push((0.90, false));
        }
        let obs = obs_from(&v);
        let cfg = EvalConfig {
            min_transitions: 2,
            ..EvalConfig::default()
        };
        let eval = evaluate(Category::Embeddings, &obs, &cfg);
        assert_eq!(eval.verdict, Verdict::Ran);
        let gbdt = eval.gbdt_accuracy.unwrap();
        let pers = eval.persistence_accuracy.unwrap();
        // Persistence is near-optimal on a piecewise-constant series.
        assert!(pers >= gbdt, "persistence {pers} should hold vs gbdt {gbdt}");
        assert_eq!(eval.gbdt_beats_persistence, Some(false));
    }

    #[test]
    fn features_and_labels_align() {
        let obs = obs_from(&[(1.0, true), (2.0, true), (3.0, true), (2.0, true)]);
        let (x, y) = build_samples(&obs, 1e-9);
        // t in {2,3} -> 2 samples, 6 features each.
        assert_eq!(x.nrows(), 2);
        assert_eq!(x.ncols(), 6);
        assert_eq!(y.len(), 2);
        // Sample for t=2 predicts dir(v[2]-v[1]) = Up.
        assert_eq!(y[0], Direction::Up.label());
        // Sample for t=3 predicts dir(v[3]-v[2]) = Down.
        assert_eq!(y[1], Direction::Down.label());
        // last_value feature of sample t=2 is v[1] = 2.0.
        assert!((x[[0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn extract_chatbot_qa_variants() {
        let fresh = serde_json::json!({ "pass_pct": 42.0 });
        assert_eq!(
            extract_value(Category::ChatbotQa, &fresh),
            Some((42.0, true))
        );
        let degraded = serde_json::json!({
            "pass_pct": null, "degraded": true, "last_known_good_pass_pct": 60.0
        });
        assert_eq!(
            extract_value(Category::ChatbotQa, &degraded),
            Some((60.0, false))
        );
        let missing = serde_json::json!({ "pass_pct": null, "note": "backend down" });
        assert_eq!(extract_value(Category::ChatbotQa, &missing), None);
    }
}
