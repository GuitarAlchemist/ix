//! File-level aggregation of per-function [`CodeMetrics`].
//!
//! Produces summary statistics (mean, p90, max) plus inequality measures
//! (Gini, Shannon entropy) over a distribution of complexity values. These
//! aggregates feed Layer 5 risk-delta gates — an entity is riskier when its
//! complexity is concentrated in a few outlier functions (high Gini) rather
//! than distributed evenly.

use crate::metrics::CodeMetrics;
use serde::{Deserialize, Serialize};

/// File-level summary of per-function metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Entity identifier (usually a file path).
    pub entity: String,
    /// Number of functions considered.
    pub n_functions: usize,
    /// Arithmetic mean of cyclomatic complexity.
    pub mean_cyclomatic: f64,
    /// 90th percentile of cyclomatic complexity.
    pub p90_cyclomatic: f64,
    /// Maximum cyclomatic complexity.
    pub max_cyclomatic: f64,
    /// Count of functions with cyclomatic >= 15.
    pub n_high_risk: usize,
    /// Count of functions with cyclomatic in [8, 15).
    pub n_medium_risk: usize,
    /// Count of functions with cyclomatic < 8.
    pub n_low_risk: usize,
    /// Gini coefficient of the cyclomatic distribution (0 = equal, 1 = concentrated).
    pub gini_complexity: f64,
    /// Normalized Shannon entropy of the cyclomatic distribution (0 = spiked, 1 = uniform).
    pub entropy_complexity: f64,
}

/// Aggregate a slice of function-level metrics into a file-level summary.
///
/// Returns a zeroed summary when `functions` is empty.
pub fn aggregate_file_metrics(functions: &[CodeMetrics]) -> AggregatedMetrics {
    let n = functions.len();
    if n == 0 {
        return AggregatedMetrics {
            entity: String::new(),
            n_functions: 0,
            mean_cyclomatic: 0.0,
            p90_cyclomatic: 0.0,
            max_cyclomatic: 0.0,
            n_high_risk: 0,
            n_medium_risk: 0,
            n_low_risk: 0,
            gini_complexity: 0.0,
            entropy_complexity: 0.0,
        };
    }

    let mut cyclos: Vec<f64> = functions.iter().map(|f| f.cyclomatic).collect();
    let sum: f64 = cyclos.iter().sum();
    let mean = sum / n as f64;
    let max = cyclos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    cyclos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Linear-interpolation 90th percentile.
    let p90 = percentile(&cyclos, 0.90);

    let n_high = functions.iter().filter(|f| f.cyclomatic >= 15.0).count();
    let n_med = functions
        .iter()
        .filter(|f| f.cyclomatic >= 8.0 && f.cyclomatic < 15.0)
        .count();
    let n_low = n - n_high - n_med;

    let gini = gini_coefficient(&cyclos);
    let entropy = shannon_entropy(&cyclos, 10);

    AggregatedMetrics {
        entity: String::new(),
        n_functions: n,
        mean_cyclomatic: mean,
        p90_cyclomatic: p90,
        max_cyclomatic: max,
        n_high_risk: n_high,
        n_medium_risk: n_med,
        n_low_risk: n_low,
        gini_complexity: gini,
        entropy_complexity: entropy,
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = q * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Standard Gini coefficient — measures inequality in a distribution.
///
/// Returns 0.0 when all values are equal (perfect equality) and approaches
/// 1.0 when one value dominates. Negative values are clamped to zero before
/// computation. Returns 0.0 for empty input or zero-sum distributions.
pub fn gini_coefficient(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.iter().map(|x| x.max(0.0)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len() as f64;
    let sum: f64 = v.iter().sum();
    if sum <= f64::EPSILON {
        return 0.0;
    }
    let mut weighted = 0.0;
    for (i, x) in v.iter().enumerate() {
        weighted += (i as f64 + 1.0) * x;
    }
    (2.0 * weighted) / (n * sum) - (n + 1.0) / n
}

/// Normalized Shannon entropy of a histogram over `values`.
///
/// Bins `values` into `bins` equal-width buckets between `min` and `max`,
/// converts counts to a probability distribution, then returns the Shannon
/// entropy divided by `log2(bins)`. Output is in [0, 1]: 0 means all mass
/// sits in one bin, 1 means uniform. Returns 0.0 when `values` is empty,
/// `bins < 2`, or the range is degenerate.
pub fn shannon_entropy(values: &[f64], bins: usize) -> f64 {
    if values.is_empty() || bins < 2 {
        return 0.0;
    }
    let min = values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() <= f64::EPSILON {
        return 0.0;
    }
    let mut counts = vec![0usize; bins];
    let width = (max - min) / bins as f64;
    for &x in values {
        let mut idx = ((x - min) / width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx >= bins as isize {
            idx = bins as isize - 1;
        }
        counts[idx as usize] += 1;
    }
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    let mut h = 0.0;
    for &c in &counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total;
        h -= p * p.log2();
    }
    h / (bins as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(name: &str, cyc: f64) -> CodeMetrics {
        CodeMetrics {
            name: name.to_string(),
            start_line: 1,
            end_line: 10,
            cyclomatic: cyc,
            cognitive: 0.0,
            n_exits: 1.0,
            n_args: 0.0,
            sloc: 1.0,
            ploc: 1.0,
            lloc: 1.0,
            cloc: 0.0,
            blank: 0.0,
            h_u_ops: 0.0,
            h_u_opnds: 0.0,
            h_total_ops: 0.0,
            h_total_opnds: 0.0,
            h_vocabulary: 0.0,
            h_length: 0.0,
            h_volume: 0.0,
            h_difficulty: 0.0,
            h_effort: 0.0,
            h_bugs: 0.0,
            maintainability_index: 100.0,
        }
    }

    #[test]
    fn test_gini_all_equal() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let g = gini_coefficient(&values);
        assert!(g.abs() < 1e-9, "gini for equal values should be 0, got {}", g);
    }

    #[test]
    fn test_gini_one_dominant() {
        let values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0];
        let g = gini_coefficient(&values);
        assert!(g > 0.85, "gini for outlier should be close to 1, got {}", g);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution should have entropy near 1.0.
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let h = shannon_entropy(&values, 10);
        assert!(h > 0.95, "uniform entropy should be near 1.0, got {}", h);
    }

    #[test]
    fn test_shannon_entropy_spike() {
        let values = vec![5.0; 50];
        let h = shannon_entropy(&values, 10);
        assert_eq!(h, 0.0, "degenerate range should yield 0 entropy");
    }

    #[test]
    fn test_aggregate_preserves_distribution() {
        // Skewed distribution: mean, p90, max must all differ.
        let funcs = vec![
            cm("a", 1.0),
            cm("b", 2.0),
            cm("c", 2.0),
            cm("d", 3.0),
            cm("e", 3.0),
            cm("f", 4.0),
            cm("g", 5.0),
            cm("h", 5.0),
            cm("i", 10.0),
            cm("j", 50.0),
        ];
        let agg = aggregate_file_metrics(&funcs);
        assert_eq!(agg.n_functions, 10);
        assert!(agg.max_cyclomatic >= agg.p90_cyclomatic);
        assert!(agg.p90_cyclomatic >= agg.mean_cyclomatic);
        assert!(
            agg.mean_cyclomatic < agg.p90_cyclomatic,
            "mean {} should be less than p90 {} for skewed data",
            agg.mean_cyclomatic,
            agg.p90_cyclomatic
        );
        assert!(
            agg.p90_cyclomatic < agg.max_cyclomatic,
            "p90 {} should be less than max {}",
            agg.p90_cyclomatic,
            agg.max_cyclomatic
        );
        assert_eq!(agg.max_cyclomatic, 50.0);
        // cyclomatic >= 15 → [50.0] = 1
        assert_eq!(agg.n_high_risk, 1);
    }

    #[test]
    fn test_aggregate_risk_buckets() {
        let funcs = vec![
            cm("low1", 3.0),
            cm("low2", 5.0),
            cm("med1", 10.0),
            cm("med2", 12.0),
            cm("high1", 20.0),
        ];
        let agg = aggregate_file_metrics(&funcs);
        assert_eq!(agg.n_low_risk, 2);
        assert_eq!(agg.n_medium_risk, 2);
        assert_eq!(agg.n_high_risk, 1);
    }

    #[test]
    fn test_aggregate_empty() {
        let agg = aggregate_file_metrics(&[]);
        assert_eq!(agg.n_functions, 0);
        assert_eq!(agg.mean_cyclomatic, 0.0);
    }
}
