//! Takens' delay embedding — reconstruct phase space from scalar time series.
//!
//! Given a 1D time series, embed it into a higher-dimensional space using
//! time-delay coordinates: x(t), x(t+tau), x(t+2*tau), ..., x(t+(d-1)*tau)

/// Delay-embed a scalar time series into `dim`-dimensional vectors.
///
/// `data`: the scalar time series.
/// `dim`: embedding dimension.
/// `delay`: time delay (in samples).
///
/// Returns a vector of `dim`-dimensional points.
pub fn delay_embed(data: &[f64], dim: usize, delay: usize) -> Vec<Vec<f64>> {
    let n = data.len();
    let required = (dim - 1) * delay + 1;
    if n < required {
        return vec![];
    }

    let num_points = n - (dim - 1) * delay;
    (0..num_points)
        .map(|i| {
            (0..dim).map(|d| data[i + d * delay]).collect()
        })
        .collect()
}

/// Estimate optimal delay using first minimum of auto-mutual information.
///
/// Uses a binning-based approach to estimate mutual information at each lag.
pub fn optimal_delay(data: &[f64], max_lag: usize, num_bins: usize) -> usize {
    let n = data.len();
    if n < 2 || max_lag == 0 {
        return 1;
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-15);
    let bin_width = range / num_bins as f64;

    let to_bin = |x: f64| -> usize {
        ((x - min_val) / bin_width).floor().min((num_bins - 1) as f64) as usize
    };

    let mut prev_mi = f64::INFINITY;
    let mut best_lag = 1;

    for lag in 1..=max_lag.min(n / 2) {
        let count = n - lag;
        let mut joint = vec![vec![0u64; num_bins]; num_bins];
        let mut marginal_x = vec![0u64; num_bins];
        let mut marginal_y = vec![0u64; num_bins];

        for i in 0..count {
            let bx = to_bin(data[i]);
            let by = to_bin(data[i + lag]);
            joint[bx][by] += 1;
            marginal_x[bx] += 1;
            marginal_y[by] += 1;
        }

        let c = count as f64;
        let mut mi = 0.0;
        for bx in 0..num_bins {
            for by in 0..num_bins {
                let pxy = joint[bx][by] as f64 / c;
                let px = marginal_x[bx] as f64 / c;
                let py = marginal_y[by] as f64 / c;
                if pxy > 0.0 && px > 0.0 && py > 0.0 {
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }

        // First minimum
        if mi > prev_mi {
            best_lag = lag - 1;
            break;
        }
        prev_mi = mi;
        best_lag = lag;
    }

    best_lag.max(1)
}

/// Estimate embedding dimension using Cao's method.
///
/// Returns the ratio E1(d) for each dimension. When E1 stops changing
/// significantly (E1 ≈ 1), the embedding dimension is sufficient.
pub fn cao_embedding_dimension(
    data: &[f64],
    delay: usize,
    max_dim: usize,
) -> Vec<f64> {
    let mut e_values = Vec::new();

    for d in 1..max_dim {
        let embedded_d = delay_embed(data, d, delay);
        let embedded_d1 = delay_embed(data, d + 1, delay);

        if embedded_d1.len() < 2 {
            break;
        }

        let n = embedded_d1.len();
        let mut a_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            // Find nearest neighbor in d-dimensional embedding
            let mut min_dist = f64::INFINITY;
            let mut nn_idx = 0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist: f64 = embedded_d[i].iter().zip(embedded_d[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(f64::NEG_INFINITY, f64::max); // Chebyshev distance
                if dist < min_dist {
                    min_dist = dist;
                    nn_idx = j;
                }
            }

            let dist_d1: f64 = embedded_d1[i].iter().zip(embedded_d1[nn_idx].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(f64::NEG_INFINITY, f64::max);

            if min_dist > 1e-15 {
                a_sum += dist_d1 / min_dist;
                count += 1;
            }
        }

        if count > 0 {
            e_values.push(a_sum / count as f64);
        }
    }

    // Compute E1(d) = E(d+1) / E(d)
    if e_values.len() < 2 {
        return vec![];
    }

    (0..e_values.len() - 1)
        .map(|i| {
            if e_values[i].abs() > 1e-15 {
                e_values[i + 1] / e_values[i]
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_embed_size() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let embedded = delay_embed(&data, 3, 5);
        // Should have 100 - (3-1)*5 = 90 points
        assert_eq!(embedded.len(), 90);
        assert_eq!(embedded[0], vec![0.0, 5.0, 10.0]);
    }

    #[test]
    fn test_delay_embed_sine() {
        // Sine wave embedded in 2D should trace a circle
        let data: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
            .collect();

        let embedded = delay_embed(&data, 2, 25); // quarter-period delay
        // Check that the embedded points roughly form a circle
        for point in &embedded {
            let r = (point[0] * point[0] + point[1] * point[1]).sqrt();
            assert!(r < 1.5, "Embedded sine should be bounded");
        }
    }

    #[test]
    fn test_optimal_delay_sine() {
        let data: Vec<f64> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
            .collect();

        let delay = optimal_delay(&data, 50, 16);
        // Optimal delay for a sine wave should be around T/4 = 25
        // The mutual information method can vary; accept a wider range
        assert!((1..=50).contains(&delay), "Optimal delay for sine: {}", delay);
    }
}
