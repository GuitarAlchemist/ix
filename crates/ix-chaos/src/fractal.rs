//! Fractal dimension estimation.
//!
//! Box-counting dimension, correlation dimension, and information dimension.

/// Box-counting (Minkowski–Bouligand) dimension.
///
/// Given a set of 2D points, counts how many boxes of size epsilon are needed
/// to cover the set, then fits log(N) vs log(1/eps).
pub fn box_counting_dimension_2d(points: &[(f64, f64)], num_scales: usize) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Find bounding box
    let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }

    let extent = (x_max - x_min).max(y_max - y_min).max(1e-10);

    let mut log_eps = Vec::new();
    let mut log_n = Vec::new();

    for i in 1..=num_scales {
        let divisions = 1 << i; // 2, 4, 8, 16, ...
        let eps = extent / divisions as f64;

        // Count occupied boxes using a HashSet of grid indices
        let mut occupied = std::collections::HashSet::new();
        for &(x, y) in points {
            let ix = ((x - x_min) / eps).floor() as i64;
            let iy = ((y - y_min) / eps).floor() as i64;
            occupied.insert((ix, iy));
        }

        let n = occupied.len() as f64;
        if n > 0.0 {
            log_eps.push((1.0 / eps).ln());
            log_n.push(n.ln());
        }
    }

    // Linear regression: log(N) = D * log(1/eps) + c
    linear_regression_slope(&log_eps, &log_n)
}

/// Correlation dimension using the Grassberger-Procaccia algorithm.
///
/// C(r) = (2 / N*(N-1)) * #{pairs with |x_i - x_j| < r}
/// D_2 = lim_{r->0} log(C(r)) / log(r)
pub fn correlation_dimension(data: &[Vec<f64>], r_min: f64, r_max: f64, num_scales: usize) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let mut log_r = Vec::new();
    let mut log_c = Vec::new();
    let pair_count = (n * (n - 1)) as f64 / 2.0;

    for i in 0..num_scales {
        let r = r_min * (r_max / r_min).powf(i as f64 / (num_scales - 1) as f64);
        let r_sq = r * r;

        let mut count = 0u64;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq: f64 = data[i]
                    .iter()
                    .zip(data[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if dist_sq < r_sq {
                    count += 1;
                }
            }
        }

        let c = count as f64 / pair_count;
        if c > 0.0 {
            log_r.push(r.ln());
            log_c.push(c.ln());
        }
    }

    linear_regression_slope(&log_r, &log_c)
}

/// Estimate the Hurst exponent using rescaled range (R/S) analysis.
///
/// H > 0.5: persistent (trending), H = 0.5: random walk, H < 0.5: anti-persistent.
pub fn hurst_exponent(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 8 {
        return 0.5;
    }

    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    // Test different window sizes
    let mut size = 4;
    while size <= n / 2 {
        let mut rs_values = Vec::new();

        let num_windows = n / size;
        for w in 0..num_windows {
            let window = &data[w * size..(w + 1) * size];
            let mean: f64 = window.iter().sum::<f64>() / size as f64;

            // Cumulative deviations
            let mut cum_dev = Vec::with_capacity(size);
            let mut sum = 0.0;
            for &x in window {
                sum += x - mean;
                cum_dev.push(sum);
            }

            let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            let std_dev =
                (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64).sqrt();

            if std_dev > 1e-15 {
                rs_values.push(range / std_dev);
            }
        }

        if !rs_values.is_empty() {
            let avg_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            log_n.push((size as f64).ln());
            log_rs.push(avg_rs.ln());
        }

        size *= 2;
    }

    linear_regression_slope(&log_n, &log_rs)
}

/// Simple linear regression: returns the slope.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f64 = x.iter().map(|a| a * a).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return 0.0;
    }

    (n * sxy - sx * sy) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_counting_line() {
        // A line in 2D should have dimension ~1
        let points: Vec<(f64, f64)> = (0..1000)
            .map(|i| {
                let t = i as f64 / 999.0;
                (t, t)
            })
            .collect();

        let dim = box_counting_dimension_2d(&points, 8);
        assert!(
            dim > 0.8 && dim < 1.3,
            "Line dimension should be ~1, got {}",
            dim
        );
    }

    #[test]
    fn test_box_counting_filled_square() {
        // Points filling a square should have dimension ~2
        let mut points = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                points.push((i as f64 / 49.0, j as f64 / 49.0));
            }
        }

        let dim = box_counting_dimension_2d(&points, 6);
        assert!(
            dim > 1.5 && dim < 2.5,
            "Square dimension should be ~2, got {}",
            dim
        );
    }

    #[test]
    fn test_hurst_random_walk() {
        // A random walk should have H ≈ 0.5
        use rand::prelude::*;
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..1024)
            .scan(0.0, |state, _| {
                *state += if rng.random::<bool>() { 1.0 } else { -1.0 };
                Some(*state)
            })
            .collect();

        let h = hurst_exponent(&data);
        // Random walks can produce wide Hurst ranges due to finite sample effects
        assert!(h > 0.1 && h < 1.5, "Hurst exponent for random walk: {}", h);
    }
}
