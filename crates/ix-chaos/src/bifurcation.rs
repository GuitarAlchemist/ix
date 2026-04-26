//! Bifurcation analysis — how system behavior changes with parameters.
//!
//! Bifurcation diagrams, period detection, and bifurcation point estimation.

/// A point on a bifurcation diagram: (parameter_value, attractor_value).
#[derive(Debug, Clone, Copy)]
pub struct BifurcationPoint {
    pub parameter: f64,
    pub value: f64,
}

/// Generate a bifurcation diagram for a 1D iterated map.
///
/// For each parameter value in [param_min, param_max], iterates the map
/// and records the attractor values after a transient.
///
/// `map`: f(x, r) where r is the parameter.
/// `x0`: initial condition.
/// `param_steps`: number of parameter values to sample.
/// `transient`: iterations to discard.
/// `record`: iterations to record per parameter value.
pub fn bifurcation_diagram<F>(
    map: F,
    x0: f64,
    param_min: f64,
    param_max: f64,
    param_steps: usize,
    transient: usize,
    record: usize,
) -> Vec<BifurcationPoint>
where
    F: Fn(f64, f64) -> f64,
{
    let mut points = Vec::with_capacity(param_steps * record);

    for step in 0..param_steps {
        let r = param_min + (param_max - param_min) * step as f64 / (param_steps - 1) as f64;
        let mut x = x0;

        // Transient
        for _ in 0..transient {
            x = map(x, r);
        }

        // Record attractor
        for _ in 0..record {
            x = map(x, r);
            points.push(BifurcationPoint {
                parameter: r,
                value: x,
            });
        }
    }

    points
}

/// Detect the period of an orbit for a 1D map at a given parameter.
///
/// Returns None if the orbit appears chaotic (period > max_period).
pub fn detect_period<F>(
    map: F,
    x0: f64,
    param: f64,
    transient: usize,
    max_period: usize,
    tolerance: f64,
) -> Option<usize>
where
    F: Fn(f64, f64) -> f64,
{
    let mut x = x0;
    for _ in 0..transient {
        x = map(x, param);
    }

    let anchor = x;

    for period in 1..=max_period {
        x = map(x, param);
        if (x - anchor).abs() < tolerance {
            return Some(period);
        }
    }

    None
}

/// Find approximate bifurcation points using period doubling detection.
///
/// Scans parameter range and detects where the period changes.
pub fn find_period_doublings<F>(
    map: F,
    x0: f64,
    param_min: f64,
    param_max: f64,
    scan_steps: usize,
    transient: usize,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let mut bifurcations = Vec::new();
    let mut prev_period = None;

    for step in 0..scan_steps {
        let r = param_min + (param_max - param_min) * step as f64 / (scan_steps - 1) as f64;
        let period = detect_period(map, x0, r, transient, 64, 1e-8);

        if let (Some(prev), Some(curr)) = (prev_period, period) {
            if curr == 2 * prev {
                bifurcations.push(r);
            }
        }
        prev_period = period;
    }

    bifurcations
}

/// Estimate Feigenbaum's constant delta from a sequence of bifurcation points.
///
/// delta ≈ 4.669... for period-doubling cascades.
pub fn feigenbaum_delta(bifurcation_points: &[f64]) -> Vec<f64> {
    if bifurcation_points.len() < 3 {
        return vec![];
    }

    let mut deltas = Vec::new();
    for i in 0..bifurcation_points.len() - 2 {
        let d1 = bifurcation_points[i + 1] - bifurcation_points[i];
        let d2 = bifurcation_points[i + 2] - bifurcation_points[i + 1];
        if d2.abs() > 1e-15 {
            deltas.push(d1 / d2);
        }
    }
    deltas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_bifurcation() {
        let map = |x: f64, r: f64| r * x * (1.0 - x);
        let diagram = bifurcation_diagram(map, 0.5, 2.5, 4.0, 100, 500, 50);
        assert!(!diagram.is_empty());

        // At r=3.0 should be near fixed point ~0.666
        let near_3: Vec<_> = diagram
            .iter()
            .filter(|p| (p.parameter - 3.0).abs() < 0.02)
            .collect();
        assert!(!near_3.is_empty());
    }

    #[test]
    fn test_period_detection() {
        let map = |x: f64, r: f64| r * x * (1.0 - x);

        // r=2.5: period-1 (fixed point)
        assert_eq!(detect_period(map, 0.5, 2.5, 1000, 64, 1e-6), Some(1));

        // r=3.2: period-2
        assert_eq!(detect_period(map, 0.5, 3.2, 1000, 64, 1e-6), Some(2));
    }

    #[test]
    fn test_feigenbaum() {
        // Known logistic map period-doubling bifurcation points
        let bif_points = vec![3.0, 3.44949, 3.54409, 3.5644];
        let deltas = feigenbaum_delta(&bif_points);
        // First delta should approach 4.669...
        if !deltas.is_empty() {
            assert!(
                deltas[0] > 3.0 && deltas[0] < 6.0,
                "Feigenbaum delta estimate: {}",
                deltas[0]
            );
        }
    }
}
