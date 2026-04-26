//! Poincaré sections and return maps.
//!
//! A Poincaré section intersects a trajectory with a hyperplane,
//! reducing continuous dynamics to a discrete map.

use crate::attractors::State3D;

/// A crossing event through a Poincaré section.
#[derive(Debug, Clone, Copy)]
pub struct Crossing {
    /// Time of crossing (interpolated).
    pub time: f64,
    /// State at crossing (linearly interpolated).
    pub state: State3D,
    /// Crossing direction: +1 or -1.
    pub direction: i8,
}

/// Extract Poincaré section crossings from a 3D trajectory.
///
/// The section is defined by `axis_value` along `axis` (0=x, 1=y, 2=z).
/// Only positive-direction crossings are kept (if `positive_only` is true).
pub fn poincare_section(
    trajectory: &[State3D],
    dt: f64,
    axis: usize,
    axis_value: f64,
    positive_only: bool,
) -> Vec<Crossing> {
    let mut crossings = Vec::new();

    let get_coord = |s: &State3D| match axis {
        0 => s.x,
        1 => s.y,
        _ => s.z,
    };

    for i in 0..trajectory.len() - 1 {
        let v0 = get_coord(&trajectory[i]) - axis_value;
        let v1 = get_coord(&trajectory[i + 1]) - axis_value;

        if v0 * v1 < 0.0 {
            // Crossing detected — linear interpolation
            let frac = v0.abs() / (v0.abs() + v1.abs());
            let s0 = &trajectory[i];
            let s1 = &trajectory[i + 1];

            let crossing = Crossing {
                time: (i as f64 + frac) * dt,
                state: State3D::new(
                    s0.x + frac * (s1.x - s0.x),
                    s0.y + frac * (s1.y - s0.y),
                    s0.z + frac * (s1.z - s0.z),
                ),
                direction: if v1 > v0 { 1 } else { -1 },
            };

            if !positive_only || crossing.direction == 1 {
                crossings.push(crossing);
            }
        }
    }

    crossings
}

/// Build a 1D return map from Poincaré section crossings.
///
/// Returns pairs (x_n, x_{n+1}) where x is the `map_axis` coordinate.
pub fn return_map(crossings: &[Crossing], map_axis: usize) -> Vec<(f64, f64)> {
    if crossings.len() < 2 {
        return vec![];
    }

    let get_coord = |c: &Crossing| match map_axis {
        0 => c.state.x,
        1 => c.state.y,
        _ => c.state.z,
    };

    crossings
        .windows(2)
        .map(|w| (get_coord(&w[0]), get_coord(&w[1])))
        .collect()
}

/// Compute mean return time from crossings.
pub fn mean_return_time(crossings: &[Crossing]) -> Option<f64> {
    if crossings.len() < 2 {
        return None;
    }

    let total_time = crossings.last().unwrap().time - crossings.first().unwrap().time;
    Some(total_time / (crossings.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attractors::{lorenz, LorenzParams};

    #[test]
    fn test_lorenz_poincare() {
        let params = LorenzParams::default();
        let traj = lorenz(State3D::new(1.0, 1.0, 1.0), &params, 0.005, 100_000);

        // Poincaré section at z = 27 (near the top of the attractor)
        let crossings = poincare_section(&traj, 0.005, 2, 27.0, true);
        assert!(crossings.len() > 10, "Should have many crossings");

        // Return map should exist
        let rmap = return_map(&crossings, 0);
        assert!(!rmap.is_empty());
    }

    #[test]
    fn test_mean_return_time() {
        let params = LorenzParams::default();
        let traj = lorenz(State3D::new(1.0, 1.0, 1.0), &params, 0.005, 100_000);
        let crossings = poincare_section(&traj, 0.005, 2, 27.0, true);

        if let Some(mrt) = mean_return_time(&crossings) {
            assert!(mrt > 0.0, "Mean return time should be positive");
        }
    }
}
