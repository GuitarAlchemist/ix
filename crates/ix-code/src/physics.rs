//! # Physics-Inspired Code Analysis (Layer 7)
//!
//! Applies techniques from dynamical systems, signal processing, spectral graph
//! theory, and Markov chain analysis to code quality metric trajectories.
//!
//! Sub-modules:
//! - [`stability`]: Chaos-theoretic stability from Lyapunov-like exponents.
//! - [`kalman_quality`]: Kalman-filtered quality estimates with innovation/anomaly
//!   detection.
//! - [`wavelet_scale`]: Multi-scale wavelet decomposition (coarse trend vs fine
//!   detail).
//! - [`markov_evolution`]: Markov model of code quality evolution between
//!   discretized states.
//! - [`laplacian`]: Graph Laplacian spectrum — algebraic connectivity, spectral
//!   gap, Fiedler vector.
//! - [`lie_symmetry`]: Stub for future Lie-group based refactoring symmetry
//!   detection.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 7a. Chaos-Theoretic Stability (ix-chaos inspired)
// ---------------------------------------------------------------------------

/// Classification of a code metric trajectory's stability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityClass {
    /// Metric is converging / bounded variation — healthy.
    Stable,
    /// Small bounded oscillations — borderline.
    Marginal,
    /// Positive Lyapunov exponent — sensitive dependence, chaotic.
    Chaotic,
    /// Rapidly growing / unbounded — refactoring bifurcation.
    Bifurcating,
}

/// Stability analysis of a scalar code metric trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStability {
    /// Estimated maximal Lyapunov exponent of the series.
    pub lyapunov_exponent: f64,
    /// Qualitative class.
    pub stability_class: StabilityClass,
}

/// Analyze code-metric trajectory stability via a finite-difference Lyapunov
/// estimator.
///
/// The series is treated as the orbit of an unknown 1D map. For consecutive
/// triples (x_{k-1}, x_k, x_{k+1}) the local slope
/// `df ≈ (x_{k+1} - x_k) / (x_k - x_{k-1})` approximates the derivative of the
/// implicit dynamics. The mean of `ln|df|` is the maximal Lyapunov exponent.
///
/// For constant or near-constant series (no information), the exponent is 0
/// and the class is `Stable`.
///
/// # Examples
/// ```
/// # #[cfg(feature = "physics")] {
/// use ix_code::physics::{analyze_code_stability, StabilityClass};
/// let flat = vec![10.0; 20];
/// let s = analyze_code_stability(&flat);
/// assert_eq!(s.stability_class, StabilityClass::Stable);
/// # }
/// ```
pub fn analyze_code_stability(metric_history: &[f64]) -> CodeStability {
    if metric_history.len() < 3 {
        return CodeStability {
            lyapunov_exponent: 0.0,
            stability_class: StabilityClass::Stable,
        };
    }

    // Detect a (near-)constant series up front and classify it as Stable.
    let min = metric_history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = metric_history
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let scale = max.abs().max(min.abs()).max(1.0);

    if range / scale < 1e-9 {
        return CodeStability {
            lyapunov_exponent: 0.0,
            stability_class: StabilityClass::Stable,
        };
    }

    // Finite-difference estimator: treat ratio of consecutive increments as an
    // approximate local derivative of an unknown 1D map.
    let mut sum = 0.0;
    let mut count = 0usize;
    let eps = 1e-12;

    for w in metric_history.windows(3) {
        let dx0 = w[1] - w[0];
        let dx1 = w[2] - w[1];
        if dx0.abs() < eps {
            continue;
        }
        let slope = (dx1 / dx0).abs();
        if slope < eps {
            continue;
        }
        sum += slope.ln();
        count += 1;
    }

    let lyap = if count == 0 { 0.0 } else { sum / count as f64 };

    // A useful auxiliary signal: the normalized total variation. Chaotic
    // trajectories oscillate with large jumps relative to their range.
    let total_variation: f64 = metric_history
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum();
    let normalized_tv =
        total_variation / (range * (metric_history.len().saturating_sub(1) as f64).max(1.0));

    let class = if !lyap.is_finite() || lyap > 2.0 || normalized_tv > 0.8 {
        if normalized_tv > 0.8 && lyap.is_finite() && lyap < 2.0 {
            StabilityClass::Chaotic
        } else {
            StabilityClass::Bifurcating
        }
    } else if lyap > 0.05 || normalized_tv > 0.4 {
        StabilityClass::Chaotic
    } else if lyap > -0.05 {
        StabilityClass::Marginal
    } else {
        StabilityClass::Stable
    };

    CodeStability {
        lyapunov_exponent: lyap,
        stability_class: class,
    }
}

// ---------------------------------------------------------------------------
// 7b. Kalman-Filtered Quality (ix-signal)
// ---------------------------------------------------------------------------

/// Output of the Kalman quality filter per time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredQuality {
    /// Posterior mean estimate.
    pub estimated_quality: f64,
    /// Posterior variance (uncertainty).
    pub uncertainty: f64,
    /// Innovation = measurement - predicted.
    pub innovation: f64,
    /// True if `|innovation|` exceeds 3 sigma of innovation covariance.
    pub is_anomaly: bool,
}

/// 1-D Kalman filter on a scalar quality trajectory.
///
/// State model: x_{k+1} = x_k + w,  w ~ N(0, q)
/// Obs model:   z_k = x_k + v,      v ~ N(0, r)
///
/// Uses [`ix_signal::kalman::KalmanFilter`] under the hood.
#[cfg(feature = "physics")]
pub fn kalman_filter_metrics(history: &[f64], q: f64, r: f64) -> Vec<FilteredQuality> {
    use ix_signal::kalman::KalmanFilter;
    use ndarray::array;

    if history.is_empty() {
        return Vec::new();
    }

    let mut kf = KalmanFilter::new(1, 1);
    kf.transition = array![[1.0]];
    kf.observation = array![[1.0]];
    kf.process_noise = array![[q]];
    kf.measurement_noise = array![[r]];
    kf.state = array![history[0]];
    kf.covariance = array![[1.0]];

    let mut out = Vec::with_capacity(history.len());
    for &z in history {
        // Predict
        kf.predict(None);

        // Compute innovation and its covariance before update.
        let predicted = kf.state[0];
        let innovation = z - predicted;
        let s = kf.covariance[[0, 0]] + r;
        let sigma = s.sqrt().max(1e-12);
        let is_anomaly = innovation.abs() > 3.0 * sigma;

        // Update
        kf.update(&array![z]);

        out.push(FilteredQuality {
            estimated_quality: kf.state[0],
            uncertainty: kf.covariance[[0, 0]],
            innovation,
            is_anomaly,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// 7c. Wavelet Multi-Scale Analysis (ix-signal)
// ---------------------------------------------------------------------------

/// Multi-scale decomposition of a metric trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleAnalysis {
    /// Coarse approximation (low-frequency trend) at the deepest level.
    pub coarse_trend: Vec<f64>,
    /// Finest-level detail coefficients (high-frequency noise/oscillation).
    pub fine_detail: Vec<f64>,
    /// Per-level energy of detail coefficients (sum of squares), outermost level first.
    pub scale_energy: Vec<f64>,
    /// Index into `scale_energy` with the largest energy — the dominant scale.
    pub dominant_scale: usize,
}

/// Multi-level Haar wavelet decomposition of a metric trajectory.
///
/// Uses [`ix_signal::wavelet::haar_dwt`]. The trajectory is padded with its
/// mean to the nearest multiple of `2^levels`.
#[cfg(feature = "physics")]
pub fn wavelet_decompose_metrics(trajectory: &[f64], levels: usize) -> MultiScaleAnalysis {
    use ix_signal::wavelet::haar_dwt;

    if trajectory.is_empty() || levels == 0 {
        return MultiScaleAnalysis {
            coarse_trend: trajectory.to_vec(),
            fine_detail: Vec::new(),
            scale_energy: Vec::new(),
            dominant_scale: 0,
        };
    }

    // Pad to a multiple of 2^levels with the signal mean for stability.
    let block = 1usize << levels;
    let mean = trajectory.iter().sum::<f64>() / trajectory.len() as f64;
    let mut padded = trajectory.to_vec();
    while padded.len() % block != 0 {
        padded.push(mean);
    }

    let (approx, details) = haar_dwt(&padded, levels);

    let scale_energy: Vec<f64> = details
        .iter()
        .map(|d| d.iter().map(|x| x * x).sum::<f64>())
        .collect();

    let dominant_scale = scale_energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // `details[0]` is the finest level (first decomposition of the raw signal).
    let fine_detail = details.first().cloned().unwrap_or_default();

    MultiScaleAnalysis {
        coarse_trend: approx,
        fine_detail,
        scale_energy,
        dominant_scale,
    }
}

// ---------------------------------------------------------------------------
// 7d. Markov Code Evolution (ix-graph)
// ---------------------------------------------------------------------------

/// Markov model of code-quality evolution between discretized quality bins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEvolutionModel {
    /// Row-stochastic transition matrix, row-major `n_states × n_states`.
    pub transition_matrix: Vec<Vec<f64>>,
    /// Bin edges (length = `n_states + 1`).
    pub states: Vec<f64>,
    /// Stationary distribution (power iteration).
    pub steady_state: Vec<f64>,
    /// Mean first-passage time (in steps) to the worst-quality bin.
    pub mean_time_to_critical: f64,
    /// Current state index (bin containing the last observation).
    pub current_state: usize,
}

/// Build a Markov model from a quality history by discretizing into `n_states`
/// equal-width bins between the min and max of the series.
#[cfg(feature = "physics")]
pub fn model_code_evolution(quality_history: &[f64], n_states: usize) -> CodeEvolutionModel {
    use ix_graph::markov::MarkovChain;
    use ndarray::Array2;

    assert!(n_states >= 2, "n_states must be at least 2");

    if quality_history.len() < 2 {
        let n = n_states;
        let uniform = vec![vec![1.0 / n as f64; n]; n];
        return CodeEvolutionModel {
            transition_matrix: uniform,
            states: (0..=n).map(|i| i as f64).collect(),
            steady_state: vec![1.0 / n as f64; n],
            mean_time_to_critical: f64::INFINITY,
            current_state: 0,
        };
    }

    let min = quality_history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = quality_history
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let span = (max - min).max(1e-12);
    let width = span / n_states as f64;

    let bin_of = |x: f64| -> usize {
        let idx = ((x - min) / width).floor() as isize;
        idx.clamp(0, n_states as isize - 1) as usize
    };

    let edges: Vec<f64> = (0..=n_states)
        .map(|i| min + width * i as f64)
        .collect();

    // Count transitions.
    let mut counts = vec![vec![0.0f64; n_states]; n_states];
    for w in quality_history.windows(2) {
        let a = bin_of(w[0]);
        let b = bin_of(w[1]);
        counts[a][b] += 1.0;
    }

    // Row-normalize with Laplace smoothing so every row sums to exactly 1.0.
    let mut matrix = vec![vec![0.0f64; n_states]; n_states];
    for (i, row) in matrix.iter_mut().enumerate() {
        let total: f64 = counts[i].iter().sum::<f64>() + n_states as f64 * 1e-6;
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (counts[i][j] + 1e-6) / total;
        }
        // Re-normalize to kill floating-point drift.
        let row_sum: f64 = row.iter().sum();
        for cell in row.iter_mut() {
            *cell /= row_sum;
        }
    }

    // Build an ndarray transition matrix and hand it to ix-graph.
    let mut arr = Array2::<f64>::zeros((n_states, n_states));
    for i in 0..n_states {
        for j in 0..n_states {
            arr[[i, j]] = matrix[i][j];
        }
    }
    let chain = MarkovChain::new(arr).expect("rows sum to 1");

    let stationary = chain.stationary_distribution(1000, 1e-10).to_vec();

    // Current state = bin of last observation. Critical = bin 0 (lowest quality).
    let current = bin_of(*quality_history.last().unwrap());
    let mean_time_to_critical = if current == 0 {
        0.0
    } else {
        chain.mean_first_passage(current, 0, 200, 1000, 0xC0DE)
    };

    CodeEvolutionModel {
        transition_matrix: matrix,
        states: edges,
        steady_state: stationary,
        mean_time_to_critical,
        current_state: current,
    }
}

// ---------------------------------------------------------------------------
// 7f. Graph Laplacian Spectrum
// ---------------------------------------------------------------------------

/// Spectrum of the graph Laplacian L = D - A.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianSpectrum {
    /// A few smallest eigenvalues (ascending), computed by inverse power
    /// iteration with deflation.
    pub eigenvalues: Vec<f64>,
    /// Algebraic connectivity = second smallest eigenvalue.
    pub algebraic_connectivity: f64,
    /// Spectral gap = lambda_2 - lambda_1.
    pub spectral_gap: f64,
    /// Number of connected components (multiplicity of eigenvalue 0).
    pub n_connected_components: usize,
    /// Fiedler vector (eigenvector associated with the algebraic connectivity).
    pub fiedler_vector: Vec<f64>,
}

/// Compute the low end of the Laplacian spectrum for an undirected graph given
/// by `n` vertices and a list of `edges`.
///
/// Uses power iteration on `M = c*I - L` to extract eigenpairs of `L`
/// corresponding to its smallest eigenvalues, with Gram-Schmidt deflation.
///
/// Computes the three smallest eigenvalues and the Fiedler vector.
pub fn compute_laplacian_spectrum(n: usize, edges: &[(usize, usize)]) -> LaplacianSpectrum {
    use ndarray::Array2;

    if n == 0 {
        return LaplacianSpectrum {
            eigenvalues: Vec::new(),
            algebraic_connectivity: 0.0,
            spectral_gap: 0.0,
            n_connected_components: 0,
            fiedler_vector: Vec::new(),
        };
    }

    // Build Laplacian L = D - A.
    let mut lap = Array2::<f64>::zeros((n, n));
    for &(u, v) in edges {
        if u == v || u >= n || v >= n {
            continue;
        }
        lap[[u, v]] -= 1.0;
        lap[[v, u]] -= 1.0;
        lap[[u, u]] += 1.0;
        lap[[v, v]] += 1.0;
    }

    // Full symmetric eigendecomposition via cyclic Jacobi rotations. For
    // Observatory-sized graphs (n up to a few hundred) this is fast and
    // numerically stable, and avoids pulling in an external LAPACK binding.
    let (eig_all, vec_all) = jacobi_eigen(&lap);

    // Sort ascending by eigenvalue. `vec_all` is indexed [i][j] = j-th
    // coordinate of the i-th eigenvector.
    let mut indexed: Vec<(f64, Vec<f64>)> = eig_all.into_iter().zip(vec_all).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues_all: Vec<f64> = indexed.iter().map(|(e, _)| e.max(0.0)).collect();

    let zero_tol = 1e-6;
    let n_components = eigenvalues_all
        .iter()
        .filter(|&&e| e < zero_tol)
        .count()
        .max(1);

    let k = 3.min(n);
    let eigenvalues: Vec<f64> = eigenvalues_all.iter().take(k).copied().collect();

    let lambda1 = eigenvalues.first().copied().unwrap_or(0.0);
    let lambda2 = eigenvalues.get(1).copied().unwrap_or(lambda1);
    let algebraic_connectivity = lambda2;
    let spectral_gap = lambda2 - lambda1;

    let fiedler_vector: Vec<f64> = indexed
        .get(1)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| vec![0.0; n]);

    LaplacianSpectrum {
        eigenvalues,
        algebraic_connectivity,
        spectral_gap,
        n_connected_components: n_components,
        fiedler_vector,
    }
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is the i-th
/// eigenvector (as a plain `Vec<f64>`) and `eigenvalues[i]` its eigenvalue.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen(matrix: &ndarray::Array2<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.nrows();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| matrix[[i, j]]).collect())
        .collect();
    // V accumulates the rotations; column j of V is the j-th eigenvector.
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 100;
    let tol = 1e-12;

    for _sweep in 0..max_sweeps {
        // Measure off-diagonal magnitude.
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1e-14 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update rows/columns p and q.
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[p][i] = a[i][p];
                        a[i][q] = s * aip + c * aiq;
                        a[q][i] = a[i][q];
                    }
                }

                // Update the accumulated rotations.
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Extract eigenvectors as rows (column j of V -> j-th eigenvector).
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| v[i][j]).collect())
        .collect();
    (eigenvalues, eigenvectors)
}

// ---------------------------------------------------------------------------
// 7e. Lie Symmetry (stub)
// ---------------------------------------------------------------------------

/// Placeholder result type for future Lie-group symmetry detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LieSymmetryReport {
    /// Human-readable status message.
    pub status: String,
    /// Number of candidate symmetries detected (currently always 0).
    pub symmetry_count: usize,
}

/// Stub: detect refactoring symmetries via Lie-group action on an AST.
///
/// Not yet implemented — this function returns a placeholder report. A future
/// implementation will infer one-parameter refactoring flows (rename,
/// extract-method, inline) as generators of a Lie algebra acting on the AST.
pub fn detect_lie_symmetries(_ast_signature: &[f64]) -> LieSymmetryReport {
    LieSymmetryReport {
        status: "not implemented — stub".to_string(),
        symmetry_count: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lyapunov_stable() {
        // Constant series: no information, lyapunov should be 0, class Stable.
        let s = analyze_code_stability(&vec![42.0; 32]);
        assert!(s.lyapunov_exponent.abs() < 1e-9);
        assert_eq!(s.stability_class, StabilityClass::Stable);
    }

    #[test]
    fn test_lyapunov_classifies_chaos() {
        // Logistic map at r=4 — known chaotic.
        let mut x = 0.1f64;
        let mut series = Vec::with_capacity(200);
        for _ in 0..200 {
            x = 4.0 * x * (1.0 - x);
            series.push(x);
        }
        let s = analyze_code_stability(&series);
        assert!(matches!(
            s.stability_class,
            StabilityClass::Chaotic | StabilityClass::Bifurcating
        ));
    }

    #[cfg(feature = "physics")]
    #[test]
    fn test_kalman_smoothing() {
        // Noisy signal around a ramp; filtered output should be smoother.
        let truth: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
        let noisy: Vec<f64> = truth
            .iter()
            .enumerate()
            .map(|(i, t)| t + 0.5 * ((i as f64 * 1.7).sin()))
            .collect();

        let filtered = kalman_filter_metrics(&noisy, 0.01, 0.5);
        assert_eq!(filtered.len(), noisy.len());

        // Total variation of the filtered signal should be strictly less than
        // that of the noisy input.
        let tv = |xs: &[f64]| -> f64 {
            xs.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
        };
        let tv_noisy = tv(&noisy);
        let est: Vec<f64> = filtered.iter().map(|f| f.estimated_quality).collect();
        let tv_filt = tv(&est);
        assert!(
            tv_filt < tv_noisy,
            "filtered TV {} should be < noisy TV {}",
            tv_filt,
            tv_noisy
        );
    }

    #[cfg(feature = "physics")]
    #[test]
    fn test_wavelet_decomposition() {
        let signal: Vec<f64> = (0..16)
            .map(|i| (i as f64 * 0.4).sin() + i as f64 * 0.1)
            .collect();
        let dec = wavelet_decompose_metrics(&signal, 2);
        assert!(!dec.coarse_trend.is_empty());
        assert_eq!(dec.scale_energy.len(), 2);
        // Energies are non-negative.
        assert!(dec.scale_energy.iter().all(|&e| e >= 0.0));
    }

    #[cfg(feature = "physics")]
    #[test]
    fn test_markov_evolution() {
        // Zig-zag series covering a range.
        let series: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 1.0 } else { 5.0 })
            .collect();
        let model = model_code_evolution(&series, 4);
        for row in &model.transition_matrix {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sum {}", s);
        }
        assert_eq!(model.states.len(), 5);
        assert_eq!(model.steady_state.len(), 4);
    }

    #[test]
    fn test_laplacian_connected() {
        // K4 complete graph.
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let spec = compute_laplacian_spectrum(4, &edges);
        assert!(spec.algebraic_connectivity > 0.5);
        assert_eq!(spec.n_connected_components, 1);
        // Smallest eigenvalue ~ 0.
        assert!(spec.eigenvalues[0] < 1e-4);
    }

    #[test]
    fn test_laplacian_disconnected() {
        // Two disjoint edges: 0-1 and 2-3.
        let edges = vec![(0, 1), (2, 3)];
        let spec = compute_laplacian_spectrum(4, &edges);
        assert!(spec.n_connected_components >= 2);
    }

    #[test]
    fn test_lie_symmetry_stub() {
        let rep = detect_lie_symmetries(&[1.0, 2.0]);
        assert_eq!(rep.symmetry_count, 0);
    }
}
