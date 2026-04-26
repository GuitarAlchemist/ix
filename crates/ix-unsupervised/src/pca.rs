//! Principal Component Analysis (PCA).
//!
//! Implements PCA via covariance matrix eigendecomposition using the
//! power iteration method (no LAPACK dependency). Supports deflation
//! to extract multiple principal components.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::traits::DimensionReducer;

/// Serializable state for a fitted [`PCA`] model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaState {
    /// Principal components, each inner Vec is one component (length = n_features).
    pub components: Vec<Vec<f64>>,
    /// Explained variance for each component.
    pub explained_variance: Vec<f64>,
    /// Column means of the training data.
    pub mean: Vec<f64>,
    /// Number of components.
    pub n_components: usize,
}

/// PCA via power iteration eigendecomposition of the covariance matrix.
pub struct PCA {
    pub n_components: usize,
    /// Principal component vectors (n_components x n_features), each row is a component.
    components: Option<Array2<f64>>,
    /// Explained variance for each component.
    explained_variance: Option<Array1<f64>>,
    /// Column means of the training data (for centering).
    mean: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            explained_variance: None,
            mean: None,
        }
    }

    /// Get explained variance ratios (proportion of total variance per component).
    pub fn explained_variance_ratio(&self) -> Option<Array1<f64>> {
        self.explained_variance.as_ref().map(|ev| {
            let total = ev.sum();
            if total > 0.0 {
                ev / total
            } else {
                ev.clone()
            }
        })
    }

    /// Get the principal components matrix (n_components x n_features).
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Save the trained model state. Returns `None` if the model has not been fitted.
    pub fn save_state(&self) -> Option<PcaState> {
        let comps = self.components.as_ref()?;
        let ev = self.explained_variance.as_ref()?;
        let mean = self.mean.as_ref()?;

        let components = (0..comps.nrows()).map(|r| comps.row(r).to_vec()).collect();

        Some(PcaState {
            components,
            explained_variance: ev.to_vec(),
            mean: mean.to_vec(),
            n_components: self.n_components,
        })
    }

    /// Reconstruct a fitted model from a previously saved state.
    pub fn load_state(state: &PcaState) -> Self {
        let n_components = state.n_components;
        let n_features = state.mean.len();

        let flat: Vec<f64> = state
            .components
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let components = Array2::from_shape_vec((n_components, n_features), flat)
            .expect("PcaState components dimensions mismatch");

        Self {
            n_components,
            components: Some(components),
            explained_variance: Some(Array1::from_vec(state.explained_variance.clone())),
            mean: Some(Array1::from_vec(state.mean.clone())),
        }
    }
}

/// Power iteration: find the dominant eigenvector of a symmetric matrix.
/// Returns (eigenvalue, eigenvector).
fn power_iteration(matrix: &Array2<f64>, max_iter: usize, tol: f64) -> (f64, Array1<f64>) {
    let n = matrix.nrows();

    // Initialize with a non-zero vector
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Multiply: v_new = M * v
        let v_new = matrix.dot(&v);

        // Compute eigenvalue (Rayleigh quotient)
        let new_eigenvalue = v.dot(&v_new);

        // Normalize
        let norm = v_new.dot(&v_new).sqrt();
        if norm < 1e-15 {
            break;
        }
        let v_normalized = &v_new / norm;

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tol {
            eigenvalue = new_eigenvalue;
            v = v_normalized;
            break;
        }

        eigenvalue = new_eigenvalue;
        v = v_normalized;
    }

    (eigenvalue, v)
}

/// Deflate matrix by removing the component of the found eigenvector.
/// M' = M - eigenvalue * v * v^T
fn deflate(matrix: &Array2<f64>, eigenvalue: f64, eigenvector: &Array1<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    let mut result = matrix.clone();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
    result
}

impl DimensionReducer for PCA {
    fn fit(&mut self, x: &Array2<f64>) {
        let n = x.nrows();
        let p = x.ncols();
        let k = self.n_components.min(p);

        // Compute column means
        let mean = x.mean_axis(Axis(0)).unwrap();

        // Center the data
        let mut centered = x.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // Compute covariance matrix: (1/(n-1)) X^T X
        let cov = centered.t().dot(&centered) / (n.max(2) - 1) as f64;

        // Extract top-k eigenvectors via repeated power iteration + deflation
        let mut components = Array2::zeros((k, p));
        let mut explained_variance = Array1::zeros(k);
        let mut current_cov = cov;

        for i in 0..k {
            let (eigenvalue, eigenvector) = power_iteration(&current_cov, 1000, 1e-10);
            components.row_mut(i).assign(&eigenvector);
            explained_variance[i] = eigenvalue.max(0.0);
            current_cov = deflate(&current_cov, eigenvalue, &eigenvector);
        }

        self.components = Some(components);
        self.explained_variance = Some(explained_variance);
        self.mean = Some(mean);
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let components = self.components.as_ref().expect("Model not fitted");
        let mean = self.mean.as_ref().expect("Model not fitted");

        // Center the data
        let mut centered = x.clone();
        for mut row in centered.rows_mut() {
            row -= mean;
        }

        // Project: X_centered * components^T -> (n_samples, n_components)
        centered.dot(&components.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_reduces_dimensions() {
        // 3D data that lives mostly on a 2D plane
        let x = array![
            [1.0, 2.0, 0.1],
            [2.0, 4.0, 0.2],
            [3.0, 6.0, 0.3],
            [4.0, 8.0, 0.4],
            [5.0, 10.0, 0.5],
        ];

        let mut pca = PCA::new(2);
        let transformed = pca.fit_transform(&x);

        assert_eq!(transformed.nrows(), 5);
        assert_eq!(transformed.ncols(), 2);
    }

    #[test]
    fn test_pca_single_component() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0],];

        let mut pca = PCA::new(1);
        let transformed = pca.fit_transform(&x);

        assert_eq!(transformed.ncols(), 1);

        // The first PC should capture nearly all variance since data is on a line
        let ev_ratio = pca.explained_variance_ratio().unwrap();
        assert!(
            ev_ratio[0] > 0.99,
            "First PC should explain >99% variance, got {}",
            ev_ratio[0]
        );
    }

    #[test]
    fn test_pca_save_load_roundtrip() {
        let x = array![
            [1.0, 0.5, 0.2],
            [2.0, 1.1, 0.3],
            [3.0, 1.4, 0.8],
            [4.0, 2.1, 1.0],
            [5.0, 2.5, 1.5],
            [1.5, 0.8, 0.1],
            [3.5, 1.8, 0.9],
        ];

        let mut pca = PCA::new(2);
        pca.fit(&x);
        let orig_transformed = pca.transform(&x);

        let state = pca.save_state().expect("fitted PCA should produce state");

        // Roundtrip through JSON
        let json = serde_json::to_string(&state).unwrap();
        let restored_state: PcaState = serde_json::from_str(&json).unwrap();
        let restored = PCA::load_state(&restored_state);

        let rest_transformed = restored.transform(&x);
        assert_eq!(orig_transformed.dim(), rest_transformed.dim());
        for i in 0..orig_transformed.nrows() {
            for j in 0..orig_transformed.ncols() {
                assert!(
                    (orig_transformed[[i, j]] - rest_transformed[[i, j]]).abs() < 1e-12,
                    "transform results must match after roundtrip"
                );
            }
        }
    }

    #[test]
    fn test_pca_save_state_unfitted() {
        let pca = PCA::new(2);
        assert!(
            pca.save_state().is_none(),
            "unfitted PCA should return None"
        );
    }

    #[test]
    fn test_pca_components_orthogonal() {
        let x = array![
            [1.0, 0.5, 0.2],
            [2.0, 1.1, 0.3],
            [3.0, 1.4, 0.8],
            [4.0, 2.1, 1.0],
            [5.0, 2.5, 1.5],
            [1.5, 0.8, 0.1],
            [3.5, 1.8, 0.9],
        ];

        let mut pca = PCA::new(2);
        pca.fit(&x);

        let comps = pca.components().unwrap();
        // Check orthogonality: dot product of PC1 and PC2 should be ~0
        let dot: f64 = comps.row(0).dot(&comps.row(1));
        assert!(
            dot.abs() < 1e-6,
            "Components should be orthogonal, dot={}",
            dot
        );
    }

    #[test]
    fn test_pca_transform_matches_dimensions() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 5.0, 7.0, 9.0],
            [4.0, 6.0, 8.0, 10.0],
        ];

        let mut pca = PCA::new(2);
        pca.fit(&x);

        let new_data = array![[1.5, 2.5, 3.5, 4.5], [2.5, 4.0, 5.5, 7.0],];

        let transformed = pca.transform(&new_data);
        assert_eq!(transformed.nrows(), 2);
        assert_eq!(transformed.ncols(), 2);
    }

    #[test]
    fn test_pca_explained_variance_decreasing() {
        let x = array![
            [1.0, 0.5, 0.1],
            [2.0, 1.2, 0.3],
            [3.0, 1.4, 0.2],
            [4.0, 2.3, 0.5],
            [5.0, 2.5, 0.4],
            [6.0, 3.1, 0.6],
        ];

        let mut pca = PCA::new(3);
        pca.fit(&x);

        let ev = pca.explained_variance.as_ref().unwrap();
        for i in 1..ev.len() {
            assert!(
                ev[i] <= ev[i - 1] + 1e-10,
                "Explained variance should be non-increasing: {} > {}",
                ev[i],
                ev[i - 1]
            );
        }
    }
}
