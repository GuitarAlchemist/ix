//! Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.
//!
//! LDA finds a linear projection that maximizes between-class variance while
//! minimizing within-class variance. Unlike PCA (which is unsupervised and
//! maximizes total variance), LDA uses class labels to find directions that
//! best separate the classes.
//!
//! For a c-class problem, LDA reduces the feature space to at most `c - 1`
//! dimensions. This is useful both as a preprocessing step for classifiers
//! and as a visualization technique for labeled data.
//!
//! # Algorithm
//!
//! 1. Compute the class means and overall mean.
//! 2. Build the within-class scatter matrix `S_W = sum_c sum_i (x_i - mu_c)(x_i - mu_c)^T`.
//! 3. Build the between-class scatter matrix `S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T`.
//! 4. Solve the generalized eigenvalue problem `S_B v = lambda S_W v`, which
//!    we convert to a standard eigenvalue problem `S_W^{-1} S_B v = lambda v`.
//! 5. Take the eigenvectors with the largest eigenvalues as the projection axes.
//!
//! # Example
//!
//! ```
//! use ix_unsupervised::lda::LinearDiscriminantAnalysis;
//! use ndarray::array;
//!
//! // Two classes, clearly separated along x-axis
//! let x = array![
//!     [0.0, 0.0],
//!     [0.1, 0.1],
//!     [0.0, 0.2],
//!     [5.0, 0.0],
//!     [5.1, 0.1],
//!     [5.0, 0.2],
//! ];
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let mut lda = LinearDiscriminantAnalysis::new(1);
//! lda.fit(&x, &y).unwrap();
//! let projected = lda.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ndarray::{Array1, Array2};
use std::collections::BTreeMap;

use ix_math::eigen::symmetric_eigen;
use ix_math::error::MathError;

/// Linear Discriminant Analysis model.
///
/// After `fit`, the model stores up to `n_components` projection vectors
/// (columns of the transformation matrix) along with the class means.
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Number of discriminant components to keep. Upper bound is `min(n_classes - 1, n_features)`.
    pub n_components: usize,
    /// Transformation matrix with shape `(n_features, n_components)`, filled by `fit`.
    pub components: Option<Array2<f64>>,
    /// Eigenvalues associated with each retained component.
    pub explained: Option<Array1<f64>>,
    /// Overall mean used for centering inputs.
    pub mean: Option<Array1<f64>>,
}

impl LinearDiscriminantAnalysis {
    /// Create a new LDA model requesting `n_components` discriminant axes.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            explained: None,
            mean: None,
        }
    }

    /// Fit the LDA model on features `x` (n_samples x n_features) and labels `y`.
    ///
    /// Labels are arbitrary `usize` values; internally they are grouped into
    /// classes via a sorted map, and the number of classes is inferred.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) -> Result<(), MathError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(MathError::EmptyInput);
        }
        if y.len() != n_samples {
            return Err(MathError::DimensionMismatch {
                expected: n_samples,
                got: y.len(),
            });
        }

        // Group row indices by class label.
        let mut by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &lbl) in y.iter().enumerate() {
            by_class.entry(lbl).or_default().push(i);
        }
        let n_classes = by_class.len();
        if n_classes < 2 {
            return Err(MathError::InvalidParameter(
                "LDA needs at least 2 classes".into(),
            ));
        }

        let max_components = (n_classes - 1).min(n_features);
        if self.n_components > max_components {
            return Err(MathError::InvalidParameter(format!(
                "n_components={} exceeds max {} for {} classes and {} features",
                self.n_components, max_components, n_classes, n_features
            )));
        }

        // Overall mean and per-class means.
        let overall_mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let mut class_means: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
        for (&lbl, indices) in &by_class {
            let mut mu = Array1::<f64>::zeros(n_features);
            for &i in indices {
                mu = mu + x.row(i).to_owned();
            }
            mu /= indices.len() as f64;
            class_means.insert(lbl, mu);
        }

        // Within-class scatter S_W = sum_c sum_i (x_i - mu_c)(x_i - mu_c)^T
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        for (&lbl, indices) in &by_class {
            let mu_c = class_means.get(&lbl).unwrap();
            for &i in indices {
                let diff = &x.row(i).to_owned() - mu_c;
                for a in 0..n_features {
                    for b in 0..n_features {
                        sw[[a, b]] += diff[a] * diff[b];
                    }
                }
            }
        }

        // Between-class scatter S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T
        let mut sb = Array2::<f64>::zeros((n_features, n_features));
        for (&lbl, indices) in &by_class {
            let mu_c = class_means.get(&lbl).unwrap();
            let diff = mu_c - &overall_mean;
            let n_c = indices.len() as f64;
            for a in 0..n_features {
                for b in 0..n_features {
                    sb[[a, b]] += n_c * diff[a] * diff[b];
                }
            }
        }

        // Regularize S_W slightly to keep it invertible even when classes
        // share degenerate directions (very small numerical safeguard).
        for i in 0..n_features {
            sw[[i, i]] += 1e-10;
        }

        // The naive formulation solves the generalized eigenvalue problem
        // `S_B v = lambda S_W v` by inverting to `(S_W^{-1} S_B) v = lambda v`.
        // But `S_W^{-1} S_B` is generally *not* symmetric, so deflation-based
        // eigensolvers silently produce wrong or duplicated components on
        // repeated eigenvalues. We symmetrize the problem instead:
        //
        //   Let S_W^{-1/2} be the symmetric inverse square root of S_W.
        //   Define  M = S_W^{-1/2} S_B S_W^{-1/2}  which IS symmetric.
        //   Solve M u = lambda u by standard symmetric eigendecomposition.
        //   The LDA direction is  v = S_W^{-1/2} u.
        //
        // This gives the same eigenvalues as the generalized problem with
        // correct eigenvectors even under spectral degeneracy.

        // Step 1: symmetric eigendecomposition of S_W -> U diag(d) U^T
        let (sw_eigenvalues, sw_eigenvectors) = symmetric_eigen(&sw)?;

        // Step 2: S_W^{-1/2} = U diag(1/sqrt(d)) U^T
        let mut sw_inv_sqrt = Array2::<f64>::zeros((n_features, n_features));
        for k in 0..n_features {
            let dk = sw_eigenvalues[k];
            if dk <= 0.0 {
                return Err(MathError::InvalidParameter(
                    "within-class scatter matrix is not positive definite".into(),
                ));
            }
            let inv_sqrt_dk = 1.0 / dk.sqrt();
            for i in 0..n_features {
                for j in 0..n_features {
                    sw_inv_sqrt[[i, j]] +=
                        inv_sqrt_dk * sw_eigenvectors[[i, k]] * sw_eigenvectors[[j, k]];
                }
            }
        }

        // Step 3: M = S_W^{-1/2} S_B S_W^{-1/2}  (symmetric by construction)
        let m = sw_inv_sqrt.dot(&sb).dot(&sw_inv_sqrt);

        // Step 4: symmetric eigendecomposition of M
        let (m_eigenvalues, m_eigenvectors) = symmetric_eigen(&m)?;

        // Step 5: LDA directions are v_k = S_W^{-1/2} u_k.
        //         m_eigenvalues/vectors are already sorted descending.
        let mut components = Array2::<f64>::zeros((n_features, self.n_components));
        let mut explained = Array1::<f64>::zeros(self.n_components);
        for k in 0..self.n_components {
            explained[k] = m_eigenvalues[k].max(0.0);
            let u_k = m_eigenvectors.column(k);
            let v_k = sw_inv_sqrt.dot(&u_k);
            for i in 0..n_features {
                components[[i, k]] = v_k[i];
            }
        }

        self.components = Some(components);
        self.explained = Some(explained);
        self.mean = Some(overall_mean);
        Ok(())
    }

    /// Project new samples into the LDA subspace. Requires `fit` first.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| MathError::InvalidParameter("model not fitted".into()))?;
        let mean = self.mean.as_ref().unwrap();

        if x.ncols() != mean.len() {
            return Err(MathError::DimensionMismatch {
                expected: mean.len(),
                got: x.ncols(),
            });
        }

        let centered = x - &mean.view().insert_axis(ndarray::Axis(0));
        Ok(centered.dot(components))
    }

    /// Convenience: fit then transform in one call.
    pub fn fit_transform(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<Array2<f64>, MathError> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_two_classes_1d_projection() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.2],
            [-0.1, 0.1],
            [5.0, 0.0],
            [5.1, 0.2],
            [4.9, 0.1],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        lda.fit(&x, &y).unwrap();
        let projected = lda.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
        // Class 0 and class 1 should be well separated in the projection
        let c0_mean: f64 = projected.slice(ndarray::s![0..3, 0]).mean().unwrap();
        let c1_mean: f64 = projected.slice(ndarray::s![3..6, 0]).mean().unwrap();
        assert!(
            (c0_mean - c1_mean).abs() > 1.0,
            "classes should be separated: {} vs {}",
            c0_mean,
            c1_mean
        );
    }

    #[test]
    fn test_lda_three_classes_2d_projection() {
        // Three clusters in 3D, LDA should find a 2D separating subspace
        let x = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [5.0, 0.0, 0.0],
            [5.1, 0.0, 0.0],
            [5.0, 0.1, 0.0],
            [0.0, 5.0, 0.0],
            [0.1, 5.0, 0.0],
            [0.0, 5.1, 0.0],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut lda = LinearDiscriminantAnalysis::new(2);
        lda.fit(&x, &y).unwrap();
        let projected = lda.fit_transform(&x, &y).unwrap();
        assert_eq!(projected.ncols(), 2);
    }

    #[test]
    fn test_lda_rejects_single_class() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0usize, 0];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_rejects_too_many_components() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![0usize, 0, 1, 1];
        // 2 classes allow max 1 component
        let mut lda = LinearDiscriminantAnalysis::new(2);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_three_classes_axis_aligned_repeated_eigenvalues() {
        // Three class clusters aligned on the three coordinate axes of a
        // 3D feature space. In the naive power-iteration-plus-deflation
        // formulation, the non-symmetric S_W^-1 S_B has repeated eigenvalues
        // and deflation silently produces duplicated components.
        //
        // After the symmetrization fix, LDA should return two distinct
        // components whose projections separate all three classes.
        let x = array![
            // class 0: along +x
            [5.0, 0.0, 0.0],
            [5.1, 0.1, 0.0],
            [4.9, 0.0, 0.1],
            // class 1: along +y
            [0.0, 5.0, 0.0],
            [0.1, 5.1, 0.0],
            [0.0, 4.9, 0.1],
            // class 2: along +z
            [0.0, 0.0, 5.0],
            [0.0, 0.1, 5.1],
            [0.1, 0.0, 4.9],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut lda = LinearDiscriminantAnalysis::new(2);
        let projected = lda.fit_transform(&x, &y).unwrap();

        // All 3 class centroids should be distinct in the 2D projection.
        let centroid = |start: usize| -> (f64, f64) {
            let mut cx = 0.0;
            let mut cy = 0.0;
            for i in start..(start + 3) {
                cx += projected[[i, 0]];
                cy += projected[[i, 1]];
            }
            (cx / 3.0, cy / 3.0)
        };
        let c0 = centroid(0);
        let c1 = centroid(3);
        let c2 = centroid(6);

        let dist =
            |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        // Each pair should be meaningfully separated in the projection.
        assert!(
            dist(c0, c1) > 1.0 && dist(c0, c2) > 1.0 && dist(c1, c2) > 1.0,
            "class centroids too close: c0={:?} c1={:?} c2={:?}",
            c0,
            c1,
            c2
        );

        // The two explained values should both be significantly positive —
        // the old deflation bug would collapse one of them to near zero.
        let explained = lda.explained.as_ref().unwrap();
        assert!(explained[0] > 1.0);
        assert!(explained[1] > 1.0);
    }

    #[test]
    fn test_lda_transform_dimension_check() {
        let x = array![[0.0, 0.0], [5.0, 0.0]];
        let y = array![0usize, 1];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        lda.fit(&x, &y).unwrap();
        let bad = array![[0.0, 0.0, 0.0]];
        assert!(lda.transform(&bad).is_err());
    }
}
