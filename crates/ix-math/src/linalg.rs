//! Linear algebra operations on ndarray matrices.

use ndarray::{Array1, Array2, Axis};

use crate::error::MathError;

/// Matrix multiply: C = A * B
pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    if a.ncols() != b.nrows() {
        return Err(MathError::DimensionMismatch {
            expected: a.ncols(),
            got: b.nrows(),
        });
    }
    Ok(a.dot(b))
}

/// Matrix-vector multiply: y = A * x
pub fn matvec(a: &Array2<f64>, x: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    if a.ncols() != x.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.ncols(),
            got: x.len(),
        });
    }
    Ok(a.dot(x))
}

/// Transpose a matrix.
pub fn transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
}

/// Determinant of a square matrix (cofactor expansion, suitable for small matrices).
pub fn determinant(a: &Array2<f64>) -> Result<f64, MathError> {
    let (n, m) = a.dim();
    if n != m {
        return Err(MathError::NotSquare { rows: n, cols: m });
    }
    Ok(det_recursive(a))
}

fn det_recursive(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    if n == 1 {
        return a[[0, 0]];
    }
    if n == 2 {
        return a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    }

    let mut det = 0.0;
    for j in 0..n {
        let minor = minor_matrix(a, 0, j);
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * a[[0, j]] * det_recursive(&minor);
    }
    det
}

fn minor_matrix(a: &Array2<f64>, row: usize, col: usize) -> Array2<f64> {
    let n = a.nrows();
    let mut result = Array2::zeros((n - 1, n - 1));
    let mut ri = 0;
    for i in 0..n {
        if i == row {
            continue;
        }
        let mut ci = 0;
        for j in 0..n {
            if j == col {
                continue;
            }
            result[[ri, ci]] = a[[i, j]];
            ci += 1;
        }
        ri += 1;
    }
    result
}

/// Inverse of a square matrix using Gauss-Jordan elimination.
pub fn inverse(a: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let (n, m) = a.dim();
    if n != m {
        return Err(MathError::NotSquare { rows: n, cols: m });
    }

    // Augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(MathError::Singular);
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Extract inverse from right half
    Ok(aug.slice(ndarray::s![.., n..]).to_owned())
}

/// Identity matrix of size n.
pub fn eye(n: usize) -> Array2<f64> {
    Array2::eye(n)
}

/// Trace of a square matrix.
pub fn trace(a: &Array2<f64>) -> Result<f64, MathError> {
    let (n, m) = a.dim();
    if n != m {
        return Err(MathError::NotSquare { rows: n, cols: m });
    }
    Ok(a.diag().sum())
}

/// Column-wise mean of a matrix: returns a row vector.
pub fn col_mean(a: &Array2<f64>) -> Array1<f64> {
    a.mean_axis(Axis(0)).unwrap()
}

/// Normalize columns to zero mean, unit variance. Returns (normalized, means, stds).
pub fn standardize(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let means = col_mean(a);
    let centered = a - &means;
    let _n = a.nrows() as f64;
    let stds = centered
        .mapv(|x| x * x)
        .mean_axis(Axis(0))
        .unwrap()
        .mapv(|x| (x).sqrt().max(1e-12));
    let normalized = &centered / &stds;
    (normalized, means, stds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn test_determinant_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let det = determinant(&a).unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let det = determinant(&a).unwrap();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let inv = inverse(&a).unwrap();
        let product = matmul(&a, &inv).unwrap();
        let identity = eye(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((product[[i, j]] - identity[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_singular_matrix() {
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(inverse(&a).is_err());
    }

    #[test]
    fn test_trace() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!((trace(&a).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (norm, means, _stds) = standardize(&a);
        // Means should be ~0
        let new_means = col_mean(&norm);
        assert!(new_means[0].abs() < 1e-10);
        assert!(new_means[1].abs() < 1e-10);
        assert!((means[0] - 3.0).abs() < 1e-10);
    }
}
