//! Hyperbolic geometry: Poincaré disk/ball model.
//!
//! The Poincaré ball model maps all of hyperbolic space into the open unit ball.
//! Distances grow exponentially toward the boundary, making it ideal for:
//! - Hierarchical embeddings (knowledge graphs, taxonomies)
//! - Agent skill dependency trees
//! - Any data with latent tree-like structure
//!
//! Key property: a tree with branching factor b has O(b^d) nodes at depth d,
//! which matches the exponential volume growth of hyperbolic space.

use ndarray::Array1;

use crate::error::MathError;

// ─── Distance ───────────────────────────────────────────────────────────────

/// Poincaré ball distance:
///   d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))
///
/// Both u and v must be inside the unit ball (||x|| < 1).
pub fn poincare_distance(u: &Array1<f64>, v: &Array1<f64>) -> Result<f64, MathError> {
    check_in_ball(u)?;
    check_in_ball(v)?;

    let diff = u - v;
    let diff_sq = diff.dot(&diff);
    let u_sq = u.dot(u);
    let v_sq = v.dot(v);

    let denom = (1.0 - u_sq) * (1.0 - v_sq);
    if denom < 1e-15 {
        return Ok(f64::INFINITY); // Points on boundary
    }

    let arg = 1.0 + 2.0 * diff_sq / denom;
    Ok(arcosh(arg))
}

/// Squared Poincaré distance (avoids arcosh for comparisons).
pub fn poincare_distance_sq(u: &Array1<f64>, v: &Array1<f64>) -> Result<f64, MathError> {
    let d = poincare_distance(u, v)?;
    Ok(d * d)
}

// ─── Möbius operations (gyrovector space) ───────────────────────────────────

/// Möbius addition: u ⊕ v
///   u ⊕ v = ((1 + 2<u,v> + ||v||²)u + (1 - ||u||²)v) / (1 + 2<u,v> + ||u||²||v||²)
///
/// This is the "addition" in hyperbolic space — not commutative!
pub fn mobius_add(u: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    check_in_ball(u)?;
    check_in_ball(v)?;

    let uv = u.dot(v);
    let u_sq = u.dot(u);
    let v_sq = v.dot(v);

    let denom = 1.0 + 2.0 * uv + u_sq * v_sq;
    if denom.abs() < 1e-15 {
        return Err(MathError::Singular);
    }

    let num = (1.0 + 2.0 * uv + v_sq) * u + (1.0 - u_sq) * v;
    let result = num / denom;

    Ok(project_to_ball(&result))
}

/// Möbius scalar multiplication: r ⊗ x
///   r ⊗ x = tanh(r * artanh(||x||)) * (x / ||x||)
pub fn mobius_scalar_mul(r: f64, x: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    check_in_ball(x)?;

    let norm = x.dot(x).sqrt();
    if norm < 1e-15 {
        return Ok(x.clone()); // Zero vector stays zero
    }

    let scaled_norm = (r * artanh(norm)).tanh();
    Ok(scaled_norm * x / norm)
}

// ─── Exponential and Logarithmic maps ───────────────────────────────────────

/// Exponential map at point p: maps a tangent vector v at p to the Poincaré ball.
///   exp_p(v) = p ⊕ (tanh(λ_p * ||v|| / 2) * v / ||v||)
/// where λ_p = 2 / (1 - ||p||²) is the conformal factor.
pub fn exp_map(p: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    check_in_ball(p)?;

    let p_sq = p.dot(p);
    let lambda = 2.0 / (1.0 - p_sq);
    let v_norm = v.dot(v).sqrt();

    if v_norm < 1e-15 {
        return Ok(p.clone());
    }

    let direction = v / v_norm;
    let scaled = (lambda * v_norm / 2.0).tanh() * &direction;

    mobius_add(p, &scaled)
}

/// Logarithmic map at point p: inverse of exp_map.
///   log_p(x) = (2 / λ_p) * artanh(||-p ⊕ x||) * (-p ⊕ x) / ||-p ⊕ x||
pub fn log_map(p: &Array1<f64>, x: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    check_in_ball(p)?;
    check_in_ball(x)?;

    let p_sq = p.dot(p);
    let lambda = 2.0 / (1.0 - p_sq);

    let neg_p = -1.0 * p;
    let add = mobius_add(&neg_p, x)?;
    let add_norm = add.dot(&add).sqrt();

    if add_norm < 1e-15 {
        return Ok(Array1::zeros(p.len()));
    }

    Ok((2.0 / lambda) * artanh(add_norm) * &add / add_norm)
}

// ─── Riemannian SGD ─────────────────────────────────────────────────────────

/// Riemannian gradient: scale Euclidean gradient by the inverse metric tensor.
///   ∇_R = ((1 - ||θ||²)² / 4) * ∇_E
pub fn riemannian_gradient(params: &Array1<f64>, euclidean_grad: &Array1<f64>) -> Array1<f64> {
    let p_sq = params.dot(params);
    let scale = (1.0 - p_sq).powi(2) / 4.0;
    scale * euclidean_grad
}

/// Riemannian SGD update step.
/// Returns updated parameters projected back into the ball.
pub fn riemannian_sgd_step(
    params: &Array1<f64>,
    euclidean_grad: &Array1<f64>,
    learning_rate: f64,
) -> Array1<f64> {
    let rgrad = riemannian_gradient(params, euclidean_grad);
    let updated = params - &(learning_rate * &rgrad);
    project_to_ball(&updated)
}

/// Full Riemannian SGD update using the exponential map (more precise).
pub fn riemannian_sgd_step_expmap(
    params: &Array1<f64>,
    euclidean_grad: &Array1<f64>,
    learning_rate: f64,
) -> Result<Array1<f64>, MathError> {
    let rgrad = riemannian_gradient(params, euclidean_grad);
    let tangent = -learning_rate * &rgrad;
    exp_map(params, &tangent)
}

// ─── Embeddings ─────────────────────────────────────────────────────────────

/// Initialize embeddings uniformly in the Poincaré ball.
/// Points are placed near the origin (small radius) for stable training.
pub fn init_embeddings(
    n_points: usize,
    dim: usize,
    max_radius: f64,
    seed: u64,
) -> Vec<Array1<f64>> {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let r = max_radius.min(0.999); // Stay inside ball

    (0..n_points)
        .map(|_| {
            let v: Array1<f64> = Array1::from_iter((0..dim).map(|_| rng.random_range(-1.0..1.0)));
            let norm = v.dot(&v).sqrt();
            if norm < 1e-15 {
                Array1::zeros(dim)
            } else {
                let radius = rng.random_range(0.0..r);
                radius * &v / norm
            }
        })
        .collect()
}

/// Compute pairwise distance matrix for embeddings.
pub fn pairwise_distances(embeddings: &[Array1<f64>]) -> ndarray::Array2<f64> {
    let n = embeddings.len();
    let mut dist = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = poincare_distance(&embeddings[i], &embeddings[j]).unwrap_or(f64::INFINITY);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    dist
}

// ─── Utilities ──────────────────────────────────────────────────────────────

/// Project a point back into the open unit ball (||x|| < 1 - eps).
pub fn project_to_ball(x: &Array1<f64>) -> Array1<f64> {
    let norm = x.dot(x).sqrt();
    let max_norm = 1.0 - 1e-5;
    if norm >= max_norm {
        max_norm * x / norm
    } else {
        x.clone()
    }
}

fn check_in_ball(x: &Array1<f64>) -> Result<(), MathError> {
    let norm_sq = x.dot(x);
    if norm_sq >= 1.0 {
        return Err(MathError::InvalidParameter(format!(
            "Point must be inside unit ball (||x||²={:.6} >= 1)",
            norm_sq
        )));
    }
    Ok(())
}

/// arcosh(x) = ln(x + sqrt(x² - 1)), for x >= 1
fn arcosh(x: f64) -> f64 {
    if x < 1.0 {
        0.0
    } else {
        (x + (x * x - 1.0).sqrt()).ln()
    }
}

/// artanh(x) = 0.5 * ln((1+x)/(1-x)), for |x| < 1
fn artanh(x: f64) -> f64 {
    let x = x.clamp(-0.99999, 0.99999);
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

// ─── Geodesics ──────────────────────────────────────────────────────────────

/// Sample points along the geodesic (shortest path) between u and v.
/// Returns `n_points` evenly spaced along the geodesic.
pub fn geodesic(
    u: &Array1<f64>,
    v: &Array1<f64>,
    n_points: usize,
) -> Result<Vec<Array1<f64>>, MathError> {
    check_in_ball(u)?;
    check_in_ball(v)?;

    let log_v = log_map(u, v)?;
    let mut points = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let t = i as f64 / (n_points - 1).max(1) as f64;
        let tangent = t * &log_v;
        points.push(exp_map(u, &tangent)?);
    }

    Ok(points)
}

// ─── Stereographic projection ───────────────────────────────────────────────

/// Map from hyperboloid model (z, x1, ..., xn) to Poincaré ball (y1, ..., yn).
///   y_i = x_i / (z + 1)
pub fn hyperboloid_to_poincare(hyperboloid_point: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    if hyperboloid_point.len() < 2 {
        return Err(MathError::InvalidParameter(
            "Need at least 2 components (z, x...)".into(),
        ));
    }
    let z = hyperboloid_point[0];
    let x = hyperboloid_point.slice(ndarray::s![1..]);
    let denom = z + 1.0;
    if denom.abs() < 1e-15 {
        return Err(MathError::Singular);
    }
    Ok(x.mapv(|xi| xi / denom))
}

/// Map from Poincaré ball (y1, ..., yn) to hyperboloid model (z, x1, ..., xn).
///   z = (1 + ||y||²) / (1 - ||y||²)
///   x_i = 2 * y_i / (1 - ||y||²)
pub fn poincare_to_hyperboloid(ball_point: &Array1<f64>) -> Result<Array1<f64>, MathError> {
    check_in_ball(ball_point)?;
    let y_sq = ball_point.dot(ball_point);
    let denom = 1.0 - y_sq;

    let z = (1.0 + y_sq) / denom;
    let x = 2.0 * ball_point / denom;

    let mut result = Array1::zeros(ball_point.len() + 1);
    result[0] = z;
    result.slice_mut(ndarray::s![1..]).assign(&x);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_poincare_distance_origin() {
        let origin = array![0.0, 0.0];
        let p = array![0.5, 0.0];
        let d = poincare_distance(&origin, &p).unwrap();
        // d(0, x) = 2 * artanh(||x||) = 2 * artanh(0.5) ≈ 1.0986
        assert!((d - 2.0 * artanh(0.5)).abs() < 1e-6, "Got {}", d);
    }

    #[test]
    fn test_poincare_distance_symmetric() {
        let u = array![0.3, 0.2];
        let v = array![-0.1, 0.4];
        let d1 = poincare_distance(&u, &v).unwrap();
        let d2 = poincare_distance(&v, &u).unwrap();
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_poincare_distance_grows_near_boundary() {
        let origin = array![0.0, 0.0];
        let near = array![0.5, 0.0];
        let far = array![0.99, 0.0];

        let d_near = poincare_distance(&origin, &near).unwrap();
        let d_far = poincare_distance(&origin, &far).unwrap();
        assert!(
            d_far > d_near * 3.0,
            "Distance should grow rapidly near boundary"
        );
    }

    #[test]
    fn test_mobius_add_identity() {
        let origin = array![0.0, 0.0];
        let p = array![0.3, 0.4];
        let result = mobius_add(&origin, &p).unwrap();
        assert!((result[0] - 0.3).abs() < 1e-10);
        assert!((result[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let p = array![0.1, 0.2];
        let v = array![0.05, -0.03]; // Tangent vector

        let q = exp_map(&p, &v).unwrap();
        let v_recovered = log_map(&p, &q).unwrap();

        assert!(
            (v[0] - v_recovered[0]).abs() < 1e-6,
            "v0: {} vs {}",
            v[0],
            v_recovered[0]
        );
        assert!(
            (v[1] - v_recovered[1]).abs() < 1e-6,
            "v1: {} vs {}",
            v[1],
            v_recovered[1]
        );
    }

    #[test]
    fn test_riemannian_sgd_stays_in_ball() {
        let params = array![0.9, 0.0]; // Near boundary
        let grad = array![1.0, 0.0]; // Pushing toward boundary

        let updated = riemannian_sgd_step(&params, &grad, 0.1);
        let norm = updated.dot(&updated).sqrt();
        assert!(norm < 1.0, "Must stay in ball, got norm {}", norm);
    }

    #[test]
    fn test_geodesic() {
        let u = array![0.0, 0.0];
        let v = array![0.5, 0.0];
        let points = geodesic(&u, &v, 5).unwrap();

        assert_eq!(points.len(), 5);
        // First point should be near u, last near v
        assert!(poincare_distance(&points[0], &u).unwrap() < 0.01);
        assert!(poincare_distance(&points[4], &v).unwrap() < 0.01);
    }

    #[test]
    fn test_hyperboloid_poincare_roundtrip() {
        let p = array![0.3, -0.2];
        let h = poincare_to_hyperboloid(&p).unwrap();
        let p2 = hyperboloid_to_poincare(&h).unwrap();

        assert!((p[0] - p2[0]).abs() < 1e-10);
        assert!((p[1] - p2[1]).abs() < 1e-10);
    }

    #[test]
    fn test_init_embeddings_in_ball() {
        let embeddings = init_embeddings(100, 10, 0.1, 42);
        assert_eq!(embeddings.len(), 100);
        for emb in &embeddings {
            assert!(emb.dot(emb).sqrt() < 1.0, "Embedding outside ball");
        }
    }

    #[test]
    fn test_outside_ball_rejected() {
        let p = array![1.0, 0.0]; // On boundary
        assert!(poincare_distance(&p, &array![0.0, 0.0]).is_err());
    }
}
