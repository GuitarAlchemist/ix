//! CPU nearest-centroid assignment — the k-means assignment step (argmin over the
//! K centroids). This is the correctness **oracle** for the GPU FlashAssign spike
//! (`examples/flash_assign_spike.rs`); it lives in the library (not the example) so
//! `cargo test --workspace` — the gate `scripts/verify.ps1` runs — actually executes
//! its tests. (Example-local `#[cfg(test)]` modules are only built, not run, by the
//! default `cargo test`.)

/// Assign each of `n` points (row-major `points`, `d` dims each) to its nearest of
/// `k` centroids (row-major `centroids`) by squared Euclidean distance. Returns one
/// cluster index per point. Ties take the lowest index (strict `<`).
pub fn nearest_centroid_cpu(
    points: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
) -> Vec<u32> {
    let mut out = vec![0u32; n];
    for (p, slot) in out.iter_mut().enumerate() {
        let pb = p * d;
        let mut best = f32::MAX;
        let mut best_k = 0u32;
        for c in 0..k {
            let cb = c * d;
            let mut dist = 0.0f32;
            for i in 0..d {
                let diff = points[pb + i] - centroids[cb + i];
                dist += diff * diff;
            }
            if dist < best {
                best = dist;
                best_k = c as u32;
            }
        }
        *slot = best_k;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_nearest_centroid() {
        // p0 near c0=(0,0); p1 near c1=(10,10).
        let points = vec![0.1, 0.0, 9.9, 10.0];
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        assert_eq!(nearest_centroid_cpu(&points, &centroids, 2, 2, 2), vec![0, 1]);
    }

    #[test]
    fn ties_take_lowest_index() {
        // Equidistant from both centroids → argmin keeps the first (strict <).
        let points = vec![5.0, 0.0];
        let centroids = vec![0.0, 0.0, 10.0, 0.0];
        assert_eq!(nearest_centroid_cpu(&points, &centroids, 1, 2, 2), vec![0]);
    }
}
