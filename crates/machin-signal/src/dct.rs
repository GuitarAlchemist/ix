//! Discrete Cosine Transform (DCT-II / DCT-III).
//! Used in JPEG, MP3, and feature extraction (MFCCs).

use std::f64::consts::PI;

/// DCT-II (the "standard" DCT used in JPEG/MP3).
pub fn dct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    xi * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos()
                })
                .sum::<f64>()
        })
        .collect()
}

/// DCT-III (inverse of DCT-II, up to scaling).
pub fn dct3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            x[0] / 2.0
                + (1..n)
                    .map(|k| {
                        x[k] * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos()
                    })
                    .sum::<f64>()
        })
        .collect()
}

/// Normalized DCT-II (orthonormal).
pub fn dct2_normalized(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mut result = dct2(x);
    result[0] *= (1.0 / n).sqrt();
    for val in result.iter_mut().skip(1) {
        *val *= (2.0 / n).sqrt();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_energy_concentration() {
        // A smooth signal should have most energy in low-frequency DCT coefficients
        let n = 16;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();
        let coeffs = dct2(&signal);

        let low_energy: f64 = coeffs[..4].iter().map(|c| c * c).sum();
        let high_energy: f64 = coeffs[4..].iter().map(|c| c * c).sum();
        assert!(low_energy > high_energy);
    }
}
