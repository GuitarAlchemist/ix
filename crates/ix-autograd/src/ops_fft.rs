//! Day 5 — Real FFT magnitude forward + backward.
//!
//! Gated behind the `fft-autograd` feature flag. Off by default.
//!
//! # Convention note
//!
//! `ix-signal::fft::rfft` is a thin wrapper around the full complex
//! FFT — it accepts a real signal and returns the full N-length
//! complex spectrum, NOT the N/2+1-length Hermitian-truncated
//! spectrum that the term "rfft" usually denotes elsewhere
//! (numpy/JAX). This means we do NOT need to apply the Hermitian
//! mirror rule from the Codex review here — the spectrum we get
//! back already contains both halves explicitly. The mirroring
//! convention will only become relevant if/when ix-signal grows a
//! true half-spectrum rfft, at which point this op gets a sibling.
//!
//! # Forward
//!
//! `rfft_magnitude(x)`:
//!   x ∈ R^N (power-of-2 length)
//!   Y = FFT(x) ∈ C^N
//!   output = |Y| ∈ R^N    (N-length magnitude vector)
//!
//! # Backward
//!
//! Chain rule on `mag[k] = sqrt(Re(Y[k])^2 + Im(Y[k])^2)`:
//!     dL/dRe(Y[k]) = g_mag[k] * Re(Y[k]) / |Y[k]|
//!     dL/dIm(Y[k]) = g_mag[k] * Im(Y[k]) / |Y[k]|
//! Combine into a complex gradient
//!     grad_Y[k] = (g_mag[k] / |Y[k]|) * Y[k]
//!
//! Then propagate grad_Y back through the FFT kernel. Because the
//! FFT is linear, `dL/dx = Re( IFFT(grad_Y) * N )` for the
//! unnormalized convention used by `ix-signal` (where
//! `ifft(fft(x)) == x`). The N factor and the `Re()` projection
//! together encode the "real input" constraint.
//!
//! See the verifier test in `tests/finite_diff.rs` for the
//! ground truth check via central finite differences.

use crate::tape::{DiffContext, TapeNode, TensorHandle};
use crate::tensor::Tensor;
use crate::{AutogradError, Result};
use ix_signal::fft::{self, Complex};
use ndarray::{Array, ArrayD, IxDyn};

/// Forward `rfft_magnitude(x)`.
///
/// `x` must be a 1-D real tensor whose length is a power of 2 and at
/// least 2. The output is the magnitude `|FFT(x)|`, also 1-D and the
/// same length as `x`.
pub fn rfft_magnitude(ctx: &mut DiffContext, x: TensorHandle) -> Result<TensorHandle> {
    let xv = ctx
        .tape
        .get(x)
        .ok_or(AutogradError::InvalidHandle(x))?
        .value
        .as_f64()
        .clone();
    if xv.ndim() != 1 {
        return Err(AutogradError::UnsupportedRank {
            op: "rfft_magnitude",
            supported: vec![1],
            actual: xv.ndim(),
        });
    }
    let n = xv.len();
    if !n.is_power_of_two() || n < 2 {
        return Err(AutogradError::Numerical(format!(
            "rfft_magnitude: input length must be a power of 2 ≥ 2, got {n}"
        )));
    }
    let signal: Vec<f64> = xv.iter().copied().collect();
    let spectrum: Vec<Complex> = fft::rfft(&signal);
    debug_assert_eq!(spectrum.len(), n);

    let mags: Vec<f64> = spectrum
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();
    let mag_array = Array::from_shape_vec(IxDyn(&[n]), mags)
        .map_err(|e| AutogradError::Numerical(format!("rfft_magnitude: shape error: {e}")))?;

    // Stash the complex spectrum so backward can recover Re(Y) and Im(Y).
    let saved = serde_json::json!({
        "n": n,
        "real": spectrum.iter().map(|c| c.re).collect::<Vec<f64>>(),
        "imag": spectrum.iter().map(|c| c.im).collect::<Vec<f64>>(),
    });

    let node = TapeNode {
        op: "rfft_magnitude",
        inputs: vec![x],
        value: Tensor::from_array(mag_array),
        grad: None,
        saved: Some(saved),
    };
    Ok(ctx.tape.push(node))
}

pub(crate) fn backward_rfft_magnitude(
    _ctx: &DiffContext,
    node: &TapeNode,
    grad_out: &ArrayD<f64>,
) -> Result<Vec<(TensorHandle, ArrayD<f64>)>> {
    let a = node.inputs[0];
    let saved = node
        .saved
        .as_ref()
        .ok_or_else(|| AutogradError::MissingSaved("rfft_magnitude.spectrum".into()))?;
    let n = saved
        .get("n")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| AutogradError::MissingSaved("rfft_magnitude.n".into()))?
        as usize;
    let real: Vec<f64> = saved
        .get("real")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AutogradError::MissingSaved("rfft_magnitude.real".into()))?
        .iter()
        .filter_map(|x| x.as_f64())
        .collect();
    let imag: Vec<f64> = saved
        .get("imag")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AutogradError::MissingSaved("rfft_magnitude.imag".into()))?
        .iter()
        .filter_map(|x| x.as_f64())
        .collect();
    if real.len() != n || imag.len() != n {
        return Err(AutogradError::Numerical(format!(
            "rfft_magnitude backward: spectrum length mismatch (got {}, expected {})",
            real.len(),
            n
        )));
    }
    let g_mag: Vec<f64> = grad_out.iter().copied().collect();
    if g_mag.len() != n {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![n],
            actual: vec![g_mag.len()],
        });
    }

    // Step 1: complex gradient on the spectrum.
    // grad_Y[k] = (g_mag[k] / |Y[k]|) * Y[k]
    // (zero at bins where the magnitude is exactly zero — magnitude
    // is non-differentiable there; standard convention.)
    let mut grad_y: Vec<Complex> = Vec::with_capacity(n);
    for k in 0..n {
        let mag = (real[k] * real[k] + imag[k] * imag[k]).sqrt();
        if mag < 1e-15 {
            grad_y.push(Complex::new(0.0, 0.0));
        } else {
            let scale = g_mag[k] / mag;
            grad_y.push(Complex::new(real[k] * scale, imag[k] * scale));
        }
    }

    // Step 2: propagate through FFT.
    //
    // Derivation: for a linear map Y = FFT(x) with x real, and
    // f(x) = g(|Y|), the gradient is
    //     dL/dx[n] = N * Re( ifft(grad_y)[n] )
    // where the N factor undoes the 1/N normalization that
    // `ix-signal::fft::ifft` applies internally. Empirically
    // verified against central finite differences in the
    // verify_rfft_magnitude_backward test.
    let inv: Vec<Complex> = fft::ifft(&grad_y);
    let n_f = n as f64;
    let real_grad: Vec<f64> = inv.iter().map(|c| c.re * n_f).collect();
    let grad_a = Array::from_shape_vec(IxDyn(&[n]), real_grad)
        .map_err(|e| AutogradError::Numerical(format!("ifft reshape: {e}")))?;
    Ok(vec![(a, grad_a)])
}
