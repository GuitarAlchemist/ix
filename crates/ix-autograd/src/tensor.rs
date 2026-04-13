//! Tape-aware tensor type backed by `ndarray`.
//!
//! Codex review decision: stay ndarray-native at the boundary and
//! internally. The `TensorData` enum leaves an explicit future door for
//! an alternative backend (candle, burn, custom GPU) without forcing
//! every downstream crate to migrate.

use ndarray::ArrayD;

/// Storage variant for a [`Tensor`]. Today this is always `F64`; the
/// enum exists so a future Candle or GPU backend can be added without
/// changing every downstream tool.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// `ndarray::ArrayD<f64>` — the only variant in the Day 1–3
    /// implementation.
    F64(ArrayD<f64>),
    // Future: Candle(candle_core::Tensor), Gpu(ix_gpu::Buffer), ...
}

/// Tape-aware tensor. Wraps a [`TensorData`] payload plus a
/// `requires_grad` flag. Tools place tensors on the tape via
/// [`crate::ops::input`] at the start of a forward pass.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying numeric payload.
    pub data: TensorData,
    /// Whether gradient tracking should be enabled for this tensor.
    /// Leaves created from trainable parameters should set this to
    /// `true`; leaves from target/observation data should set it to
    /// `false`.
    pub requires_grad: bool,
}

impl Tensor {
    /// Wrap an `ndarray::ArrayD<f64>` in a `Tensor` with gradient
    /// tracking disabled.
    pub fn from_array(array: ArrayD<f64>) -> Self {
        Self {
            data: TensorData::F64(array),
            requires_grad: false,
        }
    }

    /// Wrap an `ndarray::ArrayD<f64>` in a `Tensor` with gradient
    /// tracking enabled.
    pub fn from_array_with_grad(array: ArrayD<f64>) -> Self {
        Self {
            data: TensorData::F64(array),
            requires_grad: true,
        }
    }

    /// Shape of the underlying array.
    pub fn shape(&self) -> Vec<usize> {
        match &self.data {
            TensorData::F64(a) => a.shape().to_vec(),
        }
    }

    /// Borrow the underlying f64 array. Panics if the tensor is
    /// ever stored in a non-F64 variant (not currently possible).
    pub fn as_f64(&self) -> &ArrayD<f64> {
        match &self.data {
            TensorData::F64(a) => a,
        }
    }
}
