//! Primitive differentiable operations.
//!
//! Day 2 of Week 1 per examples/canonical-showcase/ix-roadmap-plan-v1.md §5.
//! Implements real forward + backward for add, mul, sum, matmul with
//! reverse-mode accumulation on the Wengert tape.
//!
//! Mean and variance are Day 3. FFT is Day 5 behind the `fft-autograd`
//! feature flag.

use crate::tape::{DiffContext, TapeNode, TensorHandle};
use crate::tensor::{Tensor, TensorData};
use crate::{AutogradError, Result};
use ndarray::{Array, ArrayD, Axis, IxDyn};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocate an `ArrayD<f64>` of zeros with the given shape.
pub fn alloc_like(shape: &[usize]) -> ArrayD<f64> {
    Array::zeros(IxDyn(shape))
}

/// Contract `grad` back to `target_shape` by summing over any axes that
/// were expanded by broadcasting during the forward pass.
///
/// Rules (numpy/ndarray-style):
///   1. If `grad` has more leading dims than `target_shape`, sum them away.
///   2. For any axis where `target_shape` is 1 but `grad` is >1, sum that
///      axis while keeping the dim (so the result has the target shape).
fn unbroadcast(mut grad: ArrayD<f64>, target_shape: &[usize]) -> ArrayD<f64> {
    while grad.ndim() > target_shape.len() {
        grad = grad.sum_axis(Axis(0));
    }
    for (i, &t) in target_shape.iter().enumerate() {
        if t == 1 && grad.shape()[i] != 1 {
            let summed = grad.sum_axis(Axis(i));
            grad = summed.insert_axis(Axis(i));
        }
    }
    grad
}

/// Fetch the f64 array view for a handle, returning a clone so we do
/// not hold a reference across tape mutations.
fn get_value(ctx: &DiffContext, handle: TensorHandle) -> Result<ArrayD<f64>> {
    let node = ctx
        .tape
        .get(handle)
        .ok_or(AutogradError::InvalidHandle(handle))?;
    Ok(node.value.as_f64().clone())
}

// ---------------------------------------------------------------------------
// Leaf constructor
// ---------------------------------------------------------------------------

/// Place an input tensor on the tape as a leaf node. Returns its handle.
/// Use this at the start of a forward pass to register inputs.
pub fn input(ctx: &mut DiffContext, tensor: Tensor) -> TensorHandle {
    ctx.tape.push(TapeNode {
        op: "input",
        inputs: Vec::new(),
        value: tensor,
        grad: None,
        saved: None,
    })
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

/// Element-wise addition with numpy-style broadcasting.
pub fn add(ctx: &mut DiffContext, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle> {
    let av = get_value(ctx, a)?;
    let bv = get_value(ctx, b)?;
    let out = &av + &bv; // ndarray broadcasts automatically; errors if incompatible
    let node = TapeNode {
        op: "add",
        inputs: vec![a, b],
        value: Tensor::from_array(out),
        grad: None,
        saved: None,
    };
    Ok(ctx.tape.push(node))
}

fn backward_add(
    ctx: &DiffContext,
    node: &TapeNode,
    grad_out: &ArrayD<f64>,
) -> Result<Vec<(TensorHandle, ArrayD<f64>)>> {
    let a = node.inputs[0];
    let b = node.inputs[1];
    let a_shape = ctx
        .tape
        .get(a)
        .ok_or(AutogradError::InvalidHandle(a))?
        .value
        .shape();
    let b_shape = ctx
        .tape
        .get(b)
        .ok_or(AutogradError::InvalidHandle(b))?
        .value
        .shape();
    let grad_a = unbroadcast(grad_out.clone(), &a_shape);
    let grad_b = unbroadcast(grad_out.clone(), &b_shape);
    Ok(vec![(a, grad_a), (b, grad_b)])
}

// ---------------------------------------------------------------------------
// mul
// ---------------------------------------------------------------------------

/// Element-wise multiplication with broadcasting.
pub fn mul(ctx: &mut DiffContext, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle> {
    let av = get_value(ctx, a)?;
    let bv = get_value(ctx, b)?;
    let out = &av * &bv;
    let node = TapeNode {
        op: "mul",
        inputs: vec![a, b],
        value: Tensor::from_array(out),
        grad: None,
        saved: None,
    };
    Ok(ctx.tape.push(node))
}

fn backward_mul(
    ctx: &DiffContext,
    node: &TapeNode,
    grad_out: &ArrayD<f64>,
) -> Result<Vec<(TensorHandle, ArrayD<f64>)>> {
    let a = node.inputs[0];
    let b = node.inputs[1];
    let av = get_value(ctx, a)?;
    let bv = get_value(ctx, b)?;
    let grad_a_full = grad_out * &bv;
    let grad_b_full = grad_out * &av;
    let grad_a = unbroadcast(grad_a_full, av.shape());
    let grad_b = unbroadcast(grad_b_full, bv.shape());
    Ok(vec![(a, grad_a), (b, grad_b)])
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

/// Sum-reduction over all elements. Output is a rank-0 scalar tensor.
pub fn sum(ctx: &mut DiffContext, a: TensorHandle) -> Result<TensorHandle> {
    let av = get_value(ctx, a)?;
    let total: f64 = av.iter().sum();
    let scalar = Array::from_elem(IxDyn(&[]), total);
    let node = TapeNode {
        op: "sum",
        inputs: vec![a],
        value: Tensor::from_array(scalar),
        grad: None,
        saved: None,
    };
    Ok(ctx.tape.push(node))
}

fn backward_sum(
    ctx: &DiffContext,
    node: &TapeNode,
    grad_out: &ArrayD<f64>,
) -> Result<Vec<(TensorHandle, ArrayD<f64>)>> {
    let a = node.inputs[0];
    let av = get_value(ctx, a)?;
    // grad_out is a scalar (rank-0 array); broadcast to a's shape.
    let scalar = grad_out
        .iter()
        .next()
        .copied()
        .ok_or_else(|| AutogradError::Numerical("sum backward: empty grad_out".into()))?;
    let grad_a = Array::from_elem(av.raw_dim(), scalar);
    Ok(vec![(a, grad_a)])
}

// ---------------------------------------------------------------------------
// matmul (2-D only for Day 2)
// ---------------------------------------------------------------------------

/// Matrix multiplication. 2-D inputs only on Day 2.
/// `a: [m, k]`, `b: [k, n]`, output `[m, n]`.
pub fn matmul(ctx: &mut DiffContext, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle> {
    let av = get_value(ctx, a)?;
    let bv = get_value(ctx, b)?;
    if av.ndim() != 2 || bv.ndim() != 2 {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![0, 0],
            actual: vec![av.ndim(), bv.ndim()],
        });
    }
    let a2 = av
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| AutogradError::Numerical(format!("matmul a reshape: {e}")))?;
    let b2 = bv
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| AutogradError::Numerical(format!("matmul b reshape: {e}")))?;
    let out2 = a2.dot(&b2);
    let out = out2.into_dyn();
    let node = TapeNode {
        op: "matmul",
        inputs: vec![a, b],
        value: Tensor::from_array(out),
        grad: None,
        saved: None,
    };
    Ok(ctx.tape.push(node))
}

fn backward_matmul(
    ctx: &DiffContext,
    node: &TapeNode,
    grad_out: &ArrayD<f64>,
) -> Result<Vec<(TensorHandle, ArrayD<f64>)>> {
    let a = node.inputs[0];
    let b = node.inputs[1];
    let av = get_value(ctx, a)?;
    let bv = get_value(ctx, b)?;
    let a2 = av
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| AutogradError::Numerical(format!("matmul bwd a reshape: {e}")))?;
    let b2 = bv
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| AutogradError::Numerical(format!("matmul bwd b reshape: {e}")))?;
    let g2 = grad_out
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| AutogradError::Numerical(format!("matmul bwd grad reshape: {e}")))?;
    // dL/da = dL/dz @ b^T   shape [m, k]
    let grad_a = g2.dot(&b2.t()).into_dyn();
    // dL/db = a^T @ dL/dz   shape [k, n]
    let grad_b = a2.t().dot(&g2).into_dyn();
    Ok(vec![(a, grad_a), (b, grad_b)])
}

// ---------------------------------------------------------------------------
// Day 3 stubs
// ---------------------------------------------------------------------------

/// Mean reduction. Day 3 — not implemented in Day 2.
pub fn mean(_ctx: &mut DiffContext, _a: TensorHandle) -> Result<TensorHandle> {
    Err(AutogradError::Numerical(
        "ix_autograd::ops::mean is Day 3; not implemented yet".into(),
    ))
}

/// Variance. Day 3 — not implemented in Day 2.
pub fn variance(_ctx: &mut DiffContext, _a: TensorHandle) -> Result<TensorHandle> {
    Err(AutogradError::Numerical(
        "ix_autograd::ops::variance is Day 3; not implemented yet".into(),
    ))
}

// ---------------------------------------------------------------------------
// Backward walker
// ---------------------------------------------------------------------------

impl DiffContext {
    /// Walk the tape in reverse from `output` with a seed gradient of
    /// `seed_grad` (usually a scalar 1.0 for a loss). Returns a map from
    /// every node handle to its accumulated gradient.
    ///
    /// Input leaves (op = "input") do not propagate further. Callers
    /// recover the gradient of any input by indexing into the returned
    /// map with the leaf's handle.
    pub fn backward(
        &mut self,
        output: TensorHandle,
        seed_grad: ArrayD<f64>,
    ) -> Result<std::collections::HashMap<TensorHandle, ArrayD<f64>>> {
        use std::collections::HashMap;
        let mut grads: HashMap<TensorHandle, ArrayD<f64>> = HashMap::new();
        grads.insert(output, seed_grad);

        // Walk nodes in reverse index order. Since the tape is append-only
        // and an op's inputs always have smaller indices, this is a valid
        // reverse topological order.
        let max_idx = output.0;
        for idx in (0..=max_idx).rev() {
            let handle = TensorHandle(idx);
            let grad_out = match grads.get(&handle) {
                Some(g) => g.clone(),
                None => continue, // this node is not on the path to `output`
            };

            // Clone the node so we don't hold a borrow while calling backward.
            let node = self
                .tape
                .get(handle)
                .ok_or(AutogradError::InvalidHandle(handle))?;
            let op = node.op;
            let node_clone = TapeNode {
                op,
                inputs: node.inputs.clone(),
                value: node.value.clone(),
                grad: None,
                saved: node.saved.clone(),
            };

            let input_grads = match op {
                "input" => continue, // leaves terminate the walk
                "add" => backward_add(self, &node_clone, &grad_out)?,
                "mul" => backward_mul(self, &node_clone, &grad_out)?,
                "sum" => backward_sum(self, &node_clone, &grad_out)?,
                "matmul" => backward_matmul(self, &node_clone, &grad_out)?,
                other => {
                    return Err(AutogradError::Numerical(format!(
                        "backward: unknown op `{other}`"
                    )))
                }
            };

            for (in_handle, g) in input_grads {
                grads
                    .entry(in_handle)
                    .and_modify(|existing| *existing += &g)
                    .or_insert(g);
            }
        }

        // Populate node.grad fields for inspection.
        for (&handle, g) in grads.iter() {
            if let Some(node) = self.tape.get_mut(handle) {
                node.grad = Some(Tensor {
                    data: TensorData::F64(g.clone()),
                    requires_grad: false,
                });
            }
        }

        Ok(grads)
    }
}
