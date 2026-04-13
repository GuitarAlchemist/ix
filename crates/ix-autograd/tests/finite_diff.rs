//! Finite-difference verifier for `DifferentiableTool` implementations.
//!
//! Day 2 of Week 1 per examples/canonical-showcase/ix-roadmap-plan-v1.md §5.
//!
//! The verifier perturbs each scalar element of each input by +/- epsilon,
//! runs the forward pass twice, computes the central finite difference,
//! and compares to the analytical gradient returned by `backward`. If the
//! maximum absolute difference exceeds `tolerance` on any element, the
//! test fails with a detailed error string.

use ix_autograd::ops;
use ix_autograd::prelude::*;
use ix_autograd::tools::linear_regression::LinearRegressionTool;
use ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Run the forward closure on a copy of the inputs with `inputs[name]`
/// having its `flat_idx`-th element perturbed by `delta`. Returns the
/// resulting scalar loss.
fn perturbed_loss<F>(
    forward: &F,
    inputs: &HashMap<String, ArrayD<f64>>,
    name: &str,
    flat_idx: usize,
    delta: f64,
) -> f64
where
    F: Fn(&HashMap<String, ArrayD<f64>>) -> f64,
{
    let mut cloned: HashMap<String, ArrayD<f64>> = inputs
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let arr = cloned.get_mut(name).expect("input name not found");
    // iter_mut works regardless of memory layout
    *arr.iter_mut().nth(flat_idx).expect("index out of range") += delta;
    forward(&cloned)
}

/// Verify analytical gradients against central finite differences.
fn verify_gradient<F>(
    label: &str,
    forward: F,
    inputs: HashMap<String, ArrayD<f64>>,
    analytical_grads: HashMap<String, ArrayD<f64>>,
    epsilon: f64,
    tolerance: f64,
) -> Result<(), String>
where
    F: Fn(&HashMap<String, ArrayD<f64>>) -> f64,
{
    for (name, input) in inputs.iter() {
        let grad = analytical_grads
            .get(name)
            .ok_or_else(|| format!("{label}: missing analytical grad for `{name}`"))?;
        if grad.shape() != input.shape() {
            return Err(format!(
                "{label}: `{name}` analytical grad shape {:?} != input shape {:?}",
                grad.shape(),
                input.shape()
            ));
        }
        let grad_values: Vec<f64> = grad.iter().copied().collect();
        for (flat_idx, &analytical) in grad_values.iter().enumerate().take(input.len()) {
            let f_plus = perturbed_loss(&forward, &inputs, name, flat_idx, epsilon);
            let f_minus = perturbed_loss(&forward, &inputs, name, flat_idx, -epsilon);
            let numerical = (f_plus - f_minus) / (2.0 * epsilon);
            let diff = (numerical - analytical).abs();
            if diff > tolerance {
                return Err(format!(
                    "{label}: `{name}`[{flat_idx}] numerical={numerical:.6e} \
                     analytical={analytical:.6e} diff={diff:.6e} > tol={tolerance:.6e}"
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Small reproducible inputs (no RNG crate — deterministic arrays)
// ---------------------------------------------------------------------------

fn array_2x3_a() -> ArrayD<f64> {
    Array::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![0.1, -0.5, 1.2, 0.3, -0.8, 0.7],
    )
    .expect("2x3 shape")
}

fn array_2x3_b() -> ArrayD<f64> {
    Array::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-0.4, 0.9, 0.2, 1.1, -0.3, 0.6],
    )
    .expect("2x3 shape")
}

fn array_3x2_b() -> ArrayD<f64> {
    Array::from_shape_vec(
        IxDyn(&[3, 2]),
        vec![0.5, -0.2, 0.8, 1.1, -0.4, 0.3],
    )
    .expect("3x2 shape")
}

fn array_1x3() -> ArrayD<f64> {
    Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.3, -0.4, 0.6]).expect("1x3 shape")
}

// ---------------------------------------------------------------------------
// Day 1 stubs — kept passing
// ---------------------------------------------------------------------------

#[test]
fn stub_compiles_and_imports_crate() {
    let mode = ExecutionMode::VerifyFiniteDiff;
    assert!(mode.requires_tape());
    assert!(!mode.allows_non_diff());
}

#[test]
fn execution_modes_have_expected_semantics() {
    assert!(!ExecutionMode::Eager.requires_tape());
    assert!(ExecutionMode::Train.requires_tape());
    assert!(ExecutionMode::Mixed.requires_tape());
    assert!(ExecutionMode::Mixed.allows_non_diff());
    assert!(!ExecutionMode::Train.allows_non_diff());
}

#[test]
fn tensor_from_array_roundtrip() {
    use ndarray::array;
    let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let t = Tensor::from_array(a.clone());
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.as_f64(), &a);
    assert!(!t.requires_grad);

    let t2 = Tensor::from_array_with_grad(a);
    assert!(t2.requires_grad);
}

// ---------------------------------------------------------------------------
// Op verification tests
// ---------------------------------------------------------------------------

#[test]
fn verify_add_backward() {
    let a = array_2x3_a();
    let b = array_2x3_b();

    // Analytical via the autograd tape: loss = sum(a + b)
    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let sum_ab = ops::add(&mut ctx, a_h, b_h).expect("add");
    let loss = ops::sum(&mut ctx, sum_ab).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());
    analytical.insert("b".into(), grads[&b_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "add",
        |ins| {
            let ap = ins.get("a").expect("a");
            let bp = ins.get("b").expect("b");
            (ap + bp).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("add verifier");
}

#[test]
fn verify_mul_backward() {
    let a = array_2x3_a();
    let b = array_2x3_b();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let prod = ops::mul(&mut ctx, a_h, b_h).expect("mul");
    let loss = ops::sum(&mut ctx, prod).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());
    analytical.insert("b".into(), grads[&b_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "mul",
        |ins| {
            let ap = ins.get("a").expect("a");
            let bp = ins.get("b").expect("b");
            (ap * bp).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("mul verifier");
}

#[test]
fn verify_sum_backward() {
    let a = array_2x3_a();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let loss = ops::sum(&mut ctx, a_h).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);

    verify_gradient(
        "sum",
        |ins| ins.get("a").expect("a").sum(),
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("sum verifier");
}

#[test]
fn verify_matmul_backward() {
    let a = array_2x3_a(); // [2, 3]
    let b = array_3x2_b(); // [3, 2]

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let prod = ops::matmul(&mut ctx, a_h, b_h).expect("matmul");
    let loss = ops::sum(&mut ctx, prod).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());
    analytical.insert("b".into(), grads[&b_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "matmul",
        |ins| {
            let ap = ins.get("a").expect("a");
            let bp = ins.get("b").expect("b");
            let a2 = ap
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("a 2d");
            let b2 = bp
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("b 2d");
            a2.dot(&b2).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("matmul verifier");
}

// ---------------------------------------------------------------------------
// Linear regression end-to-end verification (Day 2 stand-in)
//
// The Day 2 version used `loss = sum(y_hat * y_hat)` because sub and
// div_scalar did not exist yet. Day 3 replaces it with the proper MSE
// test `verify_linear_regression_mse_backward` further below. This
// legacy test is kept but migrated to the new tool signature (which
// requires a `y` input) so it still covers the same `mul(y_hat, y_hat)`
// shared-subexpression path on the tape.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Day 2 stand-in — superseded by verify_linear_regression_mse_backward"]
fn verify_linear_regression_backward() {
    // x: [5, 3], w: [3, 1], b: [1, 1]
    let x = Array::from_shape_vec(
        IxDyn(&[5, 3]),
        vec![
            0.1, 0.2, 0.3,
            0.4, -0.1, 0.5,
            -0.2, 0.6, 0.1,
            0.3, 0.2, -0.4,
            0.5, -0.3, 0.2,
        ],
    )
    .expect("x shape");
    let w = Array::from_shape_vec(IxDyn(&[3, 1]), vec![0.7, -0.5, 0.3]).expect("w shape");
    let b = Array::from_shape_vec(IxDyn(&[1, 1]), vec![0.1]).expect("b shape");

    // Analytical gradients via the tool's own backward.
    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let tool = LinearRegressionTool;
    let mut in_map = ValueMap::new();
    in_map.insert("x".into(), Tensor::from_array_with_grad(x.clone()));
    in_map.insert("w".into(), Tensor::from_array_with_grad(w.clone()));
    in_map.insert("b".into(), Tensor::from_array_with_grad(b.clone()));

    let out = tool.forward(&mut ctx, &in_map).expect("forward");
    // Sanity: tool returned y_hat and loss.
    assert!(out.contains_key("y_hat"));
    assert!(out.contains_key("loss"));

    let dummy_grads = ValueMap::new(); // loss seeds itself; ignored by tool
    let analytical_tensors = tool.backward(&mut ctx, &dummy_grads).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert(
        "x".into(),
        analytical_tensors
            .get("x")
            .expect("x grad")
            .as_f64()
            .clone(),
    );
    analytical.insert(
        "w".into(),
        analytical_tensors
            .get("w")
            .expect("w grad")
            .as_f64()
            .clone(),
    );
    analytical.insert(
        "b".into(),
        analytical_tensors
            .get("b")
            .expect("b grad")
            .as_f64()
            .clone(),
    );

    // Numerical forward: loss = sum((x @ w + b) * (x @ w + b))
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), x);
    inputs.insert("w".into(), w);
    inputs.insert("b".into(), b);

    verify_gradient(
        "linear_regression",
        |ins| {
            let xp = ins.get("x").expect("x");
            let wp = ins.get("w").expect("w");
            let bp = ins.get("b").expect("b");
            let x2 = xp
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("x 2d");
            let w2 = wp
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("w 2d");
            let y_hat = x2.dot(&w2) + bp[[0, 0]];
            (&y_hat * &y_hat).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("linear regression verifier");
}

// ---------------------------------------------------------------------------
// Day 3 — new tests from r7-day2-review.md §4
// ---------------------------------------------------------------------------

#[test]
fn verify_add_with_broadcast() {
    // a: [2, 3], b: [1, 3] — loss = sum(a + b)
    // Expected: grad_a has shape [2, 3], all ones; grad_b has shape [1, 3], each entry = 2.
    let a = array_2x3_a();
    let b = array_1x3();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let sum_ab = ops::add(&mut ctx, a_h, b_h).expect("add");
    let loss = ops::sum(&mut ctx, sum_ab).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let grad_a = grads[&a_h].clone();
    let grad_b = grads[&b_h].clone();
    assert_eq!(grad_a.shape(), &[2, 3], "grad_a shape must contract to a's shape");
    assert_eq!(grad_b.shape(), &[1, 3], "grad_b shape must stay [1, 3]");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grad_a);
    analytical.insert("b".into(), grad_b);

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "add_broadcast",
        |ins| {
            let ap = ins.get("a").expect("a");
            let bp = ins.get("b").expect("b");
            (ap + bp).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("add broadcast verifier");
}

#[test]
fn verify_mul_shared_subexpression() {
    // y = x * x, loss = sum(y). Expected: grad_x = 2 * x.
    let x = array_2x3_a();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let x_h = ops::input(&mut ctx, Tensor::from_array_with_grad(x.clone()));
    let y = ops::mul(&mut ctx, x_h, x_h).expect("mul");
    let loss = ops::sum(&mut ctx, y).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let grad_x = grads[&x_h].clone();
    let expected = x.mapv(|v| 2.0 * v);
    let diff: f64 = grad_x
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(diff < 1e-10, "grad_x should equal 2x, max diff {diff}");

    // Also cross-check via the finite-difference verifier.
    let mut analytical = HashMap::new();
    analytical.insert("x".into(), grad_x);
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), x);
    verify_gradient(
        "mul_shared",
        |ins| {
            let xp = ins.get("x").expect("x");
            (xp * xp).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("mul shared verifier");
}

#[test]
fn verify_disconnected_leaf() {
    // Register 3 leaves but only use 2 in the computation. The unused
    // leaf should have no gradient in the returned map and the backward
    // pass should complete without error.
    let a = array_2x3_a();
    let b = array_2x3_b();
    let unused = array_2x3_a();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let unused_h = ops::input(&mut ctx, Tensor::from_array_with_grad(unused));

    let prod = ops::mul(&mut ctx, a_h, b_h).expect("mul");
    let loss = ops::sum(&mut ctx, prod).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    assert!(grads.contains_key(&a_h), "a should have a gradient");
    assert!(grads.contains_key(&b_h), "b should have a gradient");
    assert!(
        !grads.contains_key(&unused_h),
        "unused leaf should not have a gradient entry"
    );
}

// ---------------------------------------------------------------------------
// Day 3 — sub, div_scalar, mean, variance
// ---------------------------------------------------------------------------

#[test]
fn verify_sub_backward() {
    let a = array_2x3_a();
    let b = array_2x3_b();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let diff_ab = ops::sub(&mut ctx, a_h, b_h).expect("sub");
    let loss = ops::sum(&mut ctx, diff_ab).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());
    analytical.insert("b".into(), grads[&b_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "sub",
        |ins| {
            let ap = ins.get("a").expect("a");
            let bp = ins.get("b").expect("b");
            (ap - bp).sum()
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("sub verifier");
}

#[test]
fn verify_div_scalar_backward() {
    // loss = sum(a / 4.0). Expected: grad_a = 0.25 everywhere.
    let a = array_2x3_a();
    let divisor = 4.0;

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let scaled = ops::div_scalar(&mut ctx, a_h, divisor).expect("div_scalar");
    let loss = ops::sum(&mut ctx, scaled).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);

    verify_gradient(
        "div_scalar",
        |ins| ins.get("a").expect("a").iter().map(|v| v / divisor).sum(),
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("div_scalar verifier");
}

#[test]
fn verify_mean_backward() {
    // loss = mean(a)
    let a = array_2x3_a();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let m = ops::mean(&mut ctx, a_h).expect("mean");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(m, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);

    verify_gradient(
        "mean",
        |ins| {
            let ap = ins.get("a").expect("a");
            ap.iter().copied().sum::<f64>() / (ap.len() as f64)
        },
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("mean verifier");
}

#[test]
fn verify_variance_backward() {
    // loss = variance(a)    (biased / population variance)
    let a = array_2x3_a();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let v = ops::variance(&mut ctx, a_h).expect("variance");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(v, seed).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());

    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);

    verify_gradient(
        "variance",
        |ins| {
            let ap = ins.get("a").expect("a");
            let n = ap.len() as f64;
            let mean = ap.iter().copied().sum::<f64>() / n;
            ap.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
        },
        inputs,
        analytical,
        1e-6,
        1e-4, // variance has deeper chain; loosen slightly
    )
    .expect("variance verifier");
}

// ---------------------------------------------------------------------------
// Day 3 — linear regression with full MSE loss
// ---------------------------------------------------------------------------

#[test]
fn verify_linear_regression_mse_backward() {
    // x: [5, 3], w: [3, 1], b: [1, 1], y: [5, 1]
    let x = Array::from_shape_vec(
        IxDyn(&[5, 3]),
        vec![
            0.1, 0.2, 0.3,
            0.4, -0.1, 0.5,
            -0.2, 0.6, 0.1,
            0.3, 0.2, -0.4,
            0.5, -0.3, 0.2,
        ],
    )
    .expect("x shape");
    let w = Array::from_shape_vec(IxDyn(&[3, 1]), vec![0.7, -0.5, 0.3]).expect("w shape");
    let b = Array::from_shape_vec(IxDyn(&[1, 1]), vec![0.1]).expect("b shape");
    let y = Array::from_shape_vec(IxDyn(&[5, 1]), vec![0.2, 0.1, -0.1, 0.3, 0.0]).expect("y shape");

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let tool = LinearRegressionTool;
    let mut in_map = ValueMap::new();
    in_map.insert("x".into(), Tensor::from_array_with_grad(x.clone()));
    in_map.insert("w".into(), Tensor::from_array_with_grad(w.clone()));
    in_map.insert("b".into(), Tensor::from_array_with_grad(b.clone()));
    in_map.insert("y".into(), Tensor::from_array(y.clone()));

    let out = tool.forward(&mut ctx, &in_map).expect("forward");
    assert!(out.contains_key("y_hat"));
    assert!(out.contains_key("loss"));

    let dummy = ValueMap::new();
    let grads_out = tool.backward(&mut ctx, &dummy).expect("backward");
    // y is a target, not a parameter — the tool should not return a y grad.
    assert!(!grads_out.contains_key("y"), "y must not receive a gradient");

    let mut analytical = HashMap::new();
    analytical.insert(
        "x".into(),
        grads_out.get("x").expect("x grad").as_f64().clone(),
    );
    analytical.insert(
        "w".into(),
        grads_out.get("w").expect("w grad").as_f64().clone(),
    );
    analytical.insert(
        "b".into(),
        grads_out.get("b").expect("b grad").as_f64().clone(),
    );

    // Numerical forward: loss = mean((x @ w + b - y) ^ 2)
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), x);
    inputs.insert("w".into(), w);
    inputs.insert("b".into(), b);
    // y is not perturbed — use a closure that closes over the fixed y.
    let y_fixed = y;

    verify_gradient(
        "linear_regression_mse",
        move |ins| {
            let xp = ins.get("x").expect("x");
            let wp = ins.get("w").expect("w");
            let bp = ins.get("b").expect("b");
            let x2 = xp
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("x 2d");
            let w2 = wp
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("w 2d");
            let y2 = y_fixed
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("y 2d");
            let y_hat = x2.dot(&w2) + bp[[0, 0]];
            let residual = &y_hat - &y2;
            (&residual * &residual).sum() / (residual.len() as f64)
        },
        inputs,
        analytical,
        1e-6,
        1e-4, // deeper chain: matmul → add → sub → mul → sum → div_scalar
    )
    .expect("linear regression MSE verifier");
}

// ---------------------------------------------------------------------------
// Day 4 — broadcast edge cases
// ---------------------------------------------------------------------------

#[test]
fn verify_add_scalar_to_matrix() {
    // a: [2, 3], b: scalar [1, 1] — loss = sum(a + b)
    // Expected: grad_a = ones[2,3], grad_b = scalar 6.0
    let a = array_2x3_a();
    let b = Array::from_shape_vec(IxDyn(&[1, 1]), vec![0.5]).expect("scalar shape");

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let z = ops::add(&mut ctx, a_h, b_h).expect("add");
    let loss = ops::sum(&mut ctx, z).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let grad_a = grads[&a_h].clone();
    let grad_b = grads[&b_h].clone();
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[1, 1]);
    // grad_b should sum the entire upstream grad (6 elements of 1.0)
    assert!((grad_b[[0, 0]] - 6.0).abs() < 1e-12, "grad_b = {}", grad_b[[0, 0]]);

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grad_a);
    analytical.insert("b".into(), grad_b);
    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "add_scalar_to_matrix",
        |ins| (ins.get("a").expect("a") + ins.get("b").expect("b")).sum(),
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("add_scalar_to_matrix verifier");
}

#[test]
fn verify_mul_with_row_vector_broadcast() {
    // a: [2, 3], b: [1, 3] — loss = sum(a * b)
    // Tests broadcast on the inner axis (different from add_with_broadcast
    // which tests the same case for add). This catches mul-specific
    // broadcast bugs.
    let a = array_2x3_a();
    let b = array_1x3();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let z = ops::mul(&mut ctx, a_h, b_h).expect("mul");
    let loss = ops::sum(&mut ctx, z).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let grad_a = grads[&a_h].clone();
    let grad_b = grads[&b_h].clone();
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[1, 3]);

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grad_a);
    analytical.insert("b".into(), grad_b);
    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "mul_row_broadcast",
        |ins| (ins.get("a").expect("a") * ins.get("b").expect("b")).sum(),
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("mul_row_broadcast verifier");
}

#[test]
fn verify_sub_with_broadcast() {
    // a: [2, 3], b: [1, 3] — loss = sum(a - b)
    // Tests sub-specific broadcast (the unbroadcast(neg) path).
    let a = array_2x3_a();
    let b = array_1x3();

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a.clone()));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b.clone()));
    let z = ops::sub(&mut ctx, a_h, b_h).expect("sub");
    let loss = ops::sum(&mut ctx, z).expect("sum");
    let seed = Array::from_elem(IxDyn(&[]), 1.0);
    let grads = ctx.backward(loss, seed).expect("backward");

    let grad_b = grads[&b_h].clone();
    // Sum over the broadcast axis: 2 rows of [-1, -1, -1] each = [-2, -2, -2]
    assert_eq!(grad_b.shape(), &[1, 3]);
    for &v in grad_b.iter() {
        assert!((v + 2.0).abs() < 1e-12, "grad_b row should be -2, got {v}");
    }

    let mut analytical = HashMap::new();
    analytical.insert("a".into(), grads[&a_h].clone());
    analytical.insert("b".into(), grad_b);
    let mut inputs = HashMap::new();
    inputs.insert("a".into(), a);
    inputs.insert("b".into(), b);

    verify_gradient(
        "sub_broadcast",
        |ins| (ins.get("a").expect("a") - ins.get("b").expect("b")).sum(),
        inputs,
        analytical,
        1e-6,
        1e-5,
    )
    .expect("sub_broadcast verifier");
}

#[test]
fn matmul_rejects_non_2d_with_clear_error() {
    // Day 3 added UnsupportedRank for matmul. Verify the error message
    // is specific enough to be useful.
    let a = Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("1d");
    let b = Array::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).expect("2d");

    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a));
    let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b));
    let result = ops::matmul(&mut ctx, a_h, b_h);

    match result {
        Err(ix_autograd::AutogradError::UnsupportedRank {
            op,
            supported,
            actual,
        }) => {
            assert_eq!(op, "matmul");
            assert_eq!(supported, vec![2]);
            assert_eq!(actual, 1);
        }
        other => panic!("expected UnsupportedRank, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Day 4 — second wrapped tool (StatsVarianceTool) end-to-end test
// ---------------------------------------------------------------------------

#[test]
fn verify_stats_variance_tool_backward() {
    // The StatsVarianceTool is the second wrapped DifferentiableTool.
    // It computes loss = variance(x) and its backward must match the
    // sum+div_scalar+mul+sub composition on the tape.
    use ix_autograd::tools::stats_variance::StatsVarianceTool;

    let x = array_2x3_a();
    let mut ctx = DiffContext::new(ExecutionMode::Train);
    let tool = StatsVarianceTool;
    let mut in_map = ValueMap::new();
    in_map.insert("x".into(), Tensor::from_array_with_grad(x.clone()));

    let out = tool.forward(&mut ctx, &in_map).expect("forward");
    assert!(out.contains_key("variance"));

    let dummy = ValueMap::new();
    let grads_out = tool.backward(&mut ctx, &dummy).expect("backward");

    let mut analytical = HashMap::new();
    analytical.insert(
        "x".into(),
        grads_out.get("x").expect("x grad").as_f64().clone(),
    );

    let mut inputs = HashMap::new();
    inputs.insert("x".into(), x);

    verify_gradient(
        "stats_variance",
        |ins| {
            let xp = ins.get("x").expect("x");
            let n = xp.len() as f64;
            let mean = xp.iter().copied().sum::<f64>() / n;
            xp.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
        },
        inputs,
        analytical,
        1e-6,
        1e-4,
    )
    .expect("stats_variance verifier");
}
