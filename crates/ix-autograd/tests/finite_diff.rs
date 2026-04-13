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
// Linear regression end-to-end verification
// ---------------------------------------------------------------------------

#[test]
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
