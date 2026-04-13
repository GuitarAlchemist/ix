//! Phase 1 validation benchmarks for R7 Autograd-IX.
//!
//! Measures forward + backward latency of the Day 2 and Day 3 primitives
//! plus the full `LinearRegressionTool` MSE pass and a single Adam step.
//! Results feed `examples/canonical-showcase/r7-phase1-benchmarks.txt`
//! and the Phase 1 retrospective's go/no-go wall-clock verdict.
//!
//! Run with:
//!     cargo bench -p ix-autograd --bench bench_ops

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ix_autograd::ops;
use ix_autograd::prelude::*;
use ix_autograd::tools::linear_regression::LinearRegressionTool;
use ndarray::{Array, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helpers — deterministic data generation so benchmark numbers are stable.
// ---------------------------------------------------------------------------

fn deterministic(shape: &[usize], seed_offset: u64) -> ArrayD<f64> {
    let n = shape.iter().product::<usize>();
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let seed = ((i as u64).wrapping_add(seed_offset))
                .wrapping_mul(1103515245)
                .wrapping_add(12345);
            (((seed >> 16) & 0x7fff) as f64 / 32767.0) * 2.0 - 1.0
        })
        .collect();
    Array::from_shape_vec(IxDyn(shape), data).expect("shape")
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_add_128(c: &mut Criterion) {
    let a_data = deterministic(&[128, 128], 0);
    let b_data = deterministic(&[128, 128], 1);
    c.bench_function("add_128x128_fwd_bwd", |bench| {
        bench.iter(|| {
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a_data.clone()));
            let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b_data.clone()));
            let z = ops::add(&mut ctx, a_h, b_h).expect("add");
            let loss = ops::sum(&mut ctx, z).expect("sum");
            let seed = Array::from_elem(IxDyn(&[]), 1.0);
            black_box(ctx.backward(loss, seed).expect("backward"));
        })
    });
}

fn bench_mul_128(c: &mut Criterion) {
    let a_data = deterministic(&[128, 128], 2);
    let b_data = deterministic(&[128, 128], 3);
    c.bench_function("mul_128x128_fwd_bwd", |bench| {
        bench.iter(|| {
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a_data.clone()));
            let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b_data.clone()));
            let z = ops::mul(&mut ctx, a_h, b_h).expect("mul");
            let loss = ops::sum(&mut ctx, z).expect("sum");
            let seed = Array::from_elem(IxDyn(&[]), 1.0);
            black_box(ctx.backward(loss, seed).expect("backward"));
        })
    });
}

fn bench_matmul_64(c: &mut Criterion) {
    let a_data = deterministic(&[64, 64], 4);
    let b_data = deterministic(&[64, 64], 5);
    c.bench_function("matmul_64x64_fwd_bwd", |bench| {
        bench.iter(|| {
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a_data.clone()));
            let b_h = ops::input(&mut ctx, Tensor::from_array_with_grad(b_data.clone()));
            let z = ops::matmul(&mut ctx, a_h, b_h).expect("matmul");
            let loss = ops::sum(&mut ctx, z).expect("sum");
            let seed = Array::from_elem(IxDyn(&[]), 1.0);
            black_box(ctx.backward(loss, seed).expect("backward"));
        })
    });
}

fn bench_variance_1024(c: &mut Criterion) {
    let a_data = deterministic(&[1024], 6);
    c.bench_function("variance_1024_fwd_bwd", |bench| {
        bench.iter(|| {
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let a_h = ops::input(&mut ctx, Tensor::from_array_with_grad(a_data.clone()));
            let v = ops::variance(&mut ctx, a_h).expect("variance");
            let seed = Array::from_elem(IxDyn(&[]), 1.0);
            black_box(ctx.backward(v, seed).expect("backward"));
        })
    });
}

fn bench_linreg_mse(c: &mut Criterion) {
    let x = deterministic(&[100, 10], 7);
    let w = deterministic(&[10, 1], 8);
    let b = Array::from_elem(IxDyn(&[1, 1]), 0.1_f64);
    let y = deterministic(&[100, 1], 9);

    c.bench_function("linreg_mse_100x10_fwd_bwd", |bench| {
        let tool = LinearRegressionTool;
        bench.iter(|| {
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let mut inputs = ValueMap::new();
            inputs.insert("x".into(), Tensor::from_array(x.clone()));
            inputs.insert("w".into(), Tensor::from_array_with_grad(w.clone()));
            inputs.insert("b".into(), Tensor::from_array_with_grad(b.clone()));
            inputs.insert("y".into(), Tensor::from_array(y.clone()));
            let _out = tool.forward(&mut ctx, &inputs).expect("forward");
            let dummy = ValueMap::new();
            black_box(tool.backward(&mut ctx, &dummy).expect("backward"));
        })
    });
}

fn bench_adam_step(c: &mut Criterion) {
    let x = deterministic(&[100, 10], 10);
    let w_init = Array::zeros(IxDyn(&[10, 1]));
    let b_init = Array::zeros(IxDyn(&[1, 1]));
    let y = deterministic(&[100, 1], 11);

    c.bench_function("adam_step_linreg_100x10", |bench| {
        let tool = LinearRegressionTool;
        let mut w = w_init.clone();
        let mut b = b_init.clone();
        let mut m_w: ArrayD<f64> = Array::zeros(w.raw_dim());
        let mut v_w: ArrayD<f64> = Array::zeros(w.raw_dim());
        let mut m_b: ArrayD<f64> = Array::zeros(b.raw_dim());
        let mut v_b: ArrayD<f64> = Array::zeros(b.raw_dim());
        let mut step_counter = 0_f64;

        bench.iter(|| {
            step_counter += 1.0;
            let mut ctx = DiffContext::new(ExecutionMode::Train);
            let mut inputs = ValueMap::new();
            inputs.insert("x".into(), Tensor::from_array(x.clone()));
            inputs.insert("w".into(), Tensor::from_array_with_grad(w.clone()));
            inputs.insert("b".into(), Tensor::from_array_with_grad(b.clone()));
            inputs.insert("y".into(), Tensor::from_array(y.clone()));

            let _out = tool.forward(&mut ctx, &inputs).expect("forward");
            let dummy = ValueMap::new();
            let grads = tool.backward(&mut ctx, &dummy).expect("backward");

            let grad_w = grads.get("w").expect("w grad").as_f64().clone();
            let grad_b = grads.get("b").expect("b grad").as_f64().clone();

            // Adam update
            let beta1 = 0.9_f64;
            let beta2 = 0.999_f64;
            let lr = 0.05_f64;
            let eps = 1e-8;
            m_w.zip_mut_with(&grad_w, |m, &g| *m = beta1 * *m + (1.0 - beta1) * g);
            v_w.zip_mut_with(&grad_w, |v, &g| *v = beta2 * *v + (1.0 - beta2) * g * g);
            m_b.zip_mut_with(&grad_b, |m, &g| *m = beta1 * *m + (1.0 - beta1) * g);
            v_b.zip_mut_with(&grad_b, |v, &g| *v = beta2 * *v + (1.0 - beta2) * g * g);
            let bc1 = 1.0 - beta1.powf(step_counter);
            let bc2 = 1.0 - beta2.powf(step_counter);
            for (w_e, (&m, &v)) in w.iter_mut().zip(m_w.iter().zip(v_w.iter())) {
                *w_e -= lr * (m / bc1) / ((v / bc2).sqrt() + eps);
            }
            for (b_e, (&m, &v)) in b.iter_mut().zip(m_b.iter().zip(v_b.iter())) {
                *b_e -= lr * (m / bc1) / ((v / bc2).sqrt() + eps);
            }
            black_box(&w);
        })
    });
}

criterion_group!(
    benches,
    bench_add_128,
    bench_mul_128,
    bench_matmul_64,
    bench_variance_1024,
    bench_linreg_mse,
    bench_adam_step,
);
criterion_main!(benches);
