//! End-to-end proof: CMA-ES recovers a toy synth's parameters by minimizing the
//! multi-resolution STFT loss against a target render — the full sound-matching
//! loop, in-process. (The real loop swaps this closure for the external GA WASM
//! render over the JSON contract; the optimizer/loss math is identical.)

use ndarray::Array1;

use ix_acoustic_tune::cmaes::CmaEs;
use ix_acoustic_tune::spectral_loss::multi_resolution_stft_loss;

/// Toy "synth": params = [normalized_frequency, decay_per_sample] -> damped sine.
fn render(params: &Array1<f64>, n: usize) -> Vec<f64> {
    let freq = params[0];
    let decay = params[1];
    (0..n)
        .map(|t| {
            let tt = t as f64;
            (-decay * tt).exp() * (2.0 * std::f64::consts::TAU * freq * tt).sin()
        })
        .collect()
}

#[test]
fn cmaes_recovers_toy_synth_params_via_spectral_loss() {
    const N: usize = 4096;
    let target_params = Array1::from_vec(vec![0.080, 0.0015]);
    let target = render(&target_params, N);

    // Heuristic init near the target (as the plan prescribes: f0/decay analysis
    // seeds the search), with bounds matching a plausible param range.
    let init = Array1::from_vec(vec![0.070, 0.0020]);
    let lower = Array1::from_vec(vec![0.02, 0.0]);
    let upper = Array1::from_vec(vec![0.20, 0.01]);

    let initial_loss = multi_resolution_stft_loss(&render(&init, N), &target);

    let opt = CmaEs::new(init, 0.01, 1234).with_bounds(lower, upper);
    let (best, loss) = opt.minimize(|p| multi_resolution_stft_loss(&render(p, N), &target), 200);

    // The loop must drive the spectral distance to the target sharply down...
    assert!(
        loss < 0.3 * initial_loss,
        "optimizer should cut the spectral loss (initial={initial_loss}, final={loss})"
    );
    // ...and recover the underlying parameters (decay landscape is smooth;
    // frequency to within a couple of STFT bins).
    assert!(
        (best[0] - 0.080).abs() < 0.01,
        "recovered frequency {} should be ≈0.080",
        best[0]
    );
    assert!(
        (best[1] - 0.0015).abs() < 0.0008,
        "recovered decay {} should be ≈0.0015",
        best[1]
    );
}
