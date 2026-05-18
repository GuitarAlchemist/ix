# ix-signal Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `fft::fft` and `fft::ifft` ZERO-PAD silently to the next power of two. Output length = `input.len().next_power_of_two()`, NOT `input.len()`. Callers requiring fixed length must check `is_power_of_two()` first.
- `fft::irfft` discards the imaginary parts of `ifft` output — assumes the input spectrum is conjugate-symmetric. Non-symmetric input produces a "real projection", NOT an error.
- `fft::ifft` applies the `1/N` normalization (so `ifft(fft(x)) == x` within rounding). Some libraries apply `1/sqrt(N)` symmetrically; this crate does NOT.
- `fft::Complex` has `PartialEq` derived on `(re, im)` — two complex numbers `==` iff both fields bit-equal. NaN propagates per IEEE 754 (`NaN != NaN`).
- `kalman::KalmanFilter::new` initializes `transition = I`, `observation = 0`, `process_noise = 0.01 * I`, `measurement_noise = I`, `state = 0`, `covariance = I`. Defaults are NOT a usable filter — caller must overwrite at least `observation`.
- `filter::FirFilter` applies a CAUSAL filter (no look-ahead); output length equals input length with leading samples reflecting the filter's transient response (not zero-padded mathematically symmetric).
- `wavelet::dwt_haar` requires input length to be a power of two; non-power-of-two input panics inside the internal indexing.
- `spectral::spectrogram` returns shape `(n_frames, n_freqs)` where `n_frames = (signal_len - window_size) / hop_size + 1`. Final partial frame is DROPPED, not padded.
- `timeseries::page_hinkley_detect` returns `DriftState::Drift { index }` at the FIRST detection point; subsequent drifts in the same series are NOT reported in a single call.
- `correlation::auto_correlation` returns unnormalized correlation; divide by `series.len()` for biased estimator or by `(n - lag)` for unbiased — caller's choice.

## Concurrency contracts
- All functions are pure with `&[T]` or `&Array` inputs; safe to call from multiple threads concurrently.
- `KalmanFilter` is `!Sync` for `predict`/`update` (mutates internal state via `&mut self`); clone per thread.

## Failure contracts
- FFT functions are infallible by signature — no Result. Empty input (`len() == 0`) panics inside `next_power_of_two()` (returns 1, then resize to 1 works, but `magnitude_spectrum` on empty returns empty).
- `kalman::KalmanFilter::update` panics on dimension mismatch between observation and the filter's `obs_dim`.
- `wavelet::dwt_haar` panics on non-power-of-two input.
- No `Result`-typed surface in this crate.

## Determinism contracts
- All transforms are pure functions of input — no RNG anywhere on the public surface.
- `fft` is bit-identical across runs on the same platform/toolchain. Twiddle factors are computed via `f64::cos/sin` — cross-platform rounding may differ at ULP scale.
- `KalmanFilter` predict/update are deterministic given the initial state and observation sequence.
- `spectral::spectrogram` window iteration is left-to-right with fixed hop; output order matches input order.

## Memory contracts
- `fft::fft` allocates a single `Vec<Complex>` of length `next_power_of_two(input.len())`. No in-place variant on the public surface (the in-place is private).
- `Complex` is `Copy` (no heap); slice operations are zero-allocation until output materialization.
- `KalmanFilter` allocates fresh matrices for predict/update intermediate products — no buffer reuse. For high-frequency calls, callers may want a pooling wrapper externally.
- `spectrogram` allocates `(n_frames, n_freqs)` upfront; memory scales with signal length and inverse hop size.
