---
name: machin-signal
description: Signal processing — FFT, filtering, wavelets, Kalman, spectral analysis
---

# Signal Processing

Analyze and transform time-domain and frequency-domain signals.

## When to Use
When the user has time series, audio, sensor data, or any signal that needs frequency analysis, filtering, or noise reduction.

## Capabilities
- **FFT/IFFT** — Frequency decomposition, power spectrum
- **Filtering** — Low-pass, high-pass, band-pass FIR/IIR filters
- **Wavelets** — Haar wavelet transform, multi-resolution analysis
- **Kalman Filter** — State estimation for noisy dynamic systems
- **Spectral Analysis** — Power spectral density, spectrogram
- **Windows** — Hamming, Hanning, Blackman for spectral leakage reduction
- **Convolution/Correlation** — Cross-correlation, autocorrelation
- **DCT** — Discrete cosine transform (compression, feature extraction)

## Programmatic Usage
```rust
use machin_signal::fft::{fft, ifft, power_spectrum};
use machin_signal::filter::{low_pass, high_pass};
use machin_signal::kalman::KalmanFilter;
use machin_signal::wavelet::haar_wavelet_transform;
use machin_signal::spectral::spectrogram;
use machin_signal::window::{hamming, hanning};
```

## Tips
- Apply a window function before FFT to reduce spectral leakage
- Kalman filter requires a state model — help user define F, H, Q, R matrices
- Use power spectrum to identify dominant frequencies
