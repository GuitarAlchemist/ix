//! # ix-signal
//!
//! Signal processing toolkit: transforms, filters, spectral analysis.
//!
//! ## Modules
//!
//! - **fft**: Fast Fourier Transform (Cooley-Tukey radix-2)
//! - **dct**: Discrete Cosine Transform
//! - **wavelet**: Discrete Wavelet Transform (Haar, Daubechies)
//! - **filter**: FIR/IIR digital filters, window functions
//! - **spectral**: Power spectral density, spectrograms (STFT)
//! - **correlation**: Auto/cross-correlation
//! - **kalman**: Kalman filter for state estimation
//! - **sampling**: Nyquist, interpolation, decimation
//! - **window**: Window functions (Hamming, Hanning, Blackman, Kaiser)
//! - **convolution**: Linear and circular convolution

pub mod fft;
pub mod dct;
pub mod wavelet;
pub mod filter;
pub mod spectral;
pub mod correlation;
pub mod kalman;
pub mod sampling;
pub mod window;
pub mod convolution;
pub mod timeseries;
