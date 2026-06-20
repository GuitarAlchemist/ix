//! `ix-acoustic-tune` — acoustic parameter auto-tuning (experimental).
//!
//! Closes the gap for matching a (sound-)synthesizer's parameters to a target
//! recording: an **ask/tell** black-box optimizer whose objective is evaluated
//! *externally* (render the synth → score the audio), plus the spectral
//! **features** and **losses** that form that objective. Designed for the
//! GA↔IX guitar-synth sound-matching loop (see
//! `docs/plans/2026-06-07-ix-acoustic-tune.md`).
//!
//! ## Why a new crate (experimental tier)
//!
//! The generic pieces here — CMA-ES, the [`AskTell`] driver, the spectral
//! features — are promotion candidates for `ix-optimize` / `ix-signal`. They
//! live here first so the API can iterate without tripping the stable-surface
//! gate on those stable crates. Audio analysis is pure-`f64` and **offline**:
//! this crate never enters an audio thread; the realtime synth kernel stays in
//! its own repo. We *reuse* `ix-signal`'s transforms (FFT/STFT/PSD/DCT/
//! autocorrelation) and `ix-math`'s eigendecomposition rather than reinventing.
//!
//! ## The ask/tell shape
//!
//! ```ignore
//! let mut opt = CmaEs::new(mean, sigma, seed).with_bounds(lower, upper);
//! while opt.generation() < budget {
//!     let candidates = opt.ask();              // IX proposes
//!     let scored = candidates.iter()           // host renders + scores (external)
//!         .map(|p| (p.clone(), render_and_score(p)))
//!         .collect::<Vec<_>>();
//!     opt.tell(&scored);                       // IX updates from external scores
//! }
//! let (best_params, best_loss) = opt.recommend().unwrap();
//! ```

pub mod cmaes;
pub mod contract;
pub mod features;
pub mod reference;
pub mod session;
pub mod spectral_loss;
pub mod transforms;

use ndarray::Array1;

/// Ask/tell interface for a black-box optimizer whose objective is evaluated
/// **externally** (e.g. render a synth + score the audio against a reference).
///
/// The host calls [`ask`](AskTell::ask) to get candidate parameter vectors,
/// evaluates them however it likes (in-process, a subprocess, or over the
/// GA↔IX JSON contract), then returns the `(params, loss)` pairs via
/// [`tell`](AskTell::tell). This hoists the objective evaluation *out* of the
/// optimizer's loop — the key to optimizing an external, render-required
/// objective that a closed `minimize()` loop cannot express.
pub trait AskTell {
    /// Propose the next batch of candidate parameter vectors to evaluate.
    fn ask(&mut self) -> Vec<Array1<f64>>;

    /// Report the loss for each candidate (minimization). The slice need not be
    /// in the same order as [`ask`](AskTell::ask) returned, but every candidate
    /// from the most recent `ask` should appear exactly once.
    fn tell(&mut self, evaluated: &[(Array1<f64>, f64)]);

    /// Best `(parameters, loss)` seen so far, or `None` before the first `tell`.
    fn recommend(&self) -> Option<(Array1<f64>, f64)>;

    /// Number of completed generations (calls to `tell`).
    fn generation(&self) -> usize;
}
