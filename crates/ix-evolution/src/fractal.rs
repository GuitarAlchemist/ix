//! Fractal mutation and recursive crossover operators (refs #204, gap-matrix C5/C6).
//!
//! `ix-fractal` ships the curves (`takagi`, `de_rham_interpolate`) and
//! `ix-evolution` owns the operators; before this module the two were unjoined.
//! This is the join, built so the question "does fractal noise beat Gaussian
//! noise" can actually be answered rather than asserted.
//!
//! # Why every operator here is standardised
//!
//! The Takagi function `T` is bounded on `[0, 2/3]` with mean `1/2` and
//! standard deviation `1/6`; the baseline mutation draws from `Normal(0, rate)`,
//! whose standard deviation is `rate`. Dropping raw Takagi values in as a
//! perturbation would therefore change the *step size* as well as the noise
//! shape, and any benchmark difference would measure the step size — the
//! cheapest possible confound.
//!
//! So [`TakagiNoise`] measures the curve's own mean and standard deviation at
//! construction and standardises to mean 0, sd 1. An operator with `rate = r`
//! then has the same per-gene step scale as `Normal(0, r)`, and the only
//! remaining differences are the ones under test: the *shape* of the
//! distribution and the *correlation* between genes.
//!
//! Note the mean is measured rather than assumed to be `1/2`. It only reaches
//! `1/2` in the limit — at `terms = 4` it is `0.46875`, and centring on `1/2`
//! there would inject a systematic downward drift into every mutation.
//!
//! # The three mutation arms
//!
//! Separating them is the point. A single "fractal mutation" arm that lost
//! would not say *which* property failed.
//!
//! - [`TakagiMode::Iid`] — each gene draws an independent phase. Same marginal
//!   distribution as the correlated arm, no cross-gene structure. Isolates
//!   "does the distribution shape matter".
//! - [`TakagiMode::Correlated`] — one random phase per mutation event, genes
//!   read consecutive points of the curve. Isolates "does multi-scale
//!   correlation between neighbouring genes matter" — the property that should
//!   help on an objective coupling `x[i]` to `x[i+1]`.
//! - [`fractal_noise_schedule`] — Gaussian noise whose amplitude is modulated
//!   across generations by the curve. Isolates "do multi-scale bursts help
//!   escape local optima".

use ix_fractal::de_rham::de_rham_interpolate;
use ix_fractal::takagi::takagi;
use ndarray::Array1;
use rand::Rng;

/// Resolution of the grid used to measure the curve's moments.
///
/// The Takagi curve at `terms = t` is piecewise linear with `2^t` pieces, so a
/// grid finer than that resolves it exactly. 4096 covers `terms <= 12`; beyond
/// that the moments have already converged to `1/2` and `1/6` to well past f64
/// display precision, so a coarser grid costs nothing real.
const MOMENT_GRID: usize = 4096;

/// A standardised Takagi noise source: mean 0, standard deviation 1.
///
/// Construction is O(`MOMENT_GRID`) and happens once per run, not per mutation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakagiNoise {
    terms: usize,
    mean: f64,
    sd: f64,
}

impl TakagiNoise {
    /// Measure the curve's moments at this term count and build the source.
    ///
    /// `terms` is clamped to at least 1; `ix_fractal::takagi` caps it at 53.
    pub fn new(terms: usize) -> Self {
        let terms = terms.max(1);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..MOMENT_GRID {
            let value = takagi(i as f64 / MOMENT_GRID as f64, terms);
            sum += value;
            sum_sq += value * value;
        }
        let n = MOMENT_GRID as f64;
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        Self {
            terms,
            mean,
            sd: variance.sqrt(),
        }
    }

    /// Number of terms in the series.
    pub fn terms(&self) -> usize {
        self.terms
    }

    /// The measured mean of the raw curve. Approaches `1/2` as `terms` grows.
    pub fn raw_mean(&self) -> f64 {
        self.mean
    }

    /// The measured standard deviation of the raw curve. Approaches `1/6`.
    pub fn raw_sd(&self) -> f64 {
        self.sd
    }

    /// The curve at `t`, shifted and scaled to mean 0 and standard deviation 1.
    ///
    /// `t` is taken modulo 1 by `ix_fractal::takagi`, so any real phase works.
    pub fn standardized(&self, t: f64) -> f64 {
        if self.sd <= f64::EPSILON {
            return 0.0;
        }
        (takagi(t, self.terms) - self.mean) / self.sd
    }
}

impl Default for TakagiNoise {
    /// 12 terms: the point where the measured moments have converged to `1/2`
    /// and `1/6` to six decimal places, so more terms buy nothing.
    fn default() -> Self {
        Self::new(12)
    }
}

/// How a Takagi perturbation is spread across a genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakagiMode {
    /// One independent phase per gene. No cross-gene structure.
    Iid,
    /// One phase per mutation event; gene `i` reads the curve at
    /// `phase + i / dim`, so the genome spans one period. Neighbouring genes
    /// share the curve's coarse scales and differ at its fine ones.
    Correlated,
}

/// Perturb `genes` in place with standardised Takagi noise.
///
/// `rate` is the per-gene step scale, matching `Normal(0, rate)` in the
/// baseline. `per_gene_prob` is the probability a given gene is touched at all;
/// it exists so an arm can be compared against the baseline's own 0.3 gate
/// rather than against a different one.
///
/// Returns the number of genes actually perturbed, which is what lets a test
/// check the gate without re-deriving it.
pub fn fractal_mutation_takagi(
    genes: &mut Array1<f64>,
    rate: f64,
    per_gene_prob: f64,
    noise: &TakagiNoise,
    mode: TakagiMode,
    rng: &mut impl Rng,
) -> usize {
    let dim = genes.len();
    if dim == 0 {
        return 0;
    }
    // Drawn before the loop so the correlated arm shares one phase across the
    // whole genome — that sharing is the property under test.
    let phase: f64 = rng.random::<f64>();
    let stride = 1.0 / dim as f64;

    let mut touched = 0;
    for (index, gene) in genes.iter_mut().enumerate() {
        if rng.random::<f64>() >= per_gene_prob {
            continue;
        }
        let t = match mode {
            TakagiMode::Iid => rng.random::<f64>(),
            TakagiMode::Correlated => phase + index as f64 * stride,
        };
        *gene += rate * noise.standardized(t);
        touched += 1;
    }
    touched
}

/// How a mutation's amplitude varies across a run.
///
/// The Takagi variant is the operator under test; the other two are its
/// controls, and they exist because the first benchmark run made the reason for
/// its win ambiguous.
///
/// `T(0) = T(1) = 0`, so a Takagi-modulated amplitude collapses toward zero at
/// the *end* of a run. That is annealing — a well-known technique with nothing
/// fractal about it. Without a monotone control at the same mean amplitude,
/// "the fractal schedule won" and "any decreasing schedule would have won" are
/// indistinguishable, and the first is the more flattering reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmplitudeSchedule {
    /// A fixed multiplier for the whole run. Isolates "is the average step size
    /// simply better", with no scheduling at all.
    Constant(f64),
    /// `start * (1 - t)`: monotone decreasing to zero. Plain annealing, no
    /// multi-scale structure. Isolates "does any decreasing schedule do this".
    Linear {
        /// Multiplier at `t = 0`.
        start: f64,
    },
    /// `1 + strength * standardized(T(t))`, floored just above zero.
    Takagi {
        /// The standardised noise source.
        noise: TakagiNoise,
        /// How far the multiplier swings. 0 reduces to a constant 1.
        strength: f64,
    },
}

/// Smallest multiplier any schedule may return.
///
/// A non-positive amplitude would flip the perturbation's sign rather than
/// shrink it, which is a different operator, not a smaller step.
pub const MIN_AMPLITUDE_MULTIPLIER: f64 = 1e-3;

impl AmplitudeSchedule {
    /// The multiplier at `iteration` of `total`. `total = 0` yields 1.0.
    pub fn at(&self, iteration: usize, total: usize) -> f64 {
        if total == 0 {
            return 1.0;
        }
        let t = iteration as f64 / total as f64;
        let raw = match self {
            AmplitudeSchedule::Constant(value) => *value,
            AmplitudeSchedule::Linear { start } => start * (1.0 - t),
            AmplitudeSchedule::Takagi { noise, strength } => {
                1.0 + strength * noise.standardized(t)
            }
        };
        raw.max(MIN_AMPLITUDE_MULTIPLIER)
    }

    /// The mean multiplier actually applied across a run of `total` steps.
    ///
    /// Measured over the same discrete grid the run uses, and *after* the
    /// floor, because the floor truncates the low tail and so shifts the real
    /// mean above the nominal one. This is what the controls are matched on:
    /// two schedules with different mean amplitude are two different step
    /// sizes, and the benchmark would be measuring that instead.
    pub fn mean_multiplier(&self, total: usize) -> f64 {
        if total == 0 {
            return 1.0;
        }
        let sum: f64 = (0..total).map(|i| self.at(i, total)).sum();
        sum / total as f64
    }

    /// A [`AmplitudeSchedule::Linear`] whose mean over `total` steps equals
    /// `target`, found by bisection because the floor makes the relationship
    /// non-linear near zero.
    pub fn linear_matching(target: f64, total: usize) -> Self {
        let (mut lo, mut hi) = (0.0_f64, 8.0_f64.max(target * 8.0));
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if (AmplitudeSchedule::Linear { start: mid }).mean_multiplier(total) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        AmplitudeSchedule::Linear {
            start: 0.5 * (lo + hi),
        }
    }
}

/// Multi-scale amplitude schedule: the Takagi curve read across a run.
///
/// Retained as the named candidate from the issue. Equivalent to
/// `AmplitudeSchedule::Takagi { .. }.at(iteration, total)`.
pub fn fractal_noise_schedule(
    iteration: usize,
    total: usize,
    noise: &TakagiNoise,
    strength: f64,
) -> f64 {
    AmplitudeSchedule::Takagi {
        noise: *noise,
        strength,
    }
    .at(iteration, total)
}

/// Recursive crossover: blend two parents along a de Rham fractal path.
///
/// `de_rham_interpolate` builds a midpoint-displaced path from `a` to `b`; this
/// samples one point on it uniformly. Against the baseline BLX-alpha — which
/// samples each gene independently from a box around the parents — the child
/// here sits on a single connected path, so its genes are displaced coherently.
///
/// # Cost
///
/// The shipped primitive materialises all `2^depth + 1` points to produce the
/// one we keep: `depth = 4` allocates 17 vectors per crossover. An O(`depth`)
/// single-sample walk down one branch would give the same distribution, but
/// writing one here would duplicate `ix-fractal`'s displacement logic and let
/// the two drift. Reusing the shipped curve and recording the cost is the
/// cheaper trade until a benchmark says the allocation matters.
pub fn recursive_crossover_de_rham(
    a: &Array1<f64>,
    b: &Array1<f64>,
    depth: usize,
    roughness: f64,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let path = de_rham_interpolate(a, b, depth, roughness, rng);
    if path.is_empty() {
        return a.clone();
    }
    let index = rng.random_range(0..path.len());
    path[index].clone()
}
