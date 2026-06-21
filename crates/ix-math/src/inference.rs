//! Statistical inference: distribution-shape moments, divergences, and two-sample
//! hypothesis tests — the layer both `ix-math::stats` and DuckDB built-ins lack.
//!
//! The motivating question is *"did this metric's **distribution** shift versus
//! baseline?"* — the core of any regression / RSI gate. A mean can hold steady
//! while the distribution degrades (variance blow-up, bimodality, tail shift), so
//! a principled gate needs two-sample tests, not mean thresholds.
//!
//! Conventions match SciPy defaults so results are checkable against a reference:
//! [`skewness`]/[`kurtosis`] are the biased (population) Fisher–Pearson forms,
//! [`kurtosis`] is *excess*, [`zscore`] uses population σ (ddof=0), [`quantile`]
//! is numpy's linear interpolation, [`welch_t_test`] is `ttest_ind(equal_var=False)`,
//! and [`shannon_entropy`]/[`kl_divergence`] use natural log (nats). Test values
//! are pinned to SciPy in the unit tests — **do not** change a formula without a
//! reference value.

use crate::error::MathError;

/// Outcome of a two-sample hypothesis test: the test statistic and its two-sided
/// p-value (the probability, under the null of equal distributions, of a statistic
/// at least this extreme).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

// ── descriptive: distribution shape over one sample ────────────────────────────

/// The `q`-quantile (`q ∈ [0, 1]`) by linear interpolation between order statistics
/// — numpy's default method. `q = 0.5` is the median.
pub fn quantile(x: &[f64], q: f64) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    if q.is_nan() || !(0.0..=1.0).contains(&q) {
        return Err(MathError::InvalidParameter(format!(
            "quantile q must be in [0, 1], got {q}"
        )));
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 1 {
        return Ok(s[0]);
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 < n {
        Ok(s[lo] + frac * (s[lo + 1] - s[lo]))
    } else {
        Ok(s[lo])
    }
}

/// Interquartile range: `quantile(0.75) − quantile(0.25)`.
pub fn iqr(x: &[f64]) -> Result<f64, MathError> {
    Ok(quantile(x, 0.75)? - quantile(x, 0.25)?)
}

/// Median absolute deviation: `median(|xᵢ − median(x)|)`. A robust spread estimate.
pub fn mad(x: &[f64]) -> Result<f64, MathError> {
    let med = quantile(x, 0.5)?;
    let dev: Vec<f64> = x.iter().map(|v| (v - med).abs()).collect();
    quantile(&dev, 0.5)
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// `n`-th central moment `(1/N) Σ (xᵢ − μ)ⁿ`.
fn central_moment(x: &[f64], n: i32) -> f64 {
    let mu = mean(x);
    x.iter().map(|v| (v - mu).powi(n)).sum::<f64>() / x.len() as f64
}

/// Biased (population) Fisher–Pearson skewness `g₁ = m₃ / m₂^{3/2}` — SciPy's
/// `skew` default. Symmetric data → 0; a long right tail → positive.
pub fn skewness(x: &[f64]) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let m2 = central_moment(x, 2);
    if m2 <= 0.0 {
        return Err(MathError::InvalidParameter(
            "skewness is undefined for zero-variance data".into(),
        ));
    }
    Ok(central_moment(x, 3) / m2.powf(1.5))
}

/// Biased excess kurtosis `g₂ = m₄ / m₂² − 3` — SciPy's `kurtosis` default
/// (`fisher=True`). A normal distribution → 0.
pub fn kurtosis(x: &[f64]) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let m2 = central_moment(x, 2);
    if m2 <= 0.0 {
        return Err(MathError::InvalidParameter(
            "kurtosis is undefined for zero-variance data".into(),
        ));
    }
    Ok(central_moment(x, 4) / (m2 * m2) - 3.0)
}

/// Standard score `(xᵢ − μ) / σ` with population σ (ddof = 0) — SciPy's `zscore`.
pub fn zscore(x: &[f64]) -> Result<Vec<f64>, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let mu = mean(x);
    let sigma = central_moment(x, 2).sqrt();
    if sigma <= 0.0 {
        return Err(MathError::InvalidParameter(
            "zscore is undefined for zero-variance data".into(),
        ));
    }
    Ok(x.iter().map(|v| (v - mu) / sigma).collect())
}

// ── divergences over probability vectors ───────────────────────────────────────

/// Normalize a non-negative vector to sum 1. Errors on negatives or all-zero.
fn normalize(p: &[f64]) -> Result<Vec<f64>, MathError> {
    if p.is_empty() {
        return Err(MathError::EmptyInput);
    }
    if p.iter().any(|&v| v < 0.0 || v.is_nan()) {
        return Err(MathError::InvalidParameter(
            "probability vector must be non-negative".into(),
        ));
    }
    let total: f64 = p.iter().sum();
    if total <= 0.0 {
        return Err(MathError::InvalidParameter(
            "probability vector sums to zero".into(),
        ));
    }
    Ok(p.iter().map(|v| v / total).collect())
}

/// Shannon entropy `−Σ pᵢ ln pᵢ` in nats over the normalized distribution.
pub fn shannon_entropy(p: &[f64]) -> Result<f64, MathError> {
    let p = normalize(p)?;
    Ok(-p
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v * v.ln())
        .sum::<f64>())
}

/// Kullback–Leibler divergence `Σ pᵢ ln(pᵢ / qᵢ)` in nats (asymmetric). Both
/// vectors are normalized first; `qᵢ = 0` where `pᵢ > 0` is `+∞`-undefined → error.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<f64, MathError> {
    if p.len() != q.len() {
        return Err(MathError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    let p = normalize(p)?;
    let q = normalize(q)?;
    let mut sum = 0.0;
    for (&pi, &qi) in p.iter().zip(&q) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return Err(MathError::InvalidParameter(
                    "KL divergence diverges: q is zero where p is positive".into(),
                ));
            }
            sum += pi * (pi / qi).ln();
        }
    }
    Ok(sum)
}

/// Jensen–Shannon divergence `½KL(p‖m) + ½KL(q‖m)` with `m = (p+q)/2`. Symmetric
/// and bounded in `[0, ln 2]`; `0` iff the distributions are equal.
pub fn js_divergence(p: &[f64], q: &[f64]) -> Result<f64, MathError> {
    if p.len() != q.len() {
        return Err(MathError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    let p = normalize(p)?;
    let q = normalize(q)?;
    let m: Vec<f64> = p.iter().zip(&q).map(|(&a, &b)| 0.5 * (a + b)).collect();
    Ok(0.5 * kl_divergence(&p, &m)? + 0.5 * kl_divergence(&q, &m)?)
}

// ── two-sample tests ───────────────────────────────────────────────────────────

/// Reject non-finite (NaN / ±∞) values up front. The two-sample tests sort and
/// merge with `partial_cmp` fallbacks that misbehave on NaN — in particular the KS
/// merge can pick `v = NaN` and, since `NaN != NaN`, never advance, **hanging** the
/// caller. A regression gate fed NaN telemetry must error, not spin.
fn reject_non_finite(a: &[f64], b: &[f64]) -> Result<(), MathError> {
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(MathError::InvalidParameter(
            "samples must be finite (no NaN/∞); filter missing values first".into(),
        ));
    }
    Ok(())
}

/// Two-sample Kolmogorov–Smirnov test. Statistic `D = supₓ |F_a(x) − F_b(x)|`;
/// p-value from the asymptotic Kolmogorov distribution with the Stephens
/// small-sample correction (SciPy's `method='asymp'`).
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> Result<TestResult, MathError> {
    if a.is_empty() || b.is_empty() {
        return Err(MathError::EmptyInput);
    }
    reject_non_finite(a, b)?;
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let (na, nb) = (sa.len() as f64, sb.len() as f64);
    // Walk the merged order, tracking the two empirical CDFs and their max gap. At
    // each distinct value, advance *both* CDFs past all of its copies before
    // measuring the gap — otherwise a shared (tied) value spuriously inflates D.
    let (mut i, mut j) = (0usize, 0usize);
    let mut d = 0.0_f64;
    while i < sa.len() || j < sb.len() {
        let v = if i < sa.len() && (j >= sb.len() || sa[i] <= sb[j]) {
            sa[i]
        } else {
            sb[j]
        };
        while i < sa.len() && sa[i] == v {
            i += 1;
        }
        while j < sb.len() && sb[j] == v {
            j += 1;
        }
        d = d.max((i as f64 / na - j as f64 / nb).abs());
    }
    let en = (na * nb / (na + nb)).sqrt();
    let p = kolmogorov_sf((en + 0.12 + 0.11 / en) * d);
    Ok(TestResult {
        statistic: d,
        p_value: p.clamp(0.0, 1.0),
    })
}

/// Mann–Whitney U test (rank-sum) with the normal approximation, tie correction,
/// and continuity correction — SciPy's `mannwhitneyu(method='asymptotic')`. The
/// reported statistic is `U₁` for the first sample; the p-value is two-sided.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> Result<TestResult, MathError> {
    if a.is_empty() || b.is_empty() {
        return Err(MathError::EmptyInput);
    }
    reject_non_finite(a, b)?;
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    // Average ranks over the pooled sample, recording tie-group sizes.
    let mut pooled: Vec<(f64, bool)> = a.iter().map(|&v| (v, true)).collect();
    pooled.extend(b.iter().map(|&v| (v, false)));
    pooled.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = pooled.len();
    let mut ranks = vec![0.0f64; n];
    let mut tie_term = 0.0;
    let mut k = 0;
    while k < n {
        let mut m = k + 1;
        while m < n && pooled[m].0 == pooled[k].0 {
            m += 1;
        }
        let group = m - k;
        let avg_rank = (k + 1 + m) as f64 / 2.0; // mean of ranks k+1..=m
        for r in ranks.iter_mut().take(m).skip(k) {
            *r = avg_rank;
        }
        let t = group as f64;
        tie_term += t * t * t - t;
        k = m;
    }
    let r1: f64 = ranks
        .iter()
        .zip(&pooled)
        .filter(|(_, &(_, first))| first)
        .map(|(&r, _)| r)
        .sum();
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let mu = n1 * n2 / 2.0;
    let big_n = n1 + n2;
    let sigma =
        (n1 * n2 / 12.0 * ((big_n + 1.0) - tie_term / (big_n * (big_n - 1.0)))).sqrt();
    if sigma <= 0.0 {
        return Err(MathError::InvalidParameter(
            "Mann–Whitney variance is zero (all values tied)".into(),
        ));
    }
    // Continuity correction of 0.5 toward the mean.
    let z = (u1 - mu).abs().max(0.0);
    let z = (z - 0.5).max(0.0) / sigma;
    let p = (2.0 * normal_sf(z)).clamp(0.0, 1.0);
    Ok(TestResult {
        statistic: u1,
        p_value: p,
    })
}

/// Welch's unequal-variance two-sample t-test — SciPy's
/// `ttest_ind(equal_var=False)`. Statistic `t`, two-sided p-value from the
/// Student-t distribution with Welch–Satterthwaite degrees of freedom.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> Result<TestResult, MathError> {
    reject_non_finite(a, b)?;
    if a.len() < 2 || b.len() < 2 {
        return Err(MathError::InvalidParameter(
            "Welch's t-test needs at least 2 observations per sample".into(),
        ));
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (ma, mb) = (mean(a), mean(b));
    // Sample variance (ddof = 1).
    let va = a.iter().map(|v| (v - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|v| (v - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sa = va / na;
    let sb = vb / nb;
    let se = (sa + sb).sqrt();
    if se <= 0.0 {
        return Err(MathError::InvalidParameter(
            "Welch's t-test is undefined: both samples are constant".into(),
        ));
    }
    let t = (ma - mb) / se;
    let df = (sa + sb).powi(2) / (sa * sa / (na - 1.0) + sb * sb / (nb - 1.0));
    // Two-sided p = I_x(df/2, 1/2) with x = df/(df + t²).
    let x = df / (df + t * t);
    let p = reg_incomplete_beta(df / 2.0, 0.5, x).clamp(0.0, 1.0);
    Ok(TestResult {
        statistic: t,
        p_value: p,
    })
}

// ── special functions (self-contained; ix-math has none of these) ──────────────

/// Error function via Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Upper tail of the standard normal: `P(Z > z)`.
fn normal_sf(z: f64) -> f64 {
    0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2))
}

/// Survival function of the Kolmogorov distribution
/// `Q(t) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²t²}`, via the Numerical Recipes `probks`
/// convergence logic. Crucially: for **small** `t` the alternating series converges
/// only conditionally and term-by-term truncation is numerically wrong (it would
/// report a tiny KS statistic as significant); when the terms don't shrink we fall
/// through to `Q ≈ 1`, which is the correct small-`t` limit.
fn kolmogorov_sf(t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    const EPS1: f64 = 1e-3;
    const EPS2: f64 = 1e-8;
    let a2 = -2.0 * t * t;
    let mut fac = 2.0; // the leading 2× is folded into the alternating factor
    let mut sum = 0.0;
    let mut termbf = 0.0;
    for k in 1..=100 {
        let term = fac * (a2 * (k * k) as f64).exp();
        sum += term;
        // Converged: the term is negligible vs the previous one or the running sum.
        if term.abs() <= EPS1 * termbf || term.abs() <= EPS2 * sum {
            return sum.clamp(0.0, 1.0);
        }
        fac = -fac;
        termbf = term.abs();
    }
    1.0 // did not converge (small t) → survival ≈ 1
}

/// Natural log of the gamma function (Lanczos approximation, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx).
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let tval = x + 7.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tval.ln() - tval + a.ln()
    }
}

/// Regularized incomplete beta `I_x(a, b)` via the Lentz continued fraction
/// (Numerical Recipes `betai`).
fn reg_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued-fraction core of the incomplete beta (Numerical Recipes `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;
    /// Looser tolerance for p-values from approximate special functions.
    const PTOL: f64 = 2e-3;

    #[test]
    fn quantile_linear_interpolation() {
        // numpy.quantile([1,2,3,4], q, method='linear')
        assert!((quantile(&[1., 2., 3., 4.], 0.5).unwrap() - 2.5).abs() < TOL);
        assert!((quantile(&[1., 2., 3., 4.], 0.25).unwrap() - 1.75).abs() < TOL);
        assert!((quantile(&[1., 2., 3., 4.], 0.75).unwrap() - 3.25).abs() < TOL);
        assert!(quantile(&[1.0], 1.5).is_err());
    }

    #[test]
    fn mad_robust_spread() {
        // scipy.stats.median_abs_deviation([1,2,3,4,5]) = 1.0
        assert!((mad(&[1., 2., 3., 4., 5.]).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn skewness_kurtosis_match_scipy() {
        // scipy.stats.skew([1,2,3,4,5]) = 0; .kurtosis(...) = -1.3
        assert!(skewness(&[1., 2., 3., 4., 5.]).unwrap().abs() < TOL);
        assert!((kurtosis(&[1., 2., 3., 4., 5.]).unwrap() + 1.3).abs() < TOL);
        // scipy.stats.skew([0,0,0,0,5]) = 1.5 (right-skewed)
        assert!((skewness(&[0., 0., 0., 0., 5.]).unwrap() - 1.5).abs() < TOL);
        assert!(skewness(&[2., 2., 2.]).is_err(), "zero variance is undefined");
    }

    #[test]
    fn zscore_population_sigma() {
        let z = zscore(&[1., 2., 3., 4., 5.]).unwrap();
        // mean 3, population sigma sqrt(2); endpoints ±2/sqrt(2) = ±sqrt(2)
        assert!((z[0] + 2.0_f64.sqrt()).abs() < TOL);
        assert!(z[2].abs() < TOL);
        assert!((z.iter().sum::<f64>()).abs() < TOL, "z-scores sum to 0");
    }

    #[test]
    fn entropy_and_divergences() {
        // uniform over 4 → ln 4
        assert!((shannon_entropy(&[1., 1., 1., 1.]).unwrap() - 4.0_f64.ln()).abs() < TOL);
        assert!(shannon_entropy(&[1., 0., 0., 0.]).unwrap().abs() < TOL);
        // KL([1,0] ‖ [0.5,0.5]) = ln 2
        assert!((kl_divergence(&[1., 0.], &[0.5, 0.5]).unwrap() - 2.0_f64.ln()).abs() < TOL);
        assert!(kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).unwrap().abs() < TOL);
        // JS of disjoint supports = ln 2 (the max)
        assert!((js_divergence(&[1., 0.], &[0., 1.]).unwrap() - 2.0_f64.ln()).abs() < TOL);
        assert!(js_divergence(&[1., 2., 3.], &[1., 2., 3.]).unwrap().abs() < TOL);
        // KL diverges when q is zero where p is positive.
        assert!(kl_divergence(&[1., 1.], &[1., 0.]).is_err());
    }

    #[test]
    fn ks_two_sample_separates_distributions() {
        // Identical samples → D = 0, p = 1.
        let same = ks_two_sample(&[1., 2., 3., 4.], &[1., 2., 3., 4.]).unwrap();
        assert!(same.statistic.abs() < TOL);
        assert!(same.p_value > 0.99);
        // Disjoint ranges → D = 1, tiny p.
        let diff = ks_two_sample(&[0., 1., 2.], &[10., 11., 12.]).unwrap();
        assert!((diff.statistic - 1.0).abs() < TOL);
        assert!(diff.p_value < 0.2, "clearly different → small p, got {}", diff.p_value);
    }

    #[test]
    fn two_sample_tests_reject_non_finite() {
        // A NaN must error promptly — never hang the KS merge (Codex P1).
        assert!(ks_two_sample(&[1., f64::NAN, 3.], &[1., 2., 3.]).is_err());
        assert!(mann_whitney_u(&[1., 2.], &[f64::INFINITY, 4.]).is_err());
        assert!(welch_t_test(&[1., 2., 3.], &[1., 2., f64::NAN]).is_err());
    }

    #[test]
    fn ks_small_statistic_is_not_significant() {
        // Large samples differing by a single rank → tiny D, huge n: the p-value must
        // stay ≈ 1, not be reported significant (Codex P2 — small-t Kolmogorov).
        let a: Vec<f64> = (0..400).map(|i| i as f64).collect();
        let mut b = a.clone();
        *b.last_mut().unwrap() += 0.5; // shift one point a hair
        let r = ks_two_sample(&a, &b).unwrap();
        assert!(r.statistic < 0.01, "D should be tiny, got {}", r.statistic);
        assert!(r.p_value > 0.9, "tiny D over n=400 → p≈1, got {}", r.p_value);
    }

    #[test]
    fn mann_whitney_complete_separation() {
        // x entirely below y → U1 = 0.
        let r = mann_whitney_u(&[1., 2., 3.], &[4., 5., 6.]).unwrap();
        assert!((r.statistic - 0.0).abs() < TOL, "U1 = 0, got {}", r.statistic);
        assert!(r.p_value < 0.2, "separation → small p, got {}", r.p_value);
    }

    #[test]
    fn welch_t_test_matches_scipy() {
        // ttest_ind([1,2,3,4,5],[2,3,4,5,6], equal_var=False): t = -1.0, df = 8, p ≈ 0.34659
        let r = welch_t_test(&[1., 2., 3., 4., 5.], &[2., 3., 4., 5., 6.]).unwrap();
        assert!((r.statistic + 1.0).abs() < TOL, "t = -1.0, got {}", r.statistic);
        assert!((r.p_value - 0.346_598).abs() < PTOL, "p ≈ 0.3466, got {}", r.p_value);
        // Identical samples → t = 0, p = 1.
        let same = welch_t_test(&[1., 2., 3.], &[1., 2., 3.]).unwrap();
        assert!(same.statistic.abs() < TOL);
        assert!((same.p_value - 1.0).abs() < PTOL);
    }
}
