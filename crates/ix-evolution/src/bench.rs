//! Benchmark objectives and paired comparison for operator evaluation (refs #204).
//!
//! These objectives were already written twice inside `ix-agent`'s handlers as
//! inline closures; there was no reusable definition anywhere in the workspace.
//! Rather than add a third copy inside a test, they live here where both the
//! `fractal_operator_bench` example and the operator tests can reach one
//! implementation.
//!
//! # The 2x2 that makes the benchmark say something
//!
//! A fractal mutation operator can plausibly help for two different reasons,
//! and the two predict different objectives:
//!
//! |               | separable            | coupled          |
//! |---------------|----------------------|------------------|
//! | **unimodal**  | [`sphere`]           | [`rosenbrock`]   |
//! | **multimodal**| [`rastrigin`]        | [`ackley`]       |
//!
//! Cross-gene correlation should show up on the *coupled* column; multi-scale
//! step bursts should show up on the *multimodal* row. Running only Rastrigin
//! would conflate them.
//!
//! [`shifted_rastrigin`] is the control. Sphere, Rastrigin and Ackley all put
//! their optimum at the origin, which is the centre of a symmetric search box —
//! so any operator that happens to contract toward the centre wins for a reason
//! that has nothing to do with search quality. Shifting the optimum off-centre
//! removes that artifact; a result that holds on `rastrigin` but vanishes on
//! `shifted_rastrigin` was the artifact.

use ndarray::Array1;

/// A named objective with the search box it is conventionally posed on.
pub struct Objective {
    /// Short identifier used in result tables.
    pub name: &'static str,
    /// Inclusive search bounds, applied to every dimension.
    pub bounds: (f64, f64),
    /// The function itself. Lower is better; every objective here has a global
    /// minimum of exactly 0.
    pub f: fn(&Array1<f64>) -> f64,
    /// Whether the objective has many local minima.
    pub multimodal: bool,
    /// Whether the objective couples neighbouring genes.
    pub coupled: bool,
}

/// Offset applied by [`shifted_rastrigin`], per dimension `i`.
///
/// Irrational-ish and dimension-dependent so the optimum lands on no grid an
/// operator could stumble onto, and off-centre so centre-contraction cannot win.
pub fn shift_for(index: usize) -> f64 {
    1.3 + 0.7 * ((index % 5) as f64)
}

/// `f(x) = sum(x_i^2)`. Unimodal, separable, minimum 0 at the origin.
pub fn sphere(x: &Array1<f64>) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Rosenbrock's valley. Unimodal, couples `x[i]` to `x[i+1]`, minimum 0 at
/// `(1, ..., 1)` — the only objective here whose optimum is already off-centre.
pub fn rosenbrock(x: &Array1<f64>) -> f64 {
    (0..x.len().saturating_sub(1))
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

/// Rastrigin. Multimodal, separable, minimum 0 at the origin.
pub fn rastrigin(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
            .sum::<f64>()
}

/// Rastrigin with the optimum moved off the centre of the search box.
pub fn shifted_rastrigin(x: &Array1<f64>) -> f64 {
    let shifted =
        Array1::from_iter(x.iter().enumerate().map(|(i, &v)| v - shift_for(i)));
    rastrigin(&shifted)
}

/// Ackley. Multimodal and coupled through both aggregate terms, minimum 0 at
/// the origin.
pub fn ackley(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
    let sum_cos: f64 = x
        .iter()
        .map(|&v| (2.0 * std::f64::consts::PI * v).cos())
        .sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
        + 20.0
        + std::f64::consts::E
}

/// The evaluation suite, in a fixed order so result tables are stable.
pub fn suite() -> Vec<Objective> {
    vec![
        Objective {
            name: "sphere",
            bounds: (-5.12, 5.12),
            f: sphere,
            multimodal: false,
            coupled: false,
        },
        Objective {
            name: "rosenbrock",
            bounds: (-2.048, 2.048),
            f: rosenbrock,
            multimodal: false,
            coupled: true,
        },
        Objective {
            name: "rastrigin",
            bounds: (-5.12, 5.12),
            f: rastrigin,
            multimodal: true,
            coupled: false,
        },
        Objective {
            name: "ackley",
            bounds: (-32.768, 32.768),
            f: ackley,
            multimodal: true,
            coupled: true,
        },
        Objective {
            name: "shifted_rastrigin",
            bounds: (-5.12, 5.12),
            f: shifted_rastrigin,
            multimodal: true,
            coupled: false,
        },
    ]
}

/// The outcome of comparing a candidate arm against a baseline, seed by seed.
///
/// Paired on purpose. Comparing two aggregate medians drawn from different
/// seeds measures the seeds as much as the operators; comparing the *difference*
/// within each seed cancels the shared run-to-run variance, which is the larger
/// effect here.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedResult {
    /// Seeds where the candidate reached a strictly lower fitness.
    pub wins: usize,
    /// Seeds where the baseline reached a strictly lower fitness.
    pub losses: usize,
    /// Seeds where the two were bit-identical.
    pub ties: usize,
    /// Median of `candidate - baseline` across seeds. Negative favours the
    /// candidate.
    pub median_delta: f64,
    /// Two-sided sign-test p-value over the non-tied seeds.
    pub p_value: f64,
}

/// Compare paired per-seed fitnesses. `candidate` and `baseline` must be the
/// same length and aligned by seed.
pub fn paired_compare(candidate: &[f64], baseline: &[f64]) -> PairedResult {
    assert_eq!(
        candidate.len(),
        baseline.len(),
        "paired comparison needs aligned per-seed results"
    );
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;
    let mut deltas = Vec::with_capacity(candidate.len());
    for (&c, &b) in candidate.iter().zip(baseline.iter()) {
        deltas.push(c - b);
        if c < b {
            wins += 1;
        } else if c > b {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    PairedResult {
        wins,
        losses,
        ties,
        median_delta: median(&mut deltas),
        p_value: sign_test_p(wins, losses),
    }
}

/// Median, sorting in place. Empty input yields NaN.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN fitnesses"));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Exact two-sided sign test: under "no difference", wins is Binomial(n, 0.5).
///
/// Deliberately the weakest defensible test. It assumes only that the pairing
/// is valid, makes no distributional claim about fitness values (which are
/// heavily skewed and bounded below by 0), and cannot be talked into
/// significance by one lucky seed.
pub fn sign_test_p(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let extreme = wins.max(losses);
    // sum_{i=extreme}^{n} C(n, i) * 0.5^n, computed with an incremental
    // binomial coefficient so n up to a few hundred stays exact enough.
    let mut tail = 0.0;
    let mut coefficient = 1.0_f64; // C(n, n)
    for i in (0..=n).rev() {
        if i < extreme {
            break;
        }
        tail += coefficient;
        // C(n, i-1) = C(n, i) * i / (n - i + 1)
        if i > 0 {
            coefficient = coefficient * i as f64 / (n - i + 1) as f64;
        }
    }
    (2.0 * tail * 0.5_f64.powi(n as i32)).min(1.0)
}
