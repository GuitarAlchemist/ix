//! `ix_dev_rhythm` — a Markov / HMM model of IX's own development, by IX over IX.
//!
//! Dogfood: the observation sequence is IX's git history — each commit's
//! conventional-commit **type** (feat / fix / docs / chore / refactor / test / other), in
//! chronological order. Three stages, all on IX primitives:
//!   1. **Markov chain** (`ix_graph::markov::MarkovChain`) — transition matrix + the
//!      stationary distribution (the long-run mix of commit types).
//!   2. **Validation** — *does commit order carry Markov structure, or are types i.i.d.?*
//!      A permutation null (shuffle the sequence, keep the marginal) gives a p-value for
//!      "the next commit type is predictable from the current one".
//!   3. **Hidden phases** — a 2-state HMM trained by **Baum-Welch** (`ix_graph::hmm`),
//!      then decoded with the **`ix_viterbi`** UDF on the bench, recovering latent
//!      "build vs stabilize" modes — learned, not hand-set.
//!
//! Run: `cargo run -p ix-duck --example ix_dev_rhythm --features duck`

use std::process::Command;

use ix_duck::open_bench;
use ix_graph::hmm::HiddenMarkovModel;
use ix_graph::markov::MarkovChain;
use ndarray::{Array1, Array2};

const TYPES: [&str; 7] = ["feat", "fix", "docs", "chore", "refactor", "test", "other"];
const NULLS: usize = 2000;
const SEED: u64 = 0x6174_17C0_DE17_C0DE;

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;

    // ── data: IX's commit-type sequence, chronological ───────────────────────────
    let log = Command::new("git")
        .args(["log", "--reverse", "--pretty=format:%s"])
        .output()
        .expect("git log");
    let subjects = String::from_utf8_lossy(&log.stdout);
    let seq: Vec<usize> = subjects.lines().map(type_index).collect();
    let m = TYPES.len();

    let mut marg = vec![0usize; m];
    for &t in &seq {
        marg[t] += 1;
    }
    println!("IX dev rhythm — {} commits over {} types", seq.len(), m);
    println!("   commit-type counts: {}", fmt_counts(&marg));

    // A shallow / single-commit checkout has no transitions: `seq.windows(2)` is empty,
    // so the accuracy divisor is zero and every downstream statistic is NaN. Bail with a
    // clear note rather than print garbage (Codex P2).
    if seq.len() < 30 {
        println!(
            "\ninsufficient git history ({} commits) — this demo needs a full checkout. \
             Run from a complete clone (not a shallow / `--depth 1` one).",
            seq.len()
        );
        return Ok(());
    }

    // ── 1. Markov chain + stationary distribution ────────────────────────────────
    let trans = transition_matrix(&seq, m);
    let chain = MarkovChain::new(trans.clone()).expect("row-stochastic");
    let stationary = chain.stationary_distribution(10_000, 1e-12);
    println!("\nix_graph MarkovChain — stationary distribution (long-run commit mix):");
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| stationary[b].partial_cmp(&stationary[a]).unwrap());
    for &t in &idx {
        println!("   {:<9} {:>5.1}%", TYPES[t], 100.0 * stationary[t]);
    }
    // A couple of notable one-step transitions vs the base rate.
    let base_fix = marg[1] as f64 / seq.len() as f64;
    println!(
        "   P(fix | feat) = {:.1}%  vs base P(fix) = {:.1}%   (does a fix tend to follow a feat?)",
        100.0 * trans[[0, 1]],
        100.0 * base_fix
    );

    // ── 2. validation: does order carry Markov structure? ────────────────────────
    // Statistic: next-type accuracy of "predict argmax P(next|current)". Null: shuffle
    // the sequence (destroys order, keeps the marginal), re-fit, re-score.
    let real_acc = markov_accuracy(&seq, m);
    let base_acc = *marg.iter().max().unwrap() as f64 / seq.len() as f64; // always-predict-mode
    let p = perm_p(&seq, m, real_acc);
    let verdict = if p < 0.05 { "real sequential structure" } else { "NOT beyond i.i.d. (within null)" };
    println!("\nvalidation — is the next commit type predictable from the current? [{NULLS} shuffles]");
    println!("   Markov next-type accuracy = {:.1}%   (always-predict-mode base = {:.1}%)", 100.0 * real_acc, 100.0 * base_acc);
    println!("   permutation p = {p:.4}  → {verdict}");

    // ── 3. hidden phases: Baum-Welch-trained 2-state HMM, decoded by ix_viterbi ───
    let hmm0 = seed_hmm(m);
    let hmm = hmm0.baum_welch(&seq, 200, 1e-6).expect("baum-welch");
    // Decode the most-likely hidden-state path via the bench UDF (trained params).
    let path = viterbi_via_udf(&conn, &hmm, &seq)?;
    let phase_share = |s: usize| 100.0 * path.iter().filter(|&&x| x == s).count() as f64 / path.len() as f64;
    println!("\nix_graph HMM (Baum-Welch) + ix_viterbi — 2 latent development phases:");
    for s in 0..2 {
        println!("   phase {s} ({:.0}% of history) emits: {}", phase_share(s), top_emissions(&hmm.emission, s));
    }
    println!("   recent rhythm (last 40 commits, hidden phase): {}", render_path(&path));

    Ok(())
}

/// Map a commit subject to a type index (collapsing rare/non-conventional → "other").
fn type_index(subject: &str) -> usize {
    let head = subject
        .split(['(', ':', '!', ' '])
        .next()
        .unwrap_or("");
    TYPES.iter().position(|&t| t == head).unwrap_or(m_other())
}
fn m_other() -> usize {
    TYPES.len() - 1
}

/// Row-stochastic first-order transition matrix with +1 Laplace smoothing (keeps the
/// chain ergodic even for a type that never starts a transition).
fn transition_matrix(seq: &[usize], m: usize) -> Array2<f64> {
    let mut c = Array2::<f64>::ones((m, m)); // Laplace +1
    for w in seq.windows(2) {
        c[[w[0], w[1]]] += 1.0;
    }
    for i in 0..m {
        let row_sum: f64 = c.row(i).sum();
        for j in 0..m {
            c[[i, j]] /= row_sum;
        }
    }
    c
}

/// In-sample next-type accuracy of argmax P(next | current).
fn markov_accuracy(seq: &[usize], m: usize) -> f64 {
    let t = transition_matrix(seq, m);
    let pred: Vec<usize> = (0..m)
        .map(|i| (0..m).max_by(|&a, &b| t[[i, a]].partial_cmp(&t[[i, b]]).unwrap()).unwrap())
        .collect();
    let hits = seq.windows(2).filter(|w| pred[w[0]] == w[1]).count();
    hits as f64 / (seq.len() - 1) as f64
}

/// Permutation p: fraction of shuffled sequences whose Markov accuracy ≥ the real one.
fn perm_p(seq: &[usize], m: usize, real_acc: f64) -> f64 {
    let mut rng = SEED | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = seq.to_vec();
    let mut ge = 0usize;
    for _ in 0..NULLS {
        for i in (1..s.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
        if markov_accuracy(&s, m) >= real_acc {
            ge += 1;
        }
    }
    (1 + ge) as f64 / (NULLS + 1) as f64
}

/// A symmetry-broken 2-state HMM seed so Baum-Welch has a gradient to climb.
fn seed_hmm(m: usize) -> HiddenMarkovModel {
    let initial = Array1::from_vec(vec![0.5, 0.5]);
    let transition = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8]).unwrap();
    // State 0 leans slightly to feat/refactor, state 1 to fix/docs/chore — just enough
    // asymmetry to break the symmetry; Baum-Welch learns the rest.
    let mut e = Array2::<f64>::from_elem((2, m), 1.0 / m as f64);
    e[[0, 0]] += 0.05; // feat
    e[[0, 4]] += 0.03; // refactor
    e[[1, 1]] += 0.05; // fix
    e[[1, 2]] += 0.03; // docs
    for s in 0..2 {
        let rs: f64 = e.row(s).sum();
        for j in 0..m {
            e[[s, j]] /= rs;
        }
    }
    HiddenMarkovModel::new(initial, transition, e).expect("valid hmm")
}

/// Decode the most-likely hidden-state path via the `ix_viterbi` bench UDF (trained params).
fn viterbi_via_udf(conn: &ix_duck::Connection, hmm: &HiddenMarkovModel, seq: &[usize]) -> ix_duck::Result<Vec<usize>> {
    let init = vec_json(hmm.initial.as_slice().unwrap());
    let trans = mat_json(&hmm.transition);
    let emis = mat_json(&hmm.emission);
    let obs = format!("[{}]", seq.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    let sql = format!("SELECT state FROM ix_viterbi('{init}','{trans}','{emis}','{obs}') ORDER BY step");
    let mut stmt = conn.prepare(&sql)?;
    let path: Vec<usize> = stmt
        .query_map([], |r| Ok(r.get::<_, i64>(0)? as usize))?
        .collect::<Result<_, _>>()?;
    Ok(path)
}

/// Top-3 commit types a hidden state emits.
fn top_emissions(emission: &Array2<f64>, s: usize) -> String {
    let mut idx: Vec<usize> = (0..TYPES.len()).collect();
    idx.sort_by(|&a, &b| emission[[s, b]].partial_cmp(&emission[[s, a]]).unwrap());
    idx.iter()
        .take(3)
        .map(|&j| format!("{} {:.0}%", TYPES[j], 100.0 * emission[[s, j]]))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_path(path: &[usize]) -> String {
    // phase 0 → 'A', phase 1 → 'B' (last 40 commits)
    path.iter().rev().take(40).rev().map(|&s| if s == 0 { 'A' } else { 'B' }).collect()
}

fn fmt_counts(c: &[usize]) -> String {
    TYPES.iter().zip(c).map(|(t, n)| format!("{t}:{n}")).collect::<Vec<_>>().join("  ")
}
/// Marshal a probability vector to JSON, **renormalized to sum exactly 1** at full
/// precision — `ix_viterbi` validates row-stochasticity tightly, and Baum-Welch output
/// drifts by ~1e-6, so we divide by the sum before formatting.
fn vec_json(v: &[f64]) -> String {
    let s: f64 = v.iter().sum();
    format!("[{}]", v.iter().map(|x| format!("{}", x / s)).collect::<Vec<_>>().join(","))
}
fn mat_json(m: &Array2<f64>) -> String {
    let rows: Vec<String> = (0..m.nrows()).map(|i| vec_json(m.row(i).as_slice().unwrap())).collect();
    format!("[{}]", rows.join(","))
}
