//! Hermes Spike A — learned router head.
//!
//! Trains an L2-regularized softmax regression head over frozen nomic-embed
//! vectors (16 in-scope intents), with a max-prob decline threshold for OOS,
//! and evaluates it on the independent held-out TEST set. Auto-falls back to
//! PCA→64 dims if the full-dim head overfits DEV.
//!
//! Reuses ix primitives: ix_supervised::metrics (accuracy / macro-F1),
//! ix_unsupervised::pca::PCA (dimensionality reduction).
//!
//! Inputs : state/router-spike/embeddings/{train,dev,test}.json  ({label,prompt,vec})
//! Outputs: state/router/learned-head.json        (the deployable head)
//!          state/router-spike/head-eval.json      (metrics report)
//!
//! Baseline comparison (embedding+regex) is produced separately by the C#
//! RoutingEvalHarness — this binary measures the LEARNED head only.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use ix_supervised::metrics::{accuracy, f1_avg, f1_score, Average};
use ix_unsupervised::pca::PCA;
use ix_unsupervised::traits::DimensionReducer;

const NONE: &str = "__none__";
const EMBEDDER: &str = "nomic-embed-text";

#[derive(Deserialize)]
struct Row {
    label: String,
    #[allow(dead_code)]
    prompt: String,
    vec: Vec<f64>,
}

#[derive(Serialize)]
struct LearnedHead {
    schema: String,
    embedder: String,
    dim: usize,
    normalization: String,
    pca: Option<PcaJson>,
    labels: Vec<String>,
    /// weights[feature][class]
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    tau: f64,
    note: String,
}

#[derive(Serialize)]
struct PcaJson {
    mean: Vec<f64>,
    /// components[component][feature]
    components: Vec<Vec<f64>>,
    n_components: usize,
}

#[derive(Serialize)]
struct EvalReport {
    generated_for: String,
    dim_used: usize,
    used_pca: bool,
    tau: f64,
    train_argmax_acc: f64,
    dev_argmax_acc: f64,
    test_inscope_argmax_acc: f64,
    test_inscope_acc_with_tau: f64,
    test_macro_f1: f64,
    test_oos_decline_rate: f64,
    // Cosine-1NN baseline on the SAME held-out test (router core mechanism, no regex).
    baseline_inscope_argmax_acc: f64,
    baseline_inscope_acc_with_tau: f64,
    baseline_macro_f1: f64,
    baseline_oos_decline_rate: f64,
    delta_argmax_pp: f64,
    verdict: String,
    per_intent_f1: Vec<(String, f64)>,
    note: String,
}

fn l2norm(v: &[f64]) -> Vec<f64> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

fn load(dir: &PathBuf, name: &str) -> Vec<Row> {
    let p = dir.join(name);
    let s = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {:?}: {}", p, e));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {:?}: {}", p, e))
}

/// Build (X, y) for in-scope rows only, given the fixed label ordering.
fn inscope_matrix(rows: &[Row], labels: &[String]) -> (Array2<f64>, Array1<usize>) {
    let idx = |l: &str| labels.iter().position(|x| x == l).unwrap();
    let kept: Vec<&Row> = rows.iter().filter(|r| r.label != NONE).collect();
    let d = kept[0].vec.len();
    let mut x = Array2::<f64>::zeros((kept.len(), d));
    let mut y = Array1::<usize>::zeros(kept.len());
    for (i, r) in kept.iter().enumerate() {
        for (j, v) in l2norm(&r.vec).into_iter().enumerate() {
            x[[i, j]] = v;
        }
        y[i] = idx(&r.label);
    }
    (x, y)
}

/// OOS rows as an L2-normed matrix (no labels).
fn oos_matrix(rows: &[Row]) -> Array2<f64> {
    let oos: Vec<&Row> = rows.iter().filter(|r| r.label == NONE).collect();
    if oos.is_empty() {
        return Array2::zeros((0, 0));
    }
    let d = oos[0].vec.len();
    let mut x = Array2::<f64>::zeros((oos.len(), d));
    for (i, r) in oos.iter().enumerate() {
        for (j, v) in l2norm(&r.vec).into_iter().enumerate() {
            x[[i, j]] = v;
        }
    }
    x
}

fn softmax_rows(z: &Array2<f64>) -> Array2<f64> {
    let mut out = z.clone();
    for mut row in out.rows_mut() {
        let m = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - m).exp());
        let s = row.sum();
        if s > 0.0 {
            row.mapv_inplace(|v| v / s);
        }
    }
    out
}

fn logits(x: &Array2<f64>, w: &Array2<f64>, b: &Array1<f64>) -> Array2<f64> {
    let mut z = x.dot(w);
    for mut row in z.rows_mut() {
        row += b;
    }
    z
}

/// argmax index + max probability per row.
fn predict(x: &Array2<f64>, w: &Array2<f64>, b: &Array1<f64>) -> (Array1<usize>, Array1<f64>) {
    let p = softmax_rows(&logits(x, w, b));
    let mut idx = Array1::<usize>::zeros(p.nrows());
    let mut maxp = Array1::<f64>::zeros(p.nrows());
    for (i, row) in p.rows().into_iter().enumerate() {
        let (a, m) = row
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bm), (j, &v)| {
                if v > bm {
                    (j, v)
                } else {
                    (bi, bm)
                }
            });
        idx[i] = a;
        maxp[i] = m;
    }
    (idx, maxp)
}

fn argmax_acc(x: &Array2<f64>, y: &Array1<usize>, w: &Array2<f64>, b: &Array1<f64>) -> f64 {
    let (idx, _) = predict(x, w, b);
    accuracy(y, &idx)
}

/// Cosine-1NN baseline: predict each test row's label from its nearest TRAIN
/// example. Inputs are already L2-normed, so cosine == dot product. This is the
/// production router's CORE mechanism (nearest labeled example), minus the regex
/// hint layer — the clean "does learning beat nearest-neighbor" comparison.
fn cosine_1nn(
    xtr: &Array2<f64>,
    ytr: &Array1<usize>,
    xte: &Array2<f64>,
) -> (Array1<usize>, Array1<f64>) {
    let mut pred = Array1::<usize>::zeros(xte.nrows());
    let mut best_sim = Array1::<f64>::zeros(xte.nrows());
    for (i, q) in xte.rows().into_iter().enumerate() {
        let mut bm = f64::NEG_INFINITY;
        let mut bl = 0usize;
        for (j, r) in xtr.rows().into_iter().enumerate() {
            let s = q.dot(&r);
            if s > bm {
                bm = s;
                bl = ytr[j];
            }
        }
        pred[i] = bl;
        best_sim[i] = bm;
    }
    (pred, best_sim)
}

/// L2-regularized softmax regression via full-batch GD, early-stopped on DEV.
fn train(
    x: &Array2<f64>,
    y: &Array1<usize>,
    n_classes: usize,
    xdev: &Array2<f64>,
    ydev: &Array1<usize>,
    lr: f64,
    epochs: usize,
    lambda: f64,
) -> (Array2<f64>, Array1<f64>) {
    let (n, d) = x.dim();
    let mut w = Array2::<f64>::zeros((d, n_classes));
    let mut b = Array1::<f64>::zeros(n_classes);
    let mut yoh = Array2::<f64>::zeros((n, n_classes));
    for (i, &c) in y.iter().enumerate() {
        yoh[[i, c]] = 1.0;
    }
    let mut best_dev = -1.0;
    let mut best = (w.clone(), b.clone());
    for ep in 0..epochs {
        let p = softmax_rows(&logits(x, &w, &b));
        let diff = &p - &yoh; // (n, k)
        let gw = x.t().dot(&diff) / n as f64 + &(&w * lambda);
        let gb = diff.sum_axis(Axis(0)) / n as f64;
        w = &w - &(&gw * lr);
        b = &b - &(&gb * lr);
        if ep % 10 == 0 || ep == epochs - 1 {
            let dev = argmax_acc(xdev, ydev, &w, &b);
            if dev > best_dev {
                best_dev = dev;
                best = (w.clone(), b.clone());
            }
        }
    }
    best
}

/// Sweep a confidence threshold maximizing balanced score on a calibration set:
/// 0.5*(in-scope correctly-routed at tau) + 0.5*(OOS declined at tau).
fn balanced_threshold(
    in_pred: &Array1<usize>,
    in_true: &Array1<usize>,
    in_score: &Array1<f64>,
    oos_score: &Array1<f64>,
) -> f64 {
    let mut best_t = 0.0;
    let mut best = -1.0;
    let mut t = 0.0;
    while t <= 1.0 + 1e-9 {
        let ins = in_pred
            .iter()
            .zip(in_true.iter())
            .zip(in_score.iter())
            .filter(|((p, tr), s)| **s >= t && p == tr)
            .count() as f64
            / in_pred.len() as f64;
        let oos = if oos_score.is_empty() {
            1.0
        } else {
            oos_score.iter().filter(|&&s| s < t).count() as f64 / oos_score.len() as f64
        };
        let sc = 0.5 * ins + 0.5 * oos;
        if sc > best {
            best = sc;
            best_t = t;
        }
        t += 0.01;
    }
    best_t
}

fn acc_with_tau(
    pred: &Array1<usize>,
    truth: &Array1<usize>,
    score: &Array1<f64>,
    tau: f64,
) -> f64 {
    pred.iter()
        .zip(truth.iter())
        .zip(score.iter())
        .filter(|((p, t), s)| **s >= tau && p == t)
        .count() as f64
        / pred.len() as f64
}

fn decline_rate(score: &Array1<f64>, tau: f64) -> f64 {
    if score.is_empty() {
        return f64::NAN;
    }
    score.iter().filter(|&&s| s < tau).count() as f64 / score.len() as f64
}

/// Map declined predictions (score < tau) to a sentinel class (=> counted wrong).
fn with_decline(pred: &Array1<usize>, score: &Array1<f64>, tau: f64, k: usize) -> Array1<usize> {
    let mut pt = pred.clone();
    for (i, s) in score.iter().enumerate() {
        if *s < tau {
            pt[i] = k;
        }
    }
    pt
}

fn calibrate_tau(
    xdev: &Array2<f64>,
    ydev: &Array1<usize>,
    xoos: &Array2<f64>,
    w: &Array2<f64>,
    b: &Array1<f64>,
) -> f64 {
    let (didx, dmax) = predict(xdev, w, b);
    let oos = if xoos.nrows() > 0 {
        predict(xoos, w, b).1
    } else {
        Array1::zeros(0)
    };
    balanced_threshold(&didx, ydev, &dmax, &oos)
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("..").join(".."); // repo root
    let emb = root.join("state").join("router-spike").join("embeddings");

    let train_rows = load(&emb, "train.json");
    let dev_rows = load(&emb, "dev.json");
    let test_rows = load(&emb, "test.json");

    // Fixed label ordering = sorted unique in-scope labels from TRAIN.
    let mut labels: Vec<String> = train_rows
        .iter()
        .filter(|r| r.label != NONE)
        .map(|r| r.label.clone())
        .collect();
    labels.sort();
    labels.dedup();
    let k = labels.len();
    println!("intents (k={k}): {labels:?}");

    let (xtr, ytr) = inscope_matrix(&train_rows, &labels);
    let (xdev, ydev) = inscope_matrix(&dev_rows, &labels);
    let (xte, yte) = inscope_matrix(&test_rows, &labels);
    let xoos_dev = oos_matrix(&dev_rows);
    let xoos_te = oos_matrix(&test_rows);
    println!(
        "train in-scope={} dev in-scope={} test in-scope={} | dev-oos={} test-oos={}",
        xtr.nrows(),
        xdev.nrows(),
        xte.nrows(),
        xoos_dev.nrows(),
        xoos_te.nrows()
    );

    // ---- Cosine-1NN baseline (full-dim L2 embeddings; computed before the
    //      PCA branch moves xte). The router's core mechanism, minus regex. ----
    let (bpred, bsim) = cosine_1nn(&xtr, &ytr, &xte);
    let base_argmax = accuracy(&yte, &bpred);
    let (bdpred, bdsim) = cosine_1nn(&xtr, &ytr, &xdev);
    let boos_dev = if xoos_dev.nrows() > 0 {
        cosine_1nn(&xtr, &ytr, &xoos_dev).1
    } else {
        Array1::zeros(0)
    };
    let base_tau = balanced_threshold(&bdpred, &ydev, &bdsim, &boos_dev);
    let base_acc_tau = acc_with_tau(&bpred, &yte, &bsim, base_tau);
    let base_pred_tau = with_decline(&bpred, &bsim, base_tau, k);
    let base_macro_f1 = f1_avg(&yte, &base_pred_tau, Average::Macro);
    let boos_te = if xoos_te.nrows() > 0 {
        cosine_1nn(&xtr, &ytr, &xoos_te).1
    } else {
        Array1::zeros(0)
    };
    let base_oos_decline = decline_rate(&boos_te, base_tau);

    // ---- Full-dim head ----
    let (w, b) = train(&xtr, &ytr, k, &xdev, &ydev, 1.0, 2000, 1e-4);
    let tr_acc = argmax_acc(&xtr, &ytr, &w, &b);
    let dev_acc = argmax_acc(&xdev, &ydev, &w, &b);
    println!("[full {}d] train_acc={tr_acc:.3} dev_acc={dev_acc:.3}", xtr.ncols());

    // ---- Auto PCA fallback if overfit ----
    let overfit = tr_acc - dev_acc > 0.15;
    let (used_pca, dim_used, w, b, pca_state, xte_f, xoos_te_f, tau, tr_acc, dev_acc) = if overfit {
        println!("overfit gap {:.3} > 0.15 -> PCA->64 fallback", tr_acc - dev_acc);
        let mut pca = PCA::new(64);
        let xtr_p = pca.fit_transform(&xtr);
        let xdev_p = pca.transform(&xdev);
        let xte_p = pca.transform(&xte);
        let xoos_p = if xoos_te.nrows() > 0 {
            pca.transform(&xoos_te)
        } else {
            Array2::zeros((0, 0))
        };
        let xoos_dev_p = if xoos_dev.nrows() > 0 {
            pca.transform(&xoos_dev)
        } else {
            Array2::zeros((0, 0))
        };
        let (wp, bp) = train(&xtr_p, &ytr, k, &xdev_p, &ydev, 1.0, 2000, 1e-4);
        let trp = argmax_acc(&xtr_p, &ytr, &wp, &bp);
        let devp = argmax_acc(&xdev_p, &ydev, &wp, &bp);
        let tau = calibrate_tau(&xdev_p, &ydev, &xoos_dev_p, &wp, &bp);
        let st = pca.save_state().map(|s| PcaJson {
            mean: s.mean,
            components: s.components,
            n_components: s.n_components,
        });
        println!("[pca 64d] train_acc={trp:.3} dev_acc={devp:.3} tau={tau:.2}");
        (true, 64, wp, bp, st, xte_p, xoos_p, tau, trp, devp)
    } else {
        let tau = calibrate_tau(&xdev, &ydev, &xoos_dev, &w, &b);
        println!("tau={tau:.2}");
        (false, xtr.ncols(), w, b, None, xte, xoos_te, tau, tr_acc, dev_acc)
    };

    // ---- Evaluate LEARNED HEAD on TEST ----
    let (pred, maxp) = predict(&xte_f, &w, &b);
    let test_argmax = accuracy(&yte, &pred);
    let test_acc_tau = acc_with_tau(&pred, &yte, &maxp, tau);
    let pred_tau = with_decline(&pred, &maxp, tau, k);
    let macro_f1 = f1_avg(&yte, &pred_tau, Average::Macro);
    let per_intent_f1: Vec<(String, f64)> = labels
        .iter()
        .enumerate()
        .map(|(c, l)| (l.clone(), f1_score(&yte, &pred_tau, c)))
        .collect();
    let oos_decline = if xoos_te_f.nrows() > 0 {
        decline_rate(&predict(&xoos_te_f, &w, &b).1, tau)
    } else {
        f64::NAN
    };

    // ---- Verdict (on TEST only) ----
    let delta_pp = (test_argmax - base_argmax) * 100.0;
    let min_f1 = per_intent_f1
        .iter()
        .map(|(_, f)| *f)
        .fold(f64::INFINITY, f64::min);
    let beats = delta_pp >= 3.0;
    let f1_ok = min_f1 >= 0.80;
    let oos_ok = oos_decline.is_nan() || base_oos_decline.is_nan() || oos_decline >= base_oos_decline;
    let verdict = if beats && f1_ok && oos_ok {
        "GO — head clears all gates"
    } else if beats {
        "PARTIAL — beats baseline on accuracy but a guardrail fails (see min-F1 / OOS)"
    } else {
        "KILL — head does NOT beat the cosine-1NN baseline by >=3pp on held-out"
    }
    .to_string();

    println!("\n=== TEST (held-out, in-scope=110) — argmax (threshold-free) ===");
    println!("  cosine-1NN baseline : {base_argmax:.3}");
    println!("  LEARNED head        : {test_argmax:.3}   (delta {delta_pp:+.1} pp)");
    println!("=== with decline threshold ===");
    println!(
        "  baseline  acc@tau={base_tau:.2} : {base_acc_tau:.3}  macroF1={base_macro_f1:.3}  OOSdecline={base_oos_decline:.3}"
    );
    println!(
        "  head      acc@tau={tau:.2} : {test_acc_tau:.3}  macroF1={macro_f1:.3}  OOSdecline={oos_decline:.3}"
    );
    println!("  min per-intent F1 (head): {min_f1:.3}");
    println!("\n  VERDICT: {verdict}");
    println!("  worst head intents:");
    let mut sorted = per_intent_f1.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (l, f) in sorted.iter().take(5) {
        println!("    {l:<30} f1={f:.3}");
    }

    // Misclassification dump (head, threshold-free) — diagnostic for which TEST
    // items are genuinely wrong vs ambiguous-label noise vs model error.
    let test_inscope: Vec<&Row> = test_rows.iter().filter(|r| r.label != NONE).collect();
    println!("\n  misclassified in-scope (head argmax):");
    for (i, r) in test_inscope.iter().enumerate() {
        if pred[i] != yte[i] {
            println!(
                "    \"{}\"\n        true={} pred={} p={:.2}",
                r.prompt, labels[yte[i]], labels[pred[i]], maxp[i]
            );
        }
    }

    // ---- Persist head + report ----
    let head = LearnedHead {
        schema: "router-learned-head/0.1".into(),
        embedder: EMBEDDER.into(),
        dim: dim_used,
        normalization: "lowercase+trim then L2".into(),
        pca: pca_state,
        labels: labels.clone(),
        weights: (0..w.nrows()).map(|r| w.row(r).to_vec()).collect(),
        bias: b.to_vec(),
        tau,
        note: "Apply: normalize query -> embed (nomic) -> L2 -> [PCA] -> Wx+b -> softmax; argmax if maxprob>=tau else decline.".into(),
    };
    let head_dir = root.join("state").join("router");
    fs::create_dir_all(&head_dir).unwrap();
    fs::write(
        head_dir.join("learned-head.json"),
        serde_json::to_string_pretty(&head).unwrap(),
    )
    .unwrap();

    let report = EvalReport {
        generated_for: "heldout-test (0b)".into(),
        dim_used,
        used_pca,
        tau,
        train_argmax_acc: tr_acc,
        dev_argmax_acc: dev_acc,
        test_inscope_argmax_acc: test_argmax,
        test_inscope_acc_with_tau: test_acc_tau,
        test_macro_f1: macro_f1,
        test_oos_decline_rate: oos_decline,
        baseline_inscope_argmax_acc: base_argmax,
        baseline_inscope_acc_with_tau: base_acc_tau,
        baseline_macro_f1: base_macro_f1,
        baseline_oos_decline_rate: base_oos_decline,
        delta_argmax_pp: delta_pp,
        verdict: verdict.clone(),
        per_intent_f1,
        note:
            "Baseline = cosine-1NN over TRAIN (router core, no regex hints). Production also adds +~4pp regex hints; a true production baseline needs the C# RoutingEvalHarness on heldout-test.json."
                .into(),
    };
    fs::write(
        root.join("state")
            .join("router-spike")
            .join("head-eval.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("\nwrote state/router/learned-head.json + state/router-spike/head-eval.json");
}
