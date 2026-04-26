//! Evaluation metrics for supervised learning.
//!
//! Provides regression metrics (MSE, RMSE, R²), classification metrics
//! (accuracy, precision, recall, F1), confusion matrices, ROC curves,
//! and AUC computation.
//!
//! # Example: Confusion Matrix
//!
//! ```
//! use ndarray::array;
//! use ix_supervised::metrics::ConfusionMatrix;
//!
//! let y_true = array![0, 0, 1, 1, 2, 2];
//! let y_pred = array![0, 1, 1, 1, 2, 0];
//!
//! let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 3);
//!
//! // Diagonal = correct predictions
//! assert_eq!(cm.matrix()[[0, 0]], 1); // class 0: 1 correct
//! assert_eq!(cm.matrix()[[1, 1]], 2); // class 1: 2 correct
//! assert_eq!(cm.matrix()[[2, 2]], 1); // class 2: 1 correct
//!
//! // Off-diagonal = errors
//! assert_eq!(cm.matrix()[[0, 1]], 1); // class 0 predicted as 1
//! assert_eq!(cm.matrix()[[2, 0]], 1); // class 2 predicted as 0
//!
//! assert!((cm.accuracy() - 4.0 / 6.0).abs() < 1e-10);
//! ```
//!
//! # Example: ROC/AUC for Binary Classification
//!
//! ```
//! use ndarray::array;
//! use ix_supervised::metrics::{roc_curve, roc_auc};
//!
//! let y_true = array![0, 0, 1, 1, 1];
//! let y_scores = array![0.1, 0.4, 0.35, 0.8, 0.9];
//!
//! let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_scores);
//! let auc = roc_auc(&fpr, &tpr);
//! assert!(auc > 0.7, "AUC should reflect good separation");
//! ```

use ndarray::{Array1, Array2};

// ────────────────────────── Regression Metrics ──────────────────────────

/// Mean Squared Error.
pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|v| v * v).mean().unwrap()
}

/// Root Mean Squared Error.
pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// R² (coefficient of determination).
pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot < 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Mean Absolute Error.
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::mae;
///
/// let y_true = array![1.0, 2.0, 3.0];
/// let y_pred = array![1.5, 2.5, 3.5];
/// assert!((mae(&y_true, &y_pred) - 0.5).abs() < 1e-10);
/// ```
pub fn mae(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|v| v.abs()).mean().unwrap()
}

// ────────────────────── Classification Metrics ──────────────────────────

/// Classification accuracy.
pub fn accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64 {
    let correct: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Precision for a specific class.
pub fn precision(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let tp: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p == class)
        .count();
    let fp: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t != class && p == class)
        .count();
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}

/// Recall for a specific class.
pub fn recall(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let tp: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p == class)
        .count();
    let r#fn: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p != class)
        .count();
    if tp + r#fn == 0 {
        0.0
    } else {
        tp as f64 / (tp + r#fn) as f64
    }
}

/// F1 score for a specific class.
pub fn f1_score(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let p = precision(y_true, y_pred, class);
    let r = recall(y_true, y_pred, class);
    if p + r < 1e-12 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// Averaging strategy for multi-class metrics.
#[derive(Debug, Clone, Copy)]
pub enum Average {
    /// Average per-class metrics, treating all classes equally.
    Macro,
    /// Weighted average by class support (number of true instances).
    Weighted,
}

/// Macro or weighted precision across all classes.
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::{precision_avg, Average};
///
/// let y_true = array![0, 0, 1, 1, 2, 2];
/// let y_pred = array![0, 0, 1, 2, 2, 2];
/// let macro_p = precision_avg(&y_true, &y_pred, Average::Macro);
/// assert!(macro_p > 0.5);
/// ```
pub fn precision_avg(y_true: &Array1<usize>, y_pred: &Array1<usize>, avg: Average) -> f64 {
    average_metric(y_true, y_pred, avg, precision)
}

/// Macro or weighted recall across all classes.
pub fn recall_avg(y_true: &Array1<usize>, y_pred: &Array1<usize>, avg: Average) -> f64 {
    average_metric(y_true, y_pred, avg, recall)
}

/// Macro or weighted F1 across all classes.
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::{f1_avg, Average};
///
/// let y_true = array![0, 0, 1, 1];
/// let y_pred = array![0, 1, 1, 1];
/// let macro_f1 = f1_avg(&y_true, &y_pred, Average::Macro);
/// assert!(macro_f1 > 0.5);
/// ```
pub fn f1_avg(y_true: &Array1<usize>, y_pred: &Array1<usize>, avg: Average) -> f64 {
    average_metric(y_true, y_pred, avg, f1_score)
}

fn average_metric(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    avg: Average,
    metric_fn: fn(&Array1<usize>, &Array1<usize>, usize) -> f64,
) -> f64 {
    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    match avg {
        Average::Macro => {
            let sum: f64 = (0..n_classes).map(|c| metric_fn(y_true, y_pred, c)).sum();
            sum / n_classes as f64
        }
        Average::Weighted => {
            let n = y_true.len() as f64;
            let mut total = 0.0;
            for c in 0..n_classes {
                let support = y_true.iter().filter(|&&t| t == c).count() as f64;
                total += support * metric_fn(y_true, y_pred, c);
            }
            total / n
        }
    }
}

// ──────────────────────── Confusion Matrix ──────────────────────────────

/// Confusion matrix for multi-class classification.
///
/// Row `i`, column `j` counts samples with true label `i` predicted as `j`.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::ConfusionMatrix;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_pred = array![0, 1, 0, 1];
///
/// let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 2);
/// assert_eq!(cm.tp(0), 1);
/// assert_eq!(cm.fp(0), 1); // class 1 misclassified as 0
/// assert_eq!(cm.fn_(0), 1); // class 0 misclassified as 1
/// assert_eq!(cm.tn(0), 1);
/// assert!((cm.accuracy() - 0.5).abs() < 1e-10);
/// ```
pub struct ConfusionMatrix {
    matrix: Array2<usize>,
    n_classes: usize,
}

impl ConfusionMatrix {
    /// Build a confusion matrix from true and predicted labels.
    pub fn from_labels(y_true: &Array1<usize>, y_pred: &Array1<usize>, n_classes: usize) -> Self {
        let mut matrix = Array2::zeros((n_classes, n_classes));
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            matrix[[t, p]] += 1;
        }
        Self { matrix, n_classes }
    }

    /// The raw confusion matrix (rows = true, cols = predicted).
    pub fn matrix(&self) -> &Array2<usize> {
        &self.matrix
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// True positives for a given class.
    pub fn tp(&self, class: usize) -> usize {
        self.matrix[[class, class]]
    }

    /// False positives for a given class (other classes predicted as this class).
    pub fn fp(&self, class: usize) -> usize {
        let col_sum: usize = (0..self.n_classes).map(|r| self.matrix[[r, class]]).sum();
        col_sum - self.tp(class)
    }

    /// False negatives for a given class (this class predicted as other classes).
    pub fn fn_(&self, class: usize) -> usize {
        let row_sum: usize = (0..self.n_classes).map(|c| self.matrix[[class, c]]).sum();
        row_sum - self.tp(class)
    }

    /// True negatives for a given class.
    pub fn tn(&self, class: usize) -> usize {
        let total: usize = self.matrix.sum();
        total - self.tp(class) - self.fp(class) - self.fn_(class)
    }

    /// Overall accuracy from the confusion matrix.
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.n_classes).map(|i| self.matrix[[i, i]]).sum();
        let total: usize = self.matrix.sum();
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }

    /// Per-class precision, recall, F1, and support.
    ///
    /// Returns `(precision, recall, f1, support)` vectors of length `n_classes`.
    ///
    /// ```
    /// use ndarray::array;
    /// use ix_supervised::metrics::ConfusionMatrix;
    ///
    /// let y_true = array![0, 0, 0, 1, 1, 1];
    /// let y_pred = array![0, 0, 1, 0, 1, 1];
    ///
    /// let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 2);
    /// let (prec, rec, f1, support) = cm.classification_report();
    /// assert_eq!(support, vec![3, 3]);
    /// assert!((prec[0] - 2.0 / 3.0).abs() < 1e-10);
    /// ```
    pub fn classification_report(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut prec = Vec::with_capacity(self.n_classes);
        let mut rec = Vec::with_capacity(self.n_classes);
        let mut f1 = Vec::with_capacity(self.n_classes);
        let mut support = Vec::with_capacity(self.n_classes);

        for c in 0..self.n_classes {
            let tp = self.tp(c) as f64;
            let fp = self.fp(c) as f64;
            let fn_ = self.fn_(c) as f64;

            let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f = if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            };

            prec.push(p);
            rec.push(r);
            f1.push(f);
            support.push((tp + fn_) as usize);
        }

        (prec, rec, f1, support)
    }

    /// Format the confusion matrix as a human-readable string.
    pub fn display(&self) -> String {
        let mut s = String::from("Confusion Matrix:\n");
        s.push_str("pred →");
        for c in 0..self.n_classes {
            s.push_str(&format!("\t{}", c));
        }
        s.push('\n');
        for r in 0..self.n_classes {
            s.push_str(&format!("true {}:", r));
            for c in 0..self.n_classes {
                s.push_str(&format!("\t{}", self.matrix[[r, c]]));
            }
            s.push('\n');
        }
        s
    }
}

// ──────────────────────────── ROC / AUC ─────────────────────────────────

/// Compute the ROC curve for binary classification.
///
/// Returns `(fpr, tpr, thresholds)` where `fpr` and `tpr` are vectors of
/// false-positive and true-positive rates at each threshold.
///
/// # Arguments
/// - `y_true` — binary labels (0 or 1)
/// - `y_scores` — predicted probabilities for the positive class
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::roc_curve;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_scores = array![0.1, 0.4, 0.35, 0.8];
///
/// let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_scores);
/// // Curve starts at (0, 0) and ends at (1, 1)
/// assert!(*fpr.first().unwrap() <= 0.0 + 1e-10);
/// assert!(*fpr.last().unwrap() >= 1.0 - 1e-10);
/// ```
pub fn roc_curve(y_true: &Array1<usize>, y_scores: &Array1<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y_true.len();
    let n_pos = y_true.iter().filter(|&&t| t == 1).count() as f64;
    let n_neg = n as f64 - n_pos;

    // Sort by descending score
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let mut fpr = Vec::with_capacity(n + 2);
    let mut tpr = Vec::with_capacity(n + 2);
    let mut thresholds = Vec::with_capacity(n + 2);

    let mut tp_count = 0.0;
    let mut fp_count = 0.0;

    // Start at origin
    fpr.push(0.0);
    tpr.push(0.0);
    thresholds.push(f64::INFINITY);

    let mut prev_score = f64::INFINITY;

    for &idx in &indices {
        let score = y_scores[idx];

        // Emit a point when the threshold changes
        if (score - prev_score).abs() > 1e-15 {
            fpr.push(if n_neg > 0.0 { fp_count / n_neg } else { 0.0 });
            tpr.push(if n_pos > 0.0 { tp_count / n_pos } else { 0.0 });
            thresholds.push(prev_score);
        }

        if y_true[idx] == 1 {
            tp_count += 1.0;
        } else {
            fp_count += 1.0;
        }

        prev_score = score;
    }

    // Final point at (1, 1)
    fpr.push(1.0);
    tpr.push(1.0);
    thresholds.push(prev_score);

    (fpr, tpr, thresholds)
}

/// Area Under the ROC Curve via the trapezoidal rule.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::{roc_curve, roc_auc};
///
/// // Perfect separation
/// let y_true = array![0, 0, 1, 1];
/// let y_scores = array![0.1, 0.2, 0.8, 0.9];
///
/// let (fpr, tpr, _) = roc_curve(&y_true, &y_scores);
/// let auc = roc_auc(&fpr, &tpr);
/// assert!((auc - 1.0).abs() < 1e-10);
/// ```
pub fn roc_auc(fpr: &[f64], tpr: &[f64]) -> f64 {
    let mut area = 0.0;
    for i in 1..fpr.len() {
        let dx = fpr[i] - fpr[i - 1];
        let avg_y = (tpr[i] + tpr[i - 1]) / 2.0;
        area += dx * avg_y;
    }
    area
}

/// Convenience: compute AUC directly from labels and scores.
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::auc_score;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_scores = array![0.1, 0.2, 0.8, 0.9];
/// let auc = auc_score(&y_true, &y_scores);
/// assert!((auc - 1.0).abs() < 1e-10);
/// ```
pub fn auc_score(y_true: &Array1<usize>, y_scores: &Array1<f64>) -> f64 {
    let (fpr, tpr, _) = roc_curve(y_true, y_scores);
    roc_auc(&fpr, &tpr)
}

/// Log loss (binary cross-entropy) for binary classification.
///
/// ```
/// use ndarray::array;
/// use ix_supervised::metrics::log_loss;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_prob = array![0.1, 0.2, 0.8, 0.9];
/// let loss = log_loss(&y_true, &y_prob);
/// assert!(loss < 0.3, "Low loss for good predictions");
/// ```
pub fn log_loss(y_true: &Array1<usize>, y_prob: &Array1<f64>) -> f64 {
    let eps = 1e-15;
    let n = y_true.len() as f64;
    let mut total = 0.0;
    for (&t, &p) in y_true.iter().zip(y_prob.iter()) {
        let p_clipped = p.clamp(eps, 1.0 - eps);
        if t == 1 {
            total -= p_clipped.ln();
        } else {
            total -= (1.0 - p_clipped).ln();
        }
    }
    total / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── Regression ──

    #[test]
    fn test_mse() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mse(&y_true, &y_pred).abs() < 1e-10);
    }

    #[test]
    fn test_rmse() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![2.0, 3.0, 4.0];
        assert!((rmse(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!((r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 2.5];
        assert!((mae(&y_true, &y_pred) - 0.5).abs() < 1e-10);
    }

    // ── Classification ──

    #[test]
    fn test_accuracy() {
        let y_true = array![0, 1, 1, 0];
        let y_pred = array![0, 1, 0, 0];
        assert!((accuracy(&y_true, &y_pred) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1() {
        // True: [1, 1, 0, 0]  Pred: [1, 0, 1, 0]
        // For class 1: TP=1, FP=1, FN=1
        let y_true = array![1, 1, 0, 0];
        let y_pred = array![1, 0, 1, 0];
        assert!((precision(&y_true, &y_pred, 1) - 0.5).abs() < 1e-10);
        assert!((recall(&y_true, &y_pred, 1) - 0.5).abs() < 1e-10);
        assert!((f1_score(&y_true, &y_pred, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_macro_f1() {
        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 0, 1, 1]; // perfect
        let f1 = f1_avg(&y_true, &y_pred, Average::Macro);
        assert!((f1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_precision() {
        let y_true = array![0, 0, 0, 1]; // 3 of class 0, 1 of class 1
        let y_pred = array![0, 0, 0, 1]; // perfect
        let wp = precision_avg(&y_true, &y_pred, Average::Weighted);
        assert!((wp - 1.0).abs() < 1e-10);
    }

    // ── Confusion Matrix ──

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 1, 0, 1];

        let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 2);
        assert_eq!(cm.tp(0), 1);
        assert_eq!(cm.fp(0), 1);
        assert_eq!(cm.fn_(0), 1);
        assert_eq!(cm.tn(0), 1);
        assert!((cm.accuracy() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_multiclass() {
        let y_true = array![0, 0, 1, 1, 2, 2];
        let y_pred = array![0, 0, 1, 2, 2, 2];

        let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 3);
        assert_eq!(cm.tp(0), 2);
        assert_eq!(cm.tp(1), 1);
        assert_eq!(cm.tp(2), 2);
        assert!((cm.accuracy() - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_report() {
        let y_true = array![0, 0, 0, 1, 1, 1];
        let y_pred = array![0, 0, 1, 0, 1, 1];

        let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 2);
        let (prec, rec, f1, support) = cm.classification_report();

        assert_eq!(support, vec![3, 3]);
        // class 0: TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
        assert!((prec[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((rec[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((f1[0] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_display() {
        let y_true = array![0, 1, 1, 0];
        let y_pred = array![0, 1, 0, 0];
        let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 2);
        let disp = cm.display();
        assert!(disp.contains("Confusion Matrix:"));
        assert!(disp.contains("pred"));
    }

    // ── ROC / AUC ──

    #[test]
    fn test_roc_curve_perfect() {
        let y_true = array![0, 0, 1, 1];
        let y_scores = array![0.1, 0.2, 0.8, 0.9];

        let (fpr, tpr, _) = roc_curve(&y_true, &y_scores);
        let auc = roc_auc(&fpr, &tpr);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "Perfect separation should give AUC=1.0, got {}",
            auc
        );
    }

    #[test]
    fn test_roc_curve_random() {
        // Same score for everyone → AUC ~ 0.5
        let y_true = array![0, 0, 1, 1];
        let y_scores = array![0.5, 0.5, 0.5, 0.5];

        let (fpr, tpr, _) = roc_curve(&y_true, &y_scores);
        let auc = roc_auc(&fpr, &tpr);
        assert!(
            (auc - 0.5).abs() < 0.1,
            "Random scores should give AUC≈0.5, got {}",
            auc
        );
    }

    #[test]
    fn test_roc_curve_inverted() {
        // Inverted scores → AUC ~ 0.0
        let y_true = array![0, 0, 1, 1];
        let y_scores = array![0.9, 0.8, 0.2, 0.1];

        let (fpr, tpr, _) = roc_curve(&y_true, &y_scores);
        let auc = roc_auc(&fpr, &tpr);
        assert!(
            auc < 0.1,
            "Inverted scores should give AUC≈0.0, got {}",
            auc
        );
    }

    #[test]
    fn test_auc_score_convenience() {
        let y_true = array![0, 0, 1, 1];
        let y_scores = array![0.1, 0.2, 0.8, 0.9];
        let auc = auc_score(&y_true, &y_scores);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_good_predictions() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.1, 0.1, 0.9, 0.9];
        let loss = log_loss(&y_true, &y_prob);
        assert!(loss < 0.2);
    }

    #[test]
    fn test_log_loss_bad_predictions() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.9, 0.9, 0.1, 0.1];
        let loss_bad = log_loss(&y_true, &y_prob);

        let y_prob_good = array![0.1, 0.1, 0.9, 0.9];
        let loss_good = log_loss(&y_true, &y_prob_good);

        assert!(
            loss_bad > loss_good,
            "Bad predictions should have higher loss"
        );
    }
}
