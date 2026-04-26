//! Extracted code metrics in a flat, ML-friendly format.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Flat code metrics for a single function or file scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetrics {
    /// Function/scope name (empty for top-level file scope).
    pub name: String,
    /// Start line number.
    pub start_line: usize,
    /// End line number.
    pub end_line: usize,

    // --- Complexity ---
    /// Cyclomatic complexity (number of independent paths).
    pub cyclomatic: f64,
    /// Cognitive complexity (human-perceived difficulty).
    pub cognitive: f64,
    /// Number of possible exit points.
    pub n_exits: f64,
    /// Number of function arguments.
    pub n_args: f64,

    // --- Lines of code ---
    /// Source lines of code (non-blank, non-comment).
    pub sloc: f64,
    /// Physical lines of code (all lines).
    pub ploc: f64,
    /// Logical lines of code (statements).
    pub lloc: f64,
    /// Comment lines of code.
    pub cloc: f64,
    /// Blank lines.
    pub blank: f64,

    // --- Halstead metrics ---
    /// Number of distinct operators.
    pub h_u_ops: f64,
    /// Number of distinct operands.
    pub h_u_opnds: f64,
    /// Total number of operators.
    pub h_total_ops: f64,
    /// Total number of operands.
    pub h_total_opnds: f64,
    /// Program vocabulary: u_ops + u_opnds.
    pub h_vocabulary: f64,
    /// Program length: total_ops + total_opnds.
    pub h_length: f64,
    /// Volume: length * log2(vocabulary).
    pub h_volume: f64,
    /// Difficulty: (u_ops / 2) * (total_opnds / u_opnds).
    pub h_difficulty: f64,
    /// Effort: difficulty * volume.
    pub h_effort: f64,
    /// Estimated bugs: volume / 3000.
    pub h_bugs: f64,

    // --- Maintainability ---
    /// Maintainability index (0-171 scale, higher = more maintainable).
    pub maintainability_index: f64,
}

impl CodeMetrics {
    /// Feature names for ML pipeline integration.
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "cyclomatic",
            "cognitive",
            "n_exits",
            "n_args",
            "sloc",
            "ploc",
            "lloc",
            "cloc",
            "blank",
            "h_u_ops",
            "h_u_opnds",
            "h_total_ops",
            "h_total_opnds",
            "h_vocabulary",
            "h_length",
            "h_volume",
            "h_difficulty",
            "h_effort",
            "h_bugs",
            "maintainability_index",
        ]
    }

    /// Convert metrics to a feature vector for ML pipelines.
    pub fn to_features(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.cyclomatic,
            self.cognitive,
            self.n_exits,
            self.n_args,
            self.sloc,
            self.ploc,
            self.lloc,
            self.cloc,
            self.blank,
            self.h_u_ops,
            self.h_u_opnds,
            self.h_total_ops,
            self.h_total_opnds,
            self.h_vocabulary,
            self.h_length,
            self.h_volume,
            self.h_difficulty,
            self.h_effort,
            self.h_bugs,
            self.maintainability_index,
        ])
    }

    /// Number of features in the feature vector.
    pub fn n_features() -> usize {
        20
    }
}

/// Aggregate metrics across multiple scopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetrics {
    /// Path of the analyzed file.
    pub path: String,
    /// Language detected.
    pub language: String,
    /// Top-level (file) metrics.
    pub file_scope: CodeMetrics,
    /// Per-function metrics.
    pub functions: Vec<CodeMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_length() {
        let m = CodeMetrics {
            name: "test".to_string(),
            start_line: 1,
            end_line: 10,
            cyclomatic: 1.0,
            cognitive: 0.0,
            n_exits: 1.0,
            n_args: 2.0,
            sloc: 8.0,
            ploc: 10.0,
            lloc: 5.0,
            cloc: 1.0,
            blank: 1.0,
            h_u_ops: 5.0,
            h_u_opnds: 3.0,
            h_total_ops: 10.0,
            h_total_opnds: 7.0,
            h_vocabulary: 8.0,
            h_length: 17.0,
            h_volume: 51.0,
            h_difficulty: 5.8,
            h_effort: 296.0,
            h_bugs: 0.017,
            maintainability_index: 120.0,
        };
        let features = m.to_features();
        assert_eq!(features.len(), CodeMetrics::n_features());
        assert_eq!(features.len(), CodeMetrics::feature_names().len());
    }
}
