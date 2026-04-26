//! # Metric Metabolism — Layer 3 of the Code Observatory
//!
//! Tracks how code metrics evolve over time by walking git history and
//! computing Layer 1 metrics on each historical version of a file.
//!
//! From the time series of a single metric we derive:
//!
//! - `current`  — the metric at HEAD
//! - `ewma`     — exponentially weighted moving average (smoothed level)
//! - `velocity` — first difference (rate of change per commit)
//! - `acceleration` — second difference (change in rate)
//! - `volatility`   — standard deviation of the series
//! - `trend`        — Rising / Falling / Stable / Volatile classification
//! - `discontinuities` — commit indices where the metric jumps more than
//!   3 sigma from the rolling mean (refactors, rewrites, big merges)
//!
//! The walk uses `git2` revwalk + tree lookup; it never checks files out.
//! For each commit, the file blob is read from the tree and analyzed
//! in-memory via `crate::analyze::analyze_source`.
//!
//! Below a minimum of 5 commits of history we report `confidence: 0.0`
//! — the derived signals are not yet meaningful.

use std::path::{Path, PathBuf};

use git2::{Repository, Sort};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::analyze::{analyze_source, Language};
use crate::metrics::CodeMetrics;

/// Minimum number of historical revisions required before we report a
/// non-zero confidence trajectory.
pub const MIN_COMMITS: usize = 5;

/// Maximum blob size (bytes) accepted when walking history. Blobs larger
/// than this are skipped — a trajectory on a multi-GB file is useless
/// anyway and would OOM the analyzer. 8 MB comfortably covers any
/// real-world source file while blocking adversarial mislabeled data.
pub const MAX_BLOB_SIZE: u64 = 8 * 1024 * 1024;

/// Default EWMA smoothing factor.
pub const DEFAULT_ALPHA: f64 = 0.3;

/// Qualitative classification of a metric's trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Metric is trending upward.
    Rising,
    /// Metric is trending downward.
    Falling,
    /// Metric is stable around a level.
    Stable,
    /// Metric swings widely without a clear direction.
    Volatile,
}

/// Errors that can occur while computing a metric trajectory.
#[derive(Debug, Error)]
pub enum TrajectoryError {
    /// The supplied path is not inside a git repository.
    #[error("not a git repository: {0}")]
    NotAGitRepo(PathBuf),

    /// The file had fewer than [`MIN_COMMITS`] historical versions.
    #[error("insufficient history: {found} commits, need {required}")]
    InsufficientHistory {
        /// Number of commits found touching the file.
        found: usize,
        /// Number of commits required.
        required: usize,
    },

    /// No commits touching the requested file were found.
    #[error("file not found in history: {0}")]
    FileNotFound(String),

    /// Underlying libgit2 error.
    #[error("git error: {0}")]
    Git(#[from] git2::Error),
}

/// Full trajectory of a single metric for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrajectory {
    /// File path inside the repository.
    pub file_path: String,
    /// Name of the metric being tracked (e.g. `"cyclomatic"`).
    pub metric_name: String,
    /// Value at HEAD (the newest observed value).
    pub current: f64,
    /// EWMA-smoothed series (oldest -> newest).
    pub ewma: Vec<f64>,
    /// Velocity series: first difference.
    pub velocity: Vec<f64>,
    /// Acceleration series: second difference.
    pub acceleration: Vec<f64>,
    /// Volatility (standard deviation of the raw series).
    pub volatility: f64,
    /// Qualitative classification of the trajectory.
    pub trend: Trend,
    /// Number of historical commits sampled.
    pub n_commits: usize,
    /// Confidence in the result: 0.0 when below [`MIN_COMMITS`],
    /// otherwise `min(1.0, n_commits / 20)`.
    pub confidence: f64,
    /// Indices (in the ordered, oldest->newest series) where a
    /// discontinuity was detected.
    pub discontinuities: Vec<usize>,
    /// Raw metric values (oldest -> newest). Useful for visualization.
    pub raw: Vec<f64>,
}

/// Compute the trajectory of a metric for a file by walking its git history.
///
/// * `repo_path`    — any path inside (or at the root of) the target repo
/// * `file_path`    — path to the file, relative to the repo root
/// * `metric`       — metric name, e.g. `"cyclomatic"`, `"cognitive"`,
///   `"sloc"`, `"maintainability_index"`
/// * `max_commits`  — walk at most this many commits back in history
///
/// Returns a [`MetricTrajectory`]. If fewer than [`MIN_COMMITS`] revisions
/// touch the file, the returned trajectory has `confidence = 0.0` and
/// `trend = Trend::Stable`. If zero revisions touch the file, an
/// [`TrajectoryError::FileNotFound`] is returned.
pub fn compute_trajectory(
    repo_path: &Path,
    file_path: &str,
    metric: &str,
    max_commits: usize,
) -> Result<MetricTrajectory, TrajectoryError> {
    let repo = Repository::discover(repo_path)
        .map_err(|_| TrajectoryError::NotAGitRepo(repo_path.to_path_buf()))?;

    // Walk history from HEAD, newest first.
    let mut revwalk = repo.revwalk()?;
    revwalk.set_sorting(Sort::TIME | Sort::TOPOLOGICAL)?;
    revwalk.push_head()?;

    // Detect language from the file extension once.
    let lang = Language::from_path(Path::new(file_path))
        .ok_or_else(|| TrajectoryError::FileNotFound(file_path.to_string()))?;

    // Collect (commit_order_newest_first, metric_value) for each commit
    // that contains the file with distinct content. We also dedupe
    // consecutive identical blob ids, so the trajectory reflects real
    // changes rather than every commit.
    let mut values_newest_first: Vec<f64> = Vec::new();
    let mut last_blob_id: Option<git2::Oid> = None;
    let mut walked = 0usize;

    for oid_res in revwalk {
        if walked >= max_commits {
            break;
        }
        let oid = match oid_res {
            Ok(o) => o,
            Err(_) => continue,
        };
        walked += 1;

        let commit = match repo.find_commit(oid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let tree = match commit.tree() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let entry = match tree.get_path(Path::new(file_path)) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if Some(entry.id()) == last_blob_id {
            continue;
        }
        last_blob_id = Some(entry.id());

        let blob = match repo.find_blob(entry.id()) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // Reject blobs larger than MAX_BLOB_SIZE bytes. A multi-GB file
        // (e.g. a mislabeled data file with a .rs extension) would
        // otherwise allocate bytes + UTF-8 string copies and OOM the
        // analyzer. Trajectory analysis of very large files is neither
        // fast nor informative, so skipping them is both safe and sound.
        if (blob.size() as u64) > MAX_BLOB_SIZE {
            continue;
        }
        let content = match std::str::from_utf8(blob.content()) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let metrics = analyze_source(content, lang, Path::new(file_path));
        if let Some(v) = extract_metric(&metrics.file_scope, metric) {
            values_newest_first.push(v);
        }
    }

    if values_newest_first.is_empty() {
        return Err(TrajectoryError::FileNotFound(file_path.to_string()));
    }

    // Reverse to oldest -> newest for time-series semantics.
    let mut raw: Vec<f64> = values_newest_first.into_iter().rev().collect();
    let n = raw.len();
    let current = *raw.last().unwrap();

    // Below the minimum, report a zero-confidence placeholder.
    if n < MIN_COMMITS {
        // Still provide a trivial EWMA so downstream consumers don't NPE.
        let ewma = if n > 0 {
            ix_signal::timeseries::ewma(&raw, DEFAULT_ALPHA)
        } else {
            Vec::new()
        };
        return Ok(MetricTrajectory {
            file_path: file_path.to_string(),
            metric_name: metric.to_string(),
            current,
            ewma,
            velocity: Vec::new(),
            acceleration: Vec::new(),
            volatility: 0.0,
            trend: Trend::Stable,
            n_commits: n,
            confidence: 0.0,
            discontinuities: Vec::new(),
            raw: std::mem::take(&mut raw),
        });
    }

    let ewma = ix_signal::timeseries::ewma(&raw, DEFAULT_ALPHA);
    let velocity = ix_signal::timeseries::difference(&raw, 1);
    let acceleration = ix_signal::timeseries::difference(&velocity, 1);

    let volatility = std_dev(&raw);
    let trend = classify_trend(&raw, &velocity, volatility);
    let discontinuities = detect_discontinuities(&raw);

    let confidence = ((n as f64) / 20.0).min(1.0);

    Ok(MetricTrajectory {
        file_path: file_path.to_string(),
        metric_name: metric.to_string(),
        current,
        ewma,
        velocity,
        acceleration,
        volatility,
        trend,
        n_commits: n,
        confidence,
        discontinuities,
        raw,
    })
}

/// Pull a named metric out of a [`CodeMetrics`] record.
fn extract_metric(m: &CodeMetrics, name: &str) -> Option<f64> {
    Some(match name {
        "cyclomatic" => m.cyclomatic,
        "cognitive" => m.cognitive,
        "n_exits" => m.n_exits,
        "n_args" => m.n_args,
        "sloc" => m.sloc,
        "ploc" => m.ploc,
        "lloc" => m.lloc,
        "cloc" => m.cloc,
        "blank" => m.blank,
        "h_u_ops" => m.h_u_ops,
        "h_u_opnds" => m.h_u_opnds,
        "h_total_ops" => m.h_total_ops,
        "h_total_opnds" => m.h_total_opnds,
        "h_vocabulary" => m.h_vocabulary,
        "h_length" => m.h_length,
        "h_volume" => m.h_volume,
        "h_difficulty" => m.h_difficulty,
        "h_effort" => m.h_effort,
        "h_bugs" => m.h_bugs,
        "maintainability_index" => m.maintainability_index,
        _ => return None,
    })
}

/// Population standard deviation.
fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Classify a series as Rising / Falling / Stable / Volatile.
///
/// The rule is:
///
/// * If volatility (relative to |mean|) exceeds 0.5, call it `Volatile`.
/// * Otherwise, if the mean velocity is within one standard deviation of
///   zero AND the net drift is small compared to the mean, it's `Stable`.
/// * Otherwise, the sign of the mean velocity determines Rising or Falling.
fn classify_trend(raw: &[f64], velocity: &[f64], volatility: f64) -> Trend {
    if velocity.is_empty() {
        return Trend::Stable;
    }
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let abs_mean = mean.abs().max(1e-9);
    let rel_vol = volatility / abs_mean;

    let mean_vel = velocity.iter().sum::<f64>() / velocity.len() as f64;
    let net_drift = raw.last().unwrap() - raw.first().unwrap();
    let rel_drift = (net_drift / abs_mean).abs();

    if rel_vol > 0.5 && rel_drift < 0.2 {
        return Trend::Volatile;
    }

    // Stable if neither drift nor slope is meaningful.
    if rel_drift < 0.05 && mean_vel.abs() < (volatility * 0.25).max(1e-9) {
        return Trend::Stable;
    }

    if mean_vel > 0.0 {
        Trend::Rising
    } else if mean_vel < 0.0 {
        Trend::Falling
    } else {
        Trend::Stable
    }
}

/// Flag indices where the raw series jumps more than 3 sigma away from a
/// rolling mean. The rolling window is 5 samples (or the whole series if
/// shorter). Indices are in the oldest -> newest ordering of `raw`.
fn detect_discontinuities(raw: &[f64]) -> Vec<usize> {
    let mut out = Vec::new();
    if raw.len() < 3 {
        return out;
    }
    let global_sigma = std_dev(raw).max(1e-9);
    let window = 5usize.min(raw.len());

    for i in 1..raw.len() {
        let start = i.saturating_sub(window);
        let prior = &raw[start..i];
        if prior.is_empty() {
            continue;
        }
        let mean = prior.iter().sum::<f64>() / prior.len() as f64;
        let local_sigma = std_dev(prior);
        // Use the larger of local/global sigma to avoid false positives
        // when a window happens to be flat.
        let sigma = local_sigma.max(global_sigma);
        if (raw[i] - mean).abs() > 3.0 * sigma {
            out.push(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use git2::{IndexAddOption, Repository, Signature};
    use std::fs;
    use tempfile::TempDir;

    /// Create a repo, write `contents` to `file`, and commit it.
    fn commit_file(repo: &Repository, dir: &Path, file: &str, contents: &str, msg: &str) {
        let full = dir.join(file);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, contents).unwrap();

        let mut index = repo.index().unwrap();
        index.add_all(["*"], IndexAddOption::DEFAULT, None).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = Signature::now("Test", "test@example.com").unwrap();

        let parent_commit = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .and_then(|oid| repo.find_commit(oid).ok());
        let parents: Vec<&git2::Commit> = parent_commit.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &parents)
            .unwrap();
    }

    fn init_repo() -> (TempDir, Repository) {
        let tmp = TempDir::new().unwrap();
        let repo = Repository::init(tmp.path()).unwrap();
        // Ensure we have a stable default branch.
        {
            let mut cfg = repo.config().unwrap();
            cfg.set_str("user.name", "Test").unwrap();
            cfg.set_str("user.email", "test@example.com").unwrap();
        }
        (tmp, repo)
    }

    #[test]
    fn test_insufficient_history_returns_zero_confidence() {
        let (tmp, repo) = init_repo();
        commit_file(
            &repo,
            tmp.path(),
            "lib.rs",
            "fn main() { println!(\"hi\"); }\n",
            "init",
        );

        let traj = compute_trajectory(tmp.path(), "lib.rs", "cyclomatic", 100).unwrap();
        assert_eq!(traj.n_commits, 1);
        assert_eq!(traj.confidence, 0.0);
        assert_eq!(traj.trend, Trend::Stable);
        assert!(traj.velocity.is_empty());
    }

    #[test]
    fn test_ewma_decay() {
        // Synthetic time series — just verify ix_signal ewma wiring:
        // EWMA should track the series but with lag.
        let data = vec![1.0, 10.0, 10.0, 10.0, 10.0];
        let smoothed = ix_signal::timeseries::ewma(&data, DEFAULT_ALPHA);
        assert_eq!(smoothed.len(), data.len());
        assert!((smoothed[0] - 1.0).abs() < 1e-12);
        // Each step moves ~30% of the way toward 10, so it should still
        // be below 10 after a few steps.
        assert!(smoothed[1] < 10.0);
        assert!(smoothed[1] > smoothed[0]);
        // Monotonic increase toward the new level.
        for i in 1..smoothed.len() {
            assert!(smoothed[i] >= smoothed[i - 1] - 1e-12);
        }
        // And asymptotically approach 10.
        assert!(smoothed[smoothed.len() - 1] > 5.0);
    }

    #[test]
    fn test_discontinuity_detection() {
        // Flat series then a huge jump — should be flagged.
        let mut series = vec![5.0; 10];
        series.push(500.0);
        series.extend(vec![5.0; 5]);
        let jumps = detect_discontinuities(&series);
        assert!(
            jumps.contains(&10),
            "expected discontinuity at index 10, got {jumps:?}"
        );
    }

    #[test]
    fn test_discontinuity_flat_series_none() {
        let series = vec![5.0; 12];
        let jumps = detect_discontinuities(&series);
        assert!(jumps.is_empty(), "flat series should have no jumps");
    }

    #[test]
    fn test_trend_classification() {
        // Rising.
        let rising: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let vr = ix_signal::timeseries::difference(&rising, 1);
        assert_eq!(
            classify_trend(&rising, &vr, std_dev(&rising)),
            Trend::Rising
        );

        // Falling.
        let falling: Vec<f64> = (1..=10).rev().map(|i| i as f64).collect();
        let vf = ix_signal::timeseries::difference(&falling, 1);
        assert_eq!(
            classify_trend(&falling, &vf, std_dev(&falling)),
            Trend::Falling
        );

        // Stable.
        let stable = vec![10.0; 10];
        let vs = ix_signal::timeseries::difference(&stable, 1);
        assert_eq!(
            classify_trend(&stable, &vs, std_dev(&stable)),
            Trend::Stable
        );

        // Volatile — wild swings around a mean, with small net drift.
        let volatile = vec![10.0, 1.0, 20.0, 2.0, 18.0, 3.0, 19.0, 2.0, 17.0, 11.0];
        let vv = ix_signal::timeseries::difference(&volatile, 1);
        assert_eq!(
            classify_trend(&volatile, &vv, std_dev(&volatile)),
            Trend::Volatile
        );
    }

    #[test]
    fn test_trajectory_rising_cyclomatic() {
        // Build a file that grows in cyclomatic complexity over 6 commits.
        let (tmp, repo) = init_repo();
        let versions = [
            "fn f() { let x = 1; }\n",
            "fn f(x: i32) { if x > 0 { let y = 1; } }\n",
            "fn f(x: i32) { if x > 0 { if x > 1 { let y = 1; } } }\n",
            "fn f(x: i32) { if x > 0 { if x > 1 { if x > 2 { let y = 1; } } } }\n",
            "fn f(x: i32) { if x > 0 { if x > 1 { if x > 2 { if x > 3 { let y = 1; } } } } }\n",
            "fn f(x: i32) { if x > 0 { if x > 1 { if x > 2 { if x > 3 { if x > 4 { let y = 1; } } } } } }\n",
        ];
        for (i, v) in versions.iter().enumerate() {
            commit_file(&repo, tmp.path(), "lib.rs", v, &format!("rev {i}"));
        }

        let traj = compute_trajectory(tmp.path(), "lib.rs", "cyclomatic", 100).unwrap();
        assert_eq!(traj.n_commits, versions.len());
        assert!(traj.confidence > 0.0);
        assert_eq!(traj.trend, Trend::Rising);
        assert_eq!(traj.raw.len(), versions.len());
        // Newest value should equal `current`.
        assert!((*traj.raw.last().unwrap() - traj.current).abs() < 1e-9);
        // EWMA same length as raw.
        assert_eq!(traj.ewma.len(), traj.raw.len());
        // Velocity is length n-1, acceleration n-2.
        assert_eq!(traj.velocity.len(), traj.raw.len() - 1);
        assert_eq!(traj.acceleration.len(), traj.raw.len() - 2);
    }

    #[test]
    fn test_not_a_git_repo() {
        let tmp = TempDir::new().unwrap();
        // Do NOT init a repo here.
        let err = compute_trajectory(tmp.path(), "lib.rs", "cyclomatic", 100);
        assert!(matches!(err, Err(TrajectoryError::NotAGitRepo(_))));
    }

    #[test]
    fn test_file_not_found() {
        let (tmp, repo) = init_repo();
        commit_file(&repo, tmp.path(), "other.rs", "fn main() {}\n", "init");
        let err = compute_trajectory(tmp.path(), "missing.rs", "cyclomatic", 100);
        assert!(matches!(err, Err(TrajectoryError::FileNotFound(_))));
    }
}
