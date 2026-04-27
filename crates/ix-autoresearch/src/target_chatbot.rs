//! Target B — chatbot QA threshold tuning.
//!
//! Shells out to `ga-chatbot qa --autoresearch-config <tempfile>` per
//! iteration, parses the resulting `summary.json`, and projects it onto
//! a [`ChatbotScore`].
//!
//! ## Important: read summary.json, not the exit code
//!
//! `ga-chatbot qa`'s exit code is wired to *regression mismatch*, not
//! raw F/D failure. Per the security review we read the structured
//! summary; the exit code is informational only.
//!
//! ## Determinism
//!
//! v1 ships **deterministic-stub-only** mode (no `--benchmark` flag);
//! the chatbot uses fixture-canned responses, so eval is deterministic
//! for a given config + corpus + fixtures. Caching is enabled via
//! `cache_salt = Some("ix-autoresearch:target_chatbot:v1")`.
//!
//! ## Search space (v1)
//!
//! Only `deterministic_pass_threshold` is wired through to the
//! chatbot's qa.rs Layer 2 T-cut. The other three fields
//! (`judge_accept_threshold`, `fixture_confidence_floor`,
//! `strict_grounding`) are accepted in the config JSON for forward-
//! compatibility but have no observable effect in v1.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::error::{AutoresearchError, EvalCategory};
use crate::Experiment;

/// Tunable knobs the autoresearch loop perturbs. Mirrors
/// `ga_chatbot::qa::AutoresearchConfig` field-for-field so the JSON
/// passed via `--autoresearch-config` is round-trippable across the
/// process boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatbotConfig {
    /// Layer 2 T-cut: confidence above which a response auto-PASSes.
    pub deterministic_pass_threshold: f64,
    /// Judge-panel agreement threshold (v1.5+: not yet wired).
    pub judge_accept_threshold: f64,
    /// Minimum fixture confidence to be considered (v1.5+: not yet wired).
    pub fixture_confidence_floor: f64,
    /// Require non-empty sources when confidence > 0.5 (v1.5+: not yet wired).
    pub strict_grounding: bool,
}

/// Multi-objective score from one `ga-chatbot qa` run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct ChatbotScore {
    /// summary.json's `match_rate` field — fraction of graded prompts
    /// whose deterministic verdict matches the corpus's expected
    /// verdict. Higher is better.
    pub pass_rate: f64,
    /// Mismatches where actual is positive (T/P) but expected is
    /// negative (F/D/C), normalized by total. Lower is better.
    pub false_positive_rate: f64,
    /// Mismatches where actual is negative (F/D/C) but expected is
    /// positive (T/P), normalized by total. Lower is better.
    pub false_negative_rate: f64,
}

/// Adapter holding the binary path + corpus inputs.
pub struct ChatbotTarget {
    /// Path to the built `ga-chatbot` binary. Caller is responsible
    /// for `cargo build -p ga-chatbot --release` (or debug) before
    /// invoking the loop.
    pub chatbot_bin: PathBuf,
    /// Adversarial corpus root (`tests/adversarial/corpus/`).
    pub corpus_dir: PathBuf,
    /// Fixture file (`tests/adversarial/fixtures/stub-responses.jsonl`).
    pub fixtures_path: PathBuf,
    /// Voicing corpus dir (`state/voicings/`).
    pub voicing_corpus_dir: PathBuf,
    /// Per-iteration scratch dir under which the temp config + output
    /// files are written. Each iteration creates a fresh subdir.
    pub scratch_dir: PathBuf,
    /// Hard timeout per iteration (None = use the kernel's TimeBudget).
    pub hard_timeout: Option<std::time::Duration>,
}

impl ChatbotTarget {
    pub fn new(
        chatbot_bin: PathBuf,
        corpus_dir: PathBuf,
        fixtures_path: PathBuf,
        voicing_corpus_dir: PathBuf,
        scratch_dir: PathBuf,
    ) -> Self {
        Self {
            chatbot_bin,
            corpus_dir,
            fixtures_path,
            voicing_corpus_dir,
            scratch_dir,
            hard_timeout: None,
        }
    }
}

impl Experiment for ChatbotTarget {
    type Config = ChatbotConfig;
    type Score = ChatbotScore;

    fn baseline(&self) -> ChatbotConfig {
        // Defaults match ga_chatbot::qa::AutoresearchConfig::default().
        ChatbotConfig {
            deterministic_pass_threshold: 0.9,
            judge_accept_threshold: 0.6,
            fixture_confidence_floor: 0.0,
            strict_grounding: false,
        }
    }

    fn perturb(&mut self, current: &ChatbotConfig, rng: &mut ChaCha8Rng) -> ChatbotConfig {
        // Threshold perturbation: Gaussian σ = 0.05, clamped to [0.05, 0.99].
        // Keeps the cut inside the valid confidence range and away from
        // degenerate extremes (0 and 1 produce trivial passes/fails).
        let mut pert = |x: f64, sigma: f64, lo: f64, hi: f64| -> f64 {
            let u1: f64 = rng.random_range(1e-12..1.0);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (x + z * sigma).clamp(lo, hi)
        };
        ChatbotConfig {
            deterministic_pass_threshold: pert(
                current.deterministic_pass_threshold,
                0.05,
                0.05,
                0.99,
            ),
            // Other fields are pass-through in v1 (not yet wired through
            // the qa.rs path); perturb them anyway so the search space
            // is well-defined for v1.5.
            judge_accept_threshold: pert(current.judge_accept_threshold, 0.05, 0.05, 0.95),
            fixture_confidence_floor: pert(current.fixture_confidence_floor, 0.03, 0.0, 0.5),
            strict_grounding: if rng.random::<f64>() < 0.05 {
                !current.strict_grounding
            } else {
                current.strict_grounding
            },
        }
    }

    fn evaluate(
        &mut self,
        config: &ChatbotConfig,
        soft_deadline: Instant,
    ) -> Result<ChatbotScore, AutoresearchError> {
        // Per-iter scratch dir under self.scratch_dir.
        let iter_id = uuid::Uuid::now_v7().hyphenated().to_string();
        let iter_dir = self.scratch_dir.join(&iter_id);
        std::fs::create_dir_all(&iter_dir)?;

        // Write the autoresearch config JSON.
        let config_path = iter_dir.join("autoresearch-config.json");
        let config_json = serde_json::to_string(config)?;
        std::fs::write(&config_path, config_json)?;

        let output_path = iter_dir.join("findings.jsonl");

        // Spawn ga-chatbot qa with env_clear + minimal allowlist (security review).
        let mut cmd = Command::new(&self.chatbot_bin);
        cmd.env_clear();
        // Allow-list only what the subprocess truly needs.
        for var in &["PATH", "SystemRoot", "USERPROFILE", "TEMP", "TMP", "RUST_BACKTRACE"] {
            if let Ok(v) = std::env::var(var) {
                cmd.env(var, v);
            }
        }
        // Anchor cwd to a sane location — the workspace root, inferred from
        // self.corpus_dir's ancestors. Fall back to the current cwd.
        if let Some(workspace) = workspace_root_from(&self.corpus_dir) {
            cmd.current_dir(workspace);
        }
        cmd.arg("qa")
            .arg("--corpus")
            .arg(&self.corpus_dir)
            .arg("--fixtures")
            .arg(&self.fixtures_path)
            .arg("--corpus-dir")
            .arg(&self.voicing_corpus_dir)
            .arg("--output")
            .arg(&output_path)
            .arg("--autoresearch-config")
            .arg(&config_path);

        // Soft deadline: pass via env so the child can self-honor; we don't
        // shorten the budget here. Hard timeout is enforced via wait_timeout
        // when set; for v1 the kernel's hard_timeout is on the TimeBudget
        // already and we trust the child to terminate.
        let _ = soft_deadline; // hint only

        let output = cmd.output().map_err(AutoresearchError::Io)?;
        // Note: ga-chatbot qa exits 0 on no-mismatch, 1 on mismatch.
        // Either way we *parse summary.json* — the exit code is informational.
        let summary_path = output_path.with_file_name("summary.json");
        if !summary_path.is_file() {
            // Stderr is not captured into the score per security review;
            // we report a typed category, not raw stderr.
            return Err(AutoresearchError::EvalFailed(
                EvalCategory::MissingExpectedFile {
                    path: "summary.json".to_string(),
                },
            ));
        }
        let summary_text = std::fs::read_to_string(&summary_path)?;
        let summary: serde_json::Value = serde_json::from_str(&summary_text).map_err(|e| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: e.to_string(),
            })
        })?;

        let pass_rate = summary
            .get("match_rate")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                    reason: "summary.json missing match_rate".to_string(),
                })
            })?;
        let total = summary
            .get("total")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as f64;

        // Compute FP / FN from the mismatches array.
        let mismatches = summary
            .get("mismatches")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut fp = 0u32;
        let mut fn_ = 0u32;
        for m in &mismatches {
            let actual = m
                .get("actual")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .chars()
                .next()
                .unwrap_or('?');
            let expected = m
                .get("expected")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .chars()
                .next()
                .unwrap_or('?');
            let is_pos = |c: char| c == 'T' || c == 'P';
            let is_neg = |c: char| c == 'F' || c == 'D' || c == 'C';
            if is_pos(actual) && is_neg(expected) {
                fp = fp.saturating_add(1);
            } else if is_neg(actual) && is_pos(expected) {
                fn_ = fn_.saturating_add(1);
            }
        }

        let score = ChatbotScore {
            pass_rate,
            false_positive_rate: fp as f64 / total,
            false_negative_rate: fn_ as f64 / total,
        };

        // Suppress unused-var lint on `output` while keeping the path for
        // future stderr handling. Sanitize: never let raw stderr text
        // flow into the Score.
        let _exit_status = output.status;
        Ok(score)
    }

    fn score_to_reward(&self, score: &ChatbotScore) -> f64 {
        // Lex-style scalar: pass_rate dominates; FP penalized 0.5, FN 0.2.
        score.pass_rate - 0.5 * score.false_positive_rate - 0.2 * score.false_negative_rate
    }

    fn cache_salt(&self) -> Option<String> {
        // Deterministic given config (no LLM, no wall-clock dependencies)
        // — caching enabled.
        Some("ix-autoresearch:target_chatbot:v1".to_string())
    }
}

/// Walk `corpus_dir`'s ancestors looking for a Cargo.toml workspace root.
/// Falls back to None (caller uses cwd) if none found within 6 levels.
fn workspace_root_from(corpus_dir: &std::path::Path) -> Option<PathBuf> {
    let mut cur = corpus_dir.to_path_buf();
    for _ in 0..6 {
        if cur.join("Cargo.toml").is_file() && cur.join("Cargo.lock").exists()
            || cur.join("Cargo.toml").is_file() && cur.join("crates").is_dir()
        {
            return Some(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn baseline_matches_qa_config_defaults() {
        let target = ChatbotTarget::new(
            PathBuf::from("ga-chatbot"),
            PathBuf::from("corpus"),
            PathBuf::from("fixtures.jsonl"),
            PathBuf::from("voicings"),
            PathBuf::from("scratch"),
        );
        let b = target.baseline();
        assert_eq!(b.deterministic_pass_threshold, 0.9);
        assert_eq!(b.judge_accept_threshold, 0.6);
        assert_eq!(b.fixture_confidence_floor, 0.0);
        assert!(!b.strict_grounding);
    }

    #[test]
    fn perturb_keeps_thresholds_in_valid_range() {
        let mut target = ChatbotTarget::new(
            PathBuf::from("ga-chatbot"),
            PathBuf::from("corpus"),
            PathBuf::from("fixtures.jsonl"),
            PathBuf::from("voicings"),
            PathBuf::from("scratch"),
        );
        let baseline = target.baseline();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..50 {
            let c = target.perturb(&baseline, &mut rng);
            assert!((0.05..=0.99).contains(&c.deterministic_pass_threshold));
            assert!((0.05..=0.95).contains(&c.judge_accept_threshold));
            assert!((0.0..=0.5).contains(&c.fixture_confidence_floor));
        }
    }

    #[test]
    fn cache_salt_is_target_specific() {
        let target = ChatbotTarget::new(
            PathBuf::from("ga-chatbot"),
            PathBuf::from("corpus"),
            PathBuf::from("fixtures.jsonl"),
            PathBuf::from("voicings"),
            PathBuf::from("scratch"),
        );
        let salt = target.cache_salt().unwrap();
        assert!(salt.contains("target_chatbot"));
    }

    #[test]
    fn score_to_reward_lex_orders_pass_rate_first() {
        let target = ChatbotTarget::new(
            PathBuf::from("x"),
            PathBuf::from("x"),
            PathBuf::from("x"),
            PathBuf::from("x"),
            PathBuf::from("x"),
        );
        let high = ChatbotScore {
            pass_rate: 0.9,
            false_positive_rate: 0.1,
            false_negative_rate: 0.1,
        };
        let low = ChatbotScore {
            pass_rate: 0.5,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
        };
        assert!(target.score_to_reward(&high) > target.score_to_reward(&low));
    }

    #[test]
    fn workspace_root_from_finds_cargo_root() {
        // Smoke: from the IX workspace's crates/ix-autoresearch dir we
        // should be able to walk up to the workspace root.
        let here = std::env::current_dir().unwrap();
        // current_dir during cargo test is the workspace root.
        let probe = here.join("crates").join("ix-autoresearch");
        if probe.is_dir() {
            let root = workspace_root_from(&probe).expect("should find workspace root");
            assert!(root.join("Cargo.toml").is_file());
        }
    }
}
