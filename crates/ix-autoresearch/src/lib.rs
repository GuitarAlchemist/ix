//! # ix-autoresearch
//!
//! Karpathy-style edit-eval-iterate kernel for IX subsystems.
//!
//! The kernel runs an autonomous experiment loop:
//!
//! 1. Perturb a configuration
//! 2. Evaluate it (with a soft deadline; optional hard timeout)
//! 3. Decide accept / reject via [`Strategy`]
//! 4. Append a `LogEvent::Iteration` to the JSONL log
//! 5. Update best-so-far on `(reward: f64, iteration: usize)`
//! 6. Repeat
//!
//! Three strategies ship in v1: [`Strategy::Greedy`],
//! [`Strategy::SimulatedAnnealing`], [`Strategy::RandomSearch`]. The
//! [`Strategy::Custom`] variant is reserved for v1.5+.
//!
//! Targets implement [`Experiment`]; the kernel is generic over the
//! `Config` and `Score` associated types.
//!
//! ## Quickstart
//!
//! ```no_run
//! use ix_autoresearch::{run_experiment, Strategy, TimeBudget};
//! use std::time::Duration;
//! # struct MyTarget;
//! # impl ix_autoresearch::Experiment for MyTarget {
//! #     type Config = f64; type Score = f64;
//! #     fn baseline(&self) -> f64 { 1.0 }
//! #     fn perturb(&mut self, c: &f64, _rng: &mut rand_chacha::ChaCha8Rng) -> f64 { c + 0.1 }
//! #     fn evaluate(&mut self, c: &f64, _d: std::time::Instant) -> Result<f64, ix_autoresearch::AutoresearchError> {
//! #         Ok(-c * c)
//! #     }
//! #     fn score_to_reward(&self, s: &f64) -> f64 { *s }
//! # }
//! let mut t = MyTarget;
//! let outcome = run_experiment(
//!     &mut t,
//!     Strategy::Greedy,
//!     100,
//!     TimeBudget::soft(Duration::from_secs(5)),
//!     std::path::Path::new("state/autoresearch/runs"),
//!     42,
//! ).unwrap();
//! ```
//!
//! See `docs/plans/2026-04-26-001-feat-ix-autoresearch-edit-eval-iterate-plan.md`
//! for the locked design.

pub mod cache;
pub mod error;
pub mod log;
pub mod milestones;
pub mod policy;
pub mod target_chatbot;
pub mod target_grammar;
pub mod target_optick;
pub mod time_budget;

pub use cache::CacheBridge;
pub use error::{AutoresearchError, EvalCategory};
pub use log::{CostLedger, FsyncPolicy, JsonlLog, LogEvent, SCHEMA_VERSION};
pub use milestones::{
    is_complete_milestone, promote_run, sanitize_text, validate_run_id, validate_slug,
};
pub use policy::{
    ben_ameur_t0, calibrate_initial_temperature, AcceptancePolicy, Decision, GreedyPolicy,
    RandomSearchPolicy, SimulatedAnnealingPolicy, Strategy,
};
pub use target_chatbot::{ChatbotConfig, ChatbotScore, ChatbotTarget};
pub use target_grammar::{GrammarConfig, GrammarScore, GrammarTarget};
pub use target_optick::{OpticKConfig, OpticKScore, OpticKTarget, RebuildMode};
pub use time_budget::TimeBudget;

use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::Utc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{de::DeserializeOwned, Serialize};

/// Per security review: MCP handler rejects `iterations > MCP_ITERATION_CAP`.
/// CLI is unconstrained (use `--iterations 1000000` for explicit local runs).
pub const MCP_ITERATION_CAP: usize = 10_000;

/// Per plan §Hard-kill cascade abort: after N consecutive `HardKilled`,
/// abort the run rather than continue.
pub const HARD_KILL_CASCADE_THRESHOLD: usize = 3;

// ───────────────────────── RunId ─────────────────────────

/// Newtype wrapping a UUIDv7. Lexicographically time-ordered;
/// `RunId` to_string yields a hyphenated 36-char form safe to use as
/// a directory name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(uuid::Uuid);

impl RunId {
    /// Generate a fresh UUIDv7 from the system clock.
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }

    /// Parse + validate a serialized run id. Accepts v7 (production) and
    /// v4 (testing flexibility); rejects every other UUID variant and any
    /// non-UUID string. Same path-traversal defense as
    /// [`validate_run_id`].
    pub fn parse(s: &str) -> Result<Self, AutoresearchError> {
        let canonical = validate_run_id(s)?;
        let parsed = uuid::Uuid::parse_str(&canonical)
            .expect("validate_run_id returned non-UUID string");
        Ok(Self(parsed))
    }

    pub fn as_uuid(&self) -> uuid::Uuid {
        self.0
    }

    pub fn as_string(&self) -> String {
        self.0.hyphenated().to_string()
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_string())
    }
}

// ───────────────────────── Experiment trait ─────────────────────────

/// The autoresearch experiment trait. Targets implement this; the kernel
/// runs the loop.
///
/// ## Determinism contract
///
/// `Config` MUST serialize deterministically — no `HashMap`; use
/// `BTreeMap` if an ordered map is needed. Plain structs already
/// serialize in declaration order.
///
/// `evaluate` SHOULD honor `soft_deadline` by returning
/// [`AutoresearchError::TimedOut`] when the deadline elapses. Adapters
/// that shell out should also enforce a hard kill via the
/// [`TimeBudget::hard_timeout_per_iter`] value passed in.
pub trait Experiment {
    type Config: Clone + std::fmt::Debug + Serialize + DeserializeOwned;
    type Score: Clone + std::fmt::Debug + Serialize + DeserializeOwned + PartialOrd;

    /// Initial configuration the loop perturbs from.
    fn baseline(&self) -> Self::Config;

    /// Generate a candidate by perturbing `current`. Stateless w.r.t.
    /// loop history; the kernel owns acceptance state.
    fn perturb(&mut self, current: &Self::Config, rng: &mut ChaCha8Rng) -> Self::Config;

    /// Evaluate `config`. `soft_deadline` is a hint the eval MAY honor.
    fn evaluate(
        &mut self,
        config: &Self::Config,
        soft_deadline: Instant,
    ) -> Result<Self::Score, AutoresearchError>;

    /// Project a multi-objective `Score` onto a scalar reward (higher
    /// is better) for the strategy's accept/reject decision and for
    /// `(reward, iteration)`-keyed best-so-far tracking.
    fn score_to_reward(&self, score: &Self::Score) -> f64;

    /// Cache-key salt. `Some(s)` opts in to caching keyed by
    /// `blake3(serde_json(config) || s)`; `None` disables caching for
    /// non-deterministic targets. Default `Some("v1")`.
    fn cache_salt(&self) -> Option<String> {
        Some("v1".to_string())
    }

    /// Optional content hash of the eval inputs (e.g. blake3 of a
    /// held-out corpus or the GA index). Logged on `RunStart` so replay
    /// can detect "the input set silently changed between iters" —
    /// Cerebras anti-reward-hacking pattern.
    ///
    /// **MUST be a content hash, not a path hash** — path hashes leak
    /// `C:\Users\<name>\…` into committed milestones.
    fn eval_inputs_hash(&self) -> Option<String> {
        None
    }
}

// ───────────────────────── Outcome ─────────────────────────

/// The result of a completed `run_experiment`.
#[derive(Debug, Clone)]
pub struct Outcome<E: Experiment> {
    pub run_id: RunId,
    pub baseline_config: E::Config,
    pub best_config: E::Config,
    pub best_score: Option<E::Score>,
    pub best_reward: Option<f64>,
    pub best_iteration: Option<usize>,
    pub iterations: usize,
    pub accepted: usize,
    pub log_path: PathBuf,
    pub cost: CostLedger,
    /// `Some(n)` if the run aborted after n consecutive hard kills.
    pub aborted_kills: Option<usize>,
}

// ───────────────────────── run_experiment ─────────────────────────

/// Run a fresh autoresearch experiment.
///
/// Creates `<log_dir>/<run-id>/log.jsonl`, writes `RunStart`, runs
/// `iterations` perturb→evaluate→decide→log cycles, then writes
/// `RunComplete`. Returns the [`Outcome`] including best-so-far.
///
/// Aborts early if [`HARD_KILL_CASCADE_THRESHOLD`] consecutive
/// `HardKilled` errors occur — usually a systemic failure (mmap lock,
/// OOM, GA bug), not a recoverable bad config.
pub fn run_experiment<E: Experiment>(
    experiment: &mut E,
    strategy: Strategy,
    iterations: usize,
    budget: TimeBudget,
    log_dir: &Path,
    seed: u64,
) -> Result<Outcome<E>, AutoresearchError> {
    let run_id = RunId::new();
    let run_dir = log_dir.join(run_id.as_string());
    // Per security review §State directory access: use `create_dir`
    // (fails if exists) on the leaf to defend against UUIDv7
    // pre-creation. Parents may exist.
    std::fs::create_dir_all(log_dir)?;
    std::fs::create_dir(&run_dir).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            AutoresearchError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "run dir already exists at {} (UUIDv7 collision or pre-creation)",
                    run_dir.display()
                ),
            ))
        } else {
            AutoresearchError::Io(e)
        }
    })?;
    let log_path = run_dir.join("log.jsonl");

    let baseline = experiment.baseline();
    let cache = CacheBridge::new(experiment.cache_salt());

    // ── Resolve strategy: calibrate SA T₀ if requested ──
    let resolved_strategy = resolve_sa_calibration(experiment, strategy, &baseline, &budget, seed)?;

    // ── Open log + write RunStart ──
    let mut log = JsonlLog::open(&log_path, FsyncPolicy::default())?;
    let baseline_hash = config_hash(&cache, &baseline)?;
    let target_name = std::any::type_name::<E>().to_string();
    let strategy_value = serde_json::to_value(&resolved_strategy)?;
    let git_sha = best_effort_git_sha();
    let start_event: LogEvent<E::Config, E::Score> = LogEvent::RunStart {
        schema_version: SCHEMA_VERSION,
        run_id: run_id.as_string(),
        timestamp: Utc::now(),
        target: target_name,
        strategy: strategy_value,
        seed,
        git_sha: git_sha.0,
        git_sha_reason: git_sha.1,
        baseline_config: baseline.clone(),
        eval_inputs_hash: experiment.eval_inputs_hash(),
    };
    log.append(&start_event, false)?;

    // ── Inner loop ──
    let mut policy = resolved_strategy.clone().into_policy();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let outcome = run_inner_loop(
        experiment,
        &cache,
        &mut policy,
        &mut log,
        &mut rng,
        &budget,
        baseline.clone(),
        baseline_hash,
        0..iterations,
        // resume_state for fresh runs is None.
        None,
    )?;

    finalize_run(log, &outcome)?;

    Ok(Outcome {
        run_id,
        baseline_config: baseline,
        best_config: outcome.best_config,
        best_score: outcome.best_score,
        best_reward: outcome.best_reward,
        best_iteration: outcome.best_iteration,
        iterations: outcome.iterations,
        accepted: outcome.accepted,
        log_path,
        cost: outcome.cost,
        aborted_kills: outcome.aborted_kills,
    })
}

/// Resume an interrupted run from its existing log file.
///
/// Reads the existing log, reconstructs:
/// - last accepted config (or baseline if no accept yet),
/// - best-so-far `(reward, iteration)`,
/// - SA temperature from the last `Iteration.strategy_state.temperature`
///   (if applicable),
///
/// then continues the loop in the same log file for `additional_iterations`
/// more iterations. The log file is appended to (not replaced).
///
/// **Constraints**: the resumed strategy must be the same shape as the
/// original. Switching from Greedy to SA mid-run is not supported.
pub fn resume_experiment<E: Experiment>(
    experiment: &mut E,
    log_path: &Path,
    additional_iterations: usize,
    budget: TimeBudget,
    seed: u64,
) -> Result<Outcome<E>, AutoresearchError> {
    // Replay the existing log to recover state.
    let events: Vec<LogEvent<E::Config, E::Score>> = log::read_log(log_path)?;
    if events.is_empty() {
        return Err(AutoresearchError::InvalidRunId(format!(
            "{} is empty; nothing to resume",
            log_path.display()
        )));
    }

    let replayed = replay_log(&events)?;
    let run_id = RunId::parse(&replayed.run_id)?;
    let cache = CacheBridge::new(experiment.cache_salt());
    let baseline_config = replayed.baseline;
    let replay_state = replayed.state;

    // Reconstruct strategy with resumed temperature for SA.
    let resumed_strategy = match (replayed.strategy, replayed.last_temperature) {
        (Strategy::SimulatedAnnealing { cooling_rate, .. }, Some(t)) => {
            Strategy::SimulatedAnnealing {
                initial_temperature: Some(t),
                cooling_rate,
            }
        }
        (other, _) => other,
    };

    let mut log = JsonlLog::open(log_path, FsyncPolicy::default())?;
    let mut policy = resumed_strategy.into_policy();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let start_iter = replay_state.last_iteration + 1;
    let end_iter = start_iter + additional_iterations;
    let current_config = replayed
        .last_accepted
        .unwrap_or_else(|| baseline_config.clone());
    let current_hash = config_hash(&cache, &current_config)?;

    let outcome = run_inner_loop(
        experiment,
        &cache,
        &mut policy,
        &mut log,
        &mut rng,
        &budget,
        current_config,
        current_hash,
        start_iter..end_iter,
        Some(replay_state),
    )?;

    finalize_run(log, &outcome)?;

    Ok(Outcome {
        run_id,
        baseline_config,
        best_config: outcome.best_config,
        best_score: outcome.best_score,
        best_reward: outcome.best_reward,
        best_iteration: outcome.best_iteration,
        iterations: outcome.iterations,
        accepted: outcome.accepted,
        log_path: log_path.to_path_buf(),
        cost: outcome.cost,
        aborted_kills: outcome.aborted_kills,
    })
}

// ───────────────────────── Internal helpers ─────────────────────────

#[derive(Debug, Clone, Default)]
struct ReplayState {
    last_iteration: usize,
    accepted_so_far: usize,
    best_iteration: Option<usize>,
    best_reward: Option<f64>,
    cost: CostLedger,
}

/// Internal result of `run_inner_loop` — the parts the loop computes.
/// Public `Outcome` adds `run_id`, `baseline_config`, `log_path` from the
/// caller's setup.
struct InnerOutcome<E: Experiment> {
    best_config: E::Config,
    best_score: Option<E::Score>,
    best_reward: Option<f64>,
    best_iteration: Option<usize>,
    iterations: usize,
    accepted: usize,
    cost: CostLedger,
    aborted_kills: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn run_inner_loop<E: Experiment>(
    experiment: &mut E,
    cache: &CacheBridge,
    policy: &mut Box<dyn AcceptancePolicy>,
    log: &mut JsonlLog,
    rng: &mut ChaCha8Rng,
    budget: &TimeBudget,
    initial_config: E::Config,
    initial_hash: String,
    iter_range: std::ops::Range<usize>,
    resume_state: Option<ReplayState>,
) -> Result<InnerOutcome<E>, AutoresearchError> {
    let resume = resume_state.unwrap_or_default();

    let mut current_config = initial_config.clone();
    let mut current_hash = initial_hash;

    // Best-so-far on (reward, iteration) per plan §Performance review #6.
    // Score is tracked separately for return-to-caller; the kernel's
    // accept/reject only looks at the scalar reward.
    let mut best_iteration: Option<usize> = resume.best_iteration;
    let mut best_reward: Option<f64> = resume.best_reward;
    let mut best_config: E::Config = current_config.clone();
    let mut best_score: Option<E::Score> = None;

    // Reward tracking the *current* (last-accepted) config.
    let current_reward: f64;

    // First, evaluate the *initial* config (baseline for fresh, last
    // accepted for resume) so we have something to compare against. This
    // eval is NOT logged as an `Iteration`; it's part of setup.
    if best_reward.is_none() {
        // Fresh run: evaluate baseline.
        let soft_deadline = budget.soft_deadline_from_now();
        let score = experiment.evaluate(&current_config, soft_deadline)?;
        let r = experiment.score_to_reward(&score);
        current_reward = r;
        best_reward = Some(r);
        best_iteration = Some(0);
        best_config = current_config.clone();
        best_score = Some(score);
    } else {
        // Resume: trust the recovered best as a lower bound for current.
        current_reward = best_reward.unwrap_or(f64::NEG_INFINITY);
    }
    let mut current_reward = current_reward;

    let mut iterations_run = 0usize;
    let mut accepted = resume.accepted_so_far;
    let mut consecutive_kills = 0usize;
    let mut cost = resume.cost;
    let mut aborted_kills: Option<usize> = None;

    for iter in iter_range {
        let iter_start = std::time::Instant::now();
        iterations_run += 1;

        // ── Perturb ──
        let candidate = experiment.perturb(&current_config, rng);
        let candidate_hash = config_hash(cache, &candidate)?;

        // ── Cache lookup (if salt enables it) ──
        let mut cache_hit = false;
        let candidate_score: Result<E::Score, AutoresearchError> =
            match cache.get::<E::Config, E::Score>(&candidate)? {
                Some(cached) => {
                    cache_hit = true;
                    cost.cache_hit_count = cost.cache_hit_count.saturating_add(1);
                    Ok(cached)
                }
                None => {
                    let soft_deadline = budget.soft_deadline_from_now();
                    let result = experiment.evaluate(&candidate, soft_deadline);
                    if let Ok(ref s) = result {
                        let _ = cache.set(&candidate, s);
                    }
                    result
                }
            };

        // ── Score → reward → decision ──
        let elapsed_ms = iter_start.elapsed().as_millis() as u64;
        cost.total_elapsed_ms = cost.total_elapsed_ms.saturating_add(elapsed_ms);

        let (score_opt, reward_opt, decision, error_str) = match candidate_score {
            Ok(ref score) => {
                let cand_reward = experiment.score_to_reward(score);
                consecutive_kills = 0;
                let d = policy.decide(current_reward, cand_reward, iter, rng);
                (Some(score.clone()), Some(cand_reward), d, None)
            }
            Err(AutoresearchError::HardKilled { detail }) => {
                consecutive_kills = consecutive_kills.saturating_add(1);
                cost.eval_failure_count = cost.eval_failure_count.saturating_add(1);
                (
                    None,
                    None,
                    Decision::Reject,
                    Some(format!("hard-killed: {detail}")),
                )
            }
            Err(AutoresearchError::TimedOut(d)) => {
                cost.eval_failure_count = cost.eval_failure_count.saturating_add(1);
                consecutive_kills = 0;
                (
                    None,
                    None,
                    Decision::Reject,
                    Some(format!("timed out after {d:?}")),
                )
            }
            Err(AutoresearchError::EvalFailed(cat)) => {
                cost.eval_failure_count = cost.eval_failure_count.saturating_add(1);
                consecutive_kills = 0;
                (
                    None,
                    None,
                    Decision::Reject,
                    Some(format!("eval failed: {cat}")),
                )
            }
            Err(other) => return Err(other),
        };

        let accepted_flag = matches!(decision, Decision::Accept);
        if accepted_flag {
            accepted = accepted.saturating_add(1);
            if let (Some(ref s), Some(r)) = (&score_opt, reward_opt) {
                current_config = candidate.clone();
                current_hash = candidate_hash.clone();
                current_reward = r;
                if best_reward.map_or(true, |br| r > br) {
                    best_reward = Some(r);
                    best_iteration = Some(iter);
                    best_config = candidate.clone();
                    best_score = Some(s.clone());
                }
            }
        } else {
            cost.rejected_count = cost.rejected_count.saturating_add(1);
        }

        // ── Append iteration event ──
        let iter_event: LogEvent<E::Config, E::Score> = LogEvent::Iteration {
            schema_version: SCHEMA_VERSION,
            iteration: iter,
            timestamp: Utc::now(),
            config: candidate,
            config_hash: candidate_hash,
            score: score_opt,
            reward: reward_opt,
            accepted: accepted_flag,
            previous_hash: Some(current_hash.clone()),
            error: error_str,
            elapsed_ms,
            strategy_state: policy.log_state(),
            cache_hit,
        };
        log.append(&iter_event, accepted_flag)?;

        // ── Hard-kill cascade abort ──
        if consecutive_kills >= HARD_KILL_CASCADE_THRESHOLD {
            aborted_kills = Some(consecutive_kills);
            break;
        }
    }

    // run_id / baseline_config / log_path are filled in by the caller.
    Ok(InnerOutcome {
        best_config,
        best_score,
        best_reward,
        best_iteration,
        iterations: iterations_run,
        accepted,
        cost,
        aborted_kills,
    })
}

fn finalize_run<E: Experiment>(
    mut log: JsonlLog,
    outcome: &InnerOutcome<E>,
) -> Result<(), AutoresearchError> {
    let complete: LogEvent<E::Config, E::Score> = LogEvent::RunComplete {
        schema_version: SCHEMA_VERSION,
        timestamp: Utc::now(),
        iterations: outcome.iterations,
        accepted: outcome.accepted,
        best_iteration: None, // we overwrite from caller-side fields
        best_reward: outcome.best_reward,
        consecutive_kills_at_abort: outcome.aborted_kills,
        cost: Some(outcome.cost.clone()),
    };
    log.append(&complete, false)?;
    log.finalize()?;
    Ok(())
}

/// SA T₀ calibration: when `Strategy::SimulatedAnnealing { initial_temperature: None }`,
/// run 10 baseline-relative perturbations to estimate the mean uphill ΔE
/// and resolve T₀ via Ben-Ameur 2004. Returns the strategy with
/// `initial_temperature: Some(_)` filled in.
fn resolve_sa_calibration<E: Experiment>(
    experiment: &mut E,
    strategy: Strategy,
    baseline: &E::Config,
    budget: &TimeBudget,
    seed: u64,
) -> Result<Strategy, AutoresearchError> {
    match strategy {
        Strategy::SimulatedAnnealing {
            initial_temperature: None,
            cooling_rate,
        } => {
            let baseline_score = experiment
                .evaluate(baseline, budget.soft_deadline_from_now())?;
            let baseline_reward = experiment.score_to_reward(&baseline_score);
            // Distinct seed for calibration so it doesn't deplete the main loop's RNG stream.
            let mut calib_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCA1B_EAFE));
            let t0 = calibrate_initial_temperature(baseline_reward, 10, 0.8, || {
                let candidate = experiment.perturb(baseline, &mut calib_rng);
                experiment
                    .evaluate(&candidate, budget.soft_deadline_from_now())
                    .map(|s| experiment.score_to_reward(&s))
                    .unwrap_or(baseline_reward)
            });
            Ok(Strategy::SimulatedAnnealing {
                initial_temperature: Some(t0),
                cooling_rate,
            })
        }
        other => Ok(other),
    }
}

/// Compute the cache key + a "config hash" string for log entries.
/// When caching is disabled we still produce a hash for log uniqueness.
fn config_hash<C: Serialize>(
    cache: &CacheBridge,
    config: &C,
) -> Result<String, AutoresearchError> {
    if let Some(k) = cache.key_for(config)? {
        return Ok(k);
    }
    // Caching disabled → still hash for log diagnostics.
    let bytes = serde_json::to_vec(config)?;
    Ok(format!("autoresearch:{}", blake3::hash(&bytes).to_hex()))
}

/// Best-effort `git rev-parse HEAD`. Validates output matches a 40-char
/// lowercase hex pattern; on mismatch, returns `(None, Some(reason))`.
fn best_effort_git_sha() -> (Option<String>, Option<String>) {
    use std::process::Command;
    let output = Command::new("git").arg("rev-parse").arg("HEAD").output();
    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
                (Some(s), None)
            } else {
                (None, Some("malformed sha".to_string()))
            }
        }
        Ok(_) => (None, Some("not a git checkout".to_string())),
        Err(_) => (None, Some("git not on PATH".to_string())),
    }
}

/// State recovered by [`replay_log`] from an existing run's JSONL log.
struct ReplayedRun<C> {
    run_id: String,
    baseline: C,
    strategy: Strategy,
    /// `None` iff no iteration was ever accepted.
    last_accepted: Option<C>,
    /// `None` iff no `Iteration.strategy_state.temperature` was logged.
    last_temperature: Option<f64>,
    state: ReplayState,
}

/// Replay the log to recover state needed for resume.
fn replay_log<C, S>(events: &[LogEvent<C, S>]) -> Result<ReplayedRun<C>, AutoresearchError>
where
    C: Clone,
    S: Clone,
{
    let mut run_id: Option<String> = None;
    let mut baseline: Option<C> = None;
    let mut strategy_value: Option<serde_json::Value> = None;
    let mut last_accepted: Option<C> = None;
    let mut last_temp: Option<f64> = None;
    let mut state = ReplayState::default();

    for ev in events {
        match ev {
            LogEvent::RunStart {
                run_id: rid,
                baseline_config,
                strategy,
                ..
            } => {
                run_id = Some(rid.clone());
                baseline = Some(baseline_config.clone());
                strategy_value = Some(strategy.clone());
            }
            LogEvent::Iteration {
                iteration,
                config,
                accepted,
                reward,
                strategy_state,
                ..
            } => {
                state.last_iteration = *iteration;
                if *accepted {
                    last_accepted = Some(config.clone());
                    state.accepted_so_far = state.accepted_so_far.saturating_add(1);
                    if let Some(r) = reward {
                        if state.best_reward.map_or(true, |br| *r > br) {
                            state.best_reward = Some(*r);
                            state.best_iteration = Some(*iteration);
                        }
                    }
                }
                if let Some(s) = strategy_state {
                    if let Some(t) = s.get("temperature").and_then(|v| v.as_f64()) {
                        last_temp = Some(t);
                    }
                }
            }
            LogEvent::RunComplete { cost, .. } => {
                if let Some(c) = cost {
                    state.cost = c.clone();
                }
            }
        }
    }

    let run_id = run_id.ok_or_else(|| {
        AutoresearchError::InvalidRunId("log has no RunStart event".to_string())
    })?;
    let baseline = baseline.ok_or_else(|| {
        AutoresearchError::InvalidRunId("log has no baseline_config in RunStart".to_string())
    })?;
    let strategy: Strategy =
        serde_json::from_value(strategy_value.unwrap_or(serde_json::Value::Null))?;
    Ok(ReplayedRun {
        run_id,
        baseline,
        strategy,
        last_accepted,
        last_temperature: last_temp,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_id_parses_v7_and_round_trips() {
        let id = RunId::new();
        let s = id.as_string();
        let parsed = RunId::parse(&s).unwrap();
        assert_eq!(id.as_uuid(), parsed.as_uuid());
    }

    #[test]
    fn run_id_rejects_non_uuid_strings() {
        assert!(RunId::parse("../../etc").is_err());
        assert!(RunId::parse("").is_err());
        assert!(RunId::parse("not-a-uuid").is_err());
    }

    #[test]
    fn mcp_iteration_cap_is_10k() {
        assert_eq!(MCP_ITERATION_CAP, 10_000);
    }

    #[test]
    fn hard_kill_cascade_threshold_is_3() {
        assert_eq!(HARD_KILL_CASCADE_THRESHOLD, 3);
    }
}
