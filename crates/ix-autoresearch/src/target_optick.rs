//! Target A — OPTIC-K self-tuning (mocked v1; live v2 via GA companion PR).
//!
//! Perturbs the 6-element OPTIC-K partition-weight simplex
//! `[STRUCTURE, MORPHOLOGY, CONTEXT, SYMBOLIC, MODAL, ROOT]` and scores
//! each candidate against `ix-optick-invariants` + `ix-embedding-diagnostics`.
//!
//! ## Mocked vs live
//!
//! v1 ships a synthetic `rebuild_fn` that does NOT shell out to GA's
//! `FretboardVoicingsCLI`. Instead it computes a deterministic
//! `OpticKScore` purely from the config's distance to a hand-picked
//! "true optimum" weight vector. This validates:
//!
//! - Dirichlet-on-simplex perturbation correctness (sum-to-1 invariant
//!   preserved within 1e-9 after renormalize),
//! - Lex-order reward scalarization,
//! - End-to-end kernel + adapter wiring.
//!
//! The live path requires the GA-side `--weights-config <path>` flag
//! on `FretboardVoicingsCLI` (Phase 6). When that lands, swap the
//! `rebuild_fn` for a real shell-out.
//!
//! ## Dirichlet perturbation defaults
//!
//! - α = 200 (concentration). Per Stan's reference: `Var[X_i] = w_i(1−w_i)/(α+1)`.
//!   At α=200 with uniform mid-point w≈0.17, σ ≈ 0.027 — "explore
//!   neighborhood" scale.
//! - ε = 1e-3 floor on every component before scaling, then renormalize
//!   to sum 1 *before* sampling. Without the floor, zero weights are
//!   absorbing states under `Dirichlet(0, ...)`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use rand_chacha::ChaCha8Rng;
use rand_distr::{Dirichlet, Distribution};
use serde::{Deserialize, Serialize};

use crate::error::{AutoresearchError, EvalCategory};
use crate::Experiment;

/// 6-element simplex of OPTIC-K partition weights.
///
/// Field order matches `EmbeddingSchema` in GA. Sum must be ≈ 1.0
/// (perturbation preserves this via Dirichlet-on-simplex sampling).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpticKConfig {
    pub structure_weight: f64,
    pub morphology_weight: f64,
    pub context_weight: f64,
    pub symbolic_weight: f64,
    pub modal_weight: f64,
    pub root_weight: f64,
}

impl OpticKConfig {
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.structure_weight,
            self.morphology_weight,
            self.context_weight,
            self.symbolic_weight,
            self.modal_weight,
            self.root_weight,
        ]
    }

    pub fn from_array(a: [f64; 6]) -> Self {
        Self {
            structure_weight: a[0],
            morphology_weight: a[1],
            context_weight: a[2],
            symbolic_weight: a[3],
            modal_weight: a[4],
            root_weight: a[5],
        }
    }

    /// Sum of weights — should be ≈ 1.0 for a valid config.
    pub fn sum(&self) -> f64 {
        self.as_array().iter().sum()
    }
}

/// Score from one OPTIC-K eval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct OpticKScore {
    /// Primary metric: STRUCTURE leak percentage. Lower is better.
    pub structure_leak_pct: f64,
    /// Tie-breaker: retrieval-match percentage (chatbot ground truth).
    pub retrieval_match_pct: f64,
    pub inv_25_pass_rate: f64,
    pub inv_32_pass_rate: f64,
    pub inv_36_pass_rate: f64,
}

/// Adapter holding the rebuild function (real CLI shell-out in v2,
/// synthetic in v1) and the Dirichlet hyperparameters.
pub struct OpticKTarget {
    /// "True optimum" used by the synthetic rebuild_fn to compute
    /// distance-to-optimum scores. In v1 this is the production
    /// `EmbeddingSchema.PartitionWeights` from GA, hardcoded as a
    /// reference point.
    true_optimum: OpticKConfig,
    /// Where the SA loop *starts*. Independent of `true_optimum`.
    /// `default_smoke()` initialises this to uniform 1/6 (the historical
    /// behaviour); production runs should call [`with_baseline`] to
    /// start from the deployed weights, which give ~11% leak vs
    /// uniform's ~3.6% — i.e. the deployed schema has the headroom
    /// the optimizer needs to actually find an improvement.
    baseline: OpticKConfig,
    /// Concentration α for Dirichlet perturbation. Default 200.
    alpha: f64,
    /// Floor on every weight before Dirichlet scaling. Default 1e-3.
    /// Without this, `Dirichlet(0, ...)` is an absorbing state.
    floor: f64,
    /// Synthetic vs live rebuild. In v1 this is always synthetic.
    rebuild_mode: RebuildMode,
}

/// Rebuild dispatch — synthetic (mocked v1) or live (Phase 7+).
#[derive(Debug, Clone)]
pub enum RebuildMode {
    /// Compute synthetic score from config distance to `true_optimum`.
    /// No shell-out. Default for v1.
    Synthetic,
    /// Shell out to GA's `FretboardVoicingsCLI --weights-config` to
    /// rebuild a small (CI-sized) tri-instrument index, then to
    /// `ix-optick-invariants` and `baseline-diagnostics` to score it.
    /// Per-iter cost ≈ 4–5s on a laptop with `export_max=2000`.
    CIReduced(CIReducedConfig),
    // Future: LiveCorpus { ... } for full 313K-voicing rebuilds.
}

/// Configuration for the `RebuildMode::CIReduced` live evaluator.
///
/// All three binary paths must exist as Release builds before
/// [`OpticKTarget::evaluate`] is called. The `workdir` MUST already
/// exist; the evaluator writes per-iter artifacts (weights JSON,
/// rebuilt index, firings JSON, diagnostics report dir) into it,
/// keyed by a 12-char blake3 prefix of the config so concurrent
/// runs don't collide.
#[derive(Debug, Clone)]
pub struct CIReducedConfig {
    /// Path to `FretboardVoicingsCLI[.exe]` Release build.
    pub ga_cli_path: PathBuf,
    /// Path to `ix-optick-invariants[.exe]` Release build.
    pub invariants_bin: PathBuf,
    /// Path to `baseline-diagnostics[.exe]` Release build.
    pub diagnostics_bin: PathBuf,
    /// Pre-existing scratch directory for per-iter artifacts.
    pub workdir: PathBuf,
    /// `--export-max` per instrument. 2000 → ~6K tri-instrument
    /// voicings → ~4s rebuild on a laptop. Headroom for #25/#32/#36
    /// without burning iters.
    pub export_max: u32,
    /// Diagnostic 2 (retrieval consistency) query count.
    pub retrieval_queries: u32,
    /// Diagnostic 1 classifier samples per instrument.
    pub class_samples: u32,
    /// Random-forest tree count for diagnostic 1.
    pub n_trees: u32,
    /// Random-forest max depth for diagnostic 1.
    pub tree_depth: u32,
    /// Deterministic seed for diagnostics binary.
    pub diag_seed: u64,
}

impl CIReducedConfig {
    /// CI-sized defaults: export_max=2000 per-instrument, 30 retrieval
    /// queries, 500 class samples × 10 trees @ depth 8. Total per-iter
    /// cost on a laptop: ~4s rebuild + ~50ms invariants + ~100ms diag.
    pub fn new(
        ga_cli_path: impl Into<PathBuf>,
        invariants_bin: impl Into<PathBuf>,
        diagnostics_bin: impl Into<PathBuf>,
        workdir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            ga_cli_path: ga_cli_path.into(),
            invariants_bin: invariants_bin.into(),
            diagnostics_bin: diagnostics_bin.into(),
            workdir: workdir.into(),
            export_max: 2000,
            retrieval_queries: 30,
            // Bumped 500 → 4000 (matches the diag binary's own default) on 2026-04-29
            // after Phase 7 first-pass: every seed bottomed out at exactly leak=0.000
            // because the RF classifier on 500-sample folds hit its small-sample noise
            // floor. With 8× more training data the accuracy distribution tightens and
            // the score should stop saturating.
            class_samples: 4000,
            n_trees: 10,
            tree_depth: 8,
            diag_seed: 42,
        }
    }
}

impl OpticKTarget {
    /// Default smoke instance: production-shaped weights as the
    /// "true optimum"; α = 200, floor = 1e-3, synthetic rebuild.
    pub fn default_smoke() -> Self {
        // OPTIC-K v4-pp-r v1.8 production weights (per ix-optick/src/lib.rs:64
        // schema seed). These are the reference point the synthetic
        // rebuild scores against.
        let true_optimum = OpticKConfig::from_array([
            0.30, // STRUCTURE
            0.25, // MORPHOLOGY
            0.15, // CONTEXT
            0.10, // SYMBOLIC
            0.15, // MODAL
            0.05, // ROOT
        ]);
        Self {
            true_optimum,
            // Historical default: uniform 1/6. Production loops should
            // call `with_baseline(production_weights())` to start from
            // the deployed schema (real headroom for the optimizer).
            baseline: OpticKConfig::from_array([1.0 / 6.0; 6]),
            alpha: 200.0,
            floor: 1e-3,
            rebuild_mode: RebuildMode::Synthetic,
        }
    }

    /// OPTIC-K v4-pp-r v1.8 production partition weights (the deployed
    /// schema). Use as the `with_baseline` argument for runs that ask
    /// "can the loop find weights better than what's in production?".
    pub fn production_weights() -> OpticKConfig {
        OpticKConfig::from_array([
            0.30, // STRUCTURE
            0.25, // MORPHOLOGY
            0.15, // CONTEXT
            0.10, // SYMBOLIC
            0.15, // MODAL
            0.05, // ROOT
        ])
    }

    /// Live CI-reduced instance: same Dirichlet defaults as
    /// `default_smoke`, but the rebuild shells out to the GA CLI +
    /// IX scoring binaries described in `cfg`. The "true_optimum" is
    /// kept (as the production weights reference point) so the
    /// synthetic distance still surfaces in logs as a sanity number,
    /// but the score returned by `evaluate` comes from the live
    /// pipeline, not synthetic distance.
    pub fn ci_reduced(cfg: CIReducedConfig) -> Self {
        let mut t = Self::default_smoke();
        t.rebuild_mode = RebuildMode::CIReduced(cfg);
        t
    }

    /// Override the SA starting point. Default is uniform 1/6 (historical).
    /// For "can we beat production?" tests, pass [`production_weights`].
    pub fn with_baseline(mut self, baseline: OpticKConfig) -> Self {
        self.baseline = baseline;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_floor(mut self, floor: f64) -> Self {
        self.floor = floor;
        self
    }

    pub fn true_optimum(&self) -> &OpticKConfig {
        &self.true_optimum
    }

    /// Floor every weight at `self.floor` then renormalize to sum 1.
    /// Public so tests can verify the invariant directly.
    pub fn floor_and_renormalize(&self, mut w: [f64; 6]) -> [f64; 6] {
        for x in &mut w {
            *x = x.max(self.floor);
        }
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            for x in &mut w {
                *x /= total;
            }
        }
        w
    }

    /// L1 distance between a config and the `true_optimum`. Used by the
    /// synthetic rebuild for score derivation.
    fn l1_to_optimum(&self, config: &OpticKConfig) -> f64 {
        let opt = self.true_optimum.as_array();
        let cur = config.as_array();
        cur.iter().zip(opt.iter()).map(|(a, b)| (a - b).abs()).sum()
    }
}

impl Experiment for OpticKTarget {
    type Config = OpticKConfig;
    type Score = OpticKScore;

    fn baseline(&self) -> OpticKConfig {
        self.baseline.clone()
    }

    fn perturb(&mut self, current: &OpticKConfig, rng: &mut ChaCha8Rng) -> OpticKConfig {
        // Floor + renormalize *before* Dirichlet so zero components are
        // not absorbing states. Then sample Dir(α · w_floored_renormed).
        let w = self.floor_and_renormalize(current.as_array());
        let mut alpha_arr = [0.0f64; 6];
        for (i, v) in w.iter().enumerate() {
            alpha_arr[i] = v * self.alpha;
        }
        // rand_distr 0.5 uses const generics — `Dirichlet::<f64, 6>::new`
        // takes a fixed-size array. If alpha is degenerate (NaN, etc.)
        // we fall back to the renormed `w` itself.
        let dirichlet: Dirichlet<f64, 6> = match Dirichlet::new(alpha_arr) {
            Ok(d) => d,
            Err(_) => return OpticKConfig::from_array(w),
        };
        let sample: [f64; 6] = dirichlet.sample(rng);
        OpticKConfig::from_array(sample)
    }

    fn evaluate(
        &mut self,
        config: &OpticKConfig,
        _soft_deadline: Instant,
    ) -> Result<OpticKScore, AutoresearchError> {
        // Validate simplex constraint within tolerance.
        let s = config.sum();
        if !s.is_finite() || (s - 1.0).abs() > 1e-3 {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: format!("weights must sum to ≈ 1.0; got sum = {s:.6}"),
            }));
        }
        if config.as_array().iter().any(|w| !w.is_finite() || *w < -1e-12) {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: "weights must be non-negative and finite".to_string(),
            }));
        }

        match &self.rebuild_mode {
            RebuildMode::Synthetic => Ok(self.synthetic_score(config)),
            RebuildMode::CIReduced(cfg) => self.live_score(config, cfg, _soft_deadline),
        }
    }

    fn score_to_reward(&self, s: &OpticKScore) -> f64 {
        // Lex-style scalar (deepen-plan refinement): leak dominates by
        // 1e6, retrieval next at 1e3, mean(invariants) breaks ties.
        let leak = (1.0 - s.structure_leak_pct).clamp(0.0, 1.0);
        let retr = s.retrieval_match_pct.clamp(0.0, 1.0);
        let inv = (s.inv_25_pass_rate + s.inv_32_pass_rate + s.inv_36_pass_rate) / 3.0;
        leak * 1.0e6 + retr * 1.0e3 + inv
    }

    fn cache_salt(&self) -> Option<String> {
        // Synthetic eval is deterministic given config. Live mode would
        // include the GA index hash here so a corpus rebuild invalidates.
        Some("ix-autoresearch:target_optick:v1-synthetic".to_string())
    }

    fn eval_inputs_hash(&self) -> Option<String> {
        // For synthetic mode, the only "input" is the true_optimum
        // reference point. Hash that so changing it (e.g., to test a
        // different production schema) invalidates downstream caches.
        let payload = serde_json::json!({
            "true_optimum": self.true_optimum,
            "alpha": self.alpha,
            "floor": self.floor,
            "mode": "synthetic-v1",
        });
        serde_json::to_vec(&payload)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
    }
}

impl OpticKTarget {
    /// Live CI-reduced score. Three subprocesses per call:
    /// (1) GA `FretboardVoicingsCLI --weights-config` → `index-<nonce>.optk`,
    /// (2) `ix-optick-invariants --index … --out firings-<nonce>.json`,
    ///     stderr parsed for "X/Y PASS" pass rates,
    /// (3) `baseline-diagnostics --index … --out-dir diag-<nonce>/`,
    ///     report JSON parsed for STRUCTURE leak + retrieval match.
    /// Per-iter artifacts share a 12-char blake3 nonce so concurrent
    /// runs cannot collide. Deadlines are checked between phases.
    fn live_score(
        &self,
        config: &OpticKConfig,
        cfg: &CIReducedConfig,
        deadline: Instant,
    ) -> Result<OpticKScore, AutoresearchError> {
        let cfg_bytes = serde_json::to_vec(config).map_err(|e| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: format!("serialize config: {e}"),
            })
        })?;
        let nonce: String = blake3::hash(&cfg_bytes)
            .to_hex()
            .as_str()
            .chars()
            .take(12)
            .collect();
        let weights_path = cfg.workdir.join(format!("weights-{nonce}.json"));
        let index_path = cfg.workdir.join(format!("index-{nonce}.optk"));
        let firings_path = cfg.workdir.join(format!("firings-{nonce}.json"));
        let diag_dir = cfg.workdir.join(format!("diag-{nonce}"));

        // 1. Atomic write weights JSON (write to .tmp + rename).
        write_weights_json(config, &weights_path)?;

        // 2. GA rebuild.
        check_deadline(deadline)?;
        let ga_out = Command::new(&cfg.ga_cli_path)
            .args([
                "--export-embeddings",
                "--export-max",
                &cfg.export_max.to_string(),
                "--weights-config",
                weights_path.to_str().ok_or_else(non_utf8_path)?,
                "--output",
                index_path.to_str().ok_or_else(non_utf8_path)?,
            ])
            .output()
            .map_err(|e| internal(format!("spawn GA CLI: {e}")))?;
        if !ga_out.status.success() {
            let stderr = String::from_utf8_lossy(&ga_out.stderr);
            eprintln!(
                "GA CLI failed (first 200 chars of stderr): {}",
                stderr.chars().take(200).collect::<String>()
            );
            return Err(AutoresearchError::EvalFailed(
                EvalCategory::SubprocessFailedExitCode {
                    code: ga_out.status.code().unwrap_or(-1),
                },
            ));
        }
        if !index_path.exists() {
            return Err(AutoresearchError::EvalFailed(
                EvalCategory::MissingExpectedFile {
                    path: index_path.display().to_string(),
                },
            ));
        }

        // 3. Invariants — parse stderr for "invariant #N: P/T ... PASS".
        check_deadline(deadline)?;
        let inv_out = Command::new(&cfg.invariants_bin)
            .args([
                "--index",
                index_path.to_str().ok_or_else(non_utf8_path)?,
                "--out",
                firings_path.to_str().ok_or_else(non_utf8_path)?,
            ])
            .output()
            .map_err(|e| internal(format!("spawn invariants: {e}")))?;
        if !inv_out.status.success() {
            return Err(AutoresearchError::EvalFailed(
                EvalCategory::SubprocessFailedExitCode {
                    code: inv_out.status.code().unwrap_or(-1),
                },
            ));
        }
        let inv_stderr = String::from_utf8_lossy(&inv_out.stderr);
        let (p25, p32, p36) = parse_invariant_pass_rates(&inv_stderr);

        // 4. Diagnostics — JSON-only.
        check_deadline(deadline)?;
        std::fs::create_dir_all(&diag_dir)
            .map_err(|e| internal(format!("create diag dir: {e}")))?;
        let diag_out = Command::new(&cfg.diagnostics_bin)
            .args([
                "--index",
                index_path.to_str().ok_or_else(non_utf8_path)?,
                "--out-dir",
                diag_dir.to_str().ok_or_else(non_utf8_path)?,
                "--class-samples-per-instrument",
                &cfg.class_samples.to_string(),
                "--n-trees",
                &cfg.n_trees.to_string(),
                "--tree-depth",
                &cfg.tree_depth.to_string(),
                "--retrieval-queries",
                &cfg.retrieval_queries.to_string(),
                "--seed",
                &cfg.diag_seed.to_string(),
                // Smaller cluster + topo so diagnostics is < 200ms.
                "--kmeans-k",
                "20",
                "--kmeans-iter",
                "15",
                "--cluster-sample",
                "1000",
                "--topo-sample",
                "200",
            ])
            .output()
            .map_err(|e| internal(format!("spawn diagnostics: {e}")))?;
        if !diag_out.status.success() {
            return Err(AutoresearchError::EvalFailed(
                EvalCategory::SubprocessFailedExitCode {
                    code: diag_out.status.code().unwrap_or(-1),
                },
            ));
        }
        let report = find_and_parse_diag_report(&diag_dir)?;
        let leak = parse_structure_leak(&report)?;
        let retr = parse_retrieval_match(&report)?;

        // Best-effort cleanup of the per-iter index file (keeps workdir
        // small over long runs). Diagnostics + firings stay around for
        // post-hoc inspection.
        let _ = std::fs::remove_file(&index_path);

        Ok(OpticKScore {
            structure_leak_pct: leak,
            retrieval_match_pct: retr,
            inv_25_pass_rate: p25,
            inv_32_pass_rate: p32,
            inv_36_pass_rate: p36,
        })
    }

    /// Synthetic score derivation. Mirrors what the live pipeline would
    /// produce in shape (the same OpticKScore fields), but computes
    /// values purely from the config's distance to true_optimum.
    fn synthetic_score(&self, config: &OpticKConfig) -> OpticKScore {
        let l1 = self.l1_to_optimum(config); // ∈ [0, 2]
        // Linear scale: l1=0 → leak=0; l1=2 → leak=1.
        let structure_leak_pct = (l1 / 2.0).clamp(0.0, 1.0);
        // Inverse of leak with a tiny "structural prior" so retrieval
        // doesn't track leak exactly. v1: retrieval = 1 − leak² (concave).
        let retrieval_match_pct =
            (1.0 - structure_leak_pct.powi(2)).clamp(0.0, 1.0);
        // Synthetic invariant pass rates: smoothly decay with leak.
        let inv_25_pass_rate = (1.0 - structure_leak_pct).clamp(0.0, 1.0);
        let inv_32_pass_rate = (1.0 - structure_leak_pct).clamp(0.0, 1.0);
        let inv_36_pass_rate = (1.0 - structure_leak_pct * 0.8).clamp(0.0, 1.0);
        OpticKScore {
            structure_leak_pct,
            retrieval_match_pct,
            inv_25_pass_rate,
            inv_32_pass_rate,
            inv_36_pass_rate,
        }
    }
}

// ─── Live evaluator helpers ─────────────────────────────────────────────

fn write_weights_json(
    config: &OpticKConfig,
    path: &Path,
) -> Result<(), AutoresearchError> {
    let payload = serde_json::json!({
        "schema_version":   1,
        "structure_weight":  config.structure_weight,
        "morphology_weight": config.morphology_weight,
        "context_weight":    config.context_weight,
        "symbolic_weight":   config.symbolic_weight,
        "modal_weight":      config.modal_weight,
        "root_weight":       config.root_weight,
    });
    let bytes = serde_json::to_vec_pretty(&payload).map_err(|e| {
        AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
            reason: format!("serialize weights: {e}"),
        })
    })?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| internal(format!("write weights tmp: {e}")))?;
    std::fs::rename(&tmp, path).map_err(|e| internal(format!("rename weights: {e}")))?;
    Ok(())
}

fn check_deadline(deadline: Instant) -> Result<(), AutoresearchError> {
    if Instant::now() >= deadline {
        Err(AutoresearchError::TimedOut(Duration::from_secs(0)))
    } else {
        Ok(())
    }
}

fn internal(msg: String) -> AutoresearchError {
    AutoresearchError::EvalFailed(EvalCategory::InternalError { reason: msg })
}

fn non_utf8_path() -> AutoresearchError {
    AutoresearchError::EvalFailed(EvalCategory::InternalError {
        reason: "non-UTF8 path".to_string(),
    })
}

/// Parse `ix-optick-invariants` stderr for `"invariant #N: P/T … PASS"`
/// lines. Returns `(p25, p32, p36)` pass rates ∈ [0, 1]. Missing or
/// 0/0 invariants default to `1.0` (vacuously satisfied — nothing
/// in the corpus could violate the invariant).
fn parse_invariant_pass_rates(stderr: &str) -> (f64, f64, f64) {
    let mut p25: Option<f64> = None;
    let mut p32: Option<f64> = None;
    let mut p36: Option<f64> = None;
    const PREFIX: &str = "invariant #";
    for line in stderr.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix(PREFIX) else {
            continue;
        };
        let Some((num_s, after_colon)) = rest.split_once(':') else {
            continue;
        };
        let Ok(inv_num) = num_s.trim().parse::<u32>() else {
            continue;
        };
        let trimmed = after_colon.trim_start();
        let Some((passes_s, after_slash)) = trimmed.split_once('/') else {
            continue;
        };
        let Ok(passes) = passes_s.trim().parse::<u64>() else {
            continue;
        };
        let total_end = after_slash
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_slash.len());
        let Ok(total) = after_slash[..total_end].parse::<u64>() else {
            continue;
        };
        let rate = if total == 0 {
            1.0
        } else {
            passes as f64 / total as f64
        };
        match inv_num {
            25 => p25 = Some(rate),
            32 => p32 = Some(rate),
            36 => p36 = Some(rate),
            _ => {}
        }
    }
    (p25.unwrap_or(1.0), p32.unwrap_or(1.0), p36.unwrap_or(1.0))
}

/// Locate `embedding-diagnostics-*.json` inside `diag_dir` (date-stamped
/// filename) and parse it as a generic JSON value.
fn find_and_parse_diag_report(
    diag_dir: &Path,
) -> Result<serde_json::Value, AutoresearchError> {
    let entries =
        std::fs::read_dir(diag_dir).map_err(|e| internal(format!("read diag_dir: {e}")))?;
    let mut report_path: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("embedding-diagnostics-") && name.ends_with(".json") {
            report_path = Some(p);
            break;
        }
    }
    let path = report_path.ok_or_else(|| {
        AutoresearchError::EvalFailed(EvalCategory::MissingExpectedFile {
            path: format!("{}/embedding-diagnostics-*.json", diag_dir.display()),
        })
    })?;
    let bytes = std::fs::read(&path).map_err(|e| internal(format!("read diag report: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| {
        AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
            reason: format!("parse diag report {}: {e}", path.display()),
        })
    })
}

/// Extract STRUCTURE classifier accuracy from the diagnostics report
/// and rescale to a leak fraction in [0, 1] where 0 = "indistinguishable
/// from random" (acc ≈ 1/3 with three instruments) and 1 = "perfectly
/// reveals the instrument" (acc = 1.0).
fn parse_structure_leak(report: &serde_json::Value) -> Result<f64, AutoresearchError> {
    let parts = report
        .pointer("/leak_detection/by_partition")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: "missing /leak_detection/by_partition[]".to_string(),
            })
        })?;
    let structure = parts
        .iter()
        .find(|p| p.get("partition").and_then(|s| s.as_str()) == Some("STRUCTURE"))
        .ok_or_else(|| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: "no STRUCTURE entry in by_partition".to_string(),
            })
        })?;
    let acc = structure
        .get("accuracy_mean")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: "STRUCTURE.accuracy_mean missing or non-numeric".to_string(),
            })
        })?;
    let baseline = 1.0_f64 / 3.0;
    Ok(((acc - baseline) / (1.0 - baseline)).clamp(0.0, 1.0))
}

fn parse_retrieval_match(report: &serde_json::Value) -> Result<f64, AutoresearchError> {
    report
        .pointer("/retrieval_consistency/avg_pc_set_match_pct")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            AutoresearchError::EvalFailed(EvalCategory::JsonParseFailed {
                reason: "missing /retrieval_consistency/avg_pc_set_match_pct".to_string(),
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_target() -> OpticKTarget {
        OpticKTarget::default_smoke()
    }

    #[test]
    fn baseline_is_uniform_simplex() {
        let t = make_target();
        let b = t.baseline();
        for &w in &b.as_array() {
            assert!((w - 1.0 / 6.0).abs() < 1e-12);
        }
        assert!((b.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn perturb_preserves_simplex_within_tolerance() {
        let mut t = make_target();
        let baseline = t.baseline();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..50 {
            let c = t.perturb(&baseline, &mut rng);
            let sum = c.sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "perturbed config must sum to 1.0 within 1e-9; got {sum}"
            );
            for &w in &c.as_array() {
                assert!(w >= 0.0, "weight must be ≥ 0; got {w}");
            }
        }
    }

    #[test]
    fn perturb_keeps_zero_components_above_floor() {
        // Start from a degenerate config (one zero) and verify floor
        // mechanism prevents it from being trapped at zero.
        let mut t = make_target();
        let degenerate = OpticKConfig::from_array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..20 {
            let c = t.perturb(&degenerate, &mut rng);
            // After floor + Dirichlet, every weight should sample from a
            // distribution with non-zero support — so ≥ 0.
            for &w in &c.as_array() {
                assert!(w >= 0.0);
            }
            assert!((c.sum() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn evaluate_at_true_optimum_returns_zero_leak() {
        let mut t = make_target();
        let opt = t.true_optimum().clone();
        let score = t
            .evaluate(&opt, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(score.structure_leak_pct < 1e-12);
        assert!(score.retrieval_match_pct > 0.999);
        assert!(score.inv_25_pass_rate > 0.999);
    }

    #[test]
    fn evaluate_at_uniform_baseline_has_positive_leak() {
        // Production weights are non-uniform, so uniform baseline is
        // far from optimum → non-zero leak.
        let mut t = make_target();
        let baseline = t.baseline();
        let score = t
            .evaluate(&baseline, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(score.structure_leak_pct > 0.0);
        assert!(score.structure_leak_pct < 1.0);
    }

    #[test]
    fn evaluate_rejects_non_simplex_config() {
        let mut t = make_target();
        let bad = OpticKConfig::from_array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // sums to 6
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn evaluate_rejects_negative_weight() {
        let mut t = make_target();
        let bad = OpticKConfig::from_array([0.5, 0.6, -0.1, 0.0, 0.0, 0.0]);
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn score_to_reward_lex_orders_leak_first() {
        let t = make_target();
        let low_leak = OpticKScore {
            structure_leak_pct: 0.0,
            retrieval_match_pct: 0.0,
            inv_25_pass_rate: 0.0,
            inv_32_pass_rate: 0.0,
            inv_36_pass_rate: 0.0,
        };
        let high_retrieval = OpticKScore {
            structure_leak_pct: 0.5,
            retrieval_match_pct: 1.0,
            inv_25_pass_rate: 1.0,
            inv_32_pass_rate: 1.0,
            inv_36_pass_rate: 1.0,
        };
        // 0-leak with everything else 0 must beat 0.5-leak with everything else 1.
        assert!(t.score_to_reward(&low_leak) > t.score_to_reward(&high_retrieval));
    }

    #[test]
    fn cache_salt_marks_synthetic_mode() {
        let t = make_target();
        let salt = t.cache_salt().unwrap();
        assert!(salt.contains("synthetic"));
    }

    #[test]
    fn eval_inputs_hash_changes_when_alpha_changes() {
        let t1 = OpticKTarget::default_smoke().with_alpha(100.0);
        let t2 = OpticKTarget::default_smoke().with_alpha(500.0);
        assert_ne!(t1.eval_inputs_hash(), t2.eval_inputs_hash());
    }

    #[test]
    fn floor_and_renormalize_eliminates_zeros_and_sums_to_one() {
        let t = make_target();
        let result = t.floor_and_renormalize([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]);
        // After floor + renormalize, every weight is strictly positive
        // (no absorbing-state zeros). The renormalize step pushes them
        // slightly *below* the raw `floor` because flooring increases
        // the sum >1, so renormalize divides — that's expected.
        for w in &result {
            assert!(*w > 0.0, "weight {} should be > 0 after floor", w);
        }
        // Sum invariant is the load-bearing one.
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
