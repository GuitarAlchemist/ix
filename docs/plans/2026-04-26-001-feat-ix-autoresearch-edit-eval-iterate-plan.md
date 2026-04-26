---
title: ix-autoresearch — Karpathy-style edit-eval-iterate kernel + 3 targets
type: feat
status: active
date: 2026-04-26
origin: docs/brainstorms/2026-04-26-ix-autoresearch-brainstorm.md
reversibility: |
  - two-way-door: kernel internals, target adapter perturbation policies, decision strategy
  - one-way-door: state directory layout (state/autoresearch/{runs,milestones}/), JSONL
    `ExperimentEntry` schema (versioned via `schema_version`), `run_experiment` public
    signature, MCP `ix_autoresearch_run` input_schema, GA-side `--weights-config` JSON
    contract, `Strategy` enum variants exposed to MCP
deepened: 2026-04-26 (research notes folded into Trait, Phases, Risks, Sources)
---

# ix-autoresearch — Karpathy-style edit-eval-iterate kernel + 3 targets

## Enhancement Summary (deepened 2026-04-26)

Seven parallel research agents reviewed this plan after the initial `/ce:plan` pass. Their findings are folded into the relevant sections below; the highlights:

### Corrections

- **`wait-timeout` pin**: plan said `"0.2"` (doesn't exist); the latest is `0.1.5+`. Fixed throughout.
- **`blake3`** is currently per-crate (`ix-autograd`, `ix-agent`); promote to `[workspace.dependencies]` as `blake3 = "1.5"` to share the pin.
- **`uuid v7`** is **stable** in `uuid` 1.10+ (RFC 9562 ratified May 2024) — no `unstable-` feature flag needed in 2026.

### Substantive design refinements

- **Reward scalarization**: weighted-sum can only reach **convex** parts of the Pareto front. Switch Target A's `score_to_reward` from weighted-sum to lex-order on `(1−leak%, retrieval_match, mean_invariants)` (or Chebyshev with utopia point) so non-convex tradeoffs aren't silently excluded. C and B keep weighted-sum (one objective dominates).
- **Trait additions (research-driven)**: `Experiment::cache_salt(&self) -> Option<String>` for determinism-aware caching (replaces TTL-only). Best-so-far tracking moves to `(reward: f64, iteration: usize)` tuple, not `Score: PartialOrd` (PartialOrd is partial; Score with multiple f64 fields has no total order).
- **Strategy enum**: reserve `Strategy::Custom(Box<dyn AcceptancePolicy>)` variant now to prevent a one-way door if v1.5 adds UCB-over-directions or v2 adds Bayesian opt.
- **JSONL schema additions**: every entry carries `schema_version: u32`, plus `run_start` and `run_complete` sentinel events bracket the iteration entries. Per-run metadata (`git_sha`, `seed`, `eval_inputs_hash`, `baseline_config`) lives on the `run_start` event so replay is self-describing.
- **Hard-kill failure semantics**: after N consecutive `HardKilled` events (default 3), abort the run rather than continue. Hung subprocesses usually indicate systemic failure (mmap lock, OOM, GA bug), not a recoverable bad config.
- **Dirichlet perturbation defaults**: α = 200 (concentration) for "small" perturbations; floor every weight at ε = 1e-3 before scaling so zero weights aren't absorbing states. Aitchison-style perturbation is the formal algebra.
- **SA defaults**: geometric cooling (T_{n+1} = 0.95·T_n); calibrate T₀ via 10 random-perturbation samples targeting 80% initial uphill acceptance.
- **Atomic milestone promote**: copy to `milestones/<slug>.tmp/`, atomic `rename` (Windows: `MoveFileEx`), then write `.complete` sentinel. `list` skips dirs without `.complete`.
- **JSONL torn-write defense**: serialize each entry to `Vec<u8>`, single `write_all(&[buf, b"\n"].concat())`, then `sync_all`. Replay parses each line under `serde_json::from_str`; trailing parse failure → discard last line.

### Pattern-alignment fixes

- **Cargo.toml conventions**: use `version.workspace = true` etc. (copy `ix-quality-trend`); not the `ix-optick-invariants` outlier with hardcoded values.
- **Module layout**: flatten `src/targets/{mod,grammar,chatbot,optick}.rs` to siblings `src/target_grammar.rs`, `src/target_chatbot.rs`, `src/target_optick.rs` (matches `ix-bracelet`).
- **Test layout**: collapse 6 separate test files to 3 thematic files (`tests/kernel.rs`, `tests/targets.rs`, `tests/chatbot_e2e.rs`).

### Performance refinements

- **Target A budget split** by `OpticKMode`: CI-reduced ≈ 40–60 s/iter, live ≈ 160–180 s/iter. Plan previously cited a single 140s figure that conflated the two.
- **Target B mode clarified**: 30 s/iter assumes deterministic-stub-only; judge-panel-in-loop bumps to 60–120 s/iter. v1 ships deterministic-only; judge-panel mode is a v1.5 flag.
- **fsync-on-checkpoint**, not per-line. fsync every N=10 entries OR on accept; saves ~450 ms/run with no recovery loss (replay already handles trailing partial lines).

### Karpathy autoresearch — what it actually does

For posterity (per best-practices research): Karpathy's autoresearch uses **greedy hill-climbing** on a single scalar (`val_bpb`, lower-is-better). No SA, no bandit. Better-or-equal commits accepted; everything else `git reset`-reverted. IX's choice of Greedy + SA is *richer* than Karpathy's. Cerebras's "[How to stop your autoresearch loop from cheating](https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating)" enumerates failure modes (agent drift, context pollution, reward hacking) — IX is mostly immune to the first two (numeric perturbation, no LLM context) but reward-hacking-via-metric-definition applies; mitigated by `eval_inputs_hash` per iteration and fixed held-out corpora.

## Overview

A new `ix-autoresearch` crate that runs an autonomous experiment loop
(perturb → evaluate → accept-or-revert → log → repeat) over an
`Experiment` trait, plus three target adapters wiring it to real IX
subsystems. The framework is exposed three ways from the *same* kernel —
CLI binary, MCP tool, and demo target — so external agents (GA, TARS,
Claude Code) can drive their own self-improvement loops without
duplicating the kernel.

The v1 search engine is **numeric** (greedy hill-climbing or simulated
annealing); the trait shape leaves a clean seam for LLM-driven
perturbations as a v2 plug-in. The crate ships with three target
adapters in priority order: Target C (grammar weights, sub-second smoke
test), Target B (chatbot QA threshold tuning, ~30 s/iter), Target A
(OPTIC-K partition weights, ~140 s/iter, requires a GA-side companion
PR).

## Problem Statement

IX has primitives for evaluation (`ix-optick-invariants`,
`ix-embedding-diagnostics`, `ix-quality-trend`, `ga-chatbot qa`),
search (`ix-rl::bandit`, `ix-evolution`, `ix-optimize`), and storage
(`ix-cache`), but no glue layer that composes them into an autonomous
loop. Every "tune the embedding overnight" or "find better grammar
weights" today requires hand-rolled scripts and ad-hoc logging. We've
shipped the eval substrate; we now need the loop driver that exploits
it (see brainstorm: `docs/brainstorms/2026-04-26-ix-autoresearch-brainstorm.md`).

The parallel motivation is the autoresearch *story* — Karpathy's
nanochat-autoresearch loop is a compelling demonstration of agent
autonomy. IX has a stronger eval substrate than autoresearch's
hand-built one, and pure-Rust reproducibility we can't get from a
Python+CUDA loop. The artifact this delivers is a visible overnight
metric move (e.g., "STRUCTURE leak % dropped from X to Y while we
slept"), suitable for ecosystem storytelling.

## Proposed Solution

A small kernel crate plus three target adapters and three surface
exposures, all decided in the brainstorm and locked here.

The kernel defines an `Experiment` trait whose implementations encode
*what to perturb*, *how to evaluate*, and *how to scalarize*; the
kernel owns the loop, the JSONL log, the strategy (Greedy / SA), and
the time-budget watchdog. Each target adapter is a separate module
inside the crate that implements `Experiment` for a specific
subsystem. The CLI dispatcher and the MCP handler both call the same
`run_experiment` core.

## Technical Approach

### Architecture

```
                    ┌─────────────────────────────┐
                    │  ix-autoresearch            │
                    │                             │
                    │  Experiment trait           │
                    │     ↓                       │
                    │  run_experiment(...) ──────┐│
                    │                            ││
                    │  Strategy{Greedy,SA}       ││
                    │  JsonlLog                  ││
                    │  TimeBudget                ││
                    │  CacheBridge (ix-cache)    ││
                    └────────────────────────────┼┘
                                                 │
        ┌───────────────────────┬────────────────┴┐
        ↓                       ↓                 ↓
  ┌──────────┐           ┌──────────┐       ┌──────────┐
  │  CLI     │           │  MCP     │       │  Targets │
  │ ix-      │           │ ix_auto- │       │  C  B  A │
  │ auto-    │           │ research_│       └──────────┘
  │ research │           │ run      │
  └──────────┘           └──────────┘
```

The kernel is pure-Rust, sequential v1, and parameterized over `Experiment`.
All three surfaces dispatch to the same `run_experiment(experiment,
config, strategy, log_path, budget) -> Outcome` function — no parallel
implementations.

### Crate Layout

**(deepen-plan refinement)**: layout flattened per workspace conventions. `ix-bracelet` (9 sibling files, no subdirs) and `ix-quality-trend` (4 modules, flat) are the templates; the brainstorm's `targets/` subdir was an accidental divergence. Test files collapsed from 6 → 3 thematic files matching `ix-skill`'s pattern.

```
crates/ix-autoresearch/
├── Cargo.toml
├── README.md                  # 30-line quickstart (raises bar; alternative is //! lib.rs doc)
├── src/
│   ├── lib.rs                 # Experiment trait, RunId, run_experiment, Outcome
│   ├── error.rs               # AutoresearchError
│   ├── policy.rs              # Strategy enum + AcceptancePolicy trait + T₀ calibrator
│   ├── log.rs                 # LogEvent, JsonlLog (append-before-update, atomic-write, fsync-on-checkpoint)
│   ├── time_budget.rs         # TimeBudget, hard-timeout watchdog (wait-timeout)
│   ├── cache.rs               # config_hash + ix_cache::Cache wrapper, salt-aware
│   ├── milestones.rs          # promote_run (atomic rename + .complete sentinel)
│   ├── target_grammar.rs      # Target C — fully in-process
│   ├── target_chatbot.rs      # Target B — shells out to ga-chatbot binary
│   ├── target_optick.rs       # Target A — shells out to GA CLI (mocked v1, live v2)
│   └── bin/
│       └── ix-autoresearch.rs # clap multi-verb CLI (run / list / promote / revert)
└── tests/
    ├── kernel.rs              # smoke + log roundtrip + milestone-promote (combined)
    ├── targets.rs             # grammar_e2e + optick_mocked (combined)
    └── chatbot_e2e.rs         # Target B (#[cfg(feature = "chatbot-e2e")])
```

Plus coordinated edits in:

- `Cargo.toml` (workspace) — add `crates/ix-autoresearch` member + workspace dep.
- `.gitignore` — add `state/autoresearch/runs/`.
- `state/autoresearch/{runs,milestones}/.gitkeep` — committed empty dirs (only milestones/ is tracked beyond gitkeep).
- `crates/ix-agent/src/handlers.rs` — `pub fn autoresearch_run(params: Value) -> Result<Value, String>`.
- `crates/ix-agent/src/tools.rs` — register `ix_autoresearch_run` tool.
- `crates/ix-agent/tests/parity.rs` — append `"ix_autoresearch_run"` to `EXPECTED`, bump count 67 → 68.
- `crates/ix-agent/Cargo.toml` — add `ix-autoresearch = { workspace = true }` dep.
- `governance/demerzel/schemas/capability-registry.json` — register the new tool by domain.
- `docs/MANUAL.md`, `README.md` — bump tool count, add autoresearch section.

### Trait + Struct Definitions

The trait shape is locked in the brainstorm with **four refinements** identified during planning research (the original two plus two from the deepen-plan pass):

1. `evaluate()` accepts a `soft_deadline: Instant` so eval impls can check elapsed time and self-terminate early.
2. `score_to_reward(&self, &Score) -> f64` projects a multi-objective `Score` onto a scalar for the bandit/SA. `PartialOrd` on `Score` stays for log/display.
3. **(deepen)** `cache_salt(&self) -> Option<String>` makes caching determinism-aware. Default `Some("v1")` for deterministic targets; `None` disables caching entirely; impls returning a value tied to wall-clock or external state opt out cleanly.
4. **(deepen)** Best-so-far tracking moves out of `Score: PartialOrd` and onto the kernel-side `(reward: f64, iteration: usize)` tuple — `PartialOrd` is *partial* (multi-f64 Scores aren't totally ordered), and `score_to_reward` already gives the totally-ordered scalar we need for `max_by`.

```rust
// crates/ix-autoresearch/src/lib.rs

pub trait Experiment {
    type Config: Clone + Debug + Serialize + DeserializeOwned;
    type Score:  Clone + Debug + Serialize + DeserializeOwned + PartialOrd;

    /// Initial config — the baseline the loop perturbs from.
    fn baseline(&self) -> Self::Config;

    /// Generate a candidate by perturbing `current`. Stateless w.r.t. history;
    /// the loop owns acceptance state. Implementations should respect bounds
    /// (e.g. simplex constraints for OPTIC-K weights).
    fn perturb(&mut self, current: &Self::Config, rng: &mut StdRng) -> Self::Config;

    /// Run the experiment with `config`. `soft_deadline` is a hint the eval
    /// MAY honor to self-terminate (returning `AutoresearchError::TimedOut`).
    /// The kernel separately watchdog-kills if a `hard_timeout` is set.
    fn evaluate(
        &mut self,
        config: &Self::Config,
        soft_deadline: Instant,
    ) -> Result<Self::Score, AutoresearchError>;

    /// Project the score onto a scalar reward for Strategy decisions.
    /// Higher is better. Each target encodes its own scalarization
    /// (e.g. lex-order pass_rate then -FP).
    fn score_to_reward(&self, score: &Self::Score) -> f64;

    /// Salt for the `ix-cache` lookup key. `Some(s)` opts in to caching
    /// keyed by `blake3(serde_json(config) ++ s)`; `None` disables caching
    /// (e.g. for non-deterministic evals). Default `Some("v1")` covers
    /// deterministic targets.
    fn cache_salt(&self) -> Option<String> { Some("v1".to_string()) }
}

pub struct Outcome<E: Experiment> {
    pub run_id: RunId,
    pub best_config: E::Config,
    pub best_score: E::Score,
    pub iterations: usize,
    pub accepted: usize,
    pub log_path: PathBuf,
}

pub fn run_experiment<E: Experiment>(
    experiment: &mut E,
    strategy: Strategy,
    iterations: usize,
    budget: TimeBudget,
    log_dir: &Path,
    seed: u64,
) -> Result<Outcome<E>, AutoresearchError>;
```

```rust
// crates/ix-autoresearch/src/policy.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Strategy {
    /// Accept iff reward strictly improves. Stateless. Karpathy's autoresearch
    /// uses this exclusively (`val_bpb` lower-is-better with git-revert).
    Greedy,

    /// Metropolis acceptance with geometric cooling. Reference:
    /// Andresen & Nourani — geometric outperforms logarithmic for small budgets.
    /// `initial_temperature: None` triggers the calibration helper described
    /// in Phase 1 (10 random samples → T₀ s.t. uphill accept ≈ 80%).
    SimulatedAnnealing {
        initial_temperature: Option<f64>,    // None ⇒ calibrate
        cooling_rate: f64,                   // T_{n+1} = cooling_rate * T_n; 0.95 default
    },

    /// Random search baseline (no acceptance criterion — every iter logged,
    /// best-so-far tracked separately). Useful as a control to detect
    /// reward-hacking (Cerebras autoresearch lessons).
    RandomSearch,

    /// **(deepen-plan reservation)** Plug-in acceptance policy for v1.5+
    /// (UCB-over-directions, Bayesian opt, etc.). Reserved as an enum variant
    /// now to prevent a one-way door at the MCP `input_schema` layer; not
    /// wired into the kernel in v1. When implemented, the box is a trait
    /// object: `pub trait AcceptancePolicy: Send { fn decide(...) -> Decision; }`.
    #[serde(skip)]
    Custom(Box<dyn AcceptancePolicy>),
}

pub trait AcceptancePolicy: Send {
    fn decide(
        &mut self,
        prev_reward: f64,
        candidate_reward: f64,
        iteration: usize,
        rng: &mut StdRng,
    ) -> Decision;
}

pub enum Decision { Accept, Reject }
```

#### Research Insights — Strategy defaults

- **Geometric cooling at 0.95** for ~100-iter overnight runs; T₁₀₀ ≈ T₀ · 0.006 — effectively greedy by the end, the right shape ([Andresen & Nourani SA cooling-strategy comparison](https://www.fys.ku.dk/~andresen/BAhome/ownpapers/perm-annealSched.pdf)).
- **T₀ calibration heuristic** (Ben-Ameur 2004): run 10 random perturbations, measure mean uphill ΔE, set `T₀ = -ΔE / ln(0.8)` so 80% of bad moves accept initially. Implemented as `calibrate_initial_temperature(experiment, n_samples=10, target_accept=0.8)` helper called by `Strategy::SimulatedAnnealing` when `initial_temperature` is `None`.
- **Lundy-Mees** `T_{n+1} = T_n / (1 + β·T_n)` is a known alternative; empirically slightly worse than geometric for small budgets per the same comparison; not implemented in v1.

```rust
// crates/ix-autoresearch/src/time_budget.rs

#[derive(Debug, Clone, Copy)]
pub struct TimeBudget {
    pub soft_deadline_per_iter: Duration,    // hint passed into evaluate()
    pub hard_timeout_per_iter:  Option<Duration>, // kernel kills worker if Some
}
```

```rust
// crates/ix-autoresearch/src/log.rs

/// Tagged JSONL event — every line in `log.jsonl` is one of these. The
/// `event` discriminant lets replay distinguish run-level metadata from
/// per-iteration entries, and "incomplete run" (no `RunComplete`) from
/// "graceful exit". Schema is versioned; v1 → v1.x changes are additive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum LogEvent<C, S> {
    /// First line of every run.
    #[serde(rename = "run_start")]
    RunStart {
        schema_version:    u32,                       // v1 today
        run_id:            String,
        timestamp:         chrono::DateTime<chrono::Utc>,
        target:            String,                    // "grammar" | "chatbot" | "optick"
        strategy:          serde_json::Value,         // serialized Strategy
        seed:              u64,
        git_sha:           Option<String>,            // best-effort `git rev-parse HEAD`
        baseline_config:   C,
        eval_inputs_hash:  Option<String>,            // hash of held-out corpus / GA index
    },
    /// One per evaluated candidate.
    #[serde(rename = "iteration")]
    Iteration {
        iteration:     usize,
        timestamp:     chrono::DateTime<chrono::Utc>,
        config:        C,
        config_hash:   String,                        // blake3(serde_json(config) ++ cache_salt)
        score:         Option<S>,                     // None on eval error
        reward:        Option<f64>,
        accepted:      bool,
        previous_hash: Option<String>,                // for replay revert
        error:         Option<String>,
        elapsed_ms:    u64,
    },
    /// Last line on graceful exit. Absence of this line means the run was
    /// interrupted (crash, Ctrl-C, hard-timeout abort).
    #[serde(rename = "run_complete")]
    RunComplete {
        timestamp:        chrono::DateTime<chrono::Utc>,
        iterations:       usize,
        accepted:         usize,
        best_iteration:   Option<usize>,
        best_reward:      Option<f64>,
        consecutive_kills_at_abort: Option<usize>,    // Some if aborted by hard-kill threshold
    },
}

pub struct JsonlLog<C, S> {
    /* opens file in append mode; `append(event)` serializes to Vec<u8>,
     * single `write_all([buf, b"\n"].concat())`, `sync_all` on every Nth
     * call (default N=10) plus on every `accepted: true` iteration. */
}
```

#### Research Insights — JSONL schema

- **`event` discriminant + sentinel events** (W&B / MLflow / Aim convention): replay can distinguish "interrupted" from "complete," and run-level metadata is on the first line, not implicit in the per-iter rows. Aligns with the `tars-graph-persistence` append-before-update pattern.
- **`schema_version` field** (one-way door): adding fields to `Iteration` later still parses old logs (serde `#[serde(default)]` on every additive field); changing fields requires a `schema_version` bump and a versioned-replay arm. Cheap to add now, prohibitively expensive to retrofit.
- **`git_sha`, `seed`, `eval_inputs_hash`** (Cerebras anti-cheating): replay reproducibility requires git state + RNG seed; `eval_inputs_hash` (e.g. blake3 of held-out corpus or GA index) catches reward-hacking via "the eval's input set silently shrank between iters."
- **Atomic writes**: serialize to `Vec<u8>` *before* the syscall, single `write_all`, then `sync_all`. Rust's `writeln!` over `BufWriter` can split a 1–2 KB JSON line across two `WriteFile` syscalls on Windows — not atomic. Replay uses `serde_json::from_str` per line; trailing parse failure ⇒ discard last line.
- **fsync policy**: every Nth entry OR on `accepted: true`. Per-entry fsync is ~5 ms × 100 iters ≈ 500 ms — negligible vs eval cost but wasted; durability-on-checkpoint is the right cost/benefit point.

```rust
// crates/ix-autoresearch/src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum AutoresearchError {
    #[error("evaluation timed out after {0:?}")]
    TimedOut(Duration),
    #[error("evaluation failed: {0}")]
    EvalFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("config hash collision (different config produced same hash)")]
    HashCollision,
    #[error("subprocess killed by hard timeout: {0}")]
    HardKilled(String),
}
```

### Target Schemas (locked from brainstorm)

#### Target C — Grammar evolution (smoke test)

```rust
// crates/ix-autoresearch/src/targets/grammar.rs

pub struct GrammarTarget {
    rules:        Vec<ix_grammar::weighted::WeightedRule>,
    held_out:     Vec<String>,            // corpus to parse
    grammar_text: String,                 // EBNF source
}

pub struct GrammarConfig {
    pub rule_weights: Vec<f64>,           // one per rule, must be ≥ 0
    pub temperature:  f64,                // > 0, log-uniform perturbed
}

pub struct GrammarScore {
    pub parse_success_rate: f64,          // higher = better, primary
    pub ess_stability:      f64,          // 1 - distance from ESS, tie-break
}

impl Experiment for GrammarTarget {
    type Config = GrammarConfig;
    type Score  = GrammarScore;
    fn baseline(&self) -> GrammarConfig { /* current weights, T=1.0 */ }
    fn perturb(&mut self, c: &GrammarConfig, rng: &mut StdRng) -> GrammarConfig {
        // Gaussian σ=0.1 on each weight, reflective at 0
        // Log-uniform on temperature within [0.1, 10.0]
    }
    fn evaluate(&mut self, c: &GrammarConfig, _: Instant) -> Result<GrammarScore, _> {
        // 1. Update self.rules with c.rule_weights
        // 2. Compose softmax(rules, c.temperature) into a sampling distribution
        // 3. Parse each held_out string; compute parse_success_rate
        // 4. Run replicator::simulate; compute distance from detect_ess
    }
    fn score_to_reward(&self, s: &GrammarScore) -> f64 {
        s.parse_success_rate + 0.1 * s.ess_stability  // weighted sum
    }
}
```

Held-out corpus: 50–100 strings derived from the existing grammar
catalog at `crates/ix-grammar/src/catalog.rs`. Sub-second per iter.

#### Target B — Chatbot QA threshold tuning

```rust
// crates/ix-autoresearch/src/targets/chatbot.rs

pub struct ChatbotTarget {
    chatbot_bin:     PathBuf,           // resolved to target/release/ga-chatbot
    corpus_dir:      PathBuf,           // tests/adversarial/corpus
    fixtures_path:   PathBuf,           // tests/adversarial/fixtures/stub-responses.jsonl
    workspace_root:  PathBuf,
}

pub struct ChatbotConfig {
    pub deterministic_pass_threshold: f64,    // qa.rs hardcoded 0.9 today
    pub judge_accept_threshold:       f64,    // judge panel agreement ≥ this
    pub fixture_confidence_floor:     f64,    // ignore fixtures below this
    pub strict_grounding:             bool,   // require sources when conf > 0.5
}

pub struct ChatbotScore {
    pub pass_rate:           f64,         // primary — match expected_verdict
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
}

impl Experiment for ChatbotTarget {
    fn evaluate(&mut self, c: &ChatbotConfig, deadline: Instant) -> Result<...> {
        // 1. Write c to a temp config JSON
        // 2. Spawn ga-chatbot qa --corpus <corpus_dir> --fixtures <fixtures_path>
        //                        --output <run-tmp>/findings.jsonl
        //                        --autoresearch-config <temp>
        // 3. Wait with timeout; kill on hard_timeout
        // 4. Parse <run-tmp>/summary.json (NOT the exit code — research found it
        //    is regression-wired, not raw-failure-wired)
        // 5. Compute pass_rate / FP / FN from summary fields
    }
    fn score_to_reward(&self, s: &ChatbotScore) -> f64 {
        s.pass_rate - 0.5 * s.false_positive_rate - 0.2 * s.false_negative_rate
    }
}
```

**Important**: this requires `ga-chatbot qa` to accept a new
`--autoresearch-config <path>` flag that overrides the hardcoded
constants in `qa.rs:102` (`DEFAULT_COST_THRESHOLD`) and `qa.rs:254-262`
(confidence cuts). That's a small in-repo change, not a cross-repo
dependency, but it is in-scope for this plan (Phase 4).

#### Target A — OPTIC-K self-tuning

```rust
// crates/ix-autoresearch/src/targets/optick.rs

pub struct OpticKTarget {
    ga_cli_path:       PathBuf,                   // FretboardVoicingsCLI
    invariants_bin:    PathBuf,                   // ix-optick-invariants
    diagnostics_bin:   PathBuf,                   // ix-embedding-diagnostics
    target_index:      OpticKMode,                // CIReduced | LiveCorpus(path)
    rebuild_fn:        Box<dyn FnMut(&OpticKConfig) -> Result<PathBuf, String>>,
}

#[derive(Clone)]
pub enum OpticKMode {
    /// Use the CI-reduced index for fast iteration. Default.
    CIReduced,
    /// Live GA corpus — only when GaApi is stopped (mmap lock).
    LiveCorpus(PathBuf),
}

pub struct OpticKConfig {
    pub structure_weight: f64,    // [0, 1]
    pub morphology_weight: f64,
    pub context_weight:    f64,
    pub symbolic_weight:   f64,
    pub modal_weight:      f64,
    pub root_weight:       f64,
    // Invariant: sum ≈ 1.0 (Dirichlet-on-simplex perturbation)
}

pub struct OpticKScore {
    pub structure_leak_pct:  f64,    // primary, lower = better
    pub retrieval_match_pct: f64,
    pub inv_25_pass_rate:    f64,
    pub inv_32_pass_rate:    f64,
    pub inv_36_pass_rate:    f64,
}

impl Experiment for OpticKTarget {
    fn evaluate(&mut self, c: &OpticKConfig, deadline: Instant) -> Result<...> {
        // 1. (rebuild_fn)(c) → writes config JSON, shells out to GA CLI,
        //    returns path to rebuilt index. In v1 testing this is a mock that
        //    just simulates output JSON; in v2 it runs the real rebuild.
        // 2. Spawn ix-optick-invariants --index <path> --out <tmp>/firings.json
        // 3. Spawn ix-embedding-diagnostics --index <path> --out <tmp>/diag.json
        // 4. Parse both, project into OpticKScore.
    }
    fn score_to_reward(&self, s: &OpticKScore) -> f64 {
        // **(deepen-plan refinement)** Lex-order, not weighted-sum.
        // Weighted sum can only reach convex parts of the Pareto front;
        // for OPTIC-K (where `1−leak%` and `retrieval` plausibly trade off
        // non-convexly across partition-weight regimes) lex-order makes
        // the dominance order explicit and avoids silently excluding valid
        // frontier points.
        //
        // Encoding via `f64::total_cmp` on a tuple flattened to a scalar:
        // (1−leak%) is in [0,1] and dominates by a 1.0e6 factor;
        // retrieval is next, 1.0e3 factor; mean(invariants) breaks ties.
        let leak  = (1.0 - s.structure_leak_pct).clamp(0.0, 1.0);
        let retr  = s.retrieval_match_pct.clamp(0.0, 1.0);
        let inv   = (s.inv_25_pass_rate + s.inv_32_pass_rate + s.inv_36_pass_rate) / 3.0;
        leak * 1.0e6 + retr * 1.0e3 + inv
        // Alternative: Chebyshev `min max_i w_i · |f_i − z*_i|` with utopia
        // z* = (0.0 leak, 1.0 retr, 1.0 inv) — recovers full Pareto front
        // including non-convex regions ([Tripp 2025], [Springer COAP 2023]).
        // Defer to v1.5 unless lex-order misses an obvious improvement.
    }
}
```

`rebuild_fn` is injected so tests use a mock and production uses the
real GA CLI. The real CLI requires the **GA-side companion PR** that
adds `--weights-config <path>` to `FretboardVoicingsCLI` (see
Dependencies below).

#### Research Insights — Target A scalarization + Dirichlet perturbation

- **Why lex-order over weighted-sum**: weighted sum cannot reach concave parts of the Pareto front no matter how the weights are set ([WhiteRose multi-objective survey](https://eprints.whiterose.ac.uk/86090/8/WRRO_86090.pdf)). For `(1−leak, retrieval)` the surface is plausibly non-convex; lex-order with a clear primary objective avoids the trap. Chebyshev is the principled v1.5 upgrade if lex-order plateaus.
- **Dirichlet `α = 200` default**: `Dir(α · w_current)` has `Var[X_i] = w_i(1-w_i)/(α+1)`. For 6 weights at uniform mid-point (w ≈ 0.17), α=200 gives ~σ=0.027 perturbations — "explore neighborhood" scale. α=1000 ⇒ σ=0.013 (fine-tuning). Source: [Stan Dirichlet docs](https://mc-stan.org/docs/2_21/functions-reference/dirichlet-distribution.html).
- **Zero-weight is an absorbing state**: `Dir(0, ...)` keeps that component at exactly zero forever. Mitigation: floor `w_i ← max(w_i, 1e-3)` *before* multiplying by α, then renormalize to sum 1 *before* sampling. Add this invariant to `optick_mocked.rs` test #7 (every accepted iter has `w_i ≥ 1e-3`).

### Implementation Phases

#### Phase 1: Kernel foundation

- New crate `ix-autoresearch`: `Cargo.toml` (workspace inheritance per conventions above), `README.md`, `lib.rs` skeleton.
- Implement `Experiment` trait + `Outcome`, `RunId` (UUIDv7 via `uuid::Uuid::now_v7().hyphenated().to_string()` — dir-safe, lexicographically time-sorted).
- Implement `policy.rs`:
  - `Strategy` enum (`Greedy`, `SimulatedAnnealing { initial_temperature: Option<f64>, cooling_rate: f64 }`, `RandomSearch`, `Custom(Box<dyn AcceptancePolicy>)`).
  - `AcceptancePolicy` trait (single `decide(prev_reward, candidate_reward, iteration, rng) -> Decision`).
  - `calibrate_initial_temperature(experiment, n_samples=10, target_accept=0.8) -> f64` helper called when SA's `initial_temperature` is `None`. Implements Ben-Ameur 2004: 10 random perturbations → mean uphill ΔE → `T₀ = -ΔE / ln(0.8)`.
- Implement `time_budget.rs`: `TimeBudget { soft_deadline_per_iter: Duration, hard_timeout_per_iter: Option<Duration> }`. Hard-timeout watchdog uses `wait-timeout = "0.1"`'s `Child::wait_timeout(Duration)` for shell-out targets; for in-process targets soft deadline is the only signal.
- Implement `log.rs`: `LogEvent` enum (`RunStart` / `Iteration` / `RunComplete`) with `schema_version: u32`. Each `append(event)`: serialize to `Vec<u8>`, single `write_all([buf, b"\n"].concat())`, `sync_all` per checkpoint policy (every Nth call OR on `accepted: true`).
- Implement `cache.rs`: keyed `blake3(serde_json(config) ++ cache_salt)` against `ix_cache::Cache`. `Experiment::cache_salt() -> Option<String>`; `None` disables. 24h TTL secondary safety net. **Important**: `Config` structs must serialize deterministically — document this as a `// IMPORTANT:` invariant on each `XxxConfig` struct (no `HashMap`; `BTreeMap` if ordered map needed).
- Implement `error.rs` (thiserror): `EvalFailed(String)`, `TimedOut(Duration)`, `HardKilled(String)`, `Io`, `Serde`, `HashCollision`.
- Implement `run_experiment` core: `for` loop, perturb, evaluate-with-budget, score-to-reward, strategy-decide, append to log, update best-so-far on `(reward: f64, iteration: usize)` tuple (not `Score: PartialOrd`).
- **Hard-kill cascade abort**: `consecutive_hard_kills_threshold: usize = 3`. When threshold exceeded, append `RunComplete { consecutive_kills_at_abort: Some(n), .. }` and return. Log the diagnostic context.

**Acceptance**: `tests/kernel.rs` integration tests using a deterministic mock `Experiment` (`type Config = f64; type Score = f64; perturb = +N(0, 0.1); evaluate = |c| -c*c; score_to_reward = identity; cache_salt = Some("v1")`) confirm:

- (a) Greedy converges toward 0 (after 200 iters, |best.config| < 0.05).
- (b) SA accepts uphill moves at high T (accept ratio > 30% in first 50 iters with T₀ from calibration helper).
- (c) RandomSearch logs all iters (every iter has an `Iteration` event regardless of reward).
- (d) JSONL log replays exactly (write 100 entries, close, reopen, parse each line, byte-identical roundtrip; replay reaches the same best-so-far as the original run).
- (e) `hard_timeout` kills a long-sleeping mock within 2× the budget.
- (f) **Truncated-tail tolerance**: write 50 entries, truncate the file mid-line at byte position N, replay parses 49 entries cleanly and discards the 50th.
- (g) **Hard-kill cascade abort**: mock `evaluate` that always blocks past `hard_timeout`; after 3 consecutive kills, run aborts with `RunComplete.consecutive_kills_at_abort = Some(3)`.
- (h) **Cache salt**: mock with `cache_salt = None` exercises `evaluate` on every iter (cache disabled); mock with `cache_salt = Some("v1")` hits the cache on duplicate configs.

#### Phase 2: Target C end-to-end

- Implement `targets/grammar.rs` per schema above.
- Use `ix_grammar::weighted::{WeightedRule, softmax, bayesian_update}` for weight handling, `ix_grammar::replicator::{simulate, detect_ess}` for stability scoring.
- Held-out corpus: hand-pick 50 strings from existing catalog tests, store as `crates/ix-autoresearch/tests/data/grammar-heldout.txt`.
- Integration test `grammar_e2e.rs`: 100 iterations of SA, asserts `final_reward > baseline_reward + 0.05` (some improvement) on a deterministically-seeded run.

**Acceptance**: Target C runs sub-second per iter; 100 iters complete in < 30 s; reward improves measurably.

#### Phase 3: CLI binary + MCP wrapper

- `src/bin/ix-autoresearch.rs` with clap multi-verb (`ix-skill` template, `crates/ix-skill/src/main.rs:11-100`):
  - `run --target <C|B|A> --iterations <N> --strategy <Greedy|SA> --seed <S> [--soft <secs>] [--hard <secs>]`
  - `list` — show `state/autoresearch/runs/` with summary stats per run
  - `promote <run-id> --note <slug>` — copy `runs/<id>/` to `milestones/<date>-<target>-<slug>/`
  - `revert <run-id> --to <iteration>` — print the config from a specific iteration; user pipes into target adapter manually (no autonomous revert v1)
- MCP handler: `crates/ix-agent/src/handlers.rs::autoresearch_run`. Accepts `{ target, iterations, strategy, seed, soft_seconds, hard_seconds }`. Returns `Outcome` JSON.
- MCP registration: append to `tools.rs` after `ix_grothendieck_path`. Update `parity.rs::EXPECTED` (67 → 68).
- Wire `ix-autoresearch` as a `dependency` of `ix-agent` (workspace dep).

**Acceptance**: `cargo run -p ix-autoresearch -- run --target grammar --iterations 50 --strategy sa` produces a JSONL log under `state/autoresearch/runs/<id>/log.jsonl`. MCP path returns matching JSON. Parity test passes at count = 68.

#### Phase 4: Target B end-to-end + ga-chatbot threshold flag

**(deepen-plan refinement)**: v1 ships **deterministic-stub-only mode** (~30s/iter as originally budgeted); judge-panel-in-loop is a v1.5 flag because it bumps per-iter to 60–120s and the threshold-only search space barely moves under v1's perturbation.

- Add `--autoresearch-config <path>` flag to `ga-chatbot qa` that overrides hardcoded thresholds in `qa.rs:102` (`DEFAULT_COST_THRESHOLD`) and `qa.rs:254-262` (confidence cuts `>0.9 T / >0.7 P / >0.5 U / else F`). Reads JSON `{ deterministic_pass_threshold, judge_accept_threshold, fixture_confidence_floor, strict_grounding }`.
- **Read `summary.json`, not exit code** — `ga-chatbot qa`'s exit code is wired to *regression mismatch* (mismatch between actual deterministic verdict and corpus's `expected_verdict`), not raw F/D failure. Target B parses `<run-tmp>/summary.json` for the true pass/fail signal.
- Implement `target_chatbot.rs` per schema above. Uses `Command::new(chatbot_bin)` with `--autoresearch-config <temp.json>`. Per-iter writes the config to a fresh temp file.
- Integration test `tests/chatbot_e2e.rs` (gated behind feature `chatbot-e2e`): 50 iters, SA, asserts log shape correctness more than reward improvement.

**Acceptance**: Target B runs ~30 s/iter against the existing adversarial corpus in deterministic-only mode. Log shows accept/reject distribution that's plausible (not 0% or 100%). Score parsing uses `summary.json`, not `child.status.code()`.

#### Phase 5: Target A adapter with mocked GA rebuild

- Implement `targets/optick.rs` per schema above. `rebuild_fn` defaults to a mock that:
  - Reads the config JSON
  - Synthesizes a fake `optick.index` by perturbing a baseline (or just emits a synthetic diagnostics JSON without rebuilding)
  - Returns a path to the synthetic output
- Integration test `optick_mocked.rs`: 30 iters with mock, validates schema parsing and Dirichlet-simplex perturbation correctness.

**Acceptance**: Target A's adapter compiles, the mock produces deterministic synthetic scores, the kernel exercises the OpticKConfig perturbation correctly. **This is the milestone where Target A is "code-complete on the IX side"** — live runs are gated on Phase 6.

**(deepen-plan refinement)** Per-iter budget split by `OpticKMode`:
- `OpticKMode::CIReduced` (default): ~40–60 s/iter (CI-reduced index ≤10% of full corpus, plus invariants ~10s + diagnostics ~10s).
- `OpticKMode::LiveCorpus`: ~160–180 s/iter (full 313k-voicing rebuild ~140s + invariants + diagnostics).

The plan's earlier single 140s figure conflated the two; reflect this in Phase 7's overnight-budget arithmetic (50 iters × 50s = ~40 min on CI-reduced, ~2.5 h on live).

#### Phase 6: GA-side `--weights-config` companion PR

- **Out-of-tree work** in the GA repo. Adds a `--weights-config <path>` flag to `FretboardVoicingsCLI` that reads the JSON schema from Phase 5's `targets/optick.rs` and overrides `EmbeddingSchema.PartitionWeights` at index-rebuild time.
- Companion PR commit message references this plan + the IX-side commit hash.
- **No IX-side code change required** — the existing `OpticKTarget::rebuild_fn` is replaced via dependency injection at construction time.
- Risk gate: confirm GA's CI catches any regression on existing OPTIC-K consumers (chatbot retrieval primarily).

#### Phase 7: Target A live end-to-end

- Construct `OpticKTarget` with the real `rebuild_fn` (shells out to `FretboardVoicingsCLI --weights-config <path>`).
- Run a 50-iteration overnight session against the CI-reduced index. Verify the visible metric story: `structure_leak_pct` moves measurably.
- If it moves favorably, run `ix-autoresearch promote <run-id> --note "first-overnight-tune"` to commit the milestone.
- Add an `optick_live.rs` integration test that runs 5 iters against the CI-reduced index (longer-running, gated behind `optick-live` feature flag, run only in nightly CI).

**Acceptance**: An overnight run produces a milestone snapshot with a real metric improvement. The autoresearch story is shippable.

## Alternative Approaches Considered

### LLM-driven perturbation (β) instead of numeric
**Rejected for v1**: requires sandboxing, API integration, trust boundaries, and erodes reproducibility. The autoresearch *story* lands either way. Numeric runs can be re-played from a seed; LLM runs cannot. The trait shape leaves a clean seam (`perturb()` swap) for v2 (see brainstorm: docs/brainstorms/2026-04-26-ix-autoresearch-brainstorm.md, "Why This Approach").

### `ix-pipeline::dag::Dag<N>` instead of plain `for` loop
**Rejected**: `ix-pipeline` is acyclic; iteration is not its primitive. A plain `for` loop is simpler and the kernel doesn't need DAG-shaped speculation. (Repo research: `crates/ix-pipeline/src/dag.rs:11-21`.)

### Parallel worker pool in v1
**Rejected**: Target A cannot parallelize (GA mmap lock); B's modest speedup isn't worth the v1 complexity; C is the smoke test. Sequential covers all three flagship targets with no compromise. `Strategy::Parallel(n)` reserved as a v2 enum variant (see brainstorm).

### Use UCB1 from `ix-rl::bandit` as the default policy
**Rejected for v1**: continuous configs don't fit discrete arms naturally. UCB-over-perturbation-directions is interesting but requires a perturb-with-bias signature change. Greedy + SA cover the v1 search space; UCB plug-in is a v1.5 follow-up.

### Per-target separate crates instead of one crate with target modules
**Rejected**: targets share the `Experiment` trait, the log, and the strategy. Separate crates would force the trait into a tiny crate of its own and duplicate the test fixtures. Single crate with `targets/` module tree is simpler; if a target later needs an external optional dep, we can split it then.

## System-Wide Impact

### Interaction Graph

`run_experiment(experiment, …)` triggers, in order per iteration:
1. `experiment.perturb(current_config, rng)` — pure compute.
2. `cache.get(config_hash)` — `ix_cache::Cache::get` against the in-process LRU.
3. On cache miss: `experiment.evaluate(config, soft_deadline)`. For Target A this spawns a child process (`FretboardVoicingsCLI`) that opens `optick.index` mmap (lock contention with GaApi if running). For Target B, spawns `ga-chatbot qa`. For Target C, fully in-process.
4. `cache.set_with_ttl(config_hash, score, 24h)`.
5. `score_to_reward(score)` — pure projection.
6. `strategy.decide(prev_reward, candidate_reward)` — pure.
7. `log.append(entry)` — single-line `writeln!` + `fsync` (per `2026-04-19-tars-graph-persistence.md` append-before-update pattern).
8. Best-so-far update if accepted.

### Error & Failure Propagation

Errors flow up via `Result<_, AutoresearchError>`:
- `EvalFailed(String)` — eval returned non-zero or unparseable output. Kernel logs with `error: Some(...)`, treats as reward = `f64::NEG_INFINITY` (always rejected), continues.
- `TimedOut(Duration)` — eval honored soft deadline and returned early. Treated identically to `EvalFailed`.
- `HardKilled(String)` — kernel watchdog killed the worker. Same treatment, but logs the killed PID for forensics.
- `Io` / `Serde` — kernel-level fault, aborts the run with the error logged to the JSONL.
- `HashCollision` — defensive; should never fire with blake3. Aborts the run.

The kernel does **not** auto-retry on eval failures (v1). Stagnation policy: if 50 consecutive iterations all fail or all reject, the kernel logs a warning entry and continues; the user inspects the log next morning.

### State Lifecycle Risks

- **Partial JSONL writes** (deepen-plan: HIGH concern → mitigated): `writeln!` is *not* atomic on Windows; a 1–2 KB JSON line through `BufWriter` can split across two `WriteFile` calls. Mitigation: serialize each entry to `Vec<u8>` first, single `write_all([buf, b"\n"].concat())`, then `sync_all` per checkpoint policy. Replay parses each line under `serde_json::from_str`; trailing parse failure ⇒ discard last line. `log_roundtrip.rs` test #4 covers a deliberately-truncated-tail file. A crash mid-eval (before write) leaves the log consistent (iteration simply absent). Append-before-update guarantees replay reaches identical best-so-far state.
- **Schema evolution** (deepen-plan: HIGH concern → mitigated): every `LogEvent` carries `schema_version: u32`. Replay logic must dispatch on this; v1 → v1.x changes are additive (new fields use `#[serde(default)]`). Field renames or shape changes require a `schema_version` bump and a versioned arm in the replay matcher. *This is a one-way door — flagged in frontmatter.*
- **Cache poisoning** (deepen-plan: HIGH concern → mitigated): TTL alone is the wrong primitive — a 23-h-old wall-clock-dependent score is just as wrong as a 25-h-old one. Mitigation: cache key is `blake3(serde_json(config) ++ cache_salt)`; non-deterministic targets return `cache_salt() = None` to disable caching entirely. TTL stays as a secondary safety net (24 h default). `--no-cache` flag for ad-hoc debugging.
- **Milestone promotion atomicity** (deepen-plan: HIGH concern → mitigated): `promote_run` copies to `milestones/<slug>.tmp/`, atomic-renames to `milestones/<slug>/` (Windows: `MoveFileEx` with `MOVEFILE_REPLACE_EXISTING`, exposed via `std::fs::rename`), then writes a `.complete` sentinel file *last*. `list` skips dirs missing `.complete`. The `milestone_promote.rs` integration test (collapsed into `tests/kernel.rs` per pattern alignment) covers the crashed-mid-copy case.
- **Subprocess orphans on Windows**: `Child::kill()` uses `TerminateProcess`, which **does not kill grandchildren** — exactly the leak the brainstorm flagged. Mitigation: deferred to v1.5 `JobObject` containment (`windows-sys = { version = "0.59", features = ["Win32_System_JobObjects", "Win32_Foundation"] }`, ~80 LOC modeled on `cargo`'s `src/util/job.rs`). v1 logs leaked PIDs to the `Iteration.error` field for forensics.
- **Mmap lock contention**: Target A's live mode (`OpticKMode::LiveCorpus`) cannot run concurrently with GaApi. CLI prints a warning if `tasklist /FI "IMAGENAME eq GaApi.exe"` returns a process, and `--force-live` is required to override.
- **fsync semantics on Windows**: `File::sync_all` calls `FlushFileBuffers`, which is the documented durability primitive. **Caveat**: it does not flush the disk's *write cache* unless the volume has write-caching disabled or the file was opened with `FILE_FLAG_WRITE_THROUGH`. Document this in `log.rs` rustdoc; for autoresearch's failure model (process crash, not power loss), FFB is sufficient.
- **Disk-full (ENOSPC)** (deepen-plan: LOW): on `Io` error from `log.append`, kernel propagates the error and aborts the run. Cheap fallback: also `eprintln!` to stderr with `(run_id, iteration, error)` so the orphaned `runs/<id>/` directory is at least diagnosable from terminal scrollback.
- **State-dir bloat**: `runs/` is gitignored but unbounded on disk. `ix-autoresearch list` shows total disk usage; a `prune --older-than <days>` verb is deferred to v1.5.
- **Run-id uniqueness across sessions** (deepen-plan: SAFE): UUIDv7 = 48-bit ms timestamp + 74 random bits. Two processes in the same ms have ~2⁻³⁷ collision odds; effectively zero. CI + local runs in the same minute don't collide.

### API Surface Parity

The CLI binary (`ix-autoresearch run`), the MCP handler (`ix_autoresearch_run`), and the demo target (a script that calls one of them) all dispatch to `run_experiment`. Adding a new strategy or target requires:
1. Update `Strategy` enum in `policy.rs` (single source of truth).
2. Update CLI's `--strategy` parser (clap derive picks up the enum).
3. Update MCP tool's `input_schema` enum field in `tools.rs`.
4. Update `EXPECTED` in `parity.rs` if a new top-level tool is added (not a new strategy variant).

No third surface; no parallel implementations. Deviation from this is a code-review-blocker.

### Integration Test Scenarios

1. **Greedy strategy converges on quadratic mock** (`kernel_smoke.rs`). Scenario: `mock_eval(c) = -c²`; baseline `c = 1.0`. After 200 iters, `|best.config| < 0.05`.
2. **SA accepts uphill at high T** (`kernel_smoke.rs`). Scenario: same mock with SA, `T₀ = 10`. Log shows accept ratio > 30% in first 50 iters.
3. **Hard timeout kills runaway eval** (`kernel_smoke.rs`). Scenario: mock `evaluate` does `thread::sleep(Duration::from_secs(60))`; budget hard = 1s. Kernel returns within 2s with `HardKilled` logged.
4. **JSONL log replays exactly** (`log_roundtrip.rs`). Scenario: write 100 entries, close, reopen, read back, assert byte-identical roundtrip plus monotonic iteration field.
5. **Promote-to-milestone idempotency** (`milestone_promote.rs`). Scenario: `promote_run` twice with same `--note` is idempotent (no duplicate dir, second call no-ops).
6. **Target C improves grammar parse rate** (`grammar_e2e.rs`). Scenario: 100 SA iters from a hand-degraded baseline; assert `final_reward >= baseline_reward + 0.05`.
7. **Target A mock pipeline correctness** (`optick_mocked.rs`). Scenario: 30 iters with mock rebuild; assert (a) all configs satisfy the simplex constraint within 1e-6, (b) every accepted iter has higher reward than the previous best.

## Acceptance Criteria

### Functional Requirements

- [ ] `crates/ix-autoresearch/` exists with the layout above; `cargo build -p ix-autoresearch` passes.
- [ ] `cargo test -p ix-autoresearch` passes all 7 integration tests above (3 gated tests pass under their feature flags).
- [ ] `cargo run -p ix-autoresearch -- run --target grammar --iterations 50` produces a JSONL log at `state/autoresearch/runs/<run-id>/log.jsonl` with 50 entries.
- [ ] `cargo run -p ix-autoresearch -- list` reports the run with iteration count and best reward.
- [ ] `cargo run -p ix-autoresearch -- promote <run-id> --note smoke` creates `state/autoresearch/milestones/2026-04-26-grammar-smoke/` and stages it for git.
- [ ] `ix_autoresearch_run` MCP tool is callable from Claude Code; `EXPECTED.len() == 68` in `parity.rs`.
- [ ] Target C: 100 SA iters improve reward measurably from baseline.
- [ ] Target B: 50 SA iters complete in ~30 min wall-clock; log shape is correct.
- [ ] Target A (mocked): 30 iters, all configs satisfy simplex constraint within 1e-6.
- [ ] Target A (live): an overnight 50-iter run produces a milestone with `structure_leak_pct` improvement (gated on Phase 6).

### Non-Functional Requirements

- [ ] Reproducibility: a `(target, strategy, seed, baseline)` tuple deterministically reproduces the JSONL log byte-identical (subject to wall-clock fields).
- [ ] All public APIs documented with `///` rustdoc; the crate README has a 30-line quickstart (raises bar above existing ix crates which mostly omit READMEs in favor of `//!` module docs — that's also acceptable).
- [ ] Workspace `cargo clippy --workspace --all-targets -- -D warnings` passes after this lands.
- [ ] No new external deps beyond `uuid` (new), `blake3` (promote to workspace), `wait-timeout = "0.1"` (new), and the existing `ix-*` crates. `chrono`, `clap`, `serde`, `serde_json`, `thiserror`, `tokio` already workspace-deps.

### Quality Gates

- [ ] All 7 integration tests pass on `main` after Phase 5 (Target A mocked).
- [ ] Phase 6 GA companion PR merges before Phase 7 lands.
- [ ] Phase 7's overnight run produces a committed milestone before this plan's status is set to `complete`.
- [ ] Doc count refresh: README + MANUAL show 68 MCP tools.
- [ ] Capability registry (`governance/demerzel/schemas/capability-registry.json`) registers `ix_autoresearch_run` under domain `governance`.

## Success Metrics

- **Target C**: parse-success-rate improvement ≥ 5% over baseline on the held-out corpus, end-to-end demonstration that the kernel works.
- **Target B**: any positive-delta run (the search space is fuzzy; success here is "the framework runs against the chatbot and logs sensible decisions").
- **Target A**: at least one overnight milestone where `structure_leak_pct` decreases by ≥ 1 percentage point against the CI-reduced index — the visible-metric autoresearch story.
- **Adoption signal**: at least one external caller (Claude Code, GA, TARS) uses `ix_autoresearch_run` within 30 days of merge.

## Dependencies & Prerequisites

### Internal

- `ix-rl::bandit` (UCB / Thompson references; not v1-critical but documented for v1.5).
- `ix-grammar` (Target C eval).
- `ix-cache` (memoization).
- `ga-chatbot` (Target B eval; **requires in-repo `--autoresearch-config` flag added in Phase 4**).
- `ix-optick`, `ix-optick-invariants`, `ix-embedding-diagnostics` (Target A eval — already exist).

### External (cross-repo)

- **GA companion PR**: `--weights-config <path>` flag on `FretboardVoicingsCLI`. Schema: JSON with the six float weights from `OpticKConfig`. Reads → overrides `EmbeddingSchema.PartitionWeights` at rebuild. **Phase 6 of this plan, separate PR in the GA repo, blocks Phase 7 only.**

### New crate dependencies

- `uuid = { version = "1", features = ["v7", "serde"] }` (run id). v7 is **stable** in `uuid` 1.10+ (RFC 9562 ratified May 2024); no `unstable-` flag needed in 2026. Lexicographic time-ordered, dir-safe.
- `blake3 = "1.5"` (config hashing). **Promote to `[workspace.dependencies]`** — currently per-crate in `ix-autograd/Cargo.toml:15` and `ix-agent/Cargo.toml:66`. Sharing the pin avoids drift; cheap cleanup PR.
- `wait-timeout = "0.1"` (subprocess hard timeout). **Corrected** — the brainstorm and initial plan said `"0.2"`, but the latest is `0.1.5+` (alexcrichton, last release Feb 2025).
- (v1.5 only) `windows-sys = { version = "0.59", features = ["Win32_System_JobObjects", "Win32_Foundation"] }` for `JobObject` containment to fix grandchild leak.

All other deps (`chrono`, `clap`, `serde`, `serde_json`, `thiserror`) are already in the workspace and used via `{ workspace = true }`.

#### Workspace conventions for `Cargo.toml`

Match the pattern from `ix-quality-trend` (the cleanest single-purpose example), not the outlier `ix-optick-invariants` which hardcodes `version = "0.1.0"` / `edition = "2021"` / `license = "MIT"`. The right pattern:

```toml
[package]
name = "ix-autoresearch"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
categories = ["command-line-utilities", "algorithms"]
keywords = ["autoresearch", "experiment", "loop", "tuning", "search"]
description = "Karpathy-style edit-eval-iterate kernel + target adapters for IX subsystems"
```

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GA companion PR doesn't land | Medium | Blocks Phase 7 only | Phases 1-6 ship independently; Target A "code-complete with mock" is a usable milestone. |
| Windows hard-timeout leaks grandchildren | Medium | Disk/RAM bloat on long runs | `Child::kill()` uses `TerminateProcess` which doesn't reach grandchildren. v1 logs leaked PIDs; v1.5 adds `JobObject` containment (~80 LOC modeled on cargo's job.rs). |
| Hard-kill cascades (hung subprocess pattern) | Medium | Wasted overnight run | After N=3 consecutive `HardKilled`, abort the run with diagnostic context — usually indicates systemic failure (mmap lock, OOM, GA bug), not bad config. Configurable threshold; same shape as the 50-rejection stagnation policy. |
| Mmap lock contention if user runs Target A live with GaApi up | Low | Eval fails with EBUSY | CLI checks `tasklist /FI "IMAGENAME eq GaApi.exe"` and warns before live runs. Default `OpticKMode::CIReduced` sidesteps the lock. `--force-live` required to override. |
| `cargo test` blocked by WDAC mid-session | Low (memory: `feedback_windows_app_control.md`) | All tests fail | Smoke `cargo test -p ix-autoresearch --lib` at start of every session before running long loops. |
| JSONL torn writes on Windows | Medium | Last line of crashed run unparseable | Single `write_all` of `[buf, b"\n"].concat()` (not `writeln!` over `BufWriter`); replay parses-or-discards trailing line. `log_roundtrip.rs` covers the truncated-tail case. |
| JSONL schema evolution (one-way door) | Medium | New v1.5 fields break v1 log replay | `schema_version: u32` on every event; `#[serde(default)]` on every additive field; replay matches on `schema_version`. Adding the field is cheap *now*, prohibitively expensive later. |
| Search converges to local optimum | High | Suboptimal final config | RandomSearch baseline runs as a sanity check; SA's exponential cooling provides exploration; T₀ calibrated to 80% initial uphill accept. v1.5: restart-on-stagnation. |
| Multi-objective scalarization is wrong (Target A specifically) | Medium | "Good" reward but bad real outcome | Target A uses **lex-order** (1−leak%, retrieval, mean_invariants), not weighted sum — weighted sum can only reach convex Pareto front regions. C and B keep weighted sum (one objective dominates). Document each target's encoding in module rustdoc. v1.5: Chebyshev option. |
| Cache poisoning (non-deterministic eval cached) | Low | Wrong score reused | `Experiment::cache_salt(&self) -> Option<String>`: deterministic targets return `Some("v1")`; non-deterministic return `None` (caching disabled). 24-h TTL is the secondary safety net. `--no-cache` for ad-hoc debugging. |
| Reward hacking via metric definition | Low (numeric search; high under v2 LLM) | Loop "wins" without solving the problem | `eval_inputs_hash` per iter logs the input set the eval used; if the held-out corpus or GA index changes between iters, the hash changes — replay can detect. (Cerebras autoresearch lessons.) |
| Promote-run interrupted mid-copy | Low | Half-built `milestones/<slug>/` | Copy to `<slug>.tmp/`, atomic rename to `<slug>/`, write `.complete` sentinel last. `list` skips dirs without `.complete`. Idempotent test covers crashed-mid-copy. |

## Resource Requirements

- **Engineering**: ~3-5 sessions of focused work for Phases 1-5 (kernel + Target C end-to-end + CLI/MCP + Target B + Target A mocked). Phase 6 is a separate GA-repo session. Phase 7 is one overnight run.
- **Infrastructure**: no new CI capacity; the gated tests run in nightly only.
- **Code volume**: estimated ~1500 LOC kernel + targets, ~600 LOC tests, ~200 LOC CLI/MCP wiring.

## Simplification Watch (deepen-plan)

The code-simplicity reviewer flagged the following as "v1 violations of YAGNI" that contradict the brainstorm's "build all" decision. Listed here so the user can demote any of them to v1.5 *with eyes open*; otherwise they ship as planned. None are recommended for silent demotion — each is in scope because the user explicitly chose "build all three targets" and "three surfaces from one kernel" in the brainstorm.

| Item | Reviewer's verdict | Reason | Counterargument |
|------|---|---|---|
| `Strategy::RandomSearch` | SIMPLIFY-TO-2 | SA at high T already explores; RandomSearch's only unique value is the sanity-baseline mention | Cerebras autoresearch lessons: a pure random-search baseline is the strongest detector of reward hacking; a 30-line variant is cheap insurance. **Keep**. |
| `milestones/` + `promote` verb | DEFER-TO-V1.5 | No v1 caller; first overnight run hasn't named one yet | The success metric ("at least one milestone snapshot per Phase 7") *is* the first caller. **Keep**. |
| `ix-cache` integration | DEFER-TO-V1.5 | Hit rate is ~0–2% under SA per performance review | True for SA; but Target A's 140s/iter cost makes even 5% hits worth ~7s/iter savings on long runs, and the wrapper is ~50 LOC. **Keep, behind `cache_salt = None` opt-out.** |
| `uuid v7` + `blake3` deps | SIMPLIFY (timestamp+rand + seahash) | Two new transitive deps for marginal value | Both are workspace-grade; `blake3` already used by `ix-autograd`/`ix-agent`; `uuid v7` is RFC-standard sortable. **Keep**. |
| Hard-timeout machinery | DEFER-TO-V1.5 | Soft deadline alone covers v1 evals; v1.5 can add `wait-timeout` | Target A's GA CLI is exactly the case the brainstorm identified as needing hard kills ("the C# CLI sometimes hangs"). **Keep** with the cascade-abort safety described in Phase 1. |
| MCP wrapper (`ix_autoresearch_run`) | DEFER-TO-V1.5 | No named v1 caller | Brainstorm's "one core, three surfaces" decision is explicit. **Keep**, but flag the `input_schema` as one-way door. |
| 6 → 3 test files | SIMPLIFY-TO-3 | Workspace pattern is fewer thematic files | **Accepted** — already folded into Crate Layout above. |
| `SoftDeadline` newtype | SIMPLIFY-TO-`Instant` | Wrapper buys nothing v1 doesn't have | **Accepted** — use `Instant` directly; newtype if hard-timeout returns. |

The accepted simplifications are already in the layout/trait sections above. The kept items have explicit counter-rationale tied to the brainstorm's locked decisions or to risk mitigations the simplicity review didn't weigh.

## Future Considerations

- **v1.5: UCB-over-directions Strategy** — discretize perturbation axes and use `ix-rl::bandit::UCB1` to bias `perturb()`. Requires `perturb(current, rng, bias_arm: usize)` signature update.
- **v1.5: Restart-on-stagnation** — after N consecutive rejections, jump to a random config and resume.
- **v1.5: `prune --older-than <days>`** CLI verb for `runs/` directory hygiene.
- **v2: LLM-driven perturbation** — alternative `perturb()` impl backed by an LLM call (Anthropic API or local Ollama). Captures the autoresearch *spirit*; requires sandbox + revert + cost accounting.
- **v2: Parallel worker pool** — `Strategy::Parallel(n)` for Target C and B (where eval is reentrant). A still serial.
- **v2: Bayesian optimization** — `Strategy::BayesianOpt { gp_kernel: ... }` for continuous search spaces with expensive evals (Target A is the ideal customer).
- **v3: Cross-target meta-learning** — share configs/scores across targets when they share structure (e.g. embedding-shape configs).

## Documentation Plan

- `crates/ix-autoresearch/README.md` — quickstart (30-line example), trait sketch, target list.
- Module rustdoc on each target explaining the search space and reward scalarization.
- `docs/MANUAL.md` — new section "§N. Autoresearch loops" with example invocations and the milestone promotion flow.
- `README.md` — bump tool count from 67 to 68; add `ix-autoresearch` to the "Integration" table.
- `docs/solutions/autoresearch/` — capture lessons learned during Phase 7 (subprocess kill semantics, mmap lock workarounds, etc.).
- Cross-link from this plan to `docs/brainstorms/2026-04-26-ix-autoresearch-brainstorm.md` (already in `origin:` frontmatter).

## Sources & References

### Origin

- **Brainstorm**: [docs/brainstorms/2026-04-26-ix-autoresearch-brainstorm.md](../brainstorms/2026-04-26-ix-autoresearch-brainstorm.md). Key decisions carried forward:
  - Numeric search v1 (Greedy + SA), LLM-perturb is v2 plug-in.
  - One kernel, three surfaces (CLI / MCP / demo target).
  - Sequential v1, parallel-opt-in v2 enum variant.
  - Soft deadline always + opt-in hard timeout per target.
  - Target schemas locked (Config + Score per target).
  - Gitignore `runs/`, track `milestones/` via `promote` verb.
  - Target ordering: C first (smoke test), B second, A last (cross-repo dep).

### Internal References

- Kernel surfaces:
  - `crates/ix-rl/src/bandit.rs:8-134` — UCB1 / EpsilonGreedy / Thompson (no shared trait; v1.5 use).
  - `crates/ix-cache/src/store.rs:165-213` — cache set/get API.
  - `crates/ix-pipeline/src/dag.rs:11-21` — confirmed acyclic; not used.
- Target adapters:
  - `crates/ix-grammar/src/weighted.rs:30-138` — WeightedRule + softmax + persistence (Target C).
  - `crates/ix-grammar/src/replicator.rs:59-160` — replicator dynamics + ESS (Target C).
  - `crates/ga-chatbot/src/main.rs:84-107, 396-428, 518-522` — qa subcommand surface + summary.json + exit code (Target B).
  - `crates/ga-chatbot/src/qa.rs:102, 254-262` — hardcoded thresholds to be parametrized (Target B).
  - `crates/ix-optick-invariants/src/main.rs` — invariant runner output schema (Target A).
- Surface patterns:
  - `crates/ix-skill/src/main.rs:11-100` — multi-verb CLI template.
  - `crates/ix-quality-trend/src/main.rs:14-105` — single-verb CLI + JSON artifact pattern.
  - `crates/ix-agent/src/{handlers.rs:5012-5095, tools.rs:2387-2462}` (commit 5b31e30) — recent MCP tool registration template.
  - `crates/ix-agent/tests/parity.rs:20` — parity allowlist.

### Related Work / Institutional Learnings

- `docs/plans/2026-04-19-tars-graph-persistence.md` — append-before-update JSONL pattern; single-writer locking; snapshot-every-N policy.
- `docs/solutions/integration-issues/cross-pollination-4-repo-ecosystem.md` — Demerzel governance submodule pattern, mmap lock notes.
- `docs/solutions/build-errors/windows-app-control-blocks-cargo-test-binaries.md` — WDAC OS error 4551; smoke `cargo test` at session start.
- `docs/solutions/workflow-patterns/multi-ai-review-before-merge.md` — review-while-worktree-exists pattern (relevant for v2 parallel-eval flow).

### External References

#### Karpathy autoresearch

- Karpathy, *autoresearch* — https://github.com/karpathy/autoresearch . Design we're inspired by but not adopting wholesale (Python+CUDA stack mismatch; reproducibility loss with LLM-only perturbation). The actual loop uses **greedy hill-climbing on `val_bpb`**; no SA, no bandit. IX's Greedy + SA is richer.
- Cerebras, *How to stop your autoresearch loop from cheating* — https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating . Three failure modes (agent drift, context pollution, reward hacking via metric definition); IX is mostly immune to the first two but `eval_inputs_hash` per iteration mitigates the third.
- DataCamp, *Guide to AutoResearch* — https://www.datacamp.com/tutorial/guide-to-autoresearch . Walks through the Karpathy loop semantics.

#### Multi-objective scalarization

- White Rose, *Methods for multi-objective optimization: an analysis* — https://eprints.whiterose.ac.uk/86090/8/WRRO_86090.pdf . Weighted-sum can only reach convex Pareto front regions.
- Tripp 2025, *Chebyshev Scalarization Explained* — https://www.austintripp.ca/blog/2025-05-12-chebyshev-scalarization/ . Practitioner argument for Chebyshev as default in Bayesian optimization.
- Springer J. Global Optim. 2024, *Tchebycheff weight-set decomposition* — https://link.springer.com/article/10.1007/s10898-023-01284-x . Recovers full Pareto front including non-convex points.
- Springer COAP 2023, *Pareto-front nonconvex optimal control* — https://link.springer.com/article/10.1007/s10589-023-00535-7 . Convergence-rate trade-off vs weighted-sum.

#### Dirichlet-on-simplex perturbation

- Wikipedia, *Dirichlet distribution* — https://en.wikipedia.org/wiki/Dirichlet_distribution . `Dir(α·w)` with `Var[X_i] = w_i(1-w_i)/(α+1)`.
- Stan Functions Reference: Dirichlet — https://mc-stan.org/docs/2_21/functions-reference/dirichlet-distribution.html . Reference for sampling and edge cases.
- CMU 10-701 Dirichlet recitation notes — https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/dirichlet.pdf . Concentration / corner-spiking intuition.
- Stan: Simplex Distributions — https://mc-stan.org/docs/functions-reference/simplex_distributions.html . Zero-component absorbing-state caveat.
- Aitchison's compositional-data perturbation — ResearchGate, *The shifted-scaled Dirichlet distribution in the simplex*. The formal algebra behind "perturb a simplex point."

#### SA cooling schedules

- Andresen & Nourani, *Permutational SA cooling-strategy comparison* — https://www.fys.ku.dk/~andresen/BAhome/ownpapers/perm-annealSched.pdf . Geometric outperforms logarithmic for small budgets.
- SciELO comparison (Lundy-Mees vs geometric) — https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462017000300493 .
- Banchs SA notes (Ben-Ameur calibration heuristic) — https://rbanchs.com/documents/THFEL_PR15.pdf .

#### Experiment-tracking schemas

- MLflow Tracking docs — https://mlflow.org/docs/latest/tracking/ . Field conventions and replay/resume best practices.
- Aim convert-data — https://aimstack.readthedocs.io/en/latest/quick_start/convert_data.html . Field naming conventions.
- ZenML, *MLflow vs W&B vs ZenML* — https://www.zenml.io/blog/mlflow-vs-weights-and-biases . Per-run metadata file vs inline-per-run debate (we choose inline as `RunStart` event).

#### Rust crate documentation

- `uuid` v7 — https://context7.com/uuid-rs/uuid/llms.txt . v7 stable in 1.10+, RFC 9562 ratified May 2024.
- `wait-timeout` — https://crates.io/crates/wait-timeout , https://github.com/alexcrichton/wait-timeout . `Child::wait_timeout(Duration)`; on Windows `WaitForSingleObject`; **does not kill grandchildren** under `Child::kill`.
- `blake3` — https://docs.rs/blake3/latest/blake3/ . Single-shot for <128 KiB; rayon-parallel for larger.
- `clap` derive — https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html , https://docs.rs/clap/latest/clap/trait.Subcommand.html . Multi-verb subcommand template.
- Cargo's `JobObject` impl — https://doc.rust-lang.org/nightly/nightly-rustc/src/cargo/util/job.rs.html (~80 LOC reference for v1.5 grandchild-leak fix).
- Microsoft Learn, *Win32 Job Objects* — https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects . `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` for tree-kill semantics.

### Related PRs

- IX commits this plan builds on: `5b31e30` (Grothendieck MCP tools, parity at 67), `cf87e82` (Z-pair invariant), `7b02a56` (algebra fixtures).
- GA companion PR: TBD — opens during Phase 6 with title `feat(FretboardVoicingsCLI): --weights-config flag for IX autoresearch loop`.
