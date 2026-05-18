# ix-autoresearch Context

> Fresh-session orientation for this crate. Read BEFORE touching it.

## What this crate is

Karpathy-style edit-eval-iterate kernel. Drives autonomous experiment loops over IX subsystems: perturb a config, evaluate (with deadline), accept/reject via a `Strategy`, append a JSONL `LogEvent`, update best-so-far on `(reward: f64, iteration: usize)`, repeat. Three target adapters ship today: `target_grammar` (sub-second smoke), `target_chatbot` (~30s/iter QA threshold tuning), `target_optick` (~140s/iter cross-language OPTIC-K self-tune). Per-run state is written under `state/autoresearch/runs/<uuidv7>/log.jsonl`; promotion to `state/autoresearch/milestones/<slug>/` is a separate sanitizing step.

## Key invariants (DO NOT VIOLATE)

- `Experiment::Config` MUST serialize deterministically — no `HashMap`; use `BTreeMap` or plain structs.
- Best-so-far is keyed on scalar `(reward: f64, iteration)`, NOT on `Score: PartialOrd` (PartialOrd is partial; would corrupt tie-breaks). Score is returned alongside but never compared.
- `HARD_KILL_CASCADE_THRESHOLD = 3`: after 3 consecutive `HardKilled` errors the kernel aborts the run. Do not raise this without a plan revision.
- `MCP_ITERATION_CAP = 10_000`: MCP handler rejects `iterations > cap`. CLI is unconstrained.
- `eval_inputs_hash()` MUST be a content hash, never a path hash — path hashes leak `C:\Users\<name>\…` into committed milestones (caught in security review).
- The per-run leaf dir is created with `create_dir` (not `create_dir_all`); if it exists we fail rather than overwrite — UUIDv7 pre-creation defense.
- Promote-time sanitization scrubs `sk-ant-`, `Bearer `, AWS/GitHub/Google API keys, absolute Windows + Unix paths. Add new patterns to `milestones::sanitize_text`, not callers.
- Schema-tolerant logs: every `LogEvent` carries `schema_version: u32` (currently `SCHEMA_VERSION`). Bump on breaking changes; readers tolerate older versions.
- `Strategy::Parallel(n)` and `Strategy::Custom` (LLM-perturb) are RESERVED for v2 — do not add v2 variants in v1 code paths.

## The 5-10 files that matter

- `src/lib.rs` — kernel: `run_experiment`, `resume_experiment`, `Experiment` trait, `Outcome`, the inner loop and SA-T0 calibration.
- `src/policy.rs` — `Strategy` enum + `AcceptancePolicy` impls (`GreedyPolicy`, `SimulatedAnnealingPolicy`, `RandomSearchPolicy`); Ben-Ameur T0 calibration.
- `src/log.rs` — `JsonlLog`, `FsyncPolicy`, tagged `LogEvent<Config, Score>` enum, `CostLedger`, log replay.
- `src/milestones.rs` — `promote_run`, slug/run-id validation, secret/path sanitization. The "publish" boundary.
- `src/cache.rs` — `CacheBridge`: opt-in cache keyed by `blake3(serde_json(config) || salt)`. Returns `None` when `cache_salt()` is `None`.
- `src/time_budget.rs` — `TimeBudget::soft` (deadline hint) vs `hard_timeout_per_iter` (shell-out kill).
- `src/target_grammar.rs` / `src/target_chatbot.rs` / `src/target_optick.rs` — the three reference target adapters; read these before adding a new target.
- `tests/kernel.rs` — Phase 1 acceptance suite (items a–r from the plan). Treat as the spec.
- `docs/plans/2026-04-26-001-feat-ix-autoresearch-edit-eval-iterate-plan.md` (in repo root) — locked design decisions.

## How to add a new target

1. Define `Config` and `Score` structs. `Config: Clone + Debug + Serialize + DeserializeOwned` and serializes deterministically (no `HashMap`). `Score: Clone + Debug + Serialize + DeserializeOwned + PartialOrd`.
2. Create `src/target_<name>.rs`. Implement `Experiment` with `baseline`, `perturb`, `evaluate`, `score_to_reward`.
3. Pick a `cache_salt()`: stable string (e.g. `"v1"`) if the target is deterministic; `None` if not. Bumping the salt invalidates cached evals.
4. If the target shells out, honor `soft_deadline` in `evaluate` AND enforce `TimeBudget::hard_timeout_per_iter` via process kill — return `AutoresearchError::HardKilled { detail }` so the cascade-abort counter trips.
5. Add a `pub use` line in `lib.rs`.
6. Add an integration test patterned on `tests/kernel.rs` (mock target with a known optimum). For shell-out targets, prefer the mocked-eval pattern in `tests/optick_mocked.rs`.

## What NOT to do here

- Don't add an `LLM-perturb` strategy to `Strategy` in v1. That's the reserved `Strategy::Custom` slot in v2; the plan explicitly defers it.
- Don't parallelize the inner loop. `Strategy::Parallel(n)` is reserved; the sequential `for` keeps log ordering and acceptance state coherent.
- Don't compare `E::Score` across iterations to update best. Use the scalar reward — `Score: PartialOrd` is intentionally insufficient.
- Don't widen the hard-kill cascade threshold to "just keep going". Three consecutive `HardKilled` is almost always a systemic failure (mmap lock, OOM, GA bug), not a bad config.
- Don't change the `<run-id>` directory format. Tools downstream (Demerzel governance ingest, milestone promote) parse the UUIDv7 path-traversal-safe stem.
- Don't sanitize at the caller side — keep all scrubbers in `milestones::sanitize_text`. Drift between callers was a Phase-1 review finding.

## Where to look for related context

- Crate `README.md` — quickstart, resume semantics, governance link.
- `docs/plans/2026-04-26-001-feat-ix-autoresearch-edit-eval-iterate-plan.md` — the source-of-truth design with locked decisions.
- `governance/demerzel/personas/autoresearch-driver.persona.yaml` — Demerzel invariant #34 affordances/constraints.
- `docs/MANUAL.md` — where autoresearch fits in the ix toolkit.
- Downstream consumers: chatbot QA (ga), OPTIC-K (ga/ix-optick), grammar (ix-grammar).
