---
date: 2026-04-26
topic: ix-autoresearch
---

# ix-autoresearch: a Karpathy-style edit-evaluate-iterate loop framework for IX

## What We're Building

`ix-autoresearch` — a Rust kernel that runs an autonomous experiment loop
(perturb → evaluate → accept-or-revert → log → repeat) over an `Experiment`
trait, plus three target adapters wiring it to real IX subsystems.

The framework is exposed three ways from the *same* kernel — no duplication:

1. **CLI binary** (`ix-autoresearch`) — private research tool. Run `ix-autoresearch run <target> --iterations 200`, walk away, read the JSONL log in the morning.
2. **MCP tool** (`ix_autoresearch_run`) — product capability. External agents (GA, TARS, Claude Code) can invoke autonomous tuning loops on demand.
3. **Demo target** — flagship visible metric move (OPTIC-K leak % drops overnight) suitable for ecosystem storytelling.

The kernel is **numeric-search-driven** in v1 — bandit (UCB1 from `ix-bandit`)
or simulated annealing over a constrained config space. The `Experiment` trait
leaves a clean seam for LLM-driven perturbations as a future pluggable
`perturb()` impl, but no LLM is in the v1 critical path.

## Why This Approach

We considered three engine shapes:

- **(α) Numeric search only** — Bandit / SA / DE over a constrained config. Pure Rust, deterministic, no API keys. Fits OPTIC-K weights and grammar weights trivially; fits chatbot only if the search space is reduced to parameters.
- **(β) LLM agent edits code** — Captures Karpathy's autoresearch *spirit* (an LLM rewrites `train.py` overnight). Requires sandbox + revert + API integration + trust boundaries. Heavyweight v1.
- **(γ) Hybrid** — Both engines through one trait. Most flexible, most surface area, most things to get wrong in a single session.

**Chosen: (α) with a clean seam for (β).** The autoresearch *story* lands either way ("IX self-tuned an embedding overnight"), and the bandit version is reproducible — a property the LLM version cannot offer. Adding (β) later is a different `perturb()` impl, not a kernel rewrite.

## Targets, in Priority

### A — OPTIC-K self-tuning *(flagship demo)*

- **Search space**: partition weights `[STRUCTURE, MORPHOLOGY, CONTEXT, SYMBOLIC, MODAL, ROOT] ∈ ℝ⁶` (currently fixed in `EmbeddingSchema.cs`).
- **Eval**: `ix-optick-invariants` + `ix-embedding-diagnostics` produce JSON (leak %, retrieval match, cluster ARI, #25/#32/#36 pass rates).
- **Visible metric**: STRUCTURE leak % — the catalog's #28 (currently the only embedding-side FAIL invariant).
- **Loop time**: ~140s/iter — GA must rebuild the 313k-voicing index between perturbations.
- **Cross-language**: Rust kernel triggers a C# rebuild via shell-out to `FretboardVoicingsCLI`.

### B — Chatbot QA improvement *(flagship product)*

- **Search space (v1)**: fixture-response confidence thresholds + regression-gauge accept/reject thresholds.
- **Eval**: `ga-chatbot qa --output findings.jsonl` deterministic suite + judge panel.
- **Visible metric**: adversarial-corpus pass rate.
- **Loop time**: ~30s/iter, pure Rust.
- **v2 extension** under (β): perturb fixture text or grounding logic, not just thresholds.

### C — Grammar evolution *(kernel smoke test)*

- **Search space**: `WeightedRule` weights in an existing weighted CFG.
- **Eval**: held-out parse-success rate (or ESS-stability via `ix-grammar::replicator::detect_ess`).
- **Loop time**: sub-second, fully in-process.
- **Role**: the *first* target to wire end-to-end — validates the kernel against a fast, deterministic loop before paying for A/B's external integration costs.

## Key Decisions

### Architecture

- **Numeric kernel only in v1.** LLM-perturb is a future pluggable engine via a different `perturb()` impl, not a v1 requirement. Justified by reproducibility + scope.
- **One core, three surfaces.** CLI binary, MCP tool, and demo target all dispatch to the same `run_experiment` core — no parallel implementations.
- **Decision policy default = UCB1 bandit** (from `ix-bandit`). Simulated annealing as an alternative for continuous spaces.
- **Sequential v1, parallel-opt-in v2 seam.** `run_loop` is a plain `for` loop in v1; the strategy enum reserves a `Strategy::Parallel(n)` variant so v2 doesn't break the v1 API. A is the bottleneck and *cannot* parallelize regardless (GA mmap lock).
- **Time budget = soft deadline always + opt-in hard timeout.** `run_iteration` accepts `(soft_deadline, Option<hard_timeout>)`. Eval gets the soft deadline as a hint; if `hard_timeout` is set the kernel watchdog kills the worker. C skips both (sub-second), B uses soft, A uses both (the C# CLI sometimes hangs).

### Trait shape (locked)

```rust
pub trait Experiment {
    type Config: Clone + Debug + Serialize + DeserializeOwned;
    type Score: Clone + Debug + Serialize + DeserializeOwned + PartialOrd;

    fn baseline(&self) -> Self::Config;
    fn perturb(&mut self, current: &Self::Config, rng: &mut StdRng) -> Self::Config;
    fn evaluate(&mut self, config: &Self::Config) -> Result<Self::Score, AutoresearchError>;
}
```

### Target config + score schemas (locked)

**Target C — `GrammarConfig { rule_weights: Vec<f64>, temperature: f64 }`** with score `{ parse_success_rate, ess_stability }`. Perturb: Gaussian noise on weights (reflective bound at 0); log-uniform on temperature. Eval: replicator simulation + parse against held-out corpus.

**Target B — `ChatbotConfig { deterministic_pass_threshold, judge_accept_threshold, fixture_confidence_floor: f64, strict_grounding: bool }`** with score `{ pass_rate, false_positive_rate, false_negative_rate }`. Lex-order: max pass_rate, tie-break (low FP, then low FN).

**Target A — `OpticKConfig { structure_weight, morphology_weight, context_weight, symbolic_weight, modal_weight, root_weight: f64 }`** all in [0,1], sum = 1. Score `{ structure_leak_pct, retrieval_match_pct, inv_25_pass_rate, inv_32_pass_rate, inv_36_pass_rate }`. Perturb: Dirichlet-on-simplex (preserves sum-to-1 without rejection sampling). Plumbing: Rust writes config to `state/autoresearch/<run-id>/optick-config.json`, shells out to GA's `FretboardVoicingsCLI --weights-config <path>`, parses resulting `embedding-diagnostics.json`.

### Operations

- **JSONL run log** at `state/autoresearch/runs/<run-id>/log.jsonl`. One line per iteration: timestamp, perturbation vector, eval result, accept/reject, *previous* config hash for revert.
- **Safety: opt-in autonomous revert.** Default behavior keeps every iteration's state for manual replay; autonomous revert on regression is a flag.
- **Gitignore policy: runs ignored, milestones tracked.** `state/autoresearch/runs/` is in `.gitignore`; `state/autoresearch/milestones/<date>-<target>-<note>/` is tracked. `ix-autoresearch promote <run-id>` copies the final config + summary stats into the milestones tree and stages it. Audit trail without per-iteration commit noise.
- **C ships first as the kernel validation target.** A and B come second because they require external integration; we don't want to debug the kernel through the cross-language path.
- **A's eval target = CI-reduced index, not the live `optick.index`.** Avoids GaApi's mmap lock and gives reproducible test conditions; live-corpus runs are a separate flag.

## Open Questions

- **GA CLI cross-repo prerequisite for Target A.** GA's `FretboardVoicingsCLI` does not currently accept a `--weights-config <path>` flag. A's loop cannot run end-to-end until that flag lands in the GA repo. Until then, A is dark — the Rust adapter is testable with a mocked rebuild step. Need a small companion PR on the GA side; not blocking on the IX kernel work.

## Next Steps

→ `/ce:plan` for implementation details: crate layout, file-by-file changes, trait + struct definitions, integration test strategy, and the order of work (kernel + log + policy → C end-to-end → CLI polish → MCP wrapper → B end-to-end → A adapter with mocked GA → land GA `--weights-config` flag → A end-to-end).
