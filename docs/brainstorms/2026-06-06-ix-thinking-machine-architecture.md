# Decision brief — IX "thinking machine" architecture fork

**Date:** 2026-06-06
**Status:** open decision — panel input requested
**Audience:** octo personas, Codex/Gemini CLIs, Junie (IDE), NotebookLM (web), and the human operator.

---

## 1. The goal

Make IX a "thinking machine": an AI agent that translates **natural language ↔ dynamic IX
pipelines, bidirectionally**, executes them, and is then **dogfooded to improve IX itself** in a
positive feedback loop (NL requests that fail to translate = a precise, logged backlog of IX gaps;
NL-analysis pipelines that run = concrete IX weaknesses to fix).

A 15-agent research+design workflow already produced a verified architecture ("Governed IR-Hub":
thin language shell over a Rust truth-spine, `PipelineSpec` as the hub IR, `lower()` as the sole
accept/reject oracle, fail-closed governance). This brief is about a **specific fork** that surfaced
*after* that synthesis, when the operator asked: "should we favour MCP sampling instead of an
Anthropic API key?"

## 2. Grounded findings (verified in source — not assumptions)

1. **MCP server-initiated sampling is already fully built in IX.** `ServerContext::sample()` /
   `sample_with_image()` (`crates/ix-agent/src/server_context.rs`) issue `sampling/createMessage`
   over a hand-rolled bidirectional JSON-RPC dispatcher (std-only, 30s timeout). **No API key — the
   connected MCP client provides the model.** Four tools already use it: `ix_explain_algorithm`,
   `ix_triage`, `ix_render_audit`, `ix_pipeline_compile`.
2. **An NL→pipeline compiler already exists**: `ix_pipeline_compile` (`tools.rs:587`) — NL sentence
   → `ctx.sample()` → registry validation (unknown tools, dup ids, cycles) → emits a spec runnable
   by `ix_pipeline_run`. The thinking machine is ~half-built, the key-free way.
3. **Two divergent pipeline systems exist:**
   - **System A (MCP/agent):** `{steps:[…]}` + `depends_on` + `$step_id.field` substitution; its own
     topological executor (`ix_pipeline_run`, `tools.rs:356`); its own registry validator. **Has
     sampling.**
   - **System B (CLI/canonical):** `PipelineSpec` / `ix.yaml` (`stages{}` keyed map, `deps`,
     `{"from":"stage.key"}`) + `ix-pipeline::lower()` / `executor` / `lock` (reproducibility) +
     `ix pipeline {validate,dag,run}`. The stronger, reproducible engine. **No NL/sampling yet.**
     A new `ix pipeline schema` verb (this session) emits its JSON Schema with the skill enum drawn
     live from the registry; `ix list skills --schemas` emits the bulk catalog (arg schema +
     governance tags + pipeline-callable flag). Both transport-agnostic.
4. **Sampling liveness is UNVERIFIED — green-but-dead risk.** Every sampling test uses a *stub*
   client. The tests note "the full compile flow needs a live MCP client for sampling." Whether a
   real host (Claude Code) fulfills `sampling/createMessage` is unproven; `mcp__ix__*` is not even
   connected to the current session (release binary not built). Operator decision: **verify host
   sampling support empirically before committing.**

## 3. Priors / constraints (do not relitigate)

- **Simplicity-first / YAGNI / surgical** (Karpathy rules). Minimum code for the exact problem.
- **Fail-closed governance** is the IX differentiator: `governance.check`, hexavalent T/P/U/D/F/C,
  `AssumptionGraph::fuse`, `AlignmentPolicy::should_escalate`.
- **The "did this improve IX?" oracle must be executable** (`cargo test` + sentrux + quality-trend),
  **never an LLM judge panel** (≈96% TPR / <25% TNR — confirms well, catches invalidity poorly).
- **No green-but-dead.** A tool tested green against a stub but dead against the real host is worse
  than no tool.
- Prior synthesis recommended **`PipelineSpec` as THE canonical IR**, and flagged "deprecating the
  `{steps:[…]}` path" as a **one-way door**.

## 4. The options on the table

- **O1 — Retarget the sampling compiler onto canonical PipelineSpec.** Keep `ctx.sample()` + the
  prompt pattern from `ix_pipeline_compile`, but change the target to `PipelineSpec`, validate via
  `lower()`, execute via the canonical executor. Unifies A+B on the better IR. Key-free. (Current
  lead recommendation.)
- **O2 — Keep/extend System A's `{steps:[…]}`.** Fastest (mostly exists), but entrenches the
  non-canonical IR and leaves two pipeline systems forever.
- **O3 — Standalone Python shell + Anthropic API key.** Reintroduces a key sampling already removes;
  creates a *third* pipeline path. (Current lead recommendation: reject as primary.)
- **O4 — Hybrid:** sampling-primary `ix-agent` tool on PipelineSpec, with a direct-LLM (API-key)
  fallback for headless/CI where no sampling-capable client exists.

## 5. What we're asking you

1. **Pick an option (O1–O4) or propose a better one.** Justify against the constraints in §3.
2. **Sampling liveness & fallback:** if a target host (Claude Code) does NOT implement
   `sampling/createMessage`, what is the right posture — block on it, hybrid fallback, or accept
   sampling-only and pick a different host (e.g. a custom ix federation client / "agent-blackbox")?
3. **Unification cost:** is collapsing System A into System B worth the one-way-door cost *now*, or
   should the thinking machine ship on whichever IR is faster and unify later?
4. **Walking skeleton:** the smallest end-to-end demo that proves NL→pipeline→run→NL **and** seeds
   the dogfood loop — buildable in ~1–2 days. Be concrete.
5. **Biggest risk we're not seeing?** Name the failure mode most likely to make this green-but-dead.

Answer tersely and decisively. Cite the option letters. If you'd reject the lead recommendation,
say why in one paragraph.

---

## 6. Panel resolution (2026-06-06)

Six independent perspectives (Codex/gpt-5.5, Gemini, architecture-strategist, code-simplicity-reviewer,
governance-auditor, claude-code-guide) + operator. **Converged decisively.**

**Decisive finding (claude-code-guide, citing the spec):** MCP `sampling/createMessage` is
**DEPRECATED (SEP-2577, ~mid-2026)** and **Claude Code does not implement it as a provider** (nor
Claude Desktop / Agent SDK). The spec's own migration guidance: *integrate directly with LLM
provider APIs.* → IX's `ctx.sample()` tools are **green-but-dead** against our host. This **flips the
transport question**: the key-free sampling path is a deprecated dead-end; the direct-LLM-API path is
the spec-endorsed, only-live option.

**Converged architecture (unanimous on the IR/governance core):**
- **IR: O1 — retarget onto canonical `PipelineSpec`, validate via `lower()`, execute via System B.**
  All six agree. `{steps:[…]}` (System A) is the wrong IR; do not ship the thinking machine on it.
- **Transport: NOT sampling. Direct LLM provider API** as a governed proposer (the deprecation's own
  recommendation). This is *O1's architecture with a direct-API brain instead of `ctx.sample()`* —
  not a standalone Python shell (O3 rejected as a third path), not sampling+fallback (no live
  sampling to fall back *from*, so the hybrid debate is moot).
- **Unification is CHEAP, not an engine rewrite:** `crates/ix-agent/src/registry_bridge.rs` already
  has a bijective `mcp_name()`/`mcp_to_skill_name()` map — both IRs dispatch the *same* `fn_ptr`s.
  Retarget = prompt-target swap + name normalizer + ref-syntax swap.
- **Do NOT delete System A in the same PR** (Article 3 / one-way door). Retarget + deprecate-with-shim;
  delete-trigger = zero non-test callers of the `{steps}` executor for 14 days.

**Critical correction (governance-auditor, verified in source):** `lower()` is **structure-only** —
skill existence, from-refs, acyclicity. It has **ZERO constitutional gating.** The brief/synthesis
*assumed* it was the fail-closed oracle; it is not. A generated pipeline reaches the executor
**unreviewed**. The single highest-leverage fix: add `governance.check_action` into a
`lower_governed()` wrapper (default-reject on non-`compliant` / `U` / `C` verdicts), called as a
precondition in the execution path so raw CLI/MCP runs can't bypass it.

**Three risks to bake in:**
1. *Governance gap* (above) — the gate must exist, not be assumed.
2. *Instructional drift* (Gemini) — LLMs trained on simple `{steps:[…]}` lists will hallucinate
   System A shape instead of map-based `PipelineSpec` stages → `lower()` rejects valid-looking
   requests, feels broken. Mitigate with strong `PipelineSpec` few-shots + the schema enum + the
   repair loop feeding `lower()` errors back verbatim.
3. *Schema-valid-but-useless specs* (Codex) — the dogfood "did-IX-improve?" oracle must be
   **executable** (`cargo test` + sentrux + `ix-quality-trend`), rejecting non-improvement — never an
   LLM judge.

**One-way doors needing sign-off:** hard-deletion of `{steps}`/`ix_pipeline_run`; freezing the
`PipelineSpec`/`ix pipeline schema` JSON-Schema hash; the governance-verdict contract in
`lower_governed()` output.

**Liveness posture (operator chose "verify host first"):** resolved — sampling is dead on our host,
so no probe needed and no fallback to build. Direct LLM API is primary, full stop.

