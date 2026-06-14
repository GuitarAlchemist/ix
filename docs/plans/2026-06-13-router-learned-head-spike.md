# Spike A — learned router head from synthetic trajectories

**Date:** 2026-06-13
**Origin:** unanimous `scoped-spike` verdict on `nousresearch/hermes-agent` (see `state/handoffs/2026-05-31-hermes-agent-eval-brief.md`). Borrow ONE capability — batch trajectory generation — as an offline, governed data factory. Do **not** install the Hermes harness/memory/cron/gateway.
**Status:** EXECUTED 2026-06-13 — **GO-leaning.** Learned head beats the full production router on held-out: **+6.3pp in-scope acc (81.8 vs 75.5), +7.2pp macro-F1, +31.3pp OOS-decline**. Full results: `state/router-spike/RESULTS.md`. Remaining: close scaleinfo + Phase 4 (wire into GA behind a flag, schema sign-off).
**Reversibility:** two-way door (learned head is additive, swappable behind a flag) EXCEPT the JSON weights contract (one-way-ish — see Phase 4).

---

## Hypothesis (falsifiable)

> A lightweight **learned classifier head** over the *same* query embedding the router already computes — replacing "max-cosine-to-nearest-example **+ regex hint boosts**" — generalizes better than the hand-tuned regex hints, measured on **held-out real traffic**, breaking the 93.4% teaching-to-test ceiling.

If a linear head on embeddings does **not** beat the regex baseline on held-out data, an LLM fine-tune (Phi/Llama-1B) is even less likely to be worth the operational cost → **kill the spike, keep the current router.**

## Why this is the right minimal bet (simplicity-first)

The router (`ga/Common/GA.Business.ML/Agents/Intents/SemanticIntentRouter.cs`) is *already* embedding-first: nomic-embed cosine to per-intent example centroids, threshold 0.55, plus `DefaultRoutingHintProvider` regex boosts (+0.06). The regex layer is what hit the "teaching-to-test" ceiling (hints authored against the fixed 86-prompt corpus). Swapping "nearest-centroid + regex" for a **trained linear/softmax head over the same vector** learns the decision boundary from data instead of hand-rules — the smallest change that can falsify the hypothesis. **No LLM fine-tune in Phase 1.**

---

## ⚠️ Phase 0 — GATE: get a trustworthy held-out test set

The held-out eval source (`ga/state/telemetry/routing/*.jsonl`, `RoutingTelemetryLog`, shipped 2026-05-31 expressly for this) currently holds **4 records total** — smoke pings, not traffic. The generalization guardrail has no data. **This blocks everything downstream.** Two options, pick one before Phase 1:

- **(0a) Populate telemetry, then hand-label.** Drive ~150–200 realistic queries through the *live* router (reuse the GA chatbot AFK / replay harness, or a scripted query list authored by someone/something that will NOT see the synthetic train set). Router logs each to JSONL. Hand-label the chosen intent (or `__none__` for OOS). This is the strongest held-out set — real surface forms.
- **(0b) Independent hand-authored held-out set.** A *different* model/persona than the trajectory generator authors ~100 real-style queries with gold labels, committed before any training. Weaker than 0a (still synthetic) but unblocks immediately. Strict rule: the Phase-2 generator never sees these prompts.

**Exit criterion for Phase 0:** ≥100 labeled held-out queries, ≥4 per in-scope intent, plus ≥15 OOS. Disjoint from the 86-prompt corpus and from Phase-2 synthetic data. If this can't be met, the spike is **data-blocked** — say so and stop.

---

## Phase 1 — three disjoint datasets (no leakage)

| Set | Source | Role |
|---|---|---|
| **TRAIN** | Phase-2 synthetic generation | fit the head |
| **DEV** | existing 86-prompt corpus (`ga/.../Data/routing-eval-prompts.json`) | hyperparam/threshold selection. NB: the regex baseline was tuned here → it is the baseline's home turf, so DEV flatters the baseline, not the head |
| **TEST** | Phase-0 held-out | the *only* number that decides go/no-go |

Hard rule: **synthetic data never touches TEST.** Report train→dev→test gap explicitly; a large train-test gap = synthetic distribution mismatch → distrust the head.

## Phase 2 — trajectory generation (the Hermes borrow, governed)

Generate ~30–50 labeled prompts per in-scope intent (~500–800 total) via a **direct LLM provider API behind a governance gate** (per `[[reference_mcp_sampling_deprecated]]` — `ctx.sample()` is green-but-dead; use a governed direct call). For each intent feed its production `Description` + `ExamplePrompts` and ask for diverse variants, including **near-boundary hard negatives** (e.g. chordinfo-vs-scaleinfo confusables) and conversational/jargon registers. Log every generation (Demerzel Article 7 auditability). Also generate OOS distractors for the decline class.

## Phase 3 — embed + train (IX-side, Rust)

1. **Embed all three sets with the *same* embedder the router uses** (nomic-embed-text via Ollama) — non-negotiable for transfer. Either call the same Ollama endpoint from IX, or have GA export embeddings. (Side experiment: bge-base via IX `fastembed` per `[[reference_embedding_gate_runtime_fastembed]]`, only if we also swap the production embedder — otherwise vectors won't match.)
2. **Train a linear/softmax head** in `ix-supervised` (logistic regression) — optionally a 1-hidden-layer MLP via `ix-nn` if linear underfits DEV. Output: weight matrix `[n_intents × dim]` + bias `[n_intents]`.
3. OOS handling: keep the existing threshold-decline mechanism (max softmax prob < τ → decline), τ tuned on DEV.

## Phase 4 — consumption contract (⚠️ one-way door — needs sign-off)

Export head as JSON: `state/router/learned-head.json` = `{ schema, embedder_id, dim, intent_ids[], weights[][], bias[] }`. GA loads it and applies a dot-product + softmax at route time (trivial, ~free vs the current cosine loop; no new GA runtime dependency). **The schema is the locked surface** — version it, log the decision per CLAUDE.md one-way-door rule. Reversible at runtime: keep the regex router behind a flag; the learned head is swappable/removable.

## Phase 5 — evaluate on TEST + decide

Run three configs through the existing `RoutingEvalHarness` (extended to load the learned head) on the Phase-0 **TEST** set:
- (a) current embedding **+ regex** baseline
- (b) learned head (no regex)
- (c) learned head + regex (does the boundary still want hints?)

### Success criteria (verifiable — Karpathy rule 4)

- **PRIMARY:** learned-head in-scope accuracy on **TEST** ≥ regex-baseline in-scope accuracy on the **same TEST**, by a meaningful margin (target **≥ +3 pp**).
- **GUARDRAIL 1:** no in-scope intent drops below **0.80 F1** on TEST.
- **GUARDRAIL 2:** OOS-decline rate on TEST **≥ baseline** (doesn't start forcing non-music queries into skills).
- **GUARDRAIL 3:** routing p95 latency unchanged.
- **KILL:** if (b)/(c) don't beat (a) on TEST → document the negative result, keep the current router, do not ship. Do **not** escalate to an LLM fine-tune without first explaining why embeddings carried the signal but the linear/MLP head couldn't use it.

## Risks (the three reviewers' consensus, mapped to defenses)

1. **Governance bypass [T]** — generation runs through a governed direct-API call with full audit log; no Hermes runtime, no ungoverned action surface.
2. **Synthetic overfit / teaching-to-test [P]** — TEST is real/independent and never seen by the generator; train→test gap reported; KILL criterion is on TEST only.
3. **Integration drag [T]** — zero new GA runtime dep (JSON + dot-product); training stays in IX; nothing from Hermes is installed.

## Cross-repo touch points

- **ga:** `SemanticIntentRouter` (consume head), `RoutingEvalHarness` (load head + eval on TEST), `routing-eval-prompts.json` (DEV), `state/telemetry/routing/` (Phase-0 source).
- **ix:** trajectory generation script + `ix-supervised`/`ix-nn` training + JSON export.
- **contract:** `learned-head.json` schema (Phase 4) — coordinate, version, sign off.

## First action

Resolve Phase 0 (0a vs 0b). Until ≥100 labeled held-out queries exist disjoint from the corpus, **do not generate synthetic data or train** — the result would be unfalsifiable.
