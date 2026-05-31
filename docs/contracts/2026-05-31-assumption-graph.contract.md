# Temporal Assumption Graph — Cross-Repo Contract

**Version:** 0.4.0 (Phases 1–4 implemented — **freeze-candidate**, pending operator sign-off)
**Schema version:** 1
**Status:** Phases 1–4 shipped in `ix-assumption-graph` (graph + fusion + belief-time + MCP). The `node`/`edge`/`belief_event` records, the reused enums, and the `opinion` shape are stable. Freeze (→ v1.0.0, locking the one-way-door fields) is a **one-way door requiring operator sign-off** — NOT done unilaterally.
**Producers:** `ix-assumption-graph` builder (`from_workspace`/`from_parts`), `ix-ai-annotations` (via node promotion), `/deep-research` (research-claim nodes via the `assumption-graph-loop` skill), the `revise` loop
**Consumers:** `ix-assumption-graph` fusion/`view`/`belief_at`, the MCP tools `ix_assumption_query` + `ix_assumption_belief_at`, Demerzel promotion gate, (later) Prime Radiant / dashboard visualization
**Schema file:** `docs/contracts/assumption-graph.schema.json`
**Design doc:** `docs/plans/2026-05-31-temporal-assumption-graph.md`

---

## 1. Why This Contract Exists

The `@ai:` annotation contract (`docs/contracts/ai-annotation.schema.json`) captures a line-local claim with a hexavalent truth value — but each annotation is **isolated and static**. It does not relate to other claims (no support/contradiction structure), and it does not track *how a belief changed over time* or *why*. Research output (`/deep-research`, `seldon-research`) has the same problem one level up: a report's conclusions are a snapshot that cannot later tell you one of its load-bearing claims has since been contradicted.

This contract defines the records for a **unified, longitudinal assumption graph** spanning both **dev assumptions** (promoted from `@ai:` annotations) and **research claims** (ingested from `/deep-research`), where each node carries epistemic state that is **revised over belief-time** as evidence accumulates. It is the data layer for "a semantic story that runs over a long period," not a snapshot.

It deliberately **reuses** the `@ai:` annotation vocabulary (truth-value enum, certainty markers, confidence thresholds, `source` block, deterministic SHA-256 id) so a dev annotation promotes into a graph node without re-encoding, and adds exactly four things the annotation layer lacks:

1. **A subjective-logic opinion** `(b, d, u, a)` — a fusable certainty carrier (Jøsang).
2. **An ABA `contrary`** + **edges** (`contradicts` / `supports` / `depends_on` / `refined_from`) — relational structure.
3. **A belief-time axis** — `asserted_at` / `revised_at`, distinct from record bookkeeping and from domain valid-time.
4. **A belief-event log** — append-only revision history (Phase 3).

The formal grounding for each of these is in the design doc §3 (ABA, ATMS, subjective logic, belief-time bitemporal — externally verified) and is **not re-derived here**.

---

## 2. Record Types

The graph is stored as JSONL, one record per line. Every record has `schema_version` and a `record` discriminator: `node` | `edge` | `belief_event`.

### 2.1 Node — an assumption or claim
A superset of an `@ai:` annotation. Reused fields keep identical semantics. New fields: `opinion`, `contrary`, `origin`, `belief_time`.

```json
{
  "schema_version": 1,
  "record": "node",
  "id": "sha256:f3b1c2...",
  "kind": "assumption",
  "claim": "caller holds the lock",
  "truth_value": "P",
  "certainty": "assumed",
  "confidence": 0.7,
  "opinion": { "b": 0.7, "d": 0.0, "u": 0.3, "a": 0.5 },
  "contrary": null,
  "origin": { "annotation_id": "sha256:f3b1c2...", "research_run": null },
  "source": { "author": "claude", "model": "claude-opus-4-8", "independence_class": "human-review" },
  "location": { "path": "crates/ix-x/src/y.rs", "line_start": 88, "line_end": 88 },
  "belief_time": { "asserted_at": "2026-05-31T00:00:00Z", "revised_at": "2026-05-31T00:00:00Z" },
  "created_at": "2026-05-31T00:00:00Z",
  "updated_at": "2026-05-31T00:00:00Z"
}
```

### 2.2 Edge — a relation between nodes
```json
{ "schema_version": 1, "record": "edge", "id": "sha256:...", "type": "contradicts",
  "from": "sha256:<conclusion>", "to": "sha256:<assumption>", "weight": 0.8,
  "created_at": "2026-05-31T00:00:00Z" }
```
`contradicts` carries ABA attack semantics; the acceptability decision uses **only** the `contradicts` sublattice in v1 (see §6). The other edge types are provenance/propagation hints, excluded from the formal computation.

### 2.3 Belief-event — an append-only revision (Phase 3)
```json
{ "schema_version": 1, "record": "belief_event", "node_id": "sha256:...",
  "at": "2026-06-15T00:00:00Z", "from_truth_value": "P", "to_truth_value": "F",
  "to_opinion": { "b": 0.05, "d": 0.9, "u": 0.05, "a": 0.5 },
  "trigger": "test-run", "evidence": "tests/lock_test.rs:31" }
```
Replaying belief-events reconstructs `belief_at(t)`.

---

## 3. Reused Vocabulary (see `ai-annotation.schema.json` for full tables)

- **`truth_value`** — hexavalent `T/P/U/D/F/C`, identical to the annotation contract and Demerzel `hexavalent-distribution.schema.json`.
- **`certainty`** — the eight annotation markers, plus **`adversarial-panel`** (verdict from a multi-judge panel; see §7).
- **`confidence`** — `0.0–1.0`, Demerzel thresholds (≥0.9 / ≥0.7 / ≥0.5 / ≥0.3 / <0.3).
- **`source.author`** — the annotation authors, plus **`deep-research`**.
- **`id`** — deterministic SHA-256; dev nodes use the annotation scheme (`path:line_start:kind:claim`) so an annotation and its node share identity; research nodes hash `normalized(source_url + ':' + claim)`.

---

## 4. The Opinion ↔ Truth-Value Bridge

`opinion` is the math carrier; `truth_value` is the governance/human-readable label. They are kept consistent by the producer via the projection table (design doc §5). Summary: `b` high & `u` low → `T`; `b>d`, moderate `u` → `P`; high `u` → `U`; `d>b` → `D`; `d` high → `F`; **`b` and `d` both high from _conflicting_ fused evidence → `C`** (which is outside the `b+d+u=1` simplex — Belnap supplies `C`, subjective logic supplies the rest). Projected probability `E = b + a·u` is the scalar for ranking and promotion thresholds.

---

## 5. Belief-Time vs Other Time Axes

| axis | meaning | where |
|---|---|---|
| `created_at` / `updated_at` | record bookkeeping (when the JSONL line was written) | every record |
| `belief_time.asserted_at` / `revised_at` | **when we first held / last changed this belief** | node |
| `valid_time` (future) | when the claim is true in the domain | deferred |

The `belief_time` axis is what the 2015 belief-based bitemporal model calls "belief time," distinct from transaction time (design doc §3). It is the axis `belief_at(t)` queries.

---

## 6. Acceptability (v1 scope)

Promote/demote decisions follow ABA dispute-derivation semantics over the `contradicts` sublattice only. Support edges are **deliberately excluded** from the formal acceptability computation in v1, because support pushes the framework into non-flat / bipolar argumentation where Dung's semantics do not all transfer (design doc §6, caution 2). Support/depends_on/refined_from are retained for provenance and for human/agent navigation.

---

## 7. Fusion Discipline & Panel Discipline (load-bearing)

Evidence accumulates by **subjective-logic fusion** over the per-node `opinion`. The rules below are grounded in externally-verified findings (design doc §6, and the panel mini-survey 2026-05-31).

### 7.1 Fusion
**Cumulative (additive) fusion only across distinct `independence_class` values.** Cumulative fusion assumes independent evidence; summing correlated sources over-counts and **overstates certainty**. Observations sharing a class (same model lineage, repeated runs of one agent) must use **averaging** fusion.

### 7.2 The `adversarial-panel` certainty: fail-closed gate, NOT a truth oracle
The "do multi-LLM panels improve correctness or amplify shared bias?" question is now **resolved** (it was the one fully-failed angle of the first research run; a focused, source-verified re-run settled it). The answer is asymmetric and consequential:

- **Panels confirm well, reject poorly.** LLM judges show ~96% true-positive but **<25% true-negative** rates, and **majority voting does not fix this** (Jain et al. 2025, arXiv:2510.11822). A panel is reliable at *agreeing something looks valid*, unreliable at *catching invalidity*.
- **Independence is the mechanism, not panel size.** A diverse panel beats a single judge *only* when models are disjoint families (Verga et al. 2024, arXiv:2404.18796). Same-family debate/aggregation *amplifies* shared errors — "belief entrenchment" (arXiv:2503.16814), and debate can *decrease* accuracy even when strong models outnumber weak ones (arXiv:2509.05396).
- Single-judge biases (position, verbosity, self-preference) persist at panel level; self-preference is a perplexity/familiarity effect, so same-family panels do **not** cancel it (arXiv:2306.05685, arXiv:2410.21819).

**Mandatory discipline for any `adversarial-panel` verdict driving promote/demote:**
1. **Asymmetric gate.** Require panel consensus to **promote** toward `T`; let **any single credible dissent** force `U`/`D`/`C` rather than promotion. Never let majority-approve override a substantive minority refutation.
2. **Map disagreement to hexavalent values — do not collapse to a vote.** Unanimous + high per-judge confidence → `T`/`F`; split or low-confidence → `U`/`P`; active contradiction between judges → `C`. This preserves the dissent signal a majority vote destroys.
3. **Mandate + measure judge independence.** Panel must span genuinely different pretraining lineages (distinct `independence_class`). Operationalize decorrelation (e.g. a focal-diversity metric, arXiv:2410.03953) or track per-judge error correlation on a gold set; if errors are correlated, agreement is illusory — discount it.
4. **Fail closed.** When a judge errors/abstains, never default to "survived" (`feedback_workflow_schema_catch_green_but_dead`).
5. **Guardrail metric = TNR, not agreement.** Hold out a labeled set of assumptions with known T/P/U/D/F/C verdicts and report the panel's **true-negative rate** (catch-rate on bad claims). A panel that agrees 95% of the time but has <30% TNR is "green but dead" for a demotion gate.

---

## 8. Storage Layout (planned)

| file | producer | consumer |
|---|---|---|
| `state/assumptions/graph.jsonl` | `ix-assumption-graph build` | acceptability/query engine, dashboard |
| `state/assumptions/belief-events.jsonl` | revise loop (Phase 3) | `belief_at(t)` / `diff(t1,t2)` |

---

## 9. Phase Plan (this contract = Phase 0)

- **Phase 0 (this artifact):** contract + JSON schema. **No code.**
- **Phase 1:** `ix-assumption-graph` crate — build `Dag<AssumptionNode>` from `@ai:` annotations; derive `contradicts` from `contrary`.
- **Phase 2:** opinion fusion + Belnap `C` synthesis + the §4 bridge (reuses `ix-fuzzy`, `hari`).
- **Phase 3:** belief-event log + `belief_at(t)` / `diff(t1,t2)`; the longitudinal loop ( `/deep-research` → research-claim nodes; scheduled re-verify; Demerzel promotion gate).
- **Phase 4 (shipped):** node enums split from the annotation enums — `research-claim` kind and `adversarial-panel` certainty are now first-class (`NodeKind`/`NodeCertainty`); faceted navigation `view()` (by namespace / kind / domain + escalations); MCP tools `ix_assumption_query` (faceted view) and `ix_assumption_belief_at` (point-in-time belief state). **Freeze (→ v1.0.0) pending operator sign-off.**

### Navigation & structure

The graph is navigable along two axes (`AssumptionGraph::view` → `FacetedView`):
- **namespace** — code-anchored nodes by crate (`crates/<crate>/…` → `<crate>`); research claims under `research`;
- **domain / kind** — `dev` vs `research`; and the node `kind`.

Finer "functional area" grouping (below crate/module) would need explicit tags
on annotations — not currently modeled. Humans navigate via the
`ix-assumption-graph-report` CLI or (future) Prime Radiant; agents via
`ix_assumption_query`.

---

## 10. Versioning

`v0.1.0` (draft). The `record` discriminator, the reused enums, and the `opinion` shape are intended to be stable from Phase 1. `schema_version` bumps only on a field rename/removal or an enum widening that consumers must opt into; new optional fields are additive.

### Changelog
- **v0.1.0 (2026-05-31, schema_version 1)** — initial draft: `node` / `edge` / `belief_event` records; reuses `@ai:` annotation vocabulary; adds subjective-logic `opinion`, ABA `contrary` + edges, `belief_time` axis, and the `adversarial-panel` certainty + `independence_class` fusion tag.

**One-way-door note:** the `id` scheme and the `truth_value` enum align with the `@ai:` annotation contract and Demerzel — changing them is a one-way door requiring cross-repo sign-off. Everything else in this draft is two-way until Phase 4.
