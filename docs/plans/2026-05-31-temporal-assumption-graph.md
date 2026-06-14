# Temporal Assumption Graph — Architecture Spec + Build-vs-Buy Verdict

- **Date:** 2026-05-31
- **Branch context:** `feat/adversarial-llm-panel`
- **Status:** Design spec (no code yet). Decision doc per CLAUDE.md "Log one-way doors".
- **Author:** Claude (Opus 4.8) with operator Stephane Pareilleux
- **Method:** internal prior-art sweep (Explore agent over the IX workspace) + external `/deep-research` fan-out (106 agents, 24 sources, 25 claims adversarially verified, 17 confirmed). Research-quality caveats are in §8.

---

## 0. TL;DR / The decision

**Build, don't buy. And mostly *compose*, don't build.**

The system you described — extract and maintain over time a unified graph of *assumptions* (both code-level dev assumptions and research-domain claims), each carrying a hexavalent truth value + certainty, promoted/demoted as evidence accumulates — is **~70% already present in IX as unconnected primitives**. The missing 30% is one thin crate (`ix-assumption-graph`), a time-travel query, and a longitudinal loop where `/deep-research` is the evidence arm.

**graphify and understand-anything are orthogonal** — they are snapshot GraphRAG *comprehension* tools (Tree-sitter + LLM extraction, token compression), with no truth values and no temporal belief revision. They solve "make my codebase cheaply queryable," not "track what we believe and why, over time." **You do not need either to build this.** (Optional, narrow reuse noted in §3.)

The external literature says: **don't invent the epistemics — assemble four settled formalisms** (ABA, possibilistic ATMS, subjective logic, belief-time bitemporal modeling) and map each onto a primitive you already have.

---

## 1. Problem frame (who is in pain, what changes)

**Today:** IX/GA development generates assumptions constantly — `@ai:invariant`, `@ai:hypothesis`, `@ai:assumption` annotations in code; design decisions in `docs/plans/`; research conclusions in `state/knowledge/`. They are **write-once and decay silently.** Nothing tracks "we believed X with certainty 0.7 on 2026-04-01; a test on 2026-05-01 refuted it; what downstream claims depended on X?" Research output is a *snapshot report* — it cannot tell you later that one of its load-bearing claims has since been contradicted.

**The pain:** "green-but-dead" beliefs. An assumption that was once true (or once *asserted*) is treated as still-true forever, with no mechanism to flip it to `D`/`F`/`C` when evidence arrives, and no record of *why* it flipped. This is precisely the failure class your memory keeps flagging (`feedback_green_but_dead`, `feedback_completion_bias`).

**What changes for whom:**
- *Developers:* `@ai:` assumptions become living nodes — a test run can promote `U→P→T` or refute `P→D→F`, and contradictions (`C`) escalate automatically per Demerzel thresholds.
- *Research (`/deep-research`, `seldon-research`):* a report's claims persist as graph nodes; a later run re-verifies them, and the graph shows which past conclusions have weakened. "A semantic story that runs over a long period" = the belief-time axis of this graph.
- *Governance:* assumptions crossing certainty thresholds become candidates for promotion to invariants / policy, with a full derivation trail.

---

## 2. Build-vs-buy verdict on the two products

> **Provenance flag:** the `/deep-research` workflow produced **no verified claims** on this angle (verifier sub-agents died — see §8). This verdict rests on direct page fetches I performed (graphify.net 403'd; GitHub + WebSearch + understand-anything.com succeeded) plus their READMEs. Confidence: medium-high on *what they are*, high on *that they're orthogonal*.

| | **graphify** ([safishamsi/graphify](https://github.com/safishamsi/graphify)) | **understand-anything** ([Lum1104/Understand-Anything](https://github.com/Lum1104/Understand-Anything)) | **What we need** |
|---|---|---|---|
| Job | Codebase+docs → queryable knowledge graph for AI coding assistants | Code → business-context knowledge graph (anti-"hairball") | Assumption graph with epistemic state |
| Extraction | Tree-sitter AST + LLM semantic + vision for diagrams | Tree-sitter + LLM, 26+ file types | N/A — we ingest `@ai:` annotations + research claims |
| Time model | **Snapshot** | **Snapshot** | **Belief-time + valid-time (bitemporal+)** |
| Truth model | None (concept/symbol nodes) | None | **Hexavalent T/P/U/D/F/C + certainty** |
| Verification | None | None | **Adversarial → promote/demote** |
| Headline | ~71.5× token compression vs raw-file reading | "god nodes", "surprising connections", "why" extraction | belief revision + derivation provenance |
| License | Open source (skill) | MIT | (internal crate) |

**Verdict: orthogonal comprehension tools — not building blocks for the temporal assumption graph. Do not adopt for this purpose.**

**The one defensible narrow reuse** (optional, low priority): graphify/understand-anything's *Tree-sitter+LLM extraction layer* is a mature pattern for turning prose/code into candidate nodes. But IX already has `ix-code::semantic` (call-graph/AST) and `ix-ai-annotations` (the `@ai:` extractor), so even this is **duplication, not gap-fill.** If anything, graphify is a useful *comparator* for the comprehension/RAG side of GA's chatbot — a separate concern from this spec.

---

## 3. Prior-art map: external formalism → IX primitive you already have

This is the core finding. Each row is a verified external formalism mapped to an existing Rust primitive. **The architecture is the union of these rows.**

| External formalism (verified) | What it gives | IX primitive it maps onto | Confidence |
|---|---|---|---|
| **Assumption-Based Argumentation (ABA)** — framework `⟨L, R, A, ̄ ⟩`; attacks via the *contrary* relation, always directed at assumptions; `dispute derivations` are an operational proponent/opponent proof procedure for claim acceptability | The *edge semantics* (attack/contradiction) + an **algorithmic template for the promote/demote decision** | Attack edges in `ix-pipeline::dag::Dag<N>`; dispute-derivation loop as the acceptability checker | **High (3-0)** |
| **ATMS** — node label = set of *environments* (consistent assumption sets that support it); inconsistent sets recorded as `NOGOOD`s and removed with all supersets; labels kept subset-minimal; updated **incrementally** | Support/contradiction **provenance over assumption sets** | `hari` derivation-provenance + `ix-fuzzy` contradiction synthesis; NOGOODs ≈ the `C` (Contradictory) synthesis path | **High (3-0)** |
| **Possibilistic α-ATMS** — each assumption carries a numeric weight (necessity lower bound); tracks an inconsistency degree; allows ranked solutions | **Graded** certainty on provenance (not binary) | Confidence floats already on `@ai:` annotations + Demerzel thresholds | **High (3-0)** |
| **Subjective Logic (Jøsang)** — binomial opinion `(b, d, u, a)`, `b+d+u=1`, projection `E = b + a·u`; `b=1`→TRUE, `d=1`→FALSE, `b+d<1`→uncertainty | **The formal bridge** between a discrete truth value and a continuous, *fusable* confidence | The certainty scalar on each node; see §5 for the hexavalent↔opinion bridge | **High (3-0)**; the *specific* 6-valued mapping is an interpretive bridge (medium) |
| **Subjective-logic cumulative fusion** — vector-add of evidence ≡ Dirichlet posterior update; commutative+associative (a join semigroup) | Evidence accumulation over time as a **CRDT-friendly merge** | `ix-fuzzy::observations::merge()` (G-Set CRDT + Belnap synthesis) | **High (3-0)** |
| **Belief-time bitemporal model** (Jiratanachit & Chittayasothorn 2015) — adds a **third axis, "belief time,"** distinct from transaction time; stores *multiple differing beliefs about the same fact over the same valid-time interval* | The temporal substrate for **genuine belief revision** (not snapshot storage) | The persistence layer (new) — a belief-time index over `state/` JSONL | **High (3-0)**; query mechanics unverified (paywalled) |
| **BiTRDF** (Mathematics 2025) — uniformly bitemporal RDF | Confirms uniform temporalization is sound | reference only — **theory-only, no truth values, no revision** | High (3-0) but **low reusability** |
| **OSTRICH** (JWS 2018) — RDF archive = intermittent full snapshots + independent delta chains; any version = 1 snapshot + 1 delta; **ordinal** versions, not timestamps | A concrete **versioned-store storage pattern** | Optional storage layout for the belief-time store | High (2-0); orthogonal to belief/valid time |

**Closest production comparator (angle 5, not verified but known):** **Graphiti** (getzep) is a *bi-temporal* agent-memory graph — it tracks valid-time + ingestion-time per edge and does *edge invalidation* when facts change. That is the nearest shipped thing to "beliefs over time." But it **invalidates** facts (present/absent), it does **not** carry a truth-value lattice or do adjudicated promotion/demotion. So it confirms the bi-temporal substrate is the right call, and confirms the *epistemic* layer (hexavalent + adversarial verify) is our differentiator.

---

## 4. Architecture — `ix-assumption-graph`

Three layers over existing primitives. The new code is the **wiring + the temporal index + the loop**, not the epistemics.

```
                 ┌─────────────────────────────────────────────────────┐
   INGEST        │  dev side: ix-ai-annotations  →  @ai:assumption /     │
   (existing)    │            invariant / hypothesis nodes (free)        │
                 │  research side: /deep-research verified claims        │
                 │            →  research-claim nodes (same schema)      │
                 └───────────────────────────┬─────────────────────────┘
                                              ▼
                 ┌─────────────────────────────────────────────────────┐
   GRAPH         │  ix-pipeline::dag::Dag<AssumptionNode>                │
   (existing)    │  edges: Supports / Contradicts(=ABA contrary) /       │
                 │         DependsOn / RefinedFrom                       │
                 └───────────────────────────┬─────────────────────────┘
                                              ▼
                 ┌─────────────────────────────────────────────────────┐
   BELIEF        │  ix-fuzzy::observations::merge  (CRDT + Belnap)        │
   SUBSTRATE     │  hari: derivation provenance + trust-weighted         │
   (existing)    │        consensus  (≈ ATMS environments/NOGOODs)       │
                 │  certainty fusion = subjective-logic cumulative       │
                 │        (GUARDED: independence check — see §6)         │
                 └───────────────────────────┬─────────────────────────┘
                                              ▼
                 ┌─────────────────────────────────────────────────────┐
   TEMPORAL      │  *** NEW *** belief-time index                        │
   STORE         │  belief_at(t) → Dag<AssumptionNode>                   │
   (gap #2)      │  diff(t1, t2) → what flipped & why                    │
                 └─────────────────────────────────────────────────────┘
```

### 4.1 Node schema (`AssumptionNode`)
```rust
struct AssumptionNode {
    id: Sha256,                 // stable identity — SAME scheme as @ai: annotations
    claim: String,
    kind: Kind,                 // DevInvariant | DevHypothesis | DevAssumption | ResearchClaim
    truth: Hexavalent,          // ix-types::Hexavalent — T/P/U/D/F/C
    opinion: Opinion,           // (b,d,u,a) subjective-logic — the certainty carrier
    evidence: Vec<EvidenceRef>, // src path / test id / source URL + verify vote
    contrary: Option<NodeId>,   // ABA contrary → drives Contradicts edges
    created: BeliefTime,
    updated: BeliefTime,
}
```
- `truth` and `opinion` are **redundant by design** and kept consistent via the §5 bridge. `truth` is for governance gates / human reading; `opinion` is for fusion math.
- `id` reuses the existing `SHA256(path:line:kind:claim)` so a dev assumption keeps identity across edits and across the temporal axis.

### 4.2 Edge types
- `Contradicts` ← **ABA contrary relation** (verified). The one we get formal semantics for free.
- `Supports`, `DependsOn`, `RefinedFrom` — **CAUTION:** support edges push us out of flat-ABA / Dung into **non-flat ABA / bipolar argumentation**, where "all Dung semantics apply" no longer holds (verified 3-0). See §6.

### 4.3 The longitudinal loop (gap #3 — where `/deep-research` plugs in)
```
hypothesis(U) ──gather evidence──▶ adversarial verify ──▶ fuse(opinion) ──▶ revise(truth)
     ▲                                                                          │
     └──────────────── re-check on schedule / on contradiction ◀───────────────┘
```
- "gather evidence" for research claims = a `/deep-research` run; for dev assumptions = a test run / sentrux scan / `ix-ai-annotations` reconcile.
- "revise" follows the verified hexavalent transition rules (`U→P`, `P→T`, `D→F`; never `T→F` directly — must pass through `C` or `U`).

---

## 5. The hexavalent ↔ subjective-logic bridge (the formal glue)

The research nailed the binary anchors (`b=1`→T, `d=1`→F, `b+d<1`→uncertainty, verified). Extending to all six values is a **design inference** (flagged medium-confidence) — propose this projection, to be ratified against `governance/demerzel/logic/hexavalent-logic.md`:

| Hexavalent | Subjective-logic opinion region | Projected `E=b+a·u` |
|---|---|---|
| **T** True | `b` high, `d≈0`, `u` low | → 1 |
| **P** Probable | `b > d`, `u` moderate | > 0.5 |
| **U** Unknown | `u` high (`b+d→0`) | ≈ base rate `a` |
| **D** Doubtful | `d > b`, `u` moderate | < 0.5 |
| **F** False | `d` high, `b≈0`, `u` low | → 0 |
| **C** Contradictory | **outside the simplex** — `b` *and* `d` both high from *conflicting* fused evidence | undefined → escalate |

**Key insight:** subjective logic alone *cannot* represent `C` (its `b+d+u=1` simplex has no "both" corner — that's Belnap's contribution). So the model is **Belnap FOUR for the lattice structure (gives us `C`) + subjective logic for the certainty carrier (gives us fusable confidence)**. This matches what IX already does: `ix-fuzzy` uses a *Belnap-extended* weight table for contradiction synthesis. The bridge is: fuse opinions cumulatively; if fusion yields jointly-high `b` and `d` from sources that *disagree*, that's the `C` synthesis path → node truth = `C`, escalate.

---

## 6. Risks & cautions (load-bearing — do not skip)

1. **Cumulative-fusion over-counting (verified 3-0, highest-priority trap).** Cumulative fusion assumes **independent** evidence. Summing **correlated** sources — e.g. the same agent re-run, or a multi-LLM panel sharing training bias — **overstates certainty.** This is the *same shared-bias amplification* that makes naive judge-panels unreliable. **Mitigation:** tag every `EvidenceRef` with a source-independence class; use cumulative fusion only across independent classes, averaging fusion within a class. **This directly affects the `feat/adversarial-llm-panel` branch** — the panel must not be wired as N independent votes if the models share lineage.
2. **Support edges break flat-ABA guarantees (verified 3-0).** Adding `Supports` lands us in non-flat/bipolar argumentation. **Mitigation:** keep the *acceptability decision* (dispute derivation) on the attack/`Contradicts` sublattice only; treat `Supports`/`DependsOn` as provenance/propagation hints, not as inputs to the formal semantics — at least in v1.
3. **The LLM-judge-panel question — RESOLVED 2026-05-31 (was unresolved in the first run).** A focused, source-verified re-run settled it: LLM judges have ~96% TPR but **<25% TNR** and majority voting does not fix this (arXiv:2510.11822); panel bias-reduction holds *only* for disjoint model families (arXiv:2404.18796); same-family debate amplifies shared errors (arXiv:2503.16814, 2509.05396). **Verdict:** treat the panel as a **fail-closed, asymmetric gate** — consensus required to promote toward `T`, any credible dissent forces `U`/`D`/`C`; map disagreement to hexavalent values rather than collapsing to a vote; mandate + measure judge independence; guardrail metric is **TNR**, not agreement rate. Full discipline encoded in the contract §7.2.
4. **Green-but-dead verifiers (observed this very run).** ~half the verify sub-agents died "without calling StructuredOutput." Any automation built on the deep-research panel must **fail closed** when a verifier dies, never default to "survived."
5. **Temporal-RDF prior art is thin.** BiTRDF is theory-only; the belief-time paper is paywalled. Don't over-index on RDF tooling — the belief-time axis is better served by an append-only JSONL event log in `state/` (consistent with `ix-autoresearch` / hari streaming) than by a triplestore.

---

## 7. Phased plan (composition-first)

| Phase | Deliverable | Reuses | New code | One/two-way door |
|---|---|---|---|---|
| **0** | `AssumptionNode` schema + JSON contract (`docs/contracts/`) | Hexavalent, `@ai:` id scheme | schema only | schema hash = **one-way** → sign-off |
| **1** | `ix-assumption-graph` crate: build `Dag<AssumptionNode>` from `@ai:` annotations; `Contradicts` from ABA contrary | `ix-ai-annotations`, `ix-pipeline::dag` | wiring | two-way |
| **2** | Belief substrate: opinion fusion + Belnap `C` synthesis + the §5 bridge | `ix-fuzzy`, `hari` | bridge + independence tags | bridge math = revisit-on-evidence |
| **3** | Belief-time index: `belief_at(t)`, `diff(t1,t2)` over append-only JSONL | `ix-autoresearch` event pattern | temporal index | storage layout = soft one-way |
| **4** | Longitudinal loop: `/deep-research` → research-claim nodes; scheduled re-verify; Demerzel promotion gate | `/deep-research`, Demerzel thresholds | loop + scheduler | two-way |
| **5** | (optional) MCP exposure: `ix_assumption_query`, `ix_belief_at` | `ix-agent` | tool handlers | two-way |

**Smallest useful slice (proof-of-life):** Phase 0+1 only — turn the `@ai:` annotations already in the repo into a queryable graph with `Contradicts` edges. That alone closes the "assumptions decay silently" pain for dev work, and validates the schema before any temporal/fusion complexity.

---

## 8. Research-quality caveats (full honesty)

- **Verified well (high confidence):** ABA, ATMS, possibilistic α-ATMS, subjective logic + cumulative fusion, belief-time bitemporal, BiTRDF, OSTRICH. These are settled, slow-moving fields on primary literature.
- **UNRESOLVED — angle 4 (LLM-judge panels):** every claim killed; one source (`arXiv 2603.06612`) is a **fabricated/future-dated ID**. The PoLL claims (`arXiv 2404.18796`) scored 0-0 (verifiers abstained/died, *not* refuted on merits). **Re-research independently before relying on the adversarial panel as a correctness mechanism.**
- **UNANSWERED — angle 6 (graphify/understand-anything):** workflow produced no verified claims; the §2 verdict is from my own fetches, not the panel.
- **Thin — angle 3 query mechanics:** belief-time paper paywalled (representation verified, querying not); several provenance claims (PROV-STAR, live SPARQL-star reconstruction, OCDM) refuted/unverified.
- **Interpretive bridges (not source quotes):** "subjective logic → hexavalent lattice," "cumulative fusion → CRDT merge," "ATMS → derivation provenance" are sound design inferences the verifiers endorsed, **not** claims the papers make about this architecture.
- **Tooling:** ~half the verify sub-agents died (`StructuredOutput` failure). The 17 confirmed survived genuine 3-0 / 2-0 votes; the 8 "killed" are mostly *abstentions from dead agents*, not substantive refutations — treat "killed" as "unverified," not "false."

---

## 9. Open questions (carry forward)

1. ~~When do adversarial multi-LLM panels improve correctness vs amplify shared bias?~~ **RESOLVED 2026-05-31** — see §6 caution 3 and contract §7.2. Verdict: fail-closed asymmetric gate, independence mandatory, TNR guardrail. Directly informs the `feat/adversarial-llm-panel` design.
2. **Exact 6-valued ↔ `(b,d,u,a)` projection** — ratify §5 against `hexavalent-logic.md`; how do AGM revision postulates apply to a hexavalent lattice?
3. **Source-dependence estimation** for fusion — how to classify independence of observations (multiple LLMs, repeated agent runs) in practice, to choose cumulative vs averaging fusion.
4. **Does the belief-time axis belong in `hari` or in `ix-assumption-graph`?** hari already has per-round observation tracking + replay — possibly the temporal store is a hari graduation rather than new IX code.

---

## 10. Doors & revisit triggers

- **One-way doors:** `AssumptionNode` schema hash (Phase 0); belief-time storage layout (Phase 3). Require explicit sign-off per CLAUDE.md.
- **Two-way doors:** crate wiring, MCP exposure, the loop scheduler.
- **Revisit triggers:** (a) re-check BiTRDF when its authors' 2026 implementation companion lands; (b) re-open angle 4 before the adversarial panel ships; (c) re-evaluate Graphiti as the substrate if the belief-time index proves heavy to maintain.
