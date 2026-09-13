# Gaia Artifact WorkGraph and Staleness Design v0.1

**Status:** post-S1 design input; no implementation authority  
**Date:** 2026-08-13  
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**  
**S1 relationship:** the completed S1 R1 review covers the earlier immutable four-file bundle only and returned `REQUEST_CHANGES`. This document was not in that subject and receives no inherited approval.

## 1. Core proposition

Gaia projects progress primarily through **Artifact Revisions and verified transitions between them**.

Code is an Artifact, but not a privileged exception. Specifications, Task Capsules, source snapshots, contracts, tests, builds, reviews, security attestations, acceptance decisions, knowledge packages, model checkpoints, and deployment receipts are also Artifacts. Conversations and agents may propose or produce them, but neither conversation length nor agent activity is project state.

The durable project state is therefore an **Artifact WorkGraph**: a directed graph of immutable Artifact Revisions, typed Derivation Edges, Build Recipes, evidence, and acceptance gates.

This is not a folder tree:

- one build may consume many source and configuration revisions;
- one test result may validate code, a spec requirement, and an environment;
- one review may cover several artifacts;
- one specification may affect several Task Capsules;
- one artifact may have several consumers.

The default structure is a DAG. A detected cycle is `UNKNOWN/BLOCKED` until the cycle is explicitly modelled as a bounded fixed-point computation with a convergence criterion. Gaia MUST NOT silently recurse forever.

The canonical provenance form SHOULD be bipartite:

```text
Artifact Revision --consumed-by--> Transition Receipt --produced--> Artifact Revision
```

A Transition Receipt may consume and produce several revisions. Keeping it as a first-class node preserves recipe version, fence, budget, toolchain, producer, gates, and evidence. A convenience artifact-to-artifact dependency DAG is a derived projection, not the canonical provenance record.

### 1.1 Self-similar composition

The organization is operationally **fractal** or self-similar, not mathematically fractal. The same contract repeats at each useful scale:

```text
ecosystem
  -> repository or product
     -> Mission Room
        -> Task Capsule
           -> implementation slice
              -> source/test/build/review evidence
```

Each level MAY be represented as a Composite Artifact whose implementation is a child Artifact WorkGraph. Its parent sees only a small interface:

- exact input Artifact Revisions;
- promised output Artifact Revisions;
- invariants and failure modes;
- Quality Assessments and acceptance policy;
- root digest and provenance;
- budget/performance envelope.

The child graph remains expandable for diagnosis and audit. This gives the Artifact WorkGraph the same deep-module property as good code: substantial internal behavior behind a small interface, with change localized below the seam.

Gaia MUST NOT create one node per file by reflex. A node is justified when it has an independent consumer, lifecycle, rebuild rule, review subject, or acceptance consequence. Otherwise files remain hidden inside a Composite Artifact revision. This avoids replacing code-level shallow modules with graph-level shallow nodes.

## 2. Two interface designs

### Design A — read-through automatic refresh

Illustrative interface:

```text
readFresh(artifactId) -> bytes
```

On a stale read, the module discovers dependencies, launches rebuilds, waits, and returns fresh bytes.

Advantages:

- trivial happy-path caller;
- resembles a conventional cache;
- hides orchestration.

Decisive weaknesses:

- a read unexpectedly gains mutation, routing, spend, and waiting behavior;
- parallel readers can cause rebuild stampedes;
- an unbounded downstream cascade can exceed budget or time;
- failure modes are hidden behind one apparently simple call;
- authority and evidence become ambient;
- a stale advisory read could accidentally trigger product mutation;
- testing requires observing hidden orchestration rather than a returned decision.

**Verdict: REJECT.** The interface is superficially small but not deep: callers must understand large invisible effects, costs, locks, and failure behavior.

### Design B — freshness barrier plus explicit refresh plan

Illustrative interface:

```text
assess(artifactRef, freshnessRequirement) -> FreshnessAssessment
planRefresh(targets, budget)              -> RefreshPlan | PlanningRefusal
publish(receipt, coordinatorFence)        -> PublicationDecision
```

Advantages:

- reads remain observational;
- impact calculation is deterministic and testable through the same interface callers use;
- the refresh cascade is explicit, bounded, deduplicated, and separately authorized;
- publication is atomic and fenced;
- planners, humans, and automated consumers can choose different Freshness Requirements without changing lineage truth;
- failures preserve a visible blocked frontier rather than pretending the graph is fresh.

Cost:

- one extra decision step before rebuild;
- consumers must name their freshness policy;
- a coordinator or human must authorize expensive refresh work.

**Verdict: ADOPT FOR DESIGN.** This interface provides leverage and locality: digest verification, edge semantics, impact closure, topological ordering, cycle detection, deduplication, compatibility receipts, and failure explanations remain behind three operations.

## 3. Deep-module placement

### `ArtifactWorkGraph` module

The `ArtifactWorkGraph` SHALL be a deep module. Its interface is the three operations above. Its implementation hides:

- logical Artifact identity versus immutable Artifact Revision identity;
- exact digest and schema verification;
- current accepted-head resolution;
- Derivation Edge indexing in both directions;
- compatibility rules and selectors;
- Freshness Assessment computation;
- transitive Impact Closure;
- DAG topological sorting and cycle/SCC detection;
- rebuild-key deduplication;
- partial-plan and blocked-frontier explanations;
- deterministic provenance for every assessment.

The interface SHALL return results and explanations. It SHALL NOT execute shell commands, start agents, modify product files, spend money, or update an accepted head.

### `RefreshOrchestrator` module

The `RefreshOrchestrator` consumes an accepted Refresh Plan. It owns bounded scheduling, claims, leases, retries, and evidence collection, but not freshness truth. It communicates through Gaia's existing `send`, `handoff`, `inbox`, `ack`, and `heartbeat` behavior; registration remains the existing sixth verb. No `refresh` or `invalidate` bus verb is added.

### `ArtifactPublisher` module

The `ArtifactPublisher` verifies a completed refresh receipt and uses a Coordinator Fence to atomically publish a new accepted head. The writer cannot publish its own unreviewed result. Failed compare-and-set, stale epoch, changed pre-image, incomplete evidence, or insufficient authority rejects publication.

### Recursive module interface

`ArtifactWorkGraph` uses the same interface for a leaf Artifact and a Composite Artifact. `assess` may recursively inspect a child graph, but it returns one parent-level Freshness Assessment bound to the child's root digest. `planRefresh` may expand only the affected child subgraph and then collapse it back into the parent plan. `publish` updates a composite head only after every required internal gate reaches an acceptable fixed point.

This recursive use is where “fractal” pays off: nesting changes scale, not semantics.

### Persistence seam

Persistence is an internal seam, not part of the public ArtifactWorkGraph interface.

- first production adapter: immutable local manifests plus an append-only ledger;
- test adapter: in-memory manifests with deterministic ordering and injected clocks;
- later adapter only if measured: PostgreSQL transactions and constraints.

The graph index is a rebuildable projection. Immutable manifests and accepted-head receipts are the source facts. FalkorDB or another graph engine MAY later accelerate graph queries but MUST remain a reconstructible projection, never an authority store.

## 4. Artifact record

Each Artifact Revision manifest SHALL contain at least:

```text
artifact_id              stable logical identity
revision_digest          exact content digest
artifact_kind            spec | task | source | test | build | review | decision | ...
media_type
schema_id + schema_version
producer_profile + producer_incarnation
created_at               observation only, never primary freshness truth
recipe_id + recipe_digest
toolchain_digest
subject_digest           when validating or reviewing another artifact
inputs[]                 exact Derivation Edges
evidence_refs[]
authority_receipt
supersedes_revision      optional
```

Every input edge SHALL name:

```text
upstream_artifact_id
upstream_revision_digest
relationship
selector                 optional affected subset
invalidation_policy
compatibility_rule       optional, versioned and testable
```

Filesystem modification time MAY detect suspicious drift but cannot prove freshness. Content digest, accepted heads, recipe/toolchain digests, and edge policy are authoritative inputs.

## 5. Separate state dimensions

Gaia MUST NOT collapse distinct state machines into one `status` field.

### Freshness state

- `FRESH`: every required edge is exact or has a valid Compatibility Receipt; artifact and recipe digests match.
- `SUSPECT`: an upstream head changed or a soft edge fired, but compatibility has not been decided.
- `STALE`: a hard edge is incompatible, artifact bytes drifted, or recipe/schema/toolchain invalidation applies.
- `UNKNOWN`: metadata, input, rule, or accepted head is missing or cannot be verified.

### Lifecycle state

- `CURRENT`: accepted head for its logical Artifact.
- `SUPERSEDED`: preserved evidence but no longer the accepted head.
- `QUARANTINED`: excluded from authoritative consumption.

### Refresh execution state

- `IDLE`
- `PLANNED`
- `RUNNING`
- `BLOCKED`
- `COMPLETED`
- `FAILED`

An Artifact Revision can therefore be `STALE + CURRENT + BLOCKED`, which is meaningful: it is still the published head, cannot satisfy a fresh consumer, and cannot currently be rebuilt. It must not be mislabeled `FAILED` or silently served as fresh.

### Quality state is not freshness

Quality is a vector of independent, exact-subject assessments rather than one score:

```text
Standards: APPROVE | REQUEST_CHANGES | UNKNOWN
Spec:      APPROVE | REQUEST_CHANGES | UNKNOWN
Tests:     PASS | FAIL | UNKNOWN
Security:  APPROVE | REQUEST_CHANGES | UNKNOWN
Evidence:  COMPLETE | INCOMPLETE | UNKNOWN
Clean:     TRUE | FALSE | UNKNOWN
```

An artifact can be fresh and poor-quality, or stale and historically well-reviewed. Acceptance policies combine named quality axes without erasing them. Model confidence, popularity, consensus, and scalar “quality scores” cannot replace a required axis.

## 5.1 Programmable transitions

The DAG backbone is executable through versioned **Transition Definitions**. A definition is itself an immutable Artifact Revision and declares:

- accepted input kinds, schemas, selectors, and cardinalities;
- output kinds and schemas;
- preconditions and explicit refusal modes;
- Build Recipe and toolchain digests;
- whether determinism is required and how it is tested;
- authority, Coordinator Fence, write scope, and idempotency key;
- time/token/money/resource budgets and cancellation;
- required Quality Assessments and independent reviewers;
- invalidation policy when the definition, toolchain, or input changes;
- rollback/quarantine behavior;
- evidence and Transition Receipt schema.

One execution creates a Transition Receipt bound to the exact Transition Definition revision and exact input vector. The receipt never mutates its definition or inputs.

Transitions are **reprogrammable by versioning**, not editable in history:

1. publish a new Transition Definition revision;
2. compare it to the currently accepted definition using a declared compatibility rule;
3. mark prior outputs `SUSPECT` or `STALE` along applicable recipe edges;
4. compute the affected closure;
5. obtain separate authority for refresh;
6. preserve every old definition, receipt, and output for replay.

This makes changes to prompts, skills, generators, evaluators, CI, schemas, policies, or agent harnesses first-class invalidation events. A “better prompt” cannot silently reinterpret old evidence.

Transition execution lifecycle remains distinct from output acceptance:

```text
DECLARED -> AUTHORIZED -> RUNNING -> RECEIPT_WRITTEN
                         |          |
                         v          v
                       FAILED     VERIFIED -> ACCEPTED | REJECTED
```

`RECEIPT_WRITTEN` or process exit zero does not establish `VERIFIED` or `ACCEPTED`.

## 6. Edge-specific invalidation

Invalidation MUST be edge-specific; “upstream changed, rebuild everything” is too coarse.

| Policy | Effect of upstream head change |
|---|---|
| `EXACT` | downstream becomes `STALE` |
| `COMPATIBLE` | downstream becomes `SUSPECT` until a versioned compatibility rule or reviewed receipt decides |
| `SELECTIVE` | only changes intersecting the declared selector propagate |
| `ADVISORY` | consumer is warned; authoritative freshness does not change |
| `REFERENCE_ONLY` | no invalidation; citation/provenance link remains visible |

A Compatibility Receipt is evidence, not an override flag. It binds exact old/new upstream revisions, edge, rule version, tests/review, issuer, and expiry/supersession conditions.

## 7. Consumer behavior

Every consumer declares one Freshness Requirement:

- `REQUIRE_FRESH`: return the fresh revision or a typed `REFRESH_REQUIRED/BLOCKED` decision; do not return authoritative stale content.
- `ALLOW_SUSPECT_WITH_WARNING`: useful for exploratory/advisory work; provenance and reasons accompany the content.
- `ALLOW_STALE_WITH_WARNING`: historical/debug use only; cannot satisfy a gate.
- `REQUIRE_EXACT(revisionDigest)`: replay and review against one immutable revision regardless of newer heads.

A stale read MAY cause the coordinator to receive a refresh proposal, but the read itself never executes it. `REQUEST_REFRESH_AND_WAIT` is an explicit Task Capsule policy with deadline, budget, authority, and cancellation; it is not the default.

## 8. Refresh cascade

Given changed upstream revisions, Gaia SHALL:

1. verify their accepted heads and manifests;
2. traverse reverse Derivation Edges;
3. apply selectors and invalidation policies;
4. compute the minimal Impact Closure;
5. detect cycles and unresolved compatibility;
6. partition affected nodes into `STALE`, `SUSPECT`, and `UNKNOWN`;
7. topologically order only refreshable `STALE` nodes;
8. estimate work, model/provider use, elapsed-time band, and maximum spend;
9. return a Refresh Plan and blocked frontier;
10. obtain explicit authority before execution;
11. deduplicate each rebuild using `(artifact_id, target_input_vector, recipe_digest)`;
12. claim one writer per mutable target with a lease and Coordinator Fence;
13. execute bounded Task Capsules;
14. verify exact outputs, tests, reviews, evidence, and cleanliness;
15. atomically publish accepted heads;
16. recompute downstream freshness until a fixed point or declared budget/stop condition.

Partial refresh is honest. If the budget expires, refreshed nodes may publish only when independently acceptable; the remaining frontier stays visibly stale or blocked.

## 9. Example

```text
Spec S@1
  -> Task Capsule T@1
     -> Source Snapshot C@1
        -> Test Evidence E@1
        -> Build B@1
           -> Review R@1
              -> Acceptance A@1
```

When `S@2` becomes the accepted spec head:

- `T@1` becomes stale on an `EXACT implements` edge;
- `C@1` becomes suspect or stale according to the scoped requirement selector;
- `E@1`, `R@1`, and `A@1` cannot validate new code/spec heads because their exact subjects differ;
- historical `S@1..A@1` remains immutable and replayable;
- a consumer requiring `S@2` receives a Refresh Plan, not an implicit rebuild;
- one reviewed Compatibility Receipt may preserve an unaffected source subtree, but cannot fabricate fresh test or review evidence for changed subjects.

This captures the central rule: **project progress is a monotone accumulation of accepted Artifact Revisions and fresh evidence, not a monotone accumulation of files.**

### 9.1 Composite example

```text
Product P@7 [root digest]
  contains Mission M@4 [root digest]
    contains Task T@12 [root digest]
      contains Source C@9, Tests E@5, Review R@2
```

A consumer of `Product P@7` need not learn every internal node. It depends on the Composite Artifact interface and root digest. If `Source C@10` changes inside `T`, only the affected Task subgraph expands; after its gates pass, new roots `T@13 -> M@5 -> P@8` are published. Unaffected sibling Mission Rooms remain fresh.

Root digests SHOULD be canonical Merkle-style commitments over the Composite Artifact interface, accepted child heads, Transition Definition revisions, and required receipts. They summarize lineage; they do not replace readable manifests or evidence.

## 9.2 Maturity and graph compaction

Gaia MAY contract a mature accepted subgraph into a **Graph Capsule** to reduce working-set size, query cost, prompt/context load, and operational noise.

Compaction is a representational transition:

```text
detailed immutable subgraph
        |
        | Compaction Receipt
        v
Graph Capsule macro-node in operational projections
```

The canonical ledger and original Artifact Revisions remain intact. Compaction MUST NOT rewrite history, merge identities, erase copyright/provenance, or turn lossy prose into authoritative evidence.

### Maturity Gate

A subgraph is compactable only when all required conditions hold:

1. every included revision and transition is immutable and content-addressed;
2. the selected root is accepted under named quality gates;
3. no included current node is `STALE`, `SUSPECT`, `UNKNOWN`, `RUNNING`, or `BLOCKED`;
4. no live Claim, Lease, or pending mutation crosses the proposed boundary;
5. every incoming and outgoing cross-boundary edge maps to a declared capsule input/output;
6. the boundary interface, schemas, invariants, error modes, and invalidation policies are explicit;
7. an expansion manifest enumerates every contracted node, edge, digest, recipe, receipt, and gate;
8. a canonical root digest is independently reproduced;
9. boundary reachability and required gate semantics are equivalent before and after contraction;
10. retention, licensing, security, and audit rules permit cold storage of the expanded form;
11. independent review approves the exact Compaction Receipt;
12. rollback/expansion is tested.

Age, inactivity, file count, model confidence, or “looks stable” cannot satisfy the Maturity Gate.

### Compaction Receipt

The receipt SHALL bind:

- exact input Graph Snapshot and selected node/edge set;
- capsule Artifact ID and revision/root digest;
- boundary inputs, outputs, ports, and external adjacency;
- compaction algorithm and version;
- expansion-manifest digest and storage location;
- before/after boundary-reachability witnesses;
- Quality Assessments and acceptance decision;
- independent reviewer;
- retention and re-expansion policy;
- cost and truncation facts.

### Lossless versus lossy

- **Lossless structural compaction** MAY participate in authoritative projections when the expansion manifest and equivalence tests pass.
- **Lossy semantic summarization** MAY create an advisory Artifact only. It cannot replace source nodes for provenance, legal, security, exact replay, or acceptance.

### Re-expansion and invalidation

A Graph Capsule expands when:

- a consumer requests detailed audit or diagnosis;
- a boundary input, Transition Definition, schema, or compatibility policy changes;
- a freshness assessment needs internal selectors unavailable at capsule resolution;
- a graph algorithm declares that contracted topology may affect its answer;
- the Compaction Receipt or expansion manifest fails verification.

Expansion is deterministic reconstruction of a projection, not restoration from memory. If the expansion manifest is missing or corrupt, capsule freshness becomes `UNKNOWN` and authoritative consumers fail closed.

### Analysis resolution

Every Analysis Recipe SHALL declare one resolution:

- `COLLAPSED`: analyze capsule macro-nodes only;
- `EXPANDED`: analyze all detailed nodes;
- `HYBRID`: expand only capsules intersecting the query or uncertainty frontier.

Clique, community, centrality, cut-set, and motif results depend on resolution. An Analysis Artifact MUST record which capsules were expanded. “No clique” or “no bottleneck” is invalid when relevant topology remained hidden by an undeclared collapsed view.

## 10. Negative controls

At minimum, tests SHALL prove:

1. a changed exact dependency invalidates every applicable descendant and no unrelated node;
2. a selector prevents false-positive invalidation outside its scope;
3. an advisory edge never blocks a gate and never grants authority;
4. a reference-only edge does not rebuild anything;
5. a missing manifest, rule, or head yields `UNKNOWN`, never `FRESH`;
6. byte drift yields `STALE` even when timestamps are unchanged;
7. timestamp drift alone cannot establish staleness or freshness;
8. two readers produce one deduplicated refresh intent, not a stampede;
9. a stale coordinator cannot publish a refreshed head;
10. changed inputs during rebuild reject publication;
11. a cycle fails closed with an explainable blocked SCC;
12. budget exhaustion preserves a visible partial/blocked frontier;
13. a stale review or acceptance receipt cannot validate a new subject digest;
14. a compatibility receipt for a different edge/revision is rejected;
15. a graph-index loss can be rebuilt from immutable manifests and accepted-head receipts;
16. no refresh path adds a seventh bus verb;
17. an agent or model cannot mark its own artifact fresh by assertion;
18. `REQUIRE_EXACT` replay remains possible after the logical artifact is superseded.
19. changing a Transition Definition invalidates only outputs whose recipe edge is incompatible;
20. an old Transition Receipt cannot attest an output built under a new definition;
21. a Composite Artifact root changes for an affected accepted child and remains stable for internal non-interface noise;
22. expanding then collapsing a Composite Artifact preserves the same parent-level Freshness Assessment;
23. a fresh artifact with failing quality gates cannot be accepted;
24. a stale artifact with historical approvals remains replayable but cannot satisfy a fresh gate.
25. compaction preserves declared boundary reachability and external adjacency;
26. a hidden cross-boundary edge rejects compaction;
27. a lossy summary cannot satisfy authoritative provenance or acceptance;
28. deleting the operational graph followed by expansion manifests reconstructs the same collapsed projection;
29. a corrupt or missing expansion manifest makes the capsule `UNKNOWN`;
30. a boundary change re-expands or invalidates only the affected capsule closure;
31. graph analytics at collapsed, expanded, and hybrid resolution disclose materially different answers rather than conflating them.

## 11. Specialized-system roles

- **IX** MAY compute impact closure, topological order, SCCs, and deterministic graph comparisons over copied manifests. Its output is advisory.
- **TARS** MAY validate the manifest/edge grammar and detect ambiguous relationship terms. It does not decide freshness.
- **HARI** MAY preserve contradictory freshness claims during analysis. A deterministic ArtifactWorkGraph decision remains authoritative.
- **Agent Blackbox** MAY independently replay manifests, invalidation controls, fixed points, and publication receipts.
- **PostgreSQL** MAY later implement the persistence seam when concurrent multi-host publication is authorized and measured. It is not required for T0/M2.
- **FalkorDB** MAY become a derived query projection after a benchmark proves value; it never owns accepted heads or authority.

## 11.1 Graph-analysis kernel

Gaia needs strong graph analysis, but not every graph metric belongs in the correctness path.

### Tier 1 — exact correctness kernel

These operations MAY affect freshness planning and therefore MUST be deterministic, versioned, bounded, and independently replayable:

- forward/reverse reachability and minimal Impact Closure;
- topological ordering;
- strongly connected components and cycle witnesses;
- transitive reduction for explainable dependency views;
- dominators for “all valid paths depend on this artifact/gate” analysis;
- articulation points, bridges, and bounded cut sets in named projections;
- exact changed-selector intersection;
- fixed-point detection;
- graph-digest and projection-digest verification.

The correctness kernel returns explanations and witnesses, not just booleans.

### Tier 2 — advisory structural analytics

These operations discover architecture and scheduling risks but MUST NOT grant authority or freshness by themselves:

- maximal cliques and bounded clique enumeration;
- k-core/k-truss dense-subgraph analysis;
- community detection;
- degree, betweenness, eigenvector, and PageRank-like centralities;
- motif and repeated-transition-pattern discovery;
- co-change and co-invalidation clustering;
- critical-path and slack estimates under uncertain durations;
- anomaly and change-point analysis over graph evolution.

Maximal-clique enumeration can be exponential. Every run SHALL declare size thresholds, node/edge filters, maximum results, time/memory budget, truncation semantics, and `UNKNOWN/PARTIAL` handling. Gaia MUST NOT claim “no clique” after a truncated search.

Dense clusters have an engineering interpretation: Artifacts that repeatedly change, invalidate, and get reviewed together are candidates for one deeper Composite Artifact. This is a recommendation to inspect a seam, not automatic refactoring. Conversely:

- high fan-out suggests a high-leverage interface or one-way-door contract;
- a dominator suggests a critical gate or hidden single point of failure;
- an articulation point suggests fragile decomposition;
- a large stale closure suggests an overly broad interface or invalidation policy;
- a stable community with a small cut suggests a promising deep-module seam.

### Tier 3 — predictive analytics

Learned embeddings, JEPA-style prediction, or graph neural models are post-M3 research only. They must beat simple exact graph baselines out of sample and remain advisory.

### `GraphAnalysis` deep module

Use one small interface:

```text
analyze(graphSnapshot, analysisRecipe) -> AnalysisArtifact | AnalysisRefusal
```

The versioned Analysis Recipe declares projection, algorithm, parameters, budgets, exact/approximate semantics, expected units/nulls, and falsifiers. The Analysis Artifact binds the Graph Snapshot digest, recipe digest, implementation/toolchain digest, results, witnesses, truncation, provenance, and resource use.

Two adapters justify the seam:

1. a deterministic reference adapter for small fixtures and mutation controls;
2. an IX graph adapter for bounded production-scale analysis over copied Graph Snapshots.

FalkorDB, PostgreSQL recursive queries, or another engine becomes a third adapter only after a corpus benchmark demonstrates a need. Engine-specific query syntax never enters the ArtifactWorkGraph interface.

### Required graph mutations and falsifiers

The evaluation corpus SHALL include:

- one missing dependency edge that changes Impact Closure;
- one forged Transition Receipt;
- one cycle and an exact witness;
- one dense cluster with known maximal cliques;
- one articulation point and one dominator with known answers;
- one selector-scoped change that must not escape its subgraph;
- one truncated clique search that must return `PARTIAL`, not `COMPLETE`;
- one projection-index loss followed by byte-equivalent reconstruction;
- one stale graph snapshot rejected against a newer accepted ledger position;
- one approximate result attempting to authorize refresh, which must be refused.

## 11.2 Re-codable blue/green backbone

Gaia SHOULD eventually support blue/green replacement of the **Backbone Runtime**. Blue/green does not duplicate the canonical ledger, accepted heads, or authority. It replaces one adapter behind a stable deep interface.

### Two designs

#### Design A — dual active backbones

Blue and Green each own a writable graph/database and both process live commands during migration.

**Verdict: REJECT.** Reconciliation cannot prove a unique causal order after both sides emit effects. Dual write creates divergent accepted heads, duplicate intents, inconsistent leases, and unresolvable split-brain under partition.

#### Design B — single lineage, active/shadow runtimes

Blue is the sole active Backbone Release. Green consumes the same immutable ledger and Graph Snapshots in Shadow Evaluation, writes only color-scoped derived projections/evidence, and owns no mutation fence. After comparison and drain, one atomic epoch transition makes Green active and makes every Blue fence stale.

**Verdict: ADOPT FOR LATER DESIGN.** It preserves one authority lineage while allowing the entire runtime to be rewritten, replaced, or rolled back.

### Stable seam

The ArtifactWorkGraph interface remains stable across releases:

```text
assess(artifactRef, freshnessRequirement) -> FreshnessAssessment
planRefresh(targets, budget)              -> RefreshPlan | PlanningRefusal
publish(receipt, coordinatorFence)        -> PublicationDecision
```

Blue and Green are adapters satisfying this interface. Color is deployment metadata and MUST NOT leak into Task Capsules, Artifact identities, Transition Definitions, or consumer contracts.

A separate deep `BackboneReleaseManager` hides release mechanics behind:

```text
stage(backboneRelease, evaluationContract) -> ReleaseCandidate | Refusal
compare(activeRelease, candidateRelease)   -> EquivalenceReport
cutover(candidateRelease, coordinatorFence)-> CutoverReceipt | Refusal
```

### Shared facts and isolated projections

- canonical Artifact manifests, lineage events, accepted-head receipts, and epochs remain single and append-only;
- both releases read the same content-addressed Graph Snapshot and later the same ordered event prefix;
- Blue and Green materialize separate rebuildable projections;
- Green cannot call ArtifactPublisher, acquire a mutation lease, spend, route work, or use production credentials;
- no canonical event is dual-written merely to feed both colors;
- projection loss is repaired by replay from canonical facts.

### Release modes

Every candidate declares exactly one comparison mode:

- `EQUIVALENT`: no decision delta is expected; any unexplained delta blocks cutover;
- `SEMANTIC_CHANGE`: every intended decision delta is specified, independently reviewed, and covered by migration/negative controls;
- `PERFORMANCE_ONLY`: decisions and evidence must remain equivalent while latency/resource guardrails improve;
- `EMERGENCY_ROLLBACK`: restores an earlier implementation as a new epoch, subject to schema readability and minimum safety gates.

An approximate similarity score cannot establish equivalence. Exact decision artifacts, refusal codes, impact closures, witnesses, ordering, null/unknown semantics, and provenance are compared first; performance distributions are additional evidence.

### Cutover sequence

1. create an immutable Backbone Release Artifact;
2. verify code/config/schema/toolchain/dependency digests and licence provenance;
3. declare comparison mode, expected deltas, budgets, abort thresholds, and rollback target;
4. build Green projections from one canonical Graph Snapshot;
5. replay a pinned historical, held-out, mutation, and adversarial corpus;
6. mirror live canonical events read-only until Green reaches ledger position `L`;
7. independently verify Equivalence Report, projection digests, performance, cost, and zero Green effects;
8. mark Green `READY` without mutation authority;
9. stop Blue admission and drain or hand off in-flight Task Capsules using existing idempotency keys;
10. atomically increment the active epoch and compare-and-set `active_release = Green` at ledger position `L`;
11. executors reject every delayed Blue fence and command;
12. run a bounded Green soak with rollback triggers;
13. contract/archive Blue only after the rollback window and Maturity Gate pass.

At no instant may both colors own a valid publication fence.

### Rollback

Rollback is a forward transition to a **newer epoch** selecting a previously proven Backbone Release. Epochs are never decremented or reused. Old Blue projections may accelerate recovery only after their ledger position and digest are verified; otherwise they rebuild from canonical facts.

Schema evolution follows expand -> migrate/replay -> compare -> cutover -> contract. A candidate that makes the prior safe release unable to read canonical facts is a one-way door requiring explicit approval and a different recovery plan.

### Release state dimensions

Do not collapse readiness, authority, and health:

```text
Release lifecycle: STAGED | SHADOWING | READY | ACTIVE | DRAINING | RETIRED
Authority:         NONE | READ_ONLY | MUTATION_FENCED
Health:            HEALTHY | DEGRADED | FAILED | UNKNOWN
Comparison:        MATCH | EXPECTED_DELTA | DIVERGED | INCOMPLETE
```

For example, Green may be `READY + READ_ONLY + HEALTHY + MATCH`; it is still forbidden to publish until cutover.

### Blue/green negative controls

1. simultaneous valid Blue and Green mutation fences are impossible;
2. a Green extension/tool attempting an effect is rejected and recorded;
3. a delayed Blue publication after cutover fails on epoch;
4. duplicated mirrored input produces one observation and zero duplicate effects;
5. missing/reordered Green input prevents `READY`;
6. a projection digest mismatch blocks cutover;
7. an unexplained decision delta blocks `EQUIVALENT` mode;
8. undeclared semantic delta blocks `SEMANTIC_CHANGE` mode;
9. Green lag beyond threshold aborts rather than switching partially caught up;
10. in-flight Blue Task Capsules either finish before the fence switch or hand off exactly once;
11. rollback uses a newer epoch and stale Green commands then fail;
12. irreversible schema contraction before rollback-window close is rejected;
13. unknown model/provider/accounting or dependency provenance blocks staging;
14. shadow evidence cannot self-authorize cutover;
15. canonical ledger and accepted heads remain byte-identical during shadowing.

### Timing

Blue/green backbone replacement is not required for M2 or M3. It becomes justified only after Gaia has a stable ArtifactWorkGraph contract, deterministic replay corpus, CoordinatorFence enforcement, independent verification, and a measured need for zero/low-downtime runtime evolution. Until then, versioned offline replay is sufficient and materially safer.

## 11.3 Strong-model horizon

Gaia assumes worker models may become dramatically more capable. It MUST NOT encode today's context limits, model weaknesses, vendor names, team sizes, or preferred decomposition depth into the ArtifactWorkGraph contract.

The invariant is:

> Capability may expand dynamically; authority, provenance, freshness, and acceptance remain explicit.

Consequences:

- stronger models receive larger Composite Artifacts and longer vertical slices only after Competency Evidence and risk policy justify the larger blast radius;
- Mission Rooms may need fewer agents, while retaining independent evidence and review for consequential gates;
- roles are temporary responsibilities, not assumptions that one model can or cannot perform a category of work;
- model-generated Transition Definitions, Analysis Recipes, seams, compactions, tests, or evaluators remain quarantined candidate Artifacts until held-out evidence and independent promotion;
- deterministic graph/freshness/authority kernels remain outside probabilistic model judgement;
- context is assembled from the smallest fresh relevant Artifact subgraph, not by replaying an ever-growing transcript;
- model memory is a cache; accepted Artifact Lineage is organizational memory;
- changing a worker model or harness is a versioned Transition Definition/Backbone Release change with explicit invalidation and comparison;
- a more capable model does not inherit push, deploy, spend, credential, publication, acceptance, or fence authority;
- evaluation measures outcome quality, evidence integrity, cost, latency, and negative controls on exact Task Capsules rather than treating model identity as quality.

The desirable future is therefore not an ever-larger swarm. It is a smaller number of more capable, replaceable agents operating on deeper Artifact modules through stable seams, with the Artifact WorkGraph preserving continuity and control.

## 11.4 Wayfinder fog-of-war doctrine

Gaia MUST remain useful if AGI arrives within five to ten years, arrives much later, or never arrives in a recognizable form. Model-capability forecasts are scenarios, not trusted facts or scheduling inputs.

The Wayfinder posture is:

> Advance through the largest action whose authority, evidence, reversibility, and blast radius are justified by what is currently known; preserve an explicit frontier for everything else.

Operational uncertainty is represented, not erased. Every consequential assessment or plan SHALL distinguish:

- known facts bound to immutable Artifact Revisions and provenance;
- accepted assumptions with owners, expiry/revisit triggers, and falsifiers;
- unresolved hypotheses and competing interpretations;
- missing, stale, incompatible, partial, or contradictory evidence;
- safe actions available under every surviving interpretation;
- actions blocked pending observation, authority, budget, or independent review.

Absence of evidence never becomes negative evidence, and model confidence never upgrades `UNKNOWN` to `FRESH`, `APPROVED`, or `SAFE`. When facts conflict, Gaia preserves the contradiction and computes the affected uncertainty frontier instead of selecting the most fluent narrative.

### Operational uncertainty vocabulary

One `UNKNOWN` state is insufficient. Gaia uses typed uncertainty conditions whose remedies differ:

- `MISSING`: required evidence does not exist; produce or obtain it;
- `UNOBSERVED`: evidence may exist but has not been inspected; perform a bounded observation;
- `INACCESSIBLE`: evidence is known but unavailable under current authority or connectivity; wait or request authority;
- `STALE`: evidence binds an older subject or dependency vector; refresh or revalidate its Impact Closure;
- `PARTIAL`: evidence covers only a declared subset; preserve uncovered scope explicitly;
- `AMBIGUOUS`: multiple interpretations fit the same observation; sharpen the schema or run a discriminating probe;
- `CONTRADICTORY`: credible evidence supports incompatible claims; retain both and block affected gates;
- `DIVERGENT`: implementations or replicas compute different results from the same declared inputs; compare recipes, projections, epochs, and provenance;
- `VOLATILE`: the subject changes faster than an observation or refresh can stabilize it; shrink scope or establish a quiescent snapshot;
- `NOVEL`: the observation falls outside the validated distribution or known ontology; quarantine and seek broader review;
- `UNDERDETERMINED`: available evidence cannot distinguish candidate explanations; design a higher-information probe;
- `IRREDUCIBLE`: uncertainty remains after the allowed evidence budget; choose a robust action, defer, or obtain explicit risk acceptance;
- `DECEPTIVE_RISK`: incentives or adversarial behavior make otherwise plausible evidence untrustworthy; require independent provenance and adversarial controls;
- `TEMPORAL_RISK`: a conclusion may expire before action completes; bind validity horizon and recheck point;
- `MODEL_RISK`: the evaluator, predictor, or model is outside its proven competency envelope; fall back to deterministic or independently validated methods.

These conditions are composable, not mutually exclusive. An assessment records subject, scope, provenance, detected conditions, confidence only where calibrated, consequences, candidate probes, budget, and expiry. A single scalar uncertainty score MUST NOT erase this structure.

Planning under fog uses bounded probes and tracer bullets. A Probe is selected for expected information gain only after deterministic safety, cost, reversibility, and authority constraints pass. Its result is a new Artifact or Transition Receipt; conversation and hidden model state do not count as organizational evidence.

Stronger future models may propose broader plans, identify better seams, compress larger contexts, or execute larger Composite Artifacts. They still operate within content-addressed inputs and outputs, explicit assumptions, fenced authority, idempotent effects, independent acceptance, rollback or disclosed one-way-door approval, deterministic replay, negative controls, and visible stopping conditions.

This makes the backbone capability-elastic: intelligence can improve rapidly without requiring a rewrite of Gaia's trust model. The system may widen autonomy when evidence supports it, narrow autonomy after regressions, and blue/green a new runtime without changing canonical history.

The first grammar projection is `gaia-uncertainty-grammar-v0.1.ebnf`. It is a candidate Transition Definition input, not an installed TARS grammar and not a source of truth. TARS may parse, search, mutate, or rank candidate derivations, but Gaia's semantic checks, Artifact evidence, fences, and acceptance gates remain authoritative. Grammar weights are advisory and cannot promote facts, freshness, safety, quality, or authority.

### Reproducibility and controlled variation

The deterministic-to-indeterministic spectrum is not represented by temperature or one scalar. Every model-backed Transition declares a Reproducibility Envelope across independent axes:

- model identity: content-pinned, release-pinned, provider alias, or unknown;
- decoding: greedy, seeded stochastic, unseeded stochastic, provider-opaque;
- context: exact closed Artifact set, selector-bounded dynamic set, or live/unbounded;
- tools and environment: hermetic deterministic, snapshotted, variable, or unknown;
- scheduling: serial, deterministically ordered parallel, order-sensitive concurrent, or external;
- observations: immutable snapshot, validity-bounded observation, live world, human input, or unknown;
- expected replay: `BYTE_EXACT`, `DECISION_EQUIVALENT`, `STATISTICALLY_BOUNDED`, `NON_REPLAYABLE`, or `UNKNOWN`.

`DETERMINISTIC` is earned only when every load-bearing axis is closed and replay passes. Seeded decoding alone is insufficient. Conversely, variable generation is not automatically unsafe: Gaia may spend a declared Entropy Budget to generate diverse candidate Artifacts, then cross a Determinization Boundary where schemas, deterministic gates, deduplication, authority, evidence, and independent acceptance apply.

A Semantic Equivalence Class is consumer-specific and versioned. It cannot be inferred by another unconstrained LLM. For example, formatting variants may be equivalent for a parser gate but not for a byte-addressed signature. Statistical bounds require repeated held-out trials, units, sample size, distribution assumptions, null handling, and abort thresholds.

A Refinement Policy may tighten or widen any axis. Tightening can reduce exploration after maturity; widening can increase search when progress stalls. Either change creates a new Transition Definition revision, invalidates dependent evidence according to declared edges, and requires replay comparison. No in-place tuning is hidden from lineage.

Effects remain deterministic even when proposals are not: the ArtifactPublisher accepts only exact validated receipts with idempotency keys and the current CoordinatorFence. A probabilistic model never directly owns an effect boundary.

## 11.5 Semantic GPS

The Artifact WorkGraph is a lineage and control backbone, not by itself a usable navigation surface. Gaia derives a Semantic Map that joins five projections:

1. domain topology: concepts, invariants, bounded contexts, conflicts, and terminology;
2. engineering topology: deep modules, seams, contracts, owners, and blast radius;
3. work topology: Artifact Revisions, Transition Definitions, receipts, freshness, and maturity;
4. evidence topology: claims, provenance, falsifiers, contradictions, competency, and quality axes;
5. capability topology: available actors/tools, authority, cost, latency, and proven task envelopes.

The GPS interface is deliberately small:

```text
locate(graphSnapshot, mapRecipe, evidencePolicy) -> SemanticPosition | LocalizationRefusal
route(position, destination, routePolicy)         -> SemanticRoute | RoutingRefusal
relocalize(position, newEvidence)                 -> RelocalizationReceipt | Refusal
```

A Semantic Position is not one embedding vector. It is a multi-resolution coordinate with exact anchors and uncertain regions: accepted Artifact heads, active assumptions, unresolved conditions, gate vector, current Composite Artifact path, and map/projection digests. Embeddings, graph communities, language models, TARS derivations, IX analytics, or JEPA predictions may propose nearby concepts and routes; deterministic anchors establish location.

A Semantic Destination is expressed as target invariants and acceptance gates plus non-goals, forbidden effects, budgets, and tolerable residual uncertainty. It is versioned. Changing the destination invalidates routes but does not rewrite prior travel.

Route construction is constrained multi-objective search over Transition Definitions. Candidate routes preserve separate vectors for expected evidence gain, semantic progress, freshness impact, quality risk, authority risk, blast radius, reversibility, cost, latency, and entropy. Dominated routes may be pruned; incomparable Pareto candidates remain explicit. No weighted sum is canonical. A decision policy may select among them only by declaring priorities, hard constraints, and tie-breaking.

Localization is hierarchical and fractal. Gaia first locates the ecosystem/repository/Mission Room, then the Composite Artifact and deep-module seam, then expands only the uncertainty frontier needed for the next decision. Graph Capsules act like map tiles with expansion manifests. Semantic Landmarks provide stable anchors across compaction and backbone releases.

Re-routing is triggered by changed destination, stale inputs, failed gates, contradictions, new evidence, unavailable capability, exhausted budget, or map drift. Re-routing produces a Relocalization Receipt binding the old position/route, new Graph Snapshot and Semantic Map digests, trigger, preserved landmarks, changed assumptions, abandoned route prefix, and new frontier.

Navigation remains advisory until each Transition separately passes its authority and acceptance gates. The Semantic GPS can recommend a route; it cannot publish an Artifact, spend, acquire a lease, or redefine the destination.

## 11.6 Multi-axis geometry and dimensionality reduction

Gaia's state is not canonically one real-valued vector. It is a Product State Space whose axes may be Boolean, categorical, ordinal, interval-valued, unit-bearing numeric, temporal, set-valued, lattice-valued, probabilistic or credal, textual, graph-structured, or provenance-bound. Every Axis Contract declares legal operations; subtraction, averaging, interpolation, covariance, and Euclidean distance are forbidden unless the axis semantics justify them.

The analysis pipeline separates four regimes:

1. `EXACT_DISCRETE`: identities, graphs, lattices, schemas, gates, and provenance remain symbolic and exact;
2. `LINEAR`: a declared vector-space representation supports linear combinations and a meaningful metric;
3. `LOCALLY_LINEAR`: a nonlinear region admits a bounded chart or tangent approximation with an error/radius contract;
4. `NONLINEAR`: topology, geodesics, branching, discontinuities, or constraints require graph/manifold/kernel or other nonlinear treatment.

The objective is not to force every nonlinear structure into a global linear space. It is to find the simplest faithful family of charts. One global linear map is preferred when it meets the Distortion Budget; otherwise Gaia uses several local charts connected by explicit transition maps. Some structure may remain exact and unembedded.

### Projection pipeline

```text
characterize(snapshot, axisContracts) -> GeometryAssessment | Refusal
fitChart(trainingSnapshot, projectionRecipe) -> SemanticChart | Refusal
project(chart, subjectSnapshot) -> ProjectionReceipt | ProjectionRefusal
validate(chart, heldOutSnapshot, distortionBudget) -> ChartVerdict
```

A Projection Recipe declares exact input selectors, missing/null policy, unit transformations, categorical encoding, graph projection, scaling, target dimension, algorithm, randomness/reproducibility, hyperparameters, landmarks, training/held-out split, expected consumer, and falsifiers. Fitting and validation use copied immutable Graph Snapshots.

Evaluation is multi-axis. At minimum, a Projection Receipt records:

- coverage and refused/out-of-distribution subjects;
- reconstruction error where an inverse exists;
- pairwise distance or divergence distortion only for declared metrics;
- neighborhood precision/recall plus trustworthiness and continuity;
- ordinal/rank inversions;
- landmark displacement;
- connected-component, cycle, cut, and neighborhood-topology preservation where relevant;
- constraint and invariant violations;
- downstream decision disagreement against the exact reference path;
- stability across seeds, resamples, time windows, model/runtime releases, and held-out data;
- latency, memory, sample count, units, nulls, truncation, and provenance.

No single metric establishes faithfulness. Explained variance is sufficient only for the variance question, not for semantic, neighborhood, topological, causal, or decision preservation. A weighted aggregate may be emitted for a named consumer policy, but the complete quality vector and Pareto frontier remain available.

### Linear-first, nonlinear when justified

Gaia evaluates an identity/no-reduction reference and a deterministic linear baseline before more complex candidates. A nonlinear or learned chart is accepted only if it improves declared held-out metrics beyond those baselines without violating hard semantic constraints. Complexity, compute, instability, and inability to support new points count against it.

Out-of-sample mapping is mandatory for an operational Semantic GPS. A method that only positions the training snapshot may support retrospective visualization but not live localization. New subjects outside a chart's validity region produce `PROJECTION_OOD` or `LOCALIZATION_UNKNOWN`, never silently extrapolated coordinates.

Charts are hierarchical. A coarse global chart selects a relevant Graph Capsule or domain region; exact landmarks then anchor a local chart; the canonical WorkGraph is expanded when distortion or uncertainty crosses threshold. Overlapping charts need transition functions and consistency checks. Relocalization may replace a chart revision while preserving prior Projection Receipts.

### Authority boundary

Coordinates, clusters, directions, distances, and interpolations are advisory derived Artifacts. They may rank candidate probes or routes but cannot prove dependency, freshness, compatibility, acceptance, causality, or authority. Every route selected in a reduced space is replayed against the exact WorkGraph and deterministic gates before action.

## 12. Immediate consequence for the current Gaia work

S1 R1 independently reviewed immutable bundle digest `f44ecac23f0c9d180f1bcdcbacb920163a89dc0d04cebbbe6825dbc556c32832` and returned Standards `REQUEST_CHANGES` plus Spec `REQUEST_CHANGES` in report SHA-256 `84bc1e0be2cb59837e1e0a78219a7cb4c0f150eae226db0f5a135ec2c4f6f6b2`.

That verdict remains valid for its exact subject. This new design does not modify, erase, or retroactively widen that subject. Instead it becomes a new input for the next repaired bundle. The next bundle SHALL declare the old S1 review `SUPERSEDED` only after a fresh review binds the new exact digest; until then it is a current negative verdict over the old candidate and useful evidence.

No implementation, database, package installation, runtime integration, or refresh execution is authorized by this document.

`GAIA_ARTIFACT_WORKGRAPH_STALENESS_DESIGN_V0_1_COMPLETE`
