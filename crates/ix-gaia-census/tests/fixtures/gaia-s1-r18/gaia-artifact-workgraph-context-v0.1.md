# Gaia Artifact WorkGraph Context v0.1

This context names the durable work products and freshness concepts through which a Gaia project progresses. It does not describe an implementation.

## Durable work

**Artifact**:
A durable work product that can be identified and reviewed, such as a specification, source snapshot, test result, build, review, decision, dataset, model, or deployment receipt.
_Avoid_: File, document, output

**Artifact Revision**:
One immutable, content-addressed state of an Artifact.
_Avoid_: Latest file, current copy

**Artifact Lineage**:
The directed relationships connecting an Artifact Revision to the exact revisions and decisions from which it derives or whose claims it validates.
_Avoid_: Folder tree, document chain

**Artifact WorkGraph**:
The canonical directed provenance graph whose nodes are Artifact Revisions and Transition Receipts and whose edges record exact consumption, production, validation, and supersession relationships.
_Avoid_: File graph, agent graph

**Derivation Edge**:
A declared Artifact Lineage relationship whose freshness policy says whether and how an upstream revision affects its consumer.
_Avoid_: Link, reference

**Build Recipe**:
The content-addressed procedure, toolchain, parameters, and environment contract capable of reproducing an Artifact Revision from declared inputs.
_Avoid_: Script, command

**Composite Artifact**:
An Artifact whose implementation is an internal Artifact WorkGraph and whose external interface exposes only declared inputs, outputs, invariants, gates, and one root digest.
_Avoid_: Folder, project blob

**Transition Definition**:
A versioned Artifact that declares how exact input revisions may produce output revisions, including schemas, preconditions, authority, budget, gates, invalidation, and evidence requirements.
_Avoid_: Workflow, automation rule

**Transition Receipt**:
Content-addressed evidence of one Transition Definition applied to exact inputs under a named fence, including outputs, gates, cost, and terminal execution result.
_Avoid_: Success log, job result

**Quality Assessment**:
A set of independent, subject-bound gate verdicts over an Artifact Revision. It is separate from freshness, lifecycle, and acceptance.
_Avoid_: Quality score, status

**Graph Snapshot**:
An immutable, content-addressed view of an Artifact WorkGraph at an accepted ledger position, used as the exact subject of analysis or replay.
_Avoid_: Current graph, live query

**Graph Projection**:
A reconstructible view derived from a Graph Snapshot for one purpose, such as impact, coupling, ownership, quality, or scheduling analysis.
_Avoid_: Source graph, truth database

**Graph Capsule**:
A Composite Artifact that contracts a mature, accepted subgraph behind a declared boundary interface while preserving a content-addressed expansion manifest.
_Avoid_: Squashed history, deleted nodes

**Compaction Receipt**:
Evidence that a Graph Capsule preserves required boundary reachability, lineage, gates, and replay semantics for an exact subgraph.
_Avoid_: Summary, archive marker

**Maturity Gate**:
The explicit evidence policy a subgraph must satisfy before it may be represented as a Graph Capsule.
_Avoid_: Old enough, stable-looking

## Backbone evolution

**Backbone Runtime**:
The replaceable implementation that evaluates Artifact WorkGraph rules and materializes projections from canonical lineage facts.
_Avoid_: The graph, source of truth

**Backbone Release**:
An immutable, content-addressed bundle of Backbone Runtime code, transition interpreters, schemas, configuration, and compatibility declarations.
_Avoid_: Deployment, latest runtime

**Shadow Evaluation**:
A read-only replay in which a candidate Backbone Release consumes the same Graph Snapshots as the active release and produces comparison evidence without authority to publish or cause effects.
_Avoid_: Dry run, second coordinator

**Cutover Receipt**:
Evidence that one candidate Backbone Release became active at a new monotonic epoch after shadow comparison, readiness gates, and atomic fence transfer.
_Avoid_: Toggle, deployment log

## Operational uncertainty

**Fog-of-War Envelope**:
The structured decision frontier containing known facts, assumptions, hypotheses, contradictions, missing evidence, safe actions, blocked actions, validity horizons, and falsifiers.
_Avoid_: Confidence score, general uncertainty

**Uncertainty Condition**:
A typed reason knowledge is insufficient or unsafe to use, such as `MISSING`, `UNOBSERVED`, `INACCESSIBLE`, `STALE`, `PARTIAL`, `AMBIGUOUS`, `CONTRADICTORY`, `DIVERGENT`, `VOLATILE`, `NOVEL`, `UNDERDETERMINED`, `IRREDUCIBLE`, `DECEPTIVE_RISK`, `TEMPORAL_RISK`, or `MODEL_RISK`.
_Avoid_: Unknown, low confidence

**Probe**:
A bounded, reversible Transition selected to discriminate hypotheses or reduce a declared Uncertainty Condition under explicit authority, cost, information-gain, and blast-radius constraints.
_Avoid_: Try something, exploration

**Uncertainty Frontier**:
The smallest subgraph whose decisions or gates are affected by unresolved Uncertainty Conditions.
_Avoid_: Everything unclear

**Reproducibility Envelope**:
The exact conditions under which a Transition is expected to reproduce byte-identical, decision-equivalent, statistically bounded, or explicitly non-replayable output.
_Avoid_: Temperature, deterministic mode

**Nondeterminism Source**:
A declared source of output variation, including decoding, model/version serving, context assembly, tools, environment, concurrency, external observations, or human input.
_Avoid_: LLM randomness

**Determinization Boundary**:
The seam beyond which variable model proposals are converted into validated, content-addressed, idempotent, and fenced decisions or effects.
_Avoid_: Make the LLM deterministic

**Semantic Equivalence Class**:
A versioned deterministic rule for deciding when non-byte-identical outputs are equivalent for one declared consumer and gate.
_Avoid_: Looks the same, close enough

**Entropy Budget**:
A bounded allowance for deliberate diversity across candidate generations, including sample count, decoding policy, cost, aggregation, and stopping rule.
_Avoid_: More creativity

**Refinement Policy**:
A versioned policy that tightens or widens a Reproducibility Envelope for a named Transition while declaring invalidation and review consequences.
_Avoid_: Tune settings

## Semantic navigation

**Semantic Map**:
A versioned multi-resolution projection joining domain concepts, Artifact lineage, module seams, evidence, quality gates, and available Transitions for navigation purposes.
_Avoid_: Embedding index, repository tree

**Semantic Position**:
An evidence-bound estimate of the current accepted problem, solution, implementation, and verification state within a named Semantic Map projection.
_Avoid_: Current prompt, current branch

**Semantic Destination**:
A versioned set of target invariants and acceptance gates, including explicit non-goals and forbidden regions.
_Avoid_: User request, epic title

**Semantic Landmark**:
A stable domain invariant, deep-module seam, accepted Artifact, contract, test oracle, or decision that supports localization across map revisions.
_Avoid_: Important file, keyword

**Semantic Route**:
A bounded, dependency-valid sequence or partial order of candidate Transitions from a Semantic Position toward a Semantic Destination.
_Avoid_: Generated plan, task list

**Relocalization Receipt**:
Evidence that a Semantic Position and its remaining route were recomputed after map drift, staleness, contradiction, failed probes, or newly accepted Artifacts.
_Avoid_: Re-plan message

**Axis Contract**:
The declared mathematical structure, units, null semantics, ordering, admissible operations, distance or divergence, aggregation, and provenance for one analysis axis.
_Avoid_: Feature, column

**Product State Space**:
The heterogeneous product of Axis Contracts used to represent a decision state without pretending every axis belongs to one vector space.
_Avoid_: Feature vector

**Semantic Chart**:
A bounded mapping from a region of the Product State Space into a lower-dimensional coordinate space, with declared distortion, validity region, landmarks, and out-of-distribution behavior.
_Avoid_: Embedding, visualization

**Projection Receipt**:
Evidence binding an exact input snapshot, encoding and reduction recipes, output coordinates, distortion/coverage metrics, lost information, validity region, and reproducibility envelope.
_Avoid_: Plot, model output

**Distortion Budget**:
The consumer-specific limits on neighborhood, order, distance, topology, constraint, and decision distortion permitted for a Semantic Chart.
_Avoid_: Explained variance

**Uncertainty Envelope**:
A content-addressed multi-axis assessment of one exact subject, preserving knowledge conflict, predictive bounds, noise decomposition, provenance/dependence, temporal validity, and decision sensitivity.
_Avoid_: Confidence, probability

**Credal Set**:
A coherent set of probability distributions representing unresolved predictive ambiguity for declared outcomes and assumptions.
_Avoid_: Confidence interval, probability range

**Decision Envelope**:
An Artifact containing admissible and refused actions, the Pareto frontier, policy, constraints, utility/regret bounds, witnesses, sensitivity, residual uncertainty, authority state, and expiry.
_Avoid_: Recommendation, best action

**Aggregation Recipe**:
A versioned mathematical contract for combining compatible Evidence Items or Uncertainty Envelopes under explicit dependence, conflict, null, unit, scope, and validity assumptions.
_Avoid_: Average confidence, consensus

## Freshness

**Freshness Assessment**:
A deterministic statement of whether an Artifact Revision remains usable under a named consumer policy and exact upstream heads.
_Avoid_: Cache status, last modified

**Impact Closure**:
The smallest transitively affected subgraph produced by changed revisions and the invalidation policies on their outgoing Derivation Edges.
_Avoid_: Everything downstream

**Refresh Plan**:
A bounded, dependency-ordered proposal for rebuilding or revalidating an Impact Closure under an explicit budget and authority boundary.
_Avoid_: Auto-fix, sync job

**Compatibility Receipt**:
Reviewed evidence that a changed upstream revision remains compatible with a named Derivation Edge, allowing an existing downstream Artifact Revision to remain fresh for that edge.
_Avoid_: Ignore flag, force fresh

**Freshness Requirement**:
The consumer's declared rule for accepting, warning on, or refusing a non-fresh Artifact Revision.
_Avoid_: Read mode

## Working rule

A project advances when accepted Artifact Revisions and their fresh evidence satisfy a gate. Agent activity, conversation progress, modified files, passing phases, or newly written prose do not by themselves advance the project.

`GAIA_ARTIFACT_WORKGRAPH_CONTEXT_V0_1_COMPLETE`
