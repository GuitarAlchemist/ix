# Gaia multi-axis uncertainty analysis design v0.1

Status: specification candidate; research and design only; no runtime authority.

Primary foundation: `gaia-multiaxis-uncertainty-math-primary-research.md`, exact digest recorded in the Wayfinder cursor. Grammar projection: `gaia-uncertainty-grammar-v0.1.ebnf`.

## 1. Purpose

Gaia needs to reason under fog of war without converting heterogeneous uncertainty into one confidence number. The module answers four bounded questions:

1. what is known, contradicted, missing, stale, or otherwise uncertain about an exact subject;
2. what evidence combination is mathematically justified;
3. which actions remain admissible or robust under surviving interpretations;
4. which bounded Probe has positive decision-relevant information value.

It does not establish freshness, quality, authority, or acceptance. Those remain separate gates.

## 2. Typed mathematical state

For claim `c`, the authoritative state is a heterogeneous product:

```text
U(c) = K(c) x P(c) x N(c) x R(c) x V(c) x D(c)
```

where:

- `K`: knowledge/evidence state with independent support and refutation;
- `P`: predictive uncertainty, usually a credal set or explicit refusal;
- `N`: noise decomposition, including aleatoric and epistemic estimates with assumptions;
- `R`: provenance/dependence graph;
- `V`: valid-time, observation-time, precision, expiry, and recheck policy;
- `D`: decision sensitivity, utility bounds, regret, and information value for a named decision.

This is a Product State Space, not an implicit Euclidean vector. Each Axis Contract declares its carrier set, order or preorder, units, null/unknown semantics, legal operations, aggregation recipes, precision, and witnesses.

### 2.1 Knowledge bilattice

Logical evidence uses independent support/refutation bits or evidence sets:

```text
NEITHER    = (support=0, refutation=0)
SUPPORTED  = (support=1, refutation=0)
REFUTED    = (support=0, refutation=1)
BOTH       = (support=1, refutation=1)
```

`NEITHER` is incomplete; `BOTH` is inconsistent. They MUST NOT collapse to the same scalar. Truth ordering and knowledge ordering are distinct. Evidence joins retain provenance and dependence groups; a join is illegal when the Axis Contract supplies no join.

### 2.2 Predictive uncertainty

When a defensible probability model exists, `P(c)` is a non-empty credal set `C` of distributions over declared outcomes `Omega`. Gaia exposes lower and upper expectations:

```text
lower_E_C[f] = inf_{p in C} E_p[f]
upper_E_C[f] = sup_{p in C} E_p[f]
```

An arbitrary interval attached independently to each event is not necessarily coherent and is refused. Point probabilities are a special case where `C` is a singleton. Unknown dependence remains bounds/sensitivity, not multiplied marginal confidence.

### 2.3 Aleatoric and epistemic axes

`N(c) = (A, E, estimator, assumptions, identifiability_limit)`.

- Aleatoric uncertainty describes outcome/observation variability not expected to vanish with the allowed information.
- Epistemic uncertainty describes model or knowledge uncertainty potentially reducible by evidence.

The split is estimator- and model-dependent; `UNKNOWN` is valid. Gaia MUST NOT fabricate a unique decomposition where the evidence does not identify one.

### 2.4 Provenance, dependence, and time

Every Evidence Item declares immutable subject, claim/polarity, source Entity, producing Activity, responsible Agent, derivation edges, dependence/common-cause group, observed time, valid-time interval, temporal precision, scope, and expiry/revalidation policy.

Evidence derived from the same origin is correlated unless an explicit dependence model proves otherwise. Expiry changes usability/freshness; it never erases historical provenance.

## 3. Orders and combination

The multi-axis state uses componentwise product preorders only for declared comparable axes:

```text
x <= y iff for every comparable axis i, x_i <=_i y_i
```

If one state has stronger provenance but wider probability bounds, neither dominates the other. Both remain on the Pareto frontier. No canonical total order or weighted sum exists.

Evidence aggregation requires a versioned Aggregation Recipe naming:

- compatible claim, subject, scope, units, and validity window;
- provenance/dependence assumptions;
- operator and mathematical justification;
- conflict behavior and refusal threshold;
- missing/null behavior;
- output type, bounds, witnesses, and falsifiers.

Default behavior is structural union plus explicit conflict. Naive averaging, voting, confidence multiplication, fuzzy-membership-as-probability, and unqualified normalized Dempster-Shafer fusion are rejected.

## 4. Decisions under uncertainty

For action set `A`, outcome/state set `Omega`, utility `u(a,w)`, and credal set `C`, Gaia computes when tractable:

```text
lower_utility(a) = inf_{p in C} E_p[u(a,w)]
upper_utility(a) = sup_{p in C} E_p[u(a,w)]
```

Hard authority, safety, cost, freshness, and quality constraints filter actions before utility analysis. Remaining actions may be compared by dominance. Incomparable actions remain explicit.

An optional named policy may choose minimax regret:

```text
regret(a,w) = max_{a' in A} u(a',w) - u(a,w)
MMR(a)      = sup_{p or w allowed} expected_or_statewise_regret(a)
```

The report includes the worst-case witness. Minimax regret is not the universal default: it can be excessively conservative and insensitive to likelihood.

Every selected action produces a Decision Envelope containing admissible set, rejected actions/reasons, Pareto frontier, policy, constraints, utility bounds, regret, witness states/distributions, sensitivity, residual Uncertainty Conditions, authority status, and expiry.

## 5. Probes and value of information

A Probe is proposed only relative to a pending decision. For observation design `q`:

```text
EVI(q) = expected value of optimal decision after q
       - value of optimal decision before q
net_EVI(q) = EVI(q) - acquisition_cost(q) - risk_cost(q)
```

Under imprecise probabilities, Gaia reports lower/upper net EVI or sensitivity across `C`; it does not hide sign disagreement. Positive expected information is insufficient until authority, reversibility, blast radius, time, privacy/security, and budget pass. Probe selection also declares what result would discriminate hypotheses and when to stop.

## 6. Calibration and empirical validity

Forecasts are recorded before outcomes. Repeated predictive tasks use strictly proper scores appropriate to the outcome type, with reliability/calibration by axis, source class, cohort, and horizon. Accuracy alone is insufficient.

Conformal prediction is permitted only with a declared exchangeability-style assumption and held-out calibration set. Its marginal coverage is not epistemic truth, causal evidence, one-off-claim validation, or automatic conditional coverage. Drift or coverage failure invalidates its operational receipt.

## 7. Geometry and semantic navigation

The exact typed state and evidence graph remain canonical. Geometry begins with:

1. identity/no reduction;
2. deterministic linear baseline such as PCA only over suitable numeric Axis Contracts;
3. local-linear/manifold or graph-spectral candidates if held-out distortion warrants them;
4. seeded nonlinear visualization as an exploratory lens only.

A Semantic Chart records coverage, reconstruction where meaningful, trustworthiness, continuity, rank/distance distortion for declared metrics, landmark movement, graph/topology witnesses, downstream decision disagreement, resampling/seed stability, OOD behavior, cost, and provenance.

Operational navigation requires an explicit out-of-sample map or a refusal. Apparent clusters, distances, directions, areas, or empty regions in a low-dimensional map do not establish semantic facts. Selecting a coordinate always reveals original axis values, source neighbors, provenance, valid time, map revision, and distortion vector.

## 8. Deep-module plumbing

```text
assess(subjectRevision, evidenceSet, axisContracts)
  -> UncertaintyEnvelope | AssessmentRefusal

combine(envelopes, aggregationRecipe)
  -> UncertaintyEnvelope | CombinationRefusal

decide(envelope, decisionProblem, decisionPolicy)
  -> DecisionEnvelope | DecisionRefusal

proposeProbe(decisionEnvelope, probeCatalog, budget)
  -> ProbeProposal | ProbeRefusal

calibrate(forecastReceipts, outcomeSnapshot, calibrationRecipe)
  -> CalibrationArtifact | CalibrationRefusal

project(graphSnapshot, envelopeSet, projectionRecipe)
  -> ProjectionReceipt | ProjectionRefusal
```

The implementation hides bilattice operations, credal optimization, dependency checks, temporal validity, scoring, sensitivity, Pareto computation, and projection audits. Callers never manipulate raw confidence arithmetic.

### Required Artifacts

- `AxisContract`
- `EvidenceItem`
- `UncertaintyEnvelope`
- `AggregationRecipe`
- `DecisionProblem`
- `DecisionPolicy`
- `DecisionEnvelope`
- `ProbeProposal`
- `ForecastReceipt`
- `CalibrationArtifact`
- `ProjectionRecipe`
- `ProjectionReceipt`

Every Artifact is content-addressed and binds subject/version, units, nulls, provenance, recipe/toolchain digest, reproducibility envelope, budgets, truncation/approximation, and validity.

## 9. Negative controls

1. `NEITHER` and `BOTH` remain distinguishable.
2. stronger support with worse freshness remains incomparable, not averaged.
3. shared-provenance evidence is not counted as independent corroboration.
4. total or near-total evidence conflict cannot normalize into false certainty.
5. absent probability dependence refuses marginal-confidence multiplication.
6. incoherent event-wise probability intervals are rejected.
7. epistemic/aleatoric decomposition without identifying assumptions returns `UNKNOWN`.
8. expired evidence remains in lineage but cannot satisfy a current gate.
9. a scalar dashboard score cannot hide its axis vector or policy revision.
10. robust action reports its worst-case witness and residual uncertainty.
11. positive EVI cannot bypass authority, risk, reversibility, or budget.
12. a Probe result cannot self-accept or self-authorize its downstream action.
13. a calibration claim without preregistered forecast and observed outcome is rejected.
14. conformal coverage without a valid calibration set/assumption is rejected.
15. a nonlinear map that does not beat the linear baseline on declared held-out metrics is rejected.
16. a training-only embedding cannot localize a new Artifact.
17. truncated or distorted projection cannot claim absence of a cluster, route, or bottleneck.
18. reduced-space route selection must replay against exact graph constraints before action.

## 10. Milestone boundary

This design is a candidate input to the next repaired S1 bundle. It authorizes no implementation, database, package, TARS registration, model execution, routing, spend, or product mutation. M2-M5 remain absent.

`GAIA_MULTIAXIS_UNCERTAINTY_ANALYSIS_DESIGN_V0_1_COMPLETE`
