# ADR-0005: IX is an advisory engineering co-processor

Status: proposed (2026-08-19)

## Context

IX contains useful mathematical, optimization, signal-processing, search, dynamics, pipeline, and analyst-bench modules. Some exploratory aerospace artifacts combined those modules as stand-ins for industrial CAD, structural optimization, Pareto optimization, and toolpath planning. That made the demonstration visually compelling but overstated what the executed algorithms established.

The distinction is safety-critical. A reproducible mathematical result is not automatically a valid geometric model, physics result, collision-free path, certified control action, or production decision.

## Decision

IX is an **offline/advisory engineering co-processor**.

IX may:

- analyze immutable telemetry and simulation artifacts;
- run bounded design-of-experiments, surrogate-model, and optimization studies;
- compute descriptive geometry, signal, graph, dynamics, uncertainty, and quality metrics;
- rank candidates produced or evaluated by identified external engineering adapters;
- emit content-addressed study receipts with units, provenance, assumptions, budgets, uncertainty, and typed refusals;
- support evidence-grounded retrieval over engineering artifacts.

IX does not claim to provide:

- a CAD/solid-modeling kernel or native STEP AP242 semantic model;
- a structural topology optimizer merely because `ix-topo` computes persistent homology;
- an FEA, CFD, multiphysics, meshing, or material-model solver;
- a Pareto front merely because `ix-game` computes a Nash equilibrium;
- a collision-aware robot or machine-tool path merely because `ix-graph` computes a Viterbi sequence;
- a complete digital-twin platform;
- a hard-real-time controller, safety PLC, safety function, certification verdict, distributed consensus service, or authoritative lock manager.

Physical and authoritative guarantees belong to external systems behind explicit adapters: for example a CAD geometry kernel, an independently verified CAE solver, a collision checker and motion planner, a robot controller, a safety PLC, or an authoritative transactional store. An IX result remains advisory until the responsible external system or human accepts it.

## Required evidence for an engineering claim

An IX engineering result must bind:

1. immutable input and output identities;
2. physical units and coordinate frames;
3. the exact IX and adapter revisions;
4. the external solver/tool identity and configuration when one is used;
5. declared assumptions, constraints, budgets, seeds, convergence conditions, and stop rules;
6. negative controls or an independently checkable oracle;
7. uncertainty and out-of-domain status;
8. an explicit `advisory` authority classification.

Absent any required field, the study must return a typed refusal or `UNKNOWN`; it must not silently substitute a weaker algorithm.

## Discrete-first coordination mathematics

The concluded GA research artifact [`docs/research/2026-08-09-mechanical-tensors-agentic-systems.md`](https://github.com/GuitarAlchemist/ga/blob/e430c7c19121f50f8d1bbb818c77c5c1e1923c9d/docs/research/2026-08-09-mechanical-tensors-agentic-systems.md) at revision `e430c7c19121f50f8d1bbb818c77c5c1e1923c9d` (31668 bytes, SHA-256 `9df446ffa6b5ea2fc06d51eb29a5dbbe1bcc8732a73b45854bd57db6510183a9`) is an advisory input to this decision. It supports balance laws, typed graph flows, queue backpressure, graph gradients and Laplacians, cycle decomposition, stability experiments, and cumulative-damage models. It does not establish a continuum tensor model for agent coordination.

IX therefore must:

- preserve separate physical or operational units instead of summing tokens, seconds, currency, bytes, and failures into one stress scalar;
- call an observable an edge-flow field, load matrix, queue pressure, graph gradient, or response Jacobian unless a tensor transformation law is proved;
- require versioned geometry, coordinate frames, adequate directional rank, conditioning, covariance tests, and held-out gain before accepting a tensor claim;
- keep every coordination-shape result content-addressed, read-only, and advisory;
- compare mechanics-inspired features against event-count, queue-only, and centrality-plus-queue baselines.

The first admissible vertical slice is an immutable-event-log tracer that derives typed actor and edge windows, balance residuals, tail latency, graph energy, and cycle exposure. It may predict or explain coordination failures, but it may not route work, grant authority, control a robot, or mutate the Gaia bus.

## Architectural seam

The intended seam is a deep engineering-study module whose small interface accepts a typed study plus named adapters and returns an immutable receipt or typed refusal. Adapter execution is bounded and evidence-producing. The interface must not expose raw controller writes or imply authority transfer.

The exact interface remains reversible and will be chosen through Design It Twice before implementation. This ADR fixes the safety and truth boundary, not a public Rust interface.

## Consequences

- Existing CATIA/A350 artifacts are retained as historical exploratory material with prominent warnings.
- Future demonstrations must label synthetic data and algorithmic stand-ins at the point of use.
- A true Pareto module, collision-aware planning adapter, CAD/CAE adapters, and telemetry/digital-twin read models may be added as separate vertical slices with independent validation.
- DuckDB remains an analyst's bench and IXQL remains a typed verification/planning language. Neither becomes a robot controller, safety system, consensus protocol, or lock authority.
- Mechanics-inspired coordination features remain experimental until they demonstrate invariant, held-out gain over simpler queue and graph baselines.

## Reversibility and revisit trigger

This is a two-way door for advisory capabilities and adapters, but a one-way safety boundary for claims of authority. Revisit a specific excluded capability only after IX has a dedicated module, an independent domain oracle, failure-mode tests, measured operating limits, and explicit human approval for the corresponding safety case.
