# TARS V1 Exploration Inventory (IX perspective)

Issue: [GuitarAlchemist/ix#190](https://github.com/GuitarAlchemist/ix/issues/190) — parent epic
[#189](https://github.com/GuitarAlchemist/ix/issues/189).

This is a **classification of the TARS V1 Markdown corpus from an IX perspective**. It is a research
note, not an adoption decision: nothing here has been ported, and no claim from the old exploration
text is treated as validated. The purpose is to separate the small set of documents that describe
something IX could own (a deterministic algorithm, a benchmark, a fixture, an adapter) from the much
larger set that describes TARS orchestration, TARS project history, or nothing reusable at all.

## Evidence base

| Item | Value |
| --- | --- |
| Source repository | `GuitarAlchemist/tars` (local sibling checkout `../tars`) |
| Pinned ref | `origin/main` @ `69cf427eccb25514728eacfd3530218df3975259` |
| Corpus surveyed | `v1/**/*.md`, `v1/docs/Explorations/**`, `v1/TODOs/**`, `.tars/workspace/**`, `v2/docs/4_Research/V1_Insights/**` |
| Survey date | 2026-08-03 |
| Cost incurred | 0 USD — local file reads only (`free-local`) |

Every `source_doc` path below was resolved against that checkout and its content read before the
summary was written. Paths are given relative to the tars repository root.

## Method, and what it does *not* cover

1. **Enumerated** the corpus by directory (census below) rather than sampling blindly.
2. **Filtered** to IX-relevant topics by filename and by content grep across the IX category
   vocabulary (`vector`, `graph`, `grammar`, `search`, `math`, `music`, `trace`, `duckdb`, `eval`,
   `state_space`, `tree_of_thought`, `workflow_of_thought`).
3. **Read** each surviving document (head sections in full for the long chat transcripts; complete
   for the short specification documents) before classifying it.
4. **Cross-checked** against the existing IX crate set so `recommended_action` reflects what IX
   already has.

Not covered, deliberately: the 272 Markdown files under derived or parked trees
(`v1/parked_legacy/`, `v1/parked_csharp/`, `v1/output/`, `v1/autonomous_backups/`,
`v1/integration_test_output/`, `v1/test_real_execution/`, `v1/live_demo_results/`). These are build
output, snapshots of earlier C# code, and autonomous-run artifacts; they are classified as a band
(see below) but not read document by document. No hosted summarization pass was run over the long
transcripts — the issue's `no paid full-corpus summarization` non-goal is honoured.

## Corpus census

`v1/**/*.md` = **865** files, distributed as:

| Directory | Files | Band |
| --- | ---: | --- |
| `v1/docs/` | 209 | mixed — holds `docs/Explorations/` (74) and the feature/architecture notes |
| `v1/` (root) | 192 | mixed — holds the long exported ChatGPT research docs and 95 status reports |
| `v1/parked_legacy/` | 119 | derived / parked |
| `v1/output/` | 103 | derived |
| `v1/TODOs/` | 101 | TARS-core planning, with a few IX-relevant specs |
| `v1/.specify/` | 46 | TARS-core tooling config |
| `v1/parked_csharp/` | 36 | derived / parked |
| `v1/organization/` | 16 | TARS-core org charts |
| `v1/.augment/`, `.agent-os/`, `.claude/` | 17 | agent harness config |
| everything else | 26 | derived output, samples, docs stubs |

Adjacent corpora:

| Corpus | Files | Note |
| --- | ---: | --- |
| `v1/docs/Explorations/**` | 74 | 64 in `v1/docs/Explorations/v1/Chats/`, the rest are `Reflections/` and four identical copies of one DeepThinking doc under `v2/`–`v5/` |
| `.tars/workspace/**` | 46 | 3 of its 5 `explorations/` files are byte-identical copies of `v1/docs/Explorations/v1/Chats/` files; the plans are TARS migration history |
| `v2/docs/4_Research/V1_Insights/**` | 5 | TARS's own V1→V2 salvage analysis — the highest signal-per-line documents in the whole survey |

### Classification bands

Bands partition the 865 files of `v1/**/*.md`. Band B is the remainder after the other four are
subtracted, so the five counts sum exactly to 865.

| Band | Count | What it is | IX disposition |
| --- | ---: | --- | --- |
| **A — IX candidate** | 23 | Describes an algorithm, benchmark, fixture, or adapter IX could own | Inventoried in full below |
| **B — TARS-core** | 376 | Agent orchestration, metascript/FLUX runtime, MCP wiring, deployment, F# migration plans | Stays in TARS; listed by theme below |
| **C — project history** | 115 | `*_SUCCESS.md`, `*_COMPLETE.md`, `*_SUMMARY.md`, `*_REPORT.md`, `*_FIXED*.md` status write-ups | No IX value; ignore |
| **D — derived / parked** | 272 | Build output, parked C#/legacy trees, autonomous-run artifacts | No IX value; ignore |
| **E — config / harness** | 79 | `.specify`, `.augment`, `.agent-os`, `.claude`, org charts | No IX value; ignore |

Three further band-A candidates come from the adjacent corpora rather than `v1/`: two from
`v2/docs/4_Research/V1_Insights/` and one from `.tars/workspace/`, giving **26** candidates in total
below.

Bands C, D, and E together are 466 of 865 files — over half the corpus before any judgement about
technical merit is applied. That is the single most useful result of this survey: **the reusable
fraction of TARS V1 Markdown is small**, and mining it further with a paid full-corpus pass would be
poor value.

## Candidate inventory

Field vocabulary is exactly as specified in #190: `candidate_type` ∈
`algorithm|benchmark|fixture|adapter|research_note|defer`; `priority` ∈ `P0|P1|P2|P3`; `complexity` ∈
`XS|S|M|L|XL`; `testability` ∈ `high|medium|low`; `cost_tier` ∈
`free-local|cheap-hosted|paid-agent|manual-approval`; `recommended_action` ∈
`prototype|benchmark|document|defer`.

`cost_tier` is the cost of the **next step** on that candidate, not the cost of having read it:
`free-local` = local file reads plus local Rust; `cheap-hosted` = one hosted embedding or
small-model pass is needed (used for the multi-thousand-line transcripts where a targeted extraction
is required); `manual-approval` = touches a one-way door and needs human sign-off first.

### Category: `state_space`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-State-Space for TARS.md"
summary: "Argues for modelling an agent loop as a discrete-time state-space system (x_{k+1}=Ax_k+Bu_k+w_k) and applying observability, controllability, Lyapunov stability, Kalman filtering and MPC to detect cognitive drift and looping."
ix_category: [state_space, math, eval]
candidate_type: algorithm
priority: P1
complexity: M
testability: high
cost_tier: free-local
recommended_action: prototype
follow_up: "GuitarAlchemist/ix#193"
notes: "443 lines, dated 2025-03-23. The control-theory content is standard and independently checkable; the TARS-specific mapping is speculative. IX already has ix-dynamics, ix-signal and ix-chaos, so the prototype is a small adapter over existing primitives, not new math."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/MISSED_OPPORTUNITIES_ANALYSIS.md"
summary: "TARS's own audit of techniques discussed in the v1 chats but never implemented — state-space/Kalman/MPC, topological data analysis, category theory, neural ODEs, spectral methods, causal inference, Lie-theoretic structures."
ix_category: [state_space, math, eval]
candidate_type: research_note
priority: P0
complexity: S
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#202, GuitarAlchemist/ix#194"
notes: "218 lines. The best index into the math corpus: it names the source chat for each gap, which makes it cheap to verify. Roughly half the named gaps are already closed on the IX side (ix-topo, ix-category, ix-dynamics, ix-signal), which is exactly what #202's gap matrix needs as input."
```

### Category: `vector`

```yaml
source_doc: "GuitarAlchemist/tars:v1/ChatGPT-Vector Decomposition for TARS.md"
summary: "Deep-research survey (21 cited sources) of mathematical spaces for decomposing entangled embedding vectors into interpretable constituents — matrix factorization/PCA/SVD, sparse dictionary learning, hyperbolic and manifold embeddings, tensor decomposition — with round-trip transformations."
ix_category: [vector, math]
candidate_type: research_note
priority: P1
complexity: M
testability: high
cost_tier: free-local
recommended_action: benchmark
follow_up: "GuitarAlchemist/ix#194"
notes: "1513 lines, dated 2025-06-25. It reports 21 sources and provides direct academic links, making its claims easier to audit than the uncited exploration notes. Directly adjacent to ix-optick-sae and ix-manifold; its claims are testable today against the OPTIC-K voicing corpus, which makes this the strongest benchmark candidate in the inventory."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-TARS Multi-modal Memory Space.md"
summary: "Proposes a shared projective memory vector space in homogeneous coordinates where each modality (text, 3D, code) is related by a transformation matrix."
ix_category: [vector, math]
candidate_type: research_note
priority: P2
complexity: M
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#194"
notes: "844 lines, 2025-03-24. The homogeneous-coordinate machinery is sound and already covered by ix-rotation/ix-math; the unvalidated part is the assumption that cross-modal embeddings are related by a linear map. Document the claim, do not build on it."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Genetic Memory and Vectors.md"
summary: "Explores a bit-alphabet 'genetic memory' encoding for agent state, connecting it to vector-store geometry, entropy measures and genetic-algorithm mutation/crossover."
ix_category: [vector, math]
candidate_type: research_note
priority: P2
complexity: M
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#204"
notes: "544 lines, 2025-03-30. Overlaps ix-evolution's existing operators. Useful only as prior art for the mutation-operator work in #204."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/ChatGPT-CUDA vector store evaluation.md"
summary: "External evaluation of the TARS in-memory CUDA vector store, covering indexing strategy, memory layout and benchmark methodology."
ix_category: [vector, defer]
candidate_type: defer
priority: P3
complexity: L
testability: low
cost_tier: cheap-hosted
recommended_action: defer
follow_up: "GuitarAlchemist/ix#189 (parent epic — no dedicated story)"
notes: "2678 lines, 2025-08. Defer on a stated IX constraint, not on quality: CLAUDE.md fixes GPU compute to wgpu, so a CUDA-specific store is out of scope. The benchmark *methodology* could be salvaged later, but that needs a hosted extraction pass over the transcript."
```

### Category: `graph`

```yaml
source_doc: "GuitarAlchemist/tars:v1/ChatGPT-Leveraging Primes for TARS (1).md"
summary: "Long transcript that converges on a concrete proposal: build a directed hypergraph of .trsx metascript diffs over time, embed each diff as a 16-D vector, and partition it with BSP generalised to sedenion space to surface regions of coherence or contradiction."
ix_category: [graph, vector, math]
candidate_type: algorithm
priority: P2
complexity: L
testability: medium
cost_tier: cheap-hosted
recommended_action: prototype
follow_up: "GuitarAlchemist/ix#194"
notes: "6993 lines. Credibility is mixed and must be flagged: the opening exchange mis-identifies TARS as a commercial chatbot platform and produces unusable advice, and the prime-number framing is decorative. The BSP-in-R^16 partitioning idea (lines ~746-1030) is separable from that framing and is the only part worth prototyping — IX already has ix-sedenion and ix-graph. A shorter 1164-line duplicate exists at `v1/ChatGPT-Leveraging Primes for TARS.md`; the roadmap below cites this longer one."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/TODOs/CHATGPT_LEVERAGING_PRIMES_ROADMAP.md"
summary: "Implementation roadmap distilled from the primes transcript: prime-triplet belief anchors, CUDA-accelerated generation, the TRSX hypergraph, Hurwitz quaternion lattices, and FLUX metascript integration."
ix_category: [graph, math, defer]
candidate_type: defer
priority: P3
complexity: L
testability: low
cost_tier: free-local
recommended_action: defer
follow_up: "GuitarAlchemist/ix#194"
notes: "254 lines. Useful as the map of what TARS actually intended to build, but its two load-bearing dependencies — CUDA and the FLUX metascript runtime — are both out of scope for IX. Defer the roadmap; keep the hypergraph candidate above separately."
```

### Category: `grammar`

```yaml
source_doc: "GuitarAlchemist/tars:v1/TARS_FRACTAL_GRAMMARS_README.md"
summary: "Specifies a fractal grammar system: self-similar recursive production rules with a computed fractal dimension, scale/rotate/translate/compose/recursive/conditional transformations, and emission to EBNF, ANTLR, JSON, GraphViz and SVG."
ix_category: [grammar, math]
candidate_type: algorithm
priority: P1
complexity: M
testability: high
cost_tier: free-local
recommended_action: prototype
follow_up: "GuitarAlchemist/ix#204, GuitarAlchemist/ix#203"
notes: "330 lines, written as a specification rather than a transcript — the most implementable grammar document in the corpus. Fractal dimension of a production rule is a deterministic, unit-testable quantity, and ix-grammar plus ix-fractal already cover both halves. The multi-format emitters are TARS tooling and should not be ported."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/TODOs/UNIFIED_GRAMMAR_EVOLUTION_GRANULAR_TASKS.md"
summary: "68-task breakdown for a hybrid engine combining tier-based emergent grammar evolution with fractal grammar composition across seven domains."
ix_category: [grammar]
candidate_type: research_note
priority: P2
complexity: L
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#204"
notes: "540 lines. Almost entirely F#-project mechanics (fsproj ordering, CLI wiring, DI). The reusable content is the *shape* of the tier+fractal hybrid, roughly 10% of the document."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-BNF and Parsing Alternatives.md"
summary: "Comparison of BNF, EBNF, ABNF, PEG and ANTLR, with guidance on which to pick for which purpose."
ix_category: [grammar]
candidate_type: research_note
priority: P3
complexity: XS
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "none — background only"
notes: "731 lines, 2025-03-22. Accurate but entirely general-purpose; contains nothing IX-specific. ix-grammar already settled on EBNF. Recorded so the survey is complete, not because it implies work."
```

### Category: `math`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Courbes Takagi et Rham.md"
summary: "Examines Takagi/Blancmange curves (continuous, nowhere differentiable, self-affine) and de Rham interpolatory IFS curves as computational primitives for multi-scale signals and interpolation."
ix_category: [math]
candidate_type: algorithm
priority: P1
complexity: S
testability: high
cost_tier: free-local
recommended_action: prototype
follow_up: "GuitarAlchemist/ix#203, GuitarAlchemist/ix#204"
notes: "1945 lines, 2025-03-28, French title with English body. Both curve families have closed-form constructions, so correctness is fully testable against known values — the highest-testability math candidate here. #203 already scopes exposing these through ix-fractal; this document is that story's provenance."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Nash Equilibrium in Dynamics.md"
summary: "Extends Nash equilibrium to dynamic and nonlinear systems — open-loop vs closed-loop equilibria, differential games, HJB/HJI characterisations, and the spatial/PDE case."
ix_category: [math, state_space]
candidate_type: research_note
priority: P2
complexity: L
testability: medium
cost_tier: cheap-hosted
recommended_action: benchmark
follow_up: "GuitarAlchemist/ix#194"
notes: "6406 lines — the longest document surveyed, and mostly tangent by the end. ix-game already implements static Nash; the differential-game extension is a genuine gap but proving an equilibrium numerically is hard to make into a fast test. Extracting the usable specification needs a targeted hosted pass, hence cheap-hosted."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Quaternions and Projective Geometry for TARS.md"
summary: "Surveys quaternions, dual quaternions and homogeneous/projective coordinates for kinematics, gesture modelling, and agent state representation."
ix_category: [math, vector]
candidate_type: research_note
priority: P2
complexity: S
testability: medium
cost_tier: cheap-hosted
recommended_action: document
follow_up: "GuitarAlchemist/ix#194"
notes: "4858 lines, 2025-03-23. The rigid-body content is standard and largely covered by ix-rotation; the novel part is the proposal to use dual quaternions for agent memory, which is an analogy with no stated evaluation. Length is why extraction is cheap-hosted rather than free-local."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/PAULI_MATRICES_EXPLORATION_FOR_TARS.md"
summary: "Proposes quantum-inspired agent state modelling with Pauli matrices — superposition of capabilities, interference for decision optimisation, entanglement for interdependencies, collapse on action."
ix_category: [math, state_space]
candidate_type: research_note
priority: P3
complexity: M
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#194"
notes: "300 lines. Split verdict: the algebra it states (Hermitian, unitary, traceless, the commutation and anticommutation relations) is correct and trivially unit-testable, while the agent-behaviour claims built on top are unfalsifiable as written. Adopt at most the algebra; do not import the cognitive framing."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md"
summary: "Component-by-technique matrix mapping TARS subsystems (orchestration, reasoning, code analysis, memory, Tree-of-Thought) onto state-space control, topological data analysis and fractal mathematics, with an impact rating per cell."
ix_category: [math, workflow_of_thought]
candidate_type: research_note
priority: P2
complexity: M
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#202"
notes: "248 lines. The impact ratings are asserted, not measured — every row is 'CRITICAL' or 'HIGH', which makes the ranking uninformative. Reusable as a checklist of subsystem/technique pairings for the #202 gap matrix, with the ratings discarded."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Markov Chains in AI.md"
summary: "Traces Markov-chain ideas through AI — n-gram models, MDPs and POMDPs for agent decision-making, and the probabilistic framing of token generation."
ix_category: [math, state_space]
candidate_type: research_note
priority: P3
complexity: XS
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "none — already covered"
notes: "169 lines, 2025-03-22. Introductory. IX already has Markov primitives (ix-math, memristive-markov); recorded for corpus completeness only."
```

### Category: `search`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Better Architecture than Q-star.md"
summary: "Compares speculated Q*-style RL architectures against generalist agents, MuZero-style model-based RL, neurosymbolic hybrids, retrieval-augmented memory architectures, and recursive self-improvement loops."
ix_category: [search, tree_of_thought]
candidate_type: defer
priority: P3
complexity: S
testability: low
cost_tier: free-local
recommended_action: defer
follow_up: "none"
notes: "415 lines, dated 2025-03-09 and reasoning about unreleased systems. Its central subject is speculation about an unpublished architecture; nothing here is falsifiable. This is the clearest instance of the #190 non-goal 'no unvalidated claims from old exploration text'."
```

### Category: `tree_of_thought`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/MetascriptTreeOfThought.md"
summary: "Describes the implemented Tree-of-Thought reasoning layer for metascripts: thought-node evaluation metrics, thought-tree operations, and the generate/validate/execute/analyse cycle."
ix_category: [tree_of_thought, graph]
candidate_type: fixture
priority: P1
complexity: S
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#192"
notes: "187 lines, and unlike the transcripts it documents code that existed. It is the closest thing in the corpus to a schema for a reasoning tree, so it is the natural validation source for the IX-side model already drafted in `docs/research/tars-v1-tree-of-thought-to-ix.md` (task #198, closed). Use it as a fixture, not as an implementation to port."
```

```yaml
source_doc: "GuitarAlchemist/tars:.tars/workspace/plans/todos/TODOs-Tree-of-Thought-Auto-Improvement-Pipeline-Detailed.md"
summary: "Task breakdown for a Tree-of-Thought auto-improvement pipeline that analyses code, generates candidate fixes, ranks them, and applies the winner, with per-stage completion state recorded inline."
ix_category: [tree_of_thought, workflow_of_thought]
candidate_type: fixture
priority: P2
complexity: M
testability: medium
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#192"
notes: "475 lines. Roughly the first third is F# compiler-service plumbing with no IX analogue; the pipeline decomposition itself is a usable second fixture for #192. The inline checkbox state also records which stages were never finished, which is useful signal about where the design broke down."
```

### Category: `workflow_of_thought`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/Explorations/v1/Chats/ChatGPT-Computation Expressions for AI Workflows.md"
summary: "Argues that F# computation expressions can encode a workflow grammar for AI operations (RAG, fine-tuning, inference) with composability and compile-time type safety."
ix_category: [workflow_of_thought, grammar]
candidate_type: defer
priority: P3
complexity: XS
testability: low
cost_tier: free-local
recommended_action: defer
follow_up: "none"
notes: "33 lines — a single question and answer, with no design. The mechanism is F#-specific and IX already expresses workflow composition through `ix-pipeline::dag::Dag<N>`. Defer."
```

### Category: `trace`

```yaml
source_doc: "GuitarAlchemist/tars:v2/docs/4_Research/V1_Insights/v1_component_reusability_analysis.md"
summary: "TARS's own component-by-component reuse rating of v1 for v2: VectorStore, Grammar and AgenticTraceCapture rated 70-90% reusable; AgentSystem, Inference and FLUX need refactor; CUDA, advanced math and 3D/UI explicitly deferred to v3+."
ix_category: [trace, eval]
candidate_type: adapter
priority: P1
complexity: S
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#200, GuitarAlchemist/ix#201"
notes: "696 lines, dated 2025-11-22, with real source paths and effort estimates. This is the highest signal-per-line document in the survey and it independently corroborates several deferrals reached here on separate grounds. Its trace-event taxonomy is the concrete input for an IX trace-ingest adapter (ix_trace_ingest already exists)."
```

```yaml
source_doc: "GuitarAlchemist/tars:v2/docs/4_Research/V1_Insights/v1_reuse_strategy.md"
summary: "The approved porting plan derived from the reusability analysis: which v1 file goes to which v2 project, in which phase, with the intended adaptation for each."
ix_category: [trace, workflow_of_thought]
candidate_type: adapter
priority: P2
complexity: S
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#201"
notes: "112 lines, dated 2025-12-20 and marked approved. Because it names exact source and destination paths, it is the cheapest way to check whether an IX-side assumption about a TARS artifact is still current."
```

### Category: `music`

```yaml
source_doc: "GuitarAlchemist/tars:v1/harmonic_progression_analyzer.tars.md"
summary: "Autonomous-instruction specification for a harmonic progression analyzer: chord identification, Roman-numeral functional analysis, voice-leading optimisation, modal interchange and borrowed-chord detection, substitution suggestions, and next-chord prediction."
ix_category: [music]
candidate_type: fixture
priority: P2
complexity: M
testability: high
cost_tier: free-local
recommended_action: document
follow_up: "GuitarAlchemist/ix#195"
notes: "236 lines with explicit success criteria. GA and IX already expose Roman-numeral analysis, modal-interchange, voice-leading, and voicing/search surfaces; real-time audio chord recognition, symbolic progression prediction, and DAW export remain gaps or roadmap work. Use this as a partial acceptance checklist, not as evidence that the whole analyzer exists. Its 'quaternion harmonic framework' dependency is unsubstantiated and should be dropped."
```

```yaml
source_doc: "GuitarAlchemist/tars:v1/GUITAR_ALCHEMIST_TARS_INTEGRATION_PLAN.md"
summary: "Plan for applying TARS autonomous self-improvement to the Guitar Alchemist codebase, with phased rollout and specific proposed optimisations."
ix_category: [music, workflow_of_thought]
candidate_type: defer
priority: P3
complexity: S
testability: low
cost_tier: free-local
recommended_action: defer
follow_up: "GuitarAlchemist/ix#195"
notes: "320 lines, dated 2025-08-30. Superseded by the live ga <-> ix <-> tars MCP federation described in CLAUDE.md, which is a different and better-evidenced integration than the one proposed here. Its quoted performance gain ('25%') has no measurement method attached and must not be cited."
```

### Category: `eval`

```yaml
source_doc: "GuitarAlchemist/tars:v1/docs/features/intelligence-progression-measurement.md"
summary: "Framework for scoring TARS 'intelligence' across multiple dimensions with a composite score, a benchmarking subsystem, and progression tracking over time."
ix_category: [eval]
candidate_type: defer
priority: P3
complexity: M
testability: low
cost_tier: manual-approval
recommended_action: defer
follow_up: "GuitarAlchemist/ix#205"
notes: "152 lines. The benchmark-harness shape is reusable but the composite 'intelligence score' is a self-graded metric with no external oracle — adopting it would install exactly the Goodhart failure the IX quality pipeline exists to avoid. Marked manual-approval because promoting any such score into `ga/state/quality/` is a governance decision, not an engineering one."
```

### Category: `duckdb` — no candidate

The IX category vocabulary includes `duckdb`, and the survey found **no** V1 Markdown candidate for
it. DuckDB appears exactly once across the entire corpus, as an incidental item in a list of local
vector stores (`v1/docs/Explorations/v1/Chats/ChatGPT-Nash Equilibrium in Dynamics.md`, line 1953).
The DuckDB work tracked in [#191](https://github.com/GuitarAlchemist/ix/issues/191) and
[#199](https://github.com/GuitarAlchemist/ix/issues/199) is therefore IX-originated and TARS-V2
facing; it draws nothing from this corpus and should not claim to.

## Band B — TARS-core, explicitly not IX candidates

Separating these out is an acceptance criterion of #190. The following themes are large in the
corpus and are **not** IX candidates, because IX owns reusable algorithms while TARS owns
orchestration, runtime and agent contracts:

| Theme | Representative documents | Why it stays in TARS |
| --- | --- | --- |
| Metascript / FLUX runtime | `v1/TARS_INSTRUCTION_FORMAT_SPECIFICATION.md`, `v1/docs/FLUX_Fractal_Language_Architecture.md`, `v1/docs/DSL/**` | A language runtime and execution contract, not an algorithm |
| Agent organisation & swarm | `v1/organization/**`, `.tars/workspace/plans/todos/TODOs-TARS-Swarm.md` | Agent registry and org topology are TARS contracts |
| MCP / connectivity | `v1/docs/MCP_INTEGRATION.md`, `v1/docs/MCP_SERVERS.md` | Already solved ecosystem-wide via `.mcp.json` federation |
| C#→F# migration | ~40 files under `v1/TODOs/TarsEngine_FSharp_*` | Pure TARS project history |
| Deployment / infra | Docker, Kubernetes, Hyperlight, Windows-service docs | Infrastructure, no algorithmic content |
| WebGPU visualisation | `v1/TODOs/webgpu-logistic-map-master-todo.md` + its 10 numbered sub-TODOs | Borderline: the logistic-map mathematics is already covered by `ix-chaos` and `ix-fractal`; what remains is a browser rendering project, which IX does not own |
| Self-improvement narrative | `v1/*_SUPERINTELLIGENCE_*.md`, `v1/TARS-Full-Superintelligence-Achievement.md` | Claims without oracles; band C in substance |

`v1/docs/Explorations/v1/Chats/` additionally contains ~40 chats on TTS engines, Blazor/MudBlazor,
Ollama/ONNX setup, eBPF, Docker, sci-fi trivia and similar. These are neither IX nor TARS-core
technical assets and are not itemised.

## Follow-up story linkage

The table below maps existing follow-up work to the documents that justify it. Most entries are
descendants of epic [#189](https://github.com/GuitarAlchemist/ix/issues/189); #203 and #204 are
nested through #202, while #205 is a sibling epic related to #189. Candidates without a dedicated
story remain explicitly marked `none` or point only to the parent epic. This inventory does not
propose new issues.

| Story | Scope | Candidates feeding it |
| --- | --- | --- |
| [#191](https://github.com/GuitarAlchemist/ix/issues/191) | DuckDB + IX local analytics | *(none — see the `duckdb` section above)* |
| [#192](https://github.com/GuitarAlchemist/ix/issues/192) | Tree-of-Thought / Workflow-of-Thought extraction | `docs/MetascriptTreeOfThought.md`, `TODOs-Tree-of-Thought-Auto-Improvement-Pipeline-Detailed.md` |
| [#193](https://github.com/GuitarAlchemist/ix/issues/193) | State-space & control metrics for agent loops | `ChatGPT-State-Space for TARS.md` |
| [#194](https://github.com/GuitarAlchemist/ix/issues/194) | Advanced math candidates | Vector decomposition, primes/hypergraph, Nash dynamics, quaternions, Pauli, multi-modal memory |
| [#195](https://github.com/GuitarAlchemist/ix/issues/195) | Music/guitar algorithm candidates | `harmonic_progression_analyzer.tars.md`, `GUITAR_ALCHEMIST_TARS_INTEGRATION_PLAN.md` |
| [#200](https://github.com/GuitarAlchemist/ix/issues/200) | IX analysis reports for TARS closure runs | `v1_component_reusability_analysis.md` |
| [#201](https://github.com/GuitarAlchemist/ix/issues/201) | IX pipeline contract for TARS V2 artifacts | `v1_component_reusability_analysis.md`, `v1_reuse_strategy.md` |
| [#202](https://github.com/GuitarAlchemist/ix/issues/202) | Advanced-math salvage gap matrix | `MISSED_OPPORTUNITIES_ANALYSIS.md`, `ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md` |
| [#203](https://github.com/GuitarAlchemist/ix/issues/203) | Expose ix-fractal Takagi / de Rham primitives | `ChatGPT-Courbes Takagi et Rham.md` |
| [#204](https://github.com/GuitarAlchemist/ix/issues/204) | Fractal mutation operators for ix-evolution | `TARS_FRACTAL_GRAMMARS_README.md`, `UNIFIED_GRAMMAR_EVOLUTION_GRANULAR_TASKS.md`, `ChatGPT-Genetic Memory and Vectors.md` |
| [#205](https://github.com/GuitarAlchemist/ix/issues/205) | IX use-case portfolio | `intelligence-progression-measurement.md` (as an anti-pattern) |

## Correction to the previous revision

The first revision of this file (task [#196](https://github.com/GuitarAlchemist/ix/issues/196))
inventoried **F# source files**, not the Markdown corpus #190 asks for, and cited them as bare
filenames. Re-resolving those 21 filenames against the pinned checkout:

- **7 exist**, though not always where implied: `TarsSedenionPartitioner.fs` and `TrsxHypergraph.fs`
  (`v1/src/TarsEngine.FSharp.Core/`), `AgenticTraceCapture.fs` and `HyperComplexGeometricDSL.fs`
  (`src/TarsEngine.FSharp.Core/`), and `WeightedGrammar.fs`, `ReplicatorDynamics.fs`,
  `MctsBridge.fs` — which are **V2** files under `v2/src/Tars.Evolution/`, not V1.
- **14 do not exist anywhere in the repository**: `VectorSignificance.fs`, `GraphKTheory.fs`,
  `QStarHeuristics.fs`, `PluckerLine.fs`, `Grothendieck.fs`, `SetTheory.fs`, `TarsDuckBridge.fs`,
  `Scorecard.fs`, `NeuralOde.fs`, `ToTReasoner.fs`, `WoTDerivation.fs`, `Hurwitz.fs`,
  `CudaKernels.fs`, `FluxDsl.fs`.

Several of the missing names match module names *proposed inside* the primes transcript rather than
anything ever written, which is how they entered the inventory. That revision also omitted
`candidate_type`, `priority`, `complexity` and follow-up linkage, and used a `Free`/`Paid` cost
vocabulary instead of the four tiers #190 specifies. This revision replaces it entirely.

## Limitations

- Long transcripts were read in their high-signal sections, not end to end; a claim buried in the
  tail of a 6,000-line chat could have been missed. Documents where this risk is material are marked
  `cheap-hosted`, meaning a targeted extraction pass is the honest next step.
- No claim in any source document has been validated. `testability: high` means *an oracle is
  available*, not *the claim holds*.
- Band C/D/E membership is assigned by filename and directory rule, not by reading each file, so
  individual files could be mis-banded — a document with a `*_SUMMARY.md` name could in principle
  contain real content. The counts themselves are exact. Band A was assembled by reading.
