# TARS V1 Advanced-Math Salvage Gap Matrix (IX)

Issue: [GuitarAlchemist/ix#202](https://github.com/GuitarAlchemist/ix/issues/202) — parent epic
[#189](https://github.com/GuitarAlchemist/ix/issues/189). Builds on the corrected inventory in
[`tars-v1-exploration-inventory.md`](tars-v1-exploration-inventory.md) (#190).

This matrix answers one question per row: **for this advanced-math technique that TARS V1 proposed,
what does IX have today, and is the gap an algorithm or an exposure?** It is a salvage assessment,
not an adoption decision. Nothing here has been ported, and no claim from the TARS source text is
treated as validated.

## Evidence base

| Item | Value |
| --- | --- |
| TARS source repository | `GuitarAlchemist/tars` (local sibling checkout `../tars`) |
| Pinned ref | `69cf427eccb25514728eacfd3530218df3975259` |
| How TARS paths were resolved | `git cat-file -e <SHA>:<path>` per path, plus one full `git ls-tree -r <SHA>` (15,090 paths) for basename searches |
| IX side | this worktree, branch `codex/ix-202-math-gap-matrix`, off `main` @ `4331cf7` |
| IX crate count | 82 directories under `crates/` |
| Survey date | 2026-09-07 |
| Cost incurred | 0 USD — local file reads only (`free-local`) |

### Verification rule

Every `source_doc` in this matrix was proven to exist at the pinned SHA **before** the row was
written, and every `current_ix_surface` cites a path in this worktree that was opened and read. Where
a candidate's source could not be located, the row says so explicitly rather than being dropped —
see [Unresolved and not reached](#unresolved-and-not-reached). Trusting a prior list is what produced
the phantom rows the previous inventory had to retract, so no list here — including #190's — was
taken on faith.

### Source-document existence check

All documents named by #202 and by the #190 band-A math set resolve at the pinned SHA. Line counts
are recorded because they independently corroborate #190's own figures — every one matched.

| source_doc (relative to tars root) | Exists @ SHA | Lines |
| --- | --- | ---: |
| `v1/MISSED_OPPORTUNITIES_ANALYSIS.md` | yes | 218 |
| `v1/ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md` | yes | 248 |
| `v1/COMPREHENSIVE_MATHEMATICAL_LEVERAGE_STRATEGY.md` | yes | 251 |
| `v1/docs/Explorations/v1/Chats/ChatGPT-Courbes Takagi et Rham.md` | yes | 1945 |
| `v1/docs/Explorations/v1/Chats/ChatGPT-State-Space for TARS.md` | yes | 443 |
| `v1/docs/Explorations/v1/Chats/ChatGPT-Advanced Math CS 2025.md` | yes | not measured |
| `v1/docs/Explorations/v1/Chats/ChatGPT-Quaternions and Projective Geometry for TARS.md` | yes | 4858 |
| `v1/docs/Explorations/v1/Chats/ChatGPT-Nash Equilibrium in Dynamics.md` | yes | 6406 |
| `v1/docs/Explorations/v1/Chats/ChatGPT-TARS Multi-modal Memory Space.md` | yes | 844 |
| `v1/ChatGPT-Leveraging Primes for TARS (1).md` | yes | 6993 |
| `v1/ChatGPT-Vector Decomposition for TARS.md` | yes | 1513 |
| `v1/PAULI_MATRICES_EXPLORATION_FOR_TARS.md` | yes | 300 |
| `v1/TARS_FRACTAL_GRAMMARS_README.md` | yes | 330 |
| `v1/TODOs/CHATGPT_LEVERAGING_PRIMES_ROADMAP.md` | yes | 254 |

`COMPREHENSIVE_MATHEMATICAL_LEVERAGE_STRATEGY.md` is named by #202 but is **absent from #190's
band-A list**; it is covered here for the first time, in section F and in the closure-name finding
below.

`ChatGPT-Advanced Math CS 2025.md` is cited by `MISSED_OPPORTUNITIES_ANALYSIS.md` as the source for
six of its named gaps and is likewise absent from #190's band-A list. It exists at
`v1/docs/Explorations/v1/Chats/ChatGPT-Advanced Math CS 2025.md`. Rows sourced to it are attributed
through `MISSED_OPPORTUNITIES_ANALYSIS.md`, which is the document actually read — see
[Unresolved and not reached](#unresolved-and-not-reached).

### Phantom re-check, independent of #190

#190's central correction is that the revision before it cited 14 non-existent F# files. That claim
was **re-verified here from scratch**, by extracting the complete path list at the pinned SHA and
searching it by basename rather than by trusting #190's table:

- **14 absent, confirmed:** `VectorSignificance.fs`, `GraphKTheory.fs`, `QStarHeuristics.fs`,
  `PluckerLine.fs`, `Grothendieck.fs`, `SetTheory.fs`, `TarsDuckBridge.fs`, `Scorecard.fs`,
  `NeuralOde.fs`, `ToTReasoner.fs`, `WoTDerivation.fs`, `Hurwitz.fs`, `CudaKernels.fs`, `FluxDsl.fs`.
  Zero hits across all 15,090 paths. **No row in this matrix cites any of them.**
- **7 present, confirmed**, with one refinement to #190: `AgenticTraceCapture.fs` and
  `HyperComplexGeometricDSL.fs` exist under **both** `v1/src/TarsEngine.FSharp.Core/` and the
  repository-root `src/TarsEngine.FSharp.Core/`. #190 lists only the root copy, which understates
  their V1 provenance. `TarsSedenionPartitioner.fs` and `TrsxHypergraph.fs` are V1
  (`v1/src/TarsEngine.FSharp.Core/`); `WeightedGrammar.fs`, `ReplicatorDynamics.fs` and
  `MctsBridge.fs` are V2 (`v2/src/Tars.Evolution/`), as #190 states.

### A second phantom layer: closure names

The previous retraction was about *filenames*. Verifying #202's own three named strategy documents
turned up the same failure mode one level down, in *function* names — and it changes how those three
documents must be read.

`COMPREHENSIVE_MATHEMATICAL_LEVERAGE_STRATEGY.md` asserts Phase 1 is `✅ COMPLETED` with "**23
advanced mathematical closures** implemented and accessible", and both it and
`ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md` write F# samples against named factory functions. At
the pinned SHA:

| Closure name asserted by the strategy docs | Files containing it @ SHA |
| --- | ---: |
| `createTopologicalPatternDetector` | 0 |
| `createFractalNoiseGenerator` | 0 |
| `createFractalPerturbationOptimizer` | 0 |
| `createLieAlgebraInterpolator` | 0 |
| `createKalmanFilter` | 0 |
| `createAgentMPCController` | 0 |
| `createTopologicalStabilityAnalyzer` | 3 |
| `StateSpaceOptimalAgentAssignment` | 2 |

The real factory is
`TarsEngine.FSharp.WindowsService/ClosureFactory/AdvancedMathematicalClosureFactory.fs` (1,868
lines). It exposes **20** `let create*` entry points, not 23:

`createSupportVectorMachine`, `createRandomForest`, `createAttentionMechanism`,
`createBifurcationAnalyzer`, `createChaosAnalyzer`, `createLieAlgebraStructure`,
`createLieGroupAction`, `createTransformerBlock`, `createVariationalAutoencoder`,
`createGraphNeuralNetwork`, `createPauliMatrices`, `createPauliMatrixOperations`,
`createQuantumStateEvolution`, `createBloomFilter`, `createCountMinSketch`, `createHyperLogLog`,
`createCuckooFilter`, `createProbabilisticDataStructures`, `createGraph`,
`createGraphTraversalAlgorithms`.

**None of them is Takagi, de Rham, Kalman, MPC, or persistent homology.** The file contains no
occurrence of `takagi`, `rham`, `kalman`, `mpc`, `homology` or `persistent` at all — only a
`lyapunov` exponent inside `createChaosAnalyzer`. Those five are precisely the techniques
`COMPREHENSIVE_MATHEMATICAL_LEVERAGE_STRATEGY.md` marks `✅ COMPLETED` in Phase 1.

Likewise, of the five integration *target* files the two strategy documents name, only one exists:

| Target file named by the strategy docs | Exists @ SHA |
| --- | --- |
| `TarsEngine.FSharp.Agents/AgentOrchestrator.fs` | yes (plus a `v1/parked_legacy/` copy) |
| `TarsEngine.FSharp.Core/Analysis/CodeAnalyzerService.fs` | **no** |
| `TarsEngine.FSharp.Core/LLM/AutonomousReasoningService.fs` | **no** |
| `TarsEngine.FSharp.Core/Projects/AutonomousProjectService.fs` | **no** |
| `TarsEngine.FSharp.Core/ChromaDB/HybridRAGService.fs` | **no** |

**Consequence for this matrix:** the three strategy documents are usable as a *checklist of technique
names to consider*, and nothing more. Their status markers (`✅`/`🔄`/`📋`), their percentage impact
claims — all of the form "70-90% improvement", none with a measurement method — and their "already
implemented" assertions are not evidence, and are carried into no row. Where a row below says
`already_done`, that rests on an IX file read in this worktree, never on the TARS text.

## How to read the matrix

Columns are those requested by #202. Conventions:

- **`current_ix_surface`** cites a path in this worktree. `—` means nothing was found; that is a claim
  about this survey's greps, and the patterns are recorded in the row notes and in
  [Method](#method-and-its-limits) so every negative can be re-run.
- **`status`** uses #202's vocabulary: `already_done`, `implemented_needs_exposure`,
  `partially_implemented`, `missing_algorithm`, `research_only`, `defer_v3`, `reject`.
  `implemented_needs_exposure` means the algorithm exists and is under test in a crate but is not
  reachable from DuckDB, a skill verb, or an MCP tool. `partially_implemented` means part of the
  *algorithm* is missing, not just its exposure.
- **exposure columns** record the surface as of this commit: `duckdb` = a UDF registered in
  `ix-duck`/`ix-duck-ext`; `skill` = a verb under `crates/ix-skill/src/verbs/`; `pipeline` = a stage
  reachable from `ix-pipeline`. The authoritative MCP list is the 96-entry `EXPECTED` array in
  `crates/ix-agent/tests/parity.rs`; MCP presence is noted in `notes`, since #202's column set has no
  MCP column.
- **`testability`** means *an oracle is available*, not *the claim holds*.

## The matrix

### A. State-space and control theory

Source: `v1/MISSED_OPPORTUNITIES_ANALYSIS.md` §1 (its Priority 1), amplified by
`v1/docs/Explorations/v1/Chats/ChatGPT-State-Space for TARS.md`. Feeds
[#193](https://github.com/GuitarAlchemist/ix/issues/193).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Linear state-space model `x_{k+1}=Ax_k+Bu_k+w_k` | ix-signal | `crates/ix-signal/src/state_space.rs` — `StateSpaceModel::{new,from_kalman,step,output,simulate}` | ~~missing_algorithm~~ → **partially_implemented** (corrected 2026-09-08), now **already_done** | prototype | P1 | S | high | no | no | no | **Status corrected on delivery.** The original `missing_algorithm` label overstated the gap: `crates/ix-signal/src/kalman.rs` already carried `A` (`transition`), `B` (`control`) and `C` (`observation`) as public fields, and `predict` already computed `Ax + Bu`. The real deficits were (i) no *reusable* value — the matrices could not be handed to another algorithm without dragging the estimator's mutable `state`/`covariance` along, (ii) no dimension validation, and (iii) no plant *simulation*: `predict` propagates the state **estimate** and its covariance, and `Q` is used as an uncertainty and never sampled, so the `w_k` term of the equation as literally written was genuinely absent. So A1 was mostly exposure plus a thin rollout, not a missing algorithm. **Name-collision trap (confirmed).** `crates/ix-graph/src/state_space.rs` is *not* this: it defines `trait State`, `astar()` and `beam_search()` — a discrete search space, not an LTI system. |
| A2 | Non-linear state-space `x_{k+1}=f(x_k,u_k)` | ix-dynamics | `crates/ix-dynamics/src/neural_ode.rs` | partially_implemented | prototype | P2 | M | medium | no | no | no | Neural ODE covers the *continuous-time* learned-`f` case. The discrete-time analytic-`f` case A1 needs is absent. Build A1 first; A2 is then a thin wrapper. **A1 landed 2026-09-08**, so A2 is now unblocked: it generalises `StateSpaceModel::simulate` from `A x + B u` to a caller-supplied `f(x, u)`. The rank tests do NOT generalise with it — the non-linear analogues are local (Jacobian-linearised) or Lie-bracket based, which is new work, not a wrapper. |
| A3 | Kalman filter | ix-signal | `crates/ix-signal/src/kalman.rs` — `KalmanFilter::{new,predict,update,step,filter}`, `constant_velocity_1d` | implemented_needs_exposure | document | P1 | XS | high | **yes** (`ix_kalman_smooth`) | no | no | The clearest "TARS listed it as missed, IX already had it" row. No MCP tool — `ix_kalman*` is absent from the 96-entry `EXPECTED` list — and no skill verb, so it is unreachable from an agent loop, which is exactly the consumer #193 wants. |
| A4 | Model Predictive Control | new | — | missing_algorithm | prototype | P2 | L | medium | no | no | no | Nothing matches `model_predictive` or `\bmpc\b` in any crate. `crates/ix-chaos/src/control.rs` is chaos control (`ogy_control`, `pyragas_control`, `drive_response_sync`) — a different problem. Needs A1 plus a QP solver; `ix-optimize` is the natural host. Highest-cost row in section A; do not start before A1/A5 land — **both landed 2026-09-08**, so the remaining blocker is the QP solver. |
| A5 | Observability & controllability analysis | ix-signal | `crates/ix-signal/src/state_space.rs` — `{controllability,observability}{,_matrix}`, `RankReport` | ~~missing_algorithm~~ → **already_done** (delivered 2026-09-08) | prototype | P1 | XS | high | no | no | no | **Gap confirmed before building, then closed.** All grep hits for `observability` were indeed `ix-governance`/`ix-context` telemetry, unrelated. Shipped as Kalman rank tests over `ix_math::svd`, with a LAPACK-convention relative tolerance and a `margin` (smallest singular value) so a *nearly* unobservable system is distinguishable from a healthy one. **Popov–Belevitch–Hautus was NOT shipped**: PBH needs the eigenvalues of a general non-symmetric `A`, and `ix-math` exposes only `symmetric_eigen`. It stays open as a follow-up behind a general eigensolver. |
| A6 | Lyapunov stability analysis | ix-chaos | `crates/ix-chaos/src/lyapunov.rs` | partially_implemented | prototype | P2 | M | high | no | no | no | **Two different Lyapunovs.** IX has the Lyapunov *exponent*, a chaos diagnostic exposed as MCP `ix_chaos_lyapunov`. TARS asks for a Lyapunov *function* / stability certificate for a state-space model — solving `AᵀPA − P = −Q` for `P ≻ 0`. The discrete Lyapunov equation is testable against known stable/unstable `A`. Depends on A1, **which landed 2026-09-08** — `StateSpaceModel::a()` is the input it needs. |
| A7 | State-space agent orchestration (Kalman + MPC + Lyapunov over agent state) | ix-agent | — | missing_algorithm | defer | P3 | XL | low | no | no | no | Composite of A1/A3/A4/A6, and the TARS side is unbuilt: `StateSpaceOptimalAgentAssignment` appears only in `AgentOrchestrator.fs`, whose companion closures (`createAgentMPCController`, `createKalmanFilter`) resolve to 0 files. The hard part is not the math but the missing oracle — "80-95% coordination efficiency" has no measurement method anywhere in the corpus. Defer until A1–A6 exist and a real agent-loop metric exists to move. |

### B. Techniques from the "2025 state-of-art" list

Source: `v1/MISSED_OPPORTUNITIES_ANALYSIS.md` §2, which attributes them to
`ChatGPT-Advanced Math CS 2025.md`. Feeds [#194](https://github.com/GuitarAlchemist/ix/issues/194).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Topological data analysis / persistent homology | ix-topo | `crates/ix-topo/src/persistence.rs` — `PersistenceDiagram`, `compute_persistence`, `bottleneck_distance`, `wasserstein_distance`; plus `simplex.rs`, `pointcloud.rs` | already_done | document | P1 | XS | high | no | no | no | MCP `ix_topo` exists. TARS lists this among its top-three misses; IX has had it, with both diagram metrics. The only real gap is DuckDB — see B1b. |
| B1b | Persistent homology over a DuckDB column | ix-duck | — | implemented_needs_exposure | prototype | P2 | S | high | no | no | no | Follows the shape #203 proved for `ix_takagi`. Non-trivial only because a persistence diagram is not a scalar — needs a `LIST(STRUCT(dim,birth,death))` return, or a pairwise `ix_bottleneck(a,b)` scalar form. The scalar form is the cheap version. |
| B2 | Category theory (composability) | ix-category | `crates/ix-category/src/core.rs` — `trait Category`, `trait Functor`, `trait NaturalTransformation`, `trait Monoidal`, `ComposedFunctor`; `monad.rs`, `instances.rs` | already_done | document | P2 | XS | high | no | no | no | MCP `ix_category` exists. Nothing to salvage: IX's version is more complete than anything the TARS text describes. |
| B3 | Homotopy type theory (formal verification) | — | — | reject | document | P3 | XL | low | no | no | no | HoTT is a foundation for a proof assistant, not a crate. IX's verification story is executable oracles (`cargo test`, the clippy gate, sentrux) per CLAUDE.md's `certainty := strength of live binding` rule. Adopting HoTT would need a dependently-typed host language IX does not have. Rejected on scope, not merit. The lone grep hit (`crates/ix-duck/src/loops.rs`) is the word "homotopy" in a comment. |
| B4 | Neural differential equations | ix-dynamics | `crates/ix-dynamics/src/neural_ode.rs` | implemented_needs_exposure | document | P2 | S | high | no | no | no | No `ix_dynamics`/`ix_neural_ode` MCP tool, no UDF, no skill verb. The whole `ix-dynamics` crate (`lie.rs`, `ik.rs`, `neural_ode.rs`) is agent-invisible — see the exposure-cluster note in the summary. |
| B5 | Fourier & spectral methods | ix-signal | `crates/ix-signal/src/{fft,dct,wavelet,spectral,filter,window,convolution,correlation,timeseries,sampling}.rs` | already_done | document | P1 | XS | high | **yes** (`ix_rfft`, `ix_dct`, `ix_autocorrelation`, `ix_wavelet_denoise`) | no | no | MCP `ix_fft`, `ix_spectrogram`, `ix_fir_filter`, `ix_wavelet_denoise`, `ix_spectral_distance`. Best-covered row in the matrix across all exposure surfaces. |
| B6 | Causal inference / counterfactual reasoning | new | — | missing_algorithm | benchmark | P2 | L | medium | no | no | no | **Genuine gap, and the grep is a trap:** the only `causal` hits in the workspace are `causal` *attention masks* in `crates/ix-nn/{attention,transformer,positional}.rs`, which is unrelated. No SCM, no do-calculus, no backdoor/frontdoor adjustment, no instrumental variables anywhere. IX has a plausible consumer — `ix-quality-trend` asks counterfactual questions about metric movements — which is why this is `benchmark` and not `defer`. |
| B7 | Algebraic & Lie-theoretic structures | ix-dynamics | `crates/ix-dynamics/src/lie.rs` — `so3_bracket`, `hat`, `vee`, `so3_exp`, `so3_log`, `se3_exp`, `su2_from_quaternion`, `quaternion_from_su2` | implemented_needs_exposure | document | P2 | XS | high | no | no | no | TARS wanted "Lie algebra operations for smooth state transitions"; IX has the SO(3)/SE(3)/SU(2) core already. Same invisibility as B4. |
| B8 | Cryptographic innovations / zero-knowledge proofs | — | — | reject | document | P3 | XL | low | no | no | no | No `zk`, `snark` or `zero_knowledge` anywhere, and no IX consumer: IX's crates are ML/math primitives plus governance, and governance's trust model is auditable JSON-on-disk with human merge gates, not privacy-preserving proof. Rejected on #202's non-goal "do not add esoteric algorithms without tests and a consumer". |

### C. Fractal mathematics and geometry

Sources: `v1/MISSED_OPPORTUNITIES_ANALYSIS.md` §3,
`v1/docs/Explorations/v1/Chats/ChatGPT-Courbes Takagi et Rham.md`,
`v1/TARS_FRACTAL_GRAMMARS_README.md`. Feeds
[#203](https://github.com/GuitarAlchemist/ix/issues/203) (closed) and
[#204](https://github.com/GuitarAlchemist/ix/issues/204).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Takagi / Blancmange curves | ix-fractal | `crates/ix-fractal/src/takagi.rs` — `takagi`, `takagi_series` | already_done | document | P1 | XS | high | **yes** (`ix_takagi`) | no | no | Delivered by #203 (closed; merged as PR #286, commit `db60ec9`). MCP `ix_fractal` op `takagi`. Closed-form values make this exactly testable, which is why #203 was cheap. |
| C2 | de Rham interpolatory curves | ix-fractal | `crates/ix-fractal/src/de_rham.rs`; MCP `ix_fractal` op `de_rham_1d` (`crates/ix-agent/src/handlers.rs:1740`) | partially_implemented | prototype | P2 | XS | high | **no** | no | no | #203 shipped `ix_takagi` as a UDF, but de Rham reached only MCP, bounded by `DE_RHAM_MAX_DEPTH` (2^12+1 samples). The residual `ix_de_rham` UDF is the smallest genuine exposure gap in the matrix. Status is `partially_implemented` rather than `implemented_needs_exposure` because #203 is closed, so the residue needs a new issue to stay visible. |
| C3 | **`takagi.rs` / `de_rham.rs` duplicated verbatim into `ix-chaos`** | ix-fractal / ix-chaos | `crates/ix-chaos/src/takagi.rs`, `crates/ix-chaos/src/de_rham.rs` | already_done (with defect) | document | P2 | XS | high | n/a | n/a | n/a | **Not a TARS candidate — an IX finding surfaced while verifying one.** Both files are byte-identical to their `ix-fractal` counterparts except a single doc-comment line (`//! use ix_fractal::takagi;` → `//! use ix_chaos::takagi;`); 197 lines on both sides. `ix-chaos/Cargo.toml` does not depend on `ix-fractal`, so this is a copy, not a re-export. Two risks follow: a fix to one is invisible to the other, and the stable-surface gate hashes `pub `-prefixed lines per crate, so a signature change must be made twice. It also means #203's "expose ix-fractal" framing answered half the question. |
| C4 | Dual quaternions | ix-math / ix-rotation | `crates/ix-math/src/dual_quaternion.rs` (466 L) **and** `crates/ix-rotation/src/dual_quaternion.rs` (139 L) | already_done | document | P2 | XS | high | no | no | no | MCP `ix_rotation`. Unlike C3, these are genuinely *different* implementations (466 vs 139 lines; 554 diff lines) — divergent parallel work rather than a copy. Same pattern for Plücker: `ix-math/src/plucker.rs` (264 L) vs `ix-rotation/src/plucker.rs` (156 L). Consolidation is IX hygiene, not TARS salvage; recorded so C7/D7 do not build on the wrong one. |
| C5 | Fractal mutation operators (Takagi-noise perturbation) | ix-evolution | `crates/ix-evolution/src/{genetic,differential,selection,pareto}.rs` — no fractal operator | missing_algorithm | prototype | P1 | S | high | no | no | no | Both halves exist and are unjoined: `ix-fractal::takagi_series` produces the structured noise, `ix-evolution` owns the mutation traits. Grep for `fractal_mutat`/`takagi_mutat` returns nothing. This is [#204](https://github.com/GuitarAlchemist/ix/issues/204)'s core and the strongest `missing_algorithm` here, because the composition is small and the oracle is real: seeded fractal-vs-Gaussian mutation on a fixed multi-modal objective. |
| C6 | Recursive crossover (de Rham-style blending) | ix-evolution | — | missing_algorithm | prototype | P2 | S | medium | no | no | no | Sibling of C5, same #204. Lower testability: "smoother genetic blending" has no obvious oracle beyond convergence-curve comparison, so it should ride on C5's benchmark harness rather than get its own. |
| C7 | Fractal grammars with computed fractal dimension of production rules | ix-grammar × ix-chaos | `crates/ix-grammar/src/{ebnf,abnf,weighted,constrained,catalog,replicator}.rs`; `crates/ix-chaos/src/fractal.rs` — `box_counting_dimension_2d`, `correlation_dimension`, `hurst_exponent` | missing_algorithm | prototype | P2 | M | high | partial (`ix_hurst`) | no | no | Source is the most implementable document in the grammar set (`TARS_FRACTAL_GRAMMARS_README.md`, 330 L — a specification, not a transcript). Fractal dimension of a self-similar production rule is deterministic and unit-testable. Both ingredient crates exist; the join does not. TARS's multi-format emitters (ANTLR/GraphViz/SVG) are TARS tooling and must **not** be ported. |
| C8 | Fractal memory organisation / fractal indexing | — | — | research_only | document | P3 | L | low | no | no | no | Named in both strategy documents with a "30-50% retrieval efficiency" claim and no definition of the index, no baseline, no measurement method. Its stated target file `TarsEngine.FSharp.Core/ChromaDB/HybridRAGService.fs` does not exist at the pinned SHA. Record the idea; build nothing. |

### D. Hypercomplex, projective, and vector-decomposition candidates

Sources: `v1/ChatGPT-Leveraging Primes for TARS (1).md`, `v1/ChatGPT-Vector Decomposition for TARS.md`,
`v1/docs/Explorations/v1/Chats/ChatGPT-Quaternions and Projective Geometry for TARS.md`,
`v1/docs/Explorations/v1/Chats/ChatGPT-TARS Multi-modal Memory Space.md`,
`v1/PAULI_MATRICES_EXPLORATION_FOR_TARS.md`. Feeds
[#194](https://github.com/GuitarAlchemist/ix/issues/194).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | BSP partitioning generalised to sedenion (R^16) space | ix-sedenion | `crates/ix-sedenion/src/bsp.rs` — `BspNode::{build,query_nearest,query_radius}`; `cayley_dickson.rs`, `octonion.rs`, `sedenion.rs`; also `crates/ix-math/src/bsp.rs` | already_done | document | P2 | XS | high | no | no | no | MCP `ix_sedenion`. This is the *only* separable idea in the 6,993-line primes transcript (#190 locates it at lines ~746-1030), and IX already has it. The prime-number framing around it is decorative, and the transcript's opening exchange misidentifies TARS entirely — flagged by #190, and not relied on here. |
| D2 | Directed hypergraph of `.trsx` metascript diffs | ix-graph | `crates/ix-graph/src/{graph,markov,hmm,routing,state_space}.rs` — no hypergraph | missing_algorithm | defer | P3 | L | low | no | no | no | Grep for `hypergraph` returns nothing under `crates/`. TARS's own `TrsxHypergraph.fs` **does** exist (`v1/src/TarsEngine.FSharp.Core/TrsxHypergraph.fs`), so this is not a phantom — but its input is `.trsx` metascript diffs, a TARS-runtime artifact IX has no access to and no consumer for. Deferred on missing fuel, not missing math. CLAUDE.md forbids adding `petgraph`/`daggy`, so any hypergraph must be built on `ix-graph`. |
| D3 | Vector decomposition: PCA / SVD / eigendecomposition | ix-math | `crates/ix-math/src/{svd,eigen,linalg}.rs` | already_done | document | P1 | XS | high | **yes** (`ix_pca_project`, `ix_mds_project`, `ix_tsne_project`) | no | no | MCP `ix_pca`, `ix_svd`, `ix_eigen`, `ix_tsne`. |
| D4 | Vector decomposition: sparse dictionary learning | ix-optick-sae | `crates/ix-optick-sae/src/{lib,trainer}.rs` — SAE trainer plus `SaeArtifact` schema | already_done | document | P2 | S | high | no | no | no | A sparse autoencoder *is* the learned-dictionary case, and it is already contracted cross-repo (`ga/docs/contracts/2026-05-02-optick-sae-artifact.contract.md`). The TARS document proposes this as future work; IX shipped it. |
| D5 | Vector decomposition: hyperbolic / manifold embeddings | ix-math | `crates/ix-math/src/{hyperbolic,poincare_hierarchy,geometric_space}.rs` | already_done | document | P2 | XS | high | no | no | no | **`ix-manifold` is narrower than its name suggests** — its entire public surface is `Tsne` and `BarnesHutTsne` (`crates/ix-manifold/src/lib.rs`). The hyperbolic content #202's crate list implies lives in `ix-math`. Recorded so a future reader does not search the wrong crate. |
| D6 | Vector decomposition: tensor decomposition (Tucker / CP / PARAFAC) | new | — | missing_algorithm | benchmark | P2 | M | high | no | no | no | The one genuinely missing branch of the vector-decomposition survey: no `tucker`, `parafac` or `cp_als` anywhere. Testable — reconstruct a synthetic low-rank tensor to a known error — and it has a live consumer in the OPTIC-K voicing corpus, which is why #190 rates the source its strongest benchmark candidate. |
| D7 | Homogeneous / projective coordinates, Plücker lines | ix-math / ix-rotation | `crates/ix-math/src/plucker.rs`, `crates/ix-rotation/src/plucker.rs`, `crates/ix-math/src/geometric_space.rs` | already_done | document | P2 | XS | high | no | no | no | The machinery is standard and present. The *unvalidated* part of the source is its assumption that cross-modal embeddings are related by a linear map — document, do not build on it. See C4 on the duplication. |
| D8 | Pauli matrices and the associated algebra | ix-dynamics | `crates/ix-dynamics/src/lie.rs:262` — `pauli_matrices()`, with `Complex` and SU(2) ↔ quaternion conversions | already_done | document | P3 | XS | high | no | no | no | **IX already has the only defensible half.** #190 gives the source a split verdict — correct algebra, unfalsifiable cognitive framing — and IX's copy sits in `lie.rs` where it belongs (SU(2) generators), with no agent-superposition narrative attached. Nothing to salvage; do not import the framing. |
| D9 | Quantum state evolution (Hamiltonian / unitary) | — | — | reject | document | P3 | M | medium | no | no | no | TARS's `createQuantumStateEvolution` is one of the 20 closures that genuinely exists. Rejected on the same grounds as D8's framing: no IX consumer, and the motivating claim ("superposition of agent capabilities") has no oracle. The lone `quantum` grep hit is `crates/ix-harness-signing/` (post-quantum signature naming), unrelated. |

### E. Game theory, graph, and ML closures

Sources: `v1/docs/Explorations/v1/Chats/ChatGPT-Nash Equilibrium in Dynamics.md`, and the 20 verified
closures in `AdvancedMathematicalClosureFactory.fs`. Feeds
[#194](https://github.com/GuitarAlchemist/ix/issues/194).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Static Nash equilibrium | ix-game | `crates/ix-game/src/nash.rs` — `BimatrixGame`, `fictitious_play`, `dominant_strategy_equilibrium`; plus `mean_field.rs`, `cooperative.rs`, `mechanism.rs`, `auction.rs`, `evolutionary.rs` | already_done | document | P2 | XS | high | no | no | no | MCP `ix_game_nash`. |
| E2 | Differential games / HJB–HJI, open- vs closed-loop equilibria | ix-game | — | missing_algorithm | defer | P3 | XL | low | no | no | no | A real gap — no `hamilton_jacobi`, `hjb` or `differential_game` anywhere — but the source is the longest document surveyed (6,406 lines), which #190 rates "mostly tangent by the end". Numerically certifying a differential-game equilibrium does not reduce to a fast unit test, which is what makes this `defer` rather than `prototype`. Revisit only if a concrete IX consumer appears. |
| E3 | Graph neural networks | new | — | missing_algorithm | defer | P3 | L | medium | no | no | no | `createGraphNeuralNetwork` is a real TARS closure; IX has no `gnn`/`message_passing` anywhere. Deferred because IX's graph work is spectral/combinatorial (`ix-graph`, `ix-ktheory`, `ix-topo`) and a GNN drags in a training stack — and because CLAUDE.md's "check prior art first" rule points at composing `ix-nn` with `ix-graph` before adding a crate. |
| E4 | Bifurcation analysis | ix-chaos | `crates/ix-chaos/src/bifurcation.rs` | already_done | document | P3 | XS | high | no | no | no | Named in `ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md` for "critical project decision points". |
| E5 | Chaos analysis / Lyapunov exponent | ix-chaos | `crates/ix-chaos/src/{lyapunov,attractors,embedding,poincare_map}.rs` | already_done | document | P2 | XS | high | **yes** (`ix_hurst`) | no | no | MCP `ix_chaos_lyapunov`. See A6 for why this does **not** close the Lyapunov-stability row. |
| E6 | Probabilistic data structures (Bloom, Count-Min, HyperLogLog, Cuckoo) | ix-probabilistic | `crates/ix-probabilistic/src/{bloom,count_min,cuckoo,hyperloglog}.rs` | already_done | document | P3 | XS | high | **yes** (12 UDFs: `ix_bloom_*`, `ix_cms_*`, `ix_hll_*`, `ix_cuckoo_*`) | no | no | MCP `ix_bloom_filter`, `ix_hyperloglog`. Four of TARS's 20 closures, all present in IX with fuller DuckDB exposure than anything TARS built. |
| E7 | SVM, random forest, attention, transformer | ix-supervised / ix-nn | `crates/ix-supervised/src/svm.rs` (`LinearSVM`), `decision_tree.rs`; `crates/ix-nn/src/{attention,transformer,positional}.rs` | already_done | document | P3 | XS | high | partial | no | no | MCP `ix_random_forest`, `ix_supervised`, `ix_gradient_boosting`, `ix_nn_forward`. Four more of the 20 closures, all covered. |
| E8 | Variational autoencoder | ix-nn | — | missing_algorithm | defer | P3 | M | medium | no | no | no | `createVariationalAutoencoder` exists in TARS; IX has no `vae`. Deferred: `ix-optick-sae` (D4) already covers IX's actual latent-representation consumer, so a VAE would have no user. |
| E9 | Graph construction & traversal | ix-graph | `crates/ix-graph/src/{graph,routing,state_space}.rs` | already_done | document | P3 | XS | high | **yes** (`ix_shortest_path`, `ix_centrality`, `ix_pagerank`, `ix_connected_components`) | no | no | MCP `ix_graph`. Two more closures covered. Per CLAUDE.md, do **not** add `petgraph`/`daggy`. |
| E10 | Markov chains, MDP / POMDP | ix-graph | `crates/ix-graph/src/{markov,hmm}.rs`, `crates/memristive-markov/` | already_done | document | P3 | XS | high | **yes** (`ix_viterbi`) | no | no | MCP `ix_markov`, `ix_viterbi`. #190 flags the source as introductory and already covered; confirmed. |

### F. Cross-cutting integration proposals

Sources: `v1/ADVANCED_MATHEMATICAL_INTEGRATION_STRATEGY.md` and
`v1/COMPREHENSIVE_MATHEMATICAL_LEVERAGE_STRATEGY.md`, with all status markers and impact percentages
discarded per the [closure-name finding](#a-second-phantom-layer-closure-names).

| # | candidate_name | ix_area | current_ix_surface | status | recommended_action | prio | cx | testability | duckdb | skill | pipeline | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Topological code analysis (persistent homology over code structure) | ix-topo × ix-code | `crates/ix-topo/` + `crates/ix-code/`; MCP `ix_code_analyze`, `ix_ast_query`, `ix_code_smells`; DuckDB `ix_code_complexity`, `ix_code_metrics`, `ix_code_smells`, `ix_ast_query` | implemented_needs_exposure | prototype | P2 | M | medium | partial | no | **candidate** | **Best-supported composition in this section.** Both halves are built and already exposed *separately*; nothing joins them. IX also has the fuel TARS lacked: `crates/ix-code` (the single crate CLAUDE.md refers to as `ix-code-*`) owns offline corpus AST analysis under CLAUDE.md's realtime/offline boundary. Testability is `medium`, not `high`: "does code topology predict anything" needs a labelled outcome, and the honest baseline is `ga/state/quality/`. Note the TARS target file `CodeAnalyzerService.fs` does not exist at the pinned SHA — the idea is salvage, the implementation is not. |
| F2 | Fractal complexity of code | ix-chaos × ix-code | `crates/ix-chaos/src/fractal.rs`; DuckDB `ix_hurst` | partially_implemented | benchmark | P3 | S | medium | partial | no | no | `box_counting_dimension_2d` needs a 2-D point cloud; turning a code structure into one is an unstated modelling choice in the TARS text. Cheap to try over `ix_code_metrics` output, but declare the baseline first — CLAUDE.md's "instrument before you ship" applies. |
| F3 | Topological analysis of Tree-of-Thought spaces | ix-topo × ToT | `docs/research/tars-v1-tree-of-thought-to-ix.md` (#192, closed); `crates/ix-topo/` | partially_implemented | defer | P3 | M | low | no | no | no | #192 delivered the ToT extraction packet, so the schema half now exists. The missing half is data: no IX-side ToT run produces a thought corpus to compute persistence over. Same fuel problem as D2. Defer until #192's model has a producer. |
| F4 | Advanced DSL constructs (`state` / `transition` / `prompt` blocks) | ix-pipeline | `crates/ix-pipeline/src/{dag,spec,builder,executor,lower,gate}.rs` | reject | document | P3 | M | low | no | no | n/a | TARS metascript/FLUX runtime syntax — band B in #190's taxonomy. CLAUDE.md is explicit that IX expresses workflow composition through `ix-pipeline::dag::Dag<N>`. Rejected on ownership: a language runtime is a TARS contract, not an IX algorithm. |
| F5 | Composite "intelligence" / impact scoring for the above | — | — | reject | document | P3 | S | low | no | no | no | Every impact figure in both strategy documents ("70-90% reasoning depth", "90-98% prediction accuracy", "300-400% sophistication") is asserted with no baseline, no instrument and no method. #190 reaches the same verdict for `intelligence-progression-measurement.md` on independent grounds. Adopting any of them would install exactly the Goodhart failure the IX quality pipeline exists to prevent. |

## Summary by status

48 rows total (A=7, B=9, C=8, D=9, E=10, F=5).

| status | count | rows |
| --- | ---: | --- |
| `already_done` | 21 | A1, A5, B1, B2, B5, C1, C3, C4, D1, D3, D4, D5, D7, D8, E1, E4, E5, E6, E7, E9, E10 |
| `missing_algorithm` | 11 | A4, A7, B6, C5, C6, C7, D2, D6, E2, E3, E8 |
| `implemented_needs_exposure` | 5 | A3, B1b, B4, B7, F1 |
| `partially_implemented` | 5 | A2, A6, C2, F2, F3 |
| `reject` | 5 | B3, B8, D9, F4, F5 |
| `research_only` | 1 | C8 |
| `defer_v3` | 0 | none — deferrals are expressed as `recommended_action: defer` on rows whose *status* is a factual claim about what IX has |

C3 is counted under `already_done` but carries a defect; see its row.

**The headline.** Of the techniques `MISSED_OPPORTUNITIES_ANALYSIS.md` calls TARS's biggest misses,
IX already implements the majority. Persistent homology, category theory, spectral methods, Lie
structures, Takagi and de Rham curves, neural ODEs, Pauli matrices, sedenion BSP, PCA/SVD,
probabilistic structures and Markov machinery are all present and under test. What IX lacks is
narrower and more specific than the TARS text suggests: **discrete-time control theory (A1, A4, A5,
A6), causal inference (B6), tensor decomposition (D6), and four unbuilt compositions of parts IX
already owns (C5, C6, C7, F1).**

**Correction, 2026-09-08.** A1 and A5 have since been delivered (`crates/ix-signal/src/state_space.rs`),
and building them showed the A1 label was wrong: the LTI matrices were already inside `kalman.rs`, so
A1 was `partially_implemented`, not `missing_algorithm`. A5 was a genuine gap and is confirmed as one.
The control-theory hole is therefore real but was one row smaller than this matrix claimed. See the A1
and A5 rows for the evidence.

**The second headline.** The largest exposure deficit is not per-technique but per-crate.
`ix-dynamics` (Lie algebras, inverse kinematics, neural ODEs) and `ix-signal`'s Kalman filter have
**no MCP tool, no skill verb, and — apart from `ix_kalman_smooth` — no UDF**. They are correct,
tested, and invisible to every agent in the federation. `crates/ix-skill/src/verbs/` contains no math
verb at all: the verbs are `beliefs`, `check`, `compile`, `demo`, `describe`, `embed_coverage`,
`list`, `pipeline`, `run`, `stable_surface`. The `skill_exposure` column is therefore `no` for every
row in this matrix — which is a finding, not an omission.

## Follow-up issue candidates

Proposed, **not opened** — #202 asks for candidates, and its non-goals forbid implementing anything
here. Each is scoped as a tracer-bullet vertical slice per CLAUDE.md, with the oracle named up front,
and ordered so nothing depends on something below it.

| # | Proposed issue | Rows | Why it earns an issue | Oracle | cx |
| --- | --- | --- | --- | --- | --- |
| 1 | ~~Discrete `StateSpaceModel` + observability/controllability rank tests in `ix-signal`~~ **DELIVERED 2026-09-08** | A1, A5 | Unblocks all of section A and #193; the smallest slice that makes "state-space" mean the control-theory thing in IX | Textbook `(A,B,C)` triples with known rank deficiency, plus a deficiency hidden behind a similarity transform; exact | S |
| 2 | Expose `ix-dynamics` and the Kalman filter through MCP + a skill verb | A3, B4, B7 | Largest exposure deficit in the matrix — three correct crates are agent-invisible. Must bump `EXPECTED[]` in `crates/ix-agent/tests/parity.rs`, which is the count oracle | `parity.rs` count assertion plus per-op smoke tests | S |
| 3 | Fractal mutation operator for `ix-evolution` | C5, C6 | Already scoped as [#204](https://github.com/GuitarAlchemist/ix/issues/204); this matrix confirms both halves exist and the join does not | Seeded fractal-vs-Gaussian mutation benchmark on a fixed multi-modal objective | S |
| 4 | De-duplicate `takagi.rs` / `de_rham.rs` between `ix-fractal` and `ix-chaos` | C3 | Pure IX hygiene found while verifying C1/C2; two verbatim copies will silently diverge | `cargo test --workspace` unchanged; the copy's tests must pass against the re-export | XS |
| 5 | `ix_de_rham` DuckDB UDF | C2 | Residue of closed #203; mirrors the `ix_takagi` UDF that already exists | Same closed-form known-value tests as `ix_takagi` | XS |
| 6 | Lyapunov stability certificate (discrete Lyapunov equation) in `ix-signal` | A6 | Disambiguates the two Lyapunovs; depends on issue 1 | Known stable/unstable `A`; `P ≻ 0` check | S |
| 7 | Tensor decomposition (Tucker / CP) in `ix-math` | D6 | Only missing branch of the vector-decomposition survey, with a live consumer in the OPTIC-K corpus | Reconstruct a synthetic low-rank tensor to a known error | M |
| 8 | Fractal dimension of grammar production rules | C7 | Most implementable grammar document in the corpus; joins `ix-grammar` and `ix-chaos` | Self-similar rules with analytically known dimension | M |
| 9 | Topological code-analysis spike (`ix-topo` × `ix-code`) | F1 | Both halves shipped and exposed separately; needs a declared baseline before it is worth building | `ga/state/quality/` baseline plus declared expected direction and guardrail | M |
| 10 | Causal inference primitives (SCM + backdoor adjustment) | B6 | Genuine gap with a plausible consumer in `ix-quality-trend` | Synthetic SCMs with known ground-truth effects | L |

Issues 1–5 are the recommended first tranche: each is XS or S, each has an exact oracle, and together
they close every `implemented_needs_exposure` row plus the two cheapest `missing_algorithm` rows.
Issues 7–10 should not start before a consumer is named, per #202's "no esoteric algorithms without
tests and a consumer" non-goal.

## Cost notes

| Item | Value |
| --- | --- |
| Budget from #202 | `free-local`, `max_cost_usd: 0`, `max_runner_minutes: 30` |
| Actual spend | **0 USD** |
| Method | Local `git cat-file` / `git show` / `git ls-tree` against the pinned tars checkout; local `grep` / `ls` / `diff` over this worktree; `gh issue view` for sibling-issue status |
| Hosted passes | none — no embedding, summarization or model call was made over any transcript |
| Compute | no `cargo build`, `cargo test` or `cargo clippy` run; this change adds no Rust and touches no `.github/workflows/**` |

The `cheap-hosted` tier #190 assigns to the four long transcripts (D2's 6,993 lines, E2's 6,406,
D7's 4,858, D6's 1,513) was **not** spent. Those rows rest on #190's summaries, plus
existence-and-length verification at the pinned SHA, plus the IX-side surface check. That is enough to
decide *whether IX already has it* — the question #202 asks — but not enough to extract an
implementable specification. Any row whose `recommended_action` is `prototype` and whose source is one
of those four would need that pass first; only D2 and E2 are in that position, and both are `defer`.

## Method, and its limits

1. Read #202 and the corrected inventory (#190) in full before writing any row.
2. Resolved the pinned SHA in the sibling checkout and confirmed the commit object exists. That
   checkout's `HEAD` is on an unrelated branch (`refactor/reason-feedback-seam` @ `9490f73`), so every
   TARS read went through explicit `<SHA>:<path>` addressing rather than the working tree — no row can
   have picked up post-pin content.
3. Existence-checked each source document individually, recording line counts, before classifying it.
4. Re-derived the phantom-file result independently from a full `git ls-tree -r <SHA>`, rather than
   trusting #190's table.
5. Read the three strategy documents named by #202 in full, then verified their code claims — closure
   names and target files — against the tree. That is what surfaced the closure-name finding.
6. Mapped each candidate onto IX by listing `crates/*/src/` for the 17 relevant crates and reading the
   `pub` surface of every file cited in a `current_ix_surface` cell.
7. Cross-checked exposure against three authoritative lists: registered UDF names under
   `crates/ix-duck*/src/`, the 96-entry `EXPECTED` array in `crates/ix-agent/tests/parity.rs`, and the
   verb list in `crates/ix-skill/src/verbs/`.

### Unresolved and not reached

Stated explicitly, because a gap matrix that hides its own gaps is worse than no matrix.

- **`ChatGPT-Advanced Math CS 2025.md` was not read.** It exists at the pinned SHA and is the stated
  source for six section-B rows, but those rows were built from `MISSED_OPPORTUNITIES_ANALYSIS.md`'s
  summary of it. It is absent from #190's band-A list, so it has never been read on the IX side by
  either survey. If any section-B row matters enough to act on, read it first. Its line count was not
  measured.
- **The four long transcripts were not read end to end.** D2, D6, D7 and E2 rest on #190's reading
  plus verification that the document exists at the stated length. A specification buried in a tail
  section could have been missed.
- **No source document is missing.** Every path this matrix cites resolved at the pinned SHA. Had one
  failed to resolve, its row would appear here with `current_ix_surface: source not found` rather than
  being silently dropped — no row needed that treatment.
- **Negative results are grep-bounded.** Every `—` in `current_ix_surface` means a case-insensitive
  regex over `crates/**/*.rs` found nothing; the patterns are recorded in the row notes so each can be
  re-run. An implementation under an unguessable name would read as a false gap. Two such traps were
  caught this way — `ix-graph::state_space` in A1, `causal` attention masks in B6 — so the risk is
  demonstrated, not hypothetical. There may be others.
- **No claim in any TARS source document has been validated,** and no IX algorithm cited here was
  re-verified for correctness. `already_done` means *the code exists and is under test*, not *it is
  right*.
- **Exposure columns are a snapshot** of `main` @ `4331cf7` plus this branch. #203 closed recently
  enough that C1/C2's split state is fresh; re-check before acting on issue candidate 5.
- **No build was run.** This change is documentation only and adds no Rust, so the pinned clippy gate
  (`cargo +nightly-2026-08-23 clippy --workspace --all-targets -- -D warnings`) was not exercised
  locally.
