# TARS V1 Exploration Inventory

This document catalogs IX-focused explorations from TARS V1, providing a structured inventory for potential porting, auditing, or deferral within the IX ecosystem.

## Overview

TARS V1 contained extensive explorations into exotic mathematics, agentic workflows, and specialized data structures. As the ecosystem matures, these explorations are being reviewed for inclusion in IX (the Rust-based engine).

## Inventory Categories

### Vector
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Sedenion Partitioning | `TarsSedenionPartitioner.fs` | 16D BSP trees using sedenion-valued normals for importance-weighted partitioning. | vector | High | Free | Port to `ix-sedenion` |
| Vector Significance | `VectorSignificance.fs` | Entropy-based weight assignment for high-dimensional feature vectors. | vector | High | Free | Audit for `ix-math` |

### Graph
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Semantic Hypergraph | `TrsxHypergraph.fs` | Hypergraph tracking file versions with semantic vectors and quaternion embeddings. | graph | Medium | Free | Audit for `ix-graph` |
| Graph K-Theory | `GraphKTheory.fs` | Detecting feedback cycles and resource invariants using Smith normal form. | graph | High | Free | Port to `ix-ktheory` |

### Grammar
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Weighted Grammar | `WeightedGrammar.fs` | Beta-Binomial Bayesian weight updates for grammar rules. | grammar | High | Free | Ported (`ix-grammar`) |
| Replicator Dynamics | `ReplicatorDynamics.fs` | Evolutionary game theory for grammar rule competition. | grammar | High | Free | Ported (`ix-grammar`) |

### Search
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Grammar-Guided MCTS | `MctsBridge.fs` | MCTS search over workflow graphs using EBNF constraints. | search | High | Free | Ported (`ix-grammar`) |
| Q* Heuristics | `QStarHeuristics.fs` | Adaptive search with learned cost estimates for pathfinding. | search | Medium | Free | Port to `ix-search` |

### Math
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| GeometricSpace Enum | `HyperComplexGeometricDSL.fs` | Unified API for 10+ distance metrics (Euclidean, Hyperbolic, etc.). | math | High | Free | Ported (`ix-math`) |
| Sedenion Exp/Log | `HyperComplexGeometricDSL.fs` | Transcendental functions for 16D Cayley-Dickson algebra. | math | High | Free | Ported (`ix-sedenion`) |
| Plucker Coordinates | `PluckerLine.fs` | 6D line representation for rigid body transforms. | math | High | Free | Port to `ix-math` |

### Music
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Grothendieck Delta | `Grothendieck.fs` | Signed Z⁶ delta between musical PC-sets via interval-class vectors. | music | High | Free | Audit as general delta |
| PC-Set Prime Form | `SetTheory.fs` | Canonical representation of pitch-class sets for harmonic analysis. | music | High | Free | Port to `ix-bracelet` |

### Trace
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| AgenticTraceCapture | `AgenticTraceCapture.fs` | Structured event logging for agent interactions and system snapshots. | trace | Medium | Free | Defer (Audit v2) |

### DuckDB
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Music Set-Theory UDFs | `TarsDuckBridge.fs` | DuckDB UDFs for prime form, interval-class vectors, and chord classification. | duckdb | High | Free | Port to `ix-duck` |

### Eval
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Architecture Scorecard | `Scorecard.fs` | Multi-dimensional evaluation of system architecture metrics. | eval | Medium | Free | Port to `ix-eval` |

### State Space
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Neural ODE Solver | `NeuralOde.fs` | Continuous-depth neural networks using adaptive solvers. | state_space | High | Free | Port to `ix-dynamics` |

### Tree of Thought
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Path-Based Reasoner | `ToTReasoner.fs` | Tree-search over reasoning paths with value-head pruning. | tree_of_thought | Medium | Paid | Audit for `ix-agent` |

### Workflow of Thought
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| WoT Derivation Engine | `WoTDerivation.fs` | Grammar-driven generation of execution workflows. | workflow_of_thought | High | Paid | Ported (`ix-grammar`) |

### Defer
| Candidate | Source | Summary | Category | Testability | Cost Tier | Recommended Action |
|-----------|--------|---------|----------|-------------|-----------|--------------------|
| Hurwitz Quaternions | `Hurwitz.fs` | Integer lattice quaternions for number theory. | defer | High | Free | Defer (Niche) |
| CUDA Kernels | `CudaKernels.fs` | Raw CUDA implementations for hypercomplex math. | defer | Low | Free | Defer (Use WGPU) |
| FLUX DSL | `FluxDsl.fs` | Multi-language DSL for architectural constraints. | defer | Low | Free | Defer (F#-centric) |
