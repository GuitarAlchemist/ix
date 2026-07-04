# Research: TARS V1 Tree-of-Thought to IX Data Model

This document defines the minimal IX data model for extracting and storing Tree-of-Thought (ToT) or Workflow-of-Thought (WoT) structures from TARS V1.

## Goal
Define the smallest useful data model to represent reasoning trees, enabling TARS to export its internal thinking processes into IX for analysis, visualization, and persistence.

## Data Model

### ThoughtNode
A single unit of thought or a reasoning step.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `String` | Unique identifier for the node. |
| `content` | `String` | The actual text or data representing the thought. |
| `node_type` | `String` | e.g., "observation", "hypothesis", "action", "evaluation", "candidate". |
| `metadata` | `Map<String, Value>` | Arbitrary key-value pairs (e.g., model name, timestamp, token usage). |
| `score` | `Option<f64>` | Evaluation score for this node (0.0 to 1.0). Used for candidate ranking. |

### ThoughtEdge
A directed connection between two thoughts, representing a reasoning leap, dependency, or refinement.

| Field | Type | Description |
|-------|------|-------------|
| `source` | `String` | `id` of the parent node. |
| `target` | `String` | `id` of the child node. |
| `edge_type` | `String` | e.g., "leads_to", "supports", "refutes", "refines", "proposed_fix". |
| `weight` | `f64` | Confidence or transition probability (default 1.0). |

### ThoughtTree
The container for a collection of nodes and edges forming a reasoning graph.

| Field | Type | Description |
|-------|------|-------------|
| `tree_id` | `String` | Unique identifier for the reasoning session. |
| `root_nodes` | `Vec<String>` | IDs of the starting points of the reasoning. |
| `nodes` | `Vec<ThoughtNode>` | All nodes in the tree. |
| `edges` | `Vec<ThoughtEdge>` | All edges in the tree. |
| `status` | `String` | e.g., "active", "completed", "pruned". |

## Scoring and Ranking

To support Tree-of-Thought search algorithms (like BFS, DFS, or Beam Search), nodes and branches must be evaluatable.

- **`ThoughtNode.score`**: Represents the value of a specific thought in isolation or as a terminal state. High scores indicate promising paths.
- **`ThoughtEdge.weight`**: Represents the strength of the transition.
- **Candidate Ranking**: When TARS generates multiple candidate next-steps, they are stored as children nodes. The ranking is derived from the `score` field of these candidates.

## Responsibilities

| Responsibility | System | Description |
|----------------|--------|-------------|
| **Generation** | TARS | Generating candidate thoughts using LLMs. |
| **Evaluation** | TARS | Scoring nodes based on domain-specific heuristics or LLM-as-a-judge. |
| **Search Strategy** | TARS | Deciding which branches to expand (ToT BFS/DFS/Beam). |
| **Data Structure** | IX | Defining the schema and providing graph utility functions. |
| **Storage/Persistence**| IX | Saving reasoning traces to disk/database (JSON-on-disk handoff). |
| **Analysis** | IX | Computing graph metrics (centrality, depth, branchiness) on the traces. |

## Fixture Examples

### Example 1: Code Analysis (Bug Hunting)
Exploring potential causes for a failing test in a complex system.

```json
{
  "tree_id": "bug-analysis-001",
  "root_nodes": ["start"],
  "nodes": [
    { "id": "start", "content": "Test `test_matrix_inv` fails with NaN", "node_type": "observation" },
    { "id": "h1", "content": "Input matrix is singular", "node_type": "hypothesis", "score": 0.8 },
    { "id": "h2", "content": "Numerical instability in decomposition", "node_type": "hypothesis", "score": 0.4 },
    { "id": "v1", "content": "Check determinant of input", "node_type": "action" },
    { "id": "res1", "content": "Determinant is 1e-18 (near zero)", "node_type": "observation", "score": 0.9 }
  ],
  "edges": [
    { "source": "start", "target": "h1", "edge_type": "leads_to" },
    { "source": "start", "target": "h2", "edge_type": "leads_to" },
    { "source": "h1", "target": "v1", "edge_type": "refines" },
    { "source": "v1", "target": "res1", "edge_type": "leads_to" }
  ],
  "status": "completed"
}
```

### Example 2: Fix Selection (Ranking Candidates)
Ranking different proposed fixes for the singular matrix issue.

```json
{
  "tree_id": "fix-selection-001",
  "root_nodes": ["problem"],
  "nodes": [
    { "id": "problem", "content": "Need to handle singular matrices in `inv()`", "node_type": "observation" },
    { "id": "f1", "content": "Return Error", "node_type": "candidate", "score": 0.9 },
    { "id": "f2", "content": "Add small epsilon to diagonal (Tikhonov regularization)", "node_type": "candidate", "score": 0.6 },
    { "id": "f3", "content": "Return Moore-Penrose pseudo-inverse", "node_type": "candidate", "score": 0.7 }
  ],
  "edges": [
    { "source": "problem", "target": "f1", "edge_type": "proposed_fix" },
    { "source": "problem", "target": "f2", "edge_type": "proposed_fix" },
    { "source": "problem", "target": "f3", "edge_type": "proposed_fix" }
  ],
  "status": "active"
}
```
