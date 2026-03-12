---
name: machin-search
description: Search and pathfinding with A*, Q*, MCTS, minimax, BFS/DFS
---

# Search

Find optimal paths, solve games, or explore state spaces.

## When to Use
When the user needs pathfinding, game AI, decision making under uncertainty, or combinatorial search.

## Algorithm Selection
| Problem | Algorithm | When |
|---------|-----------|------|
| Shortest path with heuristic | **A*** | Known goal, admissible heuristic available |
| Learned heuristic search | **Q*** | When you can train a Q-function on the domain |
| Game tree (deterministic) | **Minimax / Alpha-Beta** | Two-player zero-sum games |
| Game tree (stochastic) | **MCTS** | Large branching factor, simulation possible |
| Unweighted shortest path | **BFS** | All edges cost 1 |
| Exhaustive exploration | **DFS / IDDFS** | Memory-constrained, solution depth unknown |
| Local optimization | **Hill Climbing / Tabu** | Single solution improvement |

## Programmatic Usage
```rust
use machin_search::astar::{SearchState, astar};
use machin_search::qstar::{QFunction, TabularQ, qstar_search};
use machin_search::mcts::{MctsState, mcts_search};
use machin_search::adversarial::{GameState, alpha_beta};

// Implement SearchState for your domain, then:
let result = astar(start_state, |s| s.manhattan_distance());
```

## Q* vs A*
Q* uses a learned heuristic (DQN-style) instead of hand-crafted. Dramatically fewer node expansions in large action spaces. Use `compare_qstar_vs_astar()` to benchmark.
