---
name: ix-bandit
description: Multi-armed bandit simulation — epsilon-greedy, UCB1, Thompson sampling
disable-model-invocation: true
---

# Multi-Armed Bandits

Simulate bandit algorithms to find the best arm given uncertain rewards.

## When to Use
When the user needs to compare exploration-exploitation strategies, simulate A/B testing, or understand bandit algorithms.

## Capabilities
- **Epsilon-greedy** — Explore with probability ε, exploit otherwise
- **UCB1** — Upper Confidence Bound (optimism in the face of uncertainty)
- **Thompson sampling** — Bayesian approach with posterior sampling

## Key Concepts
- Higher true mean → algorithm should pull that arm more
- UCB1 gives logarithmic regret bounds
- Thompson sampling is often best in practice

## Programmatic Usage
```rust
use ix_rl::bandit::{EpsilonGreedy, UCB1, ThompsonSampling};
```

## MCP Tool
Tool name: `ix_bandit`
Parameters: `algorithm`, `true_means`, `rounds`, `epsilon`
