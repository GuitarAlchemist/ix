---
name: machin-game
description: Game theory — Nash equilibria, Shapley value, auctions, evolutionary dynamics
---

# Game Theory

Model strategic interactions between rational agents.

## When to Use
When the user needs to analyze competitive/cooperative scenarios, design auctions, compute fair allocations, or model population dynamics.

## Capabilities
- **Nash Equilibria** — Support enumeration, fictitious play, dominant strategies
- **Cooperative Games** — Shapley value, Banzhaf index, core membership
- **Auctions** — First-price, second-price (Vickrey), English, Dutch, all-pay
- **Evolutionary** — Replicator dynamics, ESS detection, Hawk-Dove, Rock-Paper-Scissors
- **Mechanism Design** — VCG mechanism, Myerson optimal reserve
- **Mean Field** — Large-population game dynamics, logit equilibrium

## Programmatic Usage
```rust
use machin_game::nash::{BimatrixGame, support_enumeration};
use machin_game::cooperative::{CooperativeGame, shapley_value};
use machin_game::auction::{second_price_auction, english_auction};
use machin_game::evolutionary::{replicator_dynamics, is_ess};
use machin_game::mean_field::{MeanFieldGame, find_equilibrium};
```

## Classic Examples
- Prisoner's Dilemma: `[[3,0],[5,1]]` vs `[[3,5],[0,1]]`
- Matching Pennies: `[[1,-1],[-1,1]]` vs `[[-1,1],[1,-1]]`
- Hawk-Dove: use `hawk_dove_matrix(v, c)` helper
