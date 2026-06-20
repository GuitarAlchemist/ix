---
name: ix-dynamics
description: Dynamical systems — inverse kinematics, Lie groups/algebras, neural ODEs
disable-model-invocation: true
---

# Dynamics

Inverse kinematics chains, Lie group representations, and neural ODE solvers.

## When to Use
When the user asks about robot arm kinematics, rigid body transformations, SO(3)/SE(3) group operations, exponential maps, or neural ordinary differential equations.

## Capabilities
- **Inverse Kinematics** — Chain of revolute joints, Jacobian-based IK solver (CCD, damped least squares)
- **Lie Groups** — SO(3) rotation group, SE(3) rigid body group, exponential/logarithmic maps
- **Lie Algebras** — so(3) and se(3) Lie algebra operations, adjoint representation
- **Neural ODEs** — ODE integration with Euler/RK4, neural network parameterized dynamics

## Programmatic Usage
```rust
use ix_dynamics::ik::{IKChain, Joint};
use ix_dynamics::lie::{SO3, SE3, so3_exp, se3_exp};
use ix_dynamics::neural_ode::{NeuralODE, euler_step, rk4_step};
```

## MCP Tool Reference
Not yet available as a dedicated MCP tool. Use the Rust API directly.
