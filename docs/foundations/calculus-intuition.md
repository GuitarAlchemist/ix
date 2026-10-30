# Calculus Intuition

> Derivatives tell you the slope. Gradients tell you the direction of steepest change. That's all you need for ML.

## The Problem

You're training a model to predict house prices. Your model has parameters (weights) that determine its predictions. Some parameter values give terrible predictions, others give great ones. You need to find the best parameters.

Imagine standing on a hilly landscape in thick fog. You can't see the lowest valley, but you *can* feel which way the ground slopes under your feet. Calculus gives you that "slope feeling" — it tells you which direction to step to go downhill. ML algorithms use this to minimize prediction errors.

## The Intuition

### Derivatives: Slope of a Curve

A **derivative** tells you how fast something is changing at a specific point.

If you're driving and your position over time is a curve, the derivative is your speedometer — it tells you your speed *right now*, not your average speed.

For a function f(x):
- If the derivative is **positive**, f is going up (increasing)
- If the derivative is **negative**, f is going down (decreasing)
- If the derivative is **zero**, you're at a flat spot (could be a minimum, maximum, or saddle point)

Think of a ball rolling in a bowl. At the bottom of the bowl, the surface is flat (derivative = 0) — that's the minimum.

### Gradients: Derivatives in Multiple Dimensions

Most ML models have many parameters, not just one. Instead of a single derivative, you get a **gradient** — a vector of derivatives, one for each parameter.

The gradient points in the direction of steepest *increase*. To minimize (go downhill), you walk in the *opposite* direction of the gradient. This is literally what gradient descent does.

```
Parameters: [w1, w2]
Gradient:   [∂f/∂w1, ∂f/∂w2]

The gradient says: "If you increase w1, the error goes up by ∂f/∂w1.
                    If you increase w2, the error goes up by ∂f/∂w2."

To reduce error: move opposite to the gradient.
```

### The Chain Rule: Why Deep Learning Works

Complex models are built from simple pieces chained together: input → layer 1 → layer 2 → output. The chain rule says:

"The derivative of a chain of functions is the product of each function's derivative."

In plain English: to know how the input affects the output, multiply how each step affects the next. This is the mathematical foundation of backpropagation in neural networks.

## How It Works

### Numerical Derivative

You don't always have a formula for the derivative. The simplest approach is to approximate it:

`f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)`

In plain English: nudge x slightly in both directions, see how much f changes, and divide by the nudge size. The smaller ε is, the more accurate (but too small causes floating-point issues).

This is called the **central difference** method. It's what ix uses when you don't provide an analytical gradient.

### Numerical Gradient (Multi-dimensional)

For a function of many variables, compute the partial derivative for each variable separately:

```
∂f/∂x₁ ≈ (f(x₁+ε, x₂, ...) - f(x₁-ε, x₂, ...)) / (2ε)
∂f/∂x₂ ≈ (f(x₁, x₂+ε, ...) - f(x₁, x₂-ε, ...)) / (2ε)
...
```

The gradient vector is `[∂f/∂x₁, ∂f/∂x₂, ...]`.

### Hessian Matrix (Second Derivatives)

The Hessian is a matrix of second derivatives. It tells you about the *curvature* — not just which way is downhill, but how steep or flat the terrain is in every direction.

```
H[i][j] = ∂²f / (∂xᵢ ∂xⱼ)
```

In plain English: the Hessian tells you whether the minimum is a sharp valley (steep curvature, can take big steps) or a shallow plain (flat curvature, need to be careful). Some advanced optimizers use this information.

## In Rust

ix provides numerical differentiation in `ix-math`:

```rust
use ndarray::array;
use ix_math::calculus;

// Scalar derivative
// f(x) = x² → f'(x) = 2x
let f = |x: f64| x * x;
let derivative_at_3 = calculus::derivative(&f, 3.0, 1e-7);
// derivative_at_3 ≈ 6.0

// Gradient of a multi-variable function
// f(x, y) = x² + y² → gradient = [2x, 2y]
let g = |x: &ndarray::Array1<f64>| x[0] * x[0] + x[1] * x[1];
let point = array![3.0, 4.0];
let grad = calculus::numerical_gradient(&g, &point, 1e-7);
// grad ≈ [6.0, 8.0]

// Hessian matrix (second derivatives)
let hessian = calculus::numerical_hessian(&g, &point, 1e-5);
// For f = x² + y², hessian ≈ [[2, 0], [0, 2]] (constant curvature)
```

### Gradients in Optimization

Here's how gradients connect to optimization — the core loop of training any model:

```rust
use ndarray::array;
use ix_optimize::gradient::SGD;
use ix_optimize::traits::{ClosureObjective, ObjectiveFunction, Optimizer};
use ix_optimize::gradient::minimize;
use ix_optimize::convergence::ConvergenceCriteria;

// Minimize f(x, y) = (x-3)² + (y-7)²
// The minimum is at (3, 7)
let objective = ClosureObjective {
    f: |x: &ndarray::Array1<f64>| {
        (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2)
    },
    dimensions: 2,
};

let mut optimizer = SGD::new(0.1);  // learning rate = 0.1
let initial = array![0.0, 0.0];    // start far from the answer
let criteria = ConvergenceCriteria {
    max_iterations: 1000,
    tolerance: 1e-8,
};

let result = minimize(&objective, &mut optimizer, initial, &criteria);
// result.best_params ≈ [3.0, 7.0]
```

Each iteration: compute gradient → take a step opposite to it → repeat.

## When To Use This

| Situation | What You Need |
|-----------|---------------|
| Training any supervised model | Gradient descent uses gradients to minimize loss |
| Debugging gradient descent | Numerical gradient to check your analytical gradient |
| Understanding optimization landscape | Hessian reveals curvature, helps choose learning rate |
| Implementing backpropagation | Chain rule composes derivatives through layers |

## Key Parameters

| Parameter | What It Controls | Too Small | Too Large |
|-----------|-----------------|-----------|-----------|
| ε (epsilon) in numerical gradient | Approximation accuracy | Floating-point noise dominates | Poor approximation of the true derivative |
| Learning rate (in gradient descent) | Step size downhill | Converges too slowly | Overshoots, diverges |

The sweet spot for numerical ε is around `1e-7` for derivatives and `1e-5` for Hessians (second derivatives amplify noise).

## Pitfalls

- **Numerical gradients are slow.** Computing the gradient of n parameters requires 2n function evaluations (one forward nudge and one backward per parameter). For neural networks with millions of parameters, analytical gradients via backpropagation are essential.
- **Local minima.** Gradient descent finds *a* minimum, not necessarily *the* minimum. For convex functions (like linear regression's loss), any minimum is the global minimum. For neural networks, there are many local minima — but in practice they're usually good enough.
- **Vanishing/exploding gradients.** In deep networks, the chain rule multiplies many small numbers (vanishing) or many large numbers (exploding). This is why activation function choice and weight initialization matter.
- **Don't confuse derivative = 0 with "found the answer."** Zero derivative could mean a maximum, minimum, or saddle point. In practice, gradient descent naturally avoids maxima and saddle points due to noise.

## Going Further

- **Next**: [Distance & Similarity](distance-and-similarity.md) — measuring closeness between data points
- **Uses this**: [Gradient Descent](../optimization/gradient-descent.md) — the optimization algorithm powered by gradients
- **Uses this**: [Backpropagation](../neural-networks/backpropagation.md) — chain rule applied to neural networks
