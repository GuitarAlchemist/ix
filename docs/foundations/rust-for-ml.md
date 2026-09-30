# Rust for ML

> The Rust patterns and crates you need to know before diving into machine learning algorithms.

## The Problem

You want to implement ML algorithms, but Rust isn't Python. There's no NumPy, no `import sklearn`. Instead, Rust gives you something better for production: zero-cost abstractions, memory safety without a garbage collector, and performance that matches C++.

The trade-off? You need to understand a few Rust-specific patterns before the ML code makes sense. This doc covers exactly those patterns — nothing more.

## The Key Crate: ndarray

Every ML algorithm in ix works with `ndarray` — Rust's equivalent of NumPy. It gives you n-dimensional arrays with fast element-wise operations.

### Array1: Vectors

A 1-dimensional array. Think of it as a list of numbers — a data point, a set of weights, or a gradient.

```rust
use ndarray::Array1;

// Create a vector
let v: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0]);

// Element-wise operations — these work exactly like NumPy
let doubled = &v * 2.0;           // [2.0, 4.0, 6.0]
let sum = &v + &doubled;          // [3.0, 6.0, 9.0]
let dot_product = v.dot(&doubled); // 1*2 + 2*4 + 3*6 = 28.0

// Useful methods
let total: f64 = v.sum();         // 6.0
let length = v.len();             // 3
let mapped = v.mapv(|x| x * x);  // [1.0, 4.0, 9.0]
```

**Important**: Notice the `&` references. In Rust, `&v * 2.0` borrows `v` without consuming it, so you can use `v` again. This is Rust's ownership system keeping your data safe.

### Array2: Matrices

A 2-dimensional array. Think of it as a dataset where each row is a sample and each column is a feature.

```rust
use ndarray::{Array2, array};

// Create a 2x3 matrix (2 samples, 3 features)
let data: Array2<f64> = array![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
];

let (rows, cols) = data.dim();     // (2, 3)
let row_0 = data.row(0);          // view of [1.0, 2.0, 3.0]
let col_1 = data.column(1);       // view of [2.0, 5.0]

// Matrix transpose
let transposed = data.t();         // 3x2 matrix
```

### The `array!` Macro

The fastest way to create small arrays for testing:

```rust
use ndarray::array;

let vector = array![1.0, 2.0, 3.0];           // Array1
let matrix = array![[1.0, 2.0], [3.0, 4.0]];  // Array2
```

## Traits: How ix Organizes Algorithms

Rust traits are like interfaces — they define what an algorithm *can do*. ix uses a few key traits across all crates:

### Regressor: Predicts a Number

```rust
pub trait Regressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}
```

Every regression algorithm (linear, polynomial, etc.) implements this. You always call `.fit()` with training data, then `.predict()` with new data.

### Classifier: Predicts a Category

```rust
pub trait Classifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
}
```

Same pattern, but labels are `usize` (integers representing classes: 0, 1, 2, ...) and there's an extra `predict_proba` for probability estimates.

### Clusterer: Finds Groups

```rust
pub trait Clusterer {
    fn fit(&mut self, x: &Array2<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn fit_predict(&mut self, x: &Array2<f64>) -> Array1<usize>;
}
```

No labels needed — the algorithm discovers structure on its own.

### Optimizer: Finds the Best Parameters

```rust
pub trait Optimizer {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &str;
}
```

Takes current parameters and a gradient, returns updated parameters.

## The Builder Pattern

Many algorithms have hyperparameters (settings you choose before training). ix uses the builder pattern to configure them fluently:

```rust
use ix_optimize::pso::ParticleSwarm;

let optimizer = ParticleSwarm::new()
    .with_particles(50)
    .with_max_iterations(1000)
    .with_bounds(-10.0, 10.0)
    .with_seed(42);
```

This pattern chains `.with_*()` calls to set options. Each method returns `Self`, so you can keep chaining. Any option you skip uses a sensible default.

## Seeded RNG for Reproducibility

ML algorithms often use randomness (random initialization, random sampling). ix takes a `seed` parameter so you get the same results every time:

```rust
use ix_unsupervised::kmeans::KMeans;

let mut kmeans = KMeans::new(3).with_seed(42);
// Running this twice with seed 42 gives identical clusters
```

Under the hood, this uses `rand::rngs::StdRng::seed_from_u64(seed)`.

## Error Handling: Result and MathError

Math operations can fail (mismatched dimensions, singular matrices). ix returns `Result<T, MathError>`:

```rust
use ix_math::linalg;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![[5.0], [6.0]];

match linalg::matmul(&a, &b) {
    Ok(result) => println!("Product: {:?}", result),
    Err(e) => println!("Error: {}", e),
}
```

In examples and quick experiments, you'll often see `.unwrap()` which panics on error — fine for learning, but handle errors properly in production code.

## f64 Everywhere

ix uses `f64` (64-bit floating point) for all CPU computations. This gives ~15 decimal digits of precision, which is more than enough for ML. GPU code uses `f32` for performance (GPUs are much faster with 32-bit floats).

## Iterators: The Rust Way to Process Data

You'll see iterator chains throughout the codebase. Here's a quick cheat sheet:

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Map: transform each element
let squared: Vec<f64> = data.iter().map(|x| x * x).collect();

// Filter: keep elements matching a condition
let big: Vec<&f64> = data.iter().filter(|&&x| x > 3.0).collect();

// Fold: reduce to a single value (like sum, but general)
let sum: f64 = data.iter().fold(0.0, |acc, x| acc + x);

// Zip: pair up two iterators
let a = vec![1.0, 2.0];
let b = vec![3.0, 4.0];
let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
// 1*3 + 2*4 = 11.0
```

These are zero-cost — the compiler optimizes them to the same machine code as a hand-written loop.

## Putting It Together

Here's a complete example that uses all these patterns — training a linear regression model:

```rust
use ndarray::array;
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::traits::Regressor;

fn main() {
    // Training data: 2 features per sample
    let x_train = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ];
    let y_train = array![2.0, 4.0, 6.0, 8.0];

    // Create and train
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train);

    // Predict
    let x_test = array![[5.0, 5.0], [6.0, 6.0]];
    let predictions = model.predict(&x_test);

    println!("Predictions: {:?}", predictions);
}
```

## Going Further

Now that you know the Rust patterns, start with the math foundations:
- **Next**: [Vectors & Matrices](vectors-and-matrices.md) — the mathematical objects behind `Array1` and `Array2`
- Or jump to any algorithm doc — they all use the patterns from this page.
