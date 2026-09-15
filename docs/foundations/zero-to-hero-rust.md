# Zero to Hero: Rust for Machine Learning

> Everything you need to know about Rust to work with ix — from absolute beginner to productive ML developer.

This guide assumes no prior Rust experience. It covers exactly the Rust you need for ML development, skipping language features you won't use.

---

## 1. Variables and Types

Rust variables are immutable by default. Add `mut` to make them mutable.

```rust
let x = 5;           // Immutable — can't change this
let mut y = 10;      // Mutable — can reassign
y = 20;              // OK

// Type annotations (usually optional — Rust infers types)
let count: usize = 42;     // Unsigned integer (used for array indices, counts)
let price: f64 = 19.99;    // 64-bit float (all ML math in ix)
let flag: bool = true;      // Boolean
let name: &str = "hello";   // String slice (borrowed text)
```

### The Types You'll See in ML

| Type | What It Is | Where You'll See It |
|------|-----------|-------------------|
| `f64` | 64-bit float | All CPU math, weights, predictions |
| `f32` | 32-bit float | GPU computations (faster on GPU) |
| `usize` | Unsigned integer (pointer-sized) | Array indices, class labels, counts |
| `bool` | True/false | Flags, convergence checks |
| `Vec<f64>` | Growable list of floats | Raw data before converting to ndarray |
| `Array1<f64>` | 1D ndarray | Vectors, predictions, labels |
| `Array2<f64>` | 2D ndarray | Datasets, weight matrices |
| `Option<T>` | Value that might not exist | Model weights before training |
| `Result<T, E>` | Value that might be an error | Math operations that can fail |

## 2. Ownership and Borrowing

This is Rust's most unique feature. It prevents memory bugs at compile time.

**The rules:**
1. Every value has exactly one **owner**
2. When the owner goes out of scope, the value is dropped (freed)
3. You can **borrow** a value without taking ownership

```rust
let data = vec![1.0, 2.0, 3.0];

// Immutable borrow (&) — you can read but not modify
let sum: f64 = data.iter().sum();  // borrows data
println!("{:?}", data);             // data still usable

// Mutable borrow (&mut) — you can modify
let mut data = vec![1.0, 2.0, 3.0];
data.push(4.0);                     // modifies data

// Moving ownership — the original variable is gone
let data = vec![1.0, 2.0, 3.0];
let data2 = data;                   // data moved to data2
// println!("{:?}", data);          // ERROR: data was moved
```

### Why This Matters for ML

When you see `&` in function signatures, it means "I'm borrowing this, not consuming it":

```rust
// This function borrows the arrays — you can still use them after
fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

let weights = array![1.0, 2.0, 3.0];
let input = array![4.0, 5.0, 6.0];
let result = dot_product(&weights, &input);  // borrow with &
// weights and input are still usable here
```

**In ix**: Most functions take `&` references. The `.fit(&mut self, x, y)` pattern borrows the data and mutates the model.

## 3. Structs and Methods

Structs are Rust's way of grouping data. Methods are functions attached to a struct.

```rust
struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    // Constructor (by convention called `new`)
    fn new(n_features: usize) -> Self {
        LinearModel {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    // Method that borrows self (read-only)
    fn predict(&self, x: &[f64]) -> f64 {
        let dot: f64 = self.weights.iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        dot + self.bias
    }

    // Method that mutably borrows self (can modify)
    fn update_bias(&mut self, new_bias: f64) {
        self.bias = new_bias;
    }
}

let mut model = LinearModel::new(3);
let prediction = model.predict(&[1.0, 2.0, 3.0]);
model.update_bias(0.5);
```

### The Builder Pattern

Many ix algorithms use this for configuration:

```rust
let optimizer = ParticleSwarm::new()     // Start with defaults
    .with_particles(50)                   // Chain configuration
    .with_max_iterations(1000)
    .with_bounds(-10.0, 10.0)
    .with_seed(42);                       // Each returns Self
```

Each `.with_*()` method takes `mut self` and returns `Self`, enabling the chain.

## 4. Traits (Interfaces)

Traits define shared behavior. If you know interfaces (Java) or protocols (Swift), traits are similar.

```rust
// ix defines traits like:
trait Classifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
}

// Multiple algorithms implement the same trait:
// KNN, DecisionTree, LogisticRegression, LinearSVM, GaussianNaiveBayes
// All have .fit() and .predict() — same interface, different algorithms
```

**Why this matters**: You can write code that works with *any* classifier:

```rust
fn evaluate<C: Classifier>(model: &C, test_x: &Array2<f64>, test_y: &Array1<usize>) -> f64 {
    let predictions = model.predict(test_x);
    metrics::accuracy(test_y, &predictions)
}

// Works with any classifier
evaluate(&knn, &test_x, &test_y);
evaluate(&tree, &test_x, &test_y);
evaluate(&svm, &test_x, &test_y);
```

## 5. Option and Result (Error Handling)

### Option: Something Might Not Exist

```rust
let mut model = LinearRegression::new();
// Before training, weights don't exist
assert!(model.weights.is_none());

model.fit(&x, &y);
// After training, weights exist
if let Some(w) = &model.weights {
    println!("Weights: {:?}", w);
}
```

`Option<T>` is either `Some(value)` or `None`. Use it for values that might not be set yet (like model weights before training).

### Result: Something Might Fail

```rust
use ix_math::linalg;

// Matrix multiplication can fail (dimension mismatch)
match linalg::matmul(&a, &b) {
    Ok(product) => println!("Result: {:?}", product),
    Err(e) => println!("Error: {}", e),
}

// Shorthand: .unwrap() panics on error (fine for examples, not production)
let product = linalg::matmul(&a, &b).unwrap();

// Shorthand: ? propagates the error to the caller
fn compute(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let product = linalg::matmul(a, b)?;  // Returns Err early if it fails
    Ok(product)
}
```

**In ix**: Math functions return `Result`. In examples you'll see `.unwrap()` everywhere — in production, handle errors properly.

## 6. Iterators

Rust iterators are zero-cost abstractions — they compile to the same code as hand-written loops.

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Map: transform each element
let squared: Vec<f64> = data.iter().map(|x| x * x).collect();
// [1.0, 4.0, 9.0, 16.0, 25.0]

// Filter: keep matching elements
let big: Vec<&f64> = data.iter().filter(|&&x| x > 3.0).collect();
// [4.0, 5.0]

// Sum
let total: f64 = data.iter().sum();
// 15.0

// Enumerate: get index + value
for (i, val) in data.iter().enumerate() {
    println!("Index {}: {}", i, val);
}

// Zip: pair up two iterators
let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];
let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
// 1*4 + 2*5 + 3*6 = 32.0

// Chain: combine operations
let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
let variance: f64 = data.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / data.len() as f64;
```

### Common Iterator Patterns in ML

```rust
// Find the argmax (index of maximum value)
let scores = vec![0.1, 0.7, 0.2];
let (best_idx, best_val) = scores.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();
// best_idx = 1, best_val = 0.7

// Normalize to probabilities
let sum: f64 = scores.iter().sum();
let probs: Vec<f64> = scores.iter().map(|x| x / sum).collect();

// Compute confusion matrix entries
let true_positives = y_true.iter().zip(y_pred.iter())
    .filter(|(&t, &p)| t == 1 && p == 1)
    .count();
```

## 7. Closures

Closures are anonymous functions. You'll use them constantly with iterators and as objective functions for optimizers.

```rust
// Simple closure
let square = |x: f64| x * x;
println!("{}", square(3.0));  // 9.0

// Closure capturing a variable
let threshold = 0.5;
let is_positive = |x: f64| x > threshold;  // captures `threshold`

// Closures as function arguments (very common in ML)
let objective = |x: &Array1<f64>| -> f64 {
    // Rosenbrock function — classic optimization test
    let mut sum = 0.0;
    for i in 0..x.len()-1 {
        sum += 100.0 * (x[i+1] - x[i]*x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
};
```

**In ix**: `ClosureObjective` wraps a closure into an `ObjectiveFunction`:

```rust
use ix_optimize::traits::ClosureObjective;

let objective = ClosureObjective {
    f: |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2),
    dimensions: 2,
};
```

## 8. Generics and Trait Bounds

Generics let you write code that works with multiple types. Trait bounds constrain which types are allowed.

```rust
// This function works with any type that implements Classifier
fn cross_validate<C: Classifier>(model: &mut C, x: &Array2<f64>, y: &Array1<usize>) -> f64 {
    model.fit(x, y);
    let preds = model.predict(x);
    metrics::accuracy(y, &preds)
}

// Multiple bounds
fn search<S: SearchState + Clone + std::fmt::Debug>(start: S) {
    // S must implement SearchState AND Clone AND Debug
}
```

You'll read generics more than write them. When you see `<S: Trait>`, just read it as "S is any type that can do Trait things."

## 9. Modules and Imports

```rust
// Import specific items
use ndarray::{Array1, Array2, array};
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::traits::Regressor;
use ix_math::distance;

// Import everything from a module (use sparingly)
use ix_math::stats::*;

// ix crate naming convention:
// Crate name: ix-supervised  (hyphen)
// Import name: ix_supervised (underscore)
```

## 10. Running Code

```bash
# Build the whole workspace
cargo build --workspace

# Run tests
cargo test --workspace

# Run a specific example
cargo run --example pso_rosenbrock

# Run with optimizations (much faster for numerical code)
cargo run --release --example pso_rosenbrock
```

**Always use `--release` for benchmarking or real workloads.** Debug builds are 10-50x slower for numerical code.

## 11. Common Gotchas for New Rustaceans

### The Borrow Checker

```rust
let mut data = vec![1.0, 2.0, 3.0];
let first = &data[0];       // Immutable borrow
// data.push(4.0);          // ERROR: can't mutate while borrowed
println!("{}", first);       // Borrow ends here
data.push(4.0);             // Now OK
```

**Fix**: Restructure so borrows don't overlap with mutations.

### Type Conversions

```rust
let n: usize = 10;
let mean: f64 = sum / n as f64;  // Must explicitly cast usize to f64

let x: f32 = 3.14;
let y: f64 = x as f64;           // f32 to f64 (widening, always safe)
let z: f32 = y as f32;           // f64 to f32 (narrowing, may lose precision)
```

### Partial Ordering (Floats)

Floats can be NaN, so they don't implement full ordering. You'll see `.partial_cmp()` and `.unwrap()`:

```rust
// This doesn't compile:
// vec.sort_by(|a, b| a.cmp(b));

// Use partial_cmp instead:
vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

### turbofish `::<>`

Sometimes Rust can't infer a type in a chain. The "turbofish" syntax tells it:

```rust
let sum = data.iter().sum::<f64>();
//                        ^^^^^^^^ turbofish: "sum produces an f64"

let collected: Vec<f64> = data.iter().copied().collect();
// OR equivalently:
let collected = data.iter().copied().collect::<Vec<f64>>();
```

## Quick Reference Card

```rust
// Create arrays
let v = array![1.0, 2.0, 3.0];                    // Array1
let m = array![[1.0, 2.0], [3.0, 4.0]];           // Array2

// Train and predict (all algorithms follow this pattern)
let mut model = Algorithm::new(/* params */);
model.fit(&train_x, &train_y);
let predictions = model.predict(&test_x);

// Evaluate
let acc = metrics::accuracy(&test_y, &predictions);

// Optimize
let result = optimizer.minimize(&objective);
println!("Best: {:?} at {}", result.best_params, result.best_value);

// Error handling
let safe_result = risky_function()?;  // Propagate error
let unsafe_result = risky_function().unwrap();  // Panic on error
```

## Going Further

- **Rust Book** (free): https://doc.rust-lang.org/book/ — the definitive guide
- **Rust by Example**: https://doc.rust-lang.org/rust-by-example/ — learn by doing
- **ndarray docs**: https://docs.rs/ndarray — the array library ix builds on
- **Next**: [Rust for ML](rust-for-ml.md) — ML-specific Rust patterns in ix
- **Start learning**: [INDEX.md](../INDEX.md) — the full learning path
