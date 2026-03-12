---
name: machin-benchmark
description: Benchmark and compare machin algorithm performance
---

# Benchmark

Profile and compare algorithm performance across the machin workspace.

## When to Use
When the user wants to know which algorithm is fastest, measure scaling behavior, or compare approaches.

## Approach
1. Use `std::time::Instant` for wall-clock timing
2. Run multiple iterations to get stable measurements
3. Report mean, min, max, and standard deviation
4. Compare algorithms on the same problem instance

## Example Benchmarks
```rust
use std::time::Instant;

// Time an operation
let start = Instant::now();
let result = expensive_operation();
let elapsed = start.elapsed();
eprintln!("Took {:?}", elapsed);

// Compare methods
for method in ["sgd", "adam", "pso", "annealing"] {
    let start = Instant::now();
    // run optimization with method...
    eprintln!("{}: {:?}", method, start.elapsed());
}
```

## Common Comparisons
- **Optimization**: SGD vs Adam vs PSO vs Annealing on standard test functions
- **Search**: A* vs Q* vs BFS — node expansions and wall time
- **Clustering**: K-Means vs DBSCAN on different data shapes
- **Pipeline**: Sequential vs parallel DAG execution speedup

## Storing Results
Use machin-cache to persist benchmark results across sessions:
```rust
use machin_cache::store::Cache;
let cache = Cache::default_cache();
cache.set("bench:optimize:sphere:pso", &elapsed_ms);
```
