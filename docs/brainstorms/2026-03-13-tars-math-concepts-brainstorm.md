---
date: 2026-03-13
topic: tars-math-concepts
---

# TARS Mathematical Concepts for MachinDeOuf

## What We're Building

Implement 9 mathematical domains from TARS v1/v2 explorations into MachinDeOuf, organized as 4 new crates + 2 crate extensions. These add algebraic types (quaternions, dual quaternions, sedenions, Plucker coordinates), number theory (prime patterns), fractal curves (Takagi, de Rham), topological data analysis, continuous dynamics (Neural ODEs, Lie groups), algebraic K-theory, and category theory primitives.

Source material: `C:\Users\spare\source\repos\tars\v1\docs\Explorations\v1\Chats\` and `C:\Users\spare\source\repos\tars\v2\docs\4_Research\Conversations\`.

## Organization: 4 New Crates + 2 Extensions

### Extend `machin-math` — Algebraic Types + Number Theory

**Quaternions** (`math/quaternion.rs`):
- `Quaternion { w, x, y, z }` — unit quaternion for 3D rotation
- `mul`, `conjugate`, `inverse`, `norm`, `normalize`
- `slerp(q1, q2, t)` — spherical linear interpolation
- `from_axis_angle`, `to_rotation_matrix` → `Array2<f64>`
- `exp`, `ln` — for continuous rotation paths

**Dual Quaternions** (`math/dual_quaternion.rs`):
- `DualQuaternion { real: Quaternion, dual: Quaternion }` — 6-DOF rigid transform
- `mul`, `conjugate`, `norm`, `normalize`
- `from_rotation_translation(q, t)`, `to_rotation_translation()`
- `sclerp(dq1, dq2, t)` — screw linear interpolation
- Encodes rotation + translation in single 8-element algebra

**Plucker Coordinates** (`math/plucker.rs`):
- `PluckerLine { direction: Array1<f64>, moment: Array1<f64> }` — 6D line representation
- `from_two_points`, `from_point_direction`
- `reciprocal_product(l1, l2)` — measures line-line distance/angle
- `intersects(l1, l2)` — coplanarity test via reciprocal product = 0
- `to_dual_quaternion()`, `from_dual_quaternion()` — conversion bridge

**Sedenions** (`math/sedenion.rs`):
- `Sedenion([f64; 16])` — 16D Cayley-Dickson algebra
- `mul` (non-associative!), `conjugate`, `norm`
- `from_octonion_pair`, `to_octonion_pair`
- Note: loses associativity and alternativity — document clearly

**BSP in High Dimensions** (`math/bsp.rs`):
- `BspNode<const D: usize>` — binary space partition tree
- `insert`, `query_region`, `nearest_neighbor`
- Works in any dimension (2D to 16D for sedenion space)

**Prime Utilities** (`math/primes.rs`):
- `sieve_of_eratosthenes(limit) -> Vec<u64>`
- `is_prime(n) -> bool` — trial division with sqrt optimization
- `prime_triplets(limit) -> Vec<(u64, u64, u64)>` — (p, p+2, p+6) patterns
- `prime_memory_hash(value, modulus) -> usize` — prime-based sparse hashing
- `nth_prime(n) -> u64`

### Extend `machin-chaos` — Fractal Curves

**Takagi Curve** (`chaos/takagi.rs`):
- `takagi(t: f64, terms: usize) -> f64` — Blancmange function
- `takagi_series(n_points: usize, terms: usize) -> Array1<f64>` — sampled curve
- `takagi_perturbation(state: &Array1<f64>, lr: f64) -> Array1<f64>` — fractal noise injection
- Continuous but nowhere differentiable — useful for non-smooth optimization landscapes

**De Rham Curves** (`chaos/derham.rs`):
- `derham_interpolate(p0, p1, depth, roughness, rng) -> Vec<Array1<f64>>` — IFS-based fractal path
- `derham_curve_1d(depth, roughness, rng) -> Array1<f64>` — 1D fractal signal
- Roughness decays 0.5x per recursion level
- Seeded RNG for reproducibility (workspace convention)

### New: `machin-topo` — Topological Data Analysis

**Simplicial Complexes** (`topo/simplex.rs`):
- `Simplex(Vec<usize>)` — k-simplex as sorted vertex set
- `SimplexStream` — filtered simplicial complex (ordered by birth time)
- `boundary(simplex) -> Vec<Simplex>` — boundary operator
- `rips_complex(points, max_dim, max_radius) -> SimplexStream` — Vietoris-Rips construction

**Persistent Homology** (`topo/persistence.rs`):
- `PersistenceDiagram { births: Vec<f64>, deaths: Vec<f64>, dimension: usize }`
- `compute_persistence(stream) -> Vec<PersistenceDiagram>` — reduction algorithm
- `betti_numbers(stream, radius) -> Vec<usize>` — beta_0, beta_1, beta_2
- `bottleneck_distance(pd1, pd2) -> f64` — compare persistence diagrams
- `wasserstein_distance(pd1, pd2, p) -> f64`

### New: `machin-dynamics` — Continuous Dynamics

**Lie Groups / Lie Algebras** (`dynamics/lie.rs`):
- `so3_bracket(a, b) -> Array2<f64>` — Lie bracket [A,B] = AB - BA
- `so3_exp(omega) -> Array2<f64>` — exponential map: so(3) -> SO(3) (Rodrigues' formula)
- `so3_log(R) -> Array1<f64>` — logarithmic map: SO(3) -> so(3)
- `se3_exp(twist) -> Array2<f64>` — SE(3) exponential for rigid body transforms
- `su2_from_quaternion(q) -> Array2<Complex<f64>>` — quaternion to SU(2) matrix
- `pauli_matrices() -> [Array2<Complex<f64>>; 3]` — sigma_x, sigma_y, sigma_z

**Neural ODE** (`dynamics/neural_ode.rs`):
- `NeuralOde { network: fn(&Array1<f64>, f64) -> Array1<f64> }`
- `solve(y0, t_span, dt) -> Vec<(f64, Array1<f64>)>` — forward integration
- `rk4_step(f, y, t, dt) -> Array1<f64>` — Runge-Kutta 4th order
- `dopri5_step(f, y, t, dt) -> (Array1<f64>, f64)` — adaptive Dormand-Prince with error estimate
- `adjoint_solve(...)` — backward pass for gradient computation (later)
- Models dy/dt = f_theta(y, t) — continuous-depth neural network layer

### New: `machin-ktheory` — Algebraic K-Theory

**K-Groups from Graphs** (`ktheory/graph_k.rs`):
- `k0_from_adjacency(A) -> (rank, generators)` — coker(I - A^T) via Smith normal form
- `k1_from_adjacency(A) -> (rank, generators)` — ker(I - A^T)
- `detect_feedback_cycles(A) -> Vec<Vec<usize>>` — eigenvalue-1 detection in A^T
- `resource_invariant(allocations, frees) -> i64` — K0 constraint check (should be 0)

**Mayer-Vietoris** (`ktheory/mayer_vietoris.rs`):
- `consistency_check(shard_a, shard_b, overlap) -> bool` — distributed data verification
- K0(A) + K0(B) - K0(A intersect B) = K0(A union B)

### New: `machin-category` — Category Theory Primitives

**Core Abstractions** (`category/core.rs`):
- `trait Category { type Obj; type Mor; fn compose(...); fn id(...); }`
- `trait Functor<C: Category, D: Category> { fn map_obj(...); fn map_mor(...); }`
- `trait NaturalTransformation<F: Functor, G: Functor> { fn component(...); }`
- `trait Monoidal: Category { fn tensor(...); fn unit(); }`

**Concrete Categories** (`category/instances.rs`):
- `VecCategory` — category of vector spaces with linear maps
- `GraphCategory` — category of graphs with graph homomorphisms
- `compose_functors(F, G) -> ComposedFunctor`

## Key Decisions

- **Quaternion as struct, not Array1<4>**: Named fields (w, x, y, z) are clearer than index-based access. Implement Into/From Array1<f64> for interop.
- **Sedenion multiplication is non-associative**: Must be clearly documented. (a*b)*c != a*(b*c). No trait that assumes associativity.
- **BSP uses const generics for dimension**: `BspNode<const D: usize>` avoids runtime dimension checks.
- **Neural ODE uses function pointers, not trait objects**: Keeps it simple. The "network" is `fn(&Array1<f64>, f64) -> Array1<f64>`. Trait wrapper later if needed.
- **TDA persistence uses matrix reduction**: Standard algorithm with Z/2 coefficients (simplest, most common).
- **Category theory uses Rust traits**: Natural mapping — Rust traits ARE morphism collections. Keep abstract but provide concrete instances.
- **All new code uses f64, ndarray, thiserror, seeded RNG** per workspace conventions.
- **Plucker coordinates bridge to dual quaternions**: Both represent lines/screws in 3D space. Conversion functions connect the two representations.

## Scope per Phase

**Phase 1 (highest value, smallest scope):**
- Quaternions + dual quaternions + SLERP + Plucker coordinates (machin-math)
- Prime utilities (machin-math)
- Takagi + de Rham curves (machin-chaos)

**Phase 2 (medium scope):**
- Sedenions + BSP (machin-math)
- Lie groups/algebras (machin-dynamics)
- K-theory for graphs (machin-ktheory)

**Phase 3 (largest scope):**
- TDA / persistent homology (machin-topo)
- Neural ODEs (machin-dynamics)
- Category theory (machin-category)

## Open Questions

- Should quaternion ops be generic over `f32`/`f64` or stick with `f64` per convention? (Recommendation: `f64` only, consistent with workspace)
- Should `machin-dynamics` depend on `machin-math` quaternions for Lie group <-> quaternion conversion? (Recommendation: yes, small directed dependency)
- Should TDA persistence computation support coefficients beyond Z/2? (Recommendation: Z/2 only for now, Z/p later if needed)

## Next Steps

-> `/ce:plan` for Phase 1 implementation details (quaternions + primes + fractal curves)
