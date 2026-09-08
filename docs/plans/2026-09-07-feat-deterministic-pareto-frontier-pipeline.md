# Deterministic Pareto frontier pipeline (issue #294)

- **Date**: 2026-09-07
- **Issue**: [GuitarAlchemist/ix#294](https://github.com/GuitarAlchemist/ix/issues/294)
- **Status**: design (this commit) then implementation
- **Reversibility**: two-way door. New `ix-evolution::frontier` module on an
  `experimental`-tier crate (`crate-maturity.toml`), plus SQL + fixtures under
  `crates/ix-duck/`. No schema hash, no public stable-tier API, no Galactic
  Protocol contract. Revisit trigger: first Gaia consumer that needs ranks
  beyond front 0, or a metric set large enough that the O(n^2 * m) pairwise
  join stops being free.

## 1. Prior art - what already ships

Before proposing anything, the three crates named in the brief were read.

| Crate | What it is | Verdict for #294 |
|---|---|---|
| `ix-evolution::pareto` (`crates/ix-evolution/src/pareto.rs`) | **Deterministic non-dominated sorting.** `rank(&[Candidate], &[Objective]) -> ParetoArchive`. Already validates empty/duplicate-id/arity/non-finite, already sorts every front by candidate id, already carries `// @ai:invariant Pareto ranks are independent of candidate row order` bound to `tests/pareto.rs`. | **REUSE.** This *is* the dominance kernel #294 needs. Re-deriving dominance in a new crate would be a shallow duplicate of a shipped, order-independent, tested primitive. |
| `ix-optimize` | Single-objective iterative optimizers - `ObjectiveFunction::evaluate -> f64`, `Optimizer::step(params, gradient)`, annealing / gradient / PSO. | **Not applicable.** #294 ranks an *already evaluated* table; there is no search space, no gradient, and the objective is a vector, not a scalar. |
| `ix-pipeline::dag::Dag<N>` | Generic DAG with cycle rejection, `topological_sort`, `parallel_levels`, `critical_path`. | **Not used, deliberately.** The Pareto pipeline is a fixed 4-stage linear transform (validate, pivot, rank, order) with no user-supplied graph. Cycle rejection guards nothing and `parallel_levels` on a chain returns four singletons. Wrapping it in a `Dag` is ceremony, not reuse. |

So the honest scope of #294 is **not** "implement Pareto". It is the layer the
kernel does not have: the **long-form objective-table contract** - fail-closed
validation of a `(subject_revision, task_class, candidate_id, metric,
direction, value)` relation, per-group isolation, a canonical metric order, and
a DuckDB surface. That layer is new; the dominance rule is borrowed.

## 2. Shape

```
long-form rows --> validate (fail closed) --> pivot to wide, metric ASC
                                                       |
                                    ix_evolution::pareto::rank  <-- REUSED
                                                       |
                                              front 0 --> canonical order
```

Two coordinated surfaces, pinned to **one** frozen golden file:

1. **Rust primitive** - `ix-evolution::frontier`, built and tested by the
   default CI path (`cargo test --workspace`). This is the reusable primitive
   IX owns.
2. **DuckDB SQL** - `crates/ix-duck/sql/pareto_frontier.sql`, table macros run
   by the real `duckdb` CLI. This is what Gaia can call without linking IX.

Neither is the source of truth for the other; **the golden CSV is**. Both are
asserted byte-identical against it, so drift in either is a test failure.

### Why not a Rust UDF in `ix-duck`?

The DuckDB code in `ix-duck` sits behind the optional `duck` / `udf` features,
which `cargo build --workspace` and CI never compile (`crates/ix-duck/Cargo.toml`
says so explicitly). A UDF would put the determinism logic where **no CI job
can see it**. Pure SQL macros run on the stock `duckdb` CLI need no build, and
the Rust half stays in a CI-visible crate.

## 3. Determinism: the total order and the tie rule

Determinism has to be argued, not asserted. Four places can leak order.

**(a) Output row order.** Rows are ordered by the triple
`(subject_revision, task_class, candidate_id)`, compared as byte sequences.
Validation rejects duplicate `(revision, task_class, candidate_id, metric)`,
which makes that triple a **key** over output rows - at most one output row per
triple. A lexicographic order on a key is *total*: for any two distinct output
rows the triples differ in at least one component, and byte comparison of
distinct byte strings is antisymmetric and never returns "equal". There is no
residual tie for a secondary rule to break. Rust compares `String` by bytes;
the default DuckDB collation is binary; fixture ids are ASCII, where the two
coincide.

**(b) Objective-value ties.** Two candidates with *identical* objective vectors
are mutually non-dominated (neither is strictly better anywhere), so **both stay
on the frontier** - a tie in values never collapses or drops a row, it just
yields two rows that (a) then orders by `candidate_id`. This is the single
tie-break rule and it is why (a) suffices.

**(c) Column order inside a row.** Metric names within a group are sorted
ascending, so the objective vector is positionally identical no matter what
order the metrics arrived in. Emitted as `metric:DIRECTION=value` with `%.6f` /
`{:.6}` fixed-precision formatting, which the C and Rust formatters agree on;
dominance itself always uses full `f64`.

**(d) Aggregate order-sensitivity in SQL.** Dominance in SQL is `bool_and(...)
AND bool_or(...)` per ordered pair - both are commutative and associative over
booleans, so DuckDB scan and parallelism order cannot change the result. The
Rust half is an O(n^2) pairwise scan whose per-pair predicate is order-free.

## 4. Fail-closed validation

Any violation aborts the whole run; there is no partial or best-effort output.

| Rule | Rust error | SQL |
|---|---|---|
| empty revision / task class / candidate id / metric | `EmptyField` | `error()` |
| direction not `MIN`/`MAX` | `UnknownDirection` | `error()` |
| duplicate `(rev, class, candidate, metric)` | `DuplicateMetric` | `error()` |
| same `(rev, class, metric)` with two directions | `MixedDirection` | `error()` |
| a candidate missing a metric its task class declares | `MissingMetric` | `error()` |
| non-finite value (`inf`, `-inf`, `NaN`) | `NonFiniteValue` | `error()` |
| no rows at all | `EmptyInput` | `error()` |

The order of the checks is itself fixed (the table order above) so the *error
message* is deterministic too, not just the success path.

## 5. Tests that can actually fail

- minimization-only, maximization-only, and mixed-direction frontiers
- ties: identical objective vectors, **both** on the frontier
- incomparable candidates, all retained
- cross-task isolation: the same `candidate_id` is **on** one task class
  frontier and **off** another one in the same revision; a second revision
  reuses the same ids with different values
- order independence: shuffle input rows (reverse plus a fixed permutation),
  byte-identical output
- each validation rule gets a rejecting test
- **mechanism-revert control**: the test file carries a deliberately weakened
  frontier (the strict-improvement requirement deleted) and asserts it produces
  a *different* answer on the golden fixture. If someone removes strict
  dominance from the real implementation, that control assertion is what
  breaks. A test that cannot fail proves nothing; this one proves the fixture
  discriminates.
- golden cross-check: Rust output equals the frozen CSV, and `duckdb` CLI
  output equals the same frozen CSV.

## 6. Out of scope

Fronts beyond rank 0, crowding distance, hypervolume, weighting, any LLM,
network, Docker, credential, or Gaia runtime dependency. Output is advisory
and read-only - nothing in this pipeline writes state or gates a merge.
