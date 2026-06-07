# Thinking-machine dogfood — 2026-06-07 (what the instrument says to build next)

Ran a 20-request realistic batch through `ix pipeline compile` (compile-only, no
`--run`; default Opus proposer `claude-opus-4-8`; default TF-IDF coverage gate),
then read the `hits.jsonl` instrument (#80) + `gaps.jsonl`.

## Instrument (cumulative, n=32)

| field | value | reading |
|---|---|---|
| `yield_rate` | **0.281** | metric — fraction that compiled |
| `coverage_refusal_rate` | **0.719** | guardrail — fraction refused at coverage/relevance |
| `governance_refusal_rate` | 0.000 | governance never refused |
| `translate_fail_rate` | 0.000 | proposer never produced an unrepairable spec |

This batch's 24 refusals: **22 `semantic-noncoverage`** (proposer emits NO_COVERAGE
*after* the cheap pre-gate passes) + 2 `coverage` (TF-IDF pre-gate). So refusals
happen at the **proposer/catalog-match stage**, not the pre-gate.

## Diagnosis: the bottleneck is catalog breadth, not the gate, model, or lower()

The gate, `lower()`, repair loop, and governance are all working: **0% translate-fail,
0% governance-refusal, and no confabulation** — the 4–9 successes map to real skills
(e.g. "standardize → PCA → classifier" correctly compiled to the composite
`ml_pipeline` skill with `normalize:true, pca_components:5, task:classify`, not a
faked PCA stage). The machine honestly refuses what it can't serve instead of
inventing specs — exactly the design goal.

The yield ceiling is the **pipeline-callable catalog**: exactly **52 arity-1 skills**,
a thin slice of IX's real capability surface. Refusals fall in three categories:

1. **Genuine catalog gaps — algorithm EXISTS in a library crate, but is not wrapped
   as a callable `#[ix_skill]`:**
   - **PCA / dimensionality reduction** — refused "compute the principal components",
     "PCA then k-means". `ix-unsupervised` implements it (`fit_transform`, `components`,
     `explained_variance_ratio`, `reconstruct`) + `classical_mds` + t-SNE. Only
     reachable today as a *preprocess flag inside* `ml_pipeline`, never standalone.
   - **DBSCAN** — refused "cluster with DBSCAN". `ix-unsupervised/src/dbscan.rs` exists;
     only `kmeans` is callable.
   - **eigendecomposition** — refused "compute the eigenvalues". `ix-math/src/eigen.rs`
     + `svd.rs` exist; neither is a callable skill.
2. **Composite-only access (no composition):** PCA exists only bundled inside
   `ml_pipeline`, so "PCA then k-means" can't be expressed — you can't pipe
   `ml_pipeline`'s internal PCA into `kmeans`. Standalone primitives unlock this.
3. **Honest capability boundaries (correct refusals, not bugs):** `nn.forward` is a
   forward pass, so "train an NN for 100 epochs" is correctly refused; `optimize`
   minimizes *benchmark* functions (Rosenbrock…), not an arbitrary user objective;
   `topo` is *topological data analysis* (persistent homology), not topological sort.
   Multi-array supervised skills (`random_forest` x_train/y_train/x_test,
   `linear_regression` X/y) are shoehorned into one data-flow socket and need inline
   data an abstract request doesn't carry.

## Next increment (data-driven): widen the callable catalog

Wrap the **already-implemented** library algorithms as arity-1 `#[ix_skill]`s, highest
refused-frequency first — **PCA**, **DBSCAN**, **eigendecomposition** (all exist in
`ix-unsupervised`/`ix-math`; pure additive registration + a test each). Then re-run
this batch to confirm yield rises via **real compilations that execute**, not
confabulation.

**Guardrail (from the instrument's own note):** a yield gain paired with a
coverage-refusal *drop* is gate-loosening unless the new skills genuinely work.
Because these algorithms already exist and are tested in their crates, exposing them
is a *legitimate* yield gain — but each new skill must ship with a test proving it
executes (no green-but-dead catalog entries). Add **one primitive at a time, each
driving a re-run that shows a real compiled+executable spec.**

Structural follow-ups (lower priority): standalone PCA enables PCA→kmeans
composition; multi-input pipeline stages for (X, y) supervised learning; `silhouette`
and graph `topological_sort` need implementing (not found as core ML).

Reproduce: `target/debug/ix.exe pipeline compile "<request>" --format json`;
aggregate via `ix pipeline hits`. Raw batch: `.ix/dogfood-2026-06-07/results.jsonl`.

---

## Increment 1 (shipped same session): standalone `pca` skill

Wrapped `ix-unsupervised`'s PCA as an arity-1 `#[ix_skill]` (`pca`; handler
`handlers::pca`; schema `data` + `n_components`; output `transformed` /
`explained_variance_ratio` / `components`). Auto-exposed as MCP tool `ix_pca` via
the registry bridge (parity 77→78, registry 52→53). Three executes-tests
(real PCA run + registered-and-arity-1 + too-many-components rejection).

**Verified effect (re-ran the refused PCA requests):**
- ✅ **Composition unblocked:** "run PCA to reduce these vectors then cluster them
  with k-means" now **compiles to a real 2-stage DAG** — `reduce {skill: pca}` →
  `cluster {skill: kmeans, data:{from: reduce.projected}}`. That chain was
  impossible before (no standalone PCA to pipe into k-means).
- ✅ **Catalog gap closed:** standalone "compute the principal components"
  refusals *changed character* from "no IX skill covers this" to
  **"no input data matrix provided to run PCA on"** (`llm-relevance`,
  deterministic) — the proposer now **recognizes** PCA and refuses only for
  missing data, which is correct (it won't fabricate inputs).

**Two NEW findings this increment surfaced (next-up backlog):**
1. **Data-binding (was a known structural follow-up, now the dominant refusal):**
   abstract requests with no inline data/file/URL are refused because required
   data args can't be filled. The fix is a data-source binding story
   (placeholder/param spec, or `--run` data binding), not more skills.
2. **Output-schema gap (NEW, exposed by the first multi-stage composition):**
   `catalog_value` (compile.rs) exposes only `name`/`doc`/`governance_tags`/
   `args_schema` — **no output schema**. So cross-stage refs *guess* the upstream
   output field name: the PCA→kmeans spec referenced `reduce.projected` but the
   handler emits `transformed`, so the compiled spec would **fail at `--run`**.
   This means "compiled" ≠ "executable" for multi-stage pipelines. Highest-value
   next fix: add each skill's output field names to the catalog (and/or validate
   cross-stage refs in `lower()`), so compositions execute, not just compile.

---

## Increment 2 (shipped same session): output schemas — compositions now EXECUTE

Closed finding #2. Skills can now declare an **output schema**, the proposer is
told the real field names, and wrong refs are caught as a repair signal:

- **Macro + registry:** added optional `output_schema_fn` to `#[ix_skill]` →
  additive `SkillDescriptor.output_schema` (default `Null` = undeclared; every
  existing skill unaffected).
- **Catalog:** `catalog_value` now exposes `output_schema` when declared; the
  system prompt instructs the proposer to reference upstream outputs ONLY by
  those declared field names (or `{from: "stage"}` for the whole output).
- **Repairable validation:** `validate_output_refs` (in `parse_and_lower`, inside
  the repair loop) rejects a `{from:"stage.field"}` whose `field` isn't in the
  producer's declared output schema — fail-OPEN for skills without a schema, so
  no existing pipeline regresses. 4 unit tests.
- **Declared output schemas** for `pca` + `kmeans` (the first composition pair).
- **On-theme bug fixed:** the `kmeans` handler *required* `max_iter` despite the
  schema marking it optional (default 100) → any composition that omitted it
  failed at `--run`. Now defaults to 100.

**Verified end-to-end:** the proposer now emits `from: reduce.transformed` (the
declared field, not the guessed `projected`); a hand-authored pca→kmeans spec
**executes through `ix pipeline run`**, producing real cluster labels
(`[1,0,1,1,1,1,1,0]`) — "compiled" is now "executable" for this composition.

This is also the first link of the **chain-of-evidence** thread (producer→consumer
field bindings are now provably valid). Next on that thread: per-run provenance
record (`stage→skill→input refs→output hash`) + hexavalent certainty propagation.

Remaining catalog primitives (still next-up): DBSCAN, eigendecomposition — and
the structural data-binding follow-up (finding #1) is now the dominant refusal.

Remaining catalog primitives (same pattern as `pca`, next passes): **DBSCAN**
(`ix-unsupervised/src/dbscan.rs`), **eigendecomposition** (`ix-math/src/eigen.rs`
+ `svd.rs`). Each: arity-1 `#[ix_skill]` + handler + executes-test + parity bump
+ dogfood re-run.

---

## Increments 3 & 4 (shipped same session): catalog breadth + data-binding

- **`dbscan`** (#87) — wrapped `ix-unsupervised` DBSCAN; an adversarial review of
  the diff caught that `Clusterer::fit_predict`'s default re-derives labels via
  `predict()`, which absorbs density-unreachable noise points → non-canonical
  labels. Fixed at source (override `fit_predict` to return `fit()`'s canonical
  labels; invisible to the stable-surface api_hash since it's a trait-method body).
- **`eigen`** (#88) — wrapped `ix-math::symmetric_eigen`; validates square +
  symmetric (the Jacobi solver silently mis-solves asymmetric input). Review
  caught the executes-test couldn't distinguish a wrong eigenvector transpose
  (2×2 with component-symmetric V); replaced with a discriminating 3×3, mutation-
  proven.
- **Data-binding via `{param}`** (#89) — closed finding #1, the dominant refusal.
  Makes the dead `PipelineSpec.params` field live: `{"param":"NAME"}` placeholders
  bound at run from `--param`, fail-closed on unbound across all three execution
  sites (run / compile / editor). Two review passes (P0: compile path didn't bind).
  **Live-verified:** `ix pipeline compile "reduce this dataset to 2 dimensions
  with PCA"` now returns `status: compiled` + `params_needed: [dataset]` (was a
  refusal), and `run --param dataset=[[...]]` executes the real PCA projection.

Registry 52 → 55 (pca/dbscan/eigen). Gap rows in `gaps.jsonl` now CLOSED: the PCA
family, eigenvalues, DBSCAN, and the "no inline data" refusals. Still OPEN (next
catalog targets, same pattern): topological sort (DAG), random-forest feature
importances, silhouette score, standalone gradient-descent on a user objective.
**Next:** a fresh dogfood batch to re-measure the refusal rate now that the
catalog is broader and `{param}` exists.
