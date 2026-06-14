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

---

## Re-measure (same session): yield 26% → 66%

Split `hits.jsonl` by `ts_ms` (the `ix pipeline hits` cumulative mean blends
pre/post and reads falsely low at 0.43 — see
`docs/solutions/workflow-patterns/2026-06-07-dogfood-yield-before-after-measurement.md`):
**pre-fix 11/42 = 26% → post-fix 21/32 = 66%** (in-domain ~87%),
`translate_fail_rate` **0%** (no confabulation; topo-sort/shortest-path verified
as real `ix_graph` ops, not hallucination). Remaining refusals are predominantly
*correct*: OOD (haiku, booking, news/scrape) + honest boundaries (NN training,
custom-objective GD) + governance correctly rejecting "delete the production
database" (Article 3). Conclusion: the **structural** bottlenecks (catalog
breadth + data binding) are resolved; only two genuine catalog gaps remained.

## Increment 5 (same session): the last two genuine gaps closed

Both remaining genuine gaps from the re-measure are now CLOSED — as **separate,
cleanly-scoped** skills (the "separate concerns" steer), each a pure module with
its own unit tests, **kept in `ix-agent` (beta)** so they don't grow a stable
crate's public surface (the workspace shares one `0.1.0` version, so a `pub fn`
in ix-unsupervised/ix-ensemble would trip the stable-surface gate — a multi-PR
version dance for a purely additive change):

- **`silhouette`** (`eval/silhouette.rs`) — clustering evaluation. Exact O(n²)
  Rousseeuw silhouette; `data` socket + `labels` wired from a cluster stage via
  `{from: "cluster.labels"}`. **Proven:** `kmeans → silhouette` compiles AND
  executes → score **0.992** on well-separated clusters.
- **`feature_importances`** (`eval/permutation_importance.rs`) — model
  explainability. **Permutation importance** (model-agnostic; needs only the
  RF's existing `predict()`, so no stable-crate surgery; inline splitmix64 PRNG,
  no `rand` dep, reproducible via `seed`) — *not* Gini/MDI (which would require
  exposing `DecisionTree` internals). **Proven:** `random_forest →
  feature_importances` compiles AND executes → importances `[0.35, 0.0]`,
  `ranking [0,1]`, correctly attributing the label-determining feature.

Registry **55 → 57**. Each ships an executes-test, a genuine `@ai:invariant`
(`::`-pathed `[T:test]` binding), READ_TOOLS classification, and parity/cli count
bumps (EXPECTED 80→82). An adversarial 3-lens review (numerical-correctness vs
sklearn / `@ai`-binding precision / schema↔handler field parity) returned
**merge_ready, no P0/P1** — only P3 honesty nits (doc-comment sklearn-parity
caveat + invariant prose narrowed to what the bound test discriminates), both
applied.

**Catalog status: complete-enough.** The structural arc (catalog breadth +
data binding + the two demand-driven gaps) is closed. Remaining refusals are
correct (OOD + honest boundaries). Next lever, if any, is incremental and
demand-gated — not structural.

---

## Catalog-gap audit + signal/SVD/GMM batch (registry 57 → 63)

Triggered by "do we have A\* and Q\*?" → a workspace-wide gap audit (7 parallel
domain agents + adversarial verification). **120 user-meaningful algorithms
surveyed: 39 exposed, 5 partial, 18 honest-boundary, 58 raw gaps, 15 confirmed
high-value gaps.** Key findings:

- **A\* is a gap, Q\* is an honest boundary.** `ix-search/astar.rs` (+ weighted/
  greedy/bidirectional) and `qstar.rs` (A\* with a learned DQN Q-heuristic) both
  exist, but the catalog's `search` skill is `search_info` (returns *docs*, not
  execution); only `graph` (dijkstra/bfs/dfs/topo/pagerank) runs. A\* is
  arity-1-feasible; **Q\* needs a user-supplied trained Q-function callback**, so
  it can't be a clean data→result skill (like Q-learning, neural-ODE, arbitrary-
  objective optimize — all correctly honest-boundary, not gaps).
- **`ix-signal` was the single biggest hole** — an 11-module crate with only
  `fft` exposed. Filled 4 modules this batch.

Shipped 6 skills, all **pure wraps of already-`pub` library functions** (so —
unlike silhouette/feature_importances — *zero* new code in the stable crates;
the wrappers live in ix-agent/beta, stable-surface untouched):

| skill | wraps | proof |
|---|---|---|
| `svd` | `ix-math::svd` | reconstructs U·diag(s)·Vᵀ; executed end-to-end (σ=[17.4,0.875,0.197], rank 3) |
| `gmm` | `ix-unsupervised::GMM` | recovers 2 blobs; soft responsibilities sum to 1 |
| `wavelet_denoise` | `ix-signal::wavelet` | reduces MSE-to-clean |
| `fir_filter` | `ix-signal::filter` | lowpass cuts high-freq diff-energy |
| `spectrogram` | `ix-signal::spectral` | tone localizes to bin 4 |
| `autocorrelation` | `ix-signal::correlation` | period-4 signal peaks at lag 4 |

All 6 live-compile via the proposer (with `{param}` binding). EXPECTED 82→88,
registry 57→63, drift snapshot 41 claims.

**Adversarial review caught 4 real P1s my power-of-two test inputs masked** (the
value of the review over green CI):
1. `spectrogram` — `rfft` zero-pads to next-pow2, so a non-pow2 `window_size`
   miscalibrates the bins → now **rejected** (require power-of-two window).
2. `wavelet_denoise` — Haar DWT drops samples when length isn't divisible by
   `2^levels`, silently shortening the output → now **rejected** (length contract
   holds for accepted input).
3. `wavelet` `@ai:invariant` over-claimed unconditional length preservation →
   narrowed to the enforced precondition.
4. `gmm` responsibilities could sum to 0 on extreme-outlier underflow → **uniform
   fallback** so every row is a valid distribution.

Each fix ships a test encoding the reviewer's exact counterexample
(`window_size=24`, `length-15`, far outlier `1e9`). The GMM responsibilities
recompute was independently verified to match the library's private
`gaussian_pdf` exactly. **Lesson (recorded):** signal/transform code tested only
on power-of-two sizes hides padding/truncation bugs — always probe non-pow2 /
non-divisible inputs (`docs/solutions/math-correctness/2026-06-07-power-of-two-test-masking.md`).
