# `ix_code_health` — a code-health map of the IX workspace, computed by IX over IX

A runnable demo that turns IX's own source tree into a queryable code-health dataset on
the DuckDB bench — **pure dogfood**, no external data, runs on any checkout.

> _Version française : [`docs/fr/pas-a-pas/sante-code.md`](../fr/pas-a-pas/sante-code.md)._

```bash
cargo run -p ix-duck --example ix_code_health --features duck
```

## The question

> **Which crates are health outliers — and does code-health predict change-proneness
> (git churn)?**

`read_text('crates/**/*.rs')` streams every Rust file into the bench; the `ix_code_*`
UDFs score each (cyclomatic + cognitive complexity, SLOC, lexical smells); a `GROUP BY`
rolls them up per crate. 79 crates, 632 files, ~154 k SLOC — all analysed in one query.

## Three stages

**1. Health table.** Per-crate complexity and smell density. The smell-density leaders
(`ix-net`, `ix-bracelet`, `ix-manifold`, `ix-dynamics`, `ix-fractal`) are all **math
crates** — a hint that lexical smell density partly measures *numeric-literal density*,
not "unhealthiness". That confound is exactly what stage 3 tests.

**2. Map.** `ix_pca_project` reduces the standardized per-crate feature vectors
`[mean_cc, max_cc, mean_cog, smell_density, ln SLOC]` to 2-D; the crates farthest from
the centroid are the multivariate outliers:

```text
ix-duck-ext  dist 9.74  (mean_cc 65.9 — the macro-heavy C-API extension)
ix-agent     dist 5.64
ix-net       dist 4.26  (smells/KLOC 468)
```

**3. Validation — does health predict churn?** Per-crate git churn (commits touching
`crates/<name>/`, normalized to commits/KLOC so the test isn't just "bigger crates change
more") is correlated against each health metric, with **2 000 permutation nulls** for
significance (the analog of the mesh's null model — an external-validity check matched to
a *predictive* claim):

| Predictor | r vs churn/KLOC | p (permutation) | Verdict |
|---|---|---|---|
| mean cyclomatic | −0.15 | 0.18 | **not predictive** |
| smell density | −0.12 | 0.30 | **not predictive** |

## The finding (an honest negative)

**Neither complexity nor lexical smell density predicts churn in this workspace** — both
correlations are within the null range, and even slightly *negative*. The negative sign
fits the stage-1 hint: smell density mostly tracks math-heaviness, and those numeric-kernel
crates (`ix-bracelet`, `ix-manifold`, …) are *mature and stable*, not churny. So the
actionable result is: **don't use these lexical smells as a change-risk proxy here** — they
have no measured external validity for churn.

A negative that survives a null model is worth more than a positive that never faced one.

## Scope and caveats

- **Advisory only** — a health *lens*, not a gate.
- `ix_code_smells` is **lexical** (heuristic: TODOs, magic numbers, long lines, …), so
  density is noisy and confounded by domain (math code trips more). Tier B
  (`code-semantic`, tree-sitter) would sharpen it.
- Churn is also driven by **crate age**, which this doesn't control for — a newer crate
  has few commits regardless of health. With n = 79 crates the test is low-powered; read
  the negative as "no *detectable* signal", not "proven independence".
- The data is the repo itself, so there's no corpus fallback — it runs anywhere `git` +
  the source tree are present.
