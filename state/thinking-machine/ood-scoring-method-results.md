# OOD scoring-method sweep — score→verdict refinement (2026-06-07)

Resolves the last open front from the embedding-gate increment
(`embedding-sweep-rust-results.md` → "Open (next)"): once the gate embeds with
bge-base-en, does a smarter **score→verdict** method (per-query calibration /
kNN-distance OOD) beat thresholding the raw cosine? Deep-research
(`reference_ood_scoring_method_research`) said yes — raw global cosine is "not
cross-query comparable," so per-query calibration should lift near-miss OOD.

This is the executable-oracle answer: a ROC sweep over the **184 validated
probes** (104 in-domain, 56 near-miss-OOD, 24 far-OOD), every method calibrated
**ID-only** (threshold = 3rd percentile of in-domain scores → recall ≈ 0.97) so
the comparison is apples-to-apples.

Reproduce: `cargo test -p ix-skill --features embeddings ood_scoring_method_sweep_over_probe_corpus -- --nocapture`

## Methods

- **A — raw mean-top-3 cosine** (the production signal).
- **B — per-query z-norm**: `(mean_top3 − μ)/σ` over the query's own 52 catalog
  cosines (the literature's per-query calibration; catalog-only, runtime-cheap).
- **C — top1−top2 margin** (relative, label-free).
- **D — raw × kNN(10)-guidance** to the in-domain query set (NNGuide-style,
  leave-one-out; needs an in-domain reference set).
- **E — fixed in-domain reference z-norm**: `(A − μ_id)/σ_id`, constants computed
  once over in-domain scores. Added after adversarial review (below).

## Results — calibrated ID-only @ recall ≈ 0.97 (n = 184)

| method | AUC | TNR | near-miss TNR | far-OOD TNR |
|---|---|---|---|---|
| **A raw-mean-top3** | **0.936** | **0.675** | **0.554** | 0.958 |
| B per-query-znorm | 0.698 | 0.075 | 0.018 | 0.208 |
| C top1-top2-margin | 0.770 | 0.138 | 0.107 | 0.208 |
| D raw×kNN-guide | 0.929 | 0.662 | 0.518 | 1.000 |
| E fixed-ref-znorm | 0.936 | 0.675 | 0.554 | 0.958 |

(Production gate today: A @ fixed 0.45 = recall 0.99, TNR 0.625, near 0.482.)

## Finding — the research hypothesis is OVERTURNED for this task

**Raw mean-top-3 cosine (A) wins outright.** Every per-query calibration is
strictly worse: B is catastrophic (near-miss TNR **0.018**, 30× worse than raw),
C is weak, D is raw-times-a-factor and loses a little.

**Mechanism.** A *coverage* gate discriminates by **absolute cosine magnitude**:
an in-domain request sits close to *some* skill (one high cosine); a near-miss
sits only moderately close to *all* of them (a uniform, low ridge). That is the
opposite of the *ranking* tasks the literature studied. Per-query z-norm (B)
divides out each query's own spread — and near-miss queries have the *smallest*
spread, so B inflates precisely the cases it must reject. This is the
transfer-gap the research itself flagged ("must validate empirically").

**The equivalence that closes it (method E).** An adversarial reviewer
(verification workflow `ww01cowgl`, 2/3 sound) argued B "should" have normalized
against a *fixed* in-domain reference, not per-query. That variant is method E —
and it is a **monotone affine map** of the raw score, so it is provably identical
to A in AUC (rank-invariant) and in any ID-quantile-calibrated TNR. The sweep
confirms it bit-for-bit: E = A = `0.936 / 0.675 / 0.554`, threshold merely shifts
to −1.644. So the only z-norm that *differs* from raw is the per-query one — and
it loses. **No fixed-reference calibration can beat raw.**

## Decision

**Keep the raw mean-top-3 cosine threshold. Wire NO new scoring method.** The
score→verdict front is closed: the candidate family cannot beat raw, by proof
(fixed-reference ≡ raw) and by sweep (per-query loses). The negative result is
the deliverable.

The **only** real lever is the **operating point**, not the method: moving the
recall floor from production's 0.99 to 0.97 trades recall for near-miss TNR
(0.625 → 0.675 overall, 0.482 → 0.554 near-miss). That is a one-line threshold
choice, deferred to a guarded A/B with a recall guardrail — not a code change.

The sweep ships as a **re-runnable guard**: the test asserts (1) raw dominates
B/C/D on AUC (A excluded from the comparison set — not a tautology), and (2) the
fixed-reference equivalence E ≡ A on all three metrics (AUC/TNR/near). If a
future embedder or probe set genuinely flips this, the test fails loudly and the
keep-raw decision is revisited.

**Enforcement caveat (not a CI gate).** The test is `#[cfg(feature = "embeddings")]`
and CI builds/tests `--workspace` with no feature flags, so it **never runs in
CI** (CI never pulls ONNX) and **skips** (early return, zero assertions) if the
probe corpus is absent. "Fails loudly" means *when a human re-runs the sweep*
after touching the embedder or probes — this is a decision record plus a manual /
PR-time check, not an auto-firing alarm. Re-run:
`cargo test -p ix-skill --features embeddings ood_scoring_method_sweep_over_probe_corpus -- --nocapture`.

Reversibility: **two-way door** (the gate already runs raw cosine; this only
records *not* adding a method). Revisit trigger: a new embedder, a materially
larger probe set, or a calibration method outside the affine/ranking family.
