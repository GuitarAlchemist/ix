# Embedding coverage-gate ROC sweep — results (2026-06-07)

Experiment for the D4 "embeddings coverage" front. Reproduce:
`python scripts/embedding_coverage_sweep.py` (needs `transformers`+`torch`; one-time
HuggingFace model download). Evaluated against the 184-probe validated corpus
(`coverage-probes.jsonl`); compared to the lexical TF-IDF baseline.

## Setup

- **Model:** `intfloat/e5-small-v2` and `intfloat/e5-base-v2` — asymmetric,
  query/passage-prefixed retrieval embedders (the deep-research-recommended
  pattern, `reference_embedding_coverage_gate_research`). Skill docs embedded as
  `passage: <doc>`, NL intent as `query: <intent>`; L2-normalized; cosine.
- **Signals:** top-1 cosine, mean-top-3 cosine, top1−top2 margin.
- **Decision:** in-domain iff signal ≥ threshold; sweep thresholds, pick the
  operating point maximizing OOD TNR subject to in-domain recall ≥ target.

## Results

| config | AUC | @recall≥0.99: overall TNR | near-miss | far |
|---|---|---|---|---|
| TF-IDF @0.08 (baseline) | — | 0.163 | 0.036 | 0.458 |
| e5-**small** top-1 cosine | 0.78 | ~0.14 | 0.071 | 0.292 |
| e5-**base** top-1 cosine | 0.885 | 0.388 | 0.179 | 0.875 |
| **e5-base mean-top-3 cosine** | 0.876 | **0.425** | **0.232** | **0.875** |

## Findings

1. **Model size dominates.** e5-small (33M) barely beats TF-IDF (AUC 0.78);
   e5-base (110M) jumps to AUC 0.885. Confirms the research's "small models are
   weaker" caveat — don't ship a tiny embedder.
2. **mean-top-3 cosine is the recommended signal/threshold:** e5-base, threshold
   ≈ **0.76**, at in-domain recall 0.99 → overall OOD **TNR 0.425** (2.6× the
   TF-IDF baseline 0.163), near-miss 0.232 (6.4×), far-OOD 0.875 (1.9×).
3. **The top1−top2 margin signal is weak** (AUC 0.67–0.76) — matches the
   research's "raw-cosine geometry tricks don't help."
4. **Near-miss OOD remains the ceiling** (TNR ~0.23). "run a t-test" / "train an
   image classifier" are genuinely *near*-OOD — semantically adjacent ML tasks
   IX doesn't implement. Embeddings give a real 6× lift but don't *solve* it; the
   fail-open LLM relevance tier keeps a role as the last catcher.

## Decision

Productionize as a **Python sidecar** (per the 2026-06-07 runtime decision +
the revised CLAUDE.md inference-only carve-out): e5-base, mean-top-3 cosine,
threshold ≈ 0.76, precomputed catalog embeddings, query embedded at gate-time;
**optional with a pure-Rust TF-IDF fallback** so the gate degrades gracefully.
Guardrail: in-domain recall must stay ≥ ~0.97.
