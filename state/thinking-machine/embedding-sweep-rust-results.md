# Embedding coverage-gate — in-process Rust sweep (2026-06-07)

Validation spike for the embedding coverage gate (step 1 of the embedding-gate
increment). Replaces the earlier Python sweep with an **in-process pure-Rust**
path (`fastembed-rs` on the `ort`/ONNX Runtime crate), per the 2026-06-07
deep-research decision (`reference_embedding_gate_runtime_fastembed`).

Reproduce: `cargo test -p ix-skill --features embeddings embedding_sweep_english_models_over_probe_corpus -- --nocapture`
(feature-gated; CI's `--workspace` default-feature path never pulls ONNX).
First run downloads each model (~270–670 MB) + the ONNX Runtime native lib.

## Setup
- **Corpus:** the 184-probe validated set (`coverage-probes.jsonl`; 104 in-domain,
  56 near-miss-OOD, 24 far-OOD).
- **Signal:** mean of the top-3 cosine similarities between the request and the
  52 pipeline-callable skills' "name + doc".
- **Per-model prefix scheme** (applied caller-side): e5 → `query:` / `passage:`;
  bge / mxbai / arctic → query instruction "Represent this sentence for searching
  relevant passages: " + bare passages; gte → none.
- **Operating point:** highest threshold holding in-domain recall ≥ 0.99 (same
  methodology as the Python sweep, so the numbers are comparable).

## Results — @ recall ≥ 0.99

| model (mean-top-3 cosine) | AUC | threshold | in-domain recall | overall OOD TNR | near-miss TNR | far-OOD TNR |
|---|---|---|---|---|---|---|
| multilingual-e5-base | 0.808 | 0.768 | 0.990 | 0.075 | 0.018 | 0.208 |
| **bge-base-en-v1.5** | **0.936** | **0.454** | **0.990** | **0.625** | **0.482** | 0.958 |
| gte-base-en-v1.5 | 0.881 | 0.347 | 0.990 | 0.375 | 0.107 | 1.000 |
| snowflake-arctic-embed-m | 0.896 | 0.203 | 0.990 | 0.500 | 0.339 | 0.875 |
| **TF-IDF baseline** | ~0.78 | — | 0.990 | 0.163 | 0.036 | 0.458 |

## Findings
1. **Winner: `bge-base-en-v1.5`** — AUC 0.936, overall OOD TNR **0.625** at
   in-domain recall 0.99. That is **3.8× the lexical baseline** (0.163) and beats
   even the Python e5-base-v2 sweep (0.425 / AUC 0.885). It is **fastembed
   built-in** — no custom ONNX export needed.
2. **It closes the headline near-miss gap.** Near-miss OOD TNR goes **0.036 →
   0.482 (13×)** — the catastrophic slice the whole embedding effort targeted.
   Far-OOD is near-solved (0.458 → 0.958).
3. **The spike reversed two assumptions.** `multilingual-e5-base` (the planned
   "start" model) is *worse than TF-IDF* at this operating point (TNR 0.075) —
   the multilingual model is far weaker on English-only discrimination. And
   `e5-base-v2` (the original sweep model) is **not needed** — bge-base-en beats
   it and ships built-in.
4. **Windows ONNX build confirmed.** `fastembed` v5.16.0 + `ort` (native ONNX
   Runtime) compile and run on Windows 11 — the main runtime risk is retired.

## Decision
Productionize the gate with **`bge-base-en-v1.5`** via in-process fastembed-rs,
mean-top-3 cosine, threshold ≈ **0.45**, behind the `embeddings` Cargo feature
(optional-dep boundary per the CLAUDE.md carve-out), with the pure-Rust TF-IDF
gate as graceful fallback. Guardrail: in-domain recall must stay ≥ ~0.97.

Resolved (next): the score→verdict *method* refinement (per-query calibration /
kNN-distance OOD vs the fixed cosine threshold) was the unsettled front — now
closed by `ood-scoring-method-results.md`. The A/B/C/D/E sweep + adversarial
verification (`ww01cowgl`) found the **raw** mean-top-3 threshold is already
optimal within the candidate family: per-query z-norm loses (AUC 0.698), and the
only calibration that wouldn't hurt (fixed in-domain reference) is provably
identical to raw. **Wire no new scoring method**; the only lever is the operating
point. The 3.8× TNR lift is from the embedder, not from any scoring refinement.
