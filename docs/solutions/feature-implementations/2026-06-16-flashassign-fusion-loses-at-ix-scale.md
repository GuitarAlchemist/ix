---
title: "FlashAssign kernel fusion loses to materialize-then-argmin at IX scale"
category: feature-implementations
date: 2026-06-16
tags: [gpu, wgpu, wgsl, kmeans, flash-kmeans, performance, spike, ix-gpu, io-aware]
symptom: "Wanted to know if Flash-KMeans' FlashAssign (fuse distance + argmin, never materialize the N×K matrix) would speed up GPU clustering/assignment in ix-gpu."
root_cause: "On a consumer GPU at IX's data scale, memory bandwidth is NOT the bottleneck — materializing even a 114 MB N×K matrix is trivial, and the materialized 2-kernel path exposes far more parallelism (N×K threads vs N threads) than the fused per-point kernel."
---

# FlashAssign fusion loses to materialize-then-argmin at IX scale

## Context

Flash-KMeans (arXiv:2603.09229, `github.com/svg-project/flash-kmeans`) is an IO-aware
*exact* k-means reporting 200× over FAISS / 33× over cuML on an NVIDIA **H200**. Its headline
trick, **FlashAssign**, fuses distance computation with an online argmin so the N×K distance
matrix is never written to global memory (the FlashAttention idea, applied to k-means
assignment). It ships as NVIDIA-only Triton kernels — incompatible with IX's pure-Rust + wgpu
(cross-vendor, f32 WGSL) stance, so adoption was never on the table. The question was whether
the *idea* transfers to WGSL and helps IX.

## The spike

`crates/ix-gpu/examples/flash_assign_spike.rs` implements the k-means **assignment step**
(assign N points to nearest of K centroids) three ways and asserts they agree:

- **MATERIALIZED** — kernel A writes the full N×K squared-distance matrix to a global buffer;
  kernel B reads it back row-by-row and argmins. (N×K global write + read.)
- **FUSED (FlashAssign)** — one kernel, one thread per point, running argmin over K centroids in
  registers; writes only N assignments + N min-dists. The N×K matrix is never materialized.
- **CPU** — the correctness oracle.

Run on an **NVIDIA RTX 5080 (Vulkan)**, median of 5 reps (after warmup):

| N | K | D | N×K matrix | materialized | fused | fused speedup |
|---|---|---|-----------|-------------|-------|---------------|
| 100 000 | 128 | 64 | 48.8 MB | 5.04 ms | 12.54 ms | **0.40×** |
| 200 000 | 64 | 128 | 48.8 MB | 15.39 ms | 21.04 ms | 0.73× |
| 50 000 | 512 | 64 | 97.7 MB | 9.05 ms | 11.21 ms | 0.81× |
| 100 000 | 256 | 128 | 97.7 MB | 22.44 ms | 46.42 ms | 0.48× |
| 15 000 | 2000 | 32 | 114.4 MB | 6.32 ms | 7.68 ms | 0.82× |

Both GPU paths matched the CPU oracle in every config.

## Finding

**At IX scale on a consumer GPU, the fused path never wins — it is 1.2×–2.5× *slower*.** The
materialized path is faster because (1) the GPU has bandwidth to spare, so writing+reading even a
114 MB matrix is cheap, and (2) kernel A launches N×K threads (millions) vs the fused kernel's N
threads (hundreds of thousands), giving far better occupancy; the naive fused kernel also re-reads
all K×D centroids from global memory per point with no reuse.

This is exactly the signal that **IX is not in the regime FlashAssign targets.** The 200× number
is a bandwidth-bound, many-GB-matrix, datacenter-GPU regime (huge N *and* K on an H200). IX clusters
small corpora (voicings, telemetry) where the matrix fits comfortably and bandwidth is not scarce.

## Caveat (what this does NOT prove)

The fused kernel here is the *naive* register-only form. A production FlashAssign tiles centroids
into **workgroup shared memory** so a workgroup cooperatively loads centroid blocks once instead of
every thread re-reading them. That could close or reverse the gap and is the only variant worth
trying if this is ever revisited. The spike disproves "naive fusion helps at our scale," not "the
FlashAssign idea is worthless."

## Decision

**Demand-gate / do not pursue.** Don't build a GPU k-means around FlashAssign on this evidence; the
simpler, more-parallel materialized assignment is faster at every scale IX actually uses. Revisit
*only* if IX starts clustering bandwidth-bound, many-GB matrices — and then start from the
shared-memory-tiled variant, not the naive one. The existing CPU k-means (`ix-unsupervised`) and the
brute-force `ix_kdist` remain the right tools at current scale.

Reproduce: `cargo run -p ix-gpu --example flash_assign_spike --release -- <N> <K> <D>`.
