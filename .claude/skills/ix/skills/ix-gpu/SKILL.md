---
name: ix-gpu
description: GPU compute via WGPU — cosine similarity, matrix multiply, batch vector search
disable-model-invocation: true
---

# GPU Compute

Cross-platform GPU computation via WGPU (Vulkan/DX12/Metal).

## When to Use
When the user needs GPU-accelerated operations: batch cosine similarity search, matrix multiplication, distance computation, or batch quaternion/sedenion operations on large datasets.

## Capabilities
- **Cosine Similarity** — GPU batch cosine similarity search across vector collections
- **Matrix Multiply** — WGPU shader-based matrix multiplication (f32)
- **Distance Matrix** — Pairwise distance computation on GPU
- **Batch Operations** — Quaternion transforms, sedenion multiplication, BSP kNN queries
- **Rips Complex** — GPU-accelerated Vietoris-Rips complex construction

## Method Selection
- **Small data** (<1000 vectors) — CPU is faster (no GPU dispatch overhead)
- **Large batches** (>10000) — GPU provides significant speedup
- **No GPU available** — All operations have CPU fallbacks

## Programmatic Usage
```rust
use ix_gpu::similarity::GpuCosineSimilarity;
use ix_gpu::matmul::GpuMatMul;
use ix_gpu::distance::GpuDistanceMatrix;
use ix_gpu::context::GpuContext;
```

## MCP Tool Reference
Not yet available as a dedicated MCP tool (requires GPU hardware). Use the Rust API directly.
