//! GPU-accelerated compute via WGPU.
//!
//! Cross-platform: Vulkan/DX12 on Windows (NVIDIA), Metal on macOS.
//! Provides cosine similarity, dot product, matrix multiply, and batch vector search.

pub mod batch;
pub mod bsp_knn;
pub mod context;
pub mod distance;
pub mod knn;
pub mod matmul;
pub mod quaternion;
pub mod rips;
pub mod sedenion;
pub mod similarity;
