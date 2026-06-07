//! Verb dispatch tables — one module per top-level verb.

pub mod beliefs;
pub mod check;
pub mod compile;
pub mod demo;
pub mod describe;
/// Optional in-process embedding coverage scorer (feature `embeddings`).
#[cfg(feature = "embeddings")]
pub mod embed_coverage;
pub mod list;
pub mod pipeline;
pub mod run;
pub mod stable_surface;
