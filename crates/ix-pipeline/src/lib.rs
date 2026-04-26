//! DAG pipeline executor for skill orchestration.
//!
//! Define pipelines as directed acyclic graphs where nodes are compute
//! functions and edges carry typed data. Independent branches run in
//! parallel, results are memoized, and the whole thing is cache-friendly.

pub mod builder;
pub mod dag;
pub mod executor;
pub mod lock;
pub mod lower;
pub mod spec;
