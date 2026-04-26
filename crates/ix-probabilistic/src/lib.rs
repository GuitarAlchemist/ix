//! # ix-probabilistic
//!
//! Space-efficient probabilistic data structures.
//! Useful for agent routing, deduplication, and cardinality estimation
//! where approximate answers save significant resources.

pub mod bloom;
pub mod count_min;
pub mod cuckoo;
pub mod hyperloglog;
