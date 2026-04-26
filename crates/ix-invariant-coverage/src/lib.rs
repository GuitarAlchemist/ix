//! Invariant-catalog coverage analysis.
//!
//! Reads a markdown catalog of claimed invariants (format: `docs/methodology/invariants-catalog.md`
//! in the GA repo) and a firings JSON from external test runs, then computes:
//!
//! - **Rank** — the GF(2) rank of the invariant×exemplar matrix. Equals the number of
//!   linearly-independent invariants. If `rank < invariants.len()`, the catalog is
//!   *denormalized*: at least one invariant is a linear combination of the others.
//! - **Redundancy pairs** — invariants whose firing sets are identical across all exemplars.
//!   Strict duplicates; first candidates for removal.
//! - **Orphan invariants** — rows with zero firings. Either untested (status N) or tested
//!   but never triggered (possibly unreachable).
//! - **Coverage gaps** — exemplars no invariant fires on. Known bugs/edge-cases that slip
//!   through the entire catalog.
//!
//! The threefold judgement of *optimal*:
//!
//! 1. `rank == invariants.len()` — no redundancy.
//! 2. no coverage gaps — every known failure mode is caught.
//! 3. no orphan invariants — every claim pulls its weight.

pub mod coverage;
pub mod invariant;
pub mod producer;
pub mod report;

pub use coverage::{CoverageMatrix, OptimalityVerdict, Report as CoverageReport};
pub use invariant::{parse_catalog, Invariant, InvariantStatus};
pub use producer::{produce_firings, PcSet};
