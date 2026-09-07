//! ix — Claude Code ML skill CLI, library entry point for tests.

pub mod output;
#[cfg(feature = "embeddings")]
pub mod local_embed;
pub mod verbs;

/// Force-link every crate containing `#[ix_skill]` registrations, so
/// `linkme::distributed_slice` elements survive LTO dead-code elimination.
///
/// **If you add a new crate with `#[ix_skill]` annotations, add it here.**
pub mod force_link {
    // Re-export `ix_agent::skills` so its `batch1`/`batch2` modules are
    // reachable from ix-skill's link graph. Without this, LTO strips the
    // distributed-slice elements and the registry appears empty.
    pub use ix_agent::skills as _ix_agent_skills;
}

/// Hexavalent exit codes — map tetra/hexa verdicts to process exit codes.
/// See `governance/demerzel/logic/hexavalent-logic.md`.
pub mod exit {
    pub const OK_TRUE: i32 = 0;
    pub const PROBABLE: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const DOUBTFUL: i32 = 3;
    pub const FALSE: i32 = 4;
    pub const CONTRADICTORY: i32 = 5;

    /// Exit code for a plain runtime error (not a governance verdict).
    pub const RUNTIME_ERROR: i32 = 10;
    /// Exit code for a CLI usage error.
    pub const USAGE: i32 = 64;
}
