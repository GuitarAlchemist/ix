//! Verdict types: [`Tier`], [`BlastRadius`], [`Evidence`], and the
//! aggregate [`ApprovalVerdict`] that the middleware mounts on every
//! action.

use serde::{Deserialize, Serialize};

/// Auto-mode-style tier classification for an action.
///
/// Tier assignment is **deterministic** — it's a pure function of the
/// action's `ActionKind`, its target file(s), and the configured
/// thresholds. No LLM classifier, no heuristic scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Auto-approved: reads, searches, in-project metadata queries.
    /// No side effects outside the workspace.
    One,
    /// Auto-approved with audit: in-project edits with a small blast
    /// radius and stable trajectory. Git provides the safety net.
    Two,
    /// Requires explicit approval: cross-module edits, shell commands,
    /// web fetches, out-of-project writes, fragile hot spots.
    Three,
}

impl Tier {
    /// `true` iff this tier auto-approves without human intervention.
    pub const fn is_auto_approved(self) -> bool {
        matches!(self, Tier::One | Tier::Two)
    }

    /// `true` iff this tier requires explicit approval.
    pub const fn requires_approval(self) -> bool {
        matches!(self, Tier::Three)
    }

    /// Short canonical name used in metadata paths and session events.
    pub const fn name(self) -> &'static str {
        match self {
            Tier::One => "tier_one",
            Tier::Two => "tier_two",
            Tier::Three => "tier_three",
        }
    }
}

/// Blast radius — how much of the workspace the action touches, directly
/// or transitively.
///
/// MVP uses a constant for in-project edits (no `ContextBundle`
/// integration yet). v2 will replace the constants with real counts
/// from an `ix_context::ContextBundle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BlastRadius {
    /// Number of nodes (functions, modules, files) transitively
    /// affected. `0` for reads, `1` for a single-file edit, `N` for a
    /// multi-file or cross-module edit.
    pub nodes_touched: usize,
    /// Distinct files touched.
    pub files_touched: usize,
    /// Distinct crates touched. `> 1` means the action crosses crate
    /// boundaries and is automatically Tier 3.
    pub crates_touched: usize,
}

impl BlastRadius {
    /// Zero-impact blast radius — the action touches nothing.
    pub const fn zero() -> Self {
        Self {
            nodes_touched: 0,
            files_touched: 0,
            crates_touched: 0,
        }
    }

    /// A minimal in-project edit: 1 file, 1 crate, 1 node.
    pub const fn minimal_edit() -> Self {
        Self {
            nodes_touched: 1,
            files_touched: 1,
            crates_touched: 1,
        }
    }
}

/// One piece of evidence that contributed to the verdict. Used for
/// auditability — the skeptical-auditor persona can replay an action
/// and see exactly which rules fired.
///
/// Tagged as `"evidence"` rather than `"kind"` because one variant
/// carries its own `kind` field (the action kind name) and the tag
/// would collide with the field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "evidence", rename_all = "snake_case")]
pub enum Evidence {
    /// The action was classified as a particular kind based on tool
    /// name pattern.
    ActionKindClassified { kind: String, from_tool: String },
    /// The blast radius was below the configured in-project threshold.
    InProjectBlastRadiusOk { nodes: usize, threshold: usize },
    /// The blast radius exceeded the configured threshold.
    BlastRadiusTooLarge { nodes: usize, threshold: usize },
    /// The action crosses crate boundaries — automatic Tier 3.
    CrossCrateBoundary { crates: usize },
    /// Out-of-project edit — automatic Tier 3.
    OutOfProjectEdit { target_hint: String },
    /// Shell command — automatic Tier 3.
    ShellCommand,
    /// Web fetch — automatic Tier 3.
    WebFetch,
    /// Unknown tool kind, defaulted to Tier 3 conservatively.
    UnknownToolDefaultedToTierThree { tool_name: String },
}

/// The full verdict attached to an action by [`crate::ApprovalMiddleware`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalVerdict {
    /// Final tier.
    pub tier: Tier,
    /// Computed blast radius.
    pub blast_radius: BlastRadius,
    /// Ordered list of evidence that contributed to the verdict. Read
    /// top-to-bottom for a human-readable audit trail.
    pub rationale: Vec<Evidence>,
    /// The action's `loop_key()` output (or `"non_tool_action"`). Used
    /// by downstream consumers to join verdicts with loop-detector
    /// state without re-parsing the action.
    pub keyed_by: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tier helpers ───────────────────────────────────────────────────

    #[test]
    fn tier_one_and_two_are_auto_approved() {
        assert!(Tier::One.is_auto_approved());
        assert!(Tier::Two.is_auto_approved());
        assert!(!Tier::Three.is_auto_approved());
    }

    #[test]
    fn only_tier_three_requires_approval() {
        assert!(!Tier::One.requires_approval());
        assert!(!Tier::Two.requires_approval());
        assert!(Tier::Three.requires_approval());
    }

    #[test]
    fn tier_names_are_stable() {
        assert_eq!(Tier::One.name(), "tier_one");
        assert_eq!(Tier::Two.name(), "tier_two");
        assert_eq!(Tier::Three.name(), "tier_three");
    }

    // ── BlastRadius constructors ───────────────────────────────────────

    #[test]
    fn blast_radius_zero_is_all_zero() {
        let b = BlastRadius::zero();
        assert_eq!(b.nodes_touched, 0);
        assert_eq!(b.files_touched, 0);
        assert_eq!(b.crates_touched, 0);
    }

    #[test]
    fn blast_radius_minimal_edit_is_one_each() {
        let b = BlastRadius::minimal_edit();
        assert_eq!(b.nodes_touched, 1);
        assert_eq!(b.files_touched, 1);
        assert_eq!(b.crates_touched, 1);
    }

    // ── Serde round-trips ──────────────────────────────────────────────

    #[test]
    fn tier_serde_snake_case() {
        // `#[serde(rename_all = "snake_case")]` converts the variant
        // identifier (One -> "one"), it doesn't prepend "tier_". Use
        // Tier::name() for the `tier_one` form.
        assert_eq!(serde_json::to_string(&Tier::One).unwrap(), r#""one""#);
        assert_eq!(serde_json::to_string(&Tier::Three).unwrap(), r#""three""#);
        let back: Tier = serde_json::from_str(r#""two""#).unwrap();
        assert_eq!(back, Tier::Two);
    }

    #[test]
    fn evidence_serde_round_trip() {
        let ev = Evidence::CrossCrateBoundary { crates: 3 };
        let json = serde_json::to_string(&ev).unwrap();
        // The Evidence enum uses `#[serde(tag = "evidence")]` rather
        // than `"kind"` because ActionKindClassified has its own
        // `kind` field that would collide with the discriminator.
        assert!(json.contains(r#""evidence":"cross_crate_boundary""#));
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn approval_verdict_full_round_trip() {
        let v = ApprovalVerdict {
            tier: Tier::Two,
            blast_radius: BlastRadius::minimal_edit(),
            rationale: vec![
                Evidence::ActionKindClassified {
                    kind: "edit_in_project".into(),
                    from_tool: "ix_code_analyze".into(),
                },
                Evidence::InProjectBlastRadiusOk {
                    nodes: 1,
                    threshold: 100,
                },
            ],
            keyed_by: "ix_code_analyze".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: ApprovalVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }
}
