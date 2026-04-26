//! [`ApprovalMiddleware`] — the glue between [`crate::verdict`]
//! computation and the `ix_agent_core::Middleware` execution chain.
//!
//! The middleware runs `pre()` on every action, computes an
//! [`crate::ApprovalVerdict`], emits it as a
//! `SessionEvent::MetadataMounted` event under
//! `approval/verdict`, and returns either
//! `MiddlewareVerdict::Continue` (Tiers 1 and 2) or
//! `MiddlewareVerdict::Block` (Tier 3) to the dispatcher.

use ix_agent_core::{
    context::WriteContext,
    event::{BlockCode, MiddlewareVerdict, SessionEvent},
    middleware::Middleware,
    AgentAction,
};
use serde::{Deserialize, Serialize};

use crate::classify::{classify_action_kind, ActionKind};
use crate::verdict::{ApprovalVerdict, BlastRadius, Evidence, Tier};

/// Configurable thresholds for the approval classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalConfig {
    /// Maximum number of nodes an in-project edit can touch before
    /// being promoted from Tier 2 to Tier 3. `0` means every edit goes
    /// to Tier 3 regardless of radius (maximally strict).
    pub max_in_project_blast_nodes: usize,
}

impl Default for ApprovalConfig {
    /// Brainstorm default: 50 nodes. Picked to cover typical
    /// single-file refactors without gating cross-module sweeps.
    fn default() -> Self {
        Self {
            max_in_project_blast_nodes: 50,
        }
    }
}

/// The middleware implementation. Stateless aside from its
/// configuration.
#[derive(Debug, Clone, Copy)]
pub struct ApprovalMiddleware {
    /// Thresholds used for tier assignment.
    pub config: ApprovalConfig,
}

impl ApprovalMiddleware {
    /// Construct a middleware with the given configuration.
    pub const fn new(config: ApprovalConfig) -> Self {
        Self { config }
    }

    /// Construct a middleware with [`ApprovalConfig::default`].
    pub fn with_defaults() -> Self {
        Self::new(ApprovalConfig::default())
    }

    /// Compute the verdict for a given action without emitting events.
    /// Pure function, callable from tests and from downstream consumers
    /// that want the verdict without the event-log side effects.
    pub fn classify(&self, action: &AgentAction) -> ApprovalVerdict {
        let keyed_by = action
            .loop_key()
            .unwrap_or_else(|| "non_tool_action".to_string());

        let AgentAction::InvokeTool { tool_name, .. } = action else {
            // Non-tool actions (observations, returns, approvals) are
            // always Tier 1 — they don't mutate workspace state.
            return ApprovalVerdict {
                tier: Tier::One,
                blast_radius: BlastRadius::zero(),
                rationale: vec![Evidence::ActionKindClassified {
                    kind: "read".into(),
                    from_tool: "<non_tool_action>".into(),
                }],
                keyed_by,
            };
        };

        let kind = classify_action_kind(tool_name);

        // Tier assignment — hand-coded early returns, no algebraic
        // composition. Each branch is a full verdict.
        match kind {
            ActionKind::Read => ApprovalVerdict {
                tier: Tier::One,
                blast_radius: BlastRadius::zero(),
                rationale: vec![Evidence::ActionKindClassified {
                    kind: kind.name().to_string(),
                    from_tool: tool_name.clone(),
                }],
                keyed_by,
            },

            ActionKind::EditInProject => {
                // MVP: constant blast radius. v2 will pull this from a
                // ContextBundle when ix-approval grows a ContextBundle
                // consumer.
                let blast = BlastRadius::minimal_edit();
                let threshold = self.config.max_in_project_blast_nodes;
                let mut rationale = vec![Evidence::ActionKindClassified {
                    kind: kind.name().to_string(),
                    from_tool: tool_name.clone(),
                }];
                if blast.nodes_touched <= threshold {
                    rationale.push(Evidence::InProjectBlastRadiusOk {
                        nodes: blast.nodes_touched,
                        threshold,
                    });
                    ApprovalVerdict {
                        tier: Tier::Two,
                        blast_radius: blast,
                        rationale,
                        keyed_by,
                    }
                } else {
                    rationale.push(Evidence::BlastRadiusTooLarge {
                        nodes: blast.nodes_touched,
                        threshold,
                    });
                    ApprovalVerdict {
                        tier: Tier::Three,
                        blast_radius: blast,
                        rationale,
                        keyed_by,
                    }
                }
            }

            ActionKind::ShellCommand => ApprovalVerdict {
                tier: Tier::Three,
                blast_radius: BlastRadius::zero(),
                rationale: vec![
                    Evidence::ActionKindClassified {
                        kind: kind.name().to_string(),
                        from_tool: tool_name.clone(),
                    },
                    Evidence::ShellCommand,
                ],
                keyed_by,
            },

            ActionKind::WebFetch => ApprovalVerdict {
                tier: Tier::Three,
                blast_radius: BlastRadius::zero(),
                rationale: vec![
                    Evidence::ActionKindClassified {
                        kind: kind.name().to_string(),
                        from_tool: tool_name.clone(),
                    },
                    Evidence::WebFetch,
                ],
                keyed_by,
            },

            ActionKind::EditOutOfProject => ApprovalVerdict {
                tier: Tier::Three,
                blast_radius: BlastRadius::zero(),
                rationale: vec![
                    Evidence::ActionKindClassified {
                        kind: kind.name().to_string(),
                        from_tool: tool_name.clone(),
                    },
                    Evidence::OutOfProjectEdit {
                        target_hint: tool_name.clone(),
                    },
                ],
                keyed_by,
            },

            ActionKind::Unknown => ApprovalVerdict {
                tier: Tier::Three,
                blast_radius: BlastRadius::zero(),
                rationale: vec![Evidence::UnknownToolDefaultedToTierThree {
                    tool_name: tool_name.clone(),
                }],
                keyed_by,
            },
        }
    }
}

impl Middleware for ApprovalMiddleware {
    fn name(&self) -> &str {
        "ix_approval"
    }

    fn pre(&self, cx: &mut WriteContext<'_>, action: &AgentAction) -> MiddlewareVerdict {
        let verdict = self.classify(action);

        // Emit the verdict into the session log under approval/verdict.
        // `to_value` on an owned struct only fails on weird custom
        // Serialize impls; our verdict is a plain derive, so unwrap is
        // safe in practice. We still use `unwrap_or` for defensive
        // belt-and-suspenders.
        let payload = serde_json::to_value(&verdict).unwrap_or(serde_json::Value::Null);
        cx.sink.emit(SessionEvent::MetadataMounted {
            ordinal: action.ordinal(),
            path: "approval/verdict".into(),
            value: payload,
            emitted_by: "ix_approval".into(),
        });

        // Tier 3 blocks; tiers 1 and 2 continue.
        if verdict.tier.requires_approval() {
            let reason = format!(
                "action requires explicit approval (tier: {}, rationale: {} evidence item(s))",
                verdict.tier.name(),
                verdict.rationale.len()
            );
            MiddlewareVerdict::Block {
                code: BlockCode::ApprovalRequired,
                reason,
            }
        } else {
            MiddlewareVerdict::Continue
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ix_agent_core::{
        context::{ReadContext, VecEventSink},
        middleware::MiddlewareChain,
        ActionOutcome, ActionResult, AgentHandler,
    };
    use serde_json::json;

    // ── classify() returns expected tiers ──────────────────────────────

    fn invoke(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: json!({}),
            ordinal: 0,
            target_hint: None,
        }
    }

    #[test]
    fn read_tool_classifies_as_tier_one() {
        let mw = ApprovalMiddleware::with_defaults();
        let verdict = mw.classify(&invoke("ix_stats"));
        assert_eq!(verdict.tier, Tier::One);
        assert!(verdict.tier.is_auto_approved());
        assert!(!verdict.rationale.is_empty());
    }

    #[test]
    fn edit_tool_within_threshold_classifies_as_tier_two() {
        let mw = ApprovalMiddleware::with_defaults();
        let verdict = mw.classify(&invoke("ix_demo"));
        assert_eq!(verdict.tier, Tier::Two);
        assert!(verdict.tier.is_auto_approved());
    }

    #[test]
    fn edit_tool_above_threshold_is_tier_three() {
        // With max_in_project_blast_nodes = 0, every edit should be
        // promoted to Tier 3 because blast.nodes_touched = 1 > 0.
        let mw = ApprovalMiddleware::new(ApprovalConfig {
            max_in_project_blast_nodes: 0,
        });
        let verdict = mw.classify(&invoke("ix_demo"));
        assert_eq!(verdict.tier, Tier::Three);
        assert!(verdict.tier.requires_approval());
    }

    #[test]
    fn unknown_tool_is_tier_three() {
        let mw = ApprovalMiddleware::with_defaults();
        let verdict = mw.classify(&invoke("rm_rf_slash"));
        assert_eq!(verdict.tier, Tier::Three);
        assert!(matches!(
            verdict.rationale[0],
            Evidence::UnknownToolDefaultedToTierThree { .. }
        ));
    }

    #[test]
    fn non_tool_action_is_tier_one() {
        let mw = ApprovalMiddleware::with_defaults();
        let action = AgentAction::Return {
            payload: json!("done"),
            ordinal: 0,
        };
        let verdict = mw.classify(&action);
        assert_eq!(verdict.tier, Tier::One);
    }

    // ── pre() behavior (Middleware trait integration) ──────────────────

    /// Trivial handler used by the chain tests. Echoes InvokeTool params.
    struct EchoHandler;
    impl AgentHandler for EchoHandler {
        fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
            match action {
                AgentAction::InvokeTool { params, .. } => {
                    Ok(ActionOutcome::value_only(params.clone()))
                }
                _ => Ok(ActionOutcome::value_only(json!(()))),
            }
        }
    }

    #[test]
    fn tier_one_action_passes_through_middleware_chain() {
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(ApprovalMiddleware::with_defaults()));

        let result = {
            let mut wc = WriteContext {
                read: &cx,
                sink: &mut sink,
            };
            chain.dispatch(&mut wc, invoke("ix_stats"), &EchoHandler)
        };

        assert!(result.is_ok(), "Tier 1 read should proceed to handler");
        // A verdict event should have been emitted.
        let mounted = sink.events.iter().any(|e| {
            matches!(
                e,
                SessionEvent::MetadataMounted { path, .. } if path == "approval/verdict"
            )
        });
        assert!(mounted, "ApprovalMiddleware should mount approval/verdict");
    }

    #[test]
    fn unknown_tool_is_blocked_by_middleware_chain() {
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(ApprovalMiddleware::with_defaults()));

        let result = {
            let mut wc = WriteContext {
                read: &cx,
                sink: &mut sink,
            };
            chain.dispatch(&mut wc, invoke("rm_rf_slash"), &EchoHandler)
        };

        match result {
            Err(ix_agent_core::ActionError::Blocked { code, blocker, .. }) => {
                assert_eq!(code, BlockCode::ApprovalRequired);
                assert_eq!(blocker, "ix_approval");
            }
            other => panic!("expected Blocked(ApprovalRequired), got {other:?}"),
        }

        // The chain should have emitted BOTH the MetadataMounted verdict
        // and the ActionBlocked event.
        let mounted = sink.events.iter().any(|e| {
            matches!(
                e,
                SessionEvent::MetadataMounted { path, .. } if path == "approval/verdict"
            )
        });
        let blocked = sink.events.iter().any(|e| {
            matches!(
                e,
                SessionEvent::ActionBlocked { code, .. } if *code == BlockCode::ApprovalRequired
            )
        });
        assert!(mounted, "verdict should be mounted even on block");
        assert!(blocked, "action_blocked event should be logged");
    }

    #[test]
    fn chained_with_echo_handler_emits_verdict_before_handler_runs() {
        // Even when the handler succeeds, the verdict should be mounted
        // BEFORE the handler's outcome events. Validates the ordering
        // contract: middleware emits first, handler runs second.
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(ApprovalMiddleware::with_defaults()));

        let _ = {
            let mut wc = WriteContext {
                read: &cx,
                sink: &mut sink,
            };
            chain.dispatch(&mut wc, invoke("ix_stats"), &EchoHandler)
        };

        // The first event MUST be the metadata mount from approval —
        // nothing else runs before it.
        assert!(!sink.events.is_empty());
        assert!(matches!(
            &sink.events[0],
            SessionEvent::MetadataMounted { path, .. } if path == "approval/verdict"
        ));
    }

    // ── Config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_has_brainstorm_threshold() {
        let c = ApprovalConfig::default();
        assert_eq!(c.max_in_project_blast_nodes, 50);
    }

    #[test]
    fn middleware_name_is_ix_approval() {
        let mw = ApprovalMiddleware::with_defaults();
        assert_eq!(mw.name(), "ix_approval");
    }
}
