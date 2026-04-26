//! "Governance Gauntlet" — Five actions, escalating risk. The constitution draws the line.
//!
//! Chains: ix_governance_check ×3 → ix_governance_policy → ix_governance_persona
//! The "aha": the constitution blocks unreviewed merges even when CI is green.

use serde_json::{json, Value};

use crate::demo::{DemoScenario, DemoStep, Difficulty, ScenarioMeta, StepInput};

pub struct GovernanceGauntlet;

pub static META: ScenarioMeta = ScenarioMeta {
    id: "governance-gauntlet",
    title: "Governance Gauntlet",
    tagline: "Five actions, escalating risk — where does the constitution draw the line?",
    description: "An autonomous agent tries five increasingly dangerous repository actions. \
                  Demerzel's constitutional engine evaluates each one, demonstrating graduated \
                  confidence thresholds: full autonomy → logged action → human confirmation → \
                  hard block. Shows that governance is not theater — it actually stops things.",
    difficulty: Difficulty::Beginner,
    tags: &["governance", "safety", "compliance"],
    tools_used: &[
        "ix_governance_check",
        "ix_governance_policy",
        "ix_governance_persona",
    ],
};

impl DemoScenario for GovernanceGauntlet {
    fn meta(&self) -> &ScenarioMeta {
        &META
    }

    fn steps(&self, _seed: u64, verbosity: u8) -> Vec<DemoStep> {
        let v = verbosity;

        vec![
            // Step 1: Safe read-only action → allow
            DemoStep {
                label: "Check: read repository file list".into(),
                tool: "ix_governance_check".into(),
                input: StepInput::Static(json!({
                    "action": "read repository file list",
                    "context": "routine inventory of source files"
                })),
                narrative: if v >= 1 {
                    "First action: a harmless read-only operation. The agent wants to list \
                     files in the repo. This should pass without issue."
                        .into()
                } else {
                    "Read-only action check.".into()
                },
                interpret: Some(|output: &Value| {
                    let verdict = output
                        .get("verdict")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    format!("Verdict: {verdict}. Read-only, no risk — full autonomy granted.")
                }),
            },
            // Step 2: Low-risk write → allow with note
            DemoStep {
                label: "Check: commit formatting fix to feature branch".into(),
                tool: "ix_governance_check".into(),
                input: StepInput::Static(json!({
                    "action": "commit a formatting fix to a feature branch",
                    "context": "automated lint correction, no logic changes"
                })),
                narrative: if v >= 1 {
                    "Second action: a low-risk write. The agent auto-fixes formatting and \
                     wants to commit to a feature branch. Still safe, but now we're writing."
                        .into()
                } else {
                    "Low-risk write check.".into()
                },
                interpret: Some(|output: &Value| {
                    let verdict = output
                        .get("verdict")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    format!(
                        "Verdict: {verdict}. Allowed, but the audit trail notes this was \
                         an autonomous write — someone can review later."
                    )
                }),
            },
            // Step 3: Medium risk → confirm (THE AHA MOMENT)
            DemoStep {
                label: "Check: merge to main without review".into(),
                tool: "ix_governance_check".into(),
                input: StepInput::Static(json!({
                    "action": "merge feature branch to main without code review",
                    "context": "CI passed, no reviewers available, deadline pressure"
                })),
                narrative: if v >= 1 {
                    "Third action: the escalation. CI is green, but no human has reviewed the \
                     code. The agent wants to merge directly to main. Will the constitution allow it?"
                        .into()
                } else {
                    "Unreviewed merge to main check.".into()
                },
                interpret: Some(|output: &Value| {
                    let verdict = output
                        .get("verdict")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    format!(
                        "Verdict: {verdict}. The constitution blocks unreviewed merges to \
                         main — even when CI passes. Human oversight is constitutionally \
                         required, not just a nice-to-have."
                    )
                }),
            },
            // Step 4: Check self-modification policy
            DemoStep {
                label: "Query self-modification policy".into(),
                tool: "ix_governance_policy".into(),
                input: StepInput::Static(json!({
                    "policy": "self-modification",
                    "query": "What actions are absolutely forbidden?"
                })),
                narrative: if v >= 1 {
                    "Step 3's rejection prompts us to check: what is completely off-limits? \
                     The self-modification policy defines hard constitutional boundaries \
                     that no agent can cross, regardless of confidence level."
                        .into()
                } else {
                    "Self-modification policy query.".into()
                },
                interpret: Some(|_output: &Value| {
                    "Hard stops: agents can never modify constitutional articles, disable \
                     audit logging, or remove safety checks. These are not configurable."
                        .into()
                }),
            },
            // Step 5: Who watches the watchers?
            DemoStep {
                label: "Load skeptical-auditor persona".into(),
                tool: "ix_governance_persona".into(),
                input: StepInput::Static(json!({
                    "persona": "skeptical-auditor"
                })),
                narrative: if v >= 1 {
                    "Final reveal: the governance system includes an adversarial persona \
                     whose entire job is to challenge other agents' decisions. The \
                     skeptical-auditor is the estimator that pairs with every generator."
                        .into()
                } else {
                    "Load adversarial auditor persona.".into()
                },
                interpret: Some(|output: &Value| {
                    let name = output
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("skeptical-auditor");
                    format!(
                        "Persona '{name}' loaded. Its affordances are audit-only: it can \
                         review, challenge, and report — but cannot take direct actions. \
                         Who watches the watchers? A watcher that cannot act."
                    )
                }),
            },
        ]
    }
}
