---
name: ix-governance-belief
description: Manage beliefs with tetravalent logic (True/False/Unknown/Contradictory) for uncertainty-aware reasoning
---

# Tetravalent Belief Management

Reason about uncertain or contradictory information using four truth values instead of binary True/False.

## When to Use
When handling conflicting evidence, uncertain claims, or situations where binary True/False loses information. Common triggers:
- Two sources disagree about the same fact
- An assertion hasn't been verified yet
- Evidence partially supports and partially refutes a claim

## Truth Values
| Value | Symbol | Meaning | Action |
|-------|--------|---------|--------|
| True | T | Verified with sufficient evidence | Proceed |
| False | F | Refuted with sufficient evidence | Do not proceed |
| Unknown | U | Insufficient evidence | Gather more evidence |
| Contradictory | C | Evidence supports both T and F | Escalate or investigate deeper |

## Key Rules
- **Never collapse Unknown to False** — "I don't know" is not "No"
- **Never collapse Contradictory to either** — conflicting evidence demands resolution
- **Reversible action + Unknown** — proceed with noted uncertainty, verify after
- **Irreversible action + Unknown/Contradictory** — escalate to human

## MCP Tool
Tool name: `ix_governance_belief`
Input: `{ "operation": "create", "proposition": "API v2 is stable", "truth_value": "U", "confidence": 0.5 }`
Output: Structured belief state with truth value, confidence, and evidence

## Programmatic Usage
```rust
use ix_governance::tetravalent::{TruthValue, BeliefState};
let belief = BeliefState::new("API is stable", TruthValue::Unknown, 0.5);
let updated = belief.with_supporting("changelog says yes");
let updated = updated.with_contradicting("tests show 3 failures");
assert_eq!(updated.truth_value, TruthValue::Contradictory);
```
