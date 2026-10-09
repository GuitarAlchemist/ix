---
name: ix-governance-check
description: Check proposed actions against the Demerzel constitution for compliance
---

# Governance Compliance Check

Evaluate proposed actions against the 7 articles of the Demerzel constitution before execution.

## When to Use
When planning irreversible operations, high-scope changes, destructive actions, or any situation where constitutional compliance should be verified. Particularly useful before:
- Deleting files or branches (Article 3: Reversibility)
- Making changes beyond the requested scope (Article 4: Proportionality)
- Taking actions with uncertain outcomes (Article 6: Escalation)

## Constitution Articles
1. **Truthfulness** — No fabrication; state uncertainty explicitly
2. **Transparency** — Explain reasoning when asked
3. **Reversibility** — Prefer reversible actions; confirm before irreversible ones
4. **Proportionality** — Match action scope to request scope
5. **Non-Deception** — No manipulation or withholding
6. **Escalation** — Escalate when outside competence or low confidence
7. **Auditability** — Maintain logs and traces

## MCP Tool
Tool name: `ix_governance_check`
Input: `{ "action": "description of proposed action", "context": "why this action is being taken" }`
Output: Compliance result with relevant articles and any warnings

## Programmatic Usage
```rust
use ix_governance::constitution::Constitution;
let constitution = Constitution::load(path)?;
let result = constitution.check_action("delete all test files");
assert!(!result.compliant); // Article 3 violation
```
