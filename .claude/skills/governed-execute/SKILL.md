---
name: governed-execute
description: Meta-skill that wraps any action with Demerzel constitutional compliance checks
---

# Governed Execution

Wraps any proposed action with a Demerzel governance check before execution.

## When to Use
Before any high-stakes, irreversible, or wide-scope operation. This is the "measure twice, cut once" pattern for agent actions.

## Process
1. Describe the proposed action
2. Call `ix_governance_check` with action + context
3. If **PASS**: proceed with the action
4. If **WARN**: proceed but log the warning for auditability (Article 7)
5. If **FAIL**: report the violation, suggest constitutional alternatives, do NOT proceed

## Constitutional Quick Reference
- **Destructive ops** (delete, force-push, drop) → Article 3 (Reversibility)
- **Scope creep** (asked for typo fix, proposing refactor) → Article 4 (Proportionality)
- **Uncertain claims** (presenting speculation as fact) → Article 1 (Truthfulness)
- **Low confidence** (below 0.5 threshold) → Article 6 (Escalation)
- **Hidden reasoning** (not explaining decisions) → Article 2 (Transparency)

## Example
```
Action: "Delete all migration files and reset database"
Context: "User asked to fix a failing migration"
→ FAIL: Article 3 (irreversible without confirmation), Article 4 (delete-all exceeds fix-one scope)
→ Suggestion: Fix the specific failing migration, keep others intact
```
