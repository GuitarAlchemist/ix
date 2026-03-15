---
name: ecosystem-audit
description: Cross-repo audit using skeptical-auditor persona — verify capabilities, detect duplication, check governance
---

# Ecosystem Audit

Comprehensive audit across all GuitarAlchemist repos using the skeptical-auditor persona.

## When to Use
When you need to verify the health, consistency, or compliance of the multi-repo ecosystem.

## Audit Dimensions

### 1. Capability Duplication
Query each MCP server's tool list. Flag tools that overlap across repos.
- Use `ix_federation_discover` to scan the capability registry
- Apply system-integrator persona constraint: "No duplicating existing functionality"

### 2. Governance Compliance
Check each repo's behavior against the Demerzel constitution.
- Use `ix_governance_check` for constitutional compliance
- Use tetravalent logic to rate compliance: T (verified), F (violated), U (unchecked), C (conflicting evidence)

### 3. Integration Health
Verify cross-repo connections are functional.
- Can ix call tars tools? Can ix call ga tools?
- Are trace export/import paths working?
- Is the capability registry up to date?

## Output Format
```
## Ecosystem Audit Report

### Capability Matrix
| Domain | ix | tars | ga | Overlap? |
|--------|-----|------|-----|----------|

### Governance Compliance
| Article | ix | tars | ga | Status |
|---------|-----|------|-----|--------|

### Integration Status
| Connection | Status | Last Verified |
|------------|--------|---------------|
```
