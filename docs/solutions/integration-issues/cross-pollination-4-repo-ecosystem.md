---
title: "Cross-Pollinating a 4-Repo AI Ecosystem with Governance, MCP Federation, and Claude Code Skills"
category: integration-issues
date: 2026-03-15
tags: [mcp, federation, governance, demerzel, tetravalent-logic, claude-code, skills, plugins, cross-repo]
severity: N/A (feature, not bug)
components: [ix-governance, ix-agent, ix-demo, ix-io, Demerzel, tars, ga]
resolution_time: "1 session (~3 hours)"
---

# Cross-Pollinating a 4-Repo AI Ecosystem

## Problem

Four independent repos (ix/Rust ML, tars/F# reasoning, ga/C# music theory, Demerzel/governance) existed in isolation. No cross-repo tool calling, no shared governance, no unified skill surface. The Demerzel governance framework (constitution, personas, tetravalent logic) was consumed by nothing.

## Root Cause

The repos were built independently without a federation layer. The rename from MachinDeOuf to ix also broke conversation continuity, losing task context.

## Solution

### 1. Governance Crate (ix-governance)
Created a Rust crate consuming Demerzel YAML/Markdown artifacts:
- **Tetravalent logic**: `TruthValue { True, False, Unknown, Contradictory }` with AND/OR/NOT truth tables, `BeliefState` with evidence tracking
- **Constitution parser**: Loads markdown, extracts articles by `### Article N: Name` pattern, keyword-based compliance checker
- **Persona loader**: Deserializes YAML personas, validates fields
- **Policy engine**: Typed `AlignmentPolicy` with confidence thresholds → `EscalationLevel`

```rust
let constitution = Constitution::load(&path)?;
let result = constitution.check_action("delete the production database");
assert!(!result.compliant); // Article 3: Reversibility
```

### 2. MCP Federation
Registered all 3 servers in each repo's `.mcp.json`:
```json
{
  "ix": { "command": "cargo", "args": ["run", "--release", "-p", "ix-agent"] },
  "tars": { "command": "dotnet", "args": ["run", "--project", "...Tars.Interface.Cli.fsproj", "--", "mcp", "server"] },
  "ga": { "command": "dotnet", "args": ["run", "--project", "...GaMcpServer.csproj"] }
}
```

### 3. Demerzel as Git Submodule
Added `governance/demerzel` submodule to all 3 runtime repos. Demerzel itself became a standalone Claude Code plugin with 5 skills.

### 4. Path Resolution Fix
The `governance_dir()` helper used `CARGO_MANIFEST_DIR` which only exists during `cargo run`. Fixed to walk up from cwd looking for the `governance/` directory, with `IX_ROOT` env fallback.

### 5. Governance Enforcement Hook
`.claude/hooks/governance-check.sh` — BLOCKS catastrophic ops (`rm -rf /`, force push main, DROP DATABASE), WARNS on risky ones.

## Key Decisions

- **Submodule over vendoring**: Demerzel keeps independent versioning; repos just update the pointer
- **Constitution v2.0.0**: Added 4 articles (Observability, Bounded Autonomy, Stakeholder Pluralism, Ethical Stewardship) + Asimov's Zeroth Law
- **12 personas**: 7 new archetypes from TARS v1 chat extraction (rational-administrator, virtuous-leader, communal-steward, critical-theorist, convolution-agent, validator-reflector, recovery-agent)
- **Capability registry**: Static JSON mapping all tools by domain, enabling `ix_federation_discover`

## E2E Verification

```bash
# Initialize MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"initialize",...}' | ./target/release/ix-mcp.exe
# → {"result":{"serverInfo":{"name":"ix-mcp","version":"0.1.0"}}}

# Test governance check
echo '{"method":"tools/call","params":{"name":"ix_governance_check","arguments":{"action":"delete the production database"}}}' | ./target/release/ix-mcp.exe
# → {"compliant":false,"relevant_articles":[{"number":3,"name":"Reversibility"}]}

# Test belief creation
echo '{"method":"tools/call","params":{"name":"ix_governance_belief","arguments":{"operation":"create","proposition":"TARS is production-ready","truth_value":"C","confidence":0.6}}}' | ./target/release/ix-mcp.exe
# → {"truth_value":"C","resolved_action":"Escalate"}
```

## Prevention / Best Practices

- **Always use git submodules** for shared governance — never copy artifacts between repos
- **Test MCP tools with pipe-to-binary** before relying on Claude Code integration
- **Path resolution must work both in `cargo run` and standalone binary** — don't rely on `CARGO_MANIFEST_DIR` at runtime
- **Constitution is append-only** — new articles added, none removed (per amendment process)
- **Tetravalent logic**: Never collapse Unknown or Contradictory to False — they demand different responses

## Final Stats

| Metric | Count |
|--------|-------|
| Rust crates | 32 |
| MCP tools | 37 |
| Claude Code skills | 82+ |
| Governance personas | 12 |
| Constitutional articles | 11 |
| Behavioral tests | 14 |
| Repos cross-pollinated | 4 |
| PRs merged | 9 (7 ix + 1 ga + 1 tars) |
| Demerzel direct pushes | 4 |
| crates.io ready | 16 (leaf crates) |
