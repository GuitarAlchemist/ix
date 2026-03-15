---
name: delegate-cli
description: Delegate tasks to Codex CLI, Gemini CLI, or other AI CLIs present on the machine — governed by Demerzel constitution
---

# Multi-CLI Delegation

Detect and delegate work to other AI CLI tools on the machine (OpenAI Codex CLI, Google Gemini CLI) while maintaining Demerzel governance.

## When to Use
When a task would benefit from a different model's strengths, or when parallelizing work across multiple AI agents. Demerzel governs all delegated work.

## Detection
Check for available CLIs:
```bash
which codex 2>/dev/null && echo "codex: available"
which gemini 2>/dev/null && echo "gemini: available"
which claude 2>/dev/null && echo "claude: available"
```

## Delegation Strategy

### Strengths-Based Routing
| CLI | Best For | Governance |
|-----|----------|------------|
| **Claude Code** | Complex reasoning, multi-file refactors, architecture | Full Demerzel governance |
| **Codex CLI** | Code generation, quick edits, test writing | Constitution check before accepting output |
| **Gemini CLI** | Long-context analysis, multimodal tasks, research | Constitution check before accepting output |

### Parallel Execution
For independent subtasks, delegate to multiple CLIs simultaneously:
1. Break the task into independent units
2. Assign each to the best-suited CLI
3. Run in parallel via background processes
4. Collect results and verify against constitution (Article 1: no fabrication, Article 7: auditability)

## Governance Rules
All delegated work is subject to the Demerzel constitution:

1. **Before delegation**: Run `governed-execute` check on the proposed delegation
2. **Prompt construction**: Include constitutional constraints in the delegated prompt
3. **After delegation**: Verify output with `ix_governance_check` before accepting
4. **Audit trail**: Log what was delegated, to which CLI, and what was returned (Article 7)

## Delegation Template
```bash
# Delegate a code generation task to Codex CLI
codex --quiet "Generate unit tests for the following function.
Rules: No fabrication (test real behavior), no destructive operations,
match the scope of the request exactly.

$(cat src/my_module.rs)"
```

## Safety
- Never delegate constitutional modification or governance bypass
- Never delegate credential/secret handling to external CLIs
- Always verify delegated output before committing
- Log all delegations for auditability
- Apply the **skeptical-auditor** persona when reviewing delegated results
