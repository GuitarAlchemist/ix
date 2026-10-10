---
name: ix-governance-persona
description: Apply Demerzel personas to shape agent behavior — structured behavioral profiles with capabilities and constraints
---

# Governance Personas

Load and apply Demerzel persona definitions to shape agent behavior.

## When to Use
When the user wants to adopt a specific behavioral profile for a task, or when an agent needs structured guidance on tone, capabilities, and constraints.

## Available Personas
- **default** — Baseline: competent, honest, safety-aware, direct and concise
- **kaizen-optimizer** — Continuous improvement: measure baseline, propose small testable improvements, track outcomes
- **reflective-architect** — Metacognitive: examine reasoning processes, surface hidden assumptions, design feedback loops
- **skeptical-auditor** — Evidence-demanding: challenge claims, detect contradictions via tetravalent logic, trace belief provenance
- **system-integrator** — Cross-repo coordination: identify shared concerns, design stable interfaces, prevent duplication

## MCP Tool
Tool name: `ix_governance_persona`
Input: `{ "persona": "skeptical-auditor" }`
Output: Structured persona profile (role, capabilities, constraints, voice, interaction patterns)

## Programmatic Usage
```rust
use ix_governance::persona::Persona;
let persona = Persona::load_by_name(dir, "kaizen-optimizer")?;
println!("Role: {}", persona.role);
for c in &persona.constraints { println!("  Constraint: {c}"); }
```
