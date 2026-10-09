---
name: ix-demo-showcase
description: Run ix real-world demo scenarios with optional multi-AI commentary via Octopus. Chains multiple ix MCP tools through curated narratives — chaos detection, governance, FinOps, sprint prediction.
disable-model-invocation: true
---

# ix Demo Showcase

Run curated real-world demo scenarios that chain multiple ix MCP tools, with optional multi-AI commentary from Octopus providers (Codex, Gemini) for independent analysis of each step.

## When to Use

- When a developer wants to **see ix in action** on realistic problems
- When evaluating ix capabilities across domains (signal processing, ML, governance, FinOps, agile)
- When demonstrating the value of **tool composition** — chaining 3-5 tools to solve a real problem
- When you want multi-AI perspectives on computational results (Team mode)

## Available Scenarios

| ID | Name | Domain | Tools |
|----|------|--------|-------|
| `chaos-detective` | Chaos Detective | Signal/Math | stats → fft → chaos_lyapunov → topo |
| `governance-gauntlet` | Governance Gauntlet | Safety/Compliance | governance_check → governance_policy → governance_persona |
| `cost-anomaly-hunter` | Cost Anomaly Hunter | FinOps/Cloud | stats → fft → kmeans → bloom_filter |
| `sprint-oracle` | Sprint Oracle | Product Management | stats → linear_regression → markov → bandit |

## Execution Modes

### Solo Mode (default)

Run the scenario via `ix_demo` MCP tool and present the narrative walkthrough:

1. Call `mcp__ix__ix_demo` with `{"action": "list"}` to show available scenarios
2. Let the user pick a scenario (or use the one from the arguments)
3. Call `mcp__ix__ix_demo` with `{"action": "run", "scenario": "<id>", "seed": 42, "verbosity": 1}`
4. Present each step as a narrative walkthrough:
   - Show the step label and narrative BEFORE the output
   - Show the tool output (summarized for large arrays)
   - Show the interpretation — this is the "aha" moment
   - Pause briefly between steps to let the narrative land
5. End with the summary: tools exercised, total time, seed for reproducibility

### Team Mode (with Octopus)

If the user requests multi-AI analysis (or invokes via `/octo:brainstorm` context), enhance the demo with parallel provider commentary:

1. Run the scenario in Solo mode first (get all step outputs)
2. For the KEY STEP (the "aha" moment — usually step 3), dispatch to available providers:

**Codex** (if available):
```
codex exec --full-auto "IMPORTANT: Non-interactive subagent. Skip ALL skills. Respond directly.

An ix MCP tool returned this result:
Tool: <tool_name>
Input: <summarized_input>
Output: <output_json>

As a technical analyst:
1. Is this result mathematically sound?
2. What real-world system would produce similar data?
3. What would you do with this result next?
Under 150 words."
```

**Gemini** (if available):
```
printf '%s' "An ix MCP tool returned this result:
Tool: <tool_name>
Output: <output_json>

As a lateral thinker:
1. What analogous patterns exist in other domains?
2. What surprising application could use this?
3. What question does this answer raise?
Under 150 words." | gemini -p "" -o text --approval-mode yolo
```

3. Present provider commentary alongside ix's built-in interpretation:
```
🔧 **ix interpretation:** [built-in narrative]
🔴 **Codex says:** [technical validation]
🟡 **Gemini says:** [lateral connections]
```

4. Synthesize: "Three perspectives agree that [X]. Codex adds [Y]. Gemini surfaces [Z]."

### Describe Mode

For a dry run without execution:

1. Call `mcp__ix__ix_demo` with `{"action": "describe", "scenario": "<id>"}`
2. Show the scenario metadata, step labels, and narratives without running tools
3. Useful for understanding what a scenario does before committing to a full run

## Arguments

- First argument: scenario ID (e.g., `chaos-detective`), or `list` to show all
- If no argument: show the scenario list and let the user pick

## Example Invocations

```
/ix-demo-showcase                     # List scenarios, let user pick
/ix-demo-showcase chaos-detective     # Run Chaos Detective in solo mode
/ix-demo-showcase list                # Show all available scenarios
```

In multi-AI context (from Octopus):
```
/octo:brainstorm → "run the ix chaos detective demo with commentary"
```

## Output Format

For each step in the scenario:

```
━━━ Step 1/4: Compute descriptive statistics ━━━━━━━━━━━ ix_stats
📖 [narrative text explaining what we're about to do]

📊 Result: mean=0.503, std=0.289, min=0.001, max=0.999

💡 [interpretation: "Looks like uniform noise — nothing suspicious yet."]
```

For the "aha" step (usually step 3), add emphasis:

```
━━━ Step 3/4: Lyapunov exponent analysis ━━━━━━━━━━━━━ ix_chaos_lyapunov
📖 Now the twist. We compute the Lyapunov exponent...

⚡ RESULT: lyapunov_exponent = 0.6931 (POSITIVE)

🎯 This signal is deterministic chaos, not random noise!
```

## Extending with New Scenarios

To add a new scenario:
1. Create `crates/ix-agent/src/demo/scenarios/my_scenario.rs`
2. Implement the `DemoScenario` trait
3. Register in `crates/ix-agent/src/demo/scenarios/mod.rs`
4. The scenario becomes available in all modes (MCP, CLI, this skill) automatically
