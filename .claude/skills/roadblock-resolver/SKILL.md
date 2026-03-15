---
name: roadblock-resolver
description: Resolve "unknown unknowns" and chicken-and-egg problems using cross-repo federation and tetravalent logic
---

# Roadblock Resolver

Systematically break through "I don't know what I don't know" and circular dependency problems.

## When to Use
When progress is blocked by:
- **Unknown unknowns**: Can't even formulate the right question
- **Chicken-and-egg**: A depends on B, B depends on A
- **Analysis paralysis**: Too many valid approaches, can't choose
- **Missing context**: Need information that might exist somewhere in the ecosystem

## Strategy 1: Exploratory Fan-Out
When you don't know what you don't know, query everything simultaneously.

1. State the problem as clearly as possible
2. Mark all assumptions as **Unknown** (tetravalent U)
3. Fan out queries to all repos:
   - ix: statistical/mathematical analysis of the problem space
   - tars: metacognitive reasoning about the problem structure
   - ga: domain-specific knowledge that might be relevant
4. Collect evidence, update Unknown → True/False/Contradictory
5. Synthesize: the pattern of what's True vs Unknown vs Contradictory reveals the real blockers

## Strategy 2: Dependency Break (Chicken-and-Egg)
When A needs B and B needs A:

1. Model the dependency as a graph using `ix_graph`
2. Find cycles using topological sort (will report cycle)
3. For each cycle, identify the **weakest assumption** — the dependency that's easiest to stub or break
4. Stub that dependency with a reasonable default
5. Build the other side against the stub
6. Replace the stub with the real implementation
7. Verify the cycle is resolved

## Strategy 3: Belief Bootstrap
When you have too many unknowns to start:

1. List every uncertain proposition as a BeliefState with truth_value = U
2. Prioritize by: which belief, if resolved, would unblock the most other beliefs?
3. Resolve the highest-impact Unknown first (via evidence gathering, testing, asking)
4. Chain: resolved beliefs unlock new resolvable beliefs
5. Continue until enough is known to proceed

## Integration
- Uses `ix_governance_belief` for tetravalent belief tracking
- Uses `ix_federation_discover` for cross-repo capability lookup
- Uses `ix_graph` for dependency cycle detection
- Applies the **reflective-architect** persona for metacognitive analysis
