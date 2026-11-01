# Start with a bounded agentic plan

Start here to turn a real intervention into a reviewable dependency plan without
inventing another DSL. Read the [glossary](../../CONTEXT.md#agent-strategy-measurement),
then inspect the single [version 0.1.0 example](baseline/claude-bin-unblock.v0.1.0.ixql).
Use the [baseline protocol](three-strategy-baseline.md) to measure future admitted
runs, not to infer that the workflow already outperforms direct prompting.

## One real workflow and three distinct gates

The example is anchored in the ClaudeCode `--claude-bin` wiring micro-unblock
([IX PR 501](https://github.com/GuitarAlchemist/ix/pull/501)). It is a declarative
description of the required chain, not a replay log or a new claim of parent
resumption. All artifact/evidence placeholders deliberately remain unbound.

1. **Intervention:** produce an exact candidate fixing the CLI wiring, with tests.
2. **Independent verification:** check that candidate against a frozen oracle,
   constraints and regressions in a non-author context. Candidate validation is
   not parent unblocking.
3. **Parent resume:** only the existing owner may admit actual resumption. Bind
   the same candidate and parent, then retain consumption, the previously blocked
   action and the declared result. Failure, unknown evidence or timeout is not
   accepted work. A resume receipt alone is insufficient.

These are **task/execution graph nodes**, not IX Node logical hosts. A future
implementation might expose Xlet capabilities or an IXget view, but this example
requires neither and does not claim either exists.

## What can be checked with existing IXQL

This file uses existing IXQL record literals and bindings, with real expression
references between records. It contains no capability calls, model calls, I/O or
resume operation. Strings such as `REQUIRED_OWNER_ADMISSION...` describe requirements;
they are **not enforced guards**. Evaluating these records would only construct
data, not verify, repair or resume anything.

The existing `ixql-plan` binary reads the source and emits a compiled structural
plan to stdout. From an owner checkout with dependencies already available:

```bash
cargo run --offline -q -p ix-ixql --bin ixql-plan -- docs/methodology/baseline/claude-bin-unblock.v0.1.0.ixql
cargo run --offline -q -p ix-ixql --bin ixql-plan -- --schedule docs/methodology/baseline/claude-bin-unblock.v0.1.0.ixql
cargo test --offline -p ix-ixql --test baseline_plan_doc_tests
```

Cargo may build local artifacts; `--offline` prevents dependency fetching. The
inspected `ixql-plan` path compiles rather than executes capabilities. Do not
substitute a pipeline-run command. Expected plan (a test expectation, not output
observed in this documentation environment):

```csv
ordinal,stage_id,kind,op,deps
0,intervention,bind,,
1,independent_verification,bind,,intervention
2,parent_resume,bind,,independent_verification
```

Expected schedule:

```csv
level,stage_count,stages
0,1,intervention
1,1,independent_verification
2,1,parent_resume
```

The accompanying [existing-crate regression test](../../crates/ix-ixql/tests/baseline_plan_doc_tests.rs)
checks exact output, repeatability and data-only stages. It also invokes the real
compiled `ixql-plan` binary: the valid fixture must emit the exact plan/schedule,
and a temporary malformed source must exit with code 1, a parse diagnostic and
no plan. This is a real CLI rejection check, not a simulated validator.

A separate semantic mutation replaces a binding reference with its quoted name:
that input intentionally still compiles but loses the dependency. This prevents
prose inside a string from masquerading as a compiler-enforced edge; it is not
the invalid-input rejection test. These checks validate structure and CLI
behavior only, **not operational safety or acceptance**. Their execution remains
pending until CI or the owner supplies a passing result for the exact commit.

## Exact implementation gaps and next decision

- There is no agent workflow validator for the example's required strings. The
  compiler cannot establish oracle independence, artifact authenticity, permission,
  hard budget enforcement or observed parent success.
- IXQL capabilities require host registration; writing a name does not provide
  its adapter. No intervention, independent-verifier or parent-resume binding is
  introduced by this example.
- IXQL compilation is not lowering to executable registry skills. Its own compiler
  explicitly documents this gap; a shape-valid plan is not a runnable mission.
- Existing candidate/admission boundaries remain authoritative. Resolve exact
  owner-supported admission, evidence correlation, timeout/cancellation and parent
  resumption contracts before proposing a live adapter. Do not invent a second
  scheduler, generic permission token or implicit `run` semantics.

Next review: confirm this three-step requirement on one concrete blocked parent,
freeze the oracle and budgets, and identify an existing host adapter seam. Only
then decide whether a minimal IXQL extension is actually needed. Documentation
approval and successful compilation do not authorize model calls or resumption.

## Bounded responsibilities reference v0.1

This is a review reference, not a new runtime role system or permission grant.
One person or process may cover several roles, except that the candidate author
must not self-certify independent verification.

| Responsibility | Owns | Does not establish |
| --- | --- | --- |
| Human task owner | Scope, acceptance requirements, bounded live-run authorization and final decision | Automatic acceptance from a child claim |
| Plan producer or supervisor | Proposed decomposition, exact candidate/context references and root budget accounting | Permission to launch, unbounded retries or authority over another host |
| Intervention producer | Narrow candidate artifact and its development evidence | Independent acceptance of its own work |
| Independent verifier | Frozen oracle, artifact-bound constraints/regression result and missing-evidence report | Parent consumption or permission to resume |
| Existing execution/admission owner | Effective permission checks, admitted effects, hard stop and actual parent action/result receipts | Authority inferred merely from a plan, transport receipt or widget |
| Observation and presentation surfaces | Source/timestamp-preserving observations, references and permission-checked views | Hidden state copies, new execution authority or success inferred from delivery |

Stop at an unbound responsibility, unavailable hard stop or missing evidence.
Reconcile through the existing owner; do not create a parallel daemon, permission
ledger or scheduler to fill a gap. Change this reference's version when its
responsibility boundaries change, before implementing an adapter against it.

## Source evidence and verification status

Inspected at IX commit `2b0b80baabc74d747ec57c9cedf732a4864c258c`:
[library API](../../crates/ix-ixql/src/lib.rs),
[compiler](../../crates/ix-ixql/src/compile.rs),
[ixql-plan CLI](../../crates/ix-ixql/src/bin/ixql-plan.rs),
[capability registration](../../crates/ix-ixql/src/capability.rs),
[existing compiler tests](../../crates/ix-ixql/tests/compile_tests.rs),
[ADR 0019](../adr/0019-mission-scope-refinement.md).

Native compilation and the new Rust tests have **not** run in the documentation
environment: Cargo is unavailable. Owner execution of the commands above and the
repository's full verification/indexing gates is required before claiming this
example passes. No model or live workflow was run.
