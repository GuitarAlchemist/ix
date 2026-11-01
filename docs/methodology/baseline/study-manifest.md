# Three strategy study manifest template

Status: NOT AUTHORIZED TO RUN. Fill and freeze before any model trial.
Use literal `NA` for unknown values, with a reason. Do not invent measurements.

## Identity and decision rule

- Study ID and version: TBD
- Manifest digest and frozen UTC date: TBD
- Operator, independent oracle owner and integration owner: TBD
- Question and task population: TBD
- Primary resource and meaningful improvement threshold: TBD
- Non-negotiable quality constraints and tolerated resource tradeoffs: TBD
- Continue / revise / stop rule: TBD

## Case inventory and independent acceptance

For each case: ID; family; selection rationale; pilot/holdout; task text digest;
base commit; initial worktree/fixture digest; allowed paths/actions; independent
oracle version/digest and command/rubric; constraints/regression checks; artifact
evidence destination; `parent_required` (0/1); exact blocked action and required
consumption/action/result proof; reviewer blinding. Do not put hidden solutions
in the author-visible manifest. Record access-controlled oracle references.

Pilot cases: TBD. Three distinct untouched real transfer cases: TBD.

## Allocation and isolation

List each planned trial ID, matched block ID, case ID, A/B/C, repeat, execution
order and randomization seed if used. State fresh-state restoration, conversation
and memory isolation, solution-leakage controls and human learning/contamination
handling. Explain matched variants if exact replay cannot be independent.

Allocation: TBD. Number of planned trials: TBD. No trials have been run.

## Models and environment

For every role: provider; exact version or exposed alias; effort; sampling
settings; CLI/tool versions; workflow/config digest; runtime date; OS/hardware;
dependency lock; permissions; network/tool access; operator; cache/context status.
Record unknown version pins explicitly, never as reproducibly pinned.

Controls across A/B/C and unavoidable confounders: TBD.

## Budgets and stop mechanism

For each cap: resource/unit; numeric limit; per-trial or root-tree scope; hard or
advisory; tested enforcement evidence; stop owner; descendant cleanup method.
Include deadline, calls/tokens if observable, retries, total agents, peak
concurrency and verification reserve. Include human availability and what waiting
counts toward the deadline. Declare treatment of late results and orphan usage.

Bounded live-run authorization and data/cost scope: NOT GRANTED.

## Construction and study setup ledger

Declare setup IDs for shared study setup and each strategy's reusable construction.
For each ID record observed wall seconds, human attention person-seconds and method,
intervention/handoff counts, usage.csv references, failure/retry work and evidence.
Shared study setup is disclosed separately, not silently charged to one strategy.
Record actual frozen workflow/prompt version and n observed reuse trials; show
setup and reuse separately. Any amortization must disclose the allocation rule.

## Timing and observation dictionary

- Wall timer and resolution: TBD
- Non-overlapping wall slice labeling/precedence rule: TBD
- Active attention interval method and resolution: TBD
- Intervention episode rule and clipboard transfer rule: protocol defaults / TBD
- Usage source and disjoint-record/deduplication rule: TBD
- Raw provider input/cache/output mapping and reasoning-token treatment: TBD
- Missing telemetry handling and capture coverage: TBD
- Optional billing evidence or estimated-price source/date: NA until evidenced
- Evidence storage, access, retention and deletion date: TBD
- Privacy: no audio/transcripts, clipboard contents, secrets or private prompts

## Result record and deviations

Link all initiated trials, including failure/timeout/cancellation/missing evidence.
List planned but uninitiated trials with reasons. Record any protocol deviation,
contamination, model/tool change or infrastructure fault before analysis. Preserve
original rows; do not silently exclude or replace bad runs. Pending rows are not
terminal observations. Confirm all children stopped or disclose residual execution.

## Review and launch checklist

Copy the checklist from [the protocol](../three-strategy-baseline.md#launch-and-decision-checklist).
Every box remains unchecked until supported. Documentation review is not live-run
authorization; no runtime integration or model execution is included in this template.
