# Three strategy baseline protocol

Status: review draft, 2026-11-01. This document defines a baseline; it does not
authorize model calls, a benchmark campaign, deployment, or a new framework.
The existing IX owner integrates it after review. Terms belong in
[CONTEXT.md](../../CONTEXT.md); this is a read-on-demand measurement procedure.

## Question and scope

For comparable real tasks, what accepted work does each strategy deliver, at what
elapsed time, human attention and model usage? Separate **work quality** from
**resource efficiency**. No strategy wins in advance; no composite intelligence,
entropy or productivity score will hide a failed acceptance gate.

| Strategy | Operation | In-scope work |
| --- | --- | --- |
| A — reference control | Stef dictates directly to Claude and uses his habitual human loop | Dictation/preparation, normal corrections, manual transfers, execution, verification and recovery |
| B — prepared prompt | Gaia prepares the prompt, then Claude executes it | All of A plus Gaia's preparation, its model usage, prompt review and the handoff |
| C — supervised workflow | A supervised branch delegates to leaves and integrates their results | Supervisor, all leaves/verifiers, context preparation, retries, integration, handoffs and recovery; initial workflow construction reported separately |

The first dynamic ClaudeCode micro-unblock is historical motivation, not a matched
A/B/C observation. A meaningful independent candidate check is evidence of useful
work, but is not proof of parent resumption, transfer, or lower/higher cost than A
or B. Do not fill this baseline from an anecdote, inferred call duration, partial
subagent token totals, or lost logs. Preserve such evidence as explicitly partial.

## Freeze before any run

Complete a private study manifest with these named sections. Hash/version it and
the oracle before the first assignment; any subsequent change creates a new cohort.

1. **Question and decision rule.** Name the task population, quality constraints,
   primary resource of interest, practically meaningful improvement, tolerated
   tradeoffs, and who decides. Do not tune these after seeing a winner.
2. **Case inventory.** Record case IDs, families, selection rationale, task text
   digest, base commit, fixture/environment digest, allowed files/actions, parent
   blocked state if applicable, and whether pilot or untouched transfer holdout.
3. **Independent oracle.** A non-author owns fixed executable checks and a written
   rubric for unavoidable judgement. Include constraints, regression tests and the
   exact parent-unblocked check. Keep hidden checks/answers out of author contexts;
   disclose requirements equally. Pin oracle source, reviewer identity/role and
   evidence location. Blind the reviewer to strategy where feasible. Author
   self-report is never the oracle. A changed oracle invalidates paired comparison
   unless every retained artifact can be re-evaluated identically and this is disclosed.
4. **Allocation.** Freeze case × strategy × repeat assignments and order, including
   seed if randomized. Pair A/B/C on the same initial state. Restore isolated
   worktrees, fixtures, conversation/memory and tool access; no solution from an
   earlier strategy may reach a later one. Counterbalance ABC, BCA, CAB (and reversed
   orders where feasible). Order balancing alone does not remove learning: if the
   same human has seen a solution, use prevalidated matched variants or label the
   block contaminated and report it separately. Never silently drop a poor result.
5. **Provenance and controls.** Pin model/provider, model version or exact exposed
   alias, effort, sampling settings if available, CLI/tool/workflow versions,
   workflow digest, run date, OS/hardware, repository and dependency state, operator,
   permission mode, network/tools and context availability. Use the same execution
   model/effort across A/B/C where possible and the same verifier setup. Record every
   Gaia/supervisor/leaf model. If quality/capability differs, the result compares
   complete systems, not workflow alone; stratify rather than claiming causality.
   Mark cache condition cold/warm/mixed/unknown and reused context explicitly;
   neither a fresh chat nor a restored worktree proves a cold provider cache.
6. **Budgets and stopping.** Declare wall deadline, model-call/token limits where
   measurable, retry count, total agents and peak concurrency, human availability,
   verification reserve and stop owner. For each cap name its unit, scope and
   enforcement: hard (tested kill/refusal including descendants) or advisory
   (observed only). Count all children against the root cap. Unknown usage cannot
   enforce a token limit; use a tested wall/call/concurrency bound instead. No live
   admission without an enforceable stop. Record residual/orphan execution and its
   resources; timeout is not proof that work was cancelled. Preserve IX permissions,
   halt markers and supervised-loop preflight when applicable; no benchmark override.
7. **Construction accounting.** Identify shared study setup separately from each
   strategy's setup. Record C's initial workflow development/tuning and any reusable
   B prompt tooling, including human/model resources and failed development attempts.
   Freeze the reusable version before evaluation. A repair during a trial belongs
   to that trial; a generalized revision creates a new version/cohort. Never hide
   a trial-specific prompt in a supposedly free setup phase.
8. **Measurement and privacy.** Name timers, usage collectors, evidence retention,
   access and deletion date. Record timing intervals, counts and artifact references,
   not audio, voice transcripts, clipboard contents, secrets or personal conversations.
   Share only redacted aggregates unless separately authorized.

## Trial lifecycle and units

Start the clock when the assigned task is released for strategy-specific preparation,
before dictation or Gaia prompt work. Stop at independent acceptance (including
required parent resumption), terminal failure, cancellation or budget deadline.
Acceptance after the cap is a late outcome, not a success within budget. Report it
separately without rewriting the original trial. Retry clocks do not reset.

| Measure | Unit and recording rule |
| --- | --- |
| Accepted | `1` only with complete independent evidence; otherwise `0` for a terminal trial. `NA` only while pending. Record why: failed check, timeout, cancellation, missing evidence, etc. |
| Candidate validated | `1`, `0`, or `NA` when not evaluated; retain artifact/check reference |
| Parent unblocked | `1`, `0`, or `NA`; declare `parent_required` first. If required, `NA` blocks acceptance |
| Elapsed | Seconds from release to stop, monotonic timer preferred; `wall_s = stop - start`. For non-success this is observed time to stop, **not** time to acceptance |
| Time to acceptance | `wall_s` for accepted trials only; otherwise unavailable/right-censored at the stop time. Do not pretend failures accepted at the timeout |
| Preparation/execution/wait/verification | Non-overlapping wall-time slices using the manifest's fixed labeling rule; sum to `wall_s`. Concurrent child durations must not be added as wall time; preserve per-agent execution separately if available. Unknown slices are `NA` |
| Human attention | Sum observed active intervals per person in seconds; union overlaps for one person, sum across people. Includes initial preparation and independent human verification. Record method/resolution; unobserved is `NA`, not whole call time |
| Interventions | Integer count after initial assignment, one per discrete correction/clarification/approval/recovery episode; a multi-sentence correction is one. Final rubric grading alone is not an intervention |
| Clipboard handoffs | Integer count per manual cross-boundary transfer, including the initial prompt transfer; independent of intervention count |
| Model usage | Tokens by provider/model: uncached input, cache write, cache read, output; all supervisor, Gaia, leaf, retry and verifier requests. Preserve source semantics and request identity so inclusive parent totals are not added to child totals |
| Failures and retries | All terminal trials and internal attempts retained, including infrastructure failures, rate limits, lost logs and cancelled children; classify reason rather than deleting |

Token categories must be mutually exclusive before summation. If a provider's input
total includes cache reads/writes, retain the raw receipt and a documented mapping;
do not guess a subtraction. Reasoning tokens included in output must not be added
twice. Report per-model vectors because tokens across tokenizers are not equal units
of work. `NA` means unknown/unavailable, never zero. Zero requires a measured absence.
A missing component makes the corresponding total unknown; a known partial subtotal
may be shown only as partial, with coverage. Billing, if available, is separate:
actual charged amount + currency + billing evidence; an estimate needs pinned rate
date and is labelled estimated. Subscription token consumption is not marginal cost.

An independent reviewer records artifact identity, checks, constraints/regressions,
and the parent-consumption → previously blocked action → result chain when needed.
Retain explicit acceptance evidence and failure logs before cleaning a worktree.
Transport receipt, successful subprocess exit and candidate validation remain
separate observations. No missing log is silently interpreted as success.

## Empty recording templates

Use [observations.csv](baseline/observations.csv) for one terminal trial per row,
[usage.csv](baseline/usage.csv) for disjoint usage records and
[study-manifest.md](baseline/study-manifest.md) for the frozen design and evidence
dictionary. These are manual documentation templates, not a new runtime schema,
telemetry service, scheduler, scorecard or cross-repo contract. They do not change
existing IX event/status vocabularies. Blank files contain headers only, no results.

`trial_id` joins observations to usage. `block_id` groups matched case/repeat trials;
`repeat` and `order_index` are positive integers; `strategy` is A/B/C. All `_s`
fields are nonnegative seconds, `_count` nonnegative integers. Flags are 0/1 except
the three evidence flags above. `outcome` is accepted/failed/timeout/cancelled/
missing_evidence. `reason` is a short sanitized explanation. `manifest_ref` and
`evidence_ref` identify retained immutable artifacts, not an unsaved conversation.
The manifest records case, environment, model, cap and workflow versions to keep
the CSV compact. `usage_complete` states whether every attributable request is
represented; it does not make unknown token fields known.

In usage rows, `scope_id` is a trial ID or a separately named setup ID declared in
the manifest; `scope_kind` is trial/construction/shared_study_setup. `record_id` is
unique. `role` is preparation/execution/supervisor/leaf/verification. `coverage` is
complete/partial/unknown; `mapping_ref` defines raw-to-category accounting and
`provenance_ref` pins provider, version/alias, effort, tool, date and raw receipt.
Use disjoint request records (preferred) or disjoint aggregate ranges, never both.
Construction human/timing/count data go in the manifest's same-unit ledger.

## Analysis without a hidden winner

For each strategy/cohort, let `N` be **all initiated assigned trials**, `S` the sum
of terminal accepted flags, and `R_i` a fully observed resource of trial i.
List planned but not initiated trials separately with reason. Do not call a cohort
complete while pending trials remain. Publish success as `S/N`, counts and failure
types; missing evidence is a non-success in this conservative observed rate, not
proof the work itself was wrong. Report its count separately.

- **Resource per accepted task:** `sum(R_i for all N trials) / S`, including failed
  attempts. If `S = 0`, this is undefined, not zero. If any relevant resource is
  missing, the full metric is `NA`; report coverage and a labelled partial subtotal,
  not an optimistic mean of only measured or successful rows.
- **Accepted-trial latency:** median/range (and quantiles only if useful) of accepted
  `wall_s`, clearly conditional on success, alongside success rate and failure stop
  times. Also report all-trial resource per accepted task; do not equate the two.
- **Matched comparisons:** show paired A/B/C outcomes and differences by case and
  repeat, not just global averages. Report quality first, resources next. No ratio
  with zero or unknown denominator. Mixed model/effort/cache cohorts stay separate.
- **Construction versus reuse:** for a resource type R, show setup `K_R`, each frozen
  use `R_i`, reuse-only `sum(R_i)/S`, and fully loaded `(K_R + sum(R_i))/S`.
  Allocation over n observed uses is `K_R/n` per use (failed uses included), not
  assumed future successes. Show hypothetical n only as scenarios, never measured
  savings. No conversion between human seconds, wall seconds and tokens.

The first pilot should be small: propose three or more distinct development cases
with two or more repeats per strategy, frozen **only after** workload and budget
review. This is a design proposal, not permission to run 18+ model trials. Report
the raw matched table, medians/ranges, denominators and missingness. Repeats of one
case are not independent task samples. If uncertainty intervals are feasible,
cluster resampling by case and state the method; with very few cases emphasize
descriptive uncertainty rather than significance or general population claims.

Separately reserve **at least three distinct real holdout cases** not used in prompt,
workflow or oracle optimization. After freezing the reusable version, evaluate
transfer under the same acceptance and budget rules. Do not train on holdout
failures then call reruns untouched holdouts; retire exposed cases and declare a
new set/version. Even three successful holdouts are limited transfer evidence,
not universal reusability. No DSPy or other framework is required for this pilot.

## Launch and decision checklist

- [ ] Owner reviewed glossary, protocol and empty template; repo links/index updated.
- [ ] A/B/C implementations and task population agreed; no strategy assumed superior.
- [ ] Cases, independent oracle, parent gate, constraints and regressions frozen.
- [ ] Pilot allocation/repeats, counterbalanced order, isolation and leakage plan frozen.
- [ ] Three untouched distinct real transfer cases reserved outside optimization.
- [ ] Model/effort/version/date/tool/environment provenance and warm/cold status recorded.
- [ ] All preparation, setup, descendants, retries, verifier and failed-trial accounting covered.
- [ ] Hard stop mechanism tested offline; concurrency, budgets and advisory gaps explicit.
- [ ] Attention measurement/privacy/retention agreed; unknown usage remains `NA`.
- [ ] Named operator explicitly authorizes the bounded live run and its data/cost scope.

After the pilot, decide **continue**, **revise and rebaseline**, or **stop** against
the frozen decision rule. A quality regression cannot be offset by a weighted score.
Inconclusive evidence warrants an inconclusive result; additional runs require a
new bounded allocation, not an open-ended campaign. Documentation approval alone
never passes the final launch checkbox.

## Related IX guidance

- [Agentic engineering](agentic-engineering.md): harness hypotheses to measure.
- [ADR 0011](../adr/0011-ix-loop-lua-sandbox-model-drafts-host-judges.md): independent
  host judgement and carved budgets; do not reinterpret calls as billed dollars.
- [ADR 0019](../adr/0019-mission-scope-refinement.md): candidate/admission/resumption
  distinctions; this protocol adds no live authority or resume mechanism.
- [Agent composition synthesis](../research/agent-composition/2026-10-30-483-decision-synthesis.md):
  missing usage is unknown and live runs need bounded admission.
