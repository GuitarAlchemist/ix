# #483 synthesis: the decisions the operator can take now, and the ones that must wait

One page over the agent-composition map [#483](https://github.com/GuitarAlchemist/ix/issues/483).
It summarizes the three research notes beside it; the evidence is in them, not here. It decides
nothing: each line is a recommendation with its door and its revisit trigger.

## Where each leaf stands (2026-10-30)

| Leaf | Evidence so far | Missing before a decision |
|---|---|---|
| [#484](https://github.com/GuitarAlchemist/ix/issues/484) session vs task | Claude Code 2.1.314 pinned; documented signal per event class; two classes (request acceptance, end of input) have none; one process per ask already exists twice (ix-loop, ix-baml) | the smallest check (two sessions, late result from an old incarnation, malformed JSONL, permission wait, exit) on synthetic fixtures; a cost measurement on 2.1.314 |
| [#485](https://github.com/GuitarAlchemist/ix/issues/485) context snapshots | inventory: memory (`ReadContext`), durable disk (`ix-recover`, SHA-256); no shared document with expected-version updates; one silent overwrite found and fixed (#491) | checks 1, 2 and 4 of the note (private snapshot, restored snapshot, missing/changed/incompatible inputs) |
| [#489](https://github.com/GuitarAlchemist/ix/issues/489) Jev decision node | 24 offline tests; three gaps pinned (timeout = provider failure, tie unflagged, missing usage counted as zero); seams named | the operator's choice on the outcome set; anything about Jev 1.14 |
| [#486](https://github.com/GuitarAlchemist/ix/issues/486) needs, [#487](https://github.com/GuitarAlchemist/ix/issues/487) traffic classes, [#488](https://github.com/GuitarAlchemist/ix/issues/488) falsifying journey | not started | everything; #486 and #488 also wait on [#420](https://github.com/GuitarAlchemist/ix/issues/420) |

## Decisions ready now

| # | Decision | Recommendation | Door | Revisit when |
|---|---|---|---|---|
| D1 | #484: first agent profile | **One process per task**, the shape ix-loop and ix-baml already run, with a validated JSON envelope; reused sessions deferred | two-way: a reused session can be added behind the same task contract | a measured fixed context cost on 2.1.314 makes per-task processes too dear, or a stream-json fixture classifier passes #484's check |
| D2 | #485: how a task receives context | **Explicit immutable artifact lists** (`id, source, sha256, schema, classification, bytes`, ix-recover's digest format); no ContextStore; no shared writes in the first slice | two-way: lists can later be served by a store | a second concrete storage mode needs shared updates, which then need expected-version checks (none exist) |
| D3 | #489: decision-node outcome | A Capability returning a **typed outcome** (`choice, abstain, invalid, timeout, provider_unavailable, missing`) with provenance; **no Verdict** derived from probabilities | one-way once a pipeline keys on the outcome names: version it (`v0`) until then | a consumer needs a value outside the six, or calibration evidence exists for a probability-derived verdict |
| D4 | next code slice | the offline recorded-response classifier for #489 (three gaps as red tests), inside the spike crate, before any IXQL wiring | two-way (throwaway-able) | D3 changes |

## Decisions that must wait

- **#420 first.** Needs (#486), the falsifying journey (#488) and any live admission depend on the
  actuator authority contract; nothing above grants or assumes authority.
- **#487** needs a saturation fixture before any transport choice; no evidence was gathered.
- **Live runs** (a recorded Claude stream, a Jev 1.14 call) need their own bounded admission: cost,
  key, data that leaves the machine.

## Consistency checks between the leaves

- `abstain` (D3) is an answer, not a Need: a branch that cannot decide raises a #486 Need, it does
  not abstain. The two must stay distinct outcome kinds.
- D1's envelope and D3's outcome share the rule #484 states: an unknown, malformed or missing record
  is never a success, and missing usage is unknown cost, never zero.
- D2's lists carry what D1's task receives; neither copies a parent's conversation into a child.

French: [`docs/fr/research/agent-composition/2026-10-30-483-decision-synthesis.md`](../../fr/research/agent-composition/2026-10-30-483-decision-synthesis.md).
