# ADR-0009 — bounded mission scope preview through IXQL

Status: proposed, draft only. [Français](../fr/adr/0009-mission-scope-refinement.md)

## Current increment

Reuse the existing IXQL parser, evaluator, `Capability`, `Capabilities::register`
and injected `Host::now`. No grammar, scheduler, registry metadata migration or
automatic registration. `proposal → mission.refine_scope` returns a binding only.

A host-bound refusal of root staging `git add .` can produce the narrower proposal
`["git","add","--","Blue/mission.txt"]` when host-declared ownership and file-kind claims
name that exact regular file and contain no overlapping foreign owner. Root, unknown refusal,
duplicate scope/claims, ambiguous ownership, path escape and stale/missing/mismatched
observations fail with explicit reasons. Directories, symlinks, unknown entry kinds and directory-only coverage refuse.
The simulated host flags are declarations, not proof of real checkout inspection.
A live producer must inspect exact targets and parent reparse/symlink escapes before
admission; no such producer is implemented here. This validates declared ownership;
it does not establish effective permission or inspect the real index.

The minimal draft contract is `mission-scope.schema.json`, version `0.1.0`.
It pins DSL dialect `1`, engine package version and exact source SHA, operation
version and exact implementation SHA, mission/attempt/revision/refusal identity.
The adapter compares pins against host composition; valid SHA strings are not
authenticated provenance. Test SHA values are fixtures. No latest/range fallback.
Input/output schemas and serde types describe this operation, without a generator.

The observation envelope is host-owned, outside proposal/task text. It reuses Go
`ts/metric/value/unit/probe/build` JSONL fields from
[prototype/probes](https://github.com/GuitarAlchemist/ix/tree/dde34d73b858a2ddd71c095f3365057bfbfa1917/prototypes/probes).
The existing Go host does not emit this correlated ownership envelope. A metric of 1
does not establish ownership or progression. No live ingestion is implemented.

Observation TTL is positive and at most 30 seconds; sample age must be less than
30 seconds. The returned expiry is the earlier deadline. Missing, future, expired,
wrong-attempt, incomplete or physically unverified evidence refuses. A heartbeat
cannot replace the sample or extend its deadline.

This handler has no subprocess, effect, permission, persistent state or resume port.
Dry-run applies to this handler only; other IXQL operations can still write.
Pure replay returns the same preview for the same evidence and clock, without an
effect idempotency receipt. Every result remains `requires_runner_admission: true`.

## Gaia owns actual admission and resumption

Owner coordination reports [Gaia246](https://github.com/GuitarAlchemist/gaia/pull/246)
at `dca0792b846199ded7bf2077992c6f5fbcd41e43`: required capabilities are exactly
Read/Write/Edit/Glob/Grep with paths plus result Write. Its effective-permission
observation is bound to cwd/permission mode and a maximum 30-second TTL.
No reliable live observer or complete CLI manifest exists; unknown stops before
launch with WAITING_PERMISSION. No Bash/git, ownership or generic resume contract.

Owner-inspected [Gaia247](https://github.com/GuitarAlchemist/gaia/pull/247)
at `1957ea8a046b221862f3e48b17eec602727e9d06` retains its existing commit request
and derives named files from bound changeSetIdentity. It refuses a dirty index
and retains existing Ed25519 expiring single-consumption authority. It does not
consume this folder proposal, provide subset selection, prove dirty-checkout
ownership or expose a generic resume API. IX does not duplicate those mechanisms.

The priority is the real resume loop tested by Gaia's owner. Neither draft supplies
authority through proposed dossiers. No direct runtime binding is introduced here.
The desired receipt → first action → result/timeout proof is still outstanding.

## Validation and ordered backlog

The first RED test compiles on unchanged source but fails because the existing
executor lacks this adapter, not because Git ran unsafely:
[CI1202 Linux job](https://github.com/GuitarAlchemist/ix/actions/runs/38080368561/job/114295858387),
head `3f820b4caa739ad73c7d693ec8290ab181cee5d4`.
GREEN must exercise the real parser/registry/handler with a simulated host,
including an exact Blue file proposal, no writes, unknown refusal, ambiguity,
directories/symlinks, duplicates, correlation, expiry, exact pins and pure replay. Additional negative
tests are included together; only the first oracle has a separately captured RED.

1. Complete the minimal adapter CI and review; provide its exact evidence to Gaia.
   No status promotion, merge, permission mutation or live execution.
2. Once Gaia publishes an exact supported contract, bind this proposal to its
   existing admission boundary and prove consumption → first action → result plus
   failure/timeout, rechecking ownership/index/TTL and effect idempotency there.
3. Bind one real bounded Go observation through existing ports, without another bus.
4. Assess the existing Incubator/CP schema before mapping this draft into its catalog;
   do not create a competing lifecycle, status enum or audit database.
5. Later: canonical interface generation; maturity versus implementation/test evidence
   at exact SHA; pinned Claude/Codex skill and MCP discovery/validation/invocation
   adapters. Instructions are not executable functions and discovery grants no rights.
6. Later: Matt Pocock skill commands through versioned artifact references; verify
   installed names/versions for diagnosing-bugs, tdd, code-review and wizard.
   ask-matt means methodological routing, never communication with Matt.
7. Later: blue/green admission for new missions only, immutable active pins, pure
   shadow replay, traceable promotion/rollback and no duplicate effects.

Local Windows verification and managed worktree creation are unavailable in this
task environment. Existing GitHub CI is used on an isolated draft branch by explicit
user direction. No claim of live probe evidence, real Git refusal, resumed streaming
execution or deployed canary. Promotion still requires the repository's review and
verification gates; docs reindexing remains unexecuted unless existing CI performs it.


## Archived WMUX transport: candidate-only increment (2026-10-11)

`request → mission.dispatch_candidate` is a second opt-in capability on the same
registry seam. It produces a candidate only; there is no Python/WMUX call, live
registration, ledger read/write, waiting loop, native cancellation, or automatic
permission/admission observer. A consumer must never treat this candidate as authority.
All outputs set `live_dispatch_available: false` and `requires_runner_admission: true`.

The local archive `snapshot/ix377-transport-20261011` at
`31d75c00409ff6227a199c4921be893e369c8b4e` freezes previously unversioned Python
prototypes. It is not an upstream WMUX commit or fix. The pinned sources are:

| Source | SHA-256 |
| --- | --- |
| cycle_dispatch.py | fe5f0d23833d4dbd54b985f12ecbb3f64a489617a484c9b56b81f708e9e011e0 |
| wmux_control.py | 4aa7cc12881982540aed0c8e9971f5a7a912d5fb382b52621307569d75b703b9 |
| wmux_mission.py | 23f0db454cd4372fc62292fb31d756bac177f4bc7f9d8d9dfc10a7080e924f4d |

These three frozen source hashes were independently recomputed locally for integration.
The archive SHA is owner-reported; the 54 existing Python fixture tests are also
owner-reported, not rerun here. No Python implementation is copied into IX.

The typed `DispatchRequest` is an adapter contract, not the existing Python API.
It pins contract/dialect/engine/implementation/archive/source hashes and the
mission/nonce, full session+workspace+surface tuple, exact brief digest and declared
brief/registry/ledger paths. Host composition injects `DispatchObservation`; no
producer verifies the real GUI, files or permissions here. Nonempty/unknown input
or non-ready admission refuses a prepared candidate. Changed tuple, nonce, mission,
artifacts or deadline refuses against that exact host declaration. Duplicate
registration cannot replace the handler.

A prepared state can suggest `dispatch`, but execution stays unavailable because
the transport does not offer atomic input reservation/compare-and-send. Empty
input in a fixture is not proof against concurrent typing. Registry lock and
GUI ownership are separate. This adapter implements no live concurrency fence.

Any attempted/ambiguous or later known transport state suggests `observe_receipt`
using the same mission, nonce and artifacts; repeated candidates are pure
projections, not effect idempotency receipts. No resend or changed identity is
generated. Unknown or inconsistent state refuses. Transport acceptance,
submitted-prompt reception, provider working and consumed author claims are
kept distinct through the projected state; none becomes a completed-tool proof,
verdict or independently verified success. `first_action_confirmed` stays false;
`effect_outcome` stays unknown, including on consumed state.

The request JSON budget is 4096 serialized bytes; IDs are at most 128 ASCII
characters and three distinct canonical relative artifact paths at most 512
characters each. This is only an adapter budget: Python brief/result/ledger
reads and subprocess capture buffers are not byte bounded by it. No file access
or payload authority is inferred from syntactic path validation.

The declared observation is fresh for less than 30 seconds, a proposed adapter
constraint. `wait_until_ms-created_at_ms` must be 1..3,600,000 ms with checked
arithmetic and no future creation. At the deadline the candidate suggests
`stop_waiting`, preserving unknown effect outcome and
`native_execution_cancelled: false`. This does not implement a transport TTL,
an actual bounded wait, abandonment or cancellation. Reconciliation belongs to
the existing transport owner, not an additional IX scheduler.

The only real Blue round trip ended at 2026-10-11T01:08:18.7337442Z. Root observed
the submitted prompt, explicit reception, native successful Read 2 files and
result. Its one-attempt `relay_send.py` did **not** use
`cycle_dispatch.dispatch`, registry lock/fsync, ledger binding or a mission token.
Its intent write was not exclusive/fsynced. This real case therefore cannot
validate those prototype mechanisms. Post-run nonempty input remained untouched;
its draft-versus-suggestion origin is unknown. No second Blue send occurred.

Evidence pins: original relay-case
`a3066b102836477cf7b590238ccea08510c48dfd6e5acfc92b07bea97dd1e778`;
redacted projection
`c025c028309b3d72956d1734313dbf3a2726bae790d026d6d075f5b0d6bb5d7a`;
root-relay-proof
`91981136ede4c3d2b5728c8a99664676e4688baab3b605fe8349f0ddedc42822`.
Digest pinning establishes artifact integrity, not authenticated execution.

Parser/adapter fixtures cover replay, wrong IDs/nonce/artifacts, unknown/nonempty
input, freshness boundaries, local deadline expiry/overflow, pins, JSON budget,
path aliases, registry/parser duplicates and evidence distinctions. They model
snapshots, not GUI interleavings, crash recovery or live transport behavior.
The first separately captured RED checks the missing capability, not a live
transport failure. There is no sequence/parallel dependency or new exception engine.

Integration proposal: expose this advisory candidate through the Incubator's
existing mission proposal view and existing permission gate only after its exact
schema/consumer is agreed. No UI/CP files are changed, no new Incubator is created,
and no candidate is an authority dossier. Revisit live dispatch only when the
transport owner supplies and tests the required admission fence and reconciliation
contract. Gaia246/247 remain separate contracts and supply no generic resume API.
