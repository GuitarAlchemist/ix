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
