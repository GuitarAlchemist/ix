---
title: "IX Windows service: reloadable modules later, as versioned admissions with a bounded drain (not hot reload)"
type: decision
status: proposed; nothing built; the first service slice (feat/ix-windows-service) does not depend on it
date: 2026-10-30
branch: feat/ix-windows-service
reversibility: two-way (no code, no contract field); becomes one-way when a version field enters the run identity on the wire or the spec schema
revisit-trigger: the first workload hosted under the service that must change version without stopping the plane (ix-loop as a supervised workload); or ADR-0016–0018 (PR #481) signed, which fixes how a durable module is versioned
---

# IX Windows service: reloadable modules later

French: [`docs/fr/plans/2026-10-30-ixctl-service-reloadable-modules.md`](../fr/plans/2026-10-30-ixctl-service-reloadable-modules.md).
User reference: [`go/ixctl/README.md`](../../go/ixctl/README.md#windows-service).

## The problem

The brief for the IX Windows service (step 6) asks how the service will later take a new version
of a module without being stopped: work in flight stays on its version, new admissions go to the
new one, the old one drains, and a rollback is possible. It also says not to promise full hot
reload. This note records the shape so the first slice does not close it off; nothing here is
built.

## What exists today (go/ixctl, main at 99f2f869)

- A plane reads its spec once, at boot (`LoadSpec`). A service's command, arguments and
  directory are fixed for the plane's life. The desired state says only `running` or `stopped`
  per service.
- Every run already carries the apply that launched it: `service.started` names its
  `generation`, and a restart keeps it. Generations are a total order (`applyMu`).
- Admissions close at a stop: once the stop is received, an apply answers 503.
- A stop is bounded and tells requested, received, and observed or unobserved apart; a process
  tree is the plane's by its Job Object and dies with the host.
- Replacing ixctl itself is a service stop and start: a new epoch at generation 0, and the
  desired state is not replayed (ADR-0014).
- ix-loop's store accepts one proposal as one file; it has no version of a script.

## Proposed shape

1. **A module version is part of the spec, not of the process.** A supervised service gains a
   `version` and the digest of what it runs, written `name@version` as probe kinds already are.
   A durable module keeps ADR-0018's identity (SemVer, a lock, a closure); the service does not
   resolve versions, it supervises what a lock names.
2. **A run keeps its version.** Run identity gains the version it launched with, as it already
   carries its generation. ADR-0018 states the same rule for a durable run: the next resume
   replays on the old bytes.
3. **Loading a version is an admission switch.** A `load` names the new version; from that
   instant every new start and every new run goes to it. Runs of the old version are not
   touched.
4. **The old version drains, bounded.** No admission reaches it; its runs end on their own or at
   the drain bound, and then stop with the same three moments and the same unobserved outcome as
   a service stop. An effect a killed run may have made is not replayed: in a durable run it
   stays `UNKNOWN` until reconciled (ADR-0016).
5. **Rollback is another load.** Loading the previous version moves admissions back. It needs
   that version's bytes, so the plane keeps a version until no run names it, plus the one before
   the current. Rollback undoes no effect a run of the newer version made.

## Not promised

- Swapping code inside a running process. A version change ends old runs; it never patches them.
- Reloading ixctl itself: that is a service restart, with ADR-0014's loss of the desired state.
- Migrating a run's state between versions, or reading an old journal with new code.

## Open questions for Stef

- Where versions live on disk: one spec file per version, or one spec naming several.
- How many old versions the plane keeps beyond the rule in point 5.
