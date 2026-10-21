---
title: "ixctl probes: a tree of in-process samplers under the control plane (story S1)"
type: decision
status: tracer merged into exp/ixctl; surface provisional; decisions D1–D5 assumed, tickets open
date: 2026-10-20
branch: exp/ixctl (PR #394)
reversibility: two-way today (no consumer outside go/ixctl; the surface is behind the plane's provisional protocol, door 1); becomes one-way when a program outside the plane creates nodes or reads `probe/` pipes
revisit-trigger: a decision ticket D1–D5 is resolved against the assumption recorded here; or Gaia / the Rust client reads a probe pipe (S4, S7); or the first cross-host pipe is wanted (D4 / S2)
---

# ixctl probes: a tree of in-process samplers under the control plane

French: [`docs/fr/plans/2026-10-20-ixctl-probes.md`](../fr/plans/2026-10-20-ixctl-probes.md).
Tracking: epic [ix#397](https://github.com/GuitarAlchemist/ix/issues/397), Wayfinder map
[ix#398](https://github.com/GuitarAlchemist/ix/issues/398), story S1
[ix#407](https://github.com/GuitarAlchemist/ix/issues/407). User reference:
[`go/ixctl/README.md`](../../go/ixctl/README.md#probes).

## The problem

The operator wants to **start probes** from the control plane: versioned samplers, written in Go
for now, compiled quickly, that read something (the host, a service, later a file or a port) and
put what they read on a pipe; to **list, create, start, stop** them; to **filter and aggregate**
their output; to **multiplex** children into a parent, parent into grand-parent, as a tree; to give
each a **time to live**; and to know **the system's limits** while doing it. The pain today: the
plane reports what a *service* costs (`last_exit`) and nothing about the host or about a running
service's inside, and the prototype probes (PR #390) are one OS process per probe, outside any
plane, with their own transport.

## Prior art read before designing

- `prototypes/probes` (PR #390): a Go host that runs template probes (WMI) as processes and writes
  JSONL for DuckDB. Useful as a source of future kinds; wrong shape for the plane (a process per
  probe, no tree, no limits).
- ix#374 (probe runtime boundary): bounded queues, **no silent drop**, versioned definitions apart
  from observations, one host multiplexing logical probes, measurement apart from composite
  evaluation, no actuators. This design follows every point.
- ix#375 (languages): Go first, Rust later.
- gaia#245: Gaia as a consumer of observations, through an envelope (S7, D7).
- The plane itself: the hub (ring 4096, channels, cursors with explicit gaps), the one `Request`
  entry point on HTTP and WebSocket, federation that mirrors `ctl` and `svc/` of a peer, the
  bearer token and loopback-only face. Everything a probe needs to publish, be driven and be
  secured already exists; the slice adds no second host, transport or token.

## Design (as built)

**Kinds** are compiled in, versioned, named `name@version`: `heartbeat@1` (its own tick count),
`cpu@1` (busy share from `GetSystemTimes` / `/proc/stat`; the first interval emits nothing),
`mem@1` (`GlobalMemoryStatusEx` / `/proc/meminfo`), `mux@1` (no sampler). A kind is a `sampler`
port (`sample() (fields, error)`) with one OS adapter per platform and a heartbeat adapter for
tests: the same light hexagon the plane already uses for its `Broker`.

**Nodes** are instances in one tree per plane: `{name, kind, every_ms, parent, filter, ttl_ms}`.
A node publishes `sample` events on its own hub channel `probe/<name>`; when its parent runs, the
sample goes up synchronously, through the parent's filter, onto `probe/<parent>` stamped with
`src`. The lifecycle (`probe.created/started/stopped/removed`, `stopped` and `removed` with a `reason`) is on `ctl`
like every plane event, so it is journaled and mirrored to peers for free.

**Filters** run before emission: `where` (typed terms, six operators, missing field = false),
`every` (one in N of what passed), `keep` (field projection; `node/kind/seq/src` always kept).
They are pure functions, tested by value.

**Leases**: `ttl_ms` counts from `probe.start`, `probe.touch` renews, expiry removes the subtree
with reason `expired`. A node without `ttl_ms` has no lease; `touch` refuses it rather than
inventing one.

**Ops**: `probe.kinds / list / create / start / stop / touch / remove`, one entry point on both
transports, HTTP routes `GET /v1/probes`, `GET /v1/probe`, `POST /v1/probe/<op>`, every op
forwardable to a peer. `ixctl probes` prints the kinds and limits offline, so a release binary
describes itself. `POST /v1/shutdown` stops every running node first (reason `shutdown`).

**Limits are part of the surface.** Every bound is a named constant reported by `probe.kinds`
under `limits` (`max_nodes` 256, `max_depth` 8, `min_every_ms` 10, `max_every_ms` 3 600 000,
`max_ttl_ms`, `ring` 4096, `max_where` 16, `max_keep` 32); every refusal names the bound it hit;
`probe.list` shows usage (`count`, `max_nodes`). The operator asked for this explicitly.

## Decisions

### Taken in this slice

- **Synchronous child→parent propagation, no inbox.** The first draft gave each mux a bounded
  inbox with a `dropped` counter. Replaced before the tests went green: an inbox adds a queue
  whose only honest behaviour under pressure is to drop, which ix#374 forbids, and the hub's
  ring already is the one bounded buffer with an explicit `gap`. A sample now either reaches the
  parent's filter in the child's tick or the child blocks on the parent's lock for a few
  microseconds. The cost is that a slow filter slows its children; the filters are pure and
  bounded by `max_where` / `max_keep`, so this is accepted. Revisit if a kind ever samples from
  something that can stall (a socket, a file): that kind gets a deadline, not the mux a queue.
- **One tree per plane, in the plane's process.** No second host. A kind that must not share the
  plane's process (a crash-prone reader) is a later kind that spawns, not a change to the tree.
- **Kinds compiled in, not loaded.** A new kind is a release (D8 / S5), never a plugin.

### Assumed, with their decision ticket (the slice keeps each reversible)

| Ticket | Assumption built on | What would change it |
|---|---|---|
| D1 ix#399 | Probes have their own ops, not a place in the desired-state document (door 2). | If probes should be declared and reconciled like services, `probe.create` becomes a plan/apply action. The tree code is unchanged; the entry point moves. |
| D2 ix#400 | TTL is a lease renewed by `touch`, not an idle timeout on samples. | An idle rule would read the last sample time instead of the touch time: one field. |
| D3 ix#401 | Two nodes of the same kind sample independently. | Shared sampling would memoize by kind and interval; the sampler port already allows it. |
| D4 ix#402 | Pipes are local; a peer's probe lifecycle is mirrored (`peer/<name>/ctl`), its pipes are not. | Mirroring `probe/` channels is one more prefix in `peers.go` once the volume and the gap semantics across hosts are decided (S2). |
| D5 ix#403 | A filter is `where` + `every` + `keep`, typed terms, no expressions. | IXQL or an expression language (D7) would sit above this, lowering to it or replacing it. |

## Measured (Windows, 2026-10-20, gate E15 / M2 on a loaded desktop)

| Measure | Value |
|---|---|
| Filtered samples that reached the mux in 1.5 s of a 20 ms heartbeat (`n > 2`, `every` 2) | 36, values 4, 6, 8, … |
| M2: samples per second on one mux fed by 16 heartbeats at 10 ms, read over SSE | 1447 of a 1600 ceiling, 16 sources, no overflow gap |
| M2: the plane's processor time over the 2.2 s window | 297 ms (about an eighth of one core, 3 200 envelopes per second published) |
| Lease of 1.5 s not touched | removed with reason `expired`, seen on `ctl` and gone from `probe.list` |

M2 also exposed a pre-existing fragility of the gate: with 6 400 envelopes per second the ring no
longer holds seq 1, so a later SSE read from the start gets a `since-too-old` gap first. The
gate's `sse()` helper now keeps gaps apart from envelopes (`SSE_GAPS`); the plane's behaviour was
already the documented one.

## Review round 20 (Codex, read-only, of `2575167`)

Four P1 and eight P2, no P0; all answered in the next commit. The P1s were races between an
op that unlocks between deciding and acting and another op landing in between: `remove`
snapshotted its subtree then unlocked, so a `create` under it left an orphan and a `start`
restarted a node about to be deleted (its run leaked); `expire` checked its due date, unlocked,
then removed, so a `touch` that landed first lost its node anyway; `shutdown` stopped the
running nodes without closing admission, so a `start` after it ran past the journal's close;
and a mux's `keep` dropped `src`, against the protocol's promise. The fixes keep the shape:
the subtree is claimed (`removing`) under the same lock that decides, and claimed nodes refuse
`start`, `touch` and new children; `expire` is `removeIf` with its due check inside that lock;
`shutdown` sets `closed` before its snapshot and `create`/`start` refuse afterwards; `keep`
always carries `src` and `error`. P2s: `emit` re-checks that the node is still the one in the
tree and running (a child's sample in flight no longer lands after the parent's `probe.stopped`);
a `stopping` node refuses a start until its run has returned; `max_depth` allows exactly
`max_depth` muxes above a node; filter values are typed at creation; Linux's `/proc/stat` total
stops at `steal` (guest time is inside user); a cpu interval whose counters did not advance
emits nothing instead of a fabricated idle; `probe.started` carries `expires_in_ms` when leased.
Codex also counted 38 gate checks; the file has 39 `check(` calls and the run prints 39.

**Order on the wire (2026-10-21, on the operator's go after the S7 contract named the gaps G6 and
G7).** A node's record is published under the tree lock with its `seq`, so a mux's records reach
its pipe in order under concurrent children (seen first as a CI failure of the mux test on
Windows) and never after the node's `probe.stopped`; `probe.created`, `probe.started`, a mux's
`probe.stopped` and `probe.removed` are published under that lock too, in the order of the state
changes they report; a leaf's `probe.stopped` waits for its loop outside the lock, and a remove
that meets that stop in progress waits for it (`sync.Cond` on the tree lock) before it says
`probe.removed`; a failed read reaches the pipe whatever the filter. Tests in
`go/ixctl/probe_order_test.go`; the contract is at v0.1.1 with G6 and G7 closed. Codex round 28
(of that commit) added: `shutdown` waits for a stop another op had in progress (its snapshot
lists running nodes only, so that leaf's `probe.stopped` could follow the shutdown's return and
the journal's close) — fixed with the same condition variable; the stop-in-progress tests now
drive a real `probe.stop` through a test kind whose read blocks until released (`block@1`,
registered for one test), and the start/stop test judges `ctl` after every round. **Trade-off
kept:** the `ctl` journal append already ran under the hub lock inside `Publish`; with lifecycle
events published under the tree lock, a stalled journal write now also holds the probe ops, not
only the publishers. Moving the journal behind an ordered writer changes when a line is on disk,
which is Door 3 territory: not done here. Round 29 (of the round-28 fix): `shutdown` also waits
for a removal in flight (a remove that met a stop in progress could say `probe.removed` after the
shutdown had returned) and `probe.remove` is refused once the plane is closed, like `create` and
`start`; a leaf's `finishStop` is deferred so a panic on the way cannot strand `stopping`; the
tests prove with a handshake that the waiting op reached the contested path before judging it,
release the blocking read in cleanup, and the start/stop test requires a lifecycle event every
round. **Design limit, recorded:** a sampler must return. The stop waits for the read in flight
so that `probe.stopped` is the pipe's last word, and shutdown waits for the stop; a kind whose
source can hang (an RPC, WMI in S6) bounds its own read. The alternative, abandoning a read on
cancel, would bound the stop but leak a goroutine per hung read; not taken. Round 30: two
removals overlapping on one node (a descendant's, then its ancestor's) both waited for the same
stop and the second, waking with a stale pointer, said `probe.removed` a second time — fixed by
rechecking that the node is still the tree's after the wait; the shutdown-waits-for-removals
claim gained a direct test holding the removals counter (the integration test only sees the order
after the fact), and the remaining receives in the tests are bounded. CI's `-race` row (ubuntu)
also caught a pre-existing unlocked read in `probe.stop`'s noop reply, exposed by the concurrent
start/stop test: the state is now copied under the lock.

## Tests (the seams agreed for TDD)

Unit, `go/ixctl/probe_test.go`, through `Plane.Request`, the pure filters and the OS kinds by
value range: create/stop/refusals naming bounds; heartbeat samples on its pipe until stopped; a
filter shapes what is emitted; a mux stamps `src`; a lease expires unless touched and a remove
takes the subtree; cpu and mem within range; kinds listed with limits; HTTP routes, the WebSocket
frame and shutdown ending every run; forwarding to a peer with its lifecycle mirrored; the CLI
printing the kinds offline. Gate: E15 (two checks) and M2 in `go/ixctl/e2e/e2e.py`, 39 checks.

## Left out on purpose

Cross-host pipes (S2), shared sampling (S3), the Rust and IXQL surface (S4), GitHub releases with
a probes manifest beyond the workflow skeleton (S5), the prototype's WMI kinds (S6), the Gaia
envelope's adapter (S7: the IX side is the draft contract
[`2026-10-21-ixctl-probe-observation.contract.md`](../contracts/2026-10-21-ixctl-probe-observation.contract.md)
with three consumer fixtures written by the real plane, reconnect, same-name recreation and
restart, 2026-10-21 on the operator's go after Gaia's design review; the adapter is Gaia's),
replay from a journal past the ring (the second slice of D9 / S8; the ring-backed
bookmarks shipped on [`2026-10-20-ixctl-bookmarks.md`](2026-10-20-ixctl-bookmarks.md)),
actuators (never: ix#374).

Once D1–D5 are resolved these decisions belong in an ADR under `docs/adr/`.
