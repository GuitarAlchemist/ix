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
`src`. The lifecycle (`probe.created/started/stopped/removed`, each with a `reason`) is on `ctl`
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
envelope (S7), bookmarks and replay of pipeline state combined with probes (D9 / S8), actuators
(never: ix#374).

Once D1–D5 are resolved these decisions belong in an ADR under `docs/adr/`.
