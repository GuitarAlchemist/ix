# Contract: a probe sample read from the plane, as a consumer's observation (`ixctl.v1` → gaia#245)

**Version:** v0.1.6 (DRAFT — Phase 0; freeze only at the named Phase 4 milestone, after Gaia's intake)
**Date:** 2026-10-21 · **Owner:** ix (`go/ixctl`, the producer side; Gaia owns the adapter, projection and UI) · **Story:** [ix#413](https://github.com/GuitarAlchemist/ix/issues/413) · **Consumer:** [gaia#245](https://github.com/GuitarAlchemist/gaia/issues/245) · **Plan:** `docs/plans/2026-10-20-ixctl-probes.md`

Says how a reader of the control plane's event stream (`subscribe` over `ixctl.v1`, `GET /v1/events`,
or `Client.Subscribe*`) maps what it reads onto the observation envelope gaia#245 asks for: probe
identity and version, schema version, observed-at and received-at, sequence and deduplication key,
units, sampling interval, freshness and the state OK / STALE / UNKNOWN / ERROR. Nothing here changes
the plane; it names what the plane already says, what the consumer must derive, and what the plane
does not say (the gaps, at the end). Three fixtures written by the real plane show the cases Gaia's
design review asked about: a reconnect, a same-name recreation of a node, and a restart of the plane.

## What the plane says

Every frame a subscription delivers is either an **envelope** or a **gap** (`go/ixctl/hub.go`,
`go/ixctl/README.md` "Protocol"):

```jsonc
// envelope: one event on one channel
{"ch": "probe/hb", "seq": 3, "epoch": "ec28e8abfbd2", "origin": "a", "t": 1792554072183,
 "ev": {"kind": "sample", "t": 1792554072183,
        "data": {"kind": "heartbeat@1", "node": "hb", "seq": 1, "n": 1}}}
// gap: what the reader did not get, and why
{"reason": "epoch-changed", "from": 1, "oldest": 1, "epoch": "78f5db688657"}
```

| Envelope field | Meaning | Locked? |
|---|---|---|
| `ch` | channel: `probe/<node>` for a node's samples, `ctl` for the lifecycle, `svc/<name>` for a service's pipe; on a parent plane, `peer/<peer>/probe/<node>` for a peer's exported, requested pipe (v0.1.3) | v1 protocol |
| `seq` | the instance's sequence, shared by every channel, from 1 at each boot | v1 protocol |
| `epoch` | the boot's id; a new one at every start of the plane | v1 protocol |
| `origin` | the instance (plane) name that produced the event; a peer mirror keeps the origin's | v1 protocol |
| `ev.kind` = `link.gap` | on a mirrored channel only, the **continuity marker**: an envelope of the parent's own (`origin` the parent, no `src_seq` / `src_epoch`) saying the mirror skipped something it cannot replay. On `peer/<peer>/probe/<node>`, every live start of the pipe: `data.reason` `live-start`, `src_epoch` the source boot now read (absent while the link has not read the boot of the instance it is connected to: a new instance whose status was unreadable and of which nothing was mirrored yet), `last_src_seq` / `last_src_epoch` the last frame mirrored on that pipe before (absent when none; across a same-name recreation it is the previous incarnation's). On `peer/<peer>/ctl`: without `prefix`, `epoch-changed` when the link sees a new boot of the peer, from the status, a queued frame or the first frame mirrored, whichever first (`src_epoch` the new boot, `last_src_epoch` the one before); with `prefix` (`ctl` or `svc/`), a gap the peer answered on the link's resumed subscription (`reason` `since-too-old` or `epoch-changed`, `from`, `oldest`, `src_epoch` as the peer said). On a pipe again (v0.1.6): `dropped`, right before the record that followed a run of records the link itself discarded for want of room in its bounded pipe queue, `dropped` their number (the link's own exact count of what it discarded, never an inference from the source's seqs; the same records are counted on the parent's `status.peers[].probes[].dropped`); and `overflow` when the peer dropped the pipe's subscription toward the link (`from`, `src_epoch` as the peer said), the pipe then reopened live and marked `live-start` as at any live start. A boundary, never a count of what the source did: only the link's own discards are counted. A local gap stays a gap frame (`reason`, `from`), v0.1.5 | S2 (ix#408) |
| `src_seq`, `src_epoch` | on a mirrored envelope (`peer/<peer>/...`) only: the source's own `seq` and `epoch`, so a consumer of the mirror keys the series `origin:src_epoch:node:<src_seq of the probe.created>` and dedups on `origin:src_epoch:src_seq`, as a consumer of the source would on `seq` / `epoch` (v0.1.4) | S2 (ix#408) |
| `t` | when the plane published it, ms since the Unix epoch, the plane's clock | v1 protocol |
| `ev.kind` | `sample` on a node's pipe; `probe.created`, `probe.started`, `probe.stopped`, `probe.removed` on `ctl`, published in the order of the state changes they report (see "Order", below) | v1 protocol |
| `ev.data.node`, `ev.data.kind` | the node's name and its kind with its version (`name@version`) | S1 (ix#407) |
| `ev.data.seq` | the node's own count of records emitted, from 1 at the node's creation; a stop and a start continue the count, a removal and a recreation under the same name start again at 1 | S1 |
| `ev.data.src` | on a mux's pipe only: the child the record came from | S1 |
| `ev.data.reason` | on `probe.stopped` / `probe.removed`: `requested`, `expired`, `removed`, `parent-removed`, `shutdown` | S1 |
| other `ev.data` fields | the kind's measurements, whose names `probe.kinds` lists (`fields`, names only); a read that failed emits a sample with `error` (the message) and no measurements, whatever the node's filter (`where` and `every` do not apply to it, and it does not count toward `every`) | per kind |

Gaps: `since-too-old` (the ring no longer holds what was asked; `oldest` is the first seq still held),
`overflow` (the reader fell behind the ring), `epoch-changed` (the seq given belongs to another boot;
the replay is the new boot from the oldest seq its ring still holds, `oldest`: 1 while that ring has
not turned, as in fixture 3), `since-ahead` (no epoch was given and the seq is past this boot's head).
With this boot's epoch and a seq past its head the plane answers no gap and starts live: a consumer
holding a seq this boot never reached has a corrupt position, and a live start with no replay after
a resume is the sign of it. `t` and `ev.t` are two separate clock reads and may differ; `ev.t` is
the event's, `t` the publication's.

**Order.** Every record of a node is published under the tree's lock with the `ev.data.seq` it was
given, so a node's records reach its pipe in `seq` order even when several children feed a mux at the
same instant, and no record of a node follows its `probe.stopped`. `probe.created`, `probe.started`, a
mux's `probe.stopped` and `probe.removed` are published under that same lock, in the order of the state
changes they report: after any interleaving of ops, the last lifecycle event of a node names its state,
and a `probe.removed` never follows the `probe.created` of a later node of the same name. A leaf's
`probe.stopped` alone is published after its loop's last tick, outside the lock; until it is out the node
refuses `probe.start` and a `probe.remove` of it waits, so `ctl` says `probe.stopped` then `probe.removed`;
the plane's shutdown waits for such a stop too, so nothing of the tree is said after it (tests
`go/ixctl/probe_order_test.go`; the mux-order case was first seen as a CI failure of
`TestAMuxMultiplexesItsChildrenStampedWithTheirSource`, the start/stop case is Codex's reasoning on the
previous code, reproduced by no run: the test stands as its guard).

## The mapping

| gaia#245 asks for | From the plane | Derived by the consumer |
|---|---|---|
| probe identity | `origin` + `ev.data.node` | a **series** is `origin:epoch:node:<seq of the probe.created that opened it>` (on a mirrored channel, `origin:src_epoch:node:<src_seq of that probe.created>`, v0.1.4): the same name is a new series after a recreation or a restart (fixtures 2 and 3). A consumer that did not read the `probe.created` (it joined late, or a gap swallowed it) cannot tell which series a sample belongs to until the plane stamps an instance id on samples (gap G1): it opens a provisional series at the first frame it reads and merges nothing across a gap |
| probe version | `ev.data.kind` (`heartbeat@1`) | |
| schema version | the protocol name, `ixctl.v1` (gap G4) | this contract's version |
| subject identity / generation | `origin` (the plane) and `epoch` (its boot); on a mirrored channel, `origin` and **`src_epoch`**: `epoch` there is the mirror's own boot and does not change when the source restarts (v0.1.4) | |
| producer / source revision | `probe.kinds` reports the binary's version; `/health` the plane's | read once per connection, kept with the series |
| observed-at | `ev.t` | |
| received-at | | the consumer's own clock at read time; its difference to `t` is the clock offset to apply when judging freshness |
| sequence | `seq`; on a mirrored channel **`src_seq`** (v0.1.4) | strictly increasing within an epoch, across every channel of that plane; the mirror's `seq` orders the mirror's own stream only |
| deduplication key | | **`origin:epoch:seq`**: a frame read twice (a reconnect that resumes below the last seq kept) is the same frame; fixture 1 shows a resume from the last seq kept gives `seq + 1` first, nothing twice; on a mirrored channel the key is **`origin:src_epoch:src_seq`**, since `seq` / `epoch` there are the mirror's own (v0.1.4) |
| units | not on the wire: `probe.kinds` → `fields` lists the names only, `doc` says the kind in words (gap G3) | a per-kind table the adapter keeps, by `kind@version` |
| sampling interval | not in the sample: `every_ms` of the node from `probe.list` | read at `probe.created`/`probe.started`, kept with the series |
| declared limits (cadence, maximum rate) | `probe.list` → `rate_per_s` (what the node's subtree declares to emit, the worst case) and a mux's `max_rate_per_s`; `probe.kinds` → `limits` (`max_rate_per_s`, `tree_rate_per_s`, the plane's budget, D10 ix#416); `bytes` on the list is what the node emitted, measured | read once per series; CPU and RSS are not attributed to probes by the plane |
| freshness TTL | not the plane's: `ttl_ms` on a node is its **lease**, not a freshness bound | the consumer's own, suggested `3 × every_ms` |
| state | | see below |

### State

- **OK**: the series' latest sample has `t` within the freshness TTL of now, corrected by the clock
  offset, and no `probe.stopped` has been read for the node since that sample.
- **STALE**: the latest sample is older than the TTL; or `probe.stopped` was read for the node
  (`reason` says why: `requested`, `expired`, `shutdown`, `removed`, `parent-removed`) and no
  `probe.started` since (fixture 2, step `stopped`). `probe.stopped` is the last thing a node's pipe
  says until it starts again, for a leaf (the stop waits for the node's loop to end) and for a mux
  (its records and its stop are published under one lock); a sample read after a `probe.stopped`
  for its node would be a plane bug, and a consumer counts it into the stopped series and stays
  STALE. After `probe.removed` the series is closed, not stale.
- **UNKNOWN**: before the first sample of a series; and after any gap (`epoch-changed`,
  `since-too-old`, `overflow`, `since-ahead`) until the next `probe.created` or sample of the node,
  because the consumer cannot say what happened in the hole. On a mirrored channel a `link.gap`
  envelope is read as a gap: on a pipe it closes that pipe's open series (the next sample opens a
  provisional one, keyed on `src_epoch`); on `peer/<peer>/ctl` with `epoch-changed` every series
  of that origin ends, with `since-too-old` every series of that origin is unknown until its next
  `probe.created` or sample (v0.1.5). After `epoch-changed` **every** series
  of that origin is unknown: the plane restarted with an empty tree (fixture 3; the tree is not
  persisted, decision D1 ix#399), and the new boot's `seq` restarts at 1 while the old boot's series
  end.
- **ERROR**: a kind that fails a read emits a `sample` whose data is `{node, kind, seq, error}`
  (the message, no measurements), whatever the node's filter, and counts it in `errors` on
  `probe.list`. ERROR is the state while the series' latest sample is fresh and carries `error`; the
  next sample without it is OK again. `errors` on `probe.list` is a cumulative count with no time
  and no recovery status: evidence that errors occurred since the node's creation, never the current
  state, which is read from the pipe. A node whose kind has no sampler on this platform refuses
  `probe.start` instead (no series).
- **Precedence**: STALE before ERROR before OK. A series whose latest sample is old, or whose node
  was stopped, is STALE whatever that sample carried; UNKNOWN applies only to the windows named
  above, and ends at the first frame of the node that follows.

## Fixtures (`go/ixctl/e2e/fixtures/consumer-*.jsonl`)

One frame per line, `{"step": "<step>", "env": {...}}` or `{"step": "<step>", "gap": {...}}`, written
by the real plane from `TestAConsumerSeesReconnectRecreationAndRestartInItsFixtures`
(`go/ixctl/consumer_fixture_test.go`) with `IXCTL_WRITE_FIXTURES=1`; on every other run the test
checks that the plane still produces, step by step, the same **shapes** (channel, event kind, node,
source, or gap reason) the files hold, and asserts the invariants below on the fresh run. Counts
and values (`seq`, `t`, `epoch`) vary with timing and are not compared.

1. **`consumer-reconnect.jsonl`**, steps `live` then `after-reconnect`. A reader of everything
   (`ch` prefix `""`) sees `probe.created`, `probe.started` and samples (the node's loop starts
   before `probe.started` is published, so its first sample may precede the event; the fixture
   check accepts either order within the step); it drops its connection
   after the sample at `seq` *L* and resumes with `since: L, epoch`. Invariants: the first frame
   after the reconnect is `seq` *L* + 1; every `seq` is the previous + 1; no gap; one `epoch`, one
   `origin`. Dedup key `origin:epoch:seq` sees nothing twice.
2. **`consumer-recreation.jsonl`**, steps `stopped`, `removed`, `recreated`. The node is stopped
   (`probe.stopped`, `reason: requested`), removed (`probe.removed`, `reason: removed`), and created
   and started again under the same name. Invariants: the recreated node's first sample has
   `ev.data.seq` 1 while its envelope `seq` is past every earlier one, same `epoch`; no gap. A
   consumer keyed on `node` alone would see the node's count fall from 6 to 1 inside one epoch;
   keyed on the series (`probe.created` at `seq` 11) it sees a new series.
3. **`consumer-restart.jsonl`**, steps `resumed-after-restart`, `recreated-after-restart`. The plane
   is restarted under its name in the same state dir; `probe.list` answers an empty tree; a reader
   resuming with the last `seq` and `epoch` it kept gets `epoch-changed` first (`from: 1`,
   `oldest: 1`, the new `epoch`), then the new boot from the oldest `seq` its ring holds, here 1:
   `probe.created`, `probe.started` and samples with `ev.data.seq` from 1 (same order caveat as
   fixture 1). Invariants: the gap names the new boot; every envelope after
   it is of the new boot; the new boot's envelope `seq` 1 is below the old boot's last `seq`, so a
   consumer keyed on `seq` without `epoch` would merge the two boots.

## Gaps (what the plane does not say today)

- **G1, no node instance id.** A recreated node has the same `node` and `kind`; only the
  `probe.created` that opened it tells the series apart (fixture 2). A consumer must read `ctl`
  beside the pipe, and one that missed the `probe.created` (joined late, or lost it in a gap) cannot
  resolve the series at all until the plane stamps an instance id on samples (a story under ix#397,
  not cut).
- **G2, the interval is not on the sample.** `every_ms` comes from `probe.list`; a reader that
  joined late reads it there.
- **G3, no types or units on the wire.** `probe.kinds` lists field names and a `doc` string per
  kind; types and units are the adapter's table, by `kind@version`.
- **G4, no schema version on the envelope.** The protocol name `ixctl.v1` and this contract's
  version stand in (`probe.kinds` → `version` is the binary's, a producer revision, not a schema
  version); a field would be a protocol change (door F1, one-way).
- **G5, the tree is not persisted across a restart** (D1 ix#399): after `epoch-changed` the
  consumer waits for the tree to be recreated by whoever owns it.
- **G6, closed in v0.1.1** (it named two windows: a child's record published after the mux's
  `probe.stopped`, and a `probe.remove` racing a `probe.stop` of the same leaf saying `probe.removed`
  before `probe.stopped`; a third, start/stop ops publishing their events in the opposite order of
  their state changes, came from Codex's review of v0.1.0). The plane now publishes records and
  lifecycle events under the tree lock and makes removal wait for a stop in progress: see "Order".
  The consumer's rule stays as a defence: `probe.removed` closes the series; a `probe.stopped` or a
  sample of the same series read after it is ignored; a sample read after `probe.stopped` counts
  into the stopped series.
- **G7, closed in v0.1.1** (a node's filter hid its error samples). A failed read now reaches the
  pipe whatever the filter; `errors` on `probe.list` remains the cumulative count.
- **G8, closed in v0.1.5** (what a mirrored pipe lost across a link outage was not marked: a pipe
  starts live on every session of the link and after every drop, never from the last mirrored
  seq, and the local ring's `gap` said nothing about it). The link now publishes a `link.gap`
  envelope on the mirrored pipe at every live start, and on the peer's mirrored `ctl` when the
  peer came back as a new instance or answered a gap on the resumed `ctl` / `svc/` (envelope
  table). What stays: the marker is a boundary, not a count, and the pipe still never resumes
  (the bounded replay and the cursor-bound list are later slices of S2).

## Changes

- v0.1.6 (2026-10-22): S2 slice 3, second cut (criterion 6 of ix#408): two more `link.gap` reasons
  on a mirrored pipe, `dropped` (records the link itself discarded for want of room in its bounded
  pipe queue, with their exact number, before the record that followed) and `overflow` (the peer
  dropped the pipe's subscription; the pipe reopens live, marked). Both are read as any pipe
  marker (State, UNKNOWN): the open series closes, the next sample opens a provisional one. The
  parent's status counts the discarded records per pipe. `ctl` and `svc/` are never shed by the
  link itself (the peer's overflow of those subscriptions ends the session, resumed on reconnect;
  a replay the peer's ring no longer holds is a `since-too-old` gap, marked). G1 unchanged.
- v0.1.5 (2026-10-22): S2 slice 3, first cut: the continuity marker `link.gap` on a mirrored
  channel (envelope table), the parent's own envelope, at a pipe's live start and on the mirrored
  `ctl` for the peer's restart or an answered gap; read as a gap by the consumer (State, UNKNOWN).
  G8 closed. G1 unchanged (a mirrored series is still provisional without the mirrored
  `probe.created`).
- v0.1.4 (2026-10-21): S2 slice 2: a mirrored envelope carries `src_seq` / `src_epoch`, the
  source's own, so the mirrored series and dedup keys are `origin:src_epoch:node:<src_seq of
  probe.created>` and `origin:src_epoch:src_seq`; the Rust client's `Event` exposes them
  (`src_seq`, `src_epoch`, absent on the plane's own events, read as a pair or not at all). The
  subject-generation and sequence rows name `src_epoch` / `src_seq` on a mirrored channel. The
  link refuses a peer frame without a seq or an epoch (it ends the session) rather than mirror
  it without provenance. G1 unchanged (no instance id on samples), G8 unchanged.
- v0.1.3 (2026-10-21): D4 (ix#402) resolved, S2 (ix#408) slice 1: `probe.created` and
  `probe.list` carry `export: true` on a node whose pipe federation may mirror; a peer that names
  the node in `peers[].probes` reads it as `peer/<peer>/probe/<node>` with `origin` kept (the
  source's `seq` / `epoch` are not on the mirrored envelope yet: slice 2 adds `src_seq` /
  `src_epoch`, and the series becomes `origin:src_epoch:node:<created seq>`). G1 unchanged; G8 opened (what a mirrored pipe loses across a link outage).
- v0.1.2 (2026-10-21): D10 (ix#416) resolved and shipped: `probe.list` carries `bytes`,
  `rate_per_s` and a mux's `max_rate_per_s`; `probe.kinds` → `limits` carries the plane's
  `max_rate_per_s` / `tree_rate_per_s`; a create past a budget is refused in advance (400, no
  event). What gaia#245 asks declared before a sample (cadence, maximum rate) is readable from the
  list and the limits; dropped samples remain none by construction (synchronous propagation; the
  ring's `gap` is the only loss); CPU and RSS are not attributed to probes.
- v0.1.1 (2026-10-21): the plane closes G6 and G7 on the operator's go (records and lifecycle
  events published under the tree lock, removal waits for a stop in progress, error samples bypass
  the filter; `go/ixctl/probe_order_test.go`). Also from Codex's review of v0.1.0: `since-ahead`
  listed under UNKNOWN; `errors` named as cumulative evidence, not a state; the series rule says
  what a consumer that missed `probe.created` can and cannot do (G1); `probe.kinds` → `version`
  moved from "schema version" to producer revision (G4).
- v0.1.0 (2026-10-21): first draft, from Gaia's design review of PR #394 (reconnect/restart and
  same-name recreation fixtures, source identity, visible staleness, no dedup collisions) and the
  operator's go. Nothing locked; a locked field set comes with Gaia's intake (Phase 4).
