# Contract: a probe sample read from the plane, as a consumer's observation (`ixctl.v1` → gaia#245)

**Version:** v0.1.0 (DRAFT — Phase 0; freeze only at the named Phase 4 milestone, after Gaia's intake)
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
| `ch` | channel: `probe/<node>` for a node's samples, `ctl` for the lifecycle, `svc/<name>` for a service's pipe | v1 protocol |
| `seq` | the instance's sequence, shared by every channel, from 1 at each boot | v1 protocol |
| `epoch` | the boot's id; a new one at every start of the plane | v1 protocol |
| `origin` | the instance (plane) name that produced the event; a peer mirror keeps the origin's | v1 protocol |
| `t` | when the plane published it, ms since the Unix epoch, the plane's clock | v1 protocol |
| `ev.kind` | `sample` on a node's pipe; `probe.created`, `probe.started`, `probe.stopped`, `probe.removed` on `ctl` | v1 protocol |
| `ev.data.node`, `ev.data.kind` | the node's name and its kind with its version (`name@version`) | S1 (ix#407) |
| `ev.data.seq` | the node's own count of records emitted, from 1 at each start of that node | S1 |
| `ev.data.src` | on a mux's pipe only: the child the record came from | S1 |
| `ev.data.reason` | on `probe.stopped` / `probe.removed`: `requested`, `expired`, `removed`, `parent-removed`, `shutdown` | S1 |
| other `ev.data` fields | the kind's measurements, named and typed by `probe.kinds` (`fields`) | per kind |

Gaps: `since-too-old` (the ring no longer holds what was asked; `oldest` is the first seq still held),
`overflow` (the reader fell behind the ring), `epoch-changed` (the seq given belongs to another boot;
the replay is the new boot from its seq 1), `since-ahead` (a seq this boot has not reached).

## The mapping

| gaia#245 asks for | From the plane | Derived by the consumer |
|---|---|---|
| probe identity | `origin` + `ev.data.node` | a **series** is `origin:epoch:node:<seq of the probe.created that opened it>`: the same name is a new series after a recreation or a restart (fixtures 2 and 3) |
| probe version | `ev.data.kind` (`heartbeat@1`) | |
| schema version | the protocol, `ixctl.v1`; the kind registry's `version` from `probe.kinds` | this contract's version |
| subject identity / generation | `origin` (the plane) and `epoch` (its boot) | |
| producer / source revision | `probe.kinds` reports the binary's version; `/health` the plane's | read once per connection, kept with the series |
| observed-at | `ev.t` (equal to `t` for a sample) | |
| received-at | | the consumer's own clock at read time; its difference to `t` is the clock offset to apply when judging freshness |
| sequence | `seq` | strictly increasing within an epoch, across every channel |
| deduplication key | | **`origin:epoch:seq`**: a frame read twice (a reconnect that resumes below the last seq kept) is the same frame; fixture 1 shows a resume from the last seq kept gives `seq + 1` first, nothing twice |
| units | `probe.kinds` → `fields` of the kind | |
| sampling interval | not in the sample: `every_ms` of the node from `probe.list` | read at `probe.created`/`probe.started`, kept with the series |
| freshness TTL | not the plane's: `ttl_ms` on a node is its **lease**, not a freshness bound | the consumer's own, suggested `3 × every_ms` |
| state | | see below |

### State

- **OK**: the series' latest sample has `t` within the freshness TTL of now, corrected by the clock
  offset, and no `probe.stopped` has been read for the node since that sample.
- **STALE**: the latest sample is older than the TTL; or `probe.stopped` was read for the node
  (`reason` says why: `requested`, `expired`, `shutdown`, `removed`, `parent-removed`) and no
  `probe.started` since. `probe.stopped` is the last thing said about a node until it starts again
  (fixture 2, step `stopped`). After `probe.removed` the series is closed, not stale.
- **UNKNOWN**: before the first sample of a series; and after any gap (`epoch-changed`,
  `since-too-old`, `overflow`) until the next `probe.created` or sample of the node, because the
  consumer cannot say what happened in the hole. After `epoch-changed` **every** series of that
  origin is unknown: the plane restarted with an empty tree (fixture 3; the tree is not persisted,
  decision D1 ix#399), and the new boot's `seq` restarts at 1 while the old boot's series end.
- **ERROR**: the plane publishes no error event on a pipe; a kind that fails a read counts it in
  `errors` on `probe.list` and emits nothing for that tick. ERROR is therefore derived from
  `probe.list` polled, not from the stream (gap G3 below).

## Fixtures (`go/ixctl/e2e/fixtures/consumer-*.jsonl`)

One frame per line, `{"step": "<step>", "env": {...}}` or `{"step": "<step>", "gap": {...}}`, written
by the real plane from `TestAConsumerSeesReconnectRecreationAndRestartInItsFixtures`
(`go/ixctl/consumer_fixture_test.go`) with `IXCTL_WRITE_FIXTURES=1`; on every other run the test
checks that the plane still produces, step by step, the same **shapes** (channel, event kind, node,
source, or gap reason) the files hold, and asserts the invariants below on the fresh run. Counts
and values (`seq`, `t`, `epoch`) vary with timing and are not compared.

1. **`consumer-reconnect.jsonl`**, steps `live` then `after-reconnect`. A reader of everything
   (`ch` prefix `""`) sees `probe.created`, `probe.started`, then samples; it drops its connection
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
   `oldest: 1`, the new `epoch`), then the new boot from `seq` 1: `probe.created`, `probe.started`,
   samples with `ev.data.seq` from 1. Invariants: the gap names the new boot; every envelope after
   it is of the new boot; the new boot's envelope `seq` 1 is below the old boot's last `seq`, so a
   consumer keyed on `seq` without `epoch` would merge the two boots.

## Gaps (what the plane does not say today)

- **G1, no node instance id.** A recreated node has the same `node` and `kind`; only the
  `probe.created` that opened it tells the series apart (fixture 2). A consumer must read `ctl`
  beside the pipe, or the plane would have to stamp a series id on samples (a story under ix#397,
  not cut).
- **G2, the interval is not on the sample.** `every_ms` comes from `probe.list`; a reader that
  joined late reads it there.
- **G3, no error event on a pipe.** Failed reads are a counter on `probe.list` (`errors`); ERROR
  is polled, not streamed.
- **G4, no schema version on the envelope.** The protocol name `ixctl.v1` and this contract's
  version stand in; a field would be a protocol change (door F1, one-way).
- **G5, the tree is not persisted across a restart** (D1 ix#399): after `epoch-changed` the
  consumer waits for the tree to be recreated by whoever owns it.

## Changes

- v0.1.0 (2026-10-21): first draft, from Gaia's design review of PR #394 (reconnect/restart and
  same-name recreation fixtures, source identity, visible staleness, no dedup collisions) and the
  operator's go. Nothing locked; a locked field set comes with Gaia's intake (Phase 4).
