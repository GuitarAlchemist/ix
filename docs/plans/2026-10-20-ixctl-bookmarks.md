---
title: "ixctl bookmarks: a named point in the instance's sequence, replayed from the ring (story S8)"
type: decision
status: ring-backed slice merged into exp/ixctl; surface provisional; D9 resolved provisionally (operator sign-off pending)
date: 2026-10-20
branch: exp/ixctl (PR #394)
reversibility: two-way today (the bookmark file is read by nothing but the plane that wrote it; the ops sit behind door 1's provisional protocol); becomes one-way when a program outside the plane keeps bookmark names, or when the journal becomes a replay source (door 3)
revisit-trigger: D9 (ix#414) is signed off differently; a replay is wanted past the ring (seconds at probe rates); Gaia or the Rust client replays from a bookmark (S4, S7)
---

# ixctl bookmarks: a named point in the instance's sequence, replayed from the ring

French: [`docs/fr/plans/2026-10-20-ixctl-bookmarks.md`](../fr/plans/2026-10-20-ixctl-bookmarks.md).
Tracking: epic [ix#397](https://github.com/GuitarAlchemist/ix/issues/397), decision D9
[ix#414](https://github.com/GuitarAlchemist/ix/issues/414), story S8
[ix#415](https://github.com/GuitarAlchemist/ix/issues/415). User reference:
[`go/ixctl/README.md`](../../go/ixctl/README.md#bookmarks-and-replay). Probes, which this
composes with: [`2026-10-20-ixctl-probes.md`](2026-10-20-ixctl-probes.md).

## The problem

The operator wants to **troubleshoot after the fact**: "les états des pipelines devraient
supporter les bookmarks et replays". A probe tree feeds a pipeline; something looks wrong; the
operator wants to mark that moment and later replay what the pipes and the pipeline carried
around it, in order. Today a reader resumes from `since` + `epoch` within the hub's ring, and a
seq is a number nobody names or keeps; the journal holds `ctl` only and nothing replays it
(door 3).

## Design (as built)

- **A bookmark names one point in the instance's sequence**: `{name, epoch, seq, t, note}`. The
  hub's `seq` is global to the instance, shared by every channel, so one bookmark is enough to
  cover a whole tree's pipes, the services' pipes and the `ctl` lifecycle at one instant. The
  alternatives considered: a position on one channel (the operator would need one per pipe, and
  they would not line up), a set of positions (the same thing, heavier), a time (`t` is kept on
  the bookmark for the human; a replay is by seq, which is exact).
- **"Here" is the head.** `bookmark.set` without `since` takes the hub's head; with `since` the
  point must be at or below the head and, when `epoch` is given, of this boot. A replay from a
  bookmark is a subscription from `since = seq` (everything after the point), through the same
  `Hub.Subscribe` every reader uses; nothing new in the hub.
- **The plane owns them.** `ixctl up --bookmarks disk` (the default) keeps them in
  `<name>.bookmarks.json` beside the journal (`{"schema": "ixctl-bookmarks/0", "bookmarks": [...]}`),
  rewritten whole on every change: a temporary file, then a rename, mode 0600, so a crash
  leaves the old file or the new one. `--bookmarks memory` keeps them for the boot only (the
  operator asked for both stores, 2026-10-20). A file the plane cannot read as bookmarks fails
  the startup with the file's name rather than being emptied. The store is a choice at startup,
  not in the spec: it says nothing about the declared world, like `--state`.
- **The ring is the only replay source** (slice 1). A bookmark older than the ring answers the
  gap the hub already produces (`since-too-old`, with the oldest seq kept), then what the ring
  holds: never a silently partial replay. A bookmark of an earlier boot answers `epoch-changed`,
  then the new boot's ring. Replay from a journal past the ring is the second slice, gated on D9;
  door 3 is untouched by this one.
- **Transports stay thin.** `subscribe` (ixctl.v1) and `GET /v1/events` accept `bookmark` in
  place of `since`/`epoch`; both together are refused ("bookmark or since"). The transport
  resolves the name through `Broker.Request(bookmark.get)`, so the `Broker` interface is
  unchanged and a transport never reaches into the plane.
- **Bounds the system reports**: `max_bookmarks` 64 per plane, the name under the instance-name
  rule, `max_note_len` 256, `ring` 4096; `bookmark.list` reports them with `count` and `store`;
  every refusal names what it hit. Setting an existing name moves its point (its note kept
  unless a new one is given): a bookmark is a cursor the operator drags, not a log.
- **Forwardable like the probe ops**: `bookmark.*` with `peer` runs on the peer, in the peer's
  sequence, in the peer's store; a replay from it is done on that peer (subscribe is local).

## Decisions

### Taken in this slice (D9, provisional)

1. Shape: a point in the instance sequence, covering every channel. 2. Owner: the plane, file
`<name>.bookmarks.json` or memory, chosen at startup. 3. Replay source: the ring; the journal is
not reopened. 4. Bounds: 64 bookmarks, note 256, instance-name rule. 5. Door 3 stays as signed
(audit log, never replayed). Operator sign-off on D9 pending; recorded on ix#414.

### Left for the second slice

A per-channel journal the plane writes on request, replayed past the ring, with the rotation
bounds the prototype's JSONL had; and whether a bookmark should publish a `ctl` event of its
own (so the journal shows that someone marked a moment).

## Measured (Windows, 2026-10-20, gate E16 on a loaded desktop)

A `mux@1` over a `heartbeat@1` at 20 ms with a filter, started after `before` and torn down
after `after` set one second later. The live reader on `probe/` saw 48 envelopes between the two
bookmarks; the replay from `before` gave back the same 48, seq for seq, no gap. The replay of
everything from `before` gave 56 envelopes on `ctl`, `probe/agg3` and `probe/hb3` in one
strictly increasing sequence, the first two `probe.created` for the mux and the leaf. A bookmark
at seq 1 after the ring had turned over answered `since-too-old` first (oldest kept 3 600+), then
the ring. A bookmark set through plane A on plane B was in B's list and not in A's, and after B's
restart it was still listed, `live: false`, and its replay began with `epoch-changed` naming the
new boot.

## Tests (the seams agreed for TDD)

Unit, `go/ixctl/bookmark_test.go`, through `Plane.Request`, the WebSocket and SSE: a bookmark
set "here" reports the head; the replay of a leaf's pipe from it is seq for seq what a live
reader saw; the replay of everything from it is the lifecycle and both pipes after the point, in
order; the SSE form; the list with its bounds and `live`; every refusal naming its cause (unknown
name, bookmark beside since, name rule, since past the head, foreign epoch, note length,
`max_bookmarks`); a set on an existing name moves it; persistence across a restart of the same
state dir with the `epoch-changed` gap; a bookmark older than the ring with `since-too-old`; the
file holding no token; the memory store writing no file and keeping nothing across a restart; a
store name that is neither refused; forwarding to a peer over HTTP. Gate: E16, three checks in
`go/ixctl/e2e/e2e.py`, 42 checks in all.

## Left out on purpose

Replay from a journal past the ring (D9, second slice); a `bookmark` event on `ctl`; the Rust
and IXQL surface (`ix.ctl.subscribe(bookmark:)`, S4); bookmarks on a mirrored peer channel
(S2); the CLI (`ixctl bookmark`, nothing a `curl` does not do today).

Once D9 is signed these decisions belong in an ADR under `docs/adr/`.
