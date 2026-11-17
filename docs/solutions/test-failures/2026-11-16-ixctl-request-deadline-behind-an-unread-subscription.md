---
title: "An ixctl test's request times out at 10 s on some CI runs, both OSes, never on main"
category: test-failures
date: 2026-11-16
tags: [go, ixctl, client, subscription, head-of-line, ci-flake, windows, linux]
symptom: "--- FAIL: TestABookmarkReplaysEveryChannelFromOnePoint (11.4s) bookmark_test.go:192: bookmark.set: context deadline exceeded"
root_cause: "The Client's read loop waits for room in a subscription's buffer before it reads the next frame. The test left two replay subscriptions (buffer 256) unread while a 10 ms heartbeat tree published about 200 frames a second, so a buffer filled after about 1.3 s and every reply behind it on the same connection stalled until the request's 10 s deadline."
---

# A request deadline behind an unread subscription

## Problem

`TestABookmarkReplaysEveryChannelFromOnePoint` failed six times in two days (twice on
ubuntu, four times on windows), on four branches, and never on main's own runs. Every failure
took 11.4 s: about 1.4 s of test, then one `bookmark.set` waiting its full 10 s. A rerun of the
failed job went green, so the first two failures were taken for a slow runner. Thirty local
runs did not reproduce it. The constant 11.4 s was the clue: a slow runner gives scattered
durations, while a fixed wait gives a fixed one.

## Root cause

`Client.readLoop` (go/ixctl/client.go) reads frames in order. It hands a reply to its waiting
request and a subscription frame to that subscription's channel, and it blocks on that channel
when its buffer is full. One unread subscription therefore stalls every reply that comes after
it on the connection. The test opened `SubscribeBookmark(..., 256)` twice, read the first 5
and 10 frames, then left both while the heartbeat (`every_ms` 10) and its mux kept publishing.
On a runner that took longer than about 1.3 s to reach the loop of 62 `bookmark.set` calls, a
buffer filled, and the next reply never reached the test.

The same test also compared `bookmark.list`'s `head` with a second read of the hub's head. A
publication between the two reads failed it once in 30 local runs.

## Fix

The test stops the tree once the replays' material exists, so nothing publishes after that.
It also gives the replays the same 4096-frame buffer as the `subscribe` helper. A 2 s pause
inserted before the loop made the old test fail 3 times out of 3 with the CI message. With the
fix it passes, with that pause and in 30 runs in a row.

## How to find it again

Put `time.Sleep(2 * time.Second)` before the `bookmark.set` loop of the old test and run
`go test -count=3 -run 'TestABookmarkReplaysEveryChannelFromOnePoint$' .` in `go/ixctl`: it
fails each time in about 12 s. More generally, when an ixctl test times out on a request:

- look for a subscription opened on the same `Client` and no longer read while something
  publishes;
- drain it, unsubscribe it, or stop what publishes.

When CI failures share one exact duration, count them across branches before calling them
transient.
