---
title: "The ixctl Go suite passes on Windows and two overflow tests fail on Linux: the loopback buffers absorb the flood"
category: test-failures
date: 2026-10-22
tags: [ixctl, go, websocket, linux, windows, ci, flaky, loopback, overflow]
symptom: "ixctl-go (ubuntu-latest) red on PR #423 while the same tests pass locally on Windows: TestWSOverflowGivesGapFrame-style flood tests time out waiting for a gap frame, 'no gap frame; last seq N'"
root_cause: "The tests count on a session queue filling behind a WebSocket writer that a non-reading client blocks. Linux loopback socket buffers take the whole flood (thousands of frames), so the writer never blocks, the queue never fills and no overflow happens; Windows buffers are smaller and the same flood overflows in a few hundred frames"
---

# The Go suite is green on Windows and red on Linux: loopback buffers differ

## Problem

PR #423's first CI run (095c507) was red on `ixctl-go (ubuntu-latest)` only: the two peer-link
flood tests waited ten seconds for an `overflow` gap that never came. The same package passed on
Windows locally and in the `ixctl-go (windows-latest)` row. Nothing in the tests named a platform;
they published a flood at a subscription whose client read nothing and asserted the bounded
session queue (256 frames) overflowed.

## Root cause

The overflow needs the writer goroutine to block on `conn.Write` so frames pile up in the
session's queue. On Linux the kernel's loopback send and receive buffers accept the flood, so the
writer keeps draining the queue into the socket and the queue never reaches its bound. On Windows
the buffers fill after a few hundred frames and the writer blocks. The tests were asserting a
kernel's buffer size, not the plane's behaviour.

## Fix

Commit 7724956: a test knob on the server, `writeDelay atomic.Int64` (go/ixctl/http.go), slept by
the writer before each frame (go/ixctl/ws.go) and zero in production. The flood tests set it to
one millisecond, so the queue fills behind the writer on every kernel; 54f2a9f gave the last
kernel-dependent test (`TestWSOverflowGivesGapFrame`) the same knob, and the overflow now comes
after 257–259 events on every run. Both OS rows green from 7724956.

The rule that came out of it (docs/CHECKLISTS.md, behavioural-claim section, item 6): a test whose
outcome depends on the OS sets a knob that makes the outcome certain, and the Linux suite runs
locally before a push that touches go/ixctl: `pwsh scripts/ixctl-test-linux.ps1` (WSL, `go vet` +
`go test -race`, the CI invocation).

## How to find it again

```powershell
# the suite under a Linux kernel, from Windows, in about a minute
pwsh scripts/ixctl-test-linux.ps1 -Run 'Overflow'
# three times under -race, when a timing-bound test is suspected
pwsh scripts/ixctl-test-linux.ps1 -Count 3
```

A green Windows run and a red Linux row on a test that waits for a bounded queue to fill is this
note.
