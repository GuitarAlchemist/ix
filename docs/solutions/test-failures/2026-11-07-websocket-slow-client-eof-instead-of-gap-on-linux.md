---
title: "A slow WebSocket client gets EOF instead of the gap frame on Linux: the write timeout closed the connection"
category: test-failures
date: 2026-11-07
tags: [go, ixmux, websocket, coder-websocket, linux, ci, loopback, overflow, timeout]
symptom: "go-modules (ubuntu-latest) red: 'slow client: failed to read: failed to read frame payload: unexpected EOF before any gap' in TestBounds; green on Windows"
root_cause: "In coder/websocket a Write whose context expires closes the whole connection. The server writes each frame under WriteTimeout (10 s); the test read its slow client only after the rest of the run, and with Linux's large loopback buffers and -race the writer blocked later and stayed blocked past 10 s, so the connection was closed before the queue overflowed and the gap was sent"
---

# A slow WebSocket client sees EOF, not a gap, on Linux

## Problem

`TestBounds` (ixmux integration) gave the server `--queue 4`, opened a client that read nothing,
flooded the journal, and expected that client to read a `gap` of reason `slow` and a 1008 close.
Windows green; the Linux CI row read `unexpected EOF before any gap`.

## Root cause

Two things stack:

1. The ixctl trap (see `2026-10-22-go-suite-green-on-windows-red-on-linux-loopback.md`): Linux
   loopback buffers absorb the flood, so the writer blocks much later than on Windows.
2. The ixmux-specific part: the writer calls `conn.Write(ctx, ...)` with a 10 s `WriteTimeout`.
   In `github.com/coder/websocket`, a write whose context expires **closes the connection**; it
   does not just fail the one write. The test read the slow client only after the fast client and
   the delivery were done, so once the writer did block, it stayed blocked over 10 s and the
   library cut the connection before any gap.

There is no server defect: a client that does not read cannot receive the gap either; closing
without a gap and resuming with `since` is the only possible behaviour (now in the README).

## Fix

Commit b343fcee:

- A test hook `server.testHookStreamLive`, called in the connection's writer before each wait on
  its queue. Holding it holds the writer before the socket, so the queue overflows on every kernel.
  `server/s5_test.go:TestSlowClientGetsGapThenClose` uses it with a queue of 2 and checks hello,
  at most 2 queued frames, the `slow` gap, the 1008 close, and that a delivery does not wait.
- `TestBounds` no longer has a slow client, and runs with `--queue 256`, above the 128 pages its
  flood produces.

The rule from the ixctl note still holds: a test whose outcome depends on socket buffers sets a
knob that makes it certain.

## How to find it again

```
# a Linux kernel is needed; the Windows run is green
go test -count=1 -run TestBounds ./cmd/ixmux/
```

An EOF where a gap frame was expected, on Linux only, with a writer under a write deadline, is
this note.
