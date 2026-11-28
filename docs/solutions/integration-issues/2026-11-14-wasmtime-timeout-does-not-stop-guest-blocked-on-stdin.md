---
title: "wasmtime run -W timeout does not stop a guest that is blocked reading stdin"
category: integration-issues
date: 2026-11-14
tags: [wasmtime, wasi, wasip1, wasip3, epoch-interruption, timeout, cancellation, xlet, sandbox]
symptom: "sleep 2 | wasmtime run -W timeout=300ms guest.wasm ends after ~2000 ms with `wasm trap: interrupt`, not after 300 ms; a guest waiting on a timer is interrupted on time"
root_cause: "-W timeout is epoch interruption: the trap fires when the guest next executes wasm code, so time spent blocked in a host call (a stdin read) is not cut short"
---

# wasmtime `-W timeout` does not stop a guest blocked on stdin

## Problem

While checking whether Wasmtime 50.0.1 can bound an untrusted Xlet's run time (2026-11-03,
scratch investigation, nothing committed), `-W timeout` gave two different behaviours:

| Guest | Limit | Ended after | Result |
|---|---|---|---|
| WASI 0.3.0 Rust guest, two async `wait-for` timers (300 + 100 ms, sequential) | 150 ms | ~170 ms | `wasm trap: interrupt`, rc 134 |
| WASI 0.3.0 echo guest, `sleep 1 \| ...` on stdin | 300 ms | ~1006 ms | `wasm trap: interrupt`, rc 134 |
| WASI 0.3.0 async stdin guest, `sleep 2 \| ...` | 300 ms | ~2006 ms | `wasm trap: interrupt`, rc 134 |
| Go `wasip1` validator, `sleep 1 \| ...` | 300 ms | ~1010 ms | one run trapped (rc 134); in the other the guest got EOF first and exited 2 |

So the deadline is enforced while the guest waits on a timer, but not while it waits on the
host for input. The outcome of the last row is a race between the interrupt and the guest's own
exit.

## Root cause

`-W timeout` is implemented with epoch interruption ("Maximum execution time of wasm code" in
`wasmtime run -W help`). The epoch check runs in wasm code, so a guest parked inside a blocking
host call does not see it until the call returns. Timer waits in the async (p3) guest return to
the guest's event loop, where the check fires.

## Fix

No code change; it is a property of the runtime. For an Xlet or validator run that may wait
on input:

- Put a wall-clock limit around the process (`timeout 5s wasmtime run ...`) as well as
  `-W timeout`.
- Better, hand the guest its whole input at start, as ix-wasm does (stdin is an in-memory
  buffer, so the guest never blocks on a pipe), and bound computation with fuel.

## How to find it again

```bash
time (sleep 2 | wasmtime run -W timeout=300ms any-guest-that-reads-stdin.wasm)
```

A wall time near 2 s rather than 0.3 s shows the deadline did not apply while blocked.
