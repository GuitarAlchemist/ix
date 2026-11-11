---
title: "A no-pid-leak check is red on Linux only: the pid 2026 is in every timestamp of the year"
category: test-failures
date: 2026-11-10
tags: [ixmux, inventory, mcp, leak-check, linux, wsl, pid, flaky]
symptom: "TestInventoryAcrossAKillDuringATurn passes on Windows and fails under WSL2 with `ixmux_inventory carries \"2026\"`, though the answer has no pid field"
root_cause: "the test looked for strconv.Itoa(pid) as a substring of the JSON answer. Linux hands out small pids (here 2026), which appear inside `read_at`/`last_activity` timestamps; Windows pids (tens of thousands) never collided"
---

# A no-pid-leak check is red on Linux only

## Problem

`ixmux_inventory` must not carry a pid. The end-to-end test checked the raw
JSON answer with `strings.Contains(text, strconv.Itoa(pid))`. It was green on
Windows and red under WSL2 (`go test -race`): the ixmux's pid was 2026, and
`"read_at":"2026-11-11T…"` contains it. No pid had leaked.

A short Linux pid also turns up inside hex epochs and uuids
(`epoch:e1aaed50feb1` contains `50`).

## Fix

Match the pid as a number of its own:

```go
regexp.MustCompile(`(^|[^0-9a-fA-F.-])` + strconv.Itoa(pid) + `($|[^0-9a-fA-F.:-])`)
```

- Not next to a hex digit, a dot or a dash: timestamps, epochs, uuids.
- Not followed by a colon: a short pid like `51` in `02:51:31`.
- A colon *before* is allowed, so a JSON value `"x":2026` is still caught.

Check the regex both ways before trusting it: on the red text, on leaking
shapes (`:2026,`, `[2026]`, `"2026"`, prose), and for a long and a short pid.
The first version excluded `:` on both sides and missed `"x":2026`, the most
likely leak. Then kill a mutant that really leaks (the pid copied into a text
field) so the check is known to bite.

## Rule

A substring search for a number in serialised output is a Linux-only flake
waiting to happen. Search for the number as a token, or decode the JSON and
compare values.

Source: `go/ixmux/cmd/ixmux/inventory_test.go` (`mcpInv`), commit `03d916b0`.
