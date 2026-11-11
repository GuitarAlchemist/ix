---
title: "A Go integration test that reads its child's output while the child runs is a DATA RACE under -race on Linux"
category: test-failures
date: 2026-11-07
tags: [go, ixmux, os-exec, race-detector, linux, ci, bytes-buffer]
symptom: "go-modules (ubuntu-latest, -race) red with WARNING: DATA RACE in TestSecurity: bytes.Buffer.String() read in the test goroutine while io.Copy -> Buffer.ReadFrom writes it; the Windows row (no -race) is green"
root_cause: "A bytes.Buffer given to exec.Cmd.Stdout/Stderr (not an *os.File) is filled by a goroutine os/exec starts; it is safe to read only after cmd.Wait returns. Reading it while the child runs (to wait for a 'listening' line, or in a failure message) races that goroutine"
---

# Reading a child's output buffer while the child runs races os/exec

## Problem

The ixmux integration tests start the server binary with `cmd.Stdout = &p.out; cmd.Stderr = &p.out`
(`p.out` a `bytes.Buffer`) and poll `p.out.String()` until the server prints its address, then
print the buffer in failure messages. Green on Windows, where the race detector does not run (no
cgo toolchain locally or on the Windows runner). The first Linux `-race` run of PR #515 reported a
DATA RACE in `TestSecurity`.

## Root cause

When `Stdout` is not an `*os.File`, `os/exec` creates a pipe and a goroutine that `io.Copy`s the
pipe into the writer. `cmd.Wait` waits for that goroutine; until it returns, the buffer is being
written concurrently. `bytes.Buffer` has no lock, so any read before `Wait` is a race, whatever
the OS. Only `-race` sees it, and only the Linux CI row has `-race`.

## Fix

Commit b343fcee: `proc.out` is a `syncBuffer`, a `bytes.Buffer` whose `Write` and `String` take
one mutex, so every read is ordered after the writes it sees. A buffer read only after `Wait`
(`deliver/kill_test.go`) needs nothing: `Wait` already orders the copy before the read.

## How to find it again

```
# on a machine with cgo (Linux CI row, or any Linux box with gcc)
go test -race -count=1 -run TestSecurity ./cmd/ixmux/
# the pattern, in any Go test: a Buffer/Builder given to Stdout or Stderr and read before Wait
rg -n 'Std(out|err) *= *&' --glob '*_test.go'
```

`scripts/ixctl-test-linux.ps1` runs `-race` under WSL for `go/ixctl` only; it does not cover
`go/ixmux`.
