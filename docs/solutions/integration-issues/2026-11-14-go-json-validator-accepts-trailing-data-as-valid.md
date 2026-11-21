---
title: "A Go JSON validator reports valid:true for input with a second value or garbage after the object"
category: integration-issues
date: 2026-11-14
tags: [go, encoding-json, wasip1, ix-wasm, ixql, validator, stdin-contract, silent-acceptance]
symptom: "printf '{...}garbage' | wasmtime run validator.p1.wasm exits 0 with {\"valid\":true,\"errors\":[]}; `{...} {\"x\":1}` is also accepted"
root_cause: "json.NewDecoder(r).Decode(&v) reads exactly one value and stops; nothing checks that the rest of stdin is empty, so only the first value is judged"
---

# A Go JSON validator accepts trailing data as valid

## Problem

The fixture validator of `prototypes/ixql-wasi-validator` (a Go program built `GOOS=wasip1`,
run by ix-wasm as an IXQL capability) promises "one JSON value in, one report out". Fed
`{"schema_version":1,"target":{"repo":"r"},"verdict":"pass"}garbage`, or the same object
followed by ` {"x":1}`, it printed `{"valid":true,"errors":[]}` and exited 0, under Wasmtime 50.0.1
and in `go test`.

Through the IXQL path itself the bug was not reachable: ix-wasm's `wasm.run` and the prototype's
`WasiValidator` serialise one value with `serde_json` and hand it over as an in-memory stdin.
It bites when the same module reads stdin from anything else (a pipe, a file, another runtime,
a future caller that concatenates inputs), where a verdict would be given on the first value only.

Nothing was red: the three end-to-end Rust tests, `run.sh` (24 assertions) and the WasmEdge
comparison all fed single-value inputs. The gap showed only when someone read `main.go`
(an `@claude` review on PR #508) and a test was written for it.

## Root cause

`json.NewDecoder` is a stream decoder. `Decode` consumes one value and leaves the reader
positioned after it. `json.Unmarshal` would have rejected the trailing bytes, but a program that
reads stdin with a `Decoder` has to check the end itself.

## Fix

Commit `9a18f6b` on `claude/gifted-archimedes-8o9lxy`: after the first `Decode`, a second
`dec.Decode(&struct{}{})` must return `io.EOF`; anything else exits 2 with
`not JSON: trailing data after the first value`. Trailing whitespace still decodes to `io.EOF`,
so fixtures ending in a newline pass.

Bound by `TestRejectedInput` (three trailing-data cases, red before the fix) and
`TestTrailingWhitespaceAccepted` in `validator/main_test.go`. Two source mutants that each
weaken the check (`err != io.EOF && err == nil`, `&& err != nil`) turn the suite red.

The prototype is not on `main`: its PR (#508) was closed unmerged, and the code and tests live
only on the branch `claude/gifted-archimedes-8o9lxy`. The fix itself is the five lines above;
copy them into any Go program that reads one JSON value from stdin.

## How to find it again

```bash
printf '%s' '{"a":1}junk' | go run ./your-validator; echo "exit=$?"
```

An exit 0 means trailing data is ignored. In any Go program that reads stdin with
`json.NewDecoder`, grep for a `Decode` that is not followed by an `io.EOF` check.
