# ADR-0006: `ix-io`'s protocol is a synchronous pull interface; async backends adapt through `BatchSource`

Status: proposed (2026-09-08) — Refs [ix#299](https://github.com/GuitarAlchemist/ix/issues/299)

## Context

`crates/ix-io/src/protocol.rs` opened with a factual claim:

> Every I/O backend implements DataSource and/or DataSink, giving a uniform
> interface for the skill layer.

It was false. `ix doctor`'s `orphan-traits` check — which exists because of this
case — recorded both traits in `state/registry/orphan-traits.allow.json` with
zero implementors, zero generic bounds, and zero mentions outside their own
declaration. Eight backends shipped in the crate and none conformed.

ix#299 offered three dispositions: implement, delete, or re-label as
aspirational. The capability the traits describe is wanted — fetching remote
data so IX algorithms can run over it — so *implement* was chosen. That forces
a design question the issue did not answer: **the eight backends have genuinely
different shapes, and a trait that fits none of them is worse than no trait.**

The shapes, as they actually are:

| backend | unit it produces | blocking or async |
| --- | --- | --- |
| `csv_io` | numeric rows from a `Read` | blocking |
| `json_io` | numeric rows (NDJSON) or a whole-document batch | blocking |
| `http` | a parsed body | async |
| `tcp` | JSON lines off a socket | async |
| `websocket` | an unbounded message stream | async |
| `pipe` | unframed `Vec<u8>` | blocking |
| `watcher` | `FileEvent` — a path and a kind, carrying no data | blocking |
| `trace_bridge` | `Trace` — a nested GA domain document | blocking |

## Decision

**1. The trait signatures do not change.** `DataSource::read_batch` /
`has_more` and `DataSink::write_batch` / `flush` keep the shapes they were
declared with. They fit a synchronous, pull-based record stream, which is what
the blocking backends are.

`has_more`'s contract is now *stated* rather than assumed: `false` is
definitive, `true` means "not known to be exhausted". A pull reader cannot
promise more without look-ahead it does not have. `pump` and `drain` are
written against that weaker guarantee and terminate on an empty read.

**2. `pump` and `drain` are added as generic consumers.** Implementors alone
would have satisfied the doctor check while leaving the "uniform interface"
claim just as empty — nothing would have been *written against* the traits.
`pump<S: DataSource, K: DataSink>` moving records between two types that have
never heard of each other is what makes the pair a seam.

**3. Async backends adapt; they do not implement.** `read_batch` is
synchronous, and blocking on a future inside `&mut self` panics within the
Tokio runtime every caller of `http`/`tcp`/`websocket` is already in. Those
backends *acquire* asynchronously and *serve* synchronously: `await` a
`DataBatch`, wrap it in `protocol::BatchSource`, which does implement
`DataSource`.

**A parallel `AsyncDataSource` trait was considered and rejected.** MSRV 1.80
permits `async fn` in traits, so it was possible. It would have declared a
second contract with no generic consumer — the exact defect ix#299 is about.
The fix for a shape mismatch is a conversion, not a second hierarchy.

**4. Three backends implement neither, with the reason in their module doc.**
`pipe` has no framing and no observable EOF, so `has_more` has no honest
answer. `watcher` events contain no data at all — it is a trigger that tells a
caller when to open a `CsvSource`. `trace_bridge`'s unit is a domain document,
and flattening it to a numeric row is a modelling choice belonging to the
analysis, not the loader.

**5. `http` is the crate's network-egress boundary and is bounded there.**
Every entry point runs through a scheme refusal (`http`/`https` only) and a
capped, timed read. `reqwest` supplies neither: `Response::text()` buffers an
entire body with no limit, and `reqwest::get` applies no timeout. Following
ix#286, the ceiling is stated and refused loudly rather than inherited
silently. Two independent guards: a `Content-Length` above the ceiling is
refused before any body byte is read, and the running byte count is checked
while streaming, so an undeclared or understated length is still bounded.

## Reversibility

**Two-way door**, with one narrower hinge.

- Adding `BatchSource`, `pump`, `drain`, the four concrete source/sink types
  and the `IoError::Limit` variant is additive. `ix-io` is `beta` in
  `crate-maturity.toml`, so the stable-surface gate warns rather than blocks,
  and nothing outside the crate consumes these names yet.
- **The hinge:** decision 1 fixes `has_more` in the public API. Dropping it in
  favour of an EOF-signalling `read_batch` is a major-version change and gets
  harder with every implementor. It is cheap to revisit *now* and should be
  batched with any other `ix-io` break rather than taken alone.
- Decision 5's numbers (8 MiB, 30 s) are constants, freely tunable.

## Revisit triggers

- **A second async transport wants the protocol generically.** Today the
  adaptation is one `BatchSource` call per backend. If a caller ever needs to
  be generic *over* async sources, re-open decision 3 — that is when an
  `AsyncDataSource` would have a consumer, which is the only thing that would
  justify it.
- **Any MCP tool or DuckDB UDF exposes `ix-io::http`.** The bounds shipped here
  are size, time, and scheme. There is **no SSRF guard**: `127.0.0.1`,
  `169.254.169.254` and RFC1918 addresses are dialled like any other, and
  redirects follow `reqwest`'s default policy. That is acceptable while callers
  are in-process Rust and unacceptable the moment a URL becomes
  attacker-influenced. Resolve it *before* adding that exposure.
- **`DataRecord::Named` / `Text` / `Bytes` gain a real producer.** The sinks
  currently skip non-`Row` records, inherited from `write_batch_csv`. If those
  variants start carrying data anyone reads, the skip becomes silent loss.
