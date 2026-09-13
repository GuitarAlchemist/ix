# Gaia M2 / IX v0a census tracer — implementation record

**Date:** 2026-08-15
**Status:** experimental local branch (`codex/gaia-m2-v0a-tracer`), uncommitted. Not published, not frozen.
**Authority:** none. This document records an implementation, not a milestone. **M2 is not established until independent review. Gaia remains NOT INTEGRATED.**

## Scope

One new experimental leaf crate, `ix-gaia-census`, implementing the narrower M2
census of specification §8.1 over copied immutable evidence. The §9.1
contradiction projection is deliberately **not** implemented.

Public seam under test: `ix_gaia_census::census(&CensusRequest)`. No binary, no
CLI verb, no MCP tool, no `#[ix_skill]` registration, no IXQL grammar, no
database, no bus call, no routing, no model, no product read.

## Subject

The exact S1-approved evidence at
`C:\tmp\gaia-s1-r18-review-e8a7d092-20260815T173104Z` — 14 files, 663,567 bytes,
fourteen-file ordinal aggregate
`e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e`. Copied
byte-for-byte into `crates/ix-gaia-census/tests/fixtures/gaia-s1-r18/`, with
`* -text` applied from `tests/fixtures/.gitattributes` (one level above the
fixture, so the evidence root keeps exactly fourteen entries).

Expected values are declared by that evidence, never recomputed by test logic:

| Value | Source |
|---|---|
| `970bb50c…311c` | manifest §3, the thirteen-file declared fixed point |
| `3c53ad57…8d74`, `457e745b…8bdb`, `ea3b98f1…23e6` | manifest §3 discriminating (wrong) constructions |
| `e8a7d092…598e` | the S1 R18 review directory identity |
| 13 files / 639,580 bytes | manifest §2 |
| 23,567 bytes, `6e0f9f73…62a3`, 214 CRLF pairs | manifest §2 and §6, the one CRLF file |

## Design decisions

- **Seam.** `ix_types::Hexavalent` carries the evidence state. No second truth
  lattice, no new registry entry, no untyped `Value` socket.
- **Refuse before emit.** A declared file absent from the evidence root is
  `Err(CensusRefusal::BinderIncomplete)` and **no artifact at all**, per
  `spec-v0.2:359`. Digest drift is the opposite: a typed observed row plus a
  deterministic advisory artifact, never authority.
- **Control-count ambiguity.** The specification states its §11 cardinality
  three ways (eighteen, twenty-six, and an enumerated list of twenty-eight).
  The **enumerated list is taken as authoritative**, unioned with the fourteen
  enumerated doctrine §4 controls: 42 rows. Each is `Covered` with a named test
  or `NotApplicable` naming the subject the tracer does not have. No lease,
  fence, bus, spend, or acceptance was built to make an irrelevant control
  apply. This records the ambiguity; it does not repair it.
- **M3 boundary.** `schema_version = 1` and named-object serialization are the
  whole forward-compatibility story. No M3 field is present. Units are only
  unit-suffixed field names, not the declared per-field Axis Contract; absence
  is only Rust type discipline, not the declared null/unknown semantics. **No
  provenance contract and no out-of-sample baseline is implemented**, and
  §8.1's "compare against simple out-of-sample baselines" has no subject here
  because the tracer fits no model.

## Reversibility

**Two-way door.** Nothing is committed, pushed, published, or frozen. Rollback
is `rm -r crates/ix-gaia-census`, revert one `[workspace] members` line and one
`crate-maturity.toml` row, delete this file. No schema is frozen
(`schema_version = 1`, tier `experimental`), no index is rebuilt, no crate is
promoted, no CI workflow changes, no external system is touched.

**Revisit trigger:** an M3 authorization, or a fresh S1 subject that moves the
fixture's fixed point.

**If this is ever committed** it becomes partly one-way: 663,567 bytes of the
Gaia S1 subject would enter git history, and the crate name would enter the
public workspace member list. Both require explicit sign-off first, and the
Streeling catalog would have to be regenerated (`cargo run -p ix-streeling --
catalog`) because this file lives under `docs/plans/`.

## Verification

`cargo test -p ix-gaia-census` after every slice (22 tests green, also green
from a clean `target/`); `cargo clippy --workspace --all-targets -- -D warnings`
green at CI's exact flags. **`cargo test --workspace` did not complete cleanly**
— first the host disk hit 0 bytes free, then three `ix-agent` tests failed
because `governance/demerzel` is an unpopulated submodule in this worktree; and
**`scripts/verify.ps1` was not run**, since it re-enters the same blocked suite.
Both gaps are recorded in the writer handoff, §8.1 and §8.2, and no claim rests
on them. Determinism was replayed under
`--test-threads=1` and under two locale/timezone environments (`UTC`/`C` and
`Pacific/Kiritimati`/`tr_TR.UTF-8`).

Full red/green record, mutation probes, and disclosures: the writer handoff at
`C:\tmp\gaia-wayfinder-plus\gaia-m2-ix-v0a-writer-handoff.md`.
