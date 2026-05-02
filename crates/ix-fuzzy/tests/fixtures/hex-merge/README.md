# Hex-merge conformance fixtures (IX-side mirror)

A test-local copy of `hari/fixtures/hex-merge/` for verifying
`ix_fuzzy::observations::merge` against the same corpus the Hari-side
test runs against. When both this test and Hari's pass, the two
implementations of Demerzel's `governance/demerzel/logic/hex-merge.md`
agree on this corpus by transitivity.

## Canonical home (open question)

The fixture corpus should ideally live in **Demerzel itself** —
`Demerzel/fixtures/hex-merge/` or `Demerzel/examples/` — so both
consumer repos read from one source instead of maintaining two
copies. That requires a Demerzel PR; until it lands, this is a
mirror of the Hari version and they must be kept in sync by hand.

## Schema

See `hari/fixtures/hex-merge/README.md` for the JSON schema and the
list of scenarios. Variants use the canonical single-letter wire
format (`T`/`P`/`U`/`D`/`F`/`C`) — same as Demerzel's
`hexavalent-state.schema.json` and `ix-types::Hexavalent`'s built-in
serde rename. Floats are compared within `1e-9`.

## Running

```
cargo test -p ix-fuzzy --test hex_merge_conformance
```

## Adding a fixture

1. Add the new file to `hari/fixtures/hex-merge/` and verify it
   passes `cargo test -p hari-lattice --test hex_merge_conformance`.
2. Copy it here.
3. Run `cargo test -p ix-fuzzy --test hex_merge_conformance`.

Both runners pick up new files automatically; no test code changes
required.
