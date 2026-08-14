# IXQL ↔ DuckDB+IX integration goes through the MCP seam, not new IXQL grammar nodes

Status: accepted (2026-06-20), implementation status corrected (2026-08-14)

## Decision

IXQL (Demerzel's governance pipeline-spec DSL) and DuckDB+IX (`ix-duck`, the analyst bench)
are **complementary layers** — specification vs. analysis — joined only by a JSON-on-disk
format contract, by design (`docs/DUCKDB.md`). When we want them to interoperate, we connect
through **the MCP tool seam and the shared file format**, *not* by adding DuckDB/Parquet
source or sink nodes to the IXQL grammar. The governance binding/record dialect now has an
executor in `ix-ixql`, but the ML-pipeline dialect remains a separate surface. A grammar change
still requires explicit Galactic-Protocol sign-off.

Concretely, the three integration opportunities resolve to:
1. **DuckDB as an IXQL data source** → expose an ix-duck query as an MCP tool; IXQL reaches it
   via its existing `mcp_tool_output("…")` / `database("…")` productions. No grammar change.
2. **Governance verdict as a callable** → expose the already-built `maintain-gate`
   (`ix-duck::maintain`, the hexavalent T/P/U/D/F/C RSI oracle — *already* "DuckDB as referee")
   as an MCP tool; a capability-verified IXQL program may gate on its typed result.
3. **IXQL → trend table** → IXQL writes JSONL with `ix.io.write()` (works today); `ix-duck`
   reads it with `read_json_auto`. Formalize the schema; no grammar change.

## Why (the trade-off)

- **The shipped executor is deliberately narrower than the full grammar corpus.** `ix-ixql`
  executes the governance binding/record dialect; the ML-pipeline dialect (`csv(…) → train(…)`)
  remains a separate surface. Adding a DuckDB grammar node would widen and couple both layers
  instead of using the existing executable adapter seam.
- **The grammar is a governed artifact.** Galactic-Protocol/grammar changes require explicit
  sign-off (CLAUDE.md "one-way doors"); the MCP seam and the file-format contract do not.
- **The MCP seam is runnable today and forward-compatible.** MCP tools are useful to agents
  immediately and remain IXQL's integration point without importing DuckDB into the compiler.
  The shared substrate exists already: IXQL data
  sources already `read_json_auto()` the same `ix/state/**/*.jsonl` files `ix-duck` reads.

## Consequences

- Exposing the maintain-gate as an MCP tool (`ix_maintain_gate`) makes `ix-agent` depend on
  `ix-duck`'s **bundled-DuckDB (`duck`) feature** — a heavy native (C++) dependency. It is
  therefore **feature-gated** (off by default) so the default agent build and `--workspace`
  CI never compile DuckDB. The duck-feature CI job (`ix-duck-chatbot.yml`) is where it should
  be exercised.
- The maintain-gate's verdict is **advisory until its ledger write-isolation (Phase-3b) lands**
  (`docs/contracts/maintain-gate.contract.md`): the proposing agent must not be able to write
  the ledger it is judged against. Until then the MCP tool reports a verdict but it is not a
  *binding* governance gate.

## Revisit trigger

The executor portion of the original trigger has fired. A first-class `duckdb(…)` source /
`write_parquet(…)` sink is now technically runnable, but remains deferred pending separate
grammar sign-off and evidence that the MCP/file seam is insufficient.
