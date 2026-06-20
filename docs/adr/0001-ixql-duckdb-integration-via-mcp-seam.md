# IXQL ↔ DuckDB+IX integration goes through the MCP seam, not new IXQL grammar nodes

Status: accepted (2026-06-20)

## Decision

IXQL (Demerzel's governance pipeline-spec DSL) and DuckDB+IX (`ix-duck`, the analyst bench)
are **complementary layers** — specification vs. analysis — joined only by a JSON-on-disk
format contract, by design (`docs/DUCKDB.md`). When we want them to interoperate, we connect
through **the MCP tool seam and the shared file format**, *not* by adding DuckDB/Parquet
source or sink nodes to the IXQL grammar — until the IXQL executor ships **and** a grammar
change gets explicit Galactic-Protocol sign-off.

Concretely, the three integration opportunities resolve to:
1. **DuckDB as an IXQL data source** → expose an ix-duck query as an MCP tool; IXQL reaches it
   via its existing `mcp_tool_output("…")` / `database("…")` productions. No grammar change.
2. **Governance verdict as a callable** → expose the already-built `maintain-gate`
   (`ix-duck::maintain`, the hexavalent T/P/U/D/F/C RSI oracle — *already* "DuckDB as referee")
   as an MCP tool; IXQL gates on it with `→ when verdict == "T"` once the executor lands.
3. **IXQL → trend table** → IXQL writes JSONL with `ix.io.write()` (works today); `ix-duck`
   reads it with `read_json_auto`. Formalize the schema; no grammar change.

## Why (the trade-off)

- **The IXQL executor is spec-only.** `ix-cli` (#103) and the tars CE are both "Draft" and
  absent from the workspace; only the tree-sitter grammar + a Node validator exist. A new
  grammar source/sink node would be syntax **nothing can execute** — premature.
- **The grammar is a governed artifact.** Galactic-Protocol/grammar changes require explicit
  sign-off (CLAUDE.md "one-way doors"); the MCP seam and the file-format contract do not.
- **The MCP seam is runnable today and forward-compatible.** MCP tools are useful to agents
  immediately and become the IXQL integration point for free when the executor ships — IXQL's
  `mcp_tool_output` already targets them. The shared substrate even exists already: IXQL data
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

When the IXQL executor (`ix-cli` #103) actually ships — at that point a first-class
`duckdb(…)` source / `write_parquet(…)` sink may be worth the grammar change (with sign-off),
because it would then be runnable and could validate schemas at pipeline-author time.
