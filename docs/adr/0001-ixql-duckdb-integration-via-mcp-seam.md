# IXQL ↔ DuckDB+IX integration goes through the MCP seam, not new IXQL grammar nodes

Status: accepted (2026-06-20)

Runtime status updated: 2026-10-13. The original revisit trigger — the IXQL executor
shipping — has fired for the governance dialect: `crates/ix-ixql` parses and runs it. The
separate ML-pipeline dialect is not parsed by that executor. Re-evaluated against that
runtime, the decision below stands, and the revisit trigger is now a measured need rather
than the executor's existence. The MCP halves of paths 1 and 2 exist (`ix_duckdb_query`,
`ix_maintain_gate`); no MCP-to-capability adapter is wired for either.

## Decision

IXQL (Demerzel's governance pipeline DSL) and DuckDB+IX (`ix-duck`, the analyst bench)
are **complementary layers** — governed coordination vs. analysis. When we want them to
interoperate, we connect through an explicit MCP-to-capability adapter or a shared
JSON-on-disk contract (`docs/DUCKDB.md`), *not* by adding DuckDB/Parquet source or sink
nodes to the IXQL grammar — unless a demonstrated need and explicit Galactic-Protocol
sign-off justify the grammar change.

Concretely, the three integration opportunities resolve to:
1. **DuckDB as an IXQL data source** → an MCP tool exposes DuckDB SQL; IXQL reaches it
   through an explicitly registered capability adapter in the governance executor. The MCP
   half exists as `ix_duckdb_query`: stock DuckDB 1.x SQL over rows the caller supplies,
   run by the operator-installed CLI in safe mode — no files and no extensions, so no
   `ix-duck` UDFs. The capability adapter does not exist yet. The separate grammar's
   `mcp_tool_output("…")` / `database("…")` productions are not proof that this executor
   supports those forms. No grammar change is needed for the adapter.
2. **Governance verdict as a callable** → the already-built `maintain-gate`
   (`ix-duck::maintain`, the hexavalent T/P/U/D/F/C RSI oracle — *already* "DuckDB as referee")
   is exposed as the feature-gated MCP tool `ix_maintain_gate`. A registered IXQL capability
   may attach that verdict to what it returns, which `→ when T >= 0.8:` verdict arms then
   match on, subject to the existing advisory-versus-binding boundary. That capability
   adapter is an integration proposal, not something already wired.
3. **IXQL → trend table** → the governance executor can write JSON via `ix.io.write()`.
   A write whose path matches no registered schema passes ungated, so register a schema for
   the target path before treating the file as a governed artifact. DuckDB can read the resulting files
   with `read_json_auto`. A JSONL export and its schema would be separate work. No grammar
   change is needed.

## Why (the trade-off)

- **The executor has a bounded dialect.** `crates/ix-ixql` parses and executes the
  binding/record governance dialect, with injected capabilities, a filesystem or memory
  host, and schema-gated writes; its tests execute a vendored governance pipeline offline.
  It does not parse the separate `csv(…) → train(…)` ML-pipeline dialect, and it does not
  supply a built-in DuckDB source/sink. Its open coverage gaps include ix#281 (element
  scope for higher-order calls) and ix#331 (`when` blocks). New grammar nodes would expand a
  governed surface before a concrete integration requires them.
- **The grammar is a governed artifact.** Galactic-Protocol/grammar changes require explicit
  sign-off (CLAUDE.md "one-way doors"); the MCP seam and the file-format contract do not.
- **The adapter and file seams are available without runtime coupling.** Existing MCP tools
  can serve agents, while IXQL's governance executor can call explicitly registered
  capabilities. A bridge from a particular MCP tool to such a capability still has to be
  built and tested. DuckDB can independently read agreed JSON/JSONL artifacts; neither
  runtime is implicitly the other's executor. One file seam already runs: `ix_ixql::compile`
  emits a pipeline's stage DAG, which `crates/ix-duck/sql/ixql_plan.sql` reads on the stock
  DuckDB CLI — inspection of a plan, not execution of it (`docs/DUCKDB.md`).

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

Revisit when a measured use case cannot be served by the registered-capability and
JSON-on-disk seams, and a first-class `duckdb(…)` source or `write_parquet(…)` sink has
an end-to-end implementation and explicit grammar sign-off. The existence of the
governance executor alone is not sufficient reason to add syntax.
