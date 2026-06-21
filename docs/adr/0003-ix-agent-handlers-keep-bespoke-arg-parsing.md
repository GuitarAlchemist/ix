# ix-agent tool handlers keep bespoke per-tool arg-parsing; the schema is NOT derived from the handler

Status: accepted (2026-06-21)

## Decision

The `ix-agent` MCP tool handlers (`crates/ix-agent/src/handlers.rs` + the `skills/batch{1,2,3}.rs`
wrappers) **keep their per-handler argument parsing** via the shared `parse_*` helpers. We do
**not** migrate them to a schema-driven shape where one `serde`/`schemars` struct per tool is the
single source of truth for both the JSON schema and the parse. The hand-built `schema.rs` DSL
remains the source of the emitted wire schema.

## Context

A `/improve-codebase-architecture` review (2026-06-21) surfaced the handler layer as a candidate
"deepening": each of ~72 tools repeats an arg-parse → call → serialize shape, and the schema
(`schema.rs`) is kept in sync with the handler by hand. The proposed deepening was "make the schema
the single source; parse derives from it."

On inspection the premise does not hold:

- **The `parse_*` calls are not pure boilerplate — they carry a tested error-message contract.**
  `crates/ix-agent/tests/graph_error_messages.rs` pins ~10 exact `ix_graph` messages
  (`edge[0]` + `'from'` + `float 0.5`, `non-negative`, `n_nodes = 3`, `unknown operation … pagerank`,
  the `[from, to, weight]` shape hint, …). Its header records that this UX "cost 15 minutes during
  the adversarial refactor oracle build." The helpers emit specific `Missing or invalid field 'X'` /
  `Non-numeric value in 'X'` strings throughout. A `serde_json::from_value` migration replaces these
  with serde's generic `missing field \`n_nodes\`` / `invalid type: floating point \`0.5\``, breaking
  that test and regressing the UX.
- **The emitted schema bytes are a stabilized external surface.** `schema.rs` was hand-built to be
  byte-faithful to the pre-existing tool schemas (the #148/#150 golden-snapshot migration). Schema
  hashes are a one-way door (CLAUDE.md "Log one-way doors") because MCP clients consume them. Deriving
  the schema from a struct (via `schemars`) changes those bytes for ~all 72 tools.
- **The `parse_*` helpers are already the deep module.** 15 helpers are reused across 72 handlers —
  that is the leverage and locality. The remaining per-handler calls are irreducible: each tool
  genuinely has different args, bespoke validation (`dimensions == 0`, enum-on-method), and a bespoke
  tested message.

## Why (the trade-off)

- **Deletion test fails for the safe variant.** Relocating the `parse_*` calls into a per-tool
  `Args::from_params` struct preserves messages and schema bytes, but deleting the struct just moves
  the calls back into the handler — complexity moves, it does not concentrate. That is a readability
  pass, not a deepening.
- **The deep variant trips two governed surfaces at once** — the tested error messages *and* the
  one-way-door schema bytes — for a leak (schema ↔ handler dual-source) that is small and stable.
- Net negative either way: the win is cosmetic; the risk is to two stabilized surfaces.

## Consequences

- Handlers continue to call `parse_str` / `parse_f64_array` / `parse_usize` / … directly. New tools
  follow the same pattern and add their schema to the `schema.rs` DSL by hand.
- The dual-source schema↔handler sync stays a known, accepted small cost. `tests/parity.rs`
  (tool-count oracle) and `tests/register_all_complexity_budget.rs` (cyclomatic ≤ 30 per `register_*`)
  remain the guards that keep it from drifting badly.

## Revisit trigger

Reconsider only if **both**: (1) a committed codegen path can derive the schema from a struct while
emitting byte-identical schema JSON (no client-visible churn), **and** (2) that path can carry the
field-level error-message contract that `graph_error_messages.rs` pins. Absent either, leave the
handler layer as it is.
