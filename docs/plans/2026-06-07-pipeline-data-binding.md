# Pipeline run-time data binding (`{param}`)

**Date:** 2026-06-07
**Status:** shipped
**Closes:** the dominant thinking-machine refusal — abstract requests with no
inline data (`"reduce this dataset with PCA"`) were refused because required
data args (`data`/`matrix`/`X`,`y`) had no value and the proposer (correctly)
won't fabricate inputs (dogfood finding #1, `state/thinking-machine/dogfood-2026-06-07-findings.md`).

## Decision

Make the already-declared-but-dead `PipelineSpec.params` field live, via a
**whole-value object-ref**: an arg of the form `{"param": "NAME"}` is replaced,
*before lowering*, with a value sourced from the spec's `params` bag (defaults)
overridden by run-time `ix pipeline run --param NAME=<json|@file>` (overrides
win). A referenced param with no value (no default / `null` default / no
override) is a hard error **before any stage executes** — never silently passed
to a skill.

This is distinct from `{"from": "stage"}` (resolved against an upstream stage's
*output* during execution). Params carry data the NL request didn't provide
inline; they are bound up front.

The NL proposer is instructed to emit `{"param": "NAME"}` + a `params: {NAME:
null}` declaration for absent data instead of `NO_COVERAGE`. `NO_COVERAGE` is
now reserved for a missing *capability*, never merely-absent data.

## Execution paths & hardening

There are two execution sites and BOTH bind before running, so the
no-unbound-`{param}` invariant holds everywhere (an adversarial review caught
that the compile path originally did not bind — the exact path the proposer
prompt enables):

- `ix pipeline run` → `pipeline::run` binds, then gates + executes.
- `ix pipeline compile --run` (and the `ix_nl_to_pipeline` MCP tool, which
  spawns it) → `compile::execute_and_narrate` binds **first** (before the
  governance gate and execution). Both `run` and `compile` take `--param`. With
  no `--param`, an unbound `{param}` is a hard "missing required param" error
  before any stage — never a `{"param":...}` blob handed to a skill. (The MCP
  tool plumbs no `--param` yet, so a `{param}` spec run via MCP fails closed; a
  params channel on the MCP surface is a follow-up.)

`--param` values are **data, not references**: a supplied value that itself
contains a `{from}`/`{param}` ref (at any depth) is rejected, so run-time input
cannot inject a DAG edge (data must not become control flow).

`ix pipeline compile` (without `--run`) reports the placeholders a template
still needs as `params_needed`, so the caller knows what to supply.

## Considered alternatives

- **Source skills** (`load_csv`/`fetch_url`): adds a file/URL read side-effect +
  governance/approval surface; the finding explicitly says the fix is a binding
  mechanism, not more skills. Deferred.
- **`${params.NAME}` string interpolation** (the field's original reserved
  design): string-only, so it cannot carry a JSON *matrix* — insufficient for
  data binding. Left reserved, unbuilt.

## Reversibility

**Two-way door.** Additive: specs without `{param}` are unchanged; the
`PipelineSpec` schema is still draft (`$id: urn:ix:pipeline:v1-draft`, not the
frozen public URL). No sibling repo consumes the PipelineSpec format (ga/tars/
Demerzel consume `optick.index` + SAE artifacts, not `ix.yaml`). The
`{param}` token can evolve without cross-repo coordination until the schema is
explicitly frozen.

**Revisit trigger:** if/when the PipelineSpec schema is promoted from
`urn:`-draft to the public `https://ix.guitaralchemist.com/...` `$id` (the
freeze milestone), the `{param}` resolution rule must be specified in the
frozen contract; re-run the dogfood batch to confirm the refusal rate drops.
