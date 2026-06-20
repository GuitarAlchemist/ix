# GA consumes the maintain *verdict* (advisory tile); it does NOT run an ix-duck chatbot gate

Status: accepted (2026-06-20)

## Decision

When GA "leverages DuckDB+IX," it does so by **consuming the fused `maintain` verdict as an
advisory dashboard tile** — read from a JSON-on-disk snapshot in the existing
`ga/state/quality/**` scorecard convention. GA does **not** run any `ix-duck` lens (e.g. the
`chatbot` regression gate) in its CI as a PR merge gate.

## Context

`ix-duck`'s `chatbot` lens diffs a run's routed `agent_id` against the canonical
`_signature.json` — a routing-flip regression gate. The obvious "leverage in GA" idea is to
run that gate on GA chatbot PRs. **Grilling against the GA codebase (2026-06-20) overturned
that idea**: GA already gates the same surface natively, in .NET/PowerShell CI:

- **`Tests/.../Corpus/CanonicalSignatureChecker.cs`** — asserts a live trace matches per
  `(name, status, agent.id)` per position against `_signature.json`. *This is exactly the
  ix-duck `chatbot` hard signal.*
- **`Scripts/compare-trace-to-canonical.ps1`** — trace-shape diff (StepMissing /
  StatusMismatch / InvariantViolated) vs `_canonical.json`.
- **`.github/workflows/semantic-regression-chatbot.yml`** — PR-time answer-text drift
  (OpenAI cosine < 0.85).

GA CI is .NET/PowerShell and **builds no Rust**.

## Why (the trade-off)

- **A GA-side ix-duck gate would duplicate GA's native gates** (the cross-repo version of the
  in-repo `telemetry`-seam-vs-`source.rs` duplication that cost a closed PR the same day) —
  and would drag a Rust toolchain + a bundled-DuckDB compile into GA's .NET CI for **zero net
  signal**.
- **What GA genuinely lacks is the *fusion*.** GA checks each signal (routing / shape /
  semantic / embeddings) **separately**; nothing combines them. `ix-duck::maintain` fuses
  routing + ood + loops + chatbot into one hexavalent **T/P/U/D/F/C** verdict
  (metric↑ ∧ guardrail held ∧ converging ∧ in-distribution — never an average). The
  cross-signal convergence view is the new value, and it belongs on a **dashboard**, not in a
  blocking gate.
- **Formats-not-coupling** (CLAUDE.md / `docs/DUCKDB.md`): GA reads an ix-emitted snapshot in
  the `ga/state/quality/**` scorecard shape — no runtime coupling, no Rust in GA CI.
- **The verdict is advisory anyway** (`docs/contracts/maintain-gate.contract.md`: binding only
  after Phase-3b ledger write-isolation), so an *advisory tile* is the honest surface — a
  merge-blocking gate would over-claim.

## Consequences

- The GA-side build is a **dashboard tile**, fed by a JSON snapshot — not a CI gate.
- It requires an **IX-side producer that doesn't exist yet**: a scheduled run of
  `maintain::evaluate` over live `../ga` that emits the verdict snapshot (today the ledger has
  only manual entries and no workflow runs it). That producer is the prerequisite — without it
  the tile is green-but-dead. (Tracked in the accompanying plan.)
- If GA ever wants a *blocking* chatbot gate, it extends its **own** `CanonicalSignatureChecker`
  / PowerShell tooling — not an imported ix-duck binary.

## Revisit trigger

If the maintain verdict becomes **binding** (Phase-3b write-isolation lands) *and* an operator
wants it to block GA merges, reconsider whether GA's pipeline should consume the verdict as a
gate (still via the contract, still no Rust in GA CI) rather than only display it.
