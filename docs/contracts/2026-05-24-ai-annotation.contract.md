# AI Source-Code Annotation — Cross-Repo Contract

**Version:** 0.2.0 (draft, Phase 0 — additive v2 kinds for value × complexity heatmap)
**Schema version:** 2
**Status:** Draft (Phase 0 of `ai-annotations` campaign, 2026-05-24)
**Producers:** `ix-ai-annotations` extractor, ga PostToolUse hook `Scripts/ai-annotation-scan.ps1`, human authors, product-owner declarations, telemetry pipelines
**Consumers:** `ix-ai-annotations` reconciler, `ix_annotations_scan` MCP tool, ga dashboard `/dev-data/ai-annotations`, ga `ValueComplexityHeatmap`, `grade-last-pr` skill, pre-merge gate
**Schema file:** `docs/contracts/ai-annotation.schema.json`

---

## 1. Why This Contract Exists

AI agents (Claude, Codex, Gemini, Junie, Auggie) routinely write code with implicit beliefs that disappear into the diff:

- "This array is sorted ascending"
- "The caller already holds the lock"
- "This is race-free under MIRI"

Today these claims live in commit messages, PR descriptions, ephemeral chat turns, or worse — nowhere. The next agent (often the same agent in a later session) has to re-derive them, frequently mis-deriving. Tests don't capture intent, only behavior. ADRs capture decisions but not micro-claims at the line level.

This contract defines a **lightweight, in-comment, structured marker** that:

1. Lives at the exact line the claim applies to.
2. Carries a **hexavalent truth value** (T/P/U/D/F/C) per Demerzel's canonical logic.
3. Carries a **certainty marker** (test / formal-proof / manually-reviewed / assumed / uncertain / inferred / dismissed).
4. Carries a **0.0–1.0 confidence** matching Demerzel's autonomous-action thresholds (≥0.9 / ≥0.7 / ≥0.5 / ≥0.3 / <0.3).
5. Optionally references **evidence** (a test path, PR number, MIRI run id).

The reconciler in Phase 1 cross-checks `[T:test]` claims against actual test runs, promotes conflicting same-line annotations to `C` (contradictory), and flags stale annotations whose file has changed without the annotation being updated.

This is complementary to `ix-invariant-coverage`, which scans an **external markdown catalog** of ecosystem-wide invariants. `ai-annotation` is **in-source, line-local, agent-authored**.

The contract is a **one-way door** on the field names and the truth_value enum (these align with Demerzel's `hexavalent-distribution.schema.json`). Field additions are additive and optional until v2.

---

## 2. The In-Source Marker Syntax

Annotations are written as comments using the language's native comment style. The parser supports:

- `//` (C, C++, Rust, JavaScript, TypeScript, C#, F#, Java, Go, Swift)
- `#` (Python, Ruby, Shell, PowerShell, YAML, TOML)
- `--` (Lua, SQL, Haskell)
- `/* ... */` (block, single-line only in v1)
- `<!-- ... -->` (HTML, XML, Markdown)

The canonical form:

```
<comment-marker> @ai:<kind> <claim> [<T>:<certainty> conf:<confidence> src:<evidence>]
```

Examples:

```rust
// @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]
// @ai:assumption caller holds the lock [P:assumed conf:0.7]
// @ai:hypothesis race-free under MIRI [U:uncertain conf:0.4]
```

```python
# @ai:contract returns non-empty iff input is non-empty [T:manually-reviewed conf:0.85 src:PR#313]
```

```fsharp
// @ai:smell deep nesting; consider extracting [D:inferred conf:0.6]
```

Field order inside the brackets is fixed: `T:certainty` first, then optional `conf:`, then optional `src:`.

---

## 3. Annotation Kinds (`kind`)

| kind             | meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `invariant`      | something that holds at this point and the surrounding code relies on it |
| `assumption`     | something the code assumes about its environment / caller / inputs       |
| `hypothesis`     | a guess to be verified; expect this to flip to T or F over time          |
| `contract`       | an externally-visible pre/post-condition (lighter than a full ADR)       |
| `smell`          | code-smell or suspicion; will likely be addressed                        |
| `decision`       | a one-line ADR pointer (the "why" of nearby code)                        |
| `hint`           | guidance for the next reader / agent (no truth claim, default `U`)       |
| `business-value` | (v2) operator-declared assertion that this code drives meaningful product value. Carries a free-text rationale + optional metric reference in `src:`. `source.author` should be `human` or `product-owner`; rarely auto-detected. Typical truth value `T` or `P`, certainty `manually-reviewed` or `assumed`. |
| `hot-path`       | (v2) measured-traffic assertion: this code path is hot in production. `source.author` should be `telemetry`; `src:` should reference the metric source (e.g., Grafana panel URL, OpenTelemetry trace id). Typical truth value `T`, certainty `test` or `manually-reviewed`. |

The `business-value` × `smell` cross-product drives the dashboard's value × complexity 2×2 heatmap (REFACTOR FIRST / DELETE CANDIDATE / KEEP STABLE / MAINTENANCE BURDEN).

---

## 4. Truth Value Enum (Hexavalent)

Aligned with `governance/demerzel/schemas/hexavalent-distribution.schema.json`:

| code | name          | meaning                                        |
|------|---------------|------------------------------------------------|
| `T`  | True          | verified with evidence                         |
| `P`  | Probable      | evidence leans true, not yet verified          |
| `U`  | Unknown       | insufficient evidence, triggers investigation  |
| `D`  | Doubtful      | evidence leans false, not yet refuted          |
| `F`  | False         | refuted with evidence                          |
| `C`  | Contradictory | conflicting evidence, triggers escalation      |

---

## 5. Certainty Marker

The `certainty` field describes **how** the truth value was reached:

| marker             | meaning                                                                |
|--------------------|------------------------------------------------------------------------|
| `test`             | a unit/integration/property test exercises this claim                  |
| `formal-proof`     | a checker (MIRI, prusti, kani, lean) verifies this                     |
| `manually-reviewed`| a human or reviewing agent inspected and signed off                    |
| `assumed`          | taken on faith from the spec / docs / caller contract                  |
| `uncertain`        | best-guess; explicit acknowledgement of low confidence                 |
| `inferred`         | derived from context (e.g., heuristic match against another test)      |
| `dismissed`        | previously flagged, now considered not-applicable                      |
| `detected-by-sentrux` | produced by the sentrux structural-rules engine (ground-truth verifier) |

The `detected-by-sentrux` certainty is reserved for annotations emitted by the
`ix-sentrux-annotations` bridge. Sentrux runs deterministic architectural rules
(e.g., `max_fn_lines`, cycle detection, redundancy) and surfaces every violation
as an `F` (False) annotation: the claim "this rule holds at this location" is
refuted by the machine. Treat it as ground-truth verifier output, not as a
human/AI opinion.

---

## 6. Confidence Thresholds (Demerzel-aligned)

| range     | behavior                          |
|-----------|-----------------------------------|
| `≥ 0.9`   | autonomous action                 |
| `≥ 0.7`   | act with note                     |
| `≥ 0.5`   | ask confirmation                  |
| `≥ 0.3`   | escalate                          |
| `< 0.3`   | do not act                        |

---

## 7. Canonical JSON Shape (extractor output)

```json
{
  "schema_version": 1,
  "id": "sha256:f3b1c2...",
  "kind": "invariant",
  "claim": "arr is sorted ascending",
  "truth_value": "T",
  "certainty": "test",
  "confidence": 0.95,
  "source": {
    "author": "claude",
    "model": "claude-opus-4-7",
    "evidence": "test_search.rs:42"
  },
  "location": {
    "path": "crates/ix-search/src/binary.rs",
    "line_start": 42,
    "line_end": 42
  },
  "created_at": "2026-05-24T18:00:00Z",
  "updated_at": "2026-05-24T18:00:00Z"
}
```

`id` is a deterministic SHA-256 of `path:line_start:kind:claim` so the same annotation hashes the same across runs (allows reconciler to track lifecycle).

The extractor in Phase 0 writes a JSONL file at `state/quality/ai-annotations.jsonl`, one annotation per line.

---

## 8. Storage Layout

| file                                                  | producer                            | consumer                          |
|-------------------------------------------------------|-------------------------------------|-----------------------------------|
| `state/quality/ai-annotations.jsonl`                  | `ix-ai-annotations extract`         | ga dashboard, reconciler          |
| `state/quality/ai-annotations-reconciliation.json`    | `ix-ai-annotations reconcile`       | ga dashboard, pre-merge gate      |

---

## 9. Phase Plan (this contract = Phase 0)

- **Phase 0 (this PR):** schema + extractor crate (`ix-ai-annotations`).
- **Phase 1:** reconciler + `ix_annotations_scan` MCP tool.
- **Phase 2:** ga dashboard tab on `/test#dev`.
- **Phase 3:** workflow hooks + pre-merge gate + `/grade-last-pr` integration.

---

## 10. Versioning

The contract is `v0.2.0` (draft). Field shapes and the truth-value enum are stable from Phase 0 onward; we only bump `schema_version` when a field is renamed or removed, OR when the `kind` enum is widened (consumers must opt into the new kinds). New optional fields are additive (no bump).

### Changelog

- **v0.2.0 (2026-05-24, schema_version 2)** — additive: two new kinds (`business-value`, `hot-path`) and two new `source.author` values (`product-owner`, `telemetry`) for the operator-decision-priority surface (value × complexity 2×2 heatmap). v1 readers will see unknown-kind annotations they must drop; v2 readers consume both. No field removals.
- **v0.1.0 (2026-05-24, schema_version 1)** — initial seven-kind contract (invariant / assumption / hypothesis / contract / smell / decision / hint).

Freeze milestone: end of Phase 3 (full pipeline shipped + ≥ 50 annotations in the codebase actively reconciled).
