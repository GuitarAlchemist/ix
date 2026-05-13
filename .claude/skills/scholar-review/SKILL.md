---
name: scholar-review
description: Deep code review of an IX crate by piping its source into Qwopus-GLM-18B (Qwen3.5 + Opus + GLM distilled frankenmerge). Offline, free, slow, reasoning-heavy.
---

# Scholar Review

Pipe an IX crate's source into the local **Qwopus-GLM-18B** reasoning model for a deep code review. The model is a frankenmerge of Qwen 3.5 + Opus-distilled + GLM-distilled layers — benchmarks above its weight class on reasoning but overthinks (30-60s typical, longer for big crates).

## When to Use

- You want a **second opinion** on a crate's architecture, invariants, or algorithmic correctness before shipping
- You want to generate **review notes for a slow-cook PR** without burning API credits
- Batch analysis across multiple crates where API cost would matter
- You explicitly want a *reasoning* model, not a tool-using agent

Do **not** use for:
- Tasks that require file edits — this is read-only analysis
- Interactive fast-feedback loops — latency is 30-60s+
- Questions that need fresh web lookups — model is fully local

## Prerequisites

1. Ollama running locally on `http://localhost:11434`
2. Model imported: `ollama list | grep qwopus-glm` → `qwopus-glm:18b` (9.8 GB)
3. If missing, import from `C:\Users\spare\models\qwopus-glm\Modelfile` via
   `ollama create qwopus-glm:18b -f Modelfile`

## Performance expectations

Measured on a Windows machine with consumer GPU:

| Input size        | Typical latency |
| ----------------- | --------------- |
| 1 file, 200 LOC   | 1-3 minutes     |
| Focused question  | 30-90 seconds   |
| Whole crate       | 10+ minutes (often times out) |

**Recommendation:** Review **one file at a time** or pair a small bundle with
a **specific `--focus`**. Avoid whole-crate dumps — the model chews on them
for minutes without delivering proportionally better insight.

## Invocation

```
/scholar-review <crate-name> [--focus "<aspect>"]
```

- `<crate-name>` — directory under `crates/` (e.g. `ix-voicings`, `ix-bracelet`)
- `--focus` — optional pointed question. Defaults to full architectural review.

Examples:

```
/scholar-review ix-voicings
/scholar-review ix-bracelet --focus "check D12 symmetry handling in the P/L/R transforms"
/scholar-review ix-quality-trend --focus "identify failure modes under concurrent writes"
```

## Pipeline

### Step 1: Assemble source bundle

Read `crates/<name>/Cargo.toml` + all `.rs` files under `crates/<name>/src/` (skip `tests/` unless the focus mentions tests). Keep the bundle under ~100 KB — truncate the largest file if needed and note the truncation.

### Step 2: Build the prompt

System prompt:

> You are a senior Rust engineer doing a deep code review. Reason carefully about invariants, edge cases, API ergonomics, performance, and test coverage. Cite the specific file/function when making claims. Output a prioritized list (HIGH/MEDIUM/LOW) with concrete suggestions. If the code looks solid, say so — don't invent problems.

User prompt:

> Review the `<crate-name>` crate. {{focus-clause}}
>
> Cargo.toml:
> ```
> {cargo_toml}
> ```
>
> Source files:
> ```
> // === src/lib.rs ===
> {lib_rs}
>
> // === src/<other>.rs ===
> {...}
> ```

### Step 3: POST to Ollama (streaming)

```bash
curl -sN http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d @request.json \
  -m 900 \
  | python -c "import json,sys
for line in sys.stdin:
    try:
        d = json.loads(line)
        chunk = d.get('message',{}).get('content','')
        if chunk: print(chunk, end='', flush=True)
        if d.get('done'): print()
    except: pass"
```

Body:

```json
{
  "model": "qwopus-glm:18b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "stream": true,
  "options": { "temperature": 0.5, "num_ctx": 8192 }
}
```

Streaming matters — `stream: false` with a 10+ minute wait looks indistinguishable from a hang. Stream so the user sees the review materialize paragraph by paragraph.

### Step 4: Present results

Print the model's review verbatim, prefixed by:

```
# Scholar Review — <crate-name>
Model: qwopus-glm:18b  ·  Duration: <N>s  ·  Input: <N> tokens
```

If `state/scholar-reviews/` exists, also write the raw review to `state/scholar-reviews/<crate>-YYYY-MM-DD.md` so future sessions can diff it.

## Notes

- Qwopus-GLM has no tool-use — this skill is read-only analysis, nothing executes.
- Multiple back-to-back reviews reuse the same model in VRAM — the first call is slow (model load), subsequent ones are 2-3× faster.
- If the model hallucinates Rust syntax or invents crates, **trust what you read in the source** over what the model claims. The scholar pass is a lens, not an oracle.
