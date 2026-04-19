---
date: 2026-04-19
reversibility: two-way-door
revisit-trigger: vocabulary-miss rate in chatbot telemetry exceeds 10%
status: design — NOT YET IMPLEMENTED
---

# MCP sampling fallback for `GaSearchVoicings`

## Problem

The MCP path is currently typed-extractor-only. Fuzzy queries like
"show me something hypnotic and evening-ish" don't map to any canonical
tag, so retrieval returns the honest-empty response. The vocabulary tool
mitigates most cases (Claude pre-canonicalizes), but unmapped descriptive
words still fall through.

## Proposed approach

Add a server-initiated `sampling/createMessage` call back to the client
when the typed extractor returns empty AND the caller explicitly opts in.
Claude's own model then extracts `{chord, mode, tags}` from the raw query
without any server-side LLM configuration.

## API shape

```csharp
[McpServerTool]
public static async Task<string> GaSearchVoicings(
    IMcpServer server,               // NEW — injected per-call by SDK
    string query,
    int limit = 10,
    string? instrument = null,
    bool allowSampling = false,      // NEW — opt-in
    CancellationToken cancellationToken = default)
```

Default behavior unchanged. Clients that want fuzzy support set
`allowSampling: true`.

## Security guards (from 2026-04-18 octopus security review)

All mandatory before the sampling call fires:

1. **Input length cap.** Reject queries > 512 chars pre-send (DoS
   mitigation). Response to user: "query too long, please shorten."
2. **Control-char strip.** Filter U+0000–U+001F from the query before it
   enters the sampling prompt (prevents prompt-injection escape).
3. **Delimited prompt wrapping.** Wrap the user query in
   `<query>...</query>` tags within the sampling system prompt. LLMs
   don't consistently respect these but it's a cheap defense-in-depth.
4. **Output-token cap.** `maxTokens: 128`. Extraction needs ~50 tokens;
   128 gives headroom. Prevents output-inflation DoS.
5. **Temperature 0.** Deterministic extraction behavior.
6. **Hard timeout 10 s.** Cancel the sampling request after 10 s,
   fall back to honest-empty. Prevents stalled-client blocking.
7. **`AsyncLocal<int>` depth counter.** Guard against Claude's sampling
   response triggering a nested tool call that re-enters this path.
   Max depth 1; deeper = abort with error.
8. **Capability probe.** At server init, read
   `InitializeResult.Capabilities.Sampling`. If absent, register the
   tool with `allowSampling` inert (always falls through to empty).
9. **Output allow-list.** Extracted `chord` must parse via
   `ChordPitchClasses.TryParse` — garbage symbols rejected. Tags must
   match `SymbolicTagRegistry.GetBitIndex` before being trusted.
10. **SHA-256 query cache.** Reuse the exact cache in
    `LlmMusicalQueryExtractor` — identical fuzzy queries don't re-sample.

## Prompt template (pinned, SHA-checked at startup)

```
System: You extract musical intent. Return JSON only — no prose.
        Schema: {"chord": "<symbol|null>", "mode": "<name|null>", "tags": [...]}
        Rules: canonical chord symbols; null if not mentioned;
        tags lowercase and from the listed vocabulary only.
User:   <query>{escaped-user-query}</query>
```

## Implementation sketch

```csharp
private static async Task<StructuredQuery?> SampleQueryAsync(
    IMcpServer server, string query, CancellationToken ct)
{
    if (query.Length > 512) return null;
    var safe = new string(query.Where(c => c >= ' ').ToArray());

    using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
    timeoutCts.CancelAfter(TimeSpan.FromSeconds(10));

    if (_samplingDepth.Value >= 1) return null;
    _samplingDepth.Value++;
    try
    {
        var req = new CreateMessageRequestParams
        {
            Messages = [new SamplingMessage {
                Role = Role.User,
                Content = new TextContent { Text = $"<query>{safe}</query>" }
            }],
            SystemPrompt = PinnedSystemPrompt,  // verified SHA-256 at startup
            MaxTokens = 128,
            Temperature = 0.0,
        };
        var response = await server.SampleAsync(req, timeoutCts.Token);
        return ParseAndAllowList(response.Content?.Text ?? "");
    }
    finally { _samplingDepth.Value--; }
}

private static readonly AsyncLocal<int> _samplingDepth = new();
```

## Rollout

1. Land code behind `allowSampling: false` default so the tool surface
   doesn't change for existing callers.
2. Update SKILL.md with "set `allowSampling: true` when user's phrasing
   doesn't match the canonical vocabulary".
3. Telemetry: log `sampling_fired=true/false, sampling_latency_ms, extracted`
   to understand real-world miss rates.
4. If miss rate stays low (< 5%), deprioritize; vocabulary is winning.
   If miss rate is high, tune the system prompt.

## Open questions

- Should the cache TTL be longer for sampling results (30 min vs 5 min)?
  Sampling is expensive; longer TTL reduces repeat cost.
- Should sampling-extracted tags be marked differently in output so the
  chatbot can tell user "I interpreted 'dreamy' as ..."? Probably yes —
  transparency.

## Related

- `ix/docs/plans/2026-04-18-optic-k-v4-pp-per-partition-norm.md`
- `ga/.claude/skills/voicing-search/SKILL.md` — mention sampling once implemented
- Security review: 2026-04-18 octopus security-auditor findings on prompt
  injection, token-budget DoS, data exfiltration via sampling
