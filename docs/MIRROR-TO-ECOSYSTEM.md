# Mirroring the ix registry pattern to tars + ga

A checklist for porting ix's `#[ix_skill]`-backed capability registry
to the F# (tars) and C# (ga) repos. The end state: all three repos
expose their tools through an identical link-time registry that unifies
CLI, MCP server, and visual surfaces.

## What the ix pattern delivers

| Problem solved | ix solution |
|---|---|
| Tool list drifts between CLI / MCP / UI | Single `IX_SKILLS` distributed slice, all surfaces read from it |
| Adding a tool requires edits in 3+ places | One `#[ix_skill]` annotation on the source function |
| Schema hand-coded for each surface | Hand-written JSON schemas referenced by `schema_fn` attribute |
| New tools miss the MCP surface | Parity test asserts exact registry count |
| Dead-code elimination silently strips tools | Documented force-link pattern |

The same shape applies to tars (F#) and ga (C#) — the languages differ
but the architectural primitives carry over.

---

## Port to tars (F# / .NET)

### Language-native equivalents

| ix concept | tars equivalent |
|---|---|
| `#[linkme::distributed_slice]` | F# `module` with `[<assembly: AssemblyAttribute>]` + reflection |
| `#[ix_skill]` proc-macro | F# attribute type + source generator (or `ModuleInitializer`) |
| `SkillDescriptor` struct | F# record: `TarsSkill = { Name; Domain; Inputs; Outputs; Handler }` |
| `linkme` crate | `System.Reflection.Assembly.GetTypes()` at startup |
| `IX_SKILLS` slice | Static `ImmutableArray<TarsSkill>` built at startup |
| `ix_registry::by_name` | `Map<string, TarsSkill>` lookup |
| `fn_ptr` dispatch | `Json -> Result<Json, string>` delegate |

### Architectural shape

```fsharp
// tars/Tars.Registry/Registry.fs
[<AttributeUsage(AttributeTargets.Method)>]
type TarsSkillAttribute(name: string, domain: string) =
    inherit Attribute()
    member _.Name = name
    member _.Domain = domain

type TarsSkill = {
    Name: string
    Domain: string
    Description: string
    Schema: unit -> JToken
    Handler: JToken -> Result<JToken, string>
}

module Registry =
    let private discover () =
        AppDomain.CurrentDomain.GetAssemblies()
        |> Seq.collect (fun a -> a.GetTypes())
        |> Seq.collect (fun t -> t.GetMethods(BindingFlags.Public ||| BindingFlags.Static))
        |> Seq.choose (fun m ->
            match m.GetCustomAttribute<TarsSkillAttribute>() with
            | null -> None
            | attr -> Some (buildSkill attr m))
        |> Seq.toArray

    let All = lazy (discover ())
    let byName name = All.Value |> Array.tryFind (fun s -> s.Name = name)
```

Then annotate tars functions:

```fsharp
[<TarsSkill("grammar.promote", "grammar")>]
let promoteGrammar (input: JToken) : Result<JToken, string> =
    // existing logic unchanged
    Ok (JToken.FromObject { promoted = true })
```

### Steps

1. Create `Tars.Registry` project with the attribute + discovery logic
2. Annotate existing tars tools (`ingest_ga_traces`, `run_promotion_pipeline`, …)
3. Wire the existing MCP server to enumerate `Registry.All.Value` for `tools/list`
4. Add a `TarsRegistryParity` xUnit test enumerating expected tool names
5. Replace hand-written MCP tool schemas with `[<TarsSkill>]`-tagged methods

---

## Port to ga (C# / .NET)

### Language-native equivalents

| ix concept | ga equivalent |
|---|---|
| `#[ix_skill]` | `[GaSkill(Name="...")]` attribute on static methods |
| `#[linkme::distributed_slice]` | `[ModuleInitializer]` method that registers into static list |
| Proc-macro | Roslyn source generator (optional — reflection works fine at 100-tool scale) |
| `ix_registry::by_name` | `ConcurrentDictionary<string, GaSkill>` |

### Architectural shape

```csharp
// ga/GuitarAlchemist.Registry/GaSkillAttribute.cs
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class GaSkillAttribute : Attribute {
    public string Name { get; }
    public string Domain { get; }
    public string[] GovernanceTags { get; init; } = Array.Empty<string>();
    public GaSkillAttribute(string name, string domain) {
        Name = name;
        Domain = domain;
    }
}

public record GaSkill(
    string Name,
    string Domain,
    string Description,
    Func<JsonNode, JsonNode> Handler,
    Func<JsonObject> Schema);

public static class Registry {
    private static readonly Lazy<IReadOnlyDictionary<string, GaSkill>> _index =
        new(() => Discover().ToDictionary(s => s.Name));

    public static GaSkill? ByName(string name) =>
        _index.Value.TryGetValue(name, out var s) ? s : null;

    public static IEnumerable<GaSkill> All => _index.Value.Values;

    private static IEnumerable<GaSkill> Discover() {
        return AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(a => a.GetTypes())
            .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(m => m.GetCustomAttribute<GaSkillAttribute>() is not null)
            .Select(BuildSkill);
    }
}
```

Then annotate ga functions:

```csharp
public static class ChordAnalysis {
    [GaSkill("chord.parse", "music-theory", GovernanceTags = new[]{"deterministic"})]
    public static JsonNode ParseChord(JsonNode input) {
        var symbol = input["symbol"]!.GetValue<string>();
        // existing GaParseChord logic
        return JsonNode.Parse($@"{{""root"":""C"",""quality"":""maj7""}}")!;
    }
}
```

### Steps

1. Create `GuitarAlchemist.Registry` project
2. Annotate ga's existing music-theory tools (chord / scale / progression APIs)
3. Wire the MCP server (or HTTP API) to enumerate `Registry.All`
4. Add xUnit parity test matching ix's pattern
5. (Optional) Roslyn source generator for compile-time schema validation

---

## Shared conventions across all three repos

After mirroring, all three repos honor the same contracts:

### Skill naming

Dotted hierarchy, MCP-ready after dot→underscore expansion:

```
ix:    supervised.linear_regression → ix_supervised_linear_regression
tars:  grammar.promote              → tars_grammar_promote
ga:    chord.parse                  → ga_chord_parse
```

### Hexavalent belief wire format

All three serialize belief state as single-letter symbols matching
`governance/demerzel/logic/hexavalent-state.schema.json`:

```json
{ "truth_value": "P", "confidence": 0.72 }
```

### Governance exit codes

```
0 = T  |  1 = P  |  2 = U  |  3 = D  |  4 = F  |  5 = C
10 = runtime error  |  64 = usage error
```

### Capability registry index

All three publish to `governance/demerzel/schemas/capability-registry.json`:

```json
{
  "repos": {
    "ix":   { "server": "ix",   "tools": {...} },
    "tars": { "server": "tars", "tools": {...} },
    "ga":   { "server": "ga",   "tools": {...} }
  }
}
```

The registry can be regenerated from each repo's native skill list
via a `tars-registry-export` / `ga-registry-export` xtask-equivalent.

### Force-link pattern

Each consumer crate / project that wants to enumerate skills from a
sibling module needs a force-link reference to prevent dead-code
elimination:

- **Rust**: `pub use sibling_crate::skills as _ix_agent_skills;`
- **F#**: reference the assembly from a startup path, or use
  `assembly: AssemblyAttribute("TarsSkillProvider", ...)` markers
- **C#**: `ModuleInitializer` in the assembly containing the skills

---

## Cross-validation checklist

Once tars + ga mirror the pattern, verify the ecosystem is coherent:

- [ ] Each repo has a registry type + discovery mechanism
- [ ] Each repo's MCP server enumerates its registry for `tools/list`
- [ ] Each repo has a parity test asserting exact tool count
- [ ] All three produce hexavalent verdicts with identical serialization
- [ ] All three can load the same Demerzel personas / policies
- [ ] `federation.discover` in ix can list tools from tars + ga
- [ ] A federated `ix.yaml` pipeline with `ga_bridge` + `tars_bridge`
      stages runs end-to-end (see
      `examples/showcase/advanced/music-theory.yaml`)

## Estimated porting effort

| Repo | Effort | Notes |
|---|---|---|
| **tars** (F#) | ~2-3 days | Attribute + reflection discovery is idiomatic F#; no source generators needed for ~10-20 tools |
| **ga** (C#) | ~2-3 days | Same shape as tars; ModuleInitializer eliminates reflection at startup |

The hardest parts are already done in ix — naming conventions, wire
formats, governance semantics, schema conventions. The mirror is
mostly mechanical translation.

## See also

- `docs/FEDERATION.md` — cross-repo integration architecture
- `crates/ix-registry/src/lib.rs` — reference Rust implementation
- `crates/ix-skill-macros/src/lib.rs` — proc-macro to model after
- `crates/ix-agent/src/skills/batch1.rs` — example annotated wrappers
- `crates/ix-agent/tests/parity.rs` — parity test pattern
