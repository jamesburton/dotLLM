# Strict Tool-Call Constraints (#104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run --tool-choice required/function` emit a fully valid, self-terminating JSON tool call (correct nested `arguments`, no leaked keys, exactly one tool's schema enforced) for single- and multi-tool, keys in any order, on weak/quantised bases.

**Architecture:** Extend the existing `SchemaTracker` JSON-schema mask engine with (1) closed-schema generation in `ToolCallSchemaBuilder`, (2) an emitted-branch dead-end fix in `PropertyNameTrie`/`SchemaTracker`, and (3) bounded parallel `anyOf` branch-narrowing (refactor single-position state into a `BranchState` struct, then run up to K=8 in parallel, OR/prune per char).

**Tech Stack:** C# / .NET 10, xUnit, `System.Text.Json`, `InlineArray` value-type state.

**Design doc:** `.planning/2026-06-27-tool-call-strict-constraints-design.md`

## Global Constraints

- Target framework .NET 10; `<Nullable>enable</Nullable>`. File-scoped namespaces.
- Zero managed allocations on the per-token mask hot path; `SchemaTracker`/`BranchState` are value types copied by value (InlineArray), no heap.
- K (max strict parallel branches) = **8**. Beyond K: fall back to the existing type-union overapproximation (do not crash), and `Debug`-log once.
- Single-branch path MUST stay behaviour-identical to today (existing constraint tests are the regression gate).
- Never run the full test suite (GPU deadlock) — always `dotnet test --filter`. GPU/model tests env-gated on `DOTLLM_BITNET_GGUF`, plain `[Fact]` + early-return when unset.
- Commits reference `(#104)`; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. PR targets `dev`.
- Build/test from the worktree: `E:\Development\dotLLM\.claude\worktrees\strict-tools`.

---

### Task 1: Closed-schema generation in `ToolCallSchemaBuilder` (Piece 1)

**Files:**
- Modify: `src/DotLLM.Engine/Constraints/ToolCallSchemaBuilder.cs`
- Test: `tests/DotLLM.Tests.Unit/Engine/Constraints/ToolCallSchemaBuilderTests.cs`

**Interfaces:**
- Produces: `ToolCallSchemaBuilder.BuildForFunction(ToolDefinition, string)` and `BuildForRequired(ToolDefinition[], string)` now emit schemas where **every object has `additionalProperties:false`**, and multi-tool `BuildForRequired` emits a top-level `{"anyOf":[ <per-tool object> … ]}` (each branch `{"type":"object","properties":{"name":{"const":"Tᵢ"},"arguments":<hardened>},"required":["name","arguments"],"additionalProperties":false}`). Single-tool `BuildForRequired` still equals `BuildForFunction`.
- Consumes: nothing new.

- [ ] **Step 1: Update the three obsolete multi-tool tests to the anyOf contract**

In `ToolCallSchemaBuilderTests.cs`, replace `BuildForRequired_MultipleTools_UsesEnumForName`, `BuildForRequired_MultipleTools_EnumContainsBothNames`, and `BuildForParallelCalls_ItemsHaveEnumName` with the anyOf forms:

```csharp
[Fact]
public void BuildForRequired_MultipleTools_UsesAnyOfPerTool()
{
    string schema = ToolCallSchemaBuilder.BuildForRequired([WeatherTool, TimeTool]);
    using var doc = JsonDocument.Parse(schema);
    Assert.True(doc.RootElement.TryGetProperty("anyOf", out var anyOf));
    Assert.Equal(2, anyOf.GetArrayLength());
    foreach (var branch in anyOf.EnumerateArray())
    {
        Assert.Equal("object", branch.GetProperty("type").GetString());
        Assert.True(branch.GetProperty("properties").GetProperty("name").TryGetProperty("const", out _));
        Assert.False(branch.GetProperty("additionalProperties").GetBoolean());
    }
}

[Fact]
public void BuildForRequired_MultipleTools_BranchesCoverBothNames()
{
    string schema = ToolCallSchemaBuilder.BuildForRequired([WeatherTool, TimeTool]);
    using var doc = JsonDocument.Parse(schema);
    var names = new List<string>();
    foreach (var branch in doc.RootElement.GetProperty("anyOf").EnumerateArray())
        names.Add(branch.GetProperty("properties").GetProperty("name").GetProperty("const").GetString()!);
    Assert.Contains("get_weather", names);
    Assert.Contains("get_time", names);
}

[Fact]
public void BuildForParallelCalls_ItemsAreAnyOf()
{
    string schema = ToolCallSchemaBuilder.BuildForParallelCalls([WeatherTool, TimeTool]);
    using var doc = JsonDocument.Parse(schema);
    var items = doc.RootElement.GetProperty("items");
    Assert.True(items.TryGetProperty("anyOf", out var anyOf));
    Assert.Equal(2, anyOf.GetArrayLength());
}
```

- [ ] **Step 2: Add new hardening tests**

```csharp
[Fact]
public void BuildForFunction_OuterObject_ForbidsAdditionalProperties()
{
    string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);
    using var doc = JsonDocument.Parse(schema);
    Assert.False(doc.RootElement.GetProperty("additionalProperties").GetBoolean());
}

[Fact]
public void BuildForFunction_ArgumentsObject_ForbidsAdditionalProperties()
{
    string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);
    using var doc = JsonDocument.Parse(schema);
    var args = doc.RootElement.GetProperty("properties").GetProperty("arguments");
    Assert.False(args.GetProperty("additionalProperties").GetBoolean());
}

[Fact]
public void BuildForFunction_NestedObjectArgs_ForbidsAdditionalPropertiesRecursively()
{
    var nested = new ToolDefinition("set_addr", "x",
        """{"type":"object","properties":{"addr":{"type":"object","properties":{"city":{"type":"string"}}}},"required":["addr"]}""");
    string schema = ToolCallSchemaBuilder.BuildForFunction(nested);
    using var doc = JsonDocument.Parse(schema);
    var addr = doc.RootElement.GetProperty("properties").GetProperty("arguments")
        .GetProperty("properties").GetProperty("addr");
    Assert.False(addr.GetProperty("additionalProperties").GetBoolean());
}

[Fact]
public void BuildForFunction_DeclaredRequiredPreserved_OptionalNotForced()
{
    // WeatherTool requires "location" only; an optional prop must stay optional.
    var tool = new ToolDefinition("t", "x",
        """{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"]}""");
    string schema = ToolCallSchemaBuilder.BuildForFunction(tool);
    using var doc = JsonDocument.Parse(schema);
    var args = doc.RootElement.GetProperty("properties").GetProperty("arguments");
    var req = new List<string>();
    foreach (var r in args.GetProperty("required").EnumerateArray()) req.Add(r.GetString()!);
    Assert.Equal(["a"], req);
}
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~ToolCallSchemaBuilderTests" -v minimal`
Expected: FAIL (anyOf / additionalProperties not yet emitted).

- [ ] **Step 4: Implement `Harden` + rewrite `BuildForRequired`/`BuildForParallelCalls`**

Add a recursive hardener and route all generated object schemas through it. In `ToolCallSchemaBuilder.cs`:

```csharp
/// <summary>
/// Recursively injects "additionalProperties":false into every object schema that does not
/// already set it, recursing through properties / items / anyOf / $defs. Declared "required"
/// is preserved as-is (optional properties stay optional). Returns a compact JSON string.
/// </summary>
private static string Harden(string schemaJson)
{
    using var doc = JsonDocument.Parse(schemaJson);
    var buf = new System.Buffers.ArrayBufferWriter<byte>();
    using (var w = new Utf8JsonWriter(buf))
        HardenElement(doc.RootElement, w);
    return System.Text.Encoding.UTF8.GetString(buf.WrittenSpan);
}

private static void HardenElement(JsonElement el, Utf8JsonWriter w)
{
    if (el.ValueKind != JsonValueKind.Object) { el.WriteTo(w); return; }

    bool isObjectType = el.TryGetProperty("type", out var t)
        && t.ValueKind == JsonValueKind.String && t.GetString() == "object";
    bool hasAddl = el.TryGetProperty("additionalProperties", out _);

    w.WriteStartObject();
    foreach (var prop in el.EnumerateObject())
    {
        w.WritePropertyName(prop.Name);
        switch (prop.Name)
        {
            case "properties":
                w.WriteStartObject();
                foreach (var p in prop.Value.EnumerateObject())
                {
                    w.WritePropertyName(p.Name);
                    HardenElement(p.Value, w);
                }
                w.WriteEndObject();
                break;
            case "items":
                HardenElement(prop.Value, w);
                break;
            case "anyOf":
            case "allOf":
            case "oneOf":
                w.WriteStartArray();
                foreach (var alt in prop.Value.EnumerateArray()) HardenElement(alt, w);
                w.WriteEndArray();
                break;
            case "$defs":
                w.WriteStartObject();
                foreach (var d in prop.Value.EnumerateObject())
                {
                    w.WritePropertyName(d.Name);
                    HardenElement(d.Value, w);
                }
                w.WriteEndObject();
                break;
            default:
                prop.Value.WriteTo(w);
                break;
        }
    }
    if (isObjectType && !hasAddl)
        w.WriteBoolean("additionalProperties", false);
    w.WriteEndObject();
}
```

Change `NormalizeParametersSchema` callers in `BuildForFunction` to wrap the params in `Harden(...)`:

```csharp
sb.Append(Harden(NormalizeParametersSchema(tool.ParametersSchema)));
```

Rewrite `BuildForRequired` multi-tool branch to anyOf (drop the enum path):

```csharp
public static string BuildForRequired(ToolDefinition[] tools, string argumentsKey = "arguments")
{
    if (tools.Length == 1)
        return BuildForFunction(tools[0], argumentsKey);

    var sb = new StringBuilder(1024);
    sb.Append("{\"anyOf\":[");
    for (int i = 0; i < tools.Length; i++)
    {
        if (i > 0) sb.Append(',');
        sb.Append(BuildForFunction(tools[i], argumentsKey));
    }
    sb.Append("]}");
    return sb.ToString();
}
```

`BuildForParallelCalls` is unchanged in shape (its `items` is `BuildForRequired(...)`, now anyOf).

- [ ] **Step 5: Run the full builder test class**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~ToolCallSchemaBuilderTests" -v minimal`
Expected: PASS (all, including the unchanged compile/parse tests and `BuildForRequired_SingleTool_SameAsBuildForFunction`).

- [ ] **Step 6: Commit**

```bash
git add src/DotLLM.Engine/Constraints/ToolCallSchemaBuilder.cs tests/DotLLM.Tests.Unit/Engine/Constraints/ToolCallSchemaBuilderTests.cs
git commit -m "feat(constraints): emit closed (additionalProperties:false) anyOf tool-call schemas (#104)"
```

---

### Task 2: Emitted-branch dead-end fix (Piece 2)

**Files:**
- Modify: `src/DotLLM.Engine/Constraints/Schema/PropertyNameTrie.cs`
- Modify: `src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs`
- Test: `tests/DotLLM.Tests.Unit/Engine/Constraints/Schema/SchemaCompilerTests.cs` (trie) and `tests/DotLLM.Tests.Unit/Engine/Constraints/JsonSchemaConstraintTests.cs` (end-to-end no-empty-mask)

**Interfaces:**
- Produces: `PropertyNameTrie.ReachableTerminalBits(int nodeIndex) → ulong` (OR of property-bit indices of all terminals reachable from a node; bit position = insertion order of the value, matching `PropertyNames` order in `SchemaCompiler`). `SchemaTracker` consults it with the per-level emitted mask so the model can never enter a key prefix whose only completions are already emitted.
- Consumes: `SchemaNode.PropertyNames` ordering (bit positions) from Task's compiler (unchanged).

- [ ] **Step 1: Write the failing trie test**

In `SchemaCompilerTests.cs`:

```csharp
[Fact]
public void PropertyNameTrie_ReachableTerminalBits_TracksReachableProperties()
{
    var trie = new DotLLM.Engine.Constraints.Schema.PropertyNameTrie(["name", "arguments"]);
    // Root reaches both (bits 0 and 1).
    Assert.Equal(0b11UL, trie.ReachableTerminalBits(0));
    // After 'n', only "name" (bit 0) is reachable.
    Assert.True(trie.TryGetChild(0, 'n', out int nNode));
    Assert.Equal(0b01UL, trie.ReachableTerminalBits(nNode));
    // After 'a', only "arguments" (bit 1) is reachable.
    Assert.True(trie.TryGetChild(0, 'a', out int aNode));
    Assert.Equal(0b10UL, trie.ReachableTerminalBits(aNode));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~SchemaCompilerTests.PropertyNameTrie_ReachableTerminalBits_TracksReachableProperties" -v minimal`
Expected: FAIL ("ReachableTerminalBits" not defined).

- [ ] **Step 3: Implement `ReachableTerminalBits` in `PropertyNameTrie`**

Bit position = index of the value in construction order. Track each terminal's bit, then propagate bottom-up. Add a `ulong[] _reachable` field computed in the constructor after freezing:

```csharp
private readonly ulong[] _reachable;

// In the constructor, AFTER building _nodes:
// 1) assign each value an index in insertion order → terminal bit.
//    Re-walk values to map terminal node → bit.
_reachable = new ulong[_nodes.Length];
{
    // Map terminal node index → bit (insertion order of values).
    int bit = 0;
    var terminalBit = new Dictionary<int, int>();
    foreach (string value in valuesList)   // capture `values` into a List<string> at top of ctor
    {
        int current = 0;
        foreach (char c in value) { TryGetChild(current, c, out current); }
        terminalBit[current] = bit++;
    }
    // Post-order propagate reachable bits up from leaves.
    ulong Compute(int node)
    {
        ulong bits = 0;
        if (terminalBit.TryGetValue(node, out int b) && b < 64) bits |= 1UL << b;
        foreach (var (_, child) in _nodes[node].Children) bits |= Compute(child);
        _reachable[node] = bits;
        return bits;
    }
    Compute(0);
}

/// <summary>OR of the bit indices (value insertion order) of all terminals reachable from node.</summary>
[MethodImpl(MethodImplOptions.AggressiveInlining)]
public ulong ReachableTerminalBits(int nodeIndex) => _reachable[nodeIndex];
```

(Capture the incoming `values` into `var valuesList = values.ToList();` at the top of the constructor and iterate `valuesList` in the build loop so it can be reused here.)

- [ ] **Step 4: Run the trie test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~SchemaCompilerTests.PropertyNameTrie_ReachableTerminalBits_TracksReachableProperties" -v minimal`
Expected: PASS.

- [ ] **Step 5: Write the failing constraint test (no empty mask / no duplicate key)**

In `JsonSchemaConstraintTests.cs` (use the existing test tokenizer/helper in that file; mirror its setup):

```csharp
[Fact]
public void StrictObject_AfterEmittingKey_NeverAllowsDeadEndPrefix()
{
    // Schema: {name, arguments} both required, additionalProperties:false.
    string schema = ToolCallSchemaBuilder.BuildForFunction(new ToolDefinition(
        "get_weather", "x",
        """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""));
    var c = new JsonSchemaConstraint(TestTokenizer, schema);
    // Drive: {"name":"get_weather",
    Feed(c, "{\"name\":\"get_weather\",");
    // At ObjectNextKey the only un-emitted key is "arguments": 'n' (re-name) must be masked out,
    // 'a' must be allowed. The mask must be non-empty.
    var mask = c.GetAllowedTokens();
    Assert.True(AnyAllowed(mask), "mask must be non-empty");
    Assert.True(TokenAllowedStartingWith(c, "\"a"), "must allow opening the arguments key");
    Assert.False(TokenAllowedStartingWith(c, "\"n"), "must not allow re-opening the name key");
}
```

(`Feed`, `AnyAllowed`, `TokenAllowedStartingWith`, `TestTokenizer` — reuse/add small helpers consistent with the existing `JsonSchemaConstraintTests` patterns; if absent, add private helpers that decode candidate tokens via the test tokenizer and check `mask.IsAllowed`.)

- [ ] **Step 6: Run to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests.StrictObject_AfterEmittingKey_NeverAllowsDeadEndPrefix" -v minimal`
Expected: FAIL (a `"n`-prefixed token is currently allowed into a dead end).

- [ ] **Step 7: Wire `ReachableTerminalBits` into `SchemaTracker` key checks**

In `SchemaTracker.cs`, replace the "some property still possible" logic with reachable-bit checks against the emitted mask:

- `IsObjectOpenCharAllowed`/`IsObjectNextKeyCharAllowed` `'"'` case: allow only if
  `!node.AdditionalPropertiesForbidden || (trie.ReachableTerminalBits(0) & ~emitted) != 0`.
- `IsTrieCharValid(c)`: when `node.AdditionalPropertiesForbidden`, allow `c` only if
  `trie.TryGetChild(_trieNodeIndex, c, out int child) && (trie.ReachableTerminalBits(child) & ~emitted) != 0`.

where `emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0`. Keep the existing `IsTrieTerminal` already-emitted rejection as a backstop.

- [ ] **Step 8: Run both tests + the existing constraint suite (regression)**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests|FullyQualifiedName~SchemaCompilerTests" -v minimal`
Expected: PASS (new tests + all existing).

- [ ] **Step 9: Commit**

```bash
git add src/DotLLM.Engine/Constraints/Schema/PropertyNameTrie.cs src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs tests/DotLLM.Tests.Unit/Engine/Constraints/Schema/SchemaCompilerTests.cs tests/DotLLM.Tests.Unit/Engine/Constraints/JsonSchemaConstraintTests.cs
git commit -m "fix(constraints): prune emitted property branches to kill dead-end key prefixes (#104)"
```

---

### Task 3: Refactor `SchemaTracker` single-position state into `BranchState` (pure refactor)

**Files:**
- Modify: `src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs`
- (no new tests — existing `JsonSchemaConstraintTests` are the behaviour-identity gate)

**Interfaces:**
- Produces: an internal `struct BranchState` holding what were `SchemaTracker`'s single-position fields (`_nodeStack`, `_stackDepth`, `_currentNodeIndex`, `_emittedProps`, `_keyBuffer`, `_keyLength`, `_trieNodeIndex`, `_enumTrieNodeIndex`, `_arrayIndices`, `_inKeyString`, `_inEnumString`) plus the methods that mutate/inspect them (`IsCharAllowed`, `OnCharAdvanced`, push/pop/key/trie helpers). `SchemaTracker` holds exactly one `BranchState` after this task and delegates to it; public API (`IsCharAllowedBySchema`, `OnCharAdvanced`, `GetSchemaStateKey`, `IsComplete`, `Reset`) is unchanged and byte-for-byte equivalent.
- Consumes: `CompiledSchema` (now stored on `BranchState`), `JsonCharParser`.

- [ ] **Step 1: Confirm the regression baseline is green**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests" -v minimal`
Expected: PASS (record the count).

- [ ] **Step 2: Extract `BranchState`**

Move every per-position field and the helper methods (`IsCharAllowedBySchema` body, `OnCharAdvanced` body, `PushObject/PushArray/PopObject/PopArray/AdvanceArrayItem`, all key/trie/enum helpers, `IsValueStartCharAllowed`, object/array char checks) into `internal struct BranchState { … }`, keeping the same `InlineArray` backing structs. `BranchState` takes `CompiledSchema` in its ctor (mirrors current `SchemaTracker` ctor). Signatures keep `in JsonCharParser parser`.

`SchemaTracker` becomes a thin wrapper:

```csharp
internal struct SchemaTracker
{
    private BranchState _b0;          // single active branch (Task 4 generalises to a set)
    public SchemaTracker(CompiledSchema schema) => _b0 = new BranchState(schema);
    public readonly bool IsComplete(in JsonCharParser p) => _b0.IsComplete(in p);
    public readonly bool IsCharAllowedBySchema(char c, in JsonCharParser p) => _b0.IsCharAllowed(c, in p);
    public void OnCharAdvanced(char c, in JsonCharParser p) => _b0.OnCharAdvanced(c, in p);
    public readonly SchemaStateKey GetSchemaStateKey(in JsonCharParser p) => _b0.GetSchemaStateKey(in p);
    public void Reset() => _b0.Reset();
}
```

- [ ] **Step 3: Build**

Run: `dotnet build src/DotLLM.Engine -v minimal`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Run the regression suite — must match the Step 1 baseline exactly**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests|FullyQualifiedName~SchemaCompilerTests" -v minimal`
Expected: PASS, same count as Step 1.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs
git commit -m "refactor(constraints): extract single-position state into BranchState (#104)"
```

---

### Task 4: Bounded parallel `anyOf` branch-narrowing (Piece 3 core)

**Files:**
- Modify: `src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs`
- Test: `tests/DotLLM.Tests.Unit/Engine/Constraints/JsonSchemaConstraintTests.cs`

**Interfaces:**
- Produces: `SchemaTracker` holding `InlineArray<BranchState>` (capacity 8) + `_liveCount`. `IsCharAllowedBySchema = OR over live branches`; `OnCharAdvanced` advances survivors, prunes the rest, collapses to 1 when possible; `IsComplete = any live branch complete`. `GetSchemaStateKey` folds the live set. Fork happens when a value position is an `anyOf` node (alt count ≤ 8); alt count > 8 → no fork (union fallback).
- Consumes: `BranchState` from Task 3; `SchemaNode.AnyOfNodeIndices`.

**Forking mechanics:** A `BranchState` exposes a readonly `TryGetAnyOfAlternatives(in JsonCharParser p, out ReadOnlySpan<int> altNodeIndices)` that returns true when the *next value to generate* at its current position is an `anyOf` node. On the transition that would start that value, `SchemaTracker` replaces the single branch with one clone per alternative, each re-seeded so its `_currentNodeIndex` is the alternative node (and, for the tool-call wrapper, its object frame uses that branch's properties/required/trie). Implement seeding by adding `BranchState.WithCurrentNode(int nodeIndex)` that returns a copy with `_currentNodeIndex` set (object frame is established normally when `{` is consumed).

- [ ] **Step 1: Write the failing multi-tool narrowing tests (both orderings)**

```csharp
private static JsonSchemaConstraint MultiToolConstraint() => new(TestTokenizer,
    ToolCallSchemaBuilder.BuildForRequired([
        new ToolDefinition("get_weather","x","""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""),
        new ToolDefinition("get_time","x","""{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}""")]));

[Fact]
public void MultiTool_NameFirst_EnforcesMatchedToolArguments()
{
    var c = MultiToolConstraint();
    Feed(c, "{\"name\":\"get_weather\",\"arguments\":{");
    // Only "city" (get_weather) may open; "zone" (get_time) must be masked.
    Assert.True(TokenAllowedStartingWith(c, "\"c"));
    Assert.False(TokenAllowedStartingWith(c, "\"z"));
}

[Fact]
public void MultiTool_ArgsFirst_NarrowsOnceNameSeen()
{
    var c = MultiToolConstraint();
    // arguments before name: both tools' first keys are live until name disambiguates.
    Feed(c, "{\"arguments\":{");
    Assert.True(TokenAllowedStartingWith(c, "\"c") || TokenAllowedStartingWith(c, "\"z"),
        "before name, at least one tool's arg key is available");
}

[Fact]
public void MultiTool_WrongToolArgKey_Rejected()
{
    var c = MultiToolConstraint();
    Feed(c, "{\"name\":\"get_time\",\"arguments\":{");
    Assert.True(TokenAllowedStartingWith(c, "\"z"));
    Assert.False(TokenAllowedStartingWith(c, "\"c"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests.MultiTool" -v minimal`
Expected: FAIL (single branch tracks the anyOf union; wrong-tool keys currently allowed).

- [ ] **Step 3: Implement the bounded branch set in `SchemaTracker`**

```csharp
[InlineArray(8)]
internal struct BranchStateArray { private BranchState _e; }

internal struct SchemaTracker
{
    private const int K = 8;
    private BranchStateArray _branches;
    private int _liveCount;

    public SchemaTracker(CompiledSchema schema) { _branches[0] = new BranchState(schema); _liveCount = 1; }

    public readonly bool IsCharAllowedBySchema(char c, in JsonCharParser p)
    {
        for (int i = 0; i < _liveCount; i++)
            if (_branches[i].IsCharAllowed(c, in p)) return true;
        return false;
    }

    public void OnCharAdvanced(char c, in JsonCharParser p)
    {
        // Fork point: if a single live branch is about to start an anyOf value, expand it.
        if (_liveCount == 1 &&
            _branches[0].TryGetAnyOfAlternatives(in p, out var alts) && alts.Length is > 1 and <= K)
        {
            var src = _branches[0];
            for (int i = 0; i < alts.Length; i++) _branches[i] = src.WithCurrentNode(alts[i]);
            _liveCount = alts.Length;
        }

        int w = 0;
        for (int i = 0; i < _liveCount; i++)
        {
            ref var b = ref _branches[i];
            if (!b.IsCharAllowed(c, in p)) continue;   // prune branches that reject the accepted char
            b.OnCharAdvanced(c, in p);
            if (w != i) _branches[w] = b;
            w++;
        }
        if (w > 0) _liveCount = w;   // never drop to 0 — parser already validated the char syntactically
    }

    public readonly bool IsComplete(in JsonCharParser p)
    {
        for (int i = 0; i < _liveCount; i++)
            if (_branches[i].IsComplete(in p)) return true;
        return false;
    }

    public readonly SchemaStateKey GetSchemaStateKey(in JsonCharParser p)
    {
        // Single-branch: identical key to pre-refactor. Multi-branch: fold all live branches.
        var k0 = _branches[0].GetSchemaStateKey(in p);
        if (_liveCount == 1) return k0;
        int h = k0.GetHashCode();
        for (int i = 1; i < _liveCount; i++) h = HashCode.Combine(h, _branches[i].GetSchemaStateKey(in p));
        return new SchemaStateKey(h, _liveCount, 0, 0);   // collision-resistant composite for the multi case
    }

    public void Reset() { _branches[0].Reset(); _liveCount = 1; }
}
```

Implement `BranchState.TryGetAnyOfAlternatives` (readonly): returns the `AnyOfNodeIndices` of the schema node that governs the value the parser is about to start (root value at `Start`/`ValueStart`, or the property value node just resolved). Return false otherwise. Implement `BranchState.WithCurrentNode(int)` returning a value copy with `_currentNodeIndex` set to the alternative and key/enum cursors reset.

> Note: when the wrapper anyOf branches are objects, each branch's `WithCurrentNode` points at that branch's object node; the existing `PushObject`/`FinishKeyString` logic then enforces that branch's properties/required/trie. The OR/prune across branches yields exact narrowing as `name`'s const and `arguments`' keys disambiguate.

- [ ] **Step 4: Run the multi-tool tests + full constraint regression**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests|FullyQualifiedName~SchemaCompilerTests" -v minimal`
Expected: PASS (multi-tool narrowing + all prior).

- [ ] **Step 5: Add the degradation test**

```csharp
[Fact]
public void MultiTool_AboveBranchCap_DoesNotCrash_AndStaysValidJson()
{
    var tools = new List<ToolDefinition>();
    for (int i = 0; i < 12; i++)
        tools.Add(new ToolDefinition($"tool_{i}", "x",
            """{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}"""));
    var c = new JsonSchemaConstraint(TestTokenizer, ToolCallSchemaBuilder.BuildForRequired(tools.ToArray()));
    Feed(c, "{\"name\":\"tool_3\",\"arguments\":{");          // must not throw
    Assert.True(AnyAllowed(c.GetAllowedTokens()));
}
```

- [ ] **Step 6: Run it**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~JsonSchemaConstraintTests.MultiTool_AboveBranchCap_DoesNotCrash_AndStaysValidJson" -v minimal`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs tests/DotLLM.Tests.Unit/Engine/Constraints/JsonSchemaConstraintTests.cs
git commit -m "feat(constraints): bounded parallel anyOf branch-narrowing for tool calls (#104)"
```

---

### Task 5: End-to-end BitNet gate + single-tool strict integration test

**Files:**
- Test: `tests/DotLLM.Tests.Integration/Engine/ConstrainedToolCallTests.cs` (create) — or extend the existing `RunToolChoiceTests` location if that is where #106 put them; match that file's env-gating pattern.
- (no src changes expected)

**Interfaces:**
- Consumes: the CLI `run --tools … --tool-choice required` path (#106) + Tasks 1–4.

- [ ] **Step 1: Single-tool strict in-process test (no model)**

Drive a `JsonSchemaConstraint` for the single-tool schema through a full valid object and assert termination:

```csharp
[Fact]
public void SingleTool_StrictArguments_RejectsUndeclaredKey_AndTerminates()
{
    string schema = ToolCallSchemaBuilder.BuildForFunction(new ToolDefinition(
        "get_weather","x","""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""));
    var c = new JsonSchemaConstraint(TestTokenizer, schema);
    Feed(c, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"");
    // Inside arguments, undeclared key must be impossible; only close is valid here.
    Assert.False(TokenAllowedStartingWith(c, ",\"q"));
    Feed(c, "}}");
    Assert.True(c.IsComplete());                          // fully-formed, self-terminated
}
```

- [ ] **Step 2: Run it**

Run: `dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~ConstrainedToolCallTests.SingleTool_StrictArguments_RejectsUndeclaredKey_AndTerminates" -v minimal`
Expected: PASS.

- [ ] **Step 3: Env-gated BitNet end-to-end (mirrors handoff)**

```csharp
[Fact]
public void BitNet_ToolChoiceRequired_ProducesValidSelfTerminatingCall()
{
    string? gguf = Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF");
    if (string.IsNullOrEmpty(gguf) || !File.Exists(gguf)) return;   // env-gated no-op
    // Run the same path the CLI uses: load model, single tool get_weather, --tool-choice required,
    // --repeat-penalty 1.3, greedy. Assert the decoded output parses to {"name":"get_weather",
    // "arguments":{"city": <string>}} and the generation stopped (EOS / IsComplete).
    // (Wire via the existing RunCommand/TextGenerator constrained path used by #106.)
}
```

Fill the body using the same constrained-generation entry point `RunToolChoiceTests` (#106) uses; assert `GenericToolCallParser` parses the output to a single `get_weather` call with a string `city` and that decoding terminated.

- [ ] **Step 4: Run env-gated (locally, with the BitNet GGUF, 3060 idle)**

Run: `DOTLLM_BITNET_GGUF=C:/Users/james/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/snapshots/a1f2f1c765812aa8af3f6eda4a313707064bba15/ggml-model-i2_s.gguf dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~ConstrainedToolCallTests" -v minimal`
Expected: PASS — output `{"name":"get_weather","arguments":{"city":"Tokyo"}}` and stops.

- [ ] **Step 5: Commit**

```bash
git add tests/DotLLM.Tests.Integration/Engine/ConstrainedToolCallTests.cs
git commit -m "test(constraints): strict single-tool + env-gated BitNet tool-call gate (#104)"
```

---

## Final integration

- [ ] Whole-branch review (opus) of the diff vs the design doc — confirm single-branch path is behaviour-identical, no allocations on the hot path, K-degradation logged.
- [ ] `dotnet build` (worktree) clean: 0 errors, 0 warnings.
- [ ] Targeted suites green: `--filter "FullyQualifiedName~ToolCallSchemaBuilderTests|FullyQualifiedName~JsonSchemaConstraintTests|FullyQualifiedName~SchemaCompilerTests"`.
- [ ] Update `docs/CONSTRAINED_DECODING.md` / `docs/TOOL_CALLING.md` with the strict-schema + anyOf-narrowing behaviour and the K=8 cap.
- [ ] Open PR → `dev`: "feat(constraints): strict order-independent tool-call decoding (#104)", body `Closes #104`, footer line. Confirm CI green (Build Debug/Release + Unit Tests; Integration's BitNet test is env-gated no-op on CI).

## Self-review notes

- **Spec coverage:** Piece 1 → Task 1; Piece 2 → Task 2; refactor → Task 3; Piece 3 narrowing + cache key + degradation → Task 4; end-to-end + single-tool strict + BitNet gate → Task 5. All design sections mapped.
- **Type consistency:** `BranchState` (Task 3) is consumed unchanged by Task 4; `ReachableTerminalBits(int)→ulong` (Task 2) used by Task 2 wiring; `TryGetAnyOfAlternatives`/`WithCurrentNode` defined and consumed in Task 4.
- **Test helpers:** `TestTokenizer`, `Feed`, `AnyAllowed`, `TokenAllowedStartingWith` must be added once to the test files if not already present, matching existing `JsonSchemaConstraintTests` conventions; Task 2 introduces them.
