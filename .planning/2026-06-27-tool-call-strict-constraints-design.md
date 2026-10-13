# Strict, order-independent tool-call constraints (#104)

**Date:** 2026-06-27
**Issue:** kkokosa/dotLLM#104 (filed on jamesburton/dotLLM)
**Branch:** `issue/104-strict-tool-constraints` (worktree `.claude/worktrees/strict-tools`, off `origin/dev`)
**Status:** design approved (2026-06-27)

## Problem

`run --tool-choice required/function` (#103/#106) forces the OUTER tool-call skeleton
(`{"name":"get_weather","arguments":{`) but does **not** strictly enforce the nested
`arguments` object, `required`, `additionalProperties:false`, or object termination. On weak /
quantised bases (BitNet b1.58) this leaks garbage inside `arguments` and never closes the JSON.
Capable bases self-terminate and mask the gap.

Observed (BitNet I2_S, `--tool-choice required`):
- base: `{"name": "get_weather", "name\n\n\n…` — a 2nd `name` key allowed.
- +tooluse adapter: `{"name": "get_weather", "arguments":{">"  : { "weather": {…` — arguments not
  schema-enforced.

## Root cause (three independent gaps)

The `SchemaTracker` mask engine already enforces `required`, `additionalProperties:false`,
property-name tries, nested object schemas, and enum/const string tries. The leaks come from:

1. **Under-constrained schemas from `ToolCallSchemaBuilder`.**
   - Multi-tool path uses `"arguments":{"type":"object"}` — fully permissive: any key leaks, never
     forced to close.
   - Single-tool path passes the tool's `ParametersSchema` verbatim; per JSON-Schema semantics that
     is *open* (`additionalProperties` defaults true) unless the author set it, so extra keys leak
     and there is no forced termination.

2. **A genuine `SchemaTracker` / `PropertyNameTrie` bug — emitted-branch dead-end.** After emitting
   `"name"`, the model may still start `"n…` (a valid trie prefix) whose only completion (`name`) is
   already emitted under `additionalProperties:false` → every continuation is rejected → **empty
   mask** → garbage / non-termination. Matches the base-model `{"name":…,"name\n\n` evidence.

3. **`anyOf` overapproximation.** Multi-tool cannot correlate `name`→that tool's `arguments` schema:
   `SchemaTracker` collapses `anyOf` to a type-union and loses per-branch structure (acknowledged
   TODO in `SchemaTracker.IsValueStartCharAllowed`).

## Goal

`--tool-choice required`/`function` yields a **fully valid, self-terminating** JSON tool call —
correct nested `arguments`, no leaked keys, exactly one tool's schema enforced — for single- and
multi-tool, **keys in any order** (per user decision: support `arguments` opening before `name`).

## Decisions (locked)

- **Approach: extend `SchemaTracker`** (not GBNF/PDA). Reuses the optimized bucketing/caching/mask
  machinery and the already-correct object key-set (any-order / each-once) semantics.
- **Order-independent** via bounded parallel branch tracking (not a name-first restriction).
- **K = 8** strict parallel-branch cap, with **graceful degradation** beyond K to today's
  type-union overapproximation (still syntactically valid + name-enum constrained), logged.

## Design — three pieces

### Piece 1 — `ToolCallSchemaBuilder` emits closed schemas

- New recursive `Harden(JsonElement) → string` that, for every object schema, injects
  `additionalProperties:false` when absent, recursing through `properties`, `items`, `anyOf`,
  `$defs`. **Keeps declared `required`** (does NOT force optional→required; optional omission stays
  valid JSON).
- `BuildForFunction` (single tool): unchanged shape `{name:const, arguments:<params>}` but
  `arguments` uses `Harden(NormalizeParametersSchema(...))`; wrapper already has
  `required:[name,arguments]`, `additionalProperties:false`.
- `BuildForRequired` (multi tool): replace `enum`-name + permissive-args with **`anyOf` of per-tool
  branches**:
  `{"anyOf":[ {"type":"object","properties":{"name":{"const":"Tᵢ"},"arguments":<hardened paramsᵢ>},
  "required":["name","arguments"],"additionalProperties":false}, … ]}`.
  Safe now because Piece 3 enforces branches.
- `BuildForParallelCalls`: array `items` = the `anyOf` wrapper (unchanged otherwise).

### Piece 2 — emitted-branch dead-end fix (`PropertyNameTrie` + `SchemaTracker`)

- Add `ReachableTerminalBits[node]` to `PropertyNameTrie`: for each trie node, the OR of the
  property-bit indices of all terminals reachable from it. Computed once at trie construction.
- In `SchemaTracker`, gate key generation on the emitted-set:
  - `IsObjectOpenCharAllowed` / `IsObjectNextKeyCharAllowed` (`"` open): allow only if
    `(ReachableTerminalBits[root] & ~emitted) != 0` (some property still un-emitted) — or additional
    properties allowed.
  - `IsTrieCharValid(c)`: allow `c` only if the child node still reaches an un-emitted terminal:
    `(ReachableTerminalBits[child] & ~emitted) != 0` — or additional properties allowed.
  - `IsTrieTerminal` keeps its existing already-emitted rejection.
- Result: the model can never enter a key prefix that has no valid (un-emitted) completion → no
  empty masks.

### Piece 3 — `anyOf` parallel-branch narrowing (`SchemaTracker` core)

- **Refactor** the current single-position state (node stack, emitted bitmask stack, key buffer,
  key-length, trie cursors, enum cursor, array indices, `inKeyString`/`inEnumString`,
  `currentNodeIndex`) into a nested value-type `BranchState` (all `InlineArray`-based, copyable by
  value). All current per-char logic moves to operate on a `ref BranchState`.
- `SchemaTracker` holds an `InlineArray<BranchState>` of capacity **K=8** plus `_liveCount`.
  **Single-branch common path is behaviour- and cost-identical to today.**
- **Fork:** when a value position resolves to an `anyOf` node, expand the current branch into one
  `BranchState` per alternative (each seeded at its alternative node). Capped at K; if the alt count
  > K, do NOT fork — fall back to the existing union behaviour for that subtree (degradation).
- **Per-char combine:**
  - `IsCharAllowedBySchema(c) = OR over live branches of branch.IsCharAllowed(c)`.
  - `OnCharAdvanced(c)`: advance every live branch that allowed `c`; **prune** the rest; if all would
    die the char was never allowed (guarded by the OR above). Collapse to single-branch
    representation when `_liveCount == 1`.
  - `IsComplete = parser.IsComplete && (any live branch complete)`.
- **Narrowing falls out naturally:** the `name` const value prunes branches whose const-trie rejects
  a char; `arguments` is validated against each surviving branch's own subschema in parallel, so
  name-first and args-first both converge.
- **Cache key:** extend `SchemaStateKey` to fold the live branch set (e.g. `_liveCount` + a stable
  hash/than-equality over the live `BranchState`s' `(currentNodeIndex, emitted, triePos)` tuples).
  Keep it collision-free for the common single-branch case (identical to today's key).

## Components & boundaries

| Unit | Responsibility | Depends on |
|------|----------------|-----------|
| `ToolCallSchemaBuilder.Harden` | Close object schemas recursively | `System.Text.Json` |
| `PropertyNameTrie.ReachableTerminalBits` | Per-node reachable-terminal bitmask | trie build |
| `BranchState` (struct) | One NFA thread: today's single-position logic | `CompiledSchema`, `JsonCharParser` |
| `SchemaTracker` | Bounded set of `BranchState`; fork / OR / prune; cache key | `BranchState` |
| `JsonSchemaConstraint` | Unchanged orchestration (mask build, bucketing, LRU) | `SchemaTracker` |

## Testing

- **`ToolCallSchemaBuilderTests`**: `Harden` injects `additionalProperties:false` recursively
  (nested objects, array items, `$defs`); single-tool keeps `const`+hardened args; multi-tool emits
  per-tool `anyOf` (not enum); declared `required` preserved, optional not forced.
- **`SchemaCompilerTests`**: `ReachableTerminalBits` correctness on multi-property tries.
- **`JsonSchemaConstraintTests`**:
  - emitted-branch dead-end: after `{"name":…,` the mask is never empty; a 2nd `name` is impossible.
  - single-tool strict: undeclared `arguments` keys rejected; required forced; terminates (only `}`).
  - multi-tool narrowing: **name-first** and **args-first** orderings both end enforcing the matched
    tool's schema; wrong-tool args rejected.
  - degradation: > K branches → no crash, still valid JSON + name constrained.
  - regression: existing single-schema tests byte-identical (single-branch path unchanged).
- **End-to-end** (env-gated `DOTLLM_BITNET_GGUF`, plain `[Fact]` + early-return per repo pattern;
  RTX 3060 idle; `--filter`, never full suite): `run <bitnet i2_s> --tools @tools.json --tool-choice
  required --repeat-penalty 1.3` → `{"name":"get_weather","arguments":{"city":"Tokyo"}}` and stops.

## Performance

Parallelism is live only while >1 branch survives — the `arguments` window before `name` is read.
Models emit `name` first, so steady state is `K_eff = 1` (today's cost). `BranchState` sized so the
K=8 worst-case clone stays bounded; first-char bucketing still eliminates >95% of tokens before any
clone. Benchmark single-tool constraint mask-build before/after to confirm no regression on the
common path.

## Risks & mitigations

- **Cache-key growth** → key folds only live branches; single-branch key identical to today.
- **`BranchState` clone cost** → K=8 cap + 1-branch common path + bucketing.
- **Refactor regressing the tested single-schema path** → single-branch must be behaviour-identical;
  existing constraint tests are the gate; land Piece 2 (and the refactor) with green tests before
  Piece 3 narrowing.

## Out of scope / follow-ups

- Non-string `const`/`enum` (numbers, bools) character-level enforcement (existing TODO).
- Recursive `$ref` schemas (compiler already rejects).
- `> K` tool strict enforcement (degrades by design).

## Workflow

Issue-driven: issue #104, branch `issue/104-strict-tool-constraints`, PR → `dev`, commits ref
`(#104)`, footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
Subagent-driven execution (fresh implementer per piece + per-piece review + final whole-branch
review). Land order: Piece 1 → Piece 2 → refactor → Piece 3, tests green at each step.
