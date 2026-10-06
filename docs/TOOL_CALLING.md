# Tool Calling — dotLLM

## Overview

Tool calling (function calling) enables models to invoke external tools by generating structured JSON that the caller can execute and feed back as context. This is the foundation for agentic workflows — the model decides *which* function to call and *what arguments* to pass, the runtime executes it, and the model incorporates the result.

dotLLM's tool calling integrates three subsystems: **chat templates** (formatting tool definitions into the prompt), **constrained decoding** (guaranteeing valid JSON for tool arguments), and **model-specific parsers** (extracting structured tool calls from model output).

## End-to-End Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Client  │────►│ Chat Template│────►│  Text Gen +  │────►│  Tool Call   │
│ (tools,  │     │  (formats    │     │  Constrained │     │  Parser      │
│ messages)│     │  tools into  │     │  Decoding     │     │  (extracts   │
│          │     │  prompt)     │     │  (optional)  │     │  ToolCall[]) │
└─────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                  │
     ┌────────────────────────────────────────────────────────────┘
     │  finish_reason: "tool_calls"
     ▼
┌─────────┐     ┌──────────────┐     ┌──────────────┐
│  Client  │────►│   Execute    │────►│ Add tool     │──── (loop back to
│ receives │     │   Tools      │     │ results to   │      Chat Template)
│ tool calls│    │   Locally    │     │ messages     │
└─────────┘     └──────────────┘     └──────────────┘
```

**Step by step:**

1. Request includes `tools` definitions (name, description, parameter JSON schema) and `tool_choice`.
2. `IChatTemplate.Apply(messages, { Tools = tools })` injects tool definitions into the prompt using the model's Jinja2 template.
3. If `tool_choice` is `required` or a specific function, `ToolCallSchemaBuilder` generates a JSON Schema and `JsonSchemaConstraint` guarantees valid output via constrained decoding.
4. Model generates text. `IToolCallParser.TryParse(output)` extracts structured `ToolCall[]`.
5. `ToolCallDetector` enriches `InferenceResponse` with parsed calls, sets `FinishReason.ToolCalls`.
6. Client executes tools, adds results as `tool` role messages with `ToolCallId`.
7. Template formats results; model generates the final response incorporating tool output.

## Core Types

### ToolDefinition

```
ToolDefinition:
  Name: string              // Function name (e.g., "get_weather")
  Description: string       // Human-readable description
  ParametersSchema: string  // JSON Schema for function parameters
```

### ToolCall

```
ToolCall:
  Id: string                // Unique call ID (e.g., "call_0")
  FunctionName: string      // Which function was called
  Arguments: string         // JSON string of arguments
```

### ToolChoice

Discriminated union controlling how the model selects tool calls. Follows the OpenAI API convention.

```
ToolChoice:
  Auto        // Model decides freely (default when tools present)
  None        // Don't call tools — text only
  Required    // Must call at least one tool
  Function    // Must call a specific function by name
```

| `tool_choice` | Constrained decoding | Parser runs | Use case |
|---------------|---------------------|-------------|----------|
| `Auto` | No | Yes — **model-family** parser | Model freely decides; detect tool calls post-hoc |
| `None` | No | No | Tools in context for reference only |
| `Required` | Yes — `anyOf` schema | Yes — **generic (markerless)** parser | Force a tool call; guaranteed valid JSON |
| `Function(name)` | Yes — single-tool schema | Yes — **generic (markerless)** parser | Force a specific function |

**Which parser parses constrained output** — see [Parsing Constrained Output](#parsing-constrained-output) below. Constrained output is *bare JSON*, never the model's envelope, so the marker-based parsers must not be used on it.

### ChatMessage (tool-related fields)

```
ChatMessage:
  Role: string           // "system" | "user" | "assistant" | "tool"
  Content: string        // Text content
  ToolCalls: ToolCall[]? // Assistant messages: tool invocations
  ToolCallId: string?    // Tool result messages: which call this answers
```

## IToolCallParser Interface

```
IToolCallParser:
  TryParse(generatedText) → ToolCall[]?   // Extract tool calls from output
  IsToolCallStart(text) → bool            // Detect partial tool call (streaming)
```

Models signal tool calls in different formats. Each parser handles one convention:

### Parser Implementations

| Parser | Marker | JSON Key | Models | Example Output |
|--------|--------|----------|--------|----------------|
| `LlamaToolCallParser` | `<\|python_tag\|>` | `name` + `parameters` | Llama 3.1+ Instruct | `<\|python_tag\|>{"name":"f","parameters":{...}}` |
| `HermesToolCallParser` | `<tool_call>`...`</tool_call>` | `name` + `arguments` | Hermes, Qwen tool-calling | `<tool_call>{"name":"f","arguments":{...}}</tool_call>` |
| `XmlToolCallParser` | `<tool_call>`...`</tool_call>` (delegates to Hermes) | `name` + `arguments` | SmolLM3 `xml_tools` branch, Qwen3/Hermes | `<tool_call>{"name":"f","arguments":{...}}</tool_call>` |
| `QwenXmlToolCallParser` | `<tool_call>`/`<function=NAME>`/`<parameter=KEY>` | XML (values are text) | Qwen3.5 / 3.6 / 3.8, Ornith, Qwen3-Coder | `<tool_call>\n<function=f>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>` |
| `Gemma4ToolCallParser` | `<\|tool_call>call:` | Gemma dict syntax (bare keys, `<\|"\|>` strings) | Gemma-4 | `<\|tool_call>call:f{city:<\|"\|>Paris<\|"\|>,n:3}<tool_call\|>` |
| `MistralToolCallParser` | `[TOOL_CALLS]` | `name` + `arguments` | Mistral Instruct | `[TOOL_CALLS][{"name":"f","arguments":{...}}]` |
| `PythonicToolCallParser` | Python call syntax | positional/kwargs → `arguments` | SmolLM3 (`python_tools` branch) | `[f(city="Tokyo")]` |
| `GenericToolCallParser` | None (bare JSON) | `name` + `arguments`/`parameters` | Fallback **and all constrained output** | `{"name":"f","arguments":{...}}` |

**Llama shapes (#771).** `LlamaToolCallParser` accepts everything Llama 3.x really emits after `<|python_tag|>`: `{"name","parameters"}`, Llama-3.2-1B's `{"type":"function","function":"get_weather","parameters":{...}}` (the `function` string *is* the name), the OpenAI envelope `{"type":"function","function":{"name","arguments"}}`, several calls as a JSON array or separated by `;`/whitespace, and the built-in-tool pythonic form `brave_search.call(query="...")`. The name/envelope normalization lives in `ToolCallJsonHelper.ParseSingle`, so every JSON-based parser benefits. Without the marker, only a response that is *entirely* tool-call JSON counts (prose quoting a schema does not).

**Qwen XML (#771)** follows llama.cpp's `common_chat_params_init_qwen3_coder`: the closing `</tool_call>` is optional (it is a stop sequence for Hermes, and the model often ends the turn first), a bare `<function=` block without `<tool_call>` is accepted, multiple blocks are parallel calls, text before the first call (a leading `</think>`, reasoning, prose) is ignored, and a `<function>` without its `</function>` is a truncated generation and is **not** reported (never execute half a call). The parser is a superset of Hermes: a `<tool_call>` body that is JSON is parsed as Hermes JSON. Each value has exactly one template newline stripped from each end (`<parameter=k>\nVALUE\n</parameter>`); a declared `string` parameter is kept verbatim, integer/number/boolean/object/array are parsed as JSON, and with no schema the text is JSON when valid, else a string. Because the server's `</tool_call>` stop sequence would cut a parallel-call completion after the first call, `ToolChoiceBinder` removes it for this parser.

**Gemma-4 (#776)** mirrors `common_chat_params_init_gemma4`: `call:NAME{key:value,...}` where strings are delimited by the `<|"|>` token and bare values are numbers/`true`/`false`/`null`, with nested objects and arrays. The reader is a recursive-descent scanner (string values may contain `{ } , :`); an unterminated argument object is not reported; `<eos>`/`<turn|>` text after the call is ignored. Separately, generation must *stop*: the GGUF declares `eos_token_id` = `<turn|>` (106) but a tool-call turn ends with `<eos>` (id 1), which used to run to `max_tokens` and leak `<eos><eos>...` into the response. `EndOfGenerationTokens` (used by `TextGenerator` and the batch scheduler) now stops on the declared EOS **plus** `<eos>`, `<end_of_turn>`, `<|eot_id|>`, `<|eom_id|>`, `<|im_end|>`, `<turn|>` when the vocabulary has an entry with exactly that text (found by scanning token text: Gemma-4's `<eos>` is **not** pre-split by `Encode`, so encoding the candidate would miss it; pinned by `EndOfGenerationRealTokenizerTests` against the real E4B vocabulary, and verified live: a tool turn now ends at 14 completion tokens instead of running to `max_tokens`), as llama.cpp's EOG set does. (`<|end|>` / `<|endoftext|>` are deliberately excluded: they delimit messages or pad in some families.)

### Argument type coercion

`IToolCallParser.TryParse(text, tools)` (default interface method) coerces each call's arguments to its tool's JSON Schema via `ToolArgumentCoercer`: `"17"` becomes `17` for `integer`/`number`, `"true"` becomes `true`, JSON text becomes an object/array, and a number/bool becomes a string where the schema says `string` (Gemma writes `zip:12345` bare). Union types (`type: [..]`, `anyOf`) are honoured, nested `properties`/`items` are recursed, a `string` in the union keeps the value verbatim, and anything that cannot be coerced is left exactly as the model wrote it. The server and CLI pass the request's tools; the schema-free `TryParse(text)` still works. The XML parser overrides the method because its wire format has no types at all.

### Key Normalization: `parameters` vs `arguments`

Llama models use `"parameters"` for function arguments; the OpenAI API and most other models use `"arguments"`. The shared `ToolCallJsonHelper` normalizes both to the `ToolCall.Arguments` field:

1. Try `"arguments"` key first.
2. Fall back to `"parameters"`.
3. If the value is a string (double-serialized JSON), detect and unwrap it.
4. Generate sequential call IDs (`call_0`, `call_1`, ...) when none provided.

### Parallel Tool Calls

All parsers support parallel tool calls:

- **Llama**: JSON array after `<|python_tag|>` — `[{call1}, {call2}]`
- **Hermes**: Multiple `<tool_call>` blocks
- **Mistral**: JSON array after `[TOOL_CALLS]`
- **Generic**: JSON array with multiple objects

### Graceful Failure

All parsers return `null` on malformed input — they never throw. This is critical because tool call detection runs on every generation when tools are present. Invalid JSON, missing `name` key, unbalanced brackets — all produce `null`, and the response is treated as normal text.

## Parser Auto-Detection

`ToolCallParserFactory.Create(Architecture, chatTemplate?)` selects the appropriate parser via a two-tier heuristic:

**Tier 1 — Template content (highest priority; ORDER MATTERS):**
```
Template contains "<|tool_call>"                   → Gemma4ToolCallParser
"python_tools" without "xml_tools"                 → PythonicToolCallParser
Template contains "<function=" AND "<parameter="   → QwenXmlToolCallParser   (before the next line!)
Template contains "<tool_call>"                    → XmlToolCallParser
Template contains "python_tag"        → LlamaToolCallParser
Template contains "[TOOL_CALLS]"      → MistralToolCallParser
```

**Tier 2 — Architecture fallback:**
```
Architecture.Llama            → LlamaToolCallParser
Architecture.Mistral          → MistralToolCallParser
Architecture.Qwen / QwenMoe   → HermesToolCallParser
Architecture.Qwen3MoeHybrid / Qwen3HybridDense → QwenXmlToolCallParser
Architecture.Gemma4           → Gemma4ToolCallParser
Architecture.SmolLM3          → XmlToolCallParser
Architecture.BitNet           → HermesToolCallParser
*                             → GenericToolCallParser
```

Template content takes priority because the template is the source of truth for the model's tool calling convention — the same architecture may have different fine-tunes with different conventions.

## Constrained Decoding for Tool Arguments

When `tool_choice` is `Required` or `Function(name)`, the model output is constrained to valid tool call JSON via the existing `JsonSchemaConstraint` infrastructure (Step 40).

### Schema Generation

`ToolCallSchemaBuilder` synthesizes a JSON Schema from `ToolDefinition[]`:

**Single function** (`BuildForFunction`):
```json
{
  "type": "object",
  "properties": {
    "name": {"const": "get_weather"},
    "arguments": {<tool's parameter schema>}
  },
  "required": ["name", "arguments"],
  "additionalProperties": false
}
```

**Multiple functions** (`BuildForRequired`, ≤ `SchemaTracker.MaxParallelBranches` = 8 tools):
```json
{
  "anyOf": [
    {"type": "object",
     "properties": {"name": {"const": "get_weather"}, "arguments": {<get_weather schema>}},
     "required": ["name", "arguments"], "additionalProperties": false},
    {"type": "object",
     "properties": {"name": {"const": "get_time"}, "arguments": {<get_time schema>}},
     "required": ["name", "arguments"], "additionalProperties": false}
  ]
}
```

Each tool is a fully-closed `anyOf` branch with its own `name` const and parameter schema. `SchemaTracker` enforces these branches via **bounded parallel branch-narrowing** (#104): up to `K=8` branches are tracked in lockstep; a character is allowed if *any* live branch allows it, and branches are pruned as the `name` const value and the `arguments` keys disambiguate — so per-tool argument schemas *are* enforced, with keys in **any order** (`name`-first or `arguments`-first both converge). Above `K` tools, `BuildForRequired` degrades to a closed `name`-`enum` flat object (`additionalProperties:false`, `name` still constrained, args a permissive object) rather than failing. Single-tool schemas (and `BuildForFunction`) use `const` with full parameter-schema enforcement. All emitted object schemas carry `additionalProperties:false`, injected recursively by `ToolCallSchemaBuilder.Harden`.

**Parallel calls** (`BuildForParallelCalls`):
```json
{
  "type": "array",
  "items": {<same flat object schema as BuildForRequired>}
}
```

The `argumentsKey` parameter handles the Llama `"parameters"` vs standard `"arguments"` difference — it's set based on the parser type (Llama parsers use `"parameters"`, all others use `"arguments"`).

### Constraint Integration

The generated schema feeds directly into the existing `ResponseFormat.JsonSchema` → `JsonSchemaConstraint` pipeline:

```
ToolDefinition[] → ToolCallSchemaBuilder.BuildForRequired()
                 → ResponseFormat.JsonSchema { Schema = generatedSchema }
                 → JsonSchemaConstraint (SchemaCompiler, SchemaTracker)
                 → TokenMask per decode step
```

`SchemaCompiler` supports `anyOf`, `const`, nested objects, and `enum`; `SchemaTracker` adds the bounded parallel `anyOf` branch-narrowing (#104) that enforces the *correct* per-tool branch. This means a constrained tool call is **structurally guaranteed** to be a valid JSON object conforming to exactly one tool's parameter schema (no leaked/duplicate keys, self-terminating). Note: structural validity is guaranteed regardless of model, but full *termination* (the model emitting the closing braces + EOS) still depends on model capability — a weak base can run to `MaxTokens` on an unbounded string value.

### Parsing Constrained Output

**Constrained output is bare JSON. It is parsed by `GenericToolCallParser`, regardless of model family.**

`ToolCallSchemaBuilder` + `JsonSchemaConstraint` emit a *JSON object* — `{"name": …, "arguments": …}` — and nothing else. The constraint machinery is a JSON-schema tracker; it has no way to force the model's envelope tokens (`<tool_call>`, `<|python_tag|>`, `[TOOL_CALLS]`) around that object, and the schema deliberately contains no such literals. Feeding constrained output to a marker-based parser therefore *always* yields `null` and the tool call is silently lost (issue #325).

So the wrapper format is owned by the **constraint layer** for `Required`/`Function` (it is: no wrapper), and every consumer must respect that:

```csharp
// argumentsKey comes from the MODEL parser (Llama → "parameters"), computed BEFORE the swap
string argumentsKey = modelParser is LlamaToolCallParser ? "parameters" : "arguments";
responseFormat = new ResponseFormat.JsonSchema { Schema = ToolCallSchemaBuilder.BuildForRequired(tools, argumentsKey) };

// then swap the parser used for OUTPUT
parser = ToolCallParserFactory.ForToolChoice(toolChoice, modelParser);
```

`ForToolChoice` returns `GenericToolCallParser` for `Required`/`Function` and the model parser unchanged for `Auto`/`None`. It is the single place this rule lives — `dotllm run`, `dotllm chat`, the integration tests, and (once server-side `tool_choice` enforcement lands) the server all route through it.

The swap also matters for **streaming**: `StreamingToolCallAccumulator` uses `IsToolCallStart`, so with a marker parser on constrained output the suppression never triggers and the raw tool-call JSON leaks to the user's console.

The marker parsers deliberately do **not** fall back to bare JSON. Under `tool_choice=auto` the absence of a marker is the signal "this is not a tool call"; a fallback would misparse prose that happens to contain `{"name": …}`.

### `tool_choice=auto` — No Constraint

When `tool_choice` is `auto`, no schema constraint is applied. The model generates freely and the parser detects tool calls post-hoc. This is because the model may choose to produce text instead of calling a tool — constraining the output format would prevent text-only responses.

## Post-Generation Detection

### ToolCallDetector

Static utility for non-streaming use:

```csharp
var response = generator.Generate(prompt, options);
response = ToolCallDetector.DetectToolCalls(response, parser);

if (response.FinishReason == FinishReason.ToolCalls)
{
    // response.ToolCalls contains parsed tool calls
}
```

Returns the original response unchanged if no tool calls are found (reference equality — no allocation).

### StreamingToolCallAccumulator

For streaming use, accumulates token text and detects tool call boundaries:

```csharp
var accumulator = new StreamingToolCallAccumulator(parser);

await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
{
    bool suppress = accumulator.Append(token.Text);
    if (!suppress)
        Console.Write(token.Text);  // show text to user
    // else: this text is part of a tool call, buffer it
}

// After generation completes:
var toolCalls = accumulator.TryParseCompleted();
```

**Behavior:**
- Before a tool call marker is detected: `Append()` returns `false` — text flows to user.
- Once `IsToolCallStart()` triggers: `Append()` returns `true` for all subsequent text — caller suppresses output.
- After generation: `TryParseCompleted()` extracts the full tool call(s).

## Chat Template Integration

The Jinja2 chat template engine (Step 16) handles tool definitions natively:

### Template Context

`JinjaChatTemplate.BuildContext()` exposes tools in the standard HuggingFace format:

```python
# Available in Jinja template:
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {<parsed JSON schema as dict>}
    }
  }
]
```

Tool calls in assistant messages:
```python
message.tool_calls = [
  {
    "id": "call_0",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{...}"}
  }
]
```

Tool result messages:
```python
message.role = "tool"
message.content = '{"temperature": 22}'
message.tool_call_id = "call_0"
```

### Template Examples

**ChatML with tools** (Qwen, Hermes):
```jinja
{% for message in messages %}
<|im_start|>{{ message.role }}
{% if message.tool_calls %}
{% for tc in message.tool_calls %}
<tool_call>{{ tc | tojson }}</tool_call>
{% endfor %}
{% else %}
{{ message.content }}
{% endif %}
<|im_end|>
{% endfor %}
{% if tools %}
Available tools: {{ tools | tojson }}
{% endif %}
```

**Llama 3.1** uses `<|python_tag|>` and formats tools as a structured system prompt section. The `tojson` filter serializes tool definitions for embedding.

### Rendering fidelity matters for tool round trips (#771)

The harness saw Llama-3.1-8B "ignore the tool result" on the second turn. The parser was not at fault (the call parsed before and after). Diagnosis found two evaluator deviations from Jinja2 (and therefore from llama.cpp's minja and HF `apply_chat_template`); both are fixed and pinned by `JinjaLlama31ToolRoundTripTests` against the byte output of reference Jinja2 3.1.6 and llama.cpp `/apply-template`. They are real rendering bugs affecting every Llama-3.x prompt, but they were **not isolated as the sole cause** of the WARN (each fix was not toggled separately), and llama.cpp on the identical, byte-matching prompt also gave a turn-2 answer that did not use the result — the 8B Q4_K_M round trip is marginal and sensitive to numerics:

- **Trailing newline.** Jinja2's default `keep_trailing_newline=False` drops one trailing newline of the template source. Llama-3.x's GGUF template ends `{%- endif %}\n`, so dotLLM prompts ended `<|start_header_id|>assistant<|end_header_id|>\n\n\n` (an extra blank line the model never trained on). `JinjaChatTemplate` now drops exactly one.
- **`is iterable` on strings.** In Jinja2 a string is iterable. Llama-3.1's tool branch is `{% if message.content is mapping or message.content is iterable %}{{ message.content | tojson }}`, so a string tool result renders as a quoted, escaped JSON string; dotLLM used to render it raw.

Result with both fixes (dotLLM, greedy): the 8B round trip now answers from the tool result ("...17 degrees Celsius and there is light rain") where before it said "The actual output of the function call is not provided". Llama-3.2-1B still re-calls the tool on turn 2 (model behaviour; llama.cpp cannot even parse the 1B's first-turn shape and returns HTTP 500).

### Known gaps

- **Hermes-family parallel calls are still truncated by the server's `</tool_call>` stop sequence** (Qwen3-4B-Instruct, SmolLM3, BitNet): generation stops after the first call's closing tag, so a second `<tool_call>` block is never produced. Only the Qwen XML parser has the stop removed (`ToolChoiceBinder`). Qwen-XML parallel calls are covered by unit tests, not yet by a real-model run.

## Multi-Turn Tool Use

A complete multi-turn conversation with tool calling:

```
messages = [
  {role: "system", content: "You are helpful."},
  {role: "user", content: "What's the weather in Paris?"},
]

// Turn 1: Model calls a tool
→ assistant: {tool_calls: [{id: "call_0", name: "get_weather", args: {"location":"Paris"}}]}
→ finish_reason: "tool_calls"

messages += [
  {role: "assistant", content: "", tool_calls: [...]},
  {role: "tool", content: '{"temp":22,"condition":"sunny"}', tool_call_id: "call_0"},
]

// Turn 2: Model uses tool result
→ assistant: "The weather in Paris is 22°C and sunny."
→ finish_reason: "stop"
```

Each turn re-applies the full chat template to the accumulated message history, including tool call and tool result messages.

## CLI Usage

```bash
# Interactive chat with tools
dotllm chat model.gguf --tools @tools.json

# Force tool calling (constrained decoding)
dotllm chat model.gguf --tools @tools.json --tool-choice required

# Force a specific function
dotllm chat model.gguf --tools @tools.json --tool-choice get_weather
```

### Tools JSON Format

`--tools` accepts a JSON array (inline or `@file`). Both flat and OpenAI-style formats supported:

**Flat format:**
```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
]
```

**OpenAI format:**
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "parameters": { ... }
    }
  }
]
```

### REPL Commands

| Command | Action |
|---------|--------|
| `/tools` | Display available tool definitions |
| `/clear` | Reset conversation (preserves system prompt) |
| `/exit` | Quit |

When tool calls are detected in the REPL:

```
>>> What's the weather in Paris?

Tool calls detected:
  [call_0] get_weather({"location": "Paris"})
Result for get_weather (Enter to skip):
[tool]>>> {"temperature": 22, "condition": "sunny"}

Generating response with tool results...
The weather in Paris is 22°C and sunny.
```

## Key Files

| File | Purpose |
|------|---------|
| `Core/Configuration/ToolChoice.cs` | `ToolChoice` discriminated union |
| `Tokenizers/ToolCall.cs` | `ToolCall` record |
| `Tokenizers/ToolDefinition.cs` | `ToolDefinition` record |
| `Tokenizers/ChatMessage.cs` | `ChatMessage` with `ToolCalls`, `ToolCallId` |
| `Tokenizers/IToolCallParser.cs` | Parser interface |
| `Tokenizers/ToolCallParsers/LlamaToolCallParser.cs` | Llama 3.1+ parser |
| `Tokenizers/ToolCallParsers/HermesToolCallParser.cs` | Hermes/Qwen parser |
| `Tokenizers/ToolCallParsers/MistralToolCallParser.cs` | Mistral parser |
| `Tokenizers/ToolCallParsers/GenericToolCallParser.cs` | Fallback parser |
| `Tokenizers/ToolCallParsers/ToolCallJsonHelper.cs` | Shared JSON extraction + normalization |
| `Tokenizers/ToolCallParsers/XmlToolCallParser.cs` | SmolLM3 / Hermes XML envelope |
| `Tokenizers/ToolCallParsers/QwenXmlToolCallParser.cs` | Qwen3-Coder XML (`<function=..><parameter=..>`): Qwen3.5/3.6/3.8, Ornith |
| `Tokenizers/ToolCallParsers/Gemma4ToolCallParser.cs` | Gemma-4 `<\|tool_call>call:name{k:<\|"\|>v<\|"\|>}<tool_call\|>` |
| `Tokenizers/ToolCallParsers/ToolArgumentCoercer.cs` | Schema-driven argument type coercion |
| `Engine/Samplers/StopConditions/EndOfGenerationTokens.cs` | EOS + vocabulary end-of-turn tokens (llama.cpp EOG set) |
| `Tokenizers/ToolCallParsers/PythonicToolCallParser.cs` | SmolLM3 Pythonic call syntax |
| `Tokenizers/ToolCallParsers/ToolCallParserFactory.cs` | Auto-detection factory + `ForToolChoice` (constrained-output rule) |
| `Engine/Constraints/ToolCallSchemaBuilder.cs` | Schema generation from tool definitions |
| `Engine/ToolCallDetector.cs` | Post-generation detection |
| `Engine/StreamingToolCallAccumulator.cs` | Streaming boundary detection |
| `Engine/InferenceResponse.cs` | `ToolCalls` property, `FinishReason.ToolCalls` |
| `Cli/Commands/ChatCommand.cs` | `--tools`, `--tool-choice`, REPL integration |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parser per model family | 4 implementations + factory | Models use fundamentally different formats — no single parser handles all |
| Template heuristic over config | Scan template for markers | Template is source of truth; same architecture can have different tool conventions |
| Schema constraint only for required/function | No constraint for auto | Model must be free to produce text instead of tool calls with `auto` |
| Wrapper format under constraint | None — bare JSON, parsed by `GenericToolCallParser` via `ToolCallParserFactory.ForToolChoice` | A JSON-schema constraint cannot emit envelope tokens; centralising the rule stops call sites forking (#325) |
| Post-generation detection (not in TextGenerator) | `ToolCallDetector` is caller responsibility | Keeps engine minimal; CLI and Server both use it differently |
| `arguments` normalization | `ToolCallJsonHelper` handles both keys | Llama uses `parameters`, everyone else uses `arguments` — normalize once |
| Sequential call IDs | `call_0`, `call_1`, ... | Simple, deterministic; models rarely provide their own IDs |
| Graceful failure | Return `null`, never throw | Parser runs on every generation — exceptions would be disruptive |

## Reference Implementations

- **llama.cpp** — Tool calling via chat template Jinja rendering, `<|python_tag|>` detection
- **vLLM** — `ToolCallParser` with model-specific handlers, guided decoding for tool args
- **Ollama** — Tool support via chat template integration, JSON extraction
- **OpenAI API** — `tool_choice`, `tools`, `finish_reason: "tool_calls"` conventions

## Known Limitations

Tracked under Wave 8 ([issue #121](https://github.com/kkokosa/dotLLM/issues/121)).

- **Streaming tool calls arrive whole, not as argument fragments.** `/v1/chat/completions` with `stream: true` no longer leaks tool-call markup as `delta.content` (#771): from the moment the model-family parser's `IsToolCallStart` recognises a call, tokens are held back (prose before the call still streams), and the parsed calls are emitted in the **final** chunk's `delta.tool_calls` with `finish_reason: "tool_calls"`. Held-back text that does not parse into a call (truncated generation) is delivered as one trailing `delta.content` chunk, never swallowed. What is NOT done: incremental `delta.tool_calls` fragments (partial `arguments` strings) — a call is reported only once complete. A marker split across tokens (`<tool` | `_call>`) leaks its first fragment, because detection runs on the accumulated text. The Anthropic stream behaves the same (`tool_use` blocks after the text block).
- **`tool_choice` other than `auto` is not enforced.** The server parses `tool_choice` from the request but does not currently constrain decoding for `"required"` or specific-function values. The short-term plan is to reject unsupported `tool_choice` values with HTTP 400; long-term is constraint-driven enforcement via `ToolCallSchemaBuilder` + `JsonSchemaConstraint`.
