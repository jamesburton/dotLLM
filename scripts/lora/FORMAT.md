# Task-LoRA train==serve format spec

This pins the prompt format used for **both** training (PyTorch/PEFT) and serving (dotLLM `--lora`).
The trainer formats every example with `tokenizer.apply_chat_template`, which has been **verified
to render identically to dotLLM's `JinjaChatTemplate`** (see the parity verdict below). Train and
serve therefore see the same string — the single biggest correctness requirement for tool-use.

Base model for the reference: `Qwen/Qwen3-4B-Instruct-2507`. Fixtures live in `scripts/lora/fixtures/`.

## 1. Instruction format (no tools)
System + user turns rendered with the Qwen3 ChatML template, `add_generation_prompt=True`, then the
assistant completion. The trainer masks loss to the assistant completion only (see `masking.py`).

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant completion}<|im_end|>
```

## 2. Tool-use format
Tools are injected by the template into the system turn as a `<tools>...</tools>` block, and the
assistant emits the call inside `<tool_call>...</tool_call>` (Hermes/ChatML), which dotLLM's
`HermesToolCallParser` parses. Golden rendering (from `fixtures/qwen3_tooluse_reference.txt`, prompt
portion):

```
<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the weather in Paris?<|im_end|>
<|im_start|>assistant
```

**Assistant target string the trainer must emit** (exact form the Hermes parser round-trips):
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

## 3. Parity verdict (U0)
**PASS** — `tests/DotLLM.Tests.Unit/Tokenizers/ChatTemplates/QwenToolFormatVerificationTests.cs`
(commit `9300c14`): dotLLM's `JinjaChatTemplate.Apply` produces output **bit-identical** (modulo
trailing whitespace) to `tokenizer.apply_chat_template`. dotLLM wraps each flat `ToolDefinition`
into the `{"type":"function","function":{...}}` object the Qwen3 template expects, and
`JinjaEvaluator.ToJson` uses Python-`json.dumps`-matching separators (`", "` / `": "`). The Hermes
parser round-trips the assistant `<tool_call>` to `ToolCall(FunctionName="get_weather", Arguments={"city":"Paris"})`.

## 4. The rule
**The trainer formats every example via `tokenizer.apply_chat_template` — never a hand-built prompt
string** — so it stays locked to what dotLLM renders at serve time. Tool definitions passed to the
template use the full `{"type":"function","function":{...}}` objects (as the HF tokenizer expects);
dotLLM reconstructs the same from its flat `ToolDefinition`.

## 5. Tool-use rendering caveat — **RESOLVED in U2 (Phase A)**

The U0 parity test (§3) validates the **serving** tool-render path: dotLLM injects tools via
`ChatTemplateOptions.Tools` → the template's `{%- if tools %}` branch → the `<tools>` block, identical
to HF `apply_chat_template(..., tools=...)`.

**Resolution (commit `0750cf8`):** `scripts/lora/tooluse_render.py` implements:
- `extract_tools(row)` — reads `row['tools']` (a JSON string already in
  `{"type":"function","function":{...}}` format; confirmed from
  `NousResearch/hermes-function-calling-v1 / glaive_func_calling`).
- `render_tooluse(tok, row)` — strips the dataset's embedded `<tools>` boilerplate from
  the system turn, then calls `tok.apply_chat_template(prompt_msgs, tools=tools,
  add_generation_prompt=True)` so the `<tools>` block is **template-generated** —
  bit-identical to dotLLM's serving render.  Supervises the first `<tool_call>` gpt
  turn as the labeled completion.

`train_task_lora.py` now routes `--task tooluse` through `render_tooluse` instead of
the generic `format_row`+`render` path (instruction/coding unchanged).
6/6 pytest tests pass (`scripts/lora/tests/test_tooluse_render.py`).

**Train==serve is now verified for all three tasks**, including the tool-use `<tools>` block.

### Minor harness notes (also U2)
- `--split` and `--max-examples` both default to 2000 and act as independent caps — set them together to
  avoid a silently-smaller dataset than intended.
- The training loop is batch-1, deterministic order, no shuffle/grad-accum — adequate for a smoke harness
  (smoke converged 3.78→0.22); a real U2 run should add shuffling + gradient accumulation.
