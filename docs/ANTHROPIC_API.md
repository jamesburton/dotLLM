# Anthropic Messages API — dotLLM

dotLLM's server exposes an **Anthropic-compatible Messages API** alongside the
OpenAI-compatible surface ([SERVER.md](SERVER.md)). Clients and SDKs written for
the Anthropic Messages API (`anthropic` Python/TypeScript SDKs, anything that
targets `POST /v1/messages`) can point at a running dotLLM server unchanged.

The engine, tokenizer, chat-template, sampler and tool-calling pipeline are
shared verbatim with the OpenAI endpoints — this layer only reshapes the wire
format. Implementation: `MessagesEndpoint`, `AnthropicConverter`, and the
`Anthropic*` DTOs in `DotLLM.Server`.

Reference: <https://docs.anthropic.com/en/api/messages>

> **Fork-only feature (#448, completed by #449).** This surface exists on this
> fork only; it is not part of upstream `kkokosa/dotLLM`. #449 added
> `POST /v1/messages/count_tokens`, the `anthropic-version` / `anthropic-beta`
> request headers, `tool_choice` enforcement, a mid-stream `error` event and
> input-side `thinking` / `redacted_thinking` blocks. dotLLM does not *emit*
> extended-thinking blocks — see [Extended thinking](#extended-thinking).

## Request headers

| Header | Behaviour |
|---|---|
| `anthropic-version` | Honoured. `2023-06-01` (what every official SDK sends) and `2023-01-01` are accepted; any other value → `400` `invalid_request_error`. A **missing** header is accepted and treated as `2023-06-01` — a deliberate deviation from the real API, which requires it, because dotLLM's server is a local development tool driven with `curl` as often as with an SDK. |
| `anthropic-beta` | Accepted and ignored, including values dotLLM has never heard of. Repeated headers and comma-joined values are both understood. No beta feature is honoured today; this must never be a `400`, because SDK helpers attach betas of their own. |
| `x-api-key` | Accepted. This is the header the official SDKs authenticate with, and it is what a client sends instead of `Authorization: Bearer`. dotLLM's server performs **no authentication at all** (see [SERVER.md § Security](SERVER.md)); `HeaderApiKeyResolver` only reads the header to partition rate-limit buckets. Nothing is validated, and no key is required. |

The version header is validated **before** the requested model is activated, so
a request pinned to an unimplemented version cannot trigger a model load.

## Endpoints

### `POST /v1/messages`

Primary endpoint. Accepts the Anthropic Messages request format; supports both
non-streaming (JSON) and streaming (named SSE events).

**Request body**:
```json
{
  "model": "llama-3-8b-q4_k_m",
  "max_tokens": 256,
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
    {"role": "user", "content": "What's the weather?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["\n\nHuman:"],
  "tools": [
    {"name": "get_weather", "description": "Get weather",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
  ],
  "tool_choice": {"type": "auto"},
  "stream": false
}
```

- `max_tokens` is **required** (per the Anthropic spec). Missing/`<= 0` → `400`.
- `system` is a top-level string **or** an array of `{"type":"text","text":"..."}`
  blocks; it becomes a leading `system` message in the chat template.
- Each message `content` is a string **or** an array of content blocks
  (`text`, `tool_use`, `tool_result`).
- `tool_choice`: `{"type":"auto"}`, `{"type":"any"}` (→ required),
  `{"type":"none"}`, or `{"type":"tool","name":"..."}`, and it is **enforced**:
  `any`/`tool` constrain decoding to a tool-call JSON schema (and parse the
  result with the markerless parser, since the constraint emits a bare JSON
  object rather than the model's `<tool_call>` envelope); `none` suppresses
  tool-call detection entirely. A forced tool that is not present in `tools`
  → `400`.
- `messages[].role` must be `user` or `assistant`; any other role → `400`.
  (The top-level `system` field is the only way to set a system prompt.)
- Unsupported / unknown content block types (`image`, `document`,
  `server_tool_use`, a typo) are **rejected with a `400`** rather than silently
  dropped: a dropped `image` block would have the model answer about a picture it
  never received. `thinking` and `redacted_thinking` blocks are accepted and
  dropped, so an extended-thinking transcript can be replayed unchanged.
- `thinking` (the request field) is accepted and ignored.
- `model` selects the resident model, exactly as on the OpenAI surface: it is
  passed to `ServerState.EnsureActiveAsync`, which activates an already-resident
  model, lazily reloads one that idled out, or loads a new one by path / HF repo
  id. A name that resolves to nothing → `400`. An Anthropic SDK's default
  `model` (e.g. `claude-sonnet-4-...`) therefore has to be overridden with the
  loaded model's id. The response `model` echoes the model that actually served
  the request, not the requested alias.
- `lora_adapter` is **not** honoured by this endpoint — per-request adapter
  selection is not implemented on the server yet (the OpenAI surface does not
  honour it either). Unknown fields are ignored, not rejected.

**Response** (non-streaming):
```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "llama-3-8b-q4_k_m",
  "content": [{"type": "text", "text": "It's sunny."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 15, "output_tokens": 8}
}
```

When tool calls are detected, `content` contains `tool_use` blocks and
`stop_reason` is `"tool_use"`:
```json
{
  "content": [
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather",
     "input": {"city": "Paris"}}
  ],
  "stop_reason": "tool_use"
}
```

### `POST /v1/messages/count_tokens`

Returns the number of input tokens the *same body* would consume on
`POST /v1/messages`, without generating anything. The body is the Messages
request minus `max_tokens` (which this route does not accept as required —
`MessageCountTokensParams` in the official SDK has no such field):

```json
{"model": "llama-3-8b-q4_k_m",
 "system": [{"type": "text", "text": "You are helpful."}],
 "messages": [{"role": "user", "content": "Hello!"}],
 "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}
```

```json
{"input_tokens": 47}
```

The count is the tokenizer's count over the **templated** prompt — system
prompt, full history and tool definitions included — produced by the same
`BuildPrompt` helper `/v1/messages` uses, so

```
count_tokens(body).input_tokens == messages.create(body).usage.input_tokens
```

holds by construction. The route needs only the tokenizer and the chat template,
so (unlike `/v1/messages`) it is not refused for masked text-diffusion models.
Validation is the same as `/v1/messages`, minus the `max_tokens` requirement.

## Streaming

With `"stream": true`, the response is a sequence of **named** SSE events
(`event: <type>\ndata: <json>\n\n`):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"It's"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":15,"output_tokens":8}}

event: message_stop
data: {"type":"message_stop"}
```

If generation fails **after** `message_start` has been written, the status line
is already on the wire, so the failure is reported the way the Anthropic stream
protocol reports it — a named `error` event, after which the stream ends without
`message_delta`/`message_stop`:

```
event: error
data: {"type":"error","error":{"type":"api_error","message":"..."}}
```

The official SDK turns this into an `APIStatusError`; without it the client sees
a truncated stream and reports a connection/parse error instead of the failure.
A client disconnect is *not* reported this way — a cancelled request is not a
server error.

Every frame's `data.type` equals its `event:` name (the SDK dispatches on the
event name and only fills `type` in when the payload omits it).

Tool calls detected during streaming are emitted after the text block closes, as
additional `tool_use` content blocks (`content_block_start` →
`content_block_delta` with `input_json_delta` → `content_block_stop`) at index
`1+`, and `stop_reason` becomes `"tool_use"`.

## Mapping reference

| Anthropic field | dotLLM engine |
|-----------------|---------------|
| `system` (string/array) | leading `system` `ChatMessage` |
| message `content` string | `ChatMessage.Content` |
| `text` block | concatenated into `ChatMessage.Content` |
| `tool_use` block (assistant) | `ChatMessage.ToolCalls` (`ToolCall`) |
| `tool_result` block (user) | separate `tool`-role `ChatMessage` keyed by `tool_use_id` |
| `tools[].input_schema` | `ToolDefinition.ParametersSchema` |
| `tool_choice` `auto`/`any`/`none`/`tool` | `ToolChoice.Auto`/`Required`/`None`/`Function` |
| `stop_sequences` | `InferenceOptions.StopSequences` |

| dotLLM `FinishReason` | Anthropic `stop_reason` |
|-----------------------|-------------------------|
| `Stop` (EOS / template stop) | `end_turn` |
| `Stop` (caller `stop_sequences` matched) | `stop_sequence` (+ `stop_sequence` field) |
| `Length` | `max_tokens` |
| `ToolCalls` | `tool_use` |

## Errors

Errors use the Anthropic envelope:
```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: field required"}}
```

| Condition | HTTP | `error.type` |
|-----------|------|--------------|
| No model loaded and no `model` given | 400 | `invalid_request_error` |
| Unknown / unloadable `model` | 400 | `invalid_request_error` |
| Empty `messages`, missing/invalid `max_tokens`, bad `role`/`content` kind | 400 | `invalid_request_error` |
| Unsupported/unknown content block type (`image`, `document`, …) | 400 | `invalid_request_error` |
| Unknown `anthropic-version` | 400 | `invalid_request_error` |
| `tool_choice` naming a tool absent from `tools` | 400 | `invalid_request_error` |
| Generation fails mid-stream (after `message_start`) | — | `api_error` in a named `error` SSE event |
| Prompt exceeds context window | 400 | `invalid_request_error` |
| Loaded model is a masked text-diffusion model | 400 | `invalid_request_error` |
| Model became unavailable after activation succeeded | 503 | `api_error` |

Note the ordering: model activation runs *before* the readiness check, so a bare
server answers `400 "No model loaded and no model specified"` (matching the
OpenAI surface's message) rather than `503`. The `503` branch is a backstop for
the model going away between activation and use.

A body that is not valid JSON, or that does not bind to the request shape, is
rejected by ASP.NET's model binding before the handler runs — that produces a
bare `400` with no Anthropic envelope. Same as the OpenAI surface.

## Limitations

- **Streaming tool calls** are detected post-generation (the engine parses tool
  calls from the full output), so `tool_use` blocks are emitted at the end of the
  stream rather than incrementally — matching the OpenAI streaming endpoint's
  post-hoc detection.
- **`image` / multimodal content blocks** are not supported (rejected, not dropped).

### Extended thinking

`thinking` and `redacted_thinking` blocks are accepted on input and **dropped**
when the prompt is built: they carry no content dotLLM can replay, and the real
API treats them as opaque. The `thinking` request field is accepted and ignored.

dotLLM never *emits* `thinking` content blocks or `thinking_delta` /
`signature_delta` stream deltas. Doing so would require splitting a model's
`<think>` span out of the token stream (the close tag straddles token
boundaries), which no dotLLM surface does today — it is a separate piece of
work, not a wire-format detail.
- **Masked text-diffusion models** are refused on this route with a `400`. The
  diffusion decode path is only wired into `/v1/chat/completions`; refusing is
  deliberate, so a diffusion checkpoint cannot silently produce autoregressive
  output here.
- **Rate limiting does not cover this route.** `RateLimitMiddleware.IsMeteredPath`
  is a path allowlist naming `/v1/chat/completions`, `/v1/completions` and
  `/v1/embeddings`, so `/v1/messages` bypasses the per-API-key limiter. The
  handler already reports actual token usage to a lease when one exists, so
  adding the path to that allowlist is the whole fix — tracked with the
  rate-limit header work in #452.
- The same single-request serialization, prompt caching, and validation rules as
  the OpenAI endpoints apply (see [SERVER.md](SERVER.md)).
