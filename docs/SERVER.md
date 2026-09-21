# Server — dotLLM

## Overview

ASP.NET Minimal API server providing OpenAI-compatible endpoints. Wires together the inference engine, tokenizer, chat templates, scheduler, and telemetry.

## Endpoints

### `POST /v1/chat/completions`
Primary chat endpoint. Accepts OpenAI-compatible request format.

**Request body**:
```json
{
  "model": "llama-3-8b-q4_k_m",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true,
  "stop": ["\n\n"],
  "tools": [...],
  "tool_choice": "auto",
  "response_format": {"type": "json_schema", "json_schema": {...}},
  "logit_bias": {"1234": -100},
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "top_n_sigma": 1.0,
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_penalty_last_n": 0,
  "dry_sequence_breakers": ["\n", ":", "\"", "*"],
  "n": 1,
  "stream_options": {"include_usage": true},
  "parallel_tool_calls": false
}
```

**`stream_options.include_usage`** (#450) — when true, the stream emits one extra chunk before
`data: [DONE]` carrying `usage` with an **empty `choices` array**. SDKs match on exactly that
shape to close out their token accounting, so it is load-bearing rather than cosmetic. The
pre-existing `finish_reason` chunk keeps its own `usage`/`timings` (a dotLLM extension the web UI
reads) — `include_usage` adds a chunk, it does not change one. Supported on all three streaming
paths: chat, diffusion chat, and `POST /v1/completions`.

**`parallel_tool_calls`** (#450) — `false` means the assistant emits at most one tool call per
turn. Nothing constrains the model during decode, so the cap is applied to the detected calls on
the way out, on both the streaming and non-streaming paths. Absent/`true` is OpenAI's default
(parallel calls allowed).

**Accepted and ignored**: `user`, `store`, `service_tier`, `reasoning_effort`, `metadata`. These
name concepts this server has no equivalent for, and a client that always sends them must never
get a 400. Genuinely unknown fields are tolerated too (STJ source-gen skips unmapped members) —
declaring these makes the intent explicit and guards against a future strict-DTO pass.

Also accepted (not shown above): `top_k`, `min_p`, `repetition_penalty` — see [SAMPLING.md](SAMPLING.md)
for the full parameter reference, including the DRY/top-nσ/logit-bias/frequency/presence-penalty
processors these fields drive. `POST /v1/completions` accepts the same sampling parameter set.

> **Adapter selection**: there is no `lora_adapter` request field (it is not part of the OpenAI
> surface either, and the server ignores unknown fields silently). LoRA serving is roadmap step 47
> and is not implemented: the engine has the `IAdapterManager` abstraction and
> `InferenceRequest.AdapterId`, but there is no implementation, no adapter admin endpoint, and no
> way to select an adapter per request. Until then, serve an adapted model by merging the adapter
> into the weights offline and loading the merged model — via `--model` at startup or
> `POST /v1/models/load`. See [LORA.md](LORA.md) for the planned design.

**Response** (non-streaming):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama-3-8b-q4_k_m",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
}
```

**Streaming**: Server-Sent Events (SSE). Each chunk:
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

### `POST /v1/completions`
Raw completion (no chat template). Same sampling parameters. Input is `prompt` (string) instead of `messages`.

### `POST /v1/messages`, `POST /v1/messages/count_tokens` (Anthropic-compatible, fork-only — #448/#449)

Anthropic Messages API endpoint, served alongside the OpenAI surface so that
`anthropic` SDK clients can talk to dotLLM unchanged. Top-level `system`,
string-or-block message `content`, `max_tokens` (required), `stop_sequences`,
`tools`/`tool_choice`, and event-based streaming SSE (`message_start`,
`content_block_*`, `message_delta`, `message_stop`). Reuses the same model
residency, chat template, scheduler, sampler and tool-call parser as
`/v1/chat/completions`; only the wire format differs.

Errors use the Anthropic envelope, not this surface's `{"error": "..."}`:
`{"type":"error","error":{"type":"invalid_request_error","message":"..."}}`.

Two caveats worth knowing here rather than in the detail doc:
- `/v1/messages` is **not** in `RateLimitMiddleware`'s metered-path allowlist,
  so it currently bypasses per-API-key rate limiting (see [Rate Limiting](#rate-limiting)).
- A masked text-diffusion model is refused on this route with a `400`; use
  `/v1/chat/completions` for those.

`POST /v1/messages/count_tokens` returns `{"input_tokens": N}` for the same body
without generating, computed from the same templated prompt `/v1/messages` bills.
`anthropic-version` is honoured (unknown value → `400`), `anthropic-beta` is
accepted and ignored, and `x-api-key` is accepted — dotLLM performs no
authentication (see [Security](#security)).

Full reference: **[ANTHROPIC_API.md](ANTHROPIC_API.md)**.

### `POST /v1/embeddings`
Extract embedding vectors from text (#451).

> **Backend coverage: CPU only.** The pooled hidden state comes from
> `IEmbeddingModel.ForwardHidden`, which only the CPU `TransformerModel` implements. When a
> Vulkan or CUDA model is loaded the endpoint returns **501 Not Implemented** with a message
> naming the model type, rather than silently returning an unvalidated vector. A GPU path is a
> follow-on, not a blocker.

**Request**

| field | type | notes |
|---|---|---|
| `input` | string \| string[] \| int[] \| int[][] | Required. A flat int array is **one** pre-tokenised sequence; a nested one is many. Pre-tokenised ids are range-checked against the vocabulary. |
| `model` | string | Optional. Activates that model (same semantics as `/v1/chat/completions`). |
| `encoding_format` | `"float"` (default) \| `"base64"` | `base64` is the raw little-endian float32 payload, base64-encoded — what the OpenAI SDK's numpy path decodes. |
| `pooling` | `"last"` \| `"mean"` \| `"cls"` | dotLLM extension, mirrors llama.cpp's `--pooling`. Omit to use the model default. |
| `normalize` | bool, default `true` | dotLLM extension. `true` is L2 / Euclidean, matching llama.cpp's `--embd-normalize 2` default and OpenAI's unit-norm vectors. |
| `dimensions` | — | **Rejected with 400.** dotLLM returns the model's full hidden size; there is no Matryoshka truncation. |

**Response**

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.1, -0.2, "..."]}],
 "model": "smollm2-135m-instruct",
 "usage": {"prompt_tokens": 21, "total_tokens": 21}}
```

`data[i]` corresponds to `input[i]`. `usage.prompt_tokens` is the sum of the per-item token counts.

**Implementation.** Each input item is its own forward pass with positions `0..n-1` (the CPU
forward has no per-sequence attention mask, so sequences are not packed), stopping after the final
output norm and before the LM head — the tensor llama.cpp names `result_norm` and assigns to
`res->t_embd`, which is what its own pooling operates on.

**Concurrency.** The request holds *both* locks that guard the model: the server request gate
(`ServerState.ExecuteAsync`, against the direct-generator path) and, when a continuous-batch
scheduler is active, `ContinuousBatchSchedulerService.AcquireModelAsync` — the scheduler drives
forward passes on the same model from its own run loop, deliberately outside the request gate,
because batching rather than serialising is the point of it. The gate alone is not enough: the
model's scratch buffers *and its compute thread pool* are shared mutable state, and an embedding
taken alongside a generation without the lease crashes the process
(`CountdownEvent … below zero` from `ComputeThreadPool`). `AcquireModelAsync` makes the run loop
finish the step it is on and block before the next, so the embedding interleaves *between* steps.
Cost to the scheduler is one uncontended semaphore per forward pass.

**Pooling default.** Precedence is: explicit `pooling` → the checkpoint's GGUF
`{arch}.pooling_type` → `last`. The GGUF value is llama.cpp's raw `llama_pooling_type` enum
(`0=none, 1=mean, 2=cls, 3=last, 4=rank`) and is mapped value-for-value. The final fallback is a
**deliberate deviation** from llama.cpp, whose `hparams.pooling_type` defaults to `NONE` when the
key is absent: `NONE` means one vector per token and is not representable in an OpenAI embeddings
response. `last` is the right default for a causal decoder — the last token is the only position
that has attended to the whole sequence — and is what llama.cpp's own tooling makes you pass
(`--pooling last`) to embed a generative model. A checkpoint that *declares* `none` or `rank` is
honoured rather than rewritten: the request fails with a 400 telling the caller to pass `pooling`
explicitly.

**Known gaps.** There is no cap on the number of input items (OpenAI's is 2048) — a large batch
holds the model lock for the whole request and stalls generation meanwhile. Batched embedding
forwards, a GPU path, and `pooling: none` (one vector per token, via a non-OpenAI response shape)
are all follow-ons.

**Correctness.** Anchored against llama.cpp, not against itself: reference vectors are captured
from `llama-server --embeddings` on the same GGUF (`tests/scripts/capture-llamacpp-embeddings.ps1`,
committed with full provenance) and compared by cosine similarity in
`EmbeddingLlamaCppParityTests`. See that test's remarks for the measured correct-vs-broken
separation the tolerance is derived from.

### `GET /v1/models`
Lists every **resident** model — the active one plus any stashed-but-loaded models (#369):
```json
{"data": [
  {"id": "llama-3-8b-q4_k_m", "object": "model", "is_active": true,
   "idle_seconds": 4.2, "keep_alive_seconds": 300, "expires_in_seconds": 295.8, "size_bytes": 4900000000}
]}
```
`expires_in_seconds` is omitted when the model's keep-alive is negative (pinned, never auto-unloads).

### `GET /v1/models/{id}`
Retrieves one model object — what the OpenAI SDK's `client.models.retrieve()` calls (#450). The
route is a catch-all (`/v1/models/{**id}`) because ids are HuggingFace repo ids and contain `/`;
the literal `/v1/models/{available,load,inspect}` routes are more specific and still win.
Resolution goes through the same list the collection endpoint returns, so retrieve can never
disagree with list — including on a bare server, where the configured-but-unloaded model id
retrieves rather than 404s. An unknown id returns `404` with
`{"error": {"type": "not_found_error", "code": "model_not_found", "param": "model", ...}}`.

## Model Keep-Alive / Idle-Unload / Multi-Model Residency (#369)

Ollama-parity daemon lifecycle: idle models unload automatically, and — when configured — more than
one model can be resident (loaded and instantly servable) at once.

### Keep-alive

Every loaded model tracks a **keep-alive** duration:

| Value | Meaning |
|-------|---------|
| *(unset)* | Server default — `ServerOptions.KeepAliveSeconds`, default **300s** (5 min, matches ollama). |
| `0` | Unload after this use (checked on the next idle-sweep tick, not synchronously — see below). |
| positive N | Unload after N seconds idle. |
| negative | Never auto-unload (pin the model resident). |

Set it per-request via `"keep_alive": <seconds>` on `POST /v1/chat/completions`, `POST /v1/completions`,
or `POST /v1/models/load`; each sets that model's override going forward (subsequent requests reuse
the last-set override unless they specify their own). A background sweep
(`ServerOptions.IdleSweepInterval`, default 5s) evicts models past their keep-alive — including the
active one, but only when it isn't mid-generation (an in-flight request is never interrupted; the
sweep just retries next tick). A later request against an idled-out model **lazily reloads it**
(from the same resolved path/options) rather than requiring an explicit `/v1/models/load` call again.

### Multi-model residency

`ServerOptions.MaxResidentModels` (default **1**) bounds how many models can be loaded
concurrently, counting the active one. Default-compatible: at `1`, loading a new model always
evicts the previous one immediately — exactly the original single-model hot-swap behavior.
Set it `> 1` (plus optionally `ServerOptions.ResidentMemoryBudgetBytes`, default 0 = unlimited byte
budget, only the count bounds residency) to hold several models resident. When a new load would
exceed the budget, the least-recently-used stashed model is evicted first (simple LRU).

Route requests to a specific resident model with the standard OpenAI `"model"` field on chat/completion
requests — reactivating an already-resident model is a cheap in-memory field-swap (no GGUF reload),
while a not-yet-resident model triggers a normal load (evicting LRU stashed models first if needed).

CLI flags: `--keep-alive <seconds>`, `--max-resident-models <n>`, `--resident-memory-budget <bytes>`.

### Concurrency scoping

Requests to different resident models are **still serialized** through the same request gate as
single-model mode — `ContinuousBatchScheduler`/`ContinuousBatchSchedulerService` have no multi-model
dispatch story, and building one was out of scope for #369. The win is reload-cost elimination
(µs-scale reactivation vs. seconds-scale disk reload), not concurrent cross-model execution. See
`docs/perf/ISSUE_369_MODEL_KEEPALIVE.md` (if still present) for the full scoping rationale.

### `POST /v1/tokenize` (extension)
**Request**: `{"text": "Hello world", "model": "..."}`
**Response**: `{"tokens": [9906, 1917], "token_strings": ["Hello", " world"], "count": 2}`

Not in OpenAI spec but widely expected for prompt engineering and billing estimation.

### `POST /v1/detokenize` (extension)
**Request**: `{"tokens": [9906, 1917], "model": "..."}`
**Response**: `{"text": "Hello world"}`

## response_format Processing

The `response_format` field maps to constrained decoding:

| `response_format.type` | Action |
|------------------------|--------|
| `"text"` | No constraint |
| `"json_object"` | `JsonConstraint` — guarantees valid JSON |
| `"json_schema"` | `JsonSchemaConstraint` compiled from `response_format.json_schema` |

The constraint is passed to the sampler pipeline and applied at every decode step.

## Tool Calling Flow

When `tools` are provided in the request:

1. **Prompt formatting**: `IChatTemplate.Apply(messages, options: { Tools = tools })` includes tool definitions in the prompt using the model's expected format.
2. **Generation**: Model generates response. If structured output is configured for tool calls, the JSON arguments are constrained to match the tool's parameter schema.
3. **Detection**: `IToolCallParser.TryParse(output)` checks if the output contains tool calls.
4. **Response**: If tool calls detected, return with `finish_reason: "tool_calls"` and structured `tool_calls` array.
5. **Continuation**: Client sends tool results as `tool` role messages. Server applies chat template again and generates final response.

## Prompt Caching

Multi-turn conversations benefit from prompt caching — reusing KV-cache state from previous turns to skip redundant prefill.

### How It Works

1. After each generation, `TextGenerator` stores the KV-cache and its full token sequence (prompt + generated) in a `PrefixCache`.
2. On the next request, the new prompt's token IDs are compared element-wise against cached entries to find the longest common prefix.
3. On cache hit: the cached KV-cache is reused, `CurrentLength` is truncated to the matched prefix, and only the new suffix tokens are prefilled.
4. On cache miss: a fresh KV-cache is allocated as usual.

This dramatically reduces time-to-first-token (TTFT) for multi-turn chat, where each turn's prompt shares a long prefix with the previous turn.

### Configuration

Prompt caching is **enabled by default** in both `chat` and `serve` commands.

| Flag | Default | Description |
|------|---------|-------------|
| `--no-prompt-cache` | `false` | Disable prompt caching |
| `--prompt-cache-size` | 1 (chat) / 4 (serve) | Maximum number of cached sessions (LRU eviction) |

### API

Cached token statistics are included in the `timings` field of streaming SSE responses:

```json
{
  "timings": {
    "prefill_time_ms": 2.1,
    "cached_tokens": 847,
    "prompt_tokens": 892
  }
}
```

### `POST /v1/cache/clear`

Clears all cached KV-cache sessions. Called automatically by the Chat UI when the conversation is cleared. Useful for freeing memory or resetting state.

**Response**: `{"status": "cleared"}`

### Scope

- CPU `SimpleKvCache` only. QuantizedKvCache and GPU caches fall back to no caching.
- Cache is cleared on model swap/reload.
- No session-based routing — single global LRU cache, serialized by the request gate.

## Rate Limiting

Per-API-key admission controls built on `System.Threading.RateLimiting`. Off by default — when no `RateLimit` configuration is present (or `Enabled: false`) the middleware short-circuits and adds zero overhead. When configured, the middleware sits between CORS and endpoint mapping and inspects every request to a metered path (see § What gets metered — everything under `/v1/` except an explicit exemption list).

Code lives in `src/DotLLM.Server/RateLimiting/`:

| File | Role |
|------|------|
| `RateLimitConfig` | Configuration record: `Enabled`, `DefaultPolicy`, `ApiKeys`, `EstimatedCompletionTokensFallback`. Loaded from `ServerOptions.RateLimit`. |
| `RateLimitPolicy` | Per-key cap: `RequestsPerMinute`, `TokensPerMinute`, `MaxConcurrent`, `Priority`, `QueueTimeout`. |
| `IApiKeyResolver` / `HeaderApiKeyResolver` | Identity surface. Default reads `X-API-Key`, falls back to `Authorization: Bearer <key>`, then `"anonymous"`. **This is NOT authentication** — host apps wiring real auth should replace the resolver. |
| `PriorityConcurrencyGate` | Per-key concurrency limiter ordered by `RequestPriority` (highest wins, FIFO within a priority class). |
| `RateLimitManager` | Owns the three limiters per resolved key. `TryAcquireAsync(key, estimatedTokens, ct)` returns a bundled `RateLimitLease` on admission or a `LimiterKind` + `RetryAfter` on rejection. |
| `RateLimitMiddleware` | Plugs into the ASP.NET pipeline; stashes the lease on `HttpContext.Items` so endpoints can call `ReportActualTokens` after generation. |

### Three independent limiters

A request is admitted only when **all three** policies admit. The first one that rejects wins — subsequent limiters are not touched:

1. **Requests/min** — `System.Threading.RateLimiting.TokenBucketRateLimiter`, replenishing at `RequestsPerMinute / 60` permits per second. `0` or negative disables.
2. **Tokens/min** — same shape. Reservation is `prompt_tokens_estimate + max_tokens` (or `EstimatedCompletionTokensFallback` when `max_tokens` is unspecified). `0` or negative disables.
3. **Max concurrent in-flight** — custom `PriorityConcurrencyGate`. Excess waiters park in a priority queue (negated `RequestPriority` + monotonic sequence number for FIFO tiebreak) and are released in priority order as slots free. Waiters that exceed `QueueTimeout` get a 429.

The `Retry-After` header is sourced from `MetadataName.RetryAfter` on the BCL limiter where available, falling back to `60s` for requests/tokens and `QueueTimeout` for concurrency.

### Configuration

The configuration record lives at `ServerOptions.RateLimit` and serializes from the standard ASP.NET options pipeline (or any path the host wires up):

```json
{
  "RateLimit": {
    "Enabled": true,
    "EstimatedCompletionTokensFallback": 256,
    "DefaultPolicy": {
      "RequestsPerMinute": 60,
      "TokensPerMinute": 100000,
      "MaxConcurrent": 5,
      "Priority": "Normal",
      "QueueTimeout": "00:00:05"
    },
    "ApiKeys": {
      "key-premium": {
        "RequestsPerMinute": 600,
        "TokensPerMinute": 1000000,
        "MaxConcurrent": 50,
        "Priority": "High"
      },
      "key-background-batch": {
        "RequestsPerMinute": 10,
        "TokensPerMinute": 50000,
        "MaxConcurrent": 1,
        "Priority": "Low"
      }
    }
  }
}
```

### Priority levels

`RequestPriority` is `Low | Normal | High | Critical`. Priority affects **admission queueing under concurrency contest**, not the requests/min or tokens/min token buckets (those are per-key and don't queue across requests). When the concurrency cap is saturated, queued waiters are released in priority order — a `High`-tier request that arrives *after* a queued `Low`-tier request jumps ahead.

> Cross-request preemption of in-flight sequences is a scheduler concern — it lives in Step 59 (Advanced scheduling) and is out of scope for this step. The middleware never interrupts a generation in progress.

### Token-budget true-up

The tokens-per-minute limiter charges `prompt_estimate + max_tokens` upfront so callers cannot bypass the cap by omitting `max_tokens`. After generation the endpoint calls `RateLimitMiddleware.GetLease(httpContext)?.ReportActualTokens(promptTokens + completionTokens)`. The difference between reservation and actuals is recorded for accounting; the BCL `TokenBucketRateLimiter` does not currently expose a public refund API, so refunds are best-effort and tracked internally for diagnostics. The reservation is the cap; charges above it are ignored.

### Response on rejection

```
HTTP/1.1 429 Too Many Requests
Retry-After: 12
X-RateLimit-Limiter: Tokens
x-request-id: 0HN7...
x-ratelimit-limit-tokens: 6000
x-ratelimit-remaining-tokens: 0
x-ratelimit-reset-tokens: 60
Content-Type: application/json

{"type":"error","error":{"message":"Rate limit exceeded (tokens-per-minute). Retry in 12s.","type":"rate_limit_error","param":null,"code":"tokens-per-minute"}}
```

| Header | Meaning |
|--------|---------|
| `Retry-After` | Seconds until the limiter can admit. Driven by the BCL limiter metadata where available. |
| `X-RateLimit-Limiter` | Which of the three limiters rejected (`Requests`, `Tokens`, `Concurrency`). Useful for client backoff decisions. |

## SDK-facing error envelope and observability headers (#452)

Every error response is `{"type": "error", "error": {"message", "type", "param", "code"}}`. The
official OpenAI and Anthropic SDKs parse this envelope to classify a failure; the flat
`{"error": "<string>"}` this server used to emit left `.type`, `.code` and `.param` unreachable, so
a 429 was indistinguishable from a 400 to anything reading the body. `param` and `code` are always
present, as explicit `null`s when unknown, matching OpenAI. The top-level `"type": "error"`
discriminator is what Anthropic's envelope requires and is inert for OpenAI clients, so one type
serves both surfaces. Error types in use: `invalid_request_error`, `rate_limit_error`,
`not_found_error`, `api_error`.

`ResponseHeadersMiddleware` (registered unconditionally, and *outside* the limiter so the headers
also land on its 429 short-circuit) emits:

| Header | Meaning |
|--------|---------|
| `x-request-id` | Correlation id. A sane inbound value is echoed; otherwise the connection's trace identifier is used. Values over 128 chars, or containing control characters, are replaced rather than reflected. |
| `openai-processing-ms` | Wall-clock milliseconds in the pipeline. Written from `Response.OnStarting`, so it is absent on a stream that started before generation finished. |
| `x-ratelimit-limit-requests` / `-remaining-requests` / `-reset-requests` | Requests-per-minute budget. Omitted entirely when that limiter is not configured — advertising a limit of 0 would make a well-behaved SDK back off against a server that is not limiting it. |
| `x-ratelimit-limit-tokens` / `-remaining-tokens` / `-reset-tokens` | Tokens-per-minute budget, same omission rule. `reset` is seconds until the bucket refills to its ceiling. |

The budget is partitioned with the **same `IApiKeyResolver` the limiter uses** — a host that
registers its own (see § Authentication note) gets headers for the right bucket. The limiter
re-stamps the `x-ratelimit-*` values after it acquires, so a success response reports the budget
including its own request rather than the state one request ago.

Inbound `OpenAI-Organization`, `OpenAI-Project`, `OpenAI-Beta` and `anthropic-beta` name concepts
this server has no equivalent for. Nothing inspects them: they are accepted and ignored, never a
400.

The deterministic headers are written *before* the inner pipeline runs. That is deliberate — the
SSE endpoints start the response on their first flush, and headers cannot be added after that.

### Authentication note

`HeaderApiKeyResolver` exists only so rate-limit buckets can be partitioned per caller. dotLLM still has no built-in authentication — see § Security. Host applications wiring real auth (OAuth, JWT, mTLS) should register their own `IApiKeyResolver` implementation that returns the authenticated principal's stable ID. The rate-limit machinery is transport-independent and will bucket on whatever opaque string you return.

### What gets metered

**Everything under `/v1/` is metered unless it is explicitly exempt.** The exemptions are
`/v1/models`, `/v1/lora`, `/v1/prompt-cache`, `/v1/cache`, `/v1/config`, `/v1/tokenize` and
`/v1/detokenize` (matched on segment boundaries, so `/v1/models/{id}` is covered by
`/v1/models`). One known inexactness: `POST /v1/prompt-cache/{id}` *does* prefill through the
model, but is exempt because it was unmetered before the list was inverted — exempting it
preserves behaviour rather than asserting it is free. Non-`/v1/` paths — `/health`, `/ready`, `/props`, the chat UI and its assets —
are never metered. These are probes, control-plane operations, or static asset serving, and
consume no inference budget.

This is deliberately an **exemption list, not an allowlist**. It used to name the three paths that
*were* metered, which meant every new generative endpoint shipped unmetered by omission with
nothing failing when someone forgot — and the list had already drifted, naming `/v1/embeddings`
(not yet built) while `/v1/messages` would have bypassed the limiter entirely. An unmetered path
can never return 429, so its configured limits are simply unenforceable. Inverted, the failure mode
is safe: forgetting to classify a new route over-meters a control-plane endpoint (visible,
harmless) instead of silently leaving a hole in the limiter. Add a route to the exemption list only
when it genuinely does not run the model.

## Warm-up

At server startup, before accepting requests:

```csharp
if (options.Warmup.Enabled)
{
    // Trigger JIT compilation of hot paths
    var dummyTokens = tokenizer.Encode("The quick brown fox");
    for (int i = 0; i < options.Warmup.Iterations; i++)
        await engine.GenerateAsync(dummyTokens, maxTokens: 16);

    // Pre-load CUDA kernels, cuBLAS handles
    // Pre-compute RoPE tables, tokenizer trie
}
```

Configuration: `WarmupOptions { Enabled, DummyPromptLength, Iterations }`.

Ensures first real request doesn't pay JIT compilation or CUDA kernel loading penalties.

## Health & Readiness

- `GET /health` — Returns 200 when server is running.
- `GET /ready` — Returns 200 only after warm-up completes and model is loaded. Used by load balancers.

## Security

**dotLLM's server is a development/local tool.** It has no authentication, no TLS, and permissive CORS. Do not expose it to the internet without a reverse proxy.

### Binding

The server binds to `localhost` by default. To expose externally, pass `--host 0.0.0.0` — but only behind a reverse proxy (nginx, Caddy, Traefik) that provides TLS and authentication.

### Authentication

No built-in auth. For network-exposed deployments, configure your reverse proxy to require `Authorization: Bearer <key>` headers.

### CORS

Default policy is permissive (`AllowAnyOrigin`) for local Chat UI development. For production, restrict origins via your reverse proxy.

### Dangerous Endpoints

- `POST /v1/models/load` — loads arbitrary GGUF files from disk
- `POST /v1/config` — changes sampling parameters

These are designed for the local Chat UI workflow and must not be internet-exposed.

## Concurrency

The server has two execution paths and picks per-request:

1. **Continuous-batch scheduler path (default for paged-KV serving)**. When `--paged` is on (the default for `serve`) and no speculative-decoding draft model is loaded, `ServerStartup` constructs a `ContinuousBatchSchedulerService` per loaded model and starts its `RunLoopAsync` on a background task tied to `IHostApplicationLifetime.ApplicationStopping`. `/v1/chat/completions` and `/v1/completions` route non-streaming requests through `EnqueueAsync` — multiple concurrent requests pipeline through a single `IModel.ForwardBatch` dispatch per scheduler iteration. The startup log prints `Continuous-batch scheduler active` when this path is engaged.
2. **Single-request gate path (fallback)**. Streaming requests, LoRA-adapter requests, logprob-capturing requests, and any backend without a paged KV-cache factory (CUDA, hybrid GPU, quantized KV) keep using the original `SemaphoreSlim(1, 1)` gate via `ServerState.ExecuteAsync`. Requests serialize FIFO. The startup log prints `Single-request mode — requests processed sequentially` when this is the only path.

### Scheduler tuning

`ContinuousBatchSchedulerOptions`:

Set the whole section via `ServerOptions.Scheduler` (bind it from `appsettings.json`, or pass `--scheduler-fairness` on the CLI to turn on fairness with defaults). `ServerStartup` forwards it to the `ContinuousBatchSchedulerService`; when omitted, scheduler defaults apply.

`--prefill-chunk-size N` (alias `--ubatch-size`, llama.cpp `-ub` analog; also bindable as `ServerOptions.PrefillChunkSize`) caps prompt-prefill work. Honest semantics per path: on the **single-request `TextGenerator` path** it truly chunks the prompt into ≤ N-token forward passes (bounding peak activation memory); on the **scheduler path** it is applied as `MaxPrefillTokensPerStep` (per-step admission cap — a single prompt longer than N still prefills in one forward pass once admitted) unless the bound `Scheduler` section already sets that cap explicitly.

| Option | Default | Meaning |
|--------|---------|---------|
| `MaxActiveSequences` | 64 | Slot cap. KV-cache pressure is the hard limit; this is a soft upper bound for batch-formation cost. |
| `MaxPrefillTokensPerStep` | 0 (disabled) | Chunked-prefill cap. When non-zero, no single Step iteration prefills more than this many tokens, even if a long prompt has more to feed. Decode tokens of already-decoding sequences keep running every step regardless — prevents head-of-line blocking. |
| `ReserveBlocksPerSequence` | 0 (disabled) | Admission KV-pressure gate: skip admission when `pagedPool.FreeBlocks < ReserveBlocksPerSequence`. |
| `EnablePreemption` | false | Allow a higher-priority request to preempt a lower-priority active sequence under block pressure (recompute-on-resume). |
| `MaxRecurrentSequences` | 0 (disabled) | Caps concurrent recurrent (Mamba/GDN) sequences to bound per-sequence recurrent-state memory. |
| `EnableFairness` | false | Per-API-key start-time fair queuing in admission so a high-volume key can't starve others sharing a priority tier. The fairness identity is `InferenceRequest.ApiKey` (the resolved API key, stashed by `RateLimitMiddleware`). |

Per-key token observability: `ContinuousBatchScheduler.GetPerKeyTokenUsage()` snapshots cumulative generated tokens per API key, and the `dotllm.engine.tokens.by_key` meter counter (tagged by `key`) records the same — zero-overhead when no listener is subscribed.

### Engine telemetry providers

Once a `ContinuousBatchSchedulerService` is constructed, it wires the observable gauges that
`EngineTelemetry` exposes:

- `dotllm.engine.request.queue_depth` → `Inner.QueueDepth + Inner.ActiveCount` (so saturation is visible — pure queue depth would underreport when sequences are already admitted).
- `dotllm.engine.kvcache.utilization` → `1.0 - FreeBlocks / TotalBlocks` of the underlying paged pool (when present).

Both providers are cleared back to `null` on `Service.Dispose` / model swap so the gauges return to their `-1` sentinel.

## Request Validation

Both `/v1/chat/completions` and `/v1/completions` validate inputs before inference:

| Check | Limit | Response |
|-------|-------|----------|
| Empty messages array | 0 | 400 `"messages array must not be empty"` |
| Messages count | > 1024 | 400 `"messages array exceeds maximum of 1024"` |
| Empty prompt (completions) | empty/null | 400 `"prompt must not be empty"` |
| `max_tokens` | &le; 0 | 400 `"max_tokens must be a positive integer"` |
| Prompt token count | &ge; `MaxSequenceLength` | 400 `"prompt (N tokens) exceeds model context length (M)"` |
| `prompt_tokens + max_tokens` | > `MaxSequenceLength` | `max_tokens` silently clamped to remaining context |
