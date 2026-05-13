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
  "n": 1,
  "lora_adapter": "customer-support"
}
```

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

### `POST /v1/embeddings`
Extract embedding vectors from text.

**Request**: `{"input": "text to embed", "model": "..."}`
**Response**: `{"data": [{"embedding": [0.1, -0.2, ...], "index": 0}]}`

Implementation: Run input through the model, capture hidden state at `PreLmHead` hook point, apply pooling (mean pool over tokens by default, configurable), L2 normalize. Minimal additional code given the hook system.

### `GET /v1/models`
List loaded models: `{"data": [{"id": "llama-3-8b-q4_k_m", "object": "model"}]}`

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

Per-API-key admission controls built on `System.Threading.RateLimiting`. Off by default — when no `RateLimit` configuration is present (or `Enabled: false`) the middleware short-circuits and adds zero overhead. When configured, the middleware sits between CORS and endpoint mapping and inspects every request to `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`.

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
Content-Type: application/json

{"error":"Rate limit exceeded (tokens-per-minute). Retry in 12s."}
```

| Header | Meaning |
|--------|---------|
| `Retry-After` | Seconds until the limiter can admit. Driven by the BCL limiter metadata where available. |
| `X-RateLimit-Limiter` | Which of the three limiters rejected (`Requests`, `Tokens`, `Concurrency`). Useful for client backoff decisions. |

### Authentication note

`HeaderApiKeyResolver` exists only so rate-limit buckets can be partitioned per caller. dotLLM still has no built-in authentication — see § Security. Host applications wiring real auth (OAuth, JWT, mTLS) should register their own `IApiKeyResolver` implementation that returns the authenticated principal's stable ID. The rate-limit machinery is transport-independent and will bucket on whatever opaque string you return.

### Unmetered endpoints

`/health`, `/ready`, `/v1/models`, `/v1/tokenize`, `/v1/detokenize`, `/v1/lora`, `/v1/cache/clear`, `/props`, `/config`, and the chat UI are deliberately unmetered — they're either probes, control-plane operations, or static asset serving.

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

The server processes one inference request at a time, serialized by a `SemaphoreSlim(1, 1)` gate. Concurrent requests queue and are served FIFO. This is by design — no batch scheduler exists yet.

The startup log prints `Single-request mode — requests processed sequentially` as a reminder.

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
