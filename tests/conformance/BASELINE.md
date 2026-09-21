# dotLLM SDK conformance matrix (#453)

- generated: 2026-09-21T10:18:48+0100
- commit: `c3f05c65`
- model: `Llama-3.2-1B-Instruct-GGUF`  (fixture: `bartowski/Llama-3.2-1B-Instruct-GGUF [Q8_0]`, device: cpu)
- python: 3.11.15
- openai: `3.16.2`  anthropic: `1.7.0`

| row | sdk | case | status | flipped by | detail |
|---|---|---|---|---|---|
| `openai/chat.completion` | openai | plain completion | **PASS** | — | content='4' finish=stop |
| `openai/chat.stream` | openai | streaming | **PASS** | — | 15 chunks, 44 chars, finish=stop |
| `openai/tool.single` | openai | tool call (single, forced) | **FAIL** | — | AssertionError: no tool_calls returned |
| `openai/tool.parallel` | openai | tool call (parallel) | **FAIL** | #450 | AssertionError: 0 tool call(s), want >= 2 |
| `openai/json.object` | openai | JSON output (json_object) | **PASS** | — | parsed { "city": "Paris", "country": "France" } |
| `openai/json.schema` | openai | structured output (json_schema, strict) | **FAIL** | — | JSONDecodeError: Expecting property name enclosed in double quotes: line 107 column 4 (char 222) |
| `openai/logprobs` | openai | logprobs + top_logprobs | **PASS** | — | 8 tokens, top_logprobs=3 |
| `openai/usage.nonstream` | openai | token counting via usage (non-stream) | **PASS** | — | prompt=39 completion=7 total=46 |
| `openai/usage.stream` | openai | token counting via stream_options.include_usage | **FAIL** | #450 | AssertionError: the usage chunk must have an empty choices array |
| `openai/models.list` | openai | model list | **PASS** | — | 1 model(s), first=Llama-3.2-1B-Instruct-GGUF |
| `openai/models.retrieve` | openai | model retrieve | **NOT-IMPL** | #450 | 404 — endpoint not implemented |
| `openai/embeddings` | openai | embeddings | **NOT-IMPL** | #451 | 404 — endpoint not implemented |
| `openai/error.envelope` | openai | 400 error envelope | **FAIL** | #452 | AssertionError: flat error envelope: the server sent a bare string, want an object with message/type/param/code — got max_tokens must be a positive integer |
| `openai/error.429` | openai | 429 → typed exception | **NOT-EXERCISED** | #452 | NOT EXERCISED — no 429 could be provoked at any configured limit: `dotllm serve` never populates ServerOptions.RateLimit and there is no config binding for it, so RateLimitMiddleware is never wired into the pipeline. Re-run with --attempt-429 against an externally hosted rate-limited server. |
| `anthropic/messages` | anthropic | plain completion | **NOT-IMPL** | #448 | 404 — endpoint not implemented |
| `anthropic/messages.stream` | anthropic | streaming (SSE event types) | **NOT-IMPL** | #448/#449 | 404 — endpoint not implemented |
| `anthropic/tool.single` | anthropic | tool call (single, forced) | **NOT-IMPL** | #448 | 404 — endpoint not implemented |
| `anthropic/structured` | anthropic | structured output (forced tool) | **NOT-IMPL** | #448 | 404 — endpoint not implemented |
| `anthropic/count_tokens` | anthropic | token counting via count_tokens | **NOT-IMPL** | #449 | 404 — endpoint not implemented |
| `anthropic/usage` | anthropic | usage on the message response | **NOT-IMPL** | #448 | 404 — endpoint not implemented |
| `anthropic/error.429` | anthropic | 429 → typed exception | **NOT-EXERCISED** | #452 | NOT EXERCISED, twice over: (1) no 429 is reachable at all (see the openai/error.429 row); (2) even once rate limiting is wired, RateLimitMiddleware.IsMeteredPath is a hardcoded allowlist of /v1/chat/completions, /v1/completions and /v1/embeddings — /v1/messages is NOT on it, so an Anthropic request can never be rate limited. This row must never be reported green until that allowlist covers /v1/messages. |

**Totals:** PASS=6, FAIL=5, NOT-IMPL=8, NOT-EXERCISED=2
