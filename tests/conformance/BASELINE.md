# dotLLM SDK conformance matrix (#453)

- generated: 2026-09-21T13:39:27+0100
- commit: `7fa06298`
- model: `Llama-3.2-1B-Instruct-GGUF`  (fixture: `(external server)`, device: cpu)
- python: 3.11.15
- openai: `3.16.2`  anthropic: `1.7.0`

| row | sdk | case | status | flipped by | detail |
|---|---|---|---|---|---|
| `openai/chat.completion` | openai | plain completion | **PASS** | — | content='4' finish=stop |
| `openai/chat.stream` | openai | streaming | **PASS** | — | 15 chunks, 44 chars, finish=stop |
| `openai/tool.single` | openai | tool call (single, forced) | **PASS** | — | get_weather({"location": "Paris"}) finish=tool_calls |
| `openai/tool.parallel` | openai | tool call (parallel) | **FAIL** | #450 | AssertionError: 1 tool call(s), want >= 2 |
| `openai/json.object` | openai | JSON output (json_object) | **PASS** | — | parsed { "city": "Paris", "country": "France" } |
| `openai/json.schema` | openai | structured output (json_schema, strict) | **FAIL** | — | JSONDecodeError: Expecting property name enclosed in double quotes: line 107 column 4 (char 222) |
| `openai/logprobs` | openai | logprobs + top_logprobs | **PASS** | — | 8 tokens, top_logprobs=3 |
| `openai/usage.nonstream` | openai | token counting via usage (non-stream) | **PASS** | — | prompt=39 completion=7 total=46 |
| `openai/usage.stream` | openai | token counting via stream_options.include_usage | **PASS** | #450 | final chunk usage total=46 |
| `openai/usage.stream.unrequested` | openai | no usage when include_usage is absent | **PASS** | #450 | no unrequested usage |
| `openai/models.list` | openai | model list | **PASS** | — | 1 model(s), first=Llama-3.2-1B-Instruct-GGUF |
| `openai/models.retrieve` | openai | model retrieve | **PASS** | #450 | retrieved Llama-3.2-1B-Instruct-GGUF |
| `openai/embeddings` | openai | embeddings | **PASS** | #451 | dim=2048 |
| `openai/error.envelope` | openai | 400 error envelope | **PASS** | #452 | error.type='invalid_request_error' |
| `openai/error.429` | openai | 429 → typed exception | **NOT-EXERCISED** | #452 | burst produced no 429 — server is not rate limited |
| `anthropic/messages` | anthropic | plain completion | **PASS** | #448 | '4' stop=end_turn |
| `anthropic/messages.stream` | anthropic | streaming (SSE event types) | **PASS** | #448/#449 | 31 events, 44 chars |
| `anthropic/tool.single` | anthropic | tool call (single, forced) | **PASS** | #448 | get_weather({"location": "Paris"}) |
| `anthropic/structured` | anthropic | structured output (forced tool) | **PASS** | #448 | {"city": "Paris", "population": 21400000} |
| `anthropic/count_tokens` | anthropic | token counting via count_tokens | **PASS** | #449 | input_tokens=47 |
| `anthropic/usage` | anthropic | usage on the message response | **PASS** | #448 | in=39 out=7 |
| `anthropic/error.429` | anthropic | 429 → typed exception | **NOT-EXERCISED** | #452 | burst produced no 429 on /v1/messages |

**Totals:** PASS=18, FAIL=2, NOT-EXERCISED=2
