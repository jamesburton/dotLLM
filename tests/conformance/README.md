# SDK conformance matrix (#453)

`sdk_conformance.py` drives a running dotLLM server with the **official vendor
SDKs** — the `openai` and `anthropic` Python packages — pointed at it through
`base_url`, against a small real GGUF. It exists so that "more compatible than
most" is a measurement instead of an opinion.

**The suite is expected to start mostly red.** Each sibling issue turns specific
rows green; the `flipped by` column in the generated matrix names which one owns
each red row. `BASELINE.md` in this directory is the honest snapshot at the
commit it records — regenerate it when a sibling issue lands.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r tests/conformance/requirements.txt
dotnet build src/DotLLM.Cli -c Release

python tests/conformance/sdk_conformance.py \
    --md-out matrix.md --json-out matrix.json
```

The harness launches `dotllm serve` itself (CPU, no UI, port 18453), waits for
`/health`, reads the model id from `/v1/models`, runs every row, and tears the
server down. Useful flags:

| flag | effect |
|---|---|
| `--base-url URL` | run against an already-running server; skips launch |
| `--only chat,tool` | run only rows whose id contains one of these substrings |
| `--attempt-429` | actually drive the 429 rows (needs a rate-limited server) |
| `--device gpu` | serve on the GPU instead of CPU (take the GPU lock first) |
| `--strict` | exit non-zero unless every row passes (default: always exit 0) |

## Model fixture

Per `CLAUDE.md`'s storage rules the model is **resolved, never copied into the
tree**:

1. `DOTLLM_CONFORMANCE_GGUF` — absolute path to a GGUF, if set;
2. otherwise the first hit in `~/.dotllm/models` among
   `bartowski/Llama-3.2-1B-Instruct-GGUF` (Q8_0),
   `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (Q8_0),
   `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (q8_0).

A **tool-calling chat template is required** for the tool rows, so base models
such as SmolLM-135M are deliberately not candidates — a base model would make
those rows fail for model reasons rather than server reasons.

## Which sibling issue owns each red row

`BASELINE.md` is generated and overwritten on every run; this table is the
hand-maintained attribution behind it, established by capturing the raw HTTP
response rather than inferring from the SDK exception. **No server code was
changed to produce it.**

| row | owner | what the server does today |
|---|---|---|
| `openai/models.retrieve` | #450 | `GET /v1/models/{id}` is not routed — 404. Only the list endpoint exists. |
| `openai/usage.stream` | #450 | `stream_options` is absent from `ChatCompletionRequest`, so `include_usage` is ignored. Usage is instead attached to the last *content* chunk — the one still carrying `choices[0].finish_reason` — plus a non-standard `timings` member. OpenAI's contract is a separate terminal chunk with `choices: []`, and usage only when requested. |
| `openai/tool.parallel` | #450 | `parallel_tool_calls` is absent from the request DTO too; the row also inherits the single-tool failure below. |
| `openai/embeddings` | #451 | `POST /v1/embeddings` is not routed — 404. Note `RateLimitMiddleware.IsMeteredPath` already lists `/v1/embeddings`, so that allowlist is drifted ahead of reality. |
| `openai/error.envelope` | #452 | A 400 returns the flat `{"error":"max_tokens must be a positive integer"}` from `Models/CommonResponses.cs`. OpenAI nests `{"message","type","param","code"}` under `error`; the SDK unwraps `body["error"]`, so `e.body` arrives as a bare string. |
| `openai/error.429`, `anthropic/error.429` | #452 | NOT-EXERCISED — see below. |
| `anthropic/messages`, `.stream`, `tool.single`, `structured`, `usage` | #448 (`.stream` also #449) | `/v1/messages` is not routed — a clean 404, so the SDK raises `NotFoundError` and the rows report NOT-IMPL instead of erroring the run out. |
| `anthropic/count_tokens` | #449 | `/v1/messages/count_tokens` is not routed — 404. |

### Why the two 429 rows are NOT-EXERCISED

Plainly: **the harness never provoked a 429, at any configured limit, because no
limit can be configured.** `RateLimitMiddleware` runs only when
`ServerOptions.RateLimit.Enabled` is set; `ServeCommand` builds `ServerOptions`
by hand, never sets `RateLimit`, and there is no configuration binding for it.
So on `dev` today no invocation of `dotllm serve` can emit a 429.

The Anthropic row is blocked a second, independent way:
`RateLimitMiddleware.IsMeteredPath` is a hardcoded allowlist of
`/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`. Even once rate
limiting is reachable, `/v1/messages` is not on it, so an Anthropic request can
never be metered — that row must stay NOT-EXERCISED until the allowlist covers
it.

When a 429 *is* reachable, run with `--attempt-429`. The row deliberately does
not accept the typed exception alone: both SDKs pick the exception class from
the status code, so `RateLimitError` would be raised even by today's flat body
and the row would pass vacuously. It additionally requires a structured error
object with `message` and a rate-limit `type`, and a `Retry-After` response
header. (`WriteRejection` already sets `Retry-After` and `X-RateLimit-Limiter`;
it is the body that is flat.)

### Defects found that no sibling issue currently owns

1. **`tool_choice` is parsed and then discarded.**
   `ChatCompletionEndpoint.cs:107` assigns
   `RequestConverter.ParseToolChoice(...)` to a local with exactly one
   reference — its own assignment. `ToolChoice.Function`, `Required` and `None`
   therefore have no effect: a forced tool call is not forced and `"none"` does
   not suppress tools. This is why `openai/tool.single` is red despite using a
   forced `tool_choice` specifically to keep model flakiness out of the row.
2. **`json_schema` constrained decoding never closes the object.** With strict
   `response_format.json_schema` the server emits
   `{"city":"Paris","population":2140000, ` followed by an unbounded whitespace
   run until `max_tokens` (`finish_reason: "length"`). Reproduced at
   `max_tokens` 64 **and** 300, so it is not a short budget: after the
   separating comma the constraint appears to permit whitespace indefinitely
   without requiring the next property name. `{"type":"json_object"}` works and
   that row passes.
3. **`<|eom_id|>` does not stop generation for Llama-3.2.**
   `CommonStopSequences` contains it, yet tool responses contain it verbatim and
   loop `<|python_tag|>{...}<|eom_id|><|start_header_id|>assistant…` until
   `max_tokens`.
4. **Llama-3.2's tool output shape is not parsed.** The model emits
   `{"type":"function","function":"get_weather","parameters":{…}}` while
   `LlamaToolCallParser` expects `name` + `parameters`. Partly model quality at
   1B, but combined with (1) the single-tool row has no path to green on this
   fixture today.

## Statuses

| status | meaning |
|---|---|
| `PASS` | the row's assertions all held |
| `FAIL` | the server responded, but wrongly — the detail says how |
| `NOT-IMPL` | the SDK got a 404: the endpoint does not exist yet |
| `NOT-EXERCISED` | the condition under test could not be provoked at all |
| `SKIP` | filtered out by `--only` |

`NOT-EXERCISED` is deliberately distinct from `PASS`. The rate-limit rows are
the live example: a "429 maps to a typed exception" row that never saw a 429
has measured nothing, and reporting it green is the vacuous-test failure mode
that #417/#420/#421 were about. See `BASELINE.md` for why no 429 is currently
reachable.

## Why the rows assert fields by hand

The `openai` SDK builds response models with `construct()` — **no validation**.
A missing `owned_by`, a `logprobs` entry with no `bytes`, a tool call with no
`id`, or a stream with no terminal `finish_reason` all come back as objects
holding `None` and would read as green. Every row therefore asserts the
discriminating fields itself. Both clients are also constructed with
`max_retries=0`, so the SDKs' default 429/5xx backoff cannot hide what the
error rows measure.

## SDK pins

`requirements.txt` pins `openai==3.16.2` and `anthropic==1.7.0` (latest on PyPI
2026-09-21). The matrix is only comparable across runs if the clients are
identical, so bump the pins deliberately and regenerate `BASELINE.md` in the
same commit. The generated matrix records the pins, the Python version, the
commit, and the fixture in its header.

## CI gate — why this does not run in CI today

The suite is **gated off by default** and run on demand. Three reasons:

1. **It needs a real multi-hundred-MB GGUF.** Model weights are never in the
   repo (`CLAUDE.md` § Model & Fixture Storage Rules) and CI has no populated
   `~/.dotllm/models`; the harness exits 1 with a clear message when no fixture
   resolves.
2. **It needs a Python environment with two third-party SDKs**, which the
   existing `.github/workflows/ci.yml` (`dotnet build` / `dotnet test` only)
   does not provision.
3. **It costs minutes of CPU inference** — model load plus warm-up plus ~20
   generations.

The suite is written so that turning the gate on is a small step once a CI
runner has a cached fixture: add a job that installs `requirements.txt`, builds
`src/DotLLM.Cli -c Release`, restores the GGUF from cache, and runs the harness
with `--json-out`. Leave `--strict` **off** until the sibling issues have
landed, or the job fails by design on the expected-red baseline.
