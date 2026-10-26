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

**Reasoning** (#767): `enable_thinking`, `chat_template_kwargs`, `reasoning_effort` and `reasoning_format` steer the chat template and the
reasoning/answer split, and the response carries `message.reasoning_content` — see [Reasoning / thinking models](#reasoning--thinking-models-767).
Messages may carry `reasoning_content` back on later turns.

**Accepted and ignored**: `user`, `store`, `service_tier`, `metadata`. These
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

### `POST /v1/systemone` (Jev-compatible decisions, fork-only - #708)

A decision endpoint with the **wire shape of TypeSafe's Jev System One API**, so Jev SDKs, the OpenJev servers and the Microsoft Agent Framework
`Microsoft.Agents.AI.TypeSafe` provider (`TypeSafeDecisionClientOptions.Endpoint = http://host:port/v1/systemone`) can point at a dotLLM server.

**Background.** Jev (TypeSafe AI) is a non-autoregressive "System One" model: a state plus typed questions in, typed answers with probabilities out, in one
forward pass. Together AI's Tev1 only imitates the *contract* (`{state, question, options[label,key,description]}` -> one letter) on an ordinary language-model
head. dotLLM answers every question with ONE prefill and **no decode loop**: the question is rendered as the Tev1 contract (system prompt, thinking block
closed), and the logits of the option letters at the first generated position are renormalised into per-option probabilities.

```json
POST /v1/systemone
{"model":"jev-latest",
 "state":"Returns are allowed within 30 days. Purchase was 12 days ago.",
 "questions":{
   "in_window":{"type":"noul","instructions":"Is the return within the window?"},
   "route":{"type":"choice","instructions":"Which action?","criteria":{"approve":"Approve","deny":"Deny","escalate":"Escalate"}},
   "urgency":{"type":"score","instructions":"How urgent?","criteria":["not urgent","low","medium","high"]}}}
```

| question `type` | `criteria` | answer fields |
|---|---|---|
| `noul` | optional `{"true": "...", "false": "..."}` | `noul` = P(true) |
| `choice` | `{name: description}` (2..52) | `choice` (argmax name), `probabilities` (by name), `confidence` = `1 - H(p)/ln K` |
| `score` | array of 2..10 level descriptions | `score` = probability-weighted level index in `[0, n-1]`, `probabilities` keyed `"0".."n-1"`, `confidence`, `legend` |

The response is `{model, answers{id:{type, ...}}, usage{input_tokens, output_tokens:0}}`. Validation failures return `422` with the standard error envelope.

Notes: `state` may be a string or any JSON value (embedded verbatim). `model` is informational (the loaded model answers). Questions run sequentially in request
order and the state is the prompt prefix, so the prefix caches reuse it across questions; concurrent requests are batched when the scheduler is on
(`--expected-concurrency`). Works on every backend with any instruction-tuned model; **Tev1-4B is the model the prompt contract was trained for**
(~65 ms per question on Strix Halo Vulkan). Probabilities are the model's own restricted softmax, **not calibrated** like Jev's, and `score` is an
expected level index, not a Jev-trained ordinal head. Option-label tokens must be single tokens in the tokenizer (A-Z, a-z).

**Calibration and option order (opt-in, #710).** Measured with `scripts/decision-calibration.py` on 480 programmatic labelled items (easy and
deliberately hard multi-rule cases) against Tev1-4B Q4_K_M on Vulkan: raw accuracy 97.9%, NLL 0.0625, ECE 0.027, and the model is slightly *under*-confident.

| setting | what it does | measured effect |
|---|---|---|
| `--decision-temperature T` | probabilities = softmax(logits / T); 0 or 1 = raw | T=0.58 fitted on half the set: held-out NLL 0.0546 -> 0.0428, ECE 0.0271 -> 0.0102 (the live server reproduces this exactly) |
| `--decision-orderings 2` | each noul/choice question is also asked with the options reversed; per-option logits are averaged | reversal flips 2.2% of choice argmaxes; choice accuracy 98.2 -> 98.9%, overall NLL 0.0480 -> 0.0452 (with T=0.58), at 2x the forward passes; score questions are never reversed (ordinal) |

Both default to off: the temperature is a fit to a synthetic set, not a guarantee for another task or model (fit yours with the script - it writes
accuracy / NLL / ECE, the fitted T and the order-flip rate), and ordering averaging doubles the cost for a small gain.

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

## Server console log (#1021)

`dotllm serve` timestamps every `[dotllm]` line (local time) and logs each `POST` request's lifecycle tersely, so an incident can be
reconstructed from the console:

```
2026-10-16 16:08:27.406 [dotllm] req#1 POST /v1/chat/completions
2026-10-16 16:08:27.459 [dotllm] req#1 model=QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf
2026-10-16 16:08:27.677 [dotllm] req#1 loading QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf from C:\...\SmolLM-135M.Q8_0.gguf (active: none)
2026-10-16 16:08:41.207 [dotllm] req#1 loaded QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf in 13.5s
2026-10-16 16:08:42.160 [dotllm] req#1 done 200 in 14.8s
2026-10-16 16:09:02.446 [dotllm] req#6 waiting for the server lock (generate); held by req#5 generate
2026-10-16 16:09:04.454 [dotllm] req#6 gave up waiting for the server lock after 2.0s (cancelled)
2026-10-16 16:09:04.454 [dotllm] req#6 cancelled (client disconnected) after 2.0s
```

Logged: arrival (method + path), the requested `model`, lock waits (only when contended, naming the holder), model loads (start, end,
duration, or the failure), idle-unload / explicit unload, and completion (status + duration) or cancellation. `GET`s (health / props polling)
are not logged. Spectre.Console output (banner, spinners) is not stamped.

## CPU KV cache type default (#1284)

With paged KV (the `serve` default) on CPU, leaving both `--cache-type-k` and `--cache-type-v` unset selects Q8_0 paged blocks on uniform,
non-hybrid models (half the memory of F16 and faster at every measured context, #1268). Startup prints
`[dotllm] INFO: KV cache: no type specified; using Q8_0 paged blocks (faster, half the memory of F16). Use --cache-type-k f16 --cache-type-v f16 for F16.`
Explicit values are honoured unchanged: `f16`/`f32` for both gives F16 blocks, `q8_0` for both gives Q8_0 blocks. Giving only K or only V
keeps the pre-#1284 behaviour (a lone `f16` side gives F16 blocks; a lone `q8_0`/`q4_0` side uses the quantized simple cache). Models the
paged pool does not serve (hybrid/recurrent, per-layer KV widths) keep their existing cache. Details: [KV_CACHE.md](KV_CACHE.md).

## Model swaps, `GET /v1/status` and SSE keep-alive (#1023)

**Free before load.** A request (or `POST /v1/models/load`) for a different model takes the server lock, then - before loading - stashes the
active model and enforces the residency budget **with the incoming model in the active slot**. By default (`MaxResidentModels = 1`) that
disposes the old model first, so the two never share device memory. With a larger budget, stashed models are additionally evicted
least-recently-used until the incoming weights fit in the CUDA device's **free** memory. `--device auto` now judges a CUDA device by free
memory (NVML, all processes): `weights + 15 % of total <= free`; an explicit `--device gpu` whose weights exceed the free memory fails at
once with the numbers (unless `--gpu-layers` asks for a partial offload). A load that can never fit an explicit GPU - weights above its
total memory, or above its free memory plus what this server's models on that device would release - is refused **before** anything is
freed, so the resident model keeps serving. Device-room eviction only evicts models on the target GPU. If the requesting client has already
disconnected, nothing is freed and nothing is loaded; once freeing has begun the load runs to completion even if the client goes away
(abandoning it would leave no model). A request queued behind a swap re-checks its model once it holds the lock and re-activates it if needed
(a stashed one is a field swap, an evicted one reloads); if that fails it gets a 503 (`code: model_not_loaded`) - it never runs on a
disposed or different model.

**`GET /v1/status`** - lock-free, unmetered, cheap to poll. Also embedded in `/props` as `activity`.

```json
{
  "state": "loading",
  "is_ready": false,
  "model_id": null,
  "loading_model_id": "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-GGUF/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf",
  "loading_since": "2026-10-16T14:12:54.921+00:00",
  "loading_elapsed_seconds": 41.3,
  "loading_request": "req#12",
  "busy_with": "req#12 model activation",
  "busy_since": "2026-10-16T14:12:54.903+00:00",
  "busy_elapsed_seconds": 41.3,
  "waiting_requests": 1
}
```

| field | meaning |
|---|---|
| `state` | `loading` (a load is in progress; wins over busy), `busy` (a request holds the execution lock), `idle` (model ready, lock free), `no_model` |
| `is_ready` / `model_id` | the active model, only when ready - `is_ready` is false for the whole of any load |
| `loading_*` | the load target, its start (UTC ISO 8601), elapsed seconds and triggering request; null when not loading |
| `busy_*` | the lock holder (`req#N generate`, `req#N model activation`, `idle sweep`, ...), since when, elapsed; null when free |
| `waiting_requests` | requests queued for the lock |

During the activation hand-off (a request that just loaded a model keeps first use of it for up to 30 s, so a waiter for another model
does not swap it away and force a reload) `/v1/status` reports `busy` with `busy_with: "req#N (activated <id>, preparing to generate)"`,
and yielding requests count in `waiting_requests`.

`/ready` answers 503 during a load. Requests arriving while a model loads queue behind it. Known gap: requests routed
through the continuous-batch scheduler (streaming or not) bypass the lock, so they are not re-validated after a swap.

**SSE keep-alive.** A streaming request (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`) that waits - for a load or for the lock,
in the model-activation phase as well as the generation phase - receives an SSE comment every 10 s:

```text
: waiting for loading owner/repo/x.gguf (started 2026-10-16T14:12:54Z, 41s)

: waiting for busy: req#7 generate (73s), 1 queued

```

Comment lines are ignored by EventSource and the OpenAI / Anthropic SDKs. Nothing is sent when the wait is shorter than 10 s, so the
uncontended stream is byte-for-byte unchanged. On `/v1/chat/completions` the role chunk (and with it the `chatcmpl-...` id) is sent only
once the request holds the lock. Once a keep-alive has committed the response (HTTP 200, `text/event-stream`), a later error arrives
in-stream: on the OpenAI surfaces as `data: {"error":{...}}` followed by `data: [DONE]`; on `/v1/messages` as Anthropic's
`event: error` / `data: {"type":"error","error":{"type":"...","message":"..."}}` (no `[DONE]`). Ollama's NDJSON `/api/*` streams get no
keep-alive (NDJSON has no comment syntax).

**Prefill keep-alive (#1025).** Once the request holds the lock, the same comments continue until the first token. The prompt's
prefill runs before it, which takes minutes for a long agent conversation:

```text
: waiting for prompt processing (40s)

```

A client that disconnects (or a server shutdown) stops the prefill at the next chunk boundary: 1,024 tokens on the CUDA Qwen3
hybrids and the Vulkan dense hybrid (Bonsai-2, #1100) (`IModel.PreferredPrefillChunkTokens`), or `--prefill-chunk-size`. Non-streaming requests observe the same cancellation.

## Model names, the hub cache and pulling (#714)

One resolver (`DotLLM.HuggingFace.ModelResolver`) turns a model name into a GGUF everywhere - `dotllm run|chat|serve|bench`, a request's `model`
field, and `POST /v1/models/load`:

| you write | resolves to |
|---|---|
| `C:\models\x.gguf` | that file |
| `owner/repo` | the repo's best local GGUF: **Q4_K_M first** (then Q4_K_S, Q4_0, Q5_K_M, Q8_0, ...), not the largest file |
| `owner/repo:Q8_0` | the file whose name contains the tag (`:latest` = the default); the legacy `--quant` is the same thing |
| `owner/repo/file.gguf` | that file |
| `hf.co/owner/repo:tag`, `hf://owner/repo` | same as `owner/repo[:tag]` |
| `Tev1-4B-experimental-Q4_K_M` | a local file by stem or filename, **only when exactly one local model has that name** (see below) |

**Ambiguous bare names are an error (#1021).** Ids are repo-qualified (`owner/repo/file.gguf`), but a bare stem or filename is still accepted
for clients holding an old id. When it matches more than one local model - typically a quant filename such as `q4_k_m.gguf` shipped by several
repos - nothing is guessed: every surface (chat, completions, messages, embeddings, `/api/*`, `POST /v1/models/load`, and the CLI) fails with the
candidate ids, and nothing is loaded:

```json
{"type":"error","error":{"message":"Model name 'q4_k_m' is ambiguous: it matches 2 local models. Use the full id of one of: Hcompany/Holo4-27B-GGUF/q4_k_m.gguf, Hcompany/Holo4-35B-A3B-GGUF/q4_k_m.gguf",
  "type":"invalid_request_error","param":"model","code":"ambiguous_model"}}
```

(HTTP 400; ollama's `/api/*` uses its own `{"error": "..."}` shape.) It used to resolve to the highest-ranked file (quant preference, then the
largest), which loaded a 35B model the client never named. `owner/repo[:tag]` is unaffected: choosing the default quant within one repo is a
choice among quants of one model, not an ambiguity. The OpenAI-shaped surfaces use `code: "ambiguous_model"` (a missing model is still
`model_not_found`); the Anthropic surface has no code field. One physical file reachable through both the flat mirror and the hub
snapshot (where it may sit in a subfolder) is one model, never ambiguous with itself.

**Stable ids.** An id is derived from the file's place in the stores - `owner/repo/<path in the snapshot>` - so it is the same whichever
copy was opened and across reloads; a file outside the stores (`dotllm serve C:\x\model.gguf`, `model_path` on `/v1/models/load`) is
identified by its full path, never a bare stem a client could persist and later see resolve to a different file. The server also remembers
every id it has handed out and maps it back to that exact file first, before any name resolution.

Local models are found in **both** the flat mirror (`~/.dotllm/models/{owner}/{repo}`) and the Hugging Face hub cache
(`HF_HUB_CACHE` / `HF_HOME/hub` / `~/.cache/huggingface/hub`, symlinked or hard-linked snapshots), so files fetched by `hf download` or another
tool need no re-pull; the same file seen through both is listed once (`GET /v1/models/available`, `dotllm model list`). Multimodal projectors
(`mmproj*`) and later shards of a split GGUF are never chosen as the model.

**Pull on a miss.** `dotllm run|chat|serve owner/repo[:tag]` downloads a missing Hub model with progress (resumable, into the hub cache, like
`ollama run`). The **server** does not download on a request unless started with `--auto-pull`: a remote client must not be able to start
multi-gigabyte transfers by naming a repo. Without it a miss answers `model_not_found` with the `dotllm model pull ...` / `POST /v1/models/pull`
hint. `dotllm model pull owner/repo:Q4_K_M` selects the file by tag without a prompt and now also writes to the hub cache. `dotllm model delete`
removes every link (mirror, snapshot) and the blob when it is unambiguous (same length and first MiB, exactly one candidate); it matches every
file of a repo unless you narrow it with `--quant`.

## Custom models: `model add`, profiles (Modelfile equivalent), `ps` / `stop` (#716)

**Add your own GGUF.** `dotllm model add C:\models\my-finetune.gguf [--name my-finetune]` links the file (hard link, no extra disk) into the model store as
`local/<name>`; `dotllm model add https://host/x.gguf` downloads a direct URL there. The file's `GGUF` magic is checked, so an HTML error page is
rejected. It is then found by name everywhere (`dotllm run my-finetune`, a request's `model`, `model list`).

**Profiles.** `dotllm model create <name> --from <base> [options]` saves `~/.dotllm/profiles/<name>.json` (`DOTLLM_PROFILES_DIR` overrides):

```
dotllm model create terse --from bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M ^
    --system "Answer in one sentence." --temperature 0.2 --max-tokens 128 --stop "###" --device vulkan --keep-alive 600
```

| field | effect when a request for `<name>` arrives |
|---|---|
| `from` | the base model: path, `owner/repo[:quant]`, local name, or another profile (chains of up to 8; the outer profile's values win) |
| `system` | prepended as the system message **only if the request has none** (chat completions and `/v1/messages`) |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `max_tokens`, `seed` | defaults for a request that omits them; a request value always wins; the global `/v1/config` defaults are never modified |
| `stop` | always added to the request's stop sequences |
| `device`, `gpu_layers`, `keep_alive` | used when the server loads the model on demand (an explicit `POST /v1/models/load` field still wins) |

The profile name is the model id: `GET /v1/models` lists it, residency and keep-alive key by it, and `model: "terse"` (or `terse:latest`) on any
endpoint loads the base model with these settings. A profile shadows a model that happens to share its name. `model show <name>` prints the merged
settings and where the base resolves, `model cp <src> <dst>` copies a profile (or aliases a model), and `model delete <name>` removes a profile without
touching its base model. `dotllm run|chat` resolve a profile to its base model but do not yet apply the profile's system prompt or sampling.

**`dotllm ps` / `dotllm stop [model|--all]`** are thin clients of a running server (`--url` or `DOTLLM_URL`, default `http://localhost:8080`):
`ps` lists the resident models with size, idle time and the auto-unload countdown; `stop` unloads (needs the server's `--allow-model-admin`).

## Using ollama's models (#718)

dotLLM reads an existing ollama installation **in place** and can pull from the same registry, so ollama names work everywhere a model is named:

| you write | what happens |
|---|---|
| `dotllm run llama3.2:3b` (also `serve`, `chat`, or `"model": "llama3.2:3b"` in a request) | found in your ollama store (`OLLAMA_MODELS` or `~/.ollama/models`): the GGUF blob is loaded in place - nothing is copied. A miss is pulled from `registry.ollama.ai` (CLI always; the server only with `--auto-pull`) |
| `dotllm model pull ollama:llama3.2:3b` | download from the ollama registry into `~/.dotllm/models/ollama/` (resumable, **sha256-verified**, a corrupt partial is discarded) and save a profile `llama3.2:3b` carrying the manifest's system prompt and parameters (temperature, top_p, top_k, repeat_penalty, stop, num_predict, seed; `num_ctx` is not mapped) |
| `ollama:user/model:tag` | a namespaced ollama model (a plain `owner/repo` always means Hugging Face) |
| `dotllm model import-ollama [name] [--force]` | turn the models of your ollama store into profiles (system prompt and parameters applied, blobs still used in place) |

`dotllm model list` and `GET /v1/models/available` list ollama-store models as `ollama/<name>`; a loaded one is keyed `name:tag`. Limits: the ollama
**prompt template** (a Go template) is not carried over - dotLLM renders the chat template embedded in the GGUF, which is what the model was
converted with; a model that only works through a custom ollama `TEMPLATE` may need a hand-written template. Only GGUF model layers are handled (vision
projector layers are ignored). dotLLM never writes into the ollama store.

## Ollama-compatible API: `/api/*` (#720)

Ollama clients (Open WebUI, Continue, Enchanted, ollama-python/js, LangChain's `ChatOllama`) can point at a dotLLM server unchanged
(`OLLAMA_HOST=http://localhost:8080`). Verified with the official `ollama` Python client against a live Vulkan server: `list`, `show`, `chat` and
`generate` (streaming and not, including `format: "json"` constrained decoding), `ps`, `pull` with progress, `delete`, `keep_alive: 0` unload, and
ollama's 404 error shape.

| route | notes |
|---|---|
| `POST /api/chat`, `POST /api/generate` | NDJSON streaming by default (`stream:false` for one object); `options` (temperature, top_p, top_k, min_p, repeat_penalty, seed, num_predict, stop), `format` (`"json"` or a JSON schema), `system`, `raw`, `keep_alive` (`5m`, `30s`, seconds, `-1`, `0`); final chunk carries `done_reason`, `total_duration`, `load_duration`, `prompt_eval_*`, `eval_*`. `/api/generate` with no prompt is the "load this model" call, and `keep_alive: 0` unloads it. Metered by the rate limiter like `/v1/*` |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `GET /api/version` | names are profiles, local models (stem) and ollama-store models (`name:tag`); `show` synthesises a Modelfile (`FROM`, `SYSTEM`, `PARAMETER`) from the profile; `version` reports the emulated API level (`0.6.0`) |
| `POST /api/pull` | `owner/repo[:tag]` (Hugging Face) or an ollama name; streams `pulling manifest` / `downloading` (`completed`/`total`) / `success` |
| `DELETE /api/delete` | removes a profile, a local model, or a model dotLLM pulled from the ollama registry (its file goes with its last profile); ollama-store models are never touched |
| `/api/embed`, `/api/embeddings`, `/api/create`, `/api/copy`, `/api/push` | `501` with a pointer (`/v1/embeddings`, `dotllm model create` / `cp`) |

`pull` and `delete` need `--allow-model-admin`, like `/v1/models/*`. `think` (`true` / `false` / `"low"` / `"medium"` / `"high"`) is honoured and the
model's reasoning comes back in `message.thinking` (`thinking` on `/api/generate`), streamed as its own chunks — see
[Reasoning / thinking models](#reasoning--thinking-models-767). Not supported inside chat: `tools` and images (use `/v1/chat/completions`;
a request with `tools` gets a `501`). Digests in `tags`/`ps` are stable placeholders, and `details` carries format and quantisation only (no family or
parameter size).

## Reasoning / thinking models (#767)

Qwen3.x, Bonsai, DeepSeek-R1-style and similar models think inside `<think>…</think>` before answering, and many chat templates **end the
generation prompt with an open `<think>`** — so the model's first tokens are its reasoning. dotLLM separates the two, the way llama.cpp's server,
vLLM and ollama do:

| surface | reasoning | answer |
|---|---|---|
| `POST /v1/chat/completions` | `message.reasoning_content` (stream: `delta.reasoning_content`, sent as separate deltas, before any `content`) | `message.content` |
| `POST /v1/messages` | a `thinking` content block (`thinking_delta` events, then an empty `signature_delta`) before the `text` block | `text` block |
| `POST /api/chat`, `/api/generate` | `message.thinking` / `thinking` (own NDJSON chunks) | `message.content` / `response` |

**Controlling it.** `enable_thinking` (bool) and `chat_template_kwargs` (any JSON object) on `/v1/chat/completions` are passed into the chat
template's Jinja context; `reasoning_effort` becomes the template's `reasoning_effort` (Qwen3.x accepts `low` / `medium` / `xhigh`; a template that
rejects a value yields a `400` naming `reasoning_effort`, and `none` means `enable_thinking=false`). Anthropic's `thinking: {"type": "enabled" |
"disabled"}` and ollama's `think` (`true` / `false` / a level string) map to the same switches; `/v1/messages` also takes `chat_template_kwargs`.
Anything left unset stays *undefined* in the template, so the template's own default applies. `chat_template_kwargs` can never override `messages`,
`tools`, `add_generation_prompt`, `bos_token` or `eos_token`, and the typed fields win over a same-named kwarg.

**`--reasoning-format none|auto|deepseek`** (default `auto`; a request may override it with `reasoning_format`):

| value | behaviour |
|---|---|
| `auto` | split when the rendered prompt ends in an unclosed `<think>`, or the model starts its output with `<think>` itself (original Qwen3 templates). A `<think>` later in the answer is left alone |
| `deepseek` | as `auto`, and `<think>…</think>` blocks are recognised anywhere in the output |
| `none` | no splitting; `content` is the raw output, `</think>` included (the pre-#767 behaviour) |

The splitter is incremental: a tag split across tokens (`</th` + `ink>`) never leaks into either stream, whitespace around the tags is dropped
(streamed text concatenates to the non-streamed result), and a generation that hits `max_tokens` mid-thought returns `content: ""` with the whole
text as reasoning and `finish_reason: "length"`.

**Everything downstream sees the answer only.** Tool-call parsing (a `<tool_call>` quoted in the thinking is not a call), stop-suffix stripping and
the Anthropic `text` block work on the answer. **Stop sequences** (other than template control tokens such as `<|im_end|>`) are suspended while the
model is inside a think block — whether the template opened it or the model did — and, after `</think>`, only match text that follows the closing tag
and the whitespace after it; EOS and `max_tokens` are never gated (`InferenceOptions.ReasoningStopGate`).

**Budget and usage.** Reasoning tokens are generated tokens: they count toward `max_tokens` and `completion_tokens`. When reasoning was produced,
`usage.completion_tokens_details.reasoning_tokens` reports the part spent thinking (exact when streaming, re-tokenised from the raw text when not),
and `completion_tokens` still includes it. A small `max_tokens` on a thinking model therefore ends mid-thought — raise it or send
`enable_thinking: false`.

**Constrained decoding.** `response_format` (`json_object`, `json_schema`, regex/grammar) and a forced `tool_choice` constrain decoding from the first
generated token, which cannot coexist with an open think block. So for such a request thinking defaults **off** (the template is rendered with
`enable_thinking=false` unless you set it explicitly) and the output is *not* split: the constrained output is the whole `content`. Pass
`enable_thinking: true` together with a constraint only if the model's template really expects it — the constraint will then apply to the reasoning
too.

**Other markups: Gemma-4 and gpt-oss (#798).** The wire markup is a property of the *model*, detected from its chat template at load
(`ReasoningMarkups.Detect`, stored on `JinjaChatTemplate.ReasoningMarkup`); `--reasoning-format` stays the user-facing policy and applies to all of them.

| family | markup the model emits | `reasoning_content` | `content` / `tool_calls` |
|---|---|---|---|
| Qwen3.x, GLM, Nemotron, DeepSeek-R1, ... | `<think>…</think>` | the think block | after `</think>` |
| Gemma-4 | `<|channel>thought\n…<channel|>` (the same incremental splitter, different tag pair, mirroring llama.cpp's `thinking_start_tag`/`thinking_end_tag`) | the thought | after `<channel|>`; a `<|tool_call>call:…<tool_call|>` there is parsed as a tool call |
| gpt-oss (Harmony) | channel messages: `<|channel|>analysis<|message|>…<|end|>`, `<|start|>assistant<|channel|>final<|message|>…`, `<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{…}<|call|>` | every `analysis` message | `final` (and a commentary *preamble* with no recipient) is `content`; a commentary message **with a recipient** is a tool call |

Harmony is a *sequence* of messages rather than one block, so it has its own splitter (`HarmonySplitter`, behind the same `IReasoningSplitter`
interface). It buffers each header until `<|message|>` (so no marker is ever split across chunks), accepts the recipient either after the channel or in the
role header (`<|start|>assistant to=functions.x<|channel|>commentary json<|message|>`), joins consecutive messages of one stream with a blank line, and passes a
recipient message through *raw* into the answer stream so the Harmony tool parser and the streaming tool-call suppressor see what they would on unsplit output.
Streamed and non-streamed results are identical under any chunking (tests feed one character at a time).

Two stop-set facts are specific to Harmony. `<|end|>` closes the *analysis* message and is followed by the final / commentary channel, so it is **not** a stop
string for a Harmony model (every other family's template wants it as one; it used to end the turn right after the thinking, so a gpt-oss request returned only
the analysis); it is dropped from the server's default stop list and from the list of control tokens that end a turn even inside reasoning, on `/v1/chat/completions` and `/v1/messages` (`ReasoningPlan.FilterStops`; the ollama surface adds no `<|end|>` stop of its own).
Generation ends on `<|return|>` (the declared EOS) or `<|call|>` (added to the end-of-generation set; llama.cpp treats both as EOG). The reasoning stop gate for Harmony
suspends user stop strings between `<|channel|>analysis<|message|>` and `<|end|>`. `reasoning_effort` (`low`/`medium`/`high`) is a real template variable on gpt-oss
and is passed through; there is no thinking *off* for it.

**Multi-turn.** An earlier assistant turn's reasoning goes back in as `reasoning_content` (OpenAI surface; `reasoning` is accepted as an alias),
as a `thinking` block (Anthropic) or `thinking` (ollama messages), and reaches the template as `message.reasoning_content`. Whether it is rendered into
the history is the template's call — Qwen3.x keeps it unless `chat_template_kwargs.preserve_thinking` is `false`. The web UI replays reasoning the same
way and shows it in a collapsible "Thinking" block above each answer (open while streaming, collapsed once the answer starts), with a Thinking
selector (model default / on / off) in the settings panel.

## Device selection: `--device auto` (#722)

`dotllm serve` defaults to `--device auto`: for each model it tries **CUDA** (when a GPU is present and the model file fits in 85% of its memory),
then **Vulkan** (when a device and the shader blobs exist), then the **CPU**, and a load that fails on one device falls through to the next with a log
line. The server keeps `auto` as its configured device, so each on-demand load chooses again for its own size; the device actually used is
`ResolvedDevice` (what `GET /api/ps` reports as `size_vram`). Pass `cpu`, `gpu:N` or `vulkan` to force one. `dotllm stop --server` and
`POST /v1/admin/shutdown` stop a running server gracefully (`--allow-model-admin`).

**No silent CPU (#790).** One parser (`DeviceSpec`: `auto`, `cpu`, `vulkan`, `gpu[:N]`/`cuda[:N]`) and one load dispatch (`DeviceModelLoader`) back
`serve`, `run`, `chat` and `bench`. An unknown value is rejected with the accepted list. An explicit GPU that cannot be honoured (no such device,
SPIR-V missing, load failure) fails with what/why/model-vs-device-memory and the `--device cpu` opt-in - the server does not start on the CPU.
`auto` ending on the CPU always sets `device_fallback_warning` in `/props` (no GPU servable, or the GPU load failed and why) and prints it at startup.
`/props` reports `device` (as requested) and `resolved_device` (as loaded) for every load path, explicit or `auto`; `dotllm ps` shows the same.

## GPU memory on Windows: no silent spill into system RAM (#1068)

On Windows the NVIDIA driver (WDDM) does not fail a CUDA allocation that exceeds dedicated VRAM by default. It backs the
excess with **shared system memory** ("sysmem fallback"), and every kernel that touches those buffers then reads them over
PCIe. On a 12 GB RTX 3060 serving Bonsai-2 27B (7.1 GB of weights) a long agent conversation grew its KV cache past VRAM
this way and requests went from minutes to hours: `nvidia-smi` showed 100% GPU utilization but a 0-3% memory-controller
load, and the Windows `GPU Process Memory` counter showed ~2 GB of *Shared Usage* for the server process.

The CUDA hybrid model (Qwen3HybridDense: Bonsai, Qwen3.5/3.6 dense) now refuses that instead:

- **Admission.** Every device buffer that grows with the context (the KV cache in any storage format, the F16 dequant
  scratch for quantized KV, the F32 attention staging, the MTP head's KV cache, recurrent-state checkpoints and row
  snapshots) is checked against free VRAM before it is allocated. A growth that would not leave a reserve free fails the
  request with a clear error naming the buffer, the size and the free memory, rather than spilling. The server answers
  it as `400 context_length_exceeded`, so agent clients compact and retry. The reserve is 256 MiB on top of CUDA's own
  free figure; `DOTLLM_CUDA_VRAM_RESERVE_MB` changes it. On WDDM, CUDA's free figure is the DXGI per-process budget minus
  usage, and on an RTX 3060 the spill began about 280 MiB past that budget, so the effective margin is about 0.5 GB.
- **Detection.** The server polls this process's `\GPU Process Memory(pid_*)\Shared Usage` counter (at most every 5 s,
  and after every KV growth) and logs a `WARNING: GPU memory has spilled into shared system memory` line each time it rises
  another 256 MiB above its post-load level. That catches a spill admission cannot prevent, for example when another
  program takes the VRAM the server was counting on.

### The served context is fitted to free VRAM

The checkpoint's window (262,144 tokens for Bonsai-2) is far more than a 12 GB card holds next to 7 GB of weights, and a
client that believes it keeps the conversation growing until the KV cache no longer fits. So at load, after warm-up, the
server runs one prefill tile (and, with MTP, one verify) to allocate every fixed-size buffer, then divides the free VRAM
left after the reserve by the per-token cost (KV rows in every attention layer under the chosen `--kv-quant-k/-v`, the
quantized-KV dequant scratch when a path still needs it, and the MTP head's KV) and serves that many tokens, rounded down to
1024:

```
[dotllm] Context window: 26624 of 262144 tokens, fitted to free VRAM (1826 MiB free after load, 256 MiB reserve, 149 MiB fixed, 54.0 KiB/token, q8_0 KV + MTP). Set --ctx-size to override.
```

`GET /props` reports it as `max_sequence_length`, and requests are validated against it (a prompt past it is rejected as
`context_length_exceeded`, and `max_tokens` is clamped to what remains), so agent clients such as CodeSharp compact in
time. `--ctx-size N` (`-c N`) sets the window explicitly, capped at the checkpoint's; a value above the fit is logged as a
warning, and a request that then outgrows VRAM fails with an out-of-memory error rather than spilling.

Measured on an RTX 3060 12 GB (desktop on the same card) with Bonsai-2 27B PQ2_0-MTP. The window counts the prompt plus
`max_tokens`:

| Flags | Fitted window | Cost per token |
|---|---|---|
| `--kv-quant-k q8_0 --kv-quant-v q8_0` (MTP on) | 26,624 | 54 KiB (KV 34, dequant scratch 4, MTP head 2 x 8; before #1094/#1095) |
| same, plus `DOTLLM_CUDA_PREFILL_CHUNK=512` | 35,840 | 54 KiB, but ~480 MiB less fixed prefill scratch |
| `--kv-quant-k q8_0 --kv-quant-v q8_0 --no-mtp` | 53,248 | 38 KiB |
| f16 KV (MTP on) | 14,336 | 80 KiB |

Since #1094 the quantized-KV flash prefill reads the K/V blocks directly, so the 4 KiB/token dequant scratch is gone from
every quantized row above: q4_0 without MTP costs **18.0 KiB/token** (was 22) and q4_0 with MTP 34 KiB (was 38). On the same
card with nothing else on it (2,782 MiB free after load at the 512 tile), q4_0 / `--no-mtp` / 512 tile fits
(2,782 − 256 − 149) MiB ÷ 18 KiB ≈ **134k** (was 110,592). The measured line on a card with another process holding ~0.9 GB:

```
[dotllm] Context window: 97280 of 262144 tokens, fitted to free VRAM (2130 MiB free after load, 256 MiB reserve, 149 MiB fixed, 18.0 KiB/token, q4_0 KV). Set --ctx-size to override.
```

Since #1095 the MTP head keeps its KV in the trunk's type and grows it in place, so it costs 1,156 B/token at q4_0 (was
16 KiB counted: F32 rows, doubled for the copying growth). q4_0 with MTP is **19.1 KiB/token** (was 34). Measured on the
same card with nothing else on it, 512 tile:

| Flags | Free after load | Fitted window | Cost per token |
|---|---|---|---|
| `--kv-quant-k q4_0 --kv-quant-v q4_0` (MTP on) | 2,316 MiB | **101,376** (~57k at the old 34 KiB) | 19.1 KiB (KV 18, MTP head 1.1) |
| same, `--no-mtp` | 2,782 MiB | 135,168 | 18.0 KiB |

The remaining gap to `--no-mtp` is the ~466 MiB MTP holds regardless of context (the head's weights and draft scratch),
not a per-token cost.

The 512-token prefill tile costs about 15% of prefill speed. `--no-mtp` doubles the window at the cost of MTP's decode
speed-up. An agent client that sends a large `max_tokens` (CodeSharp sends 16,384) spends that much of the window on
the reply, so its prompts must compact well below the window.

### Prefix reuse across agent turns (#1080)

An agent client resends the whole conversation every turn. The server keeps one recurrent-prefix snapshot (the GDN state
at a token boundary, plus the KV rows before it), and a turn that starts with it prefills only the rest. Two things used to
defeat that on the CUDA hybrid:

- **The KV growth needed twice the KV.** A turn needs a longer KV than the last one, and growing the model-owned storage
  copied it, so the old and new storage coexisted. Near the fitted window that never fits, the admission refused it, and the
  turn re-prefilled everything (`cached_tokens` 0). The KV now grows **in place** with CUDA virtual memory: each attention
  layer's K and V range is reserved once for the window (`cuMemAddressReserve`), and physical pages are mapped in as it
  grows (`cuMemCreate` + `cuMemMap`). Nothing is copied and a growth costs only the added pages (2 MiB granules), so it fits
  whenever the window does. Devices or drivers without the VMM API keep the copying growth; `DOTLLM_CUDA_KV_VMM=0` forces it.
- **The snapshot lagged a turn.** It sat at what a prompt shared with the previous one, so each hit reused the prompt from
  two turns back. A prompt that extends the whole previous one is now snapshotted at its own end, so the next turn prefills
  only what it adds. Other prompts (a fixed system prompt with different questions) keep the shared-prefix boundary.

Measured on an RTX 3060 12 GB (T5500, PCIe 2.0, desktop on the card) with Bonsai-2 27B PQ2_0-MTP, `--kv-quant-k q8_0
--kv-quant-v q8_0`, replaying an agent conversation that grows about 3.5k tokens a turn (`max_tokens` 4096, greedy):

| MTP | prompt | before: cached / time to first token | after: cached / time to first token |
|---|---|---|---|
| on (window 26,624) | 11,196 | 3,861 / 52 s | 7,568 / 27 s |
| on | 18,369 | 0 / 138 s | 14,696 / 28 s |
| off (window 53,248) | 25,102 | 0 / 195 s | 21,716 / 28 s |
| off | 35,751 | 0 / 286 s | 32,127 / 31 s |
| off | 46,481 | 0 / 385 s | 42,854 / 33 s |

Shared GPU memory stayed at its idle level (74-78 MiB) in every run; peak dedicated memory rose by 56-214 MiB, because the
KV is kept rather than freed and re-made. When #1080 landed, a turn that hit the prefix ran without MTP speculation (the MTP
head needs the whole sequence absorbed), so decode on those turns ran at the plain rate (43-51 ms/token here). #1088,
below, lifts that.

### Continuing sessions at 100k+ tokens (#1093)

Two more things defeated prefix reuse on a long chat session on a hybrid (recurrent) model:

- **The snapshot boundary sat inside the template tail.** A chat turn renders as `...assistant reply<end>` and the next turn
  re-renders that reply, but the generation prompt that closed the previous request (for ChatML templates,
  `<|im_start|>assistant
`) is not part of the re-rendered history. A snapshot taken at the end of the prompt therefore
  covered tokens the next prompt does not contain, and the next turn matched only an older boundary. The rule:
  - **The suffix.** The server renders the chat template twice, with `add_generation_prompt=true` and `=false`. The suffix is
    the remainder of the `true` render after the `false` render, and is used only when the `false` render is a prefix of the
    `true` one (`ReasoningSupport.GenerationPromptSuffix`). It reaches the engine as `InferenceOptions.GenerationPromptSuffix`.
  - **The boundary.** The snapshot goes **just before the suffix's tokens** (`RecurrentSnapshotBoundary`). When the legacy
    boundary (below) is also available, the later of the two wins, capped at prompt end - 1.
  - **Fallback.** With no usable suffix the boundary is the #1080/#1091 behaviour: prompt end - 1 when the prompt extends the
    previous one exactly, otherwise the prefix shared with the previous prompt. No usable suffix means: raw completions, a
    template that cannot render both ways (or whose `false` render is not a prefix), an empty remainder, or suffix tokens
    that are not the tail of the prompt's tokens (the suffix merged with the preceding text under BPE).
  - **Paths.** It applies on the single-slot, tiered and scheduler paths, and the superseded entry is released when the
    boundary moves forward. The single-slot and scheduler paths use the hint only when the prompt continues the previous
    conversation (it shares everything before the previous prompt's own suffix), so a classifier-style prompt, with a fixed
    system prompt and a different question each time, keeps its shared-prefix boundary. The scheduler also uses it when the
    prompt shares less than the minimum snapshot length with the previous one (a conversation starting). The tiered path uses
    it whenever a suffix boundary exists, falling back to prompt end - 1 when the prompt extends the previous one exactly.
- **A hit that needed to grow its KV could turn into a miss.** The growth asked for the needed size plus an eighth of
  headroom. Near the window the device refused that, the hit became a miss, and the whole prompt was prefilled again
  (916 s at 112k tokens). The growth now retries at exactly the needed size (capped at the window). If even that is refused,
  a prompt that would have to **refill 4,096 or more tokens** fails with HTTP 400 `context_length_exceeded` instead of
  silently re-prefilling for minutes; a shorter refill still misses and runs. The check happens before any state changes, so
  the snapshot survives for a request that fits. The tiered path, which never grows, rethrows the refusal and keeps the entry.

Measured on an RTX 3060 12 GB with Bonsai-2 27B PQ2_0-MTP, `--kv-quant-k q4_0 --kv-quant-v q4_0 --no-mtp --ctx-size 110592`,
`DOTLLM_CUDA_PREFILL_CHUNK=512`, greedy, `max_tokens` 64. Each turn appends about 34,000 characters (roughly 8k tokens) of
wikitext as a new user message and keeps the model's reply. Reproduce with `scripts/session-growth.py`
(`--model <id from /v1/models> --turn-chars 34000 --turns 15 --out growth.tsv`); the fitted window for other KV settings
comes from `scripts/context-fit-probe.ps1`.

| turn | prompt | before: cached / prefill | after: cached / prefill |
|---|---|---|---|
| 2 | 17,037 | 7,822 / 64 s | 7,818 / 63 s |
| 3 | 25,060 | 7,822 / 121 s | 17,032 / 57 s |
| 4 | 32,934 | 17,036 / 115 s | 25,055 / 58 s |
| 8 | 64,320 | 48,782 / 125 s | 56,406 / 64 s |
| 13 | 104,149 | 89,111 / 144 s | 96,651 / 69 s |
| 14 | 112,017 | **0 / 916 s** | refused: exceeds the 110,592-token window |

After the change every turn reuses the whole previous prompt (cached is the previous prompt minus 5 tokens), so each turn
prefills only the roughly 8k tokens it adds, at 57-74 s. Before, the cached prefix lagged a turn and prefill time
stretched to 115-144 s. With MTP on (window 51,200 on this card) the first three turns reuse the same way
(0 / 7,818 / 17,032 cached) and decode at 21-25 tokens/s.

### Continuing sessions on Vulkan (#1100)

Two things kept a long Bonsai-2 session off the Vulkan backend:

- **The KV never grew.** `VulkanNemotronHKvCache` kept the interface's never-grow `TryGrowTo`, and the model made an eager
  cache of exactly the size a miss asked for (needed + 256). Every turn of a growing session needs more, so turn 2 was
  refused ("could not grow to 17101 positions ... it holds 8143"). The Bonsai and LFM2 Vulkan hosts now make **lazy, growable**
  caches capped at the window. The forward grows the buffers before recording and drops descriptor caches that may name freed
  buffers (see `KV_CACHE.md`, "Growable caches for a continuing session").
- **Long prefills lost the device.** The model reported no prefill tile, so a whole prompt was one forward, and each layer's
  submission covered every row. A fresh ~16k prompt hit `VK_ERROR_DEVICE_LOST` after ~40 s. Two fixes:
  - **Prefill tile.** The model now reports a 1,024-token tile (`PreferredPrefillChunkTokens`). `DOTLLM_VULKAN_PREFILL_CHUNK`
    overrides it (0 = no tile), and `--prefill-chunk-size` overrides both.
  - **Bounded submissions.** The tile alone was not enough: an attention layer's work grows with the depth, and a 1,024-row tile
    still lost the device at about 85k positions (fresh ~128k prompt, two attention layers per submission). Submissions are
    now **bounded by estimated GPU time** (`VulkanPrefillSubmitBudget`, 1 s, `DOTLLM_VK_SUBMIT_BUDGET_MS`). The cost model has
    two terms:
    - dense GFLOPs (2 x parameters x rows);
    - causal attention GFLOPs (4 x rows x mean KV x heads x head dim).

    Each term has its own rate, refitted from every submission's measured wall time; Strix fits 11-25 TFLOP/s dense (it varies with the tile) and
    ~1 TFLOP/s attention. Layers are packed up to the budget. When one attention layer alone would exceed it, the tile runs as
    row slices: 512 rows at 64k and ~233 at 128k on Strix. Tiles under 128 rows are costed as 128, because they do not fill the
    GPU. `DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT` still fixes the count. `DOTLLM_VK_PREFILL_SUBMIT_LOG=1` prints one line per
    prefill forward (submissions, longest, fitted rates).

Measured on Strix Halo (Radeon 8060S, Vulkan) with Bonsai-2 27B PQ2_0-MTP, `--no-mtp --ctx-size 131072`, greedy, `max_tokens`
64, F32 KV. Fresh single prompts of wikitext, server restarted between them:

| prompt | before (dev `e2219f5a`) | after: prefill | prefill tok/s | decode tok/s | process working set |
|---|---|---|---|---|---|
| 17,003 | `VK_ERROR_DEVICE_LOST` | 127.6 s | 133 | 16.1 | 11.7 GiB |
| 32,831 | `VK_ERROR_DEVICE_LOST` | 344.6 s | 95 | 12.2 | 13.1 GiB |
| 64,077 | (not reached) | 1,039.9 s | 62 | 8.3 | 17.0 GiB |

The longest single submission in those prefills was 1.05 s, except for one: the 58-row tail of the 33k prompt took 1.65 s
before short tiles were costed at 128 rows.

Growth run: `scripts/session-growth.py --turn-chars 34000 --turns 16 --pid <server>`, adding ~8k tokens of wikitext per turn.
"Before" is dev `e2219f5a`. Working set and private bytes are the server process's after each turn. On this UMA device the KV
shows up there and as "GPU shared" memory: 24.5 GiB at the end, with 291 MiB dedicated.

| turn | prompt | before: cached | after: cached | new | TTFT | prefill tok/s | decode tok/s | working set | private |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7,823 | 0 | 0 | 7,823 | 45.7 s | 172 | 19.4 | 10.0 GiB | 10.2 GiB |
| 2 | 17,037 | **refused** | 7,818 | 9,219 | 83.7 s | 110 | 15.9 | 11.4 GiB | 11.5 GiB |
| 3 | 25,060 | | 17,032 | 8,028 | 98.8 s | 81 | 13.8 | 12.6 GiB | 12.8 GiB |
| 4 | 32,934 | | 25,055 | 7,879 | 119.5 s | 66 | 12.0 | 13.5 GiB | 13.7 GiB |
| 5 | 41,026 | | 32,929 | 8,097 | 146.4 s | 55 | 10.7 | 14.6 GiB | 14.7 GiB |
| 6 | 48,783 | | 41,021 | 7,762 | 163.4 s | 48 | 9.7 | 15.6 GiB | 15.7 GiB |
| 7 | 56,411 | | 48,778 | 7,633 | 181.3 s | 42 | 8.9 | 16.5 GiB | 16.6 GiB |
| 8 | 64,320 | | 56,406 | 7,914 | 212.0 s | 37 | 8.2 | 17.5 GiB | 17.6 GiB |
| 9 | 72,287 | | 64,315 | 7,972 | 235.8 s | 34 | 7.6 | 18.6 GiB | 18.7 GiB |
| 10 | 80,915 | | 72,282 | 8,633 | 281.6 s | 31 | 7.0 | 19.7 GiB | 19.8 GiB |
| 11 | 89,112 | | 80,910 | 8,202 | 309.4 s | 27 | 6.6 | 20.7 GiB | 20.8 GiB |
| 12 | 96,656 | | 89,107 | 7,549 | 307.4 s | 25 | 6.2 | 21.6 GiB | 21.7 GiB |
| 13 | 104,149 | | 96,651 | 7,498 | 327.1 s | 23 | 5.8 | 22.5 GiB | 22.6 GiB |
| 14 | 112,017 | | 104,144 | 7,873 | 352.1 s | 22 | 5.5 | 23.5 GiB | 23.5 GiB |
| 15 | 120,386 | | 112,012 | 8,374 | 392.6 s | 21 | 5.2 | 24.6 GiB | 24.6 GiB |
| 16 | 128,254 | | 120,381 | 7,873 | 394.6 s | 20 | 5.0 | 25.5 GiB | 25.5 GiB |

What the table shows:

- **Prefix reuse.** Every turn after the first reuses the whole previous prompt (cached = previous prompt - 5), and there were
  no refusals or device losses.
- **Memory.** It grows ~1 GiB a turn, the F32 KV's 128 KiB per token. These tables were measured with F32 KV; #1102 made
  F16 (64 KiB per token) the Vulkan default. Its table follows.
- **Prefill and decode speed.** Both fall with depth because the attention reads the whole F32 history. Prefill attention runs
  at ~1 TFLOP/s on this hd256 path. Narrower KV (design V2) and a faster hd256 prefill kernel (V3, #441) are the next steps.
- **Total time.** All 16 turns, from an empty session to 128k, took 63 minutes.

**With F16 KV (#1102, now the default).** The same growth run, 64 KiB per token, with the GQA-grouped split-KV decode pass.
The F32 arm is the same build with `--cache-type-k/-v f32`:

| turn | prompt | decode tok/s F32 | decode tok/s F16 | prefill tok/s F16 | private F32 | private F16 |
|---|---|---|---|---|---|---|
| 1 | 7,823 | 19.6 | 21.2 | 179 | 10.2 GiB | 9.7 GiB |
| 4 | 32,934 | 12.6 | 15.9 | 66 | 13.7 GiB | 11.6 GiB |
| 8 | 64,320 | 8.7 | 12.0 | 38 | 17.6 GiB | 13.6 GiB |
| 12 | 96,656 | 6.7 | 9.6 | 25 | 21.7 GiB | 15.8 GiB |
| 16 | 128,254 | 5.4 | **8.1** | 21 | 25.6 GiB | **17.7 GiB** |

- Memory grows 70 KiB per token with F16 against 134 KiB with F32.
- Decode at 128k is 1.5x the F32 arm and 1.6x the table above.
- Prefill barely moves: it is bound by the ~1 TFLOP/s hd256 attention (#441), not by KV bytes.
- The F16 arm ran a paired-load variant of the grouped pass that is ~12 % faster per layer but wrong on Intel and was
  withdrawn (#1113). With the shipped pass expect ~7.4 tok/s at 128k.

### Past 262,144 tokens: YaRN for Bonsai-2 / Qwen3.8 (#1101)

Bonsai-2 and Qwen3.8 train to 262,144 tokens and reach about 1M with opt-in YaRN (factor 4). Turn it on at load:

```
dotllm serve <bonsai> --device vulkan --no-mtp --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 --ctx-size 327680
```

The window becomes 1,048,576 (`factor × original`), and `--ctx-size` caps it: on CUDA the VRAM fit and `--ctx-size` work as
above, and on the Vulkan hybrid `--ctx-size` now lowers the window too (`[dotllm] Context window: 327680 of 1048576 tokens
(--ctx-size).`); before #1101 Vulkan ignored the flag. The KV caches grow up to that window and no further.

Cost: YaRN's rotation itself is free, but it also scales every position by `0.1·ln 4 + 1 = 1.139`, so short prompts produce
slightly different output. Measured on Bonsai-2 (CUDA, wikitext-2, 16 × 512-token windows): perplexity 10.8345 off vs
10.8297 on (±0.47), and greedy text that diverges after a few dozen tokens while staying coherent. Leave it off unless a
session will pass 262,144 tokens. The KV is the real cost: F16 (the
Vulkan default since #1102) 64 KiB/token (300k ≈ 18 GiB, 1M = 64 GiB), F32 128 KiB/token (1M = 128 GiB), q8_0 about
34 KiB/token (#1112). See `docs/POSITION_ENCODING.md`
for the math and `scripts/needle-in-haystack.py` for a retrieval check.

### MTP on prefix-hit turns (#1088)

The single-slot snapshot now keeps the MTP head that absorbed the prefix, so a hit rolls the head back with the trunk and
drafts from it (see `KV_CACHE.md`, "MTP on a single-slot prefix hit"). Two related changes:

- **The first turn snapshots too.** With no snapshot held, a prompt that shares nothing usable with the previous one (the
  first turn of a conversation) now snapshots at its own end. The second agent turn then hits: `cached_tokens` is the
  first prompt minus one, where it used to be 0.
- **Plain turns keep the head in step.** A turn that takes the snapshot carries the head through its prefill even when it
  decodes plain. The adaptive gate's size prior sends Bonsai-2's first request to plain decode, and before this fix that
  left every later turn without a head, so the gate never tried MTP for the whole session.

Measured on the T5500 (RTX 3060 12 GB, Bonsai-2 27B PQ2_0-MTP, `--cache-type-k q8_0 --cache-type-v q8_0`, fitted window
28,672 in both builds). The replay is an agent conversation: a system prompt, then a C# source file of about 4k tokens as
a tool result each turn, with the model's replies kept. Settings: `max_tokens` 1024, greedy, thinking off, streamed. MTP is
forced (`DOTLLM_MTP_ADAPTIVE=0`) to show what the head does on hit turns. "Tokens per round" is committed tokens per verify
round at K=3 (`3 x accepted / drafted`), so 1.0 means no draft was ever accepted.

| turn | before: cached | before: TTFT | before: ms/token | before: tokens/round | after: cached | after: TTFT | after: ms/token | after: tokens/round |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 19.6 s | 45.6 | 2.21 | 0 | 19.9 s | 43.0 | 2.41 |
| 1 | **0** | **57.4 s** | 48.0 | 2.32 | **2,927** | **38.3 s** | 42.4 | 2.68 |
| 2 | 7,551 | 29.1 s | 44.0 | plain | 7,911 | 34.2 s | 47.0 | **2.55** |
| 3 | 11,378 | 35.8 s | 45.8 | plain | 12,177 | 40.1 s | 46.7 | **2.74** |
| 4 | 15,871 | 32.0 s | 45.2 | plain | 17,090 | 36.0 s | 50.9 | **2.65** |
| 5 | 19,797 | 42.9 s | 45.9 | plain | 21,353 | 48.3 s | 47.0 | **3.07** |

- **Hit turns now speculate.** They accept as many drafts as a cold turn does or more (2.5-3.1 committed tokens per round,
  against 2.2-2.4 on the cold turns), so the head resumed from the snapshot is the right one.
- **Speculation does not pay on this card at these lengths.** The decode rate with MTP on hit turns (42-51 ms/token) is the
  same as with MTP on cold turns (43-48), but plain decode is no slower (41-46 ms/token in the `--no-mtp` run of the same
  build). The adaptive gate (the default) therefore tries MTP on the first hit, measures it, and falls back to plain. Its
  warm-up measures MTP faster at a 16-token context (33-35 against 42-46 ms/token), so the loss comes from verifying K+1
  rows against a long KV.
- **TTFT.** The turn-1 hit cuts its time to first token from 57 s to 38 s. Later turns are within run-to-run noise
  once the prompt lengths are matched: the replies differ between builds, so the prompts do too.
- **GPU memory.** Idle is unchanged at 9,379 / 76 MiB (dedicated / shared). Peak dedicated rose from 10,597 to 10,827 MiB,
  because the head's KV now stays allocated through hit turns. Peak shared was 78 MiB in both builds, so it never left idle.

With the adaptive gate (the default, no environment overrides) the same replay runs as follows. "Drafts" counts draft tokens
before the round controller fell back to plain decode. Before this change no hit turn could draft, and the size prior had
already sent the cold turns to plain.

| turn | before: prompt / cached | before: TTFT | before: ms/token | after: prompt / cached | after: TTFT | after: ms/token | after: drafts |
|---|---|---|---|---|---|---|---|
| 0 | 2,928 / 0 | 18.0 s | 39.6 | 2,928 / 0 | 18.6 s | 40.4 | 0 (prior: plain) |
| 1 | 7,646 / **0** | **51.3 s** | 39.5 | 7,912 / **2,927** | **36.3 s** | 41.7 | 307, then plain |
| 2 | 11,398 / 7,645 | 27.0 s | 42.3 | 11,856 / 7,911 | 30.8 s | 43.8 | 8, then plain |
| 3 | 16,143 / 11,397 | 35.7 s | 42.8 | 16,557 / 11,855 | 37.5 s | 44.6 | 14, then plain |
| 4 | 20,161 / 16,142 | 30.5 s | 44.1 | 20,820 / 16,556 | 35.3 s | 45.3 | 6, then plain |
| 5 | 25,423 / 20,160 | 42.1 s | 45.0 | 26,420 / 20,819 | 47.9 s | 45.8 | 12, then plain |

Peak GPU memory was 10,597 / 78 MiB before and 10,807 / 78 MiB after (dedicated / shared), against 9,379 / 76 MiB idle.
On a card where MTP pays at agent-length contexts, the hit turns now keep it.

**Time to first token on hit turns (PR #1091 review).** Feeding the head through a turn's prefill is not free: the CUDA
absorb measured about 0.2 ms per prompt token on this card (3% of the trunk's prefill). In the first push of #1091, every
plain turn that took the snapshot fed the head, so every hit turn paid it even after the gate had measured plain decode
faster. A plain turn now feeds the head only while the gate forecasts MTP for a coming request
(`MtpAdaptiveGate.LikelyUsesMtpNext`). Interleaved replay on the T5500 (default gate, q8_0 KV, MTP on; the assistant
replies in the history are fixed text, so every build sees identical prompts; mean ± SD of TTFT in seconds over 6 `dev`
runs and 4 runs of each branch build, alternating the build order every round):

| turn | prompt / cached | `dev` @ 257605eb | first push | with the gated feed | first push - `dev` | gated feed - `dev` |
|---|---|---|---|---|---|---|
| 0 | 2,928 / 0 | 19.1 ± 0.9 | 20.3 ± 0.3 | 20.1 ± 0.4 | +1.2 | +1.1 |
| 1 | 7,012 / 2,927 (`dev`: 0) | 51.5 ± 2.9 | 31.7 ± 0.4 | 31.6 ± 0.6 | -19.8 | -19.9 |
| 2 | 10,376 / 7,011 | 25.3 ± 0.9 | 26.7 ± 0.3 | 25.8 ± 0.4 | +1.4 | +0.6 |
| 3 | 14,388 / 10,375 | 30.0 ± 0.8 | 31.8 ± 0.2 | 30.6 ± 0.6 | +1.7 | +0.6 |
| 4 | 17,750 / 14,387 | 26.2 ± 0.7 | 27.4 ± 0.4 | 26.6 ± 0.6 | +1.2 | +0.3 |
| 5 | 22,449 / 17,749 | 37.2 ± 1.1 | 38.8 ± 0.5 | 37.6 ± 0.8 | +1.6 | +0.3 |

- Turn 1 now hits, 20 s sooner. Its MTP exploration still pays the head's absorb, which is why turn 0 (a cold turn that
  feeds the head so turn 1 can explore MTP) is about 1 s slower.
- Hit turns 2-5 are within run-to-run noise of `dev`. The gate measured plain faster after turn 1, so the session's
  later hits decode plain without a head, as on `dev`.
- Shared GPU memory stayed at idle (78 MiB peak against 76 idle) in every run.

### Turning sysmem fallback off in the driver

For a dedicated inference box, make the driver fail the allocation instead of spilling. In **NVIDIA Control Panel >
Manage 3D settings** (Global, or Program Settings for `dotllm.exe` / `DotLLM.Cli.exe` / the `dotnet` host that runs it),
set **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback** (driver 536.40 or later). With it set, an
over-allocation surfaces as `CUDA_ERROR_OUT_OF_MEMORY` and fails that request, which is what you want: a failed request
is visible and retryable, a spilled one silently takes hours. Leave it on *Driver Default* if other applications on the
same GPU rely on oversubscription. Linux drivers never spill this way.

## MTP self-speculation is on by default (#757)

When the loaded GGUF carries an embedded MTP head (`SupportsMtp`; e.g. Qwen3.6-MTP builds), `dotllm serve` enables MTP self-speculative decoding
automatically with the adaptive gate (it backs off to plain decode when drafts stop paying) - no flag needed. It engages only when **all** hold:

| Condition | If not met |
|-----------|-----------|
| the GGUF has an MTP head | silently not applicable (a model without a head logs nothing; `--mtp` on such a model logs that it is ignored) |
| no `--speculative-model` (external draft) | the draft model wins; startup log says MTP was skipped |
| `--expected-concurrency` < 5 | the continuous-batch scheduler wins (+73% aggregate decode at 8 streams); startup log says MTP was skipped |
| `--no-mtp` not given | MTP off; startup log says it was disabled |

The startup log always says whether MTP was enabled automatically or skipped, and why. **Tradeoff:** while MTP is active the continuous-batch
scheduler is off for that model (MTP's single-sequence speculative loop does not batch requests), so for many concurrent clients pass `--no-mtp`
or `--expected-concurrency 5` (or more). `--mtp` is still accepted for back-compat; it only changes the log wording and does **not** override the
rules above. `GET /props` reports `mtp_active` and `mtp_status` (`active`, `off (--no-mtp)`, `unavailable (no MTP head)`, `skipped (...)`); the
web UI shows the same on the model badge. `POST /v1/models/load` accepts `"mtp": false` to opt out for one load.

**qwen4exp (Qwen3.8-Flash-Next) ships its head as a separate `mtp-*.gguf`** beside the quantisation folders (#820). `serve` attaches an
auto-detected sibling head when it fits the device (measured on Vulkan / Strix Halo with the 2..8-row GEMVs of #902: every available quantisation wins with
speculation, UD-IQ4_XS and ISTA GSQ-RCO 1.3-1.5x, and the head costs ~2 GiB of device memory); `--mtp-head <path>` names a head explicitly. A head that
does not fit is declined with a startup message and serving continues without speculation. Sampled requests
(`temperature > 0`) speculate too, with exact speculative sampling (the output is distributed as plain sampling; greedy output is the verify path's greedy, which
equals plain decode up to near-tie flips from the multi-row kernels, see SPECULATIVE.md). The adaptive gate keeps separate greedy and sampled arms, measures whole-round wall time, adapts the draft length K per model, drops to plain
decode inside a request whose measured rounds lose, and re-probes a losing arm with exponential back-off. See [SPECULATIVE.md](SPECULATIVE.md) (#900).

`GET /v1/models/inspect` returns `has_mtp` (true for a qwen35 checkpoint whose trailing block really carries the `nextn.*` tensors). The web UI's
Load Model modal uses it: for such models the draft-model picker is hidden and replaced with "MTP: built into this model - enabled automatically"
plus an opt-out checkbox; for models without a head the picker is offered as an optional **external draft (speculative) model** (unselected by default).

The modal also defaults the compute device to the best servable GPU (`GET /v1/devices` -> `recommended_device`, same ordering as `--device auto`;
optionally `?model_bytes=N`), and to CPU only when no GPU is servable. An explicit GPU device that fails to load returns an error - there is no
silent CPU fallback (only `--device auto` falls back, with a prominent `device_fallback_warning` in `/props`).

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

The `response_format` field maps to constrained decoding (on a thinking model it also turns thinking off by default and disables the
reasoning split for that request — see [Reasoning / thinking models](#reasoning--thinking-models-767)):

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

## Vulkan device (`--device vulkan`)

`dotllm serve --device vulkan` (and `dotllm run --device vulkan`) load the model through
`VulkanModelLoader.CreateSharedFromGguf`, the same per-architecture dispatch `bench`/`perplexity` use, on a
process-wide Vulkan device. Requests run one at a time through `TextGenerator` with the model's own device-resident
KV-cache: no `ForwardBatch` scheduler, no paged/quantized KV, no cross-request prefix cache yet. Before this,
`--device vulkan` silently fell back to the CPU (the string does not start with `gpu`); `chat` now rejects it
explicitly. Measured on Strix Halo with Tev1-4B Q4_K_M (108-token decision prompt, 1 answer token):
~0.3 s/request on Vulkan vs ~5.7-9 s on CPU (#613).

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
| `--prompt-cache-step` | 0 (off) | Hybrid recurrent models: checkpoint the recurrent state every N prefill tokens, so a prompt that diverges mid-history resumes from the nearest one ([KV_CACHE.md](KV_CACHE.md#mid-history-resume-step-checkpoints-1286), #1286) |
| `--prompt-cache-step-keep` | 8 | Most `--prompt-cache-step` checkpoints kept (deepest first) |

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

1. **Continuous-batch scheduler path (default for paged-KV serving)**. When `--paged` is on (the default for `serve`) and no speculative-decoding draft model is loaded, `ServerStartup` constructs a `ContinuousBatchSchedulerService` per loaded model and starts its `RunLoopAsync` on a background task tied to `IHostApplicationLifetime.ApplicationStopping`. `/v1/chat/completions`, `/v1/completions`, `/v1/messages` and the Ollama `/api/generate` / `/api/chat` route requests through it — non-streaming via `EnqueueAsync`, streaming via `StreamAsync` (#1266), which yields each token as the run loop samples it from an unbounded per-request channel, so a slow client never stalls the loop. Multiple concurrent requests, streaming or not, pipeline through a single `IModel.ForwardBatch` dispatch per scheduler iteration; neither path takes the request gate or the model lease. The startup log prints `Continuous-batch scheduler active` when this path is engaged.
2. **Single-request gate path (fallback)**. LoRA-adapter requests, logprob-capturing requests, and any backend without a paged KV-cache factory (CUDA, Vulkan, hybrid GPU, quantized KV) keep using the original `SemaphoreSlim(1, 1)` gate via `ServerState.ExecuteAsync`. Requests serialize FIFO. The startup log prints `Single-request mode — requests processed sequentially` when this is the only path.

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
| Messages count | > `--max-messages` (default 8192; 0 = unlimited) | 400 `"messages array exceeds maximum of N messages (got M); raise it with --max-messages or DOTLLM_MAX_MESSAGES (0 = unlimited)"` (Anthropic route: `"messages: array exceeds ..."`) |
| Empty prompt (completions) | empty/null | 400 `"prompt must not be empty"` |
| `max_tokens` | &le; 0 | 400 `"max_tokens must be a positive integer"` |
| Prompt token count | &ge; `MaxSequenceLength` | 400 `"prompt (N tokens) exceeds model context length (M)"` |
| `prompt_tokens + max_tokens` | > `MaxSequenceLength` | `max_tokens` silently clamped to remaining context |

### Configuring limits

The message-count cap applies to both `/v1/chat/completions` and `/v1/messages` (#835). It defaults to 8192 (it was a
hard-coded 1024). Set it with `dotllm serve --max-messages <n>` or the `DOTLLM_MAX_MESSAGES` environment variable; the flag
wins over the variable. `0` removes the cap; a negative or non-integer value fails at startup. It is only a cheap structural
bound - the prompt-length-vs-context check remains the real protection, so long agent conversations are limited by the
model's context window, not by this count.

### Vulkan continuous-batch scheduler (hybrid GDN models)

Vulkan hybrid models (Qwen3.5 / Tev1, Qwen3.6) serve through the serial per-request generator by default. Pass `--expected-concurrency N`
(N >= 5) to serve them through the continuous-batch scheduler instead: measured on Tev1-4B, aggregate decode throughput at 8 concurrent
streams is +73% (102 vs 59 tok/s), parity at 4 or fewer and for a single stream. It is off when MTP (default-on for embedded-head models) or a draft model is active, and a
repeated identical prompt is slower than on the serial path (which has the whole-prompt cache). `DOTLLM_VK_SCHEDULER=0|1` overrides the hint.
# Multi-session recurrent prefix caching (#1033)

For serialized agent workloads that alternate between conversations, configure independent
recurrent-prefix snapshots rather than repeatedly prefilling each full conversation:

```text
dotllm serve <model.gguf> --prefix-snapshots 20 --prefix-snapshot-host-budget 8GiB
```

`--prefix-snapshots` bounds the number of retained entries (default 1).
`--prefix-snapshot-host-budget` bounds page-locked host retention (default 0, disabled),
accepting bytes or units such as `MiB` and `GiB`. The same flags are accepted by the
standalone server and interactive `chat`. Snapshot transport must be supported by the
loaded model; inspect `GET /v1/prompt-cache` for the effective state and occupancy.

For example, a 30,000-token prefix at 64 KiB of KV per token plus 150 MiB of recurrent
state occupies about 1.98 GiB. An 8 GiB host budget holds approximately four such snapshots;
the exact count depends on KV quantization, layer geometry and prefix lengths. More tracked
entries do not imply that all of them remain resident when the byte budget is exhausted.

The streaming `timings` extension reports `cache_tier` and `cache_restore_ms` alongside
`cached_tokens`. OpenAI usage responses include `prompt_tokens_details.cached_tokens`.
The prompt-cache administration endpoint reports recurrent entries, host bytes, hits by
tier and restore failures. Clearing the prompt cache also clears recurrent snapshots;
model changes invalidate them. Disk retention is the separate #1034 follow-up.

### Recurrent prefix disk spill

Add `--prefix-snapshot-disk-dir PATH --prefix-snapshot-disk-budget 40GiB` to the
multi-slot/host configuration above to enable bounded ephemeral spill. The directory
and positive budget are required together; decimal and binary byte suffixes use the
same syntax as the host budget. Startup logs the implemented disk transport once.
The current `file-stream` fallback uses pinned CUDA copies where the provider supports
them; native DirectStorage/cuFile paths are not implemented.

Streaming timings identify disk hits with `cache_tier: disk`; cached tokens also appear
in OpenAI `usage.prompt_tokens_details.cached_tokens`. `/v1/prompt-cache` exposes disk
occupancy, budget and hit counts. Clearing the cache or unloading its model removes its
owned files. Files are private to the cache instance and are not reused after restart;
see [KV cache disk spill](KV_CACHE.md#disk-spill-below-the-host-tier-1034) for transient
budget headroom and crash-orphan cleanup.
