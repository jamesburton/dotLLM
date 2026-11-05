# Chunk cache (server side)

Server-side notes for the `chunk_cache` wire convention (chunk-level K/V reuse for agent harnesses). The convention, its
fields and its error codes are specified outside this repo (ChunkedkvCache, spec `2026-11-02-chunk-cache-integration.md`,
wire contract `docs/contracts/chunk-cache-v1/`). This page records what the dotLLM server does; it grows with each server
task. Everything the convention adds sits behind `dotllm serve --chunk-cache` (off by default); with the flag off, every
request and response is what it was before the convention, for any body.

## Switches

| flag | default | meaning |
|---|---|---|
| `--chunk-cache` | off | honour `chunk_cache` annotations on `/v1/chat/completions` and `/v1/messages`: policies `exact` (default) and `shadow`, whose output is identical to the unannotated request, plus a `chunk_cache` report in the response (below). Off: the annotations are never read; the body behaves exactly as without them (pinned over HTTP by `ChunkCacheFlagHttpTests` and `FlagOff_AnnotatedBodies_AnsweredLikePlain_EverySurface`, blocking and streaming, both surfaces) |
| `--chunk-cache-max-chunks <n>` | 4096 | most annotations read per request (1 to 65536, so the per-request parse work stays bounded); the rest are counted from the array length and dropped with `chunk_too_many` without being read |
| `--chunk-store <dir>` | none | the chunk store (e.g. built by `chunkcache build` from the same GGUF, without `--strict-model-id`) the server looks annotated chunks up in. **Read-only**: the server never writes, encodes into or creates it (the sole-writer encode path, its budget and admission rules come with a later task). Needs `--chunk-cache` (rejected without it); a relative path is resolved at startup. No store: every verified chunk is a miss, no warning. A configured root that does not exist: misses plus `chunk_store_unavailable` (strict: 503) |

All three exist on `dotllm serve` and on `ServerOptions.Parse` (`ServerOptions.ChunkCache`).

## Request parsing (`ChunkAnnotationParser`)

The parser reads the request and nothing else: no store, no tokenizer, no reuse. It turns the annotations into
`ParsedChunkRequest` (accepted spans, warnings with counts, the strict-mode error, the request options), or returns null
when the flag is off or the request carries no annotation. Both chat endpoints call it right after request validation;
a strict-mode defect is returned there (see "Execution" below for the error shapes).
**Packaging:** this step does not ship alone: the parser, span location and `exact` / `shadow` execution (plan Tasks 10,
11 and 13) go upstream as one PR, so the flag does something when it is merged. The parser and its records are
`internal` until then.

### Where a span can be

| surface | placement | addresses |
|---|---|---|
| OpenAI `/v1/chat/completions` | manifest `chunk_cache.chunks[]`, `at: {message, part: 0[, start, end]}` | a message's **string** `content`. Array content stays the binding 400 with the flag on (general array support is a separate issue). A `chunk` field on a message object is an unknown field, ignored |
| Anthropic `/v1/messages` | inline `chunk` on a `text` block, or on a text block inside a `tool_result`'s content array | that block's text |
| Anthropic | manifest, `at: {message, part[, start, end]}` | a top-level `text` block, string message content (`part: 0`), or a `tool_result` with **string** content. A `tool_result` with array content, `tool_use`, `thinking`: not a text part (`chunk_span_invalid`); its inner blocks are reachable inline only |

`message` / `part` index the request's own `messages` and `content` arrays (the Anthropic top-level `system` is not
addressable). `start` / `end` are UTF-8 byte offsets into the decoded text, end exclusive; omitted means the whole part.

Each accepted span also carries `EngineSpan(EngineMessage, StartChar, EndChar)`: where it lies in the converter's
`ChatMessage` list (before `ProfileSystemPrompt.Apply` may prepend a profile system message), in UTF-16 chars. On
Anthropic the converter records these positions itself while it joins (`AnthropicTextOrigins`), so they follow its rules
exactly: text blocks joined with one `'\n'` (written only once text has accumulated), a `tool_result` ending the turn and
becoming its own `tool` message, the text after it starting a new message at offset 0, a `tool_result` content array
joined with `'\n'`. The parser then checks that the engine message holds the span text at that position.

### Validation, in order (one defect per annotation; contract rulings 23–35 and the Task 10 review amendments)

1. Request level. Structural defects (ruling 23) are `chunk_annotation_malformed`: `chunk_cache` not an object or `chunks`
   not an array (nothing accepted); a wrong JSON type for `v`, `strict`, `policy`, `namespace`, `report`, `on_miss`,
   `min_chunk_tokens` (or a negative value) or `query`; a string System.Text.Json cannot decode; an unknown `report` or
   `on_miss` value, or a `query.message` that is not the index of a request message (ruling 34; the default applies).
   `v` other than 1 →
   `chunk_version_unsupported` (nothing accepted); an unknown `policy` name → `chunk_policy_unavailable` (`exact`
   applies, spans kept); a `namespace` other than the caller's → `chunk_namespace_ignored` (spans kept). Inline and
   manifest placements together →
   `chunk_manifest_conflict`, served unannotated (an options-only `chunk_cache` next to inline annotations is not a
   conflict). More than `--chunk-cache-max-chunks` annotations → `chunk_too_many`: the first *n* in document order are
   read, the rest are counted from the array length and never read.
2. Per annotation, in this order (ruling 33):
   1. **structure** → `chunk_annotation_malformed`: not an object; inline without `v`; `v` not a number; `id` or `hash`
      missing or not a string; `chunker` / `positioning` / `stability` not strings; `no_store` not a boolean; `at` not
      `{message, part[, start, end]}` with integers; any string that cannot be decoded (a lone-surrogate escape);
   2. **version** → `chunk_version_unsupported`: `v` ≠ 1, or `positioning` other than `isolated`;
   3. **id** → `chunk_span_invalid`: not 1–256 characters (Unicode scalar values);
   4. **eligibility** → `chunk_not_wire_eligible`: a `chunker` outside the wire-eligible list (`symbol-512` today); the
      span is dropped (rulings 26, 35: a warning, 400 when strict) and reserves no range;
   5. **range / boundary** → `chunk_span_invalid`: message or part out of range, a non-text part, an empty or
      out-of-range byte range, an offset inside a multi-byte character;
   6. **overlap** → `chunk_span_invalid`: overlaps an earlier span of the same part (the later one is dropped; a span
      that then fails its hash still holds its range);
   7. **hash**, last → `chunk_hash_mismatch`: not `"sha256:"` + lowercase hex of the span's UTF-8 bytes (decoded JSON
      text, no normalisation: CRLF stays CRLF);
   8. a span the engine message does not hold at the mapped position → `chunk_span_invalid` (defect kind
      `not_carried`; a converter/mapping bug guard).
3. Non-strict: one warning per code with the number of annotations (or requests) it covers; a dropped span is plain
   text. `strict: true`: `Error` is the first defect in document order (amended ruling 4: request-level first, then
   inline annotations in (message, part) order, then manifest entries in array order): `400 invalid_request_error` with
   the code (`chunk_store_unavailable` alone is `503 api_error`; nothing is 422). Only `chunk_ignored_too_short`,
   `chunk_encode_skipped` and `chunk_encode_queued` are never errors.

**Lone surrogates (ruling 17).** A string that holds a lone surrogate (in-process callers) hashes as UTF-8 with U+FFFD.
Over HTTP a lone-surrogate *escape* (`"\ud800"`, valid JSON) is a client error, flag on or off: System.Text.Json parses
it but refuses to return it as a string. OpenAI message text fails at model binding (400). Anthropic message text is a
raw `JsonElement` that the converter reads later; `/v1/messages` and `/v1/messages/count_tokens` catch that refusal
(around the `chunk_cache` parse, which converts the messages first, and in prompt building) and answer
`400 invalid_request_error` (`Anthropic_LoneSurrogateEscapeInTextBlock_Is400`: annotated, unannotated, flag off; before
the fix all three were a 500). An annotation field with one is `chunk_annotation_malformed`, never an exception.

**Cost bounds.** The parser never throws on body content and does bounded work: the manifest is cut at its array length
before any entry is read; each annotated part is encoded to UTF-8 once per request, with a byte-to-char checkpoint table
(one per 1,024 chars); overlapping spans are rejected before hashing, so the bytes hashed per request never exceed the
request's own text (4,096 whole-part spans on a 4 MiB part hash once: `CostBound_*` tests). An unannotated Anthropic
request is detected by a pre-scan and never converted twice.

**`report: per_chunk`** is accepted from any caller here; restricting it to an authenticated caller in its own
namespace (§10) is the authentication task's job.

Request options: `policy` (`exact` default; `balanced` / `fast` / `auto` parse to their own values and are degraded
later), `strict`, `report` (`summary` default), `min_chunk_tokens` (32), `on_miss` (`prefill`), `query` (`{text}` or
`{message}`; default: the last engine message with role `user`, so Anthropic tool results never count as the query).
A valid `query.message` that points at a message with no text parts (e.g. an assistant turn with only tool calls) gives
an empty query and no warning (ruling 34 asks only for the index of a request message).
**Namespace.** The store namespace is always server-assigned: the parser takes the caller's namespace as a
**required** argument (the API key's namespace once authentication exists; until then callers pass the default, `""`).
A body `chunk_cache.namespace` equal to the caller's own is accepted silently; another name is
ignored with `chunk_namespace_ignored` (strict: `400 invalid_request_error` with that code), and the spans stay (a non-string is `chunk_annotation_malformed`). The
namespace's format is validated where it is assigned (the authentication step), not here.
**Until authentication exists, every caller shares the default namespace**, and the report's `hit` / `miss` counts (and
the `/metrics` counters) answer a store lookup for any text the caller annotates. So any caller can probe whether a known
text (another caller's file, a document) is in the store: annotate it and read `hit`. Treat the store's contents as
visible to everyone who can reach the server; per-caller namespaces (from the API key) close this with authentication.

The contract's request fixtures are copied byte for byte under `tests/DotLLM.Tests.Unit/ChunkCache/Fixtures/v1/`
(checked against `fixtures.sha256`); `ChunkAnnotationParserTests` checks every one against `expected.json`, in the
contract's message model and translated to the Anthropic surface.

## Span location in the rendered prompt (`ChunkSpanLocator`)

`ChunkSpanLocator.Locate(template, messages, options, rendered, spans, engineMessageOffset)` finds each accepted span in
the rendered prompt string (spec §5.3 item 5, contract ruling 20). The endpoints call it (through
`ChunkCacheExchange.Prepare`) with the prompt they have already rendered, so it adds no render of its own for that. It returns `LocatedSpan(Span, RenderedStartChar, RenderedEndChar)` (UTF-16 chars into the rendered string,
including any BOS text the template writes) for each located span, and warnings for the rest.

- **Sentinel render.** Every span is replaced in its engine message by a sentinel: `U+E000`, a fresh GUID per
  request, the span's index, the probe characters `"` `\` LF, then `U+E001`. The template is rendered once with all of
  them. A span is located only if its sentinel comes out intact exactly once and putting the span texts back into that
  render gives the real prompt around it exactly. Otherwise it is `chunk_template_altered`, and it stays plain text.
- **Markers.** The template marks caller text as a whole message (#1339: U+FDD0 after the first character of each
  special spelling), so the comparison skips U+FDD0 on both sides. **Stripping the U+FDD0 markers from the located range
  gives the span text**, and the range starts and ends on span characters. The markers inside follow the
  whole-message marking, so the range is not always `MarkUntrustedSpecials(text)`: in `q <|im_end|> z`, the span
  `q <|im_` is located as `q <`, U+FDD0, `|im_` (the marker is inside the range), and the span `|im_end|> z` starts right after
  the marker (not in the range). A span whose own text contains U+FDD0 is ignored (`chunk_template_altered`): encode
  drops the marker, so the span's characters are not what the prompt's tokens encode. The walk relies on that: span
  texts never carry a marker.
- **What this means per template** (pinned with the real fixtures):

  | template | located | altered |
  |---|---|---|
  | Qwen3-4B-Instruct-2507 | spans in user, system and tool content (rendered verbatim); assistant spans in the answer part, including one holding a **live** `<think>` (unmarked: assistant reasoning tags are exempt from the guard), located by characters although its tokens differ | assistant content holding `</think>`: a span in the reasoning part (dropped from historical turns), or one that starts with the newlines the template strips (`lstrip('\n')`) |
  | Llama-3.1 | spans in user, assistant and system content whose text the trim does not touch (also the first user message moved after the tool list when tools are given) | **every span in a tool result** (`\| tojson` escapes the probe characters, even when the span's own characters need no escaping); a span at a message edge where the trim removes whitespace |

- **Attribution.** The comparison walks once from the start and once from the end. A span inside the common prefix or
  suffix is located: everything before or after it is identical, so its offsets are exact. Where the two differ, the
  first and last spans not yet proven are ignored, and the spans between them are tried again in another render.
  Spans whose sentinel broke are dropped from that render as well. This is conservative: a span is never located on a
  guess, and an innocent span between template-altered ones may be ignored once the render budget is spent.
- **Cost.** At most `MaxSentinelRenders` (4) sentinel renders per request, whatever the number of spans. Each pass is
  linear in the prompt length, plus O(spans log spans): it never renders once per span. 4,096 spans on a 4 MiB message
  take one sentinel render (`CostBound_ManySpans_BoundedRenders`).
- **Profile system prompt (ruling 31).** Engine positions are counted before `ProfileSystemPrompt.Apply` prepends a
  message. The caller passes `engineMessageOffset` (the number of messages prepended), and the parameter is required.
  A span that is not at its stated position is `chunk_span_invalid`, never located elsewhere.
- **Never throws on request content.** If a template throws on the sentinel render, or drops or duplicates content,
  every span it covers is ignored. Positions out of range, overlaps and lone surrogates are handled without a throw.
- Character equality is not token equality at a span's edges (BPE merges with neighbouring template text; a
  SentencePiece leading space; live reasoning tags in assistant turns). Execution (below) checks the span's token ids
  against the whole-prompt tokenisation over the located range before counting a span as reusable.

## Execution: `exact` and `shadow` (`ChunkCachePipeline`, `ChunkCacheReport`)

Only `exact` (the default) and `shadow` run. **Both generate exactly what the unannotated request generates**: the prompt
is rendered and tokenised as a plain request (spec §5.3 item 7), the same dispatch runs (scheduler or generator), and the
prompt cache is looked up with exact lineage only (Task 12: a snapshot's approximate rows are never restored, so
`cached_tokens` / `timings.cache_n` keep meaning exact prefix reuse). The annotations only produce counts, and under
`shadow` a plan. Pinned over HTTP on a synthetic model, blocking and streaming, on three prompt-cache configurations
(single-slot prefix cache; the default paged KV + scheduler; paged + scheduler with the prefix trie wired in):
`Exact_IsBitIdenticalToUnannotated`, `Exact_AfterShadow_SameSlot_Identical`, `CachedTokens_UnchangedByAnnotations`,
`Exact_WithApproxSnapshotOnSlot_RestoresExactRowsOnly`. Note: `ServerStartup` builds the prefix trie but does not pass it
to the scheduler, so the default paged path has no cross-request prefix reuse at all (`cached_tokens` 0, annotated or
not); the tests wire it in to check warm-slot neutrality on the trie path too.

- **Policies.** `balanced`, `fast` and `auto` parse and are served `exact` with `reason: "chunk_policy_unavailable"`
  and that warning once (rulings 42; strict: 400) until applied reuse ships. On a model or backend that cannot assemble
  chunks (recurrent or hybrid layers, a diffusion model, a backend other than CPU or Vulkan, a RoPE variant the
  re-rotation does not reproduce, or a model whose identity cannot be computed) every annotation is ignored with
  `chunk_model_unsupported` (strict: 400); `reason` carries it when the requested policy was not `exact` (ruling 41).
  Diffusion models get no report.
- **Per located span**, in prompt order, at most `MaxPlannedSpans` (1,024) per request:
  1. ids = `ChunkTokenization.Encode(text)` (`EncodeRaw(MarkUntrustedSpecials(text))`, ruling 21);
  2. fewer than `min_chunk_tokens` (default 32) ids: `chunk_ignored_too_short` (warning only);
  3. **token check**: the span counts only if those ids occur in the whole-prompt ids exactly where its characters are.
     A span the check finds different is `chunk_template_altered` (strict: 400): a span glued to the text before it where
     the tokenizer merges across the edge, a SentencePiece leading-space piece on the span itself, a Qwen3 assistant span
     holding a live `<think>` (`Span_TokenIdsDifferAtBoundary_NotCountedReusable`, `AssistantSpanWithThinkTag_VerifiedByIds`).
     How: with a token boundary known at the end of the previous verified span, the glue up to the next span is encoded
     alone and compared with the prompt's ids; a tokenizer that adds a leading piece to every encode (SentencePiece's
     dummy space prefix) is handled by accepting the glue when everything after its first piece matches within two ids
     of the boundary (`SentencePieceLeadingSpace_ManyValidSpans_AllVerified_StrictOk`: 150 spans, no prefix encode after
     the first). Otherwise the prefix up to the span is encoded (as the prompt) and must be a prefix of the prompt's ids,
     which proves or disproves a token boundary at the span start. Prefix encodes are capped at 4 × the prompt length
     (+64 Ki chars). A span the check cannot place once that cap is spent, and every span past `MaxPlannedSpans`, is
     ignored with **`chunk_verification_skipped`** (ruling 49: a warning only, never a strict error; never reported as
     altered);
  4. store lookup (`IChunkStore.Contains`, which reads at most the first 8 KiB of an entry: its header) under
     `ChunkKey` of the server's identity (below), `isolated` positioning and the caller's namespace (server-assigned; the
     default namespace until authentication exists). A store failure is a miss.
- **Cost before generation.** All of this runs on the request thread before the scheduler queue or model gate, under
  `exact` too (ruling 5 counts hits): up to 4 sentinel renders (locator), one extra whole-prompt encode, the glue and
  span encodes (linear in the prompt), prefix encodes up to the cap above, and at most 1,024 store header reads
  (`ManySpans_CappedAtMaxPlannedSpans_RestSkipped`). Concurrent annotated requests therefore compete with inference for
  CPU; the prompt itself is already bounded by the context-length check.
- **Identity.** `ServerModelIdentity`: content model id (`GgufIdentity.ModelId`, sampled), tokenizer id
  (`GgufIdentity.TokenizerId`), quantisation label (`GgufIdentity.QuantizationLabel`, the helper the CLI's label must
  use too), backend tag (`cpu` / `vulkan`; a Vulkan model with layers left on the CPU is still `vulkan`) and stored K/V
  type F32, folded into the same fingerprint (`ChunkFingerprint.Compute`) the `chunkcache` CLI uses. A CLI-built entry is
  a hit here and nothing else is (`ShadowKey_ChunkWithThinkTag_HitsStoreEntryFromCli`; `ServerIdentity_EqualsCliModelHandle`
  pins model id, tokenizer id, label, fingerprint and producer against the CLI's own model handle). Computed once per
  loaded model under a lock; a failure (unreadable or malformed GGUF) is cached as unsupported, never a 500.
- **Where the server and a store silently disagree (always misses, never false hits).** The key covers every input that
  changes K/V, so these give `miss` for every chunk with no warning:
  - a store built with `chunkcache --strict-model-id` (full-file hash; the server always uses the sampled id);
  - a server started with `--rope-*` overrides or another context configuration that changes the RoPE fields of the
    fingerprint;
  - a CPU-built store served on Vulkan, or the reverse (the backend tag is in the key and the entry header);
  - a quantisation label of `unknown` on one side only.
  To diagnose, compare the fingerprint inputs. The capabilities endpoint (next task) must expose them: `model.id`
  (`GgufIdentity.ModelId`, and whether it is sampled or full), `model.tokenizer_id`, the quantisation label, the backend
  tag, the K/V dtype (F32), `model.producer` (the producer hash, the key of the empty sequence in the caller's namespace,
  which folds in the model fingerprint, RoPE parameters, schema version, backend, dtype and namespace) and the RoPE
  parameters (`rope_dim`, `theta`, pairing) that enter the key.
- **Strict mode.** Parser defects are returned before rendering. After it, the error is the first of the request-level
  defects (`chunk_model_unsupported`, `chunk_policy_unavailable`, `chunk_store_unavailable`), else the span-level defect
  of the span first in document order (ruling 4; locator and token-check defects alike). Warning-only codes
  (`chunk_ignored_too_short`, `chunk_verification_skipped`) never become an error. OpenAI: the usual error body with
  `code` (and `param: "chunk_cache"`); 400 `invalid_request_error`, or 503 `api_error` for `chunk_store_unavailable`.
  Anthropic: the usual envelope with `error.code` (ruling 29; omitted on every other error).
- **Never throws on request content.** An unexpected failure while planning ignores every annotation (the request is
  served as plain) and logs only the exception type; reports and logs never carry chunk text, hashes or namespaces.

### The response `chunk_cache` object (spec §5.9; contract rulings 5–10, 46–50)

Written on annotated requests only (flag on), unless the request asked for `report: "none"` (`per_chunk` gives the
summary for now: per-chunk reports need authentication, ruling 45). Where: the blocking response (both surfaces), the
OpenAI streaming **finish-reason chunk**, the one that carries `timings` (ruling 47; dotLLM emits `usage` on a separate
trailing `include_usage` chunk, which stays SDK-shaped), and the Anthropic `message_delta` event beside `usage` (whose
fields are unchanged: no `cache_read_input_tokens`, rulings 40 and 46). Shapes match the contract fixtures
(`Report_PartitionHolds`):

> **Contract change (v1, ruling 46).** Ruling 9 originally had the Anthropic `message_delta` usage carry
> `cache_read_input_tokens` (= `prefix_reused`). Ruling 40 (an annotated request's `usage` must equal the unannotated
> one's, and dotLLM's Anthropic usage never carried the field) prevailed; ruling 46 amended ruling 9 and removed the field
> from `response-anthropic-delta.json` in the contract folder itself (ChunkedkvCache `cdda32e`). Our copy of that
> fixture, its `fixtures.sha256` line and the pinned list hash are a re-copy of the amended contract set (the client
> tests on `dev` carry the same list hash since `b0728217`), not a local edit. `check.py` still accepts the field when
> present, provided it equals `tokens.prefix_reused`.

- `chunks`: `annotated = prefix + hit + miss + ignored`. `prefix`: a verified chunk wholly inside the exact prefix the
  prompt cache restored. `hit` / `miss`: the store lookup, under `exact` too (ruling 5). `ignored`: everything else
  (dropped by the parser, altered, too short, skipped by the verification budget, unsupported model).
- `tokens`: nothing is applied in v1, so `prompt = prefix_reused + prefilled`, `prefix_reused` = `cached_tokens`
  (exact rows), `prefix_reused_approx`, `chunk_reused`, `chunk_recomputed`, `encode_queued` are 0. `timings` gains
  `chunk_n` and `chunk_recomputed_n`, present (0) on annotated requests only.
- `shadow` adds `plan` (ruling 7) with its own partition: hits outside the prefix become `chunk_reused`, the first
  ⌈budget × n⌉ rows of each are `chunk_recomputed` (`selector: "chunk_starts"`; `budget` 0.05 by default, or the
  request's `chunk_cache.recompute.budget`, a number from 0 to 1, ruling 50; any other value is
  `chunk_annotation_malformed` and the default applies), `predicted_ttft_ms` = assembly + recompute (µs per recomputed
  row × visible key) + prefill of the rest, and `full_prefill_estimate_ms` as below. The constants are provisional
  (`ChunkCache/chunk-cost-model.json`, embedded, with sources: Vulkan 16 µs per row × key from G2, CPU from E1); they
  are a prediction, calibrated later.
- `timings_ms`: `assembly`, `probe`, `recompute` 0; `prefill` measured; `full_prefill_estimate` is an **estimate** of the
  unannotated request's prefill of the non-prefix rows (ruling 48): at this request's own measured prefill rate (v1
  applies no chunk reuse, so that is the unannotated rate and the estimate equals `prefill`), or at the backend's
  default rate when nothing was measured. A policy that applies reuse must never use its own measured rate here.
- `warnings`: parser, locator and execution warnings merged per code with counts, in order of first occurrence
  (ruling 38).

### `/metrics`

`dotllm_chunk_cache_chunks_total{status}` (`prefix`, `hit`, `miss`, `ignored`) and
`dotllm_chunk_cache_tokens_total{kind}` (each key of `tokens`), counted for every annotated request (also with
`report: "none"`). Labels carry no namespace, text or hash.

### Packaging and dependencies

`DotLLM.Server` references `DotLLM.ChunkReuse`, which brings `Microsoft.CodeAnalysis.CSharp` (Roslyn, for the symbol
chunker) and `DotLLM.Vulkan` into the dependency graph of the server package and of the `dotllm` CLI. With the flag off
the server never constructs the chunk-cache runtime (it is created on first use), so the store and key code is not
reached; whether the CLR loads the `DotLLM.ChunkReuse` assembly at all with the flag off has not been measured. The
opt-in AOT publish of `dotllm` (docs/AOT.md) has not been re-verified with this reference. Before upstreaming, the key,
store, tokenisation and identity types are candidates for a small assembly without Roslyn or Vulkan.

## Pinned server behaviour (before the convention)

What the server does **today** with the three things the convention leans on. Each fact is pinned by a test in
`tests/DotLLM.Tests.Unit/Server/ChunkCache/ServerPinTests.cs`, which passes on unmodified `dev`. If a pin breaks, the
convention's assumptions moved: update this section with it.

Templates are the GGUF `tokenizer.chat_template` of `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` and
`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (the repo fixtures `qwen3-4b-instruct-2507-chat-template.jinja` and
`llama-3.1-chat-template.jinja`, checked byte-identical to the GGUF metadata).

### Multi-part content

| surface | behaviour | evidence |
|---|---|---|
| OpenAI `/v1/chat/completions` | `messages[i].content` is a string only. An array of content parts fails model binding: **400 with an empty body** (no OpenAI error envelope); the handler never runs. | `ChatMessageDto.Content` is `string?` (`Models/ChatCompletionRequest.cs`) |
| Anthropic `/v1/messages`, `count_tokens` | `text` blocks of one message are joined with **one `'\n'`** before the template sees them. The separator is written only once text has accumulated: `["", "b"]` → `b`, `["a", "", "b"]` → `a\n\nb`. | `AnthropicConverter.AppendMessage` |
| Anthropic `tool_result` | Ends the current turn: text before it becomes its own message, the result becomes a `tool` message (array content joined with `'\n'`), text after it starts a new message. An Anthropic (message, block) index does not map 1:1 onto the rendered messages. | `AnthropicConverter.FlushPending`, `ConcatTextBlocks` |
| Anthropic `system` array | Joined with `'\n'`. | `AnthropicConverter.ExtractText` |

Consequences for span location (spec §5.3 item 5):

- On the OpenAI surface only part 0 exists, and the inline `chunk` placement on a text part cannot be sent at all. A
  manifest span (`at: {message, part: 0, start, end}`) on string content is the only OpenAI placement that reaches the
  server today.
- On the Anthropic surface a part's offset in the flattened message is the length of the earlier parts plus one `'\n'`
  before each part except while the accumulated text is still empty (see the empty-block rule above). The separator is outside every span, so a span covering a
  whole part is not altered by the join.

### Templates

| template | role | rendered as | evidence |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | `user`, `tool`, `system` | verbatim between fixed literals; no trim, no escaping | `Qwen3_UserAndToolContent_AreRenderedVerbatim` |
| Llama-3.1-8B-Instruct | `user`, `assistant`, `system` | `\| trim` (.NET `string.Trim()`: all leading and trailing whitespace); interior kept | `Llama31_UserContent_IsTrimmed_InteriorKept`, `Llama31_SystemContent_IsTrimmed` |
| Llama-3.1-8B-Instruct | `tool` | `\| tojson`: a quoted, JSON-escaped string (`\n` → `\\n`, `"` → `\"`) | `Llama31_ToolContent_IsJsonStringEncoded` |

Consequences: on Llama-3.1 a span touching either edge of a user message is altered (`chunk_template_altered`), and
**every span inside a tool result is altered**, because the whole tool content is re-encoded as a JSON string. Agent file
reads land in tool results, so on Llama-3.1 the convention can only locate spans placed in user messages (in the middle,
away from the edges).

### Special-token spellings in content

- **The chat path tokenizes them as plain text** (#1339, `ControlTokenGuard`). The template marks every special spelling
  in caller text (content, tool results, tool arguments, tool definitions, kwargs) by inserting U+FDD0 after its first
  character; encode strips the marker. In assistant turns only, the template's own reasoning tags stay live when the template
  uses think markup (`<think>` / `</think>`).
- **The rendered prompt string therefore does not contain such content verbatim**: it contains the marked copy
  (`tokenizer.MarkUntrustedSpecials(content)`). A substring or sentinel search for a span over the rendered string must
  compare against the marked span text, or strip markers from both sides.
- **A caller-supplied U+FDD0 is dropped on encode**: content `a﷐b` tokenizes exactly like `ab`.
- **`ITokenizer.EncodeRaw` (what `SymbolChunker` and `FixedSizeChunker` use) parses special spellings** as control tokens
  (scan-everything semantics, also `/v1/completions` and `/tokenize`). So a chunk whose text contains a special spelling
  has different ids in the chunk store than in a chat prompt. `EncodeRaw(tokenizer.MarkUntrustedSpecials(text))` equals
  the plain-text encoding, i.e. the chat path's.
- The spellings are the vocabulary's token types 3 and 4 longer than one character: 26 on Qwen3-4B-Instruct-2507,
  including `<think>`, `</think>`, `<tool_call>`, `<tool_response>`, which occur in ordinary source code and docs.

Consequence for spec §5.3 item 6: the spec assumed the chat path and `EncodeRaw` agree. They do not. Either a chunk whose
text contains a special spelling is ignored (`chunk_special_tokens`), or chunk text is encoded through the guard
(`EncodeRaw(MarkUntrustedSpecials(text))`) wherever the store must match the chat path.

### Unknown fields

| what | OpenAI chat | Anthropic messages |
|---|---|---|
| unknown top-level field (`chunk_cache`) | ignored; rendered prompt unchanged | ignored; rendered prompt and token count unchanged |
| unknown field on a message object (`chunk`) | ignored | (not a convention placement) |
| unknown field on a `text` part / block (`chunk`, `cache_control`) | unreachable: array content is a 400 | ignored; validation inspects only `type` |
| unknown content block **type** | n/a | **400 `invalid_request_error`**, `messages[i].content[j].type: unsupported content block type '<t>'` (#449) |
| blocks inside a `tool_result`'s content array | n/a | **not type-checked**: any block with a string `text` is joined in, whatever its type; any other block is dropped silently |

Unknown fields are skipped because the source-generated `ServerJsonContext` keeps `JsonUnmappedMemberHandling.Skip`
(see `docs/SERVER.md`, "Accepted and ignored"). A strict-DTO pass would turn every annotated request into a 400 on a
server without the convention.
