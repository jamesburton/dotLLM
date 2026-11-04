# Chunk cache (server side)

Server-side notes for the `chunk_cache` wire convention (chunk-level K/V reuse for agent harnesses). The convention, its
fields and its error codes are specified outside this repo (ChunkedkvCache, spec `2026-11-02-chunk-cache-integration.md`,
wire contract `docs/contracts/chunk-cache-v1/`). This page records what the dotLLM server does; it grows with each server
task. Everything the convention adds sits behind `dotllm serve --chunk-cache` (off by default); with the flag off, every
request and response is what it was before the convention, for any body.

## Switches

| flag | default | meaning |
|---|---|---|
| `--chunk-cache` | off | parse and validate `chunk_cache` annotations on `/v1/chat/completions` and `/v1/messages` (reuse arrives with the next server tasks; today no response changes). Off: the annotations are never read; the body behaves exactly as without them (pinned over HTTP by `ChunkCacheFlagHttpTests`) |
| `--chunk-cache-max-chunks <n>` | 4096 | most annotations read per request (≥ 1; no upper bound, so `int.MaxValue` disables the cap); the rest are counted from the array length and dropped with `chunk_too_many` without being read |

Both exist on `dotllm serve` and on `ServerOptions.Parse` (`ServerOptions.ChunkCache`).

## Request parsing (`ChunkAnnotationParser`)

The parser reads the request and nothing else: no store, no tokenizer, no reuse. It turns the annotations into
`ParsedChunkRequest` (accepted spans, warnings with counts, the strict-mode error, the request options), or returns null
when the flag is off or the request carries no annotation. **It is not yet called by the endpoints**: the next server
task applies the parse (strict 400s, the `chunk_cache` response object); until then the flag changes no response.
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

**Lone surrogates (ruling 17).** In-process callers only: a string with a lone surrogate hashes as UTF-8 with U+FFFD.
Over HTTP it cannot happen: System.Text.Json parses a lone-surrogate escape but refuses to return it as a string, so
message text with one fails before the parser (as it always did), and an annotation field with one is
`chunk_annotation_malformed`, never an exception.

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
**Namespace.** The store namespace is always server-assigned: the parser takes the caller's namespace as a
**required** argument (the API key's namespace once authentication exists; until then callers pass the default, `""`).
A body `chunk_cache.namespace` equal to the caller's own is accepted silently; another name is
ignored with `chunk_namespace_ignored` (strict: `400 invalid_request_error` with that code), and the spans stay (a non-string is `chunk_annotation_malformed`). The
namespace's format is validated where it is assigned (the authentication step), not here.

The contract's request fixtures are copied byte for byte under `tests/DotLLM.Tests.Unit/ChunkCache/Fixtures/v1/`
(checked against `fixtures.sha256`); `ChunkAnnotationParserTests` checks every one against `expected.json`, in the
contract's message model and translated to the Anthropic surface.

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
