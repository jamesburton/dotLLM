# Chunk cache (server side)

Server-side notes for the `chunk_cache` wire convention (chunk-level K/V reuse for agent harnesses). The convention, its
fields and its error codes are specified outside this repo (ChunkedkvCache, spec `2026-11-02-chunk-cache-integration.md`,
wire contract `docs/contracts/chunk-cache-v1/`). This page records what the dotLLM server does; it grows with each server
task. Everything the convention adds sits behind `dotllm serve --chunk-cache` (off by default); with the flag off, every
request and response is what it was before the convention, for any body.

The `chunkcache` command-line tool that builds, selects, inspects and evicts the store is documented in [CHUNK_CACHE_CLI.md](CHUNK_CACHE_CLI.md).

## Switches

| flag | default | meaning |
|---|---|---|
| `--chunk-cache` | off | honour `chunk_cache` annotations on `/v1/chat/completions` and `/v1/messages`: policies `exact` (default) and `shadow`, whose output is identical to the unannotated request, plus a `chunk_cache` report in the response (below). Off: the annotations are never read; the body behaves exactly as without them (pinned over HTTP by `ChunkCacheFlagHttpTests` and `FlagOff_AnnotatedBodies_AnsweredLikePlain_EverySurface`, blocking and streaming, both surfaces) |
| `--chunk-cache-max-chunks <n>` | 4096 | most annotations read per request (1 to 65536, so the per-request parse work stays bounded); the rest are counted from the array length and dropped with `chunk_too_many` without being read |
| `--chunk-store <dir>` | none | the chunk store (e.g. built by `chunkcache build` from the same GGUF, without `--strict-model-id`) the server looks annotated chunks up in. **Read-only unless `--chunk-store-write`**: the server then never writes, encodes into or creates it. Needs `--chunk-cache` (rejected without it); a relative path is resolved at startup. No store: every verified chunk is a miss, no warning. A configured root that does not exist (read-only): misses plus `chunk_store_unavailable` (strict: 503) |
| `--chunk-store-write` | off | the server becomes the store's **sole writer** ([Store writes](#store-writes-the-sole-writer)): it owns the directory (created if missing; a second server on it refuses to start), keeps the server ledger, and encodes admitted misses in the background while idle. Needs `--chunk-store` and `--chunk-store-budget` |
| `--chunk-store-budget <bytes>` | none | the store's byte budget (bytes or units: `20GiB`, `500MB`), **required** with `--chunk-store-write` (no unlimited default) and rejected without it; above 0 |
| `--chunk-store-index-only` | off | with `--chunk-store-write`: store only chunks that an index file in the store (`index-*.json`, from `chunkcache build` over a scanned, deny-listed tree) references, so text from ad-hoc reads is never persisted (spec §10). Rejected without `--chunk-store-write` |
| `--chunk-store-namespace-budget <bytes>` | off | with `--chunk-store-write`: the most bytes of ledger entries one namespace (API key) may hold, a **cap, not a reservation**. When set, budget eviction takes **only the writing namespace's own** rows ([Budget](#store-writes-the-sole-writer)), so one tenant's prewarm or encodes never evict another tenant's entries; non-admin capabilities report it as `budget_bytes`. Off: one shared LRU. Rejected without `--chunk-store-write`; above 0 and **below `--chunk-store-budget`** (startup error otherwise); a startup warning when the caps of all namespaces (the keys file's, or 2 without one) add up to more than the store budget |
| `--chunk-cache-api-keys <file>` | none | the API keys of the chunk-cache surface (below, [Authentication and namespaces](#authentication-and-namespaces)): `<key> [admin] [ns=<namespace>]` per line, keys of at least 32 characters; each key is its own store namespace unless `ns=` names one. Needs `--chunk-cache`. Unset: the control plane answers local callers only, all in the default namespace. Read once at startup (revoking a key = restart). **Security-sensitive** |
| `--chunk-cache-allow-short-keys` | off | accept keys shorter than 32 characters, each with a startup warning (otherwise they refuse startup). Needs `--chunk-cache-api-keys` |

All nine exist on `dotllm serve` and on `ServerOptions.Parse` (`ServerOptions.ChunkCache`).

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
   4. **eligibility** → `chunk_not_wire_eligible`: a `chunker` outside the wire-eligible list (`symbol-512` and `lines-512`); the
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

**`report: per_chunk`** is accepted from any caller by the parser; the response carries per-chunk status only for an
authenticated caller, in its own namespace (see [Authentication and namespaces](#authentication-and-namespaces)).

Request options: `policy` (`exact` default; `balanced` / `fast` / `auto` parse to their own values and are degraded
later), `strict`, `report` (`summary` default), `min_chunk_tokens` (32), `on_miss` (`prefill`), `query` (`{text}` or
`{message}`; default: the last engine message with role `user`, so Anthropic tool results never count as the query).
A valid `query.message` that points at a message with no text parts (e.g. an assistant turn with only tool calls) gives
an empty query and no warning (ruling 34 asks only for the index of a request message).
**Namespace.** The store namespace is always server-assigned: the parser takes the caller's namespace as a
**required** argument, from `ChunkCacheAuth` (the API key's namespace, or the default `""` for a local caller or an
unauthenticated one).
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

#### Chunk edges and joining (token stability)

A span counts only if its own ids occur in the whole-prompt ids at its position, so what a harness puts **next to** a chunk
decides whether it is reusable:

- **Back to back is safe.** `symbol-512` and `lines-512` accept an edge only where
  `Encode(left + right) == Encode(left) ++ Encode(right)` (`ChunkBoundary`, with `ChunkTokenization.Encode`): an edge that
  would merge (a chunk's last newline with the blank line starting the next, `"\n" + "\n"` becoming one `"\n\n"` token) moves
  to a stable edge (symbol: over the blank lines to the previous chunk; lines: back to the previous line end). Census on the
  three frozen corpora (Qwen3-4B tokenizer): `symbol-512` had 5,910 of 6,684 edges unstable (88 %), now 0; `lines-512` had
  0 of 2,542 and still 0 (`experiments/corpus.md`). If no stable edge is in reach, the greedy cut stays and that edge can
  still merge (not seen on the corpora).
- **An index built before this change is re-chunked, not kept.** The chunkers kept their names, so each index entry records
  the chunker's algorithm version (`chunkerVersion` per file; absent = 1, the old cuts; `symbol-*` and `lines-*` are now 2).
  `chunkcache status` reports a file cut by another version as stale even when its text is unchanged, and the next
  `chunkcache build` re-chunks it. The new chunks have new keys: the old entries miss (never hit wrongly) and become orphans
  that `chunkcache evict` removes.
- **A newline join is not safe.** Every chunk ends in a newline (both chunkers cut after line ends), and a newline
  followed by an inserted `"\n"` merges. The Anthropic converter joins text blocks of one message with `"\n"`, so inline
  chunk blocks that end in a newline are all `chunk_template_altered` (e2e A6: 3 of 3). The same holds for any glue that
  starts with a newline. Client rules: send the chunks of an Anthropic request as a **manifest over one string message**
  (e2e A6m works), or put glue that does not start with a newline between chunks and end a header with a non-whitespace
  character (a `// path` line, e2e A3g). Census of chunks ending in a newline: 99.6 % (`symbol-512`) and 99.7 %
  (`lines-512`) of the left-hand chunks merge under a newline join. The chunkers do not try to avoid a trailing newline:
  every chunk is whole lines, and the alternative (ending before the newline) only moves the merge to the start of the
  next chunk.

## Execution: `exact` and `shadow` (`ChunkCachePipeline`, `ChunkCacheReport`)

Only `exact` (the default) and `shadow` run. **Both generate exactly what the unannotated request generates**: the prompt
is rendered and tokenised as a plain request (spec §5.3 item 7), the same dispatch runs (scheduler or generator), and the
prompt cache is looked up with exact lineage only (Task 12: a snapshot's approximate rows are never restored, so
`cached_tokens` / `timings.cache_n` keep meaning exact prefix reuse). The annotations only produce counts, and under
`shadow` a plan. Pinned over HTTP on a synthetic model, blocking and streaming, on both prompt-cache configurations
(single-slot prefix cache; the default paged KV + scheduler, which holds the cross-request prefix trie since #1596):
`Exact_IsBitIdenticalToUnannotated`, `Exact_AfterShadow_SameSlot_Identical`, `CachedTokens_UnchangedByAnnotations`,
`Exact_WithApproxSnapshotOnSlot_RestoresExactRowsOnly`. Both paths reuse exact prefixes across requests, so a warm
request reports `cached_tokens` > 0, and a verified chunk wholly inside that restored prefix is reported `prefix`, not
`hit` (a test that wants a `hit` must make its prompt diverge before the chunk).

- **Policies.** `balanced`, `fast` and `auto` parse and are served `exact` with `reason: "chunk_policy_unavailable"`
  and that warning once (rulings 42; strict: 400) until applied reuse ships. On a model or backend that cannot assemble
  chunks (recurrent or hybrid layers, a diffusion model, a backend other than CPU or Vulkan, a RoPE variant the
  re-rotation does not reproduce, or a model whose identity cannot be computed) every annotation is ignored with
  `chunk_model_unsupported` (strict: 400); `reason` carries it when the requested policy was not `exact` (ruling 41). Weights the base GGUF's model id does not cover are part of the
  identity (#1915): the SHA-256 of the `--tensor-overlay` file and of each `--ternary-adapter` file, and of a request's
  `lora_adapter` (its rank, alpha and every factor byte; not its name), go into an order-stable modifier digest
  (`ModifierSetDigest`) that `ChunkFingerprint.WithModifiers` folds into the fingerprint. Contents, never paths: the same
  files elsewhere, or listed in another order, are the same key space; any other set is a different one, so a base entry is a
  `miss` under an adapter and the reverse. With no modifiers the fingerprint is exactly the one written before #1915, so
  existing entries stay valid for the base model and the store layout and index version are unchanged. A request with
  `lora_adapter` is looked up under its own keys but **never admits** (no sighting, no queue): the background encoder runs the served
  model without the adapter. The overlay and adapter files are hashed once per loaded model, at the first annotated request
  (an unreadable file makes the identity unsupported, `chunk_model_unsupported`, never a base-keyed one).
  Diffusion models get no report.
- **Per located span**, in prompt order, at most `MaxPlannedSpans` (1,024) per request:
  1. ids = `ChunkTokenization.Encode(text)` (`EncodeRaw(MarkUntrustedSpecials(text))`, ruling 21);
  2. fewer than `min_chunk_tokens` (default 32) ids: `chunk_ignored_too_short` (warning only);
  3. **token check**: the span counts only if those ids occur in the whole-prompt ids exactly where its characters are.
     A span the check finds different is `chunk_template_altered` (strict: 400): a span glued to the text before it where
     the tokenizer merges across the edge (in particular a chunk ending in a newline followed by the Anthropic `"\n"` join of
     inline text blocks: see "Chunk edges and joining" above), a SentencePiece leading-space piece on the span itself, a Qwen3 assistant span
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
     `ChunkKey` of the server's identity (below), `isolated` positioning and the caller's namespace (server-assigned by
     `ChunkCacheAuth`). An unauthenticated caller's chunks are never looked up (each is a miss, no warning: the store is a
     membership oracle). A store failure is a miss.
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
  To diagnose, compare the fingerprint inputs, which `GET /v1/chunk-cache` exposes (below): `model.id`
  (`GgufIdentity.ModelId`) with `model.id_kind` (`sampled`), `model.tokenizer_id`, `model.quantization`, `model.backend`,
  `model.kv_dtype`, `model.producer` (the producer hash, the key of the empty sequence in the caller's namespace, which
  folds in the model fingerprint, RoPE parameters, schema version, backend, dtype and namespace) and `model.rope`
  (`dim`, `theta`, `type`).
- **Strict mode.** Parser defects are returned before rendering. After it, the error is the first of the request-level
  defects (`chunk_model_unsupported`, `chunk_policy_unavailable`, `chunk_store_unavailable`), else the span-level defect
  of the span first in document order (ruling 4; locator and token-check defects alike). Warning-only codes
  (`chunk_ignored_too_short`, `chunk_verification_skipped`) never become an error. OpenAI: the usual error body with
  `code` (and `param: "chunk_cache"`); 400 `invalid_request_error`, or 503 `api_error` for `chunk_store_unavailable`.
  Anthropic: the usual envelope with `error.code` (ruling 29; omitted on every other error).
- **Never throws on request content.** An unexpected failure while planning ignores every annotation (the request is
  served as plain) and logs only the exception type; reports and logs never carry chunk text, hashes or namespaces.

### The response `chunk_cache` object (spec §5.9; contract rulings 5–10, 46–50)

Written on annotated requests only (flag on), unless the request asked for `report: "none"`. With `report:
"per_chunk"` from an authenticated caller the object also carries `per_chunk`: one `{id, status, why}` per annotation the
server read, in document order (`status` `prefix` / `hit` / `miss` / `ignored`; `why` is the §5.8 code that dropped an
`ignored` one, else null, ruling 55). Any other caller gets the summary (spec §10, ruling 45). Where: the blocking response (both surfaces), the
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

## Store writes: the sole writer

With `--chunk-store-write` (spec §5.10, §10, §14 L1/L2; `ChunkStoreService`, `ChunkEncodeQueue`, `SightingTable`) the
server is the only process that writes K/V into the store. Without it nothing below happens and the store is read-only
exactly as before.

**Ownership.** `dotllm serve` takes `<store>/.chunkstore.owner.lock` exclusively **before the model loads** and holds it
for the process lifetime (`StoreOwnerLock`; the OS releases it if the process dies). A second server with
`--chunk-store-write` on the same directory refuses to start within seconds, with a message naming the lock; a read-only
server on it is fine. An offline `chunkcache` writer must take the same lock with `StoreOwnerLock.TryAcquire` and **hold
it for its whole write**; `StoreOwnerLock.IsHeldByAnother` is a racy probe (a server can start between the probe and the
write), for status display only. The owner lock is not the store lock: every write still takes the short store lock
(`.chunkstore.lock`), so a CLI index update or orphan pass interleaves with the server between entries. **Local file
systems only:** the lock relies on the OS releasing an exclusive handle when its process dies (verified on NTFS; `flock`
on local Linux file systems); on network shares (SMB, NFS) two servers may both believe they own the store.

**Admission** (request thread; never blocks and never touches a lock file). After `exact` / `shadow` planning, each
verified **miss** of an **authenticated** caller (a local caller without keys, or a valid key: the control plane's rule;
anyone else gets no lookups and no side effects) on an inference request (never `explain`, never a request that fails
strict) is queued for a background encode only when all of these hold:

1. not `no_store` (no sighting is recorded either) and not `stability: "volatile"`;
2. its `source.path`, when the annotation carries one, is not on the secrets deny-list (`ChunkDenyList`, by file name
   at any depth: `.env*`, `*.env`, `*.pem`, `*.key`, `*.pfx`, `*.p12`, `id_*`, `id_rsa*`, `secrets.*`,
   `appsettings*.json`, `credentials*` (so `.aws/credentials`), `.npmrc`, `.netrc`, `.pgpass`, `*.keystore`, `*.jks`,
   `kubeconfig*`, `*.tfvars`, `*.tfstate`; and anything under a `.git`, `.ssh`, `.kube` or `.docker` directory;
   case-insensitive on every platform). Windows spellings of the same file are normalised first: trailing dots and
   spaces per segment are dropped, a path containing `::` (an alternate data stream) is denied, a segment with a colon
   is checked on both sides of it (`secrets.json:stream`, and the drive-relative `C:secrets.json`), and an 8.3 short
   name is denied when its stem could be a denied name (`SECRET~1.JSO`) or, in NTFS's hashed form (`SE1A2B~1.JSO`),
   when its extension could be a denied one. A `source` that is not an object, or a non-string or over-long `path`,
   cannot be checked and is never admitted. The chunk **text** is also refused when a PEM armour line
   (`-----BEGIN ` or `-----END `) says `PRIVATE KEY` (RSA, EC, OPENSSH, ENCRYPTED, PGP `PRIVATE KEY BLOCK`); a key body
   that a chunker split into a chunk with neither armour line is not detected (known limit: harnesses set `no_store`,
   and the path list covers key files). Not on the list (a later deny-list flag): `*.kdbx`, `*.p8`, `.pypirc`,
   `.gnupg/`. The server cannot see files: it
   trusts `no_store` and matches only the path it is told (`source` never affects lookup and is never an annotation
   defect). When the same text occurs twice in one request, the most restrictive occurrence decides;
3. the served backend is CPU (a Vulkan server writes nothing until the per-backend schema question is settled: Vulkan
   K/V also depends on the shader binaries and the driver) and the chunk has at most **512 tokens** (`MaxCpuEncodeTokens`,
   a constant; ruling after the Task 15 re-review): a longer chunk is refused at admission and the response's
   `chunk_encode_skipped` carries `detail: "too_large_for_cpu"`. This bounds how long a request can wait behind one
   encode (below);
4. it has been seen in at least **2 requests** in the caller's namespace (`SightingTable`: in memory, keyed by namespace
   and text hash, at most 100,000 rows and 25,000 per namespace, least recently seen dropped first, a namespace at its
   share dropping its own; one sighting per request and text; `source.path` stored verbatim up to 256 characters, longer
   as a prefix plus `#` and 16 hex digits of its SHA-256, cut at a character boundary; a later forget must compare
   `SightingTable.StoredPath(requested)`, not the raw path). `on_miss: "encode"` bypasses this rule only (the request
   itself is still served `exact`). Texts past a request's cap of 16 queued texts record no sighting in that request;
5. the writer is not stopped (unreadable ledger, budget exhausted: below);
6. with `--chunk-store-index-only`, an index references its key (a snapshot refreshed when the index files change,
   checked again under the store lock before the put);
7. the queue has room: at most 256 items (the one being encoded included), at most 64 of them from one namespace, and
   16 Mi characters of text; one request queues at most 16 texts; a text already queued is not queued twice;
8. once the writer has stored an entry (so it knows this model's bytes per token), a chunk whose entry could never fit
   the budget is refused before any encode (`over_budget`).

The response's `chunk_cache.warnings` then carry `chunk_encode_queued` and / or `chunk_encode_skipped` with counts (one
per verified miss), and `tokens.encode_queued` counts the queued tokens. Hits refresh their ledger rows' last use. The
refusal reason is not reported per chunk (it would be a membership probe on the deny-list and on the sightings).

**Background encode** (`ChunkEncodeQueue`: one dedicated, below-normal-priority thread). An item is encoded only after
the server has been **quiet for 2 s** (a fixed constant: no request running or waiting, no model loading or activating,
the continuous-batch scheduler idle, and no request started meanwhile, counted by `ServerState.ActivityStamp` so a
request between two 25 ms polls also restarts the wait), and then only through `ServerState.TryAcquireIdleModel`. The
worker holds the gate and the scheduler's model lease for **one** chunk's forward pass and releases them before any disk
I/O. After 4 encodes in one quiet window it pauses 10 s. **Worst-case wait for a request:** one encode, i.e. a request
that arrives just after the worker took the model waits for that chunk (at most 512 tokens on CPU) to finish. Measured on the
synthetic test model (CPU, 1 thread): 86 tokens about 35 ms, 2048 tokens about 2.0-2.2 s. **Measured on the reference
model** (Qwen3-4B-Instruct-2507 Q4_K_M, CPU, 8 threads, Strix Halo, one run, CPU otherwise lightly loaded): a 512-token
chunk 8.35 s, a 2048-token chunk 43.5 s. With the 512-token CPU cap the worst-case request wait on that model and
machine is therefore about **8.4 s** (the 2048-token figure is why the cap exists). Other models, thread counts and
hardware are **unmeasured**; the wait scales roughly with model size and chunk length.
The quiet-period and window clocks are monotonic (`Stopwatch`), so a wall-clock step never shortens or extends them. Transient memory per encode is about twice the chunk's F32
K/V (the encode cache plus the host copy), e.g. about 1 GiB for 2048 tokens of an 8B model. The chunk is tokenised again with `ChunkTokenization.Encode` (ruling 21: the wire path's tokenisation), its key
is recomputed with the served model's identity (an item whose key no longer matches, e.g. after a model swap, is
dropped), and it is encoded in isolation at positions `0..n-1` on the server's own model (`ChunkEncoder`,
`CpuChunkBackend`). The entry is F32 (the store's only dtype; capabilities `kv_dtype`), stamped with the model
fingerprint and the backend tag: exactly what `chunkcache build` writes for the same GGUF, so a CLI reader and the
server's own lookups agree (`EntriesTaggedWithServerBackend` compares the K/V bit for bit). **Shutdown** cancels the
worker and drops what is queued; an encode in progress cannot be interrupted (the forward takes no cancellation). The
server waits up to 30 s for it; if the worker is still encoding after that, the model, the request gate and the store's
ownership are **not** freed (the model is left for the process exit, logged by type only), so a running forward never
touches freed memory and no second writer can start on the store before the encode has ended.

**Writing** (worker thread only, so the store lock's 20 ms retry loop never runs on a request thread). Per entry,
**one** store-lock scope holds: budget eviction if needed, the put (temp file + rename), the ledger row
(`ServerLedger.Record`) and the ledger save (temp file + rename; it requires the held lock handle). The file comes first,
then its row, so no other lock holder ever sees a row without a file; a crash leaves at most a file without a row (an
orphan, removable by an orphans pass), and a row whose file is gone is dropped on the next load. A failed put (disk
full, I/O error) stores nothing and records nothing. A failed ledger save leaves the old ledger file, reloads it, and
deletes the just-written file unless an index references its key. If the ledger cannot be read back then (after brief
retries for a sharing violation), the unsaved row leaves memory and **writes stop** (`ledger_unreadable`): no admission,
no encode, no write, and the file is never written over; the worker retries a read every 30 s and only a successful
(verified) read re-enables writes. The store is never wrapped in `LockingChunkStore` (the lock is not re-entrant). The
lock timeout is 10 s; a timeout drops the item (a later sighting can queue it again).

**Budget.** The byte total of all entry files in the store (CLI entries included) stays within `--chunk-store-budget`.
When an entry does not fit, ledger rows are evicted: the writing namespace's own rows first, then the others', each in
least-recently-used order (last write or hit, then key). Keys are content-addressed, so a CLI index may reference the
same key as a ledger row: such a row is removed but its file is kept and frees nothing (decided under the lock, after the
ledger save, from every index and the remaining ledger). Entries only an index references are never evicted by the
server. When evicting cannot make room, **nothing is evicted**, and the decision never depends on how large the current
entry is: the writer stops (`budget_exhausted`, re-checked every 30 s until a minimum-size entry fits) only when not even
a **minimum-size entry** (`min_chunk_tokens` = 32 tokens of this model) would fit after evicting everything the server
may evict, i.e. the store is genuinely full of entries the server may not evict. In every other case only the current
entry is dropped (`entry_too_large`; the writer keeps running, so no caller can stop it for everyone). An entry larger
than the whole budget is dropped before the store is even counted. An unreadable CLI index file also stops it (eviction cannot be decided; fail closed).

**Per-namespace budget** (`--chunk-store-namespace-budget`, off by default). **A cap, not a reservation.** Each
namespace's ledger bytes must also stay within it. Budget eviction then takes **only the writing namespace's own rows**
(least recently used first), for both limits: a namespace over its share evicts its own rows. Nothing is reserved for a
namespace: when the store is full of **other** namespaces' rows, the entry is dropped (never another tenant's entry,
never a writer stop) and the namespace is **blocked** at admission (`chunk_encode_skipped` with `detail:
"namespace_budget_blocked"`, reason label `namespace_budget_blocked`; queued items of it are not encoded either), so no
encode is spent and thrown away, until a minimum-size entry fits again (re-checked every 30 s, and at once after an evict
that freed space) (`NamespaceBudget_PrewarmEvictsOnlyOwnRows`). Only `namespaces x namespace budget <= store budget` (minus
CLI entries and orphans, which are not counted per namespace, ruling 69) is a real quota: the budget must be below the
store budget (startup error), and the server warns at startup when the keys file's namespaces (or 2 without a keys file)
times the namespace budget exceed it. Admission refuses (`over_budget`) a chunk that could never fit the smaller of the two
budgets. Without the option the shared LRU above applies (own rows first, then the others').

**Store problems in responses.** While the writer cannot store, responses with at least one verified miss of an
authenticated caller carry `chunk_store_unavailable` (a warning, never a strict-mode error here: the request itself is
unaffected) with an additive `detail`: `ledger_unreadable` (writes disabled until a verified reload), `budget_exhausted`
(encoding stopped: the store is full of entries the server may not evict), `entry_too_large` (the last encoded entry
could not fit and was dropped), `lock_timeout` (another writer held the store lock past the timeout; never reported as a
plain busy store) or `write_failed` (the last put or ledger save failed). `entry_too_large`, `lock_timeout` and
`write_failed` clear on the next successful write. These details are writer-wide (every authenticated tenant sees
them). A configured root that is missing in read-only mode keeps its earlier behaviour (strict: 503).

**Unreadable ledger at startup** (`ledger-server.json` exists but cannot be read after brief retries): the server starts
and inference runs, but writes stay disabled: the ledger is never replaced by an empty one and never saved over (an
empty ledger would turn every server-written entry into an orphan). The startup log names the file. Writes resume only
when a later read of the file succeeds (every 30 s), e.g. after it is repaired or restored.

**Privacy.** Nothing here logs chunk text, hashes, keys or namespaces (log lines carry exception types, sizes and
counts). The ledger records each row's namespace and first `source.path` (tree-relative paths only; others are dropped),
and the sighting table holds text hashes, in memory only. **The store holds the source code**: every entry stores the
chunk's token ids beside its K/V, so treat the store root like the repository: the same access controls, outside synced
folders, git-ignored, and excluded from crash dumps and telemetry (spec §10). The writer creates a missing root with
inherited permissions.

**Metrics** (`/metrics`, fixed label sets, no namespace): `dotllm_chunk_store_writes_total{result}` (`stored`, `failed`,
`too_large`), `dotllm_chunk_store_evictions_total`, `dotllm_chunk_store_skips_total{reason}` (the admission reasons:
`no_store`, `volatile`, `denied_path`, `source_unreadable`, `secret_content`, `backend_not_writable`, `too_long`,
`not_seen_twice`, `index_only`, `writes_disabled`, `budget_exhausted`, `queue_full`, `namespace_queue_share`,
`request_cap`, `over_budget`, `too_large_for_cpu`, `namespace_budget_blocked`) and the gauge `dotllm_chunk_store_queue_depth`.

**Known limits (multi-tenant).** Isolation between API keys stops at fairness: (a) the queue is shared (each namespace
may hold at most 64 of its 256 items, one request at most 16); (b) the sighting table is shared (each namespace at most
25,000 of its 100,000 rows); (c) memory per sighting row is bounded by the 256-character path cap; (d) the byte budget
and its LRU are shared: eviction takes the writing namespace's own rows first, but a tenant with no rows of its own
evicts others' least recently used entries, unless `--chunk-store-namespace-budget` is set (then only its own).
The writer-wide `detail` values above are visible to every authenticated tenant.

## Control plane: capabilities and explain (spec §5.9, §7.2)

Both routes are **mapped only with `--chunk-cache`**: with the flag off they do not exist (the usual bare 404 for any
method, pinned by `FlagOff_404`). Both authenticate (below). Neither writes the store, generates, or counts in `/metrics`
(prewarm and evict, which do write, are [below](#control-plane-prewarm-and-evict-spec-72-10-the-store-tasks)).

**Rate limiting** applies to them only when it is configured (`--rate-limit-*`; off by default): `/v1/` paths are
metered unless exempted, and these are deliberately not exempt. The limiter's buckets are keyed by the raw presented key
string (pre-existing): a client sending a fresh random key per request gets a fresh bucket each time, but with keys
configured such a request only reaches the cheap 401 path. Explain bounds itself whatever the limiter does (below).

**CSRF hardening** (in place before prewarm / evict add side effects): a request with `Sec-Fetch-Site: cross-site` is
`403 permission_error` on both routes whatever the key, and a control-plane POST must be `Content-Type:
application/json` (parameters such as `charset` allowed), else `415`. That forces a browser to preflight, which the
`Origin` rule then refuses (`ControlPlane_RequiresJson_RefusesCrossSite`).

### `GET /v1/chunk-cache`

The contract's `capabilities.json` shape (`Capabilities_MatchesFixtureShape`), plus the chunk-key identity in `model`
(ruling 51: clients ignore unknown keys):

```json
{"versions": [1], "policies": ["exact", "shadow", "balanced", "fast", "auto"],
 "selectors": ["chunk_starts"], "wire_chunkers": ["lines-512", "symbol-512"], "auto_validated": [],
 "model": {"id": "<GgufIdentity.ModelId>", "supported": true, "reason": null, "id_kind": "sampled",
           "tokenizer_id": "<GgufIdentity.TokenizerId>", "producer": "<64 hex: keyOf([]) in the caller's namespace>",
           "quantization": "ftype-15", "backend": "cpu", "kv_dtype": "f32",
           "rope": {"dim": 128, "theta": 1000000, "type": "neox"}},
 "store": {"entries": 0, "bytes": 0, "budget_bytes": 0, "dtype": "f32"}, "max_chunks_per_request": 4096}
```

- `model.id` is the content model id, not the served model name. On an unsupported model or backend: `id` null,
  `supported: false`, `reason: "chunk_model_unsupported"`, and no identity fields.
- `model.producer` is per caller: the key of no tokens in the caller's namespace. A server-backed `chunkcache` finds its
  index from `model.id` and `model.producer`: `index-<chunker>.<ModelTag(model.id)>.<producer[..8]>.<ns-or-default>.json`.
  The namespace itself is never in a response; the CLI derives it from its own key (`ChunkNamespace.ForApiKey`), or
  is told the `ns=` name by its operator, or uses `default`.
- `store` is always an object (ruling 52). The store-wide figures come from a walk over the entry files that is reused for 5 s
  (the route has no slot), so they can be that old. With `--chunk-store-write`, a **non-admin** sees its own namespace's figures:
  `entries` and `bytes` of its ledger rows (what the server wrote for it; a CLI's offline entries are not counted) and
  `budget_bytes` = the per-namespace budget when set, else the store budget. An **admin** sees store-wide figures: every
  entry file in the store (CLI entries and orphans included) and the store budget (`Capabilities_ReportsStoreFigures`).
  Without a writer there is no ledger: a non-admin sees 0, an admin the read-only store's files, and `budget_bytes` is 0.
  `dtype` is `f32`. Without `--chunk-store` it carries `"configured": false`; with one, `configured` is absent
  (`Capabilities_StoreConfiguredFalse_WithoutStore`).
- `wire_chunkers` is the server's wire-eligible list (`lines-512`, `symbol-512`; sorted). `lines-512` is eligible on correctness
  (its text round-trips by construction); its reuse **quality** is not validated, so `auto_validated` stays `[]` (Phase I5's gate).

### `POST /v1/chunk-cache/explain`

A dry run of the request's annotations. Body: `{"request": <a /v1/chat/completions body>}` (contract ruling 15) and
nothing else: any other top-level field, a non-object `request`, a body that is not JSON, or a `request` the chat endpoint
could not bind (e.g. array `content`) is `400 invalid_request_error` (`Explain_HashListBody_Rejected`). There is no hash
lookup: the text must be sent. The request is validated, parsed in the caller's namespace (a body `namespace` other than
the caller's is ignored with `chunk_namespace_ignored`), rendered by the chat endpoint's own render (`TryRender`),
tokenised once (that one encode serves the context-length check and the plan), then located, token-checked and looked up
exactly as an inference request. A strict request's defect is returned with the same status and error as
`/v1/chat/completions` would return. The model named in the body is not loaded: explain uses the model being served.

**Bounds** (explain renders and tokenises on the request thread, outside the scheduler): at most
`MaxConcurrentExplains` (2) per server at once, the next gets `429 rate_limit_error` (`chunk_explain_busy`,
`Retry-After: 1`); a body over `MaxExplainBodyBytes` (4 MiB) is `413`, with or without a `Content-Length` (a truncated
body is 400, never 500) (`Explain_ConcurrencyCapped_429_BodyLimited_413`). The body is read before a slot is taken,
within `ExplainBodyReadTimeoutSeconds` (10 s); a slower body is `408` and its connection closed, so a sender trickling a
large body at Kestrel's minimum data rate (~240 B/s) never holds a slot (`Explain_SlowBody_DoesNotHoldASlot_408`).

```json
{"policy_would_apply": "exact", "reason": null,
 "chunks": [{"id": "a", "status": "hit", "why": null}, {"id": "b", "status": "miss", "why": null},
            {"id": "c", "status": "ignored", "why": "chunk_hash_mismatch"}],
 "counts": {"annotated": 3, "prefix": 0, "hit": 1, "miss": 1, "ignored": 1},
 "predicted_ms": {"reuse": 12.0, "full_prefill": 40.0}, "warnings": ["chunk_hash_mismatch"]}
```

- `chunks` (per-annotation status, ruling 55: `why` a §5.8 code or null) is returned to an authenticated caller only,
  for its own namespace; `counts` always. Every caller the control plane admits is authenticated, so in practice `chunks`
  is always present here; the builder still enforces it (`Explain_PerChunkOnlyForOwnNamespace`).
- Explain does not probe the prompt cache, so nothing is `prefix`.
- `predicted_ms` is the shadow plan's prediction whatever the requested policy: `reuse` = assembly + recompute (the
  request's `recompute.budget`, else 0.05) + prefill of the rest; `full_prefill` = the whole prompt at the backend's
  default rate (`chunk-cost-model.json`). `warnings` lists the codes (counts are in the inference report only).
- OpenAI surface only in v1 (the CLI's `select --server` sends OpenAI string-content messages, ruling 19).

## Control plane: prewarm and evict (spec §7.2, §10; the store tasks)

Both routes exist only with `--chunk-cache`, change the store only through the sole writer, and need
`--chunk-store-write` (without it they answer `503 api_error` / `chunk_store_unavailable`). The gates run in this order:
`Sec-Fetch-Site: cross-site` is `403`; a `Content-Type` other than `application/json` is `415` (a browser must preflight,
which the `Origin` rule refuses: no web page can drive them, CSRF); authentication as for the other routes (`401` /
`403`); the body is then read **before any slot is taken**, within `BodyReadTimeout` (10 s, a constant: later is `408
request_timeout`) and the size cap (`413`), so a trickling body holds nothing; then **one call per namespace at a time**
and at most `MaxConcurrentWrites` (2) prewarm / evict calls per server at once (`429 chunk_control_busy`, `Retry-After:
1`) (`WriteRoutes_BodyReadBeforeTheGate_PerNamespaceAndGlobalSlots`). When rate limiting is configured they are metered like inference. Bodies are parsed strictly: an
unknown or repeated field, a wrong type or a body that is not a JSON object is `400 invalid_request_error`, whose message
never echoes the content. Responses carry **counts only**: no text, hash, key, namespace or path, and so do the logs
(`WriteRoutes_NoNamespaceKeyTextOrPathInBodiesOrLogs`).

### `POST /v1/chunk-cache/prewarm`

Queues texts for a background encode in the caller's namespace. **Text only** (no hash list exists: a hash-only API
would be a membership oracle):

```json
{"texts": [{"id": "Program.cs#3", "text": "...", "source": {"path": "src/Program.cs"}}], "max_tokens": 20000}
→ {"requested": 1, "already_stored": 0, "queued": 1, "queued_tokens": 431, "refused": 0, "refused_reasons": []}
```

- Each text is tokenised as the chat path tokenises chunk text (`ChunkTokenization.Encode`, ruling 21) and keyed with the
  served model's identity in the caller's namespace. A text whose entry is already stored counts as `already_stored`
  (and refreshes its ledger row's last use). Every other text goes through the sole writer's admission
  (`ChunkStoreService.AdmitPrewarm`) with **every rule of [Store writes](#store-writes-the-sole-writer) except
  "seen in 2 requests"**, and **without recording a sighting**: the deny-list on `source.path` (`source` is advisory,
  ruling 14: `{path}` only; the server cannot rely on it), the PEM private-key sniff, CPU only and at most 512 tokens
  (ruling 65), the budget (`over_budget`), the writer not stopped, index-only, and the queue's bounds (256 items, 64 per
  namespace). Encodes then run under the idle rule exactly as encodes on miss.
- **Caps**: at most `MaxPrewarmItems` (256) texts per call (more is `400 chunk_too_many`, nothing queued); a body over
  `MaxPrewarmBodyBytes` (4 MiB) is `413`; `max_tokens` (optional, a positive integer) bounds the tokens one call queues:
  a text that would pass it is refused with `max_tokens`. The same text twice in one call is decided once.
- `refused_reasons` lists the distinct admission reasons (the `dotllm_chunk_store_skips_total` reason labels, plus
  `max_tokens` and `empty`), not per text. `queued` counts every text that is queued, a repeat of a text in the same call
  included, while `queued_tokens` counts each distinct text once. An unsupported model or backend is `400 chunk_model_unsupported`; no model
  loaded is `503 model_not_loaded`.

### `POST /v1/chunk-cache/evict`

```json
{"mode": "path", "path": "src/**", "dry_run": false}
→ {"evicted": 12, "freed_bytes": 48211968, "kept_referenced": 1, "dry_run": false,
   "removed": {"ledger_rows": 13, "index_files": 0, "sightings": 40, "queued": 2}}
```

- **`dry_run` defaults to `true`**: the counts of what would go, and nothing changes. Only an explicit JSON `false`
  deletes (a string `"false"` is a 400). `mode` defaults to `orphans`. `removed` and `warnings` are additive to the spec's
  output (`evicted`, `freed_bytes`, `kept_referenced`, `dry_run`).
- **Namespaces are the caller's** (server-assigned). The body's `namespace` is honoured only for an **admin** (an `admin`
  key, or the local user of a server without keys): `default` names the default namespace, otherwise a valid namespace
  name (`ns=` names, `k-<16 hex>`), else 400. From anyone else it is ignored with the warning `chunk_namespace_ignored`
  and the call acts on the caller's own namespace (`NonAdmin_EvictBodyNamespace_Ignored`).
- **Deleting store-wide or in another namespace needs an admin key** (ruling on the security review): `orphans`, an
  admin's store-wide `all`, or a body namespace other than the caller's, with `dry_run: false`, need a key marked `admin`
  in `--chunk-cache-api-keys`. The local user of a server **without keys** may prewarm, delete in its own (default)
  namespace, and dry-run anything, but gets `403 chunk_cache_admin_key_required` for those deletes
  (`Keyless_StoreWideDestructive_NeedsAdminKey`). **Proxy hazard:** without keys, anything that reaches the server through
  a same-host reverse proxy looks local, whatever forwarding headers it adds (`Forwarded`, `X-Forwarded-*` and `X-Real-IP`
  are not looked at; `Keyless_ForwardedHeaders_StayLocal_ButStoreWideDeletesStillNeedAnAdminKey`): with keys off, such a
  caller is the server's local admin and could read store-wide figures, prewarm, and forget the default namespace.
  **Set `--chunk-cache-api-keys` whenever anything other than local processes can reach the server.**
- The default `mode` is `orphans`, which is admin-only: a non-admin's `{}` is a 403; send `"mode": "namespace"` (or
  `path`).
- **Modes**:
  - `orphans`: entry files that neither an index file nor the server ledger references (store-wide, so **admin only**;
    others get `403 chunk_cache_admin_only`). Ledger keys and index keys stay (`Evict_Orphans_KeepsLedgerAndIndexKeys`).
  - `path` ("forget this repo"): in the namespace, the ledger rows whose first-seen `source.path` matches the glob
    (`PathGlob`: relative, `**` any depth, a pattern without `/` matches a file name at any depth, case-insensitive on
    Windows; rooted, `..` and negated `!` globs and globs over 1,024 characters are a 400), their entry files, the
    sighting rows and queued encodes with a matching provenance. Rows and sightings that carry no `source.path` (an
    annotation without `source`) are never reached by a path: use `namespace` for a full forget. Paths over 256 characters are held in the sighting table as a prefix plus a hash; forget also
    compares the requested path in that stored form, so a long literal path matches (`Evict_Path_LongPath_MatchesStoredForm`).
  - `namespace`: everything of the namespace: its ledger rows, its own **index files** (current names,
    `index-<chunker>.<model12>.<p8>.<ns-or-default>.json`; their index locks are taken first), the entries those rows
    and indexes referenced, its sighting rows and queued encodes
    (`EvictNamespaceMode_RemovesThatNamespacesIndexFilesRowsAndSightingsOnly`). Another namespace is never touched
    (`Evict_Namespace_CannotTouchOtherNamespace`).
  - `all`: the caller's namespace as `namespace` for a non-admin; for an admin without a body namespace, the **whole
    store**: every ledger row, every current-name index file, every sighting and queued encode, and every entry file
    (orphans and CLI entries too) except those a remaining (legacy-named) index references.
- **What forget keeps** (Task 15 rule (c)): an entry file whose key an index the call does not remove still references
  (a CLI index, which the server never edits; a legacy-named index, never deleted) keeps its file and is counted in
  `kept_referenced`; its ledger row still goes. So `path` removes K/V, ledger rows and sightings, and **reports** rather
  than edits the index references (the server owns none): rebuild or forget them with `chunkcache` (Task 17).
  The store registry (`chunkcache.json`) is advisory and drops rows of deleted index files on its next load.
- **No marker**: removing the rows and saving suffices. A later request with the same text is a first sighting again,
  so it is re-admitted only after two more requests (`Forget_ClearsSightings_NoMarker`). An encode already running when
  its namespace is forgotten is dropped before its write by a fence (the queue's sequence number), so nothing forgotten
  comes back (`Forget_DropsQueuedAndInFlightItems`). A forgotten item is checked again before it is encoded, so it is
  never encoded only to be thrown away. `namespace` / `all` drop every queued and in-flight item of the namespace (or the
  store); `path` drops only the items whose provenance matches, queued or in flight
  (`Forget_WriteTimeFence_InFlightOnly_PathDropsMatchingItemsOnly`). A request whose second sighting was counted before
  a forget but whose item is queued after it is a later request: its entry is stored.
- **Fail closed** (Review Focus): with writes disabled (an unreadable ledger at startup or after a failed save), a ledger
  file that no longer reads, or an unreadable index file, **nothing is deleted**, dry runs included: `503 api_error` /
  `chunk_store_unavailable` with the cause (`ledger_unreadable`, `index_unreadable`) in the message
  (`Evict_UnreadableLedger_FailsClosed`, `Evict_UnreadableIndex_FailsClosed`). The ledger file is retried briefly first
  (4 x 100 ms, awaited, a sharing violation is not corruption); a real (non-dry) evict that still cannot read it disables
  writes until it reads back, exactly as the writer's own failed save; a dry run never changes the writer's state.
  Fail-closed is store-wide: one unreadable index file (perhaps another tenant's CLI index) blocks every tenant's forget
  until it is repaired or removed.
- **Partial deletes are reported** (counts only): an index or entry file that cannot be deleted (held open by a scanner
  or a reader) is left in place, and the response adds the warning `chunk_evict_partial` and `failed: {entry_files,
  index_files}` (an index left in place keeps its entries referenced; an entry left in place is an orphan for a later
  `orphans` pass). A ledger save that fails is a 503 whose message says that no entry or index file was deleted and the
  rows stay, but the matching sightings and queued encodes (counted in the message) were already dropped: they come back
  with new sightings (`Evict_PartialDelete_WarnsWithCounts_FailedSave_SaysWhatWasDropped`). The rollback after a failed
  save re-reads the ledger with brief retries on the request thread (failure path only).
- **Locks** (lock order index then store): the index locks of the index files it deletes, then the writer's in-memory
  ledger gate (shared with the background writer), then the store lock for **one short scope**: verify the ledger and the
  indexes, save the ledger without the forgotten rows, delete the index files, re-read every reference from disk, delete
  the entry files no longer referenced. Each index file is read once for that decision, and every reference again just
  before the entry files are deleted. All locks are waited for asynchronously within the writer's lock timeout (10 s),
  never by sleeping on a request thread; a timeout is `503` (`lock_timeout`) with nothing deleted
  (`Evict_StoreLockHeld_LockTimeout_NothingDeleted`, `Evict_WhileWriterHoldsLedgerGate_WaitsThenSeesRow_OrTimesOutCleanly`:
  an evict that starts while the writer is mid-write waits for it and then sees its row). A **dry run** takes no lock
  file and reads the index files and the ledger file once, before the ledger gate, so it never stalls the writer on disk
  reads. Matching sightings holds the sighting table's lock while it scans (at most 100,000 rows), briefly delaying
  inference sightings.
- Deleted entry files count in `dotllm_chunk_store_evictions_total`; one log line per call gives the counts and the mode.

### Authentication and namespaces

dotLLM has no authentication anywhere else (`docs/SERVER.md` § Security). This is opt-in, covers the chunk-cache surface
only, and is **security-sensitive**: a per-chunk store answer tells the caller whether some text was cached, i.e. a
membership probe on source code (spec §10).

- **`--chunk-cache-api-keys <file>`**: UTF-8, one key per line: `<key> [admin] [ns=<namespace>]` (fields separated by
  spaces or tabs, each at most once, in any order after the key). Blank lines are skipped, and so is **every line whose
  first non-blank character is `#`**: a key can therefore never start with `#` (a line meant as such a key is silently a
  comment; the recipe below never produces one). Each line is trimmed. A key is printable ASCII without spaces. Any
  other character (non-ASCII, a control character, a no-break space) refuses the file at startup with the line number,
  never the key; so do a repeated key, an unknown or repeated field, an invalid `ns=` and a file with no key. An
  unusable path is a clean startup error, not a stack trace.
- **Keys must be at least 32 characters** (startup refuses a shorter one, naming its line) unless
  `--chunk-cache-allow-short-keys` is given, which accepts them with a warning per line. Length is not entropy: generate
  keys, e.g. 32 random bytes as base64url (43 characters):

  ```sh
  openssl rand 32 | basenc --base64url | tr -d '=\n'          # or: openssl rand -hex 32
  ```
  ```powershell
  [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).TrimEnd('=').Replace('+','-').Replace('/','_')
  ```
- **`admin`**: store-wide capabilities figures, `evict` of `orphans`, `evict` of another namespace (the body's
  `namespace`) and of the whole store (`all` without a namespace). Without a keys file the local caller is admin for
  figures, dry runs and its own namespace only: store-wide or cross-namespace **deletes** always need an `admin` key
  (proxy hazard: a same-host reverse proxy makes every caller look local).
- **`ns=<namespace>`**: the key uses that namespace (`ChunkNamespace.IsValid`, not empty, never `default`) instead of
  the one derived from it. Two keys may name the same namespace on purpose (a team; a rotated key keeping the old
  key's entries). A named namespace is not derived from the key, so the store's file names say nothing about the key
  (`KeysFile_NsField_MapsKeyToNamedNamespace`).
- **With keys**: the request must carry a listed key in `X-API-Key` (exactly one value; a blank one is no key, with no
  fall-back to `Bearer`) or `Authorization: Bearer`. Matching is constant-time
  (`CryptographicOperations.FixedTimeEquals` over the digests of every key, no early exit). Otherwise
  `401 authentication_error` (`WWW-Authenticate: Bearer`). A loopback caller needs a key too.
- **Without keys**: local callers only, in the default namespace (admin for figures and dry runs, not for store-wide
  deletes), else `403 permission_error`. Local means all of: the
  connection's remote address is loopback (IPv4-mapped too; an unknown address, as on a Unix socket, is not local); the
  `Host` header is `localhost` or a loopback address (any port); and there is no `Origin` header. The last two stop a
  web page in the user's browser: the server's CORS policy allows any origin, and DNS rebinding reaches loopback under a
  foreign host name. "Local" still admits **every process on the machine**, including a service that can be made to
  fetch URLs for someone else (SSRF), and behind a reverse proxy on the same host every caller is local: configure keys
  in both cases.
- **Rotation and revocation**: the keys file is read once at startup and never reloaded. **Revoking a key means
  editing the file and restarting the server.** A replacement key gets a new, empty derived namespace and strands the
  old key's entries (nothing maps one to the other; they age out with the store's eviction) unless both lines carry the
  same `ns=`, which is the rotation path.
- **Namespaces** are server-assigned, never caller-chosen: the key's `ns=`, else `ChunkNamespace.ForApiKey(key)` (`k-` +
  the first 16 hex of SHA-256 of the key), or `ChunkNamespace.Default`. They are hashed into every chunk key, so one
  key's **store** entries are never hits for another (`TwoKeys_CannotSeeEachOthersEntries`). They never appear
  in a response or a log line, nor do keys or key digests (`NamespaceNeverInResponseBodiesOrLogs` checks every body and
  the console and `ILogger` output). They **do** appear on disk: in store file names (`index-*.<ns>.json`) and, later,
  the server ledger. Treat the store root like the repository (spec §10).
- **Isolation stops at the store.** The prompt (prefix) cache is shared by all callers and knows nothing about
  namespaces: `prefix` status, `chunks.prefix`, `usage.cached_tokens` / `timings.cache_n` and time to first token show
  whether *anyone* recently sent the same prompt prefix. That exposure predates the convention (`cached_tokens` is
  unauthenticated) and nothing here adds to it, but a multi-tenant deployment does not get full isolation; a
  per-namespace prompt cache is an open item. Likewise `/metrics` is unauthenticated and its
  `dotllm_chunk_cache_*_total` counters are server-wide (no namespace labels): they show aggregate hit volume of every
  caller.
- **Raw keys in memory**: the keys file's keys are kept as SHA-256 digests only, but a presented key string lives on in
  the rate limiter's per-key buckets (for the process lifetime, when rate limiting is on), in `HttpContext.Items` and on
  the inference request (scheduler fairness). None of these is logged.
- **Inference** stays unauthenticated: an annotated request is never refused here. A caller the control plane would
  refuse is served normally, but its chunks are never looked up in the store (all misses, no warning) and
  `report: "per_chunk"` gives it the summary (`ReportPerChunk_UnauthenticatedGetsSummary`). The same rule gates store
  writes (prewarm, background encode, evict).
- **Stack traces**: the server runs `UseDeveloperExceptionPage()` unconditionally (pre-existing). The chunk-cache handlers
  catch their own failures (explain answers 500 `api_error` with a fixed message, capabilities degrades to an
  unsupported model), so no stack trace leaves them; other endpoints may still return one.
- **Known limit: the derived namespace is unsalted.** `ForApiKey` is a plain SHA-256 prefix, so anyone who can read the
  store root can test guessed keys offline at GPU speed (on the order of 10^10 guesses per second); the 64-bit
  truncation does not slow that down, and besides the `k-<16 hex>` in index file names, the producer tag in those names
  and the ledger rows can also confirm a guess. Long random keys (the 32-character minimum, the recipe above) or `ns=`
  names avoid it. The alternative is `HMAC-SHA256(server secret, key)` with the secret outside the store root: offline
  guessing would need the secret, but changing or losing the secret (or the derivation) re-keys every derived namespace
  (a key-scheme bump, `chunk-key/v5`) and orphans every stored entry, and a server-backed CLI could no longer derive its
  own namespace from its key. Left to the maintainer; v1 ships the unsalted form.

### Packaging and dependencies

`DotLLM.Server` references `DotLLM.ChunkReuse`, which brings `Microsoft.CodeAnalysis.CSharp` (Roslyn, for the symbol
chunker) and `DotLLM.Vulkan` into the dependency graph of the server package and of the `dotllm` CLI. With the flag off
the server never constructs the chunk-cache runtime (it is created on first use), so the store and key code is not
reached; whether the CLR loads the `DotLLM.ChunkReuse` assembly at all with the flag off has not been measured. The
opt-in AOT publish of `dotllm` (docs/AOT.md) has not been re-verified with this reference. Before upstreaming, the key,
store, tokenisation and identity types are candidates for a small assembly without Roslyn or Vulkan.

### MCP server (`chunkcache mcp`)

`chunkcache mcp [--store DIR] [--server URL] [--model FILE.gguf] [--threads N]` runs a stdio Model Context Protocol server
(newline-delimited JSON-RPC) with six tools: `chunkcache_build`, `chunkcache_select`, `chunkcache_status`,
`chunkcache_evict`, `chunkcache_prewarm`, `chunkcache_explain`. It is a thin wrapper: each call is turned into the argument
list of the matching `chunkcache <verb> --json`, run in process, and the verb's JSON is projected onto the tool's
`outputSchema` and returned as `structuredContent` plus a serialised text copy. A failing verb is a tool result with
`isError` (an `{"error": {"code", "message"}}` object), not a protocol error. Unknown arguments are rejected by the schema.

Defaults that protect: `chunkcache_evict` is a dry run unless `dry_run` is `false`; `chunkcache_build` only indexes
(tokenizer, nothing is encoded) unless `encode` is `true`, which queues the missing chunks on the server (an error without
a server); `max_encode_tokens` is a strict token cap, not a chunk count. The API key is read from `CHUNKCACHE_API_KEY` only
and is sent as `X-API-Key`, to the launch `--server`'s origin only (with the key set, a call whose `server` names another
origin is refused); it never appears in a frame or on stderr. Tools that open a model take an optional `model`
(a GGUF path) that overrides the launch `--model`.

Claude Code (`.mcp.json`):

```json
{"mcpServers": {"chunkcache": {"command": "chunkcache", "args": ["mcp", "--store", "./store", "--model", "M.gguf", "--server", "http://localhost:8080"]}}}
```

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
