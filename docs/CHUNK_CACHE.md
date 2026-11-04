# Chunk cache (server side)

Server-side notes for the `chunk_cache` wire convention (chunk-level K/V reuse for agent harnesses). The convention, its
fields and its error codes are specified outside this repo (ChunkedkvCache, spec `2026-11-02-chunk-cache-integration.md`,
wire contract `docs/contracts/chunk-cache-v1/`). This page records what the dotLLM server does; it grows with each server
task. Nothing here is behind a flag yet: there is no `--chunk-cache` option today.

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
