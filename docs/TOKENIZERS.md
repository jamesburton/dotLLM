# Tokenizers & Chat Templates — dotLLM

## Tokenizer Types

### tiktoken-style BPE (Llama 3, GPT-4)

1. **Regex pre-tokenization**: Split input using a compiled regex pattern that separates words, numbers, whitespace, punctuation. Pattern is model-specific (e.g., Llama 3 uses a complex pattern handling Unicode categories).
2. **Byte-level BPE**: Each pre-token is converted to bytes, then BPE merges are applied using a **priority queue** (merge with lowest rank first). This is 3-6× faster than the iterative pair-counting approach.
3. **Vocabulary**: Direct byte-pair → token ID mapping.

Implementation: Trie for prefix matching, compiled regex. Vocabulary loaded from GGUF `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`.

### SentencePiece BPE (Llama 2)

- Unicode code-point level (not byte-level).
- Space represented as `▁` (U+2581) prepended to words.
- Token scores (float) determine merge priority.
- Protobuf `.model` files (but GGUF embeds the vocabulary directly).

Vocabulary from GGUF: `tokenizer.ggml.tokens` + `tokenizer.ggml.scores`.

### HuggingFace tokenizer.json

JSON format containing: model type, vocabulary, merges, pre-tokenizer config, normalizer, post-processor, added tokens. Full specification of the tokenization pipeline.

Used when loading models from SafeTensors (which don't embed tokenizer in the weight file).

#### Adapter: `HfTokenizerJsonParser` + `HfBpeTokenizerFactory`

The `DotLLM.Tokenizers.Hf` namespace parses `tokenizer.json` and routes the
pipeline to the matching BPE encoder. The parser strips down to the fields
actually used for encode/decode (model vocab/merges, pre-tokenizer,
normalizer, decoder, added tokens); post-processor / template logic is
handled elsewhere.

Two pipelines are supported today:

| Pipeline | Pre-tokenizer (JSON) | Decoder (JSON) | Norm | BPE encoder | Checkpoints |
|---|---|---|---|---|---|
| **SentencePiece / Metaspace** | `Metaspace` or `null` | `Sequence[Replace, ByteFallback, Fuse, Strip]` | — | `BpeTokenizer.CreateSentencePiece` | Llama 1/2, Mistral, TinyLlama, Phi-3.5-mini, ib-ssm Mamba-3 |
| **GPT-2 / ByteLevel** | `ByteLevel` **or** `Sequence[Split, ByteLevel]` | `ByteLevel` | NFC (Qwen) | `BpeTokenizer.CreateTiktoken{,WithRegex}` | Qwen2 / Qwen2.5, Granite-3 dense, GPT-2 proper |

##### ByteLevel contract

`ByteLevelPreTokenizer` owns the GPT-2 `bytes_to_unicode` mapping that
defines the alphabet the BPE merges operate over:

- Printable ASCII 33–126 and Latin-1 supplement 161–255 (minus 173) map to
  their own code points.
- All remaining bytes (0–32, 127–160, 173) — the control and whitespace
  bytes — are pushed into `U+0100 + n` where `n` counts unassigned bytes in
  ascending order. The effect is that BPE never has to merge across
  boundaries caused by control bytes, because those bytes now live in a
  private-range of valid Latin-extended characters.
- UTF-8 continuation bytes are naturally handled: `é` = `0xC3 0xA9` →
  `"Ã©"` in the byte-level alphabet, which the merge table resolves to a
  single token if the vocabulary contains that byte pair.

`ByteLevelDecoder` is the inverse: reverse each token's chars to bytes,
concatenate across the full sequence, UTF-8 decode.

##### Regex ordering (correctness-critical)

HF `ByteLevel` applies its pre-tokenization regex to the **raw text**
before byte-mapping. The dotLLM encoder mirrors that order — applying the
regex to byte-mapped text would misclassify multi-byte chars (`é` splits
to `Ã` + `©` across segment boundaries, losing the merge).

Regex source priority inside the factory:

1. If `Sequence[Split(regex=X), ByteLevel(use_regex=false)]` (Qwen2 shape),
   use `X` compiled from the JSON. The Qwen2 pattern is
   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
   — close to but distinct from the built-in GGUF `llama3` pattern.
2. Else if standalone `ByteLevel(use_regex=true)` (Granite-3, GPT-2), use
   the cached default GPT-2 pattern in `ByteLevelPreTokenizer.DefaultGpt2Regex`.
3. Else (`use_regex=false` with no upstream Split), feed the whole byte-
   mapped string to BPE as one segment.

##### Normalizer

When the JSON declares a `normalizer` of kind NFC/NFD/NFKC/NFKD, the
factory wraps the inner tokenizer in a `NormalizingTokenizer` decorator
that `text.Normalize(form)`s input before encode. Decode is untouched.
Qwen2/Qwen2.5 declare NFC — necessary to keep composed vs decomposed
forms of accented chars on the same code path the merge table was
trained against.

##### Architecture → pipeline map

| Arch | Tokenizer | Pipeline |
|---|---|---|
| Llama 1 / 2 | SPM BPE | Metaspace |
| Mistral | SPM BPE | Metaspace |
| TinyLlama | SPM BPE | Metaspace |
| Phi-3.5-mini | SPM BPE | Metaspace (not ByteLevel as some other Phi variants) |
| Mamba-3 (ib-ssm) | SPM BPE | Metaspace |
| Qwen2 / Qwen2.5 | GPT-2 BPE | `Sequence[Split, ByteLevel]` + NFC |
| Llama 3 | GPT-2 BPE | ByteLevel (standalone, use_regex=true) |
| Granite-3 dense | GPT-2 BPE | ByteLevel (standalone, use_regex=true) |
| Granite-3 MoE | GPT-2 BPE (vocab.json, no tokenizer.json) | — out of adapter scope; uses slow-tokenizer path |
| GPT-2 / GPT-4 | GPT-2 BPE | ByteLevel |

##### Known gaps

- **Multi-step Sequence pre-tokenizers.** DeepSeek-V2/V2-Lite uses
  `Sequence[Split, Split, Split, Split, Split, Digits, ByteLevel]`. The
  adapter only recognizes the `[Split, ByteLevel]` shape; other
  compositions surface as `HfPreTokenizerKind.Sequence` and the factory
  throws. Tracked as a P0.1 follow-up.
- **Slow-tokenizer checkpoints.** Granite-3 MoE (and older Llama-family
  models) ship `vocab.json` + `merges.txt` rather than `tokenizer.json`.
  A separate parser path is required — not in scope for the
  `tokenizer.json` adapter.
- **Chat templates and post-processors** (BOS/EOS insertion, token type
  IDs) are handled by callers, not the adapter.

##### Entry points

- `HfTokenizerJsonParser.Parse(jsonContent)` → `HfTokenizerSpec`.
- `HfBpeTokenizerFactory.Create(spec, bosId=-1, eosId=-1)` → `ITokenizer`.
- `HfBpeTokenizerFactory.TryLoadFromDirectory(dir)` → `ITokenizer?`
  (one-liner for HF checkpoint dirs).
- `ModelLoader.LoadTokenizerFromHfDirectory(path)` → same, from the model
  loader's public API.

## ITokenizer Interface

```
ITokenizer:
  Encode(text) → int[]
  Decode(tokenIds) → string
  DecodeToken(tokenId) → string
  VocabSize → int
  BosTokenId → int
  EosTokenId → int
  CountTokens(text) → int   // Fast count without full encode
```

## Chat Template Engine

### Purpose

Models require specific prompt formatting. The OpenAI API sends `messages[]` — the engine must format them correctly.

### Template Format

Templates use **Jinja2 syntax** (HuggingFace standard), stored in:
- GGUF: `tokenizer.chat_template` metadata key
- HuggingFace: `tokenizer_config.json` → `chat_template` field

### Required Jinja2 Subset

Full Jinja2 is not needed. Required features:
- Variable interpolation: `{{ message.content }}`
- For loops: `{% for message in messages %}`
- Conditionals: `{% if message.role == "system" %}`
- String filters: `{{ text | trim }}`, `{{ text | strip }}`
- `raise_exception("error message")`
- Basic expressions and comparisons

### IChatTemplate Interface

```
IChatTemplate:
  Apply(messages: IReadOnlyList<ChatMessage>, options: ChatTemplateOptions) → string
```

```
ChatMessage:
  Role: string ("system" | "user" | "assistant" | "tool")
  Content: string
  ToolCalls: ToolCall[]?     (for assistant messages with tool calls)
  ToolCallId: string?        (for tool result messages)

ChatTemplateOptions:
  AddGenerationPrompt: bool  (append assistant turn prefix)
  Tools: ToolDefinition[]?   (for tool-calling models)
```

### Known Template Formats

**Llama 3**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**ChatML** (Qwen, many others):
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

**Mistral**:
```
[INST] {user_message} [/INST]
```

### Fallback

If no template found in model metadata, use configurable default (ChatML). Log a warning.

## Tool Calling Protocol

### Flow

1. Request includes `tools` definitions (name, description, parameter JSON schema).
2. Chat template formats tool definitions into prompt.
3. Model generates tool call JSON: `{"name": "func", "arguments": {...}}`.
4. **Constrained decoding** ensures valid JSON matching the tool schema.
5. Server detects tool call, returns `finish_reason: "tool_calls"`.
6. Client executes tool, sends result as `tool` role message.
7. Template formats result; model generates final response.

### IToolCallParser

```
IToolCallParser:
  TryParse(generatedText) → ToolCall[]?
  IsToolCallStart(text) → bool
```

Models signal tool calls differently:
- Special tokens (`<|tool_call|>`, `<|python_tag|>`)
- JSON patterns in output
- Model-specific formats

The parser is associated with the chat template — each template knows its model's tool calling convention.