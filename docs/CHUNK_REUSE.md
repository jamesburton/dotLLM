# Chunk-level KV reuse (`DotLLM.ChunkReuse`)

Library for reusing the K/V of a **chunk** of tokens (a file, a function, a block of lines) across prompts, instead of
re-running prefill over it every time. A chunk is encoded **once, in isolation** at positions `0..n-1`; later prompts that
contain the same chunk at another offset get its K/V copied into their cache with the keys **re-rotated** to the new
positions. Tracking issue: #1542.

Prefix caching (`docs/KV_CACHE.md`) reuses K/V only for an identical prompt *prefix*. Chunk reuse also covers a chunk in
the middle of a prompt, at the price of exactness: a reused chunk never attended to what precedes it in the new prompt (see
[Exactness](#exactness)).

`DotLLM.ChunkReuse` is the core: no Roslyn and no Vulkan dependency, so `DotLLM.Server` can reference it. Nothing in the
engine or the server uses it yet.

## Encode and assemble

| type | role |
|---|---|
| `ChunkEncoder` | `Encode(model, tokenIds)`: one isolated prefill at `0..n-1` into a fresh exact-F32 cache, then a host copy of every layer's K/V rows as a `ChunkKvEntry`. `EncodeResident` keeps the cache itself (`ResidentChunkKvEntry`). Entries carry the producer fingerprint and the backend tag. |
| `ChunkAssembler` | `Assemble(destination, chunks, expectedFingerprint)`: writes the chunks back to back from `destination.CurrentLength`, re-rotating each chunk's keys by its offset; values are copied unchanged. Refuses an entry of another fingerprint. |
| `RopeShift` | `Apply(keys, headDim, delta, rope)`: dotLLM stores K **after** RoPE, so a key written at `p` is `R(p)·k`; rotating by `R(Δ)` gives `R(p+Δ)·k`, the key the model writes at `p+Δ`. |
| `RopeParameters` | `FromConfig(config)`: rotated dims, theta and pairing (NeoX / normal). Exact for plain RoPE. NoPE layers are left untouched by the assembler. |
| `ChunkedContext` | Token layout of an assembled context: chunk starts, the query tail, `ChunkOf(token)`. |

**Backends.** Row IO and the key shift go through `IChunkBackend`; `CpuChunkBackend` is the reference (host
`SimpleKvCache`, exact F32). Every entry is stamped with its backend tag (`ChunkBackendTags.Cpu` = `cpu`); the tag is part of
the fingerprint and of the key, so CPU and other-backend entries never mix: a mismatch is a miss, never a conversion.

## Identity: fingerprint and key

A stored chunk's K/V depend on the model, its weights, the tokenizer, the RoPE configuration, the KV layout, the backend and
the engine's K/V output. Two layers capture that:

- **`ChunkFingerprint.Compute(config, modelId, quantization, tokenizerId, backend)`**: the producer identity. Pass the real
  config: RoPE scaling (type, factor, original context, YaRN betas), NoPE layers and per-layer KV layout reach the key only
  through the fingerprint.
- **`GgufIdentity`**: content-based ids for a GGUF, so a renamed file keeps its entries.
  - `ModelId(gguf)` hashes every shard (index, length, header with metadata and tensor table, first and last 4 KiB of each
    tensor). It misses edits strictly inside a tensor more than 4 KiB from both ends with identical metadata; dense changes
    (fine-tunes, merges, re-quantisation) change every tensor's head. `ModelId(gguf, fullHash: true)` hashes every byte
    (`gguf-full-sha256:` prefix).
  - `TokenizerId(metadata)`: SHA-256 of the `tokenizer.ggml.*` metadata (the chat template does not take part).
  - `QuantizationLabel(metadata)`: from `general.file_type` (e.g. `ftype-15`).
- **`ChunkReuseSchema.KvOutputVersion`** (an integer, not the build SHA, so a rebuild keeps entries valid): bump it by hand
  whenever a change moves the K/V a chunk encodes to. `ChunkReuseSchemaTests` holds a golden K/V summary per version for
  the synthetic test model (relative tolerance 1e-4, because CPU SIMD paths differ by ISA).
- **`ChunkKey.Create(fingerprint, tokenizerId, rope, backend, kvDtype, tokenIds, kvOutputVersion, positioning, @namespace)`**: the content
  address, key scheme `chunk-key/v4`: SHA-256 over a length-prefixed canonical encoding of the scheme, the namespace, the
  fingerprint, tokenizer id, RoPE parameters, schema version, backend, KV dtype, positioning and the token ids. No file path,
  so identical chunk text in two files is one entry.
  - `ChunkPositioning.Isolated` (an encoder output at `0..n-1`) is the only kind an assembler places; K/V computed with
    preceding context belong under `Contextual` and never answer an isolated lookup.
  - **Namespaces** (`ChunkNamespace`) partition a store: `Default` (`""`, a single-user local store, hashed as a zero
    length, never skipped) or `ForApiKey(key)` = `"k-"` + 16 hex of SHA-256 of the key (never the key itself). A valid
    namespace matches `^(|k-[0-9a-f]{16}|[a-z0-9._-]{1,64})$` except the reserved `default`.

## Chunkers and tokenisation

`IChunker.Chunk(SourceFile)` returns `Chunk`s (file path, 1-based lines, UTF-16 offsets, token ids, `WireEligible`).

- **`ChunkTokenization.Encode(tokenizer, text)`** is the one way library code turns chunk text into ids:
  `EncodeRaw(MarkUntrustedSpecials(text))`, exactly as the chat path tokenises user / tool content (#1339). A special-token
  spelling such as `<think>` inside a chunk is encoded as plain text, as the server sees it. No BOS. Plain text encodes as
  `EncodeRaw` (existing entries stay valid). It is the same *function* as the chat path, not a promise that a span's ids
  inside a rendered prompt equal its standalone ids (assistant reasoning tags, template-boundary merges): callers that
  locate spans must compare ids.
- **`LinesChunker`** (`lines-N`): text first. The file text is cut into pieces of at most N tokens (binary search over
  code-point boundaries, preferring the last line end; never splitting a surrogate pair or a CRLF), and every piece is
  tokenised alone, so a server that receives the piece as text reproduces its ids exactly.
- **`FixedSizeChunker`** (`fixed-N`): token-id slices of the whole file's ids. Cheap, but a slice is not text-reproducible
  (BPE merges across a boundary, a slice ending inside a multi-byte character).
- **`WireChunking`**: which chunker names may be used for chunks that travel as text (`Eligible`, `IsWireChunker`;
  `lines-512` waits in `PendingE4c` for its eligibility gate), the wire hash (`sha256:` + lowercase hex of the UTF-8 text) and `VerifyRoundTrip`, which re-encodes
  each chunk's text and reports the chunks whose ids differ.

## On-disk store

`IChunkStore` (`TryGet`, `Put`, `Contains`, `Evict`, `Keys`, `TotalBytes`) and `FileChunkStore`, one file per entry at
`root/<hex[0..2]>/<hex>.ckv`.

- **Format** (little-endian, `FileChunkStore.FormatVersion` = 2): magic `CKVS`, version, KV dtype, the 64-byte key, model
  fingerprint, **backend tag**, token count, layer count, per-layer K and V element counts, payload size; then the token ids,
  K of every layer, V of every layer. For Qwen3-4B (36 layers, 8 KV heads, head dim 128) F32 K+V cost 8,192 bytes per token
  per layer: a 512-token chunk is ≈ 144 MiB.
- **Atomic writes.** `Put` writes a unique temp file in the shard directory, flushes it, then renames it over the final
  name; a crash or a concurrent writer never leaves a partial `.ckv` at the final path. `SweepTempFiles(olderThan)` removes
  temp files a crashed writer left behind; `GetStats()` counts entries and temp files separately.
- **Reads are validated.** A read memory-maps the file and checks magic, version, dtype, key, backend tag and the exact file
  length. Truncation, trailing bytes, another version (including version 1, which had no backend tag), another key's file
  or an unsupported dtype is a **miss**, never a wrong hit. `Contains` and `TryReadFingerprint` read the header only.
- **Busy destination** (Windows cannot replace a mapped file): if the destination is already a valid entry for the key,
  `Put` returns; if it is invalid and still held by a reader, `Put` throws `ChunkStoreBusyException` at once and stores
  nothing (no retry loop; a later `Put` repairs it).
- **Backend tag.** The store reports the tag as written; a reader compares it with its own backend. CPU and other-backend
  entries never mix: the backend is in the key and in the header.
- `FileChunkStore.OpenExisting(root)` opens without creating the directory (read-only consumers).
- `FileChunkStore` takes no lock: each put is atomic on its own. Locks for writers that share a root with eviction come
  with the chunk index (later PR).

## Store-backed assembly

`StoredContextAssembler.Assemble(chunks, model, store, destination?, query, @namespace)` returns
`StoredAssembly(Context, Stats, Entries)`:

- **Lookup**: each chunk under `KeyOf(config, ids, @namespace)` (isolated positioning, the assembler's backend, F32).
- **Counts**: `AssemblyStats(Hits, Misses, EncodedTokens, Repeats)`, with `Hits + Misses + Repeats` = chunk count; a key
  that repeats within one call reuses the first entry (no read, no encode).
- **Hit check**: a hit must carry this fingerprint, this backend, exactly the chunk's token ids and one `[n, stride(l)]` K and
  V array per KV layer. A well-formed entry that fails is evicted, re-encoded and replaced (`Rejected`).
- **Store failures are misses**: a damaged, truncated, old-format or foreign entry is re-encoded and stored. If the store
  cannot write (`ChunkStoreBusyException` on put, an I/O or access error evicting a rejected entry) the fresh entry is used
  unstored and counted in `StoreFailures`. A store-lock timeout (`ChunkStoreLockTimeoutException`) propagates.
- **Validation first**: chunks, model identity, namespace and destination (exact host F32, layer count, strides, capacity)
  are checked before any store read or encode; on a throw the destination is unchanged.
- **Non-empty destination**: chunks are appended at `CurrentLength` (`DestinationOffset`); keys are re-rotated to those
  absolute positions.
- **Identity contract.** `new StoredContextAssembler(fingerprint, tokenizerId)` trusts the caller: a fingerprint that does
  not belong to the model passed to `Assemble` would store one model's K/V under another's keys (store poisoning).
  `StoredContextAssembler.ForModel(config, modelId, quant, tokenizerId)` keeps the fingerprint inputs and refuses a model
  whose configuration does not reproduce the fingerprint; use content-based ids (`GgufIdentity`) for `modelId`.
- **Isolated encodes only**: entries must be `ChunkEncoder` outputs of the chunk alone at `0..n-1`.
## Exactness

Assembly is the "naive concatenation" cache: each chunk's K/V are exact for the chunk alone, with correct positions, but
carry no attention to earlier chunks or the prompt before them. Output with reused chunks is therefore **not** identical to a
full prefill. Callers that need exact output must not place reused chunks (the server's planned `exact` policy only reports
what would hit). Measured quality of reuse is tracked in #1542.

## Precision

`ChunkKvEntry` holds F32 rows because the CPU reference cache (`SimpleKvCache`) is F32; the encoder reads them unchanged.
The KV dtype is part of the key, so compact formats (F16, Q8_0) can be added later without colliding with F32 entries.
