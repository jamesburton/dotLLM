# KV-Cache — dotLLM

## Purpose

The KV-cache stores previously computed key and value vectors for all layers, avoiding O(n²) recomputation during autoregressive generation. At each decode step, only the new token's K/V are computed and appended.

## Memory Consumption

Llama 3 8B, FP16, 2048 tokens:
```
2 (K+V) × 32 layers × 8 KV heads × 128 head_dim × 2048 tokens × 2 bytes
= ~1 GB
```
Scales linearly with sequence length and batch size. Dominant memory consumer in production.

## Simple KV-Cache (Phase 1)

Pre-allocated contiguous buffer per sequence:
```
K_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
V_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
```
Simple indexing: `K_cache[layer][head][pos] = new_K`. Wastes memory for short sequences.

## Paged KV-Cache

Inspired by OS virtual memory paging. This is the **memory management** half of PagedAttention (vLLM): block-based allocation, ref counting, CoW. The **kernel** half (attention reading non-contiguous blocks directly) is a future step — current kernels see contiguous buffers via staging-buffer gather.

### Why

**Batch serving memory efficiency**: `SimpleKvCache` pre-allocates `maxSeqLen × layers` per sequence. With 10 concurrent requests averaging 200 tokens but `maxSeqLen=4096`, that's 40K tokens worth of buffers for 2K tokens of actual data — ~95% waste. A shared block pool allocates on demand and reclaims blocks on completion, keeping waste <4%.

**Foundation for downstream steps**: The ref counting and CoW infrastructure enables:
- **Step 37 (advanced prefix sharing)** — hard requirement. Cross-sequence block sharing (e.g., shared system prompt) is impossible without block tables and ref counting.
- **Step 43 (speculative decoding)** — benefits from cheap block-level rollback on draft rejection and CoW fork for draft branching, though `SimpleKvCache.SetCurrentLength()` truncation works as a fallback.
- **Beam search** (future) — beams share prefix blocks via `Fork`, diverge via CoW.

**When it doesn't help**: Single-sequence CLI inference. One sequence uses most of its allocation, and the staging-buffer gather adds overhead. This is why paged is opt-in for `run`/`chat` and default-on only for `serve`.

### Design
- Divide cache into fixed-size **blocks** of B tokens (B = 16).
- **Block table** per sequence: maps logical positions to physical blocks (page table).
- **Free pool**: blocks allocated on demand, returned on completion.
- Memory waste: <4% (vs ~60% for static pre-allocation).

### Operations
- **Allocate**: When sequence needs more blocks, pop from free pool.
- **Free**: On sequence completion/eviction, return all blocks to pool.
- **Copy-on-write**: For beam search — beams share prefix blocks (ref-counted). On divergence, copy the shared block.
- **Fork**: For prompt caching — new sequence references existing prefix blocks.

### Attention Integration (v1: Staging Buffer)

Current implementation uses a staging buffer approach: `GetKeysRef()`/`GetValuesRef()` gather block data into a pre-allocated contiguous buffer before each attention call. This avoids modifying attention kernels. The gather cost is O(seqLen × headDim) — negligible compared to attention's O(seqLen² × headDim).

Future optimization: specialized paged attention kernels that read blocks directly via block table indirection, eliminating the staging copy.

### Key Classes

- `KvBlockPool` — Fixed-size pool of KV blocks. Per-layer contiguous storage, 64-byte aligned. Free list with `Interlocked` ref counting for thread safety.
- `KvBlockTable` — Per-sequence mapping from logical positions to physical block IDs. Supports `Fork` (shared prefix) and `EnsureWritable` (copy-on-write).
- `PagedKvCache : IKvCache` — Paged cache with staging-buffer gather. Drop-in replacement for `SimpleKvCache`.
- `PagedKvCacheFactory` — Creates `PagedKvCache` instances backed by a shared `KvBlockPool`.

### CLI

```
# CLI (run/chat): paged is opt-in
dotllm chat model.gguf --paged

# Server (serve): paged is on by default, opt-out with --no-paged
dotllm serve model.gguf                 # paged KV-cache active
dotllm serve model.gguf --no-paged      # simple KV-cache

# API-only server (no web UI)
dotllm serve model.gguf --no-ui
```

### Limitations (v1)

- CPU `PagedKvCache` only. CUDA and hybrid models fall back to their native KV-cache.
- Not compatible with quantized KV-cache (`--cache-type-k`/`--cache-type-v`). Falls back to `QuantizedKvCache` with a warning.
- Staging buffer means `GetKeysRef`/`GetValuesRef` results are only valid until the next call (shared buffer).

## KV-Cache Quantization

Compress cached K/V to extend context capacity:

| Format | CPU (vs FP32) | GPU (vs FP16) | Quality Impact |
|--------|---------------|---------------|----------------|
| FP32 (CPU) / FP16 (GPU) | 1× (baseline) | 1× (baseline) | None |
| Q8_0 | 3.76× | 1.88× | Minimal |
| Q4_0 | 7.11× | 3.56× | Small (for older tokens) |

### Implementation

**Dual-region storage** with quantize-on-evict:
1. **Quantized buffer** (append-only): Q8_0/Q4_0 blocks for positions outside the window
2. **Full-precision ring buffer**: FP32 (CPU) or FP16 (GPU) for the most recent W tokens

On each new token write, the oldest window entry is quantized and appended to the quantized buffer. Separate key/value quantization types are supported (e.g., Q8_0 keys + Q4_0 values).

### Key Classes
- `QuantizedKvCache` — CPU implementation in `DotLLM.Engine`
- `CudaQuantizedKvCache` — GPU implementation in `DotLLM.Cuda`
- `IQuantizedKvCache` — Interface extending `IKvCache` for quantized access
- `KvCacheConfig { KeyDType, ValueDType, MixedPrecisionWindowSize }` — Configuration

### Attention Integration
- **CPU**: Per-tile dequantization inside tiled attention kernel (`Attention.ExecuteTiledQuantizedHead`). Phase 1 processes quantized region with on-the-fly dequant, Phase 2 reads FP32 window directly.
- **GPU**: Scratch-buffer approach — dequant quantized region + ring-ordered window copy into temporary FP16 buffer, then standard attention kernel.

### CLI
```
--cache-type-k q8_0    # key quantization (f32, q8_0, q4_0)
--cache-type-v q4_0    # value quantization (f32, q8_0, q4_0)
--cache-window 64      # recent tokens in full precision (0 = all quantized)
```

Orthogonal to weight quantization — Q4_K_M model can use Q8_0 KV-cache.

## MLA KV-Cache (DeepSeek-V2 / V3)

MLA decouples Q head-dim from V head-dim (V2-Lite: qk=192, v=128) and adds
a shared MQA-style rope-K that broadcasts across heads — neither fits the
per-head-uniform `IKvCache` shape that GQA/MHA caches assume. A dedicated
`MlaExpandedKvState` lives next to `TransformerModel` for this reason.

**Loader default**: HF and GGUF config extractors both set `MlaConfig.UseHybridMlaCache = true` for `Architecture.DeepSeekV2` / `Architecture.DeepSeekV3`, so production code paths get **Phase C** (hybrid latent + absorbed decode) without needing per-call configuration. Phase A remains active and is the numerical oracle; tests that build `MlaConfig` directly (bypassing the loader) still default to Phase A. The default flip lives at commits `4b54a72` (HF) and `4724397` (GGUF) — pre-flip, V2-Lite at `max_position_embeddings=163840` allocated ~68 GB and OOM'd on most hosts.

### Phase A — expanded reference cache

`src/DotLLM.Models/Architectures/MlaExpandedKvState.cs`. Per layer:

- `K_nope[layer]` : `[maxSeqLen, numHeads * qkNopeHeadDim]` — per-head non-rope K
- `V[layer]` : `[maxSeqLen, numHeads * vHeadDim]` — per-head V
- `KPe[layer]` : `[maxSeqLen, qkRopeHeadDim]` — shared rope-K (post-rotation)

All 64-byte aligned native memory, lazily constructed on the first MLA
forward and reset when the caller signals a fresh sequence by passing
`positions[0] == 0`. Not re-entrant; single-stream only (beam search or
batching needs per-sequence instances). Caller-supplied `IKvCache` is
ignored for MLA layers.

This layout is the PoC scalar kernel's scratch layout made persistent —
storage is 1:1 with what the kernel already computes, so zero shape
translation. Memory: ~16.6 KB per token per layer at F32 for V2-Lite;
on a 27-layer 8K context that's ~3.6 GB. The purpose of Phase A is
**correctness oracle**: generation works end-to-end and a split call
(prefill + step-by-step decode) produces logits that match a single-call
forward over the combined range within 1e-4.

### Phase B — pure latent + W_UK absorbed (landed)

The production memory win (per the DeepSeek-V2 paper, §2.1.2): store
the *compressed* latent `c_kv[kv_lora_rank]` per token (512 floats for
V2-Lite) alongside the shared `k_pe[qk_rope_head_dim]` (64 floats) —
a single `[kv_lora_rank + qk_rope_head_dim] = 576` value per token per
layer, 7.2× smaller than Phase A at F32. Attention math absorbs `W_UK`
into Q on-the-fly: `Q_latent[h] = Q_nope[h] @ W_UK_T[h]` (size
kv_lora_rank) and `score[h, t, s] = Q_latent[h] · c_kv[s] + Q_pe[h]
· k_pe[s]`. Output uses absorbed `W_UV`: `out[h] = W_UV[h] @ (softmax
· c_kv)`. vLLM's MLA backend is the reference implementation.

Lives at `src/DotLLM.Models/Architectures/MlaLatentKvState.cs` +
`src/DotLLM.Cpu/Kernels/MlaAttention.ExecuteLatent`. Selected via
`MlaConfig.UseLatentCache = true` (mutually exclusive with
`UseHybridMlaCache`).

### Phase C — hybrid: latent persistence + Phase A-equivalent prefill expand + absorbed decode (landed, default)

The production-shipping path mirrors vLLM's MLA backend: prefill
(`seqLen > 1`) expands cached latents through `W_UK` / `W_UV` into
local scratch and runs the standard 192-dim per-head MHA loop
(compute-bound at long seqKv); decode (`seqLen == 1`) delegates to
`ExecuteLatent` — the absorbed 576-dim MQA-style read of the compact
latent cache (bandwidth-bound at decode). Both paths persist the
SAME latent form (c_kv + k_pe per token) to `MlaLatentKvState` —
Phase A's expanded per-head K_nope/V is local prefill scratch and is
discarded. A decode step therefore consumes exactly the latents a
pure-Phase-B prefill would have written, so the absorbed kernel can
run over them without re-expansion.

Lives at `src/DotLLM.Cpu/Kernels/MlaAttention.ExecuteLatentHybrid`.
Selected via `MlaConfig.UseHybridMlaCache = true` (mutually
exclusive with `UseLatentCache`). **Default for DeepSeek-V2/V3 from
the loaders.**

Phase A is the numerical oracle for Phase B / Phase C: oracle tests
prove split-call match against the expanded-cache reference at 1e-3
drift on real-weight prompts.

## Simple Prompt Caching (Step 54)

Live KV-cache reuse for multi-turn conversations. No paged attention required — works with `SimpleKvCache`.

### Design: Live Reuse (not Snapshots)

After each generation, the KV-cache is transferred to a `PrefixCache` instead of being disposed. On the next call:

1. `PrefixCache.FindMatch(promptTokenIds)` scans entries (max 1–4) with element-wise prefix comparison (`MemoryExtensions.CommonPrefixLength`, SIMD-vectorized).
2. On hit: `SimpleKvCache.SetCurrentLength(matchedTokens)` truncates visible length. Suffix tokens are prefilled at positions `[matchedLen..promptLen)`.
3. On miss: fresh KV-cache allocated, full prefill as usual.

No data copying on cache hit. The same KV-cache object persists across calls.

### Key Classes

- `PrefixCache` — LRU cache with configurable max entries. Owns cached KV-cache instances.
- `PrefixCacheEntry` — Token sequence + live KV-cache + LRU timestamp.
- `SimpleKvCache.SetCurrentLength(int)` — Truncates visible length for prefix reuse.

### Multi-turn Chat Pattern

Each turn's prompt = previous prompt + assistant response + new user message. The stored token sequence (prompt + generated) shares a prefix with the new prompt. Typical cache hit rate: near 100%.

### CLI

```
--no-prompt-cache      # Disable (enabled by default in chat/serve)
--prompt-cache-size 1  # Max cached sessions (1 for chat, 4 for serve)
```

### Scope

- CPU `SimpleKvCache` only. QuantizedKvCache and GPU caches fall back to no caching.
- Cache cleared on model swap, `/clear` command (CLI), or `POST /v1/cache/clear` (server).

### Stats

`InferenceTimings.CachedTokenCount` reports how many prompt tokens were served from cache. Displayed in CLI output, API `timings.cached_tokens`, and Chat UI stats bar.

## Advanced Prompt Caching / Prefix Sharing (Future — Step 36+)

Requires paged KV-cache. Enables cross-request prefix sharing (e.g., shared system prompts across users).

### Problem
Many requests share the same system prompt (e.g., all chat requests in a deployment).
Recomputing KV-cache for the shared prefix is wasteful.

### Solution: Prefix Trie
- Maintain a **trie** of recently computed prompt prefixes, keyed by token sequences.
- On new request: walk the trie matching the prompt's token sequence.
- If match found: share the cached KV blocks (read-only), only prefill the new suffix.

### Implementation
- Shared blocks use **reference counting**. Freed when all referencing sequences complete.
- **LRU eviction** when memory scarce. Frequently used prefixes (system prompts) stay cached.
- **Explicit registration**: Server API accepts `prefix_id` for deterministic caching.

### Integration with PagedAttention
The prefix trie stores references to physical KV blocks. New sequences get their own block table with shared prefix entries pointing to existing blocks, plus new blocks for the suffix. Copy-on-write if modification needed (rare — KV cache is append-only).