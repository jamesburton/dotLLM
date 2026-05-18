# Attention Mechanisms — dotLLM

## Attention in dotLLM

Attention is called directly in each backend's forward pass (CPU and GPU have separate implementations optimized for their respective hardware). There is no shared `IAttentionMechanism` interface — this is intentional, as abstracting across backends would lose CPU-specific optimizations (fused ops, weight repacking) and GPU-specific optimizations (cuBLAS GEMM, PTX kernels).

## Grouped-Query Attention (GQA)

Single implementation covers three variants via `num_kv_heads`:

| Config | Variant | Models |
|--------|---------|--------|
| `kv_heads == attn_heads` | MHA | GPT-2, older models |
| `kv_heads == 1` | MQA | Falcon, PaLM |
| `1 < kv_heads < attn_heads` | GQA | Llama 2/3, Mistral, Qwen2 |

### Forward Pass

1. Project: Q = x @ W_q, K = x @ W_k, V = x @ W_v
2. Reshape to heads: Q[batch, num_heads, seq, head_dim], K/V[batch, kv_heads, seq, head_dim]
3. Apply position encoding (RoPE) to Q and K
4. Update KV-cache (append K, V)
5. GQA broadcast: expand KV heads by `group_size = num_heads / kv_heads`
6. Scores = (Q @ K.T) / sqrt(head_dim) + mask
7. Weights = softmax(scores)
8. Output = weights @ V → reshape → output @ W_o

### Sliding Window

Mask modifier, not separate mechanism. Limits attention to `[pos - window_size, pos]`. KV-cache evicts older entries. Configured via `ModelConfig.SlidingWindowSize`.

## Multi-head Latent Attention (MLA)

DeepSeek-V2/V3. Compresses KV into low-rank latent space.

1. Compress: `c_kv = x @ W_dkv` → latent_dim (e.g., 512 vs 4096)
2. Store `c_kv` in cache (not full K, V — 8-16× smaller)
3. Decompress at attention time: `K = c_kv @ W_uk`, `V = c_kv @ W_uv`
4. Separate RoPE handling for rope and non-rope dimensions
5. Standard attention computation

Requires its own attention implementation with `LatentKvCache`.

**CPU**: complete. Three phases coexist behind `MlaConfig` flags:
- `MlaAttention.Execute` — Phase A naive expanded (per-head K_nope/V cache); the numerical oracle.
- `MlaAttention.ExecuteLatent` — Phase B absorbed-form attention over latent `[c_kv, k_pe]` cache (`MlaLatentKvState`).
- `MlaAttention.ExecuteLatentHybrid` — Phase C vLLM-style: prefill expands + MHA, decode runs absorbed.
- All three handle low-rank Q (`q_a_proj` + `q_a_layernorm` + `q_b_proj`), low-rank KV with `kv_a_layernorm`, RoPE on the rope-only sub-dim, causal mask, and YaRN's `mscale²` softmax-scale multiplier (`MlaConfig.ComputeYarnSoftmaxScaleMultiplier`).
- Verified end-to-end on tiny-random DeepSeek-V2/V3 fixtures and (gated by checkpoint availability) DeepSeek-V2-Lite real weights.

**CUDA**: Phase A primitives landed (`CudaMlaAttention.Forward`, `attention_mla_f32` kernel, `mla_helpers.cu`, `CudaMlaWeights`, `CudaMlaKvCache`). F32 throughout for now, validated against the CPU oracle within FP16 noise. **Not yet wired into `CudaTransformerModel.Forward`** — that wiring blocks on the CUDA MoE FFN port (DeepSeek-V2/V3 layers are MLA + MoE FFN, and the FFN GPU path doesn't exist). Phase B/C and FP16/quantized weight paths are deferred follow-ups.

## Vulkan Flash Attention (prefill GQA path)

Vulkan attention has two F32 paths sharing one descriptor surface:

| Kernel | Shader | Workgroup unit | Use |
|--------|--------|----------------|-----|
| `AttentionF32Kernel` | `attention_f32.comp` (+ `_sg`, `_coopmat`) | one (query token, head) | Decode (seq_q = 1); fallback for FA-ineligible shapes |
| `VulkanFlashAttentionF32Kernel` | `attention_flash_f32.comp` | one (head, query-tile of BR=16 rows) | Prefill (seq_q > 1), head_dim ≤ 128 |

The FA shader is Flash-Attention-v2 style: each workgroup holds BR=16 Q-rows in shared memory and walks the KV stream in BC=64 column tiles. Each KV row is read **once per Q-tile** (= BR× amortisation vs the per-token shader). Online softmax is maintained per Q-row (per-row running max + sum_exp), with one workgroup-wide tree reduce per (Q-row, KV-tile) pair using subgroupMax / subgroupAdd + cross-subgroup shared-memory combine — portable across subgroup widths.

Dispatch decision (`VulkanTransformerModel.RecordAttention` and analogous sites in `VulkanNemotronHTransformerModel` / `VulkanQwen3MoeHybridTransformerModel`):

```
if (_flashAttention != null && seqQ > 1 && headDim <= 128) -> FA
else                                                        -> naive per-token
```

Env-var opt-out: `DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1` forces every dispatch onto the legacy per-token kernel. The FA path is null when the SPV is missing (older builds) or when head_dim exceeds the shader bound — both gates fall back automatically.

Tile sizing rationale (Strix Halo / RDNA3.5, 64-wide wavefronts, 64 KB LDS):
- WG_SIZE = BC = 64: one wavefront per workgroup, reductions in a single subgroup step.
- BR = 16, MAX_HEAD_DIM = 128: qTile 8 KB + outAccum 8 KB + scoreMatrix 4 KB ≈ 20 KB. Headroom for raising BR or MAX_HEAD_DIM later.
- Soft-cap (Gemma 2 / Qwen3 style): optional push constant; raw scores pass through `softCap * tanh(s / softCap)` before softmax when non-zero.

Per-shape kernel microbench (dev-laptop iGPU, GQA-4, 32/8 heads, head_dim 64): **FA is 2.46-2.49× faster than the naive per-token shader at pp512 / pp2048**, matching the BR-amortisation prediction; see `benchmarks/DotLLM.Benchmarks/VulkanFlashAttentionBenchmarks.cs` and `tests/DotLLM.Tests.Unit/Vulkan/VulkanFlashAttentionF32KernelTests.cs` for parity coverage (MHA / GQA-4 / GQA-8, prompt 128 / 512 / 2048, sliding window, soft-cap, ALiBi).

MLA-attention (DeepSeek-V2/V3) keeps `AttentionMlaF32Kernel`; the FA path is GQA-only by design — an MLA FA variant is a separate workstream.

Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S iGPU) measurement at pp512/pp2048/pp4096 with 32/8 heads, head_dim 64: FA speedup is 1.35× / 2.06× / 2.72× — scales with sequence length per the BR=BC K-amortisation prediction. See `docs/PERFORMANCE.md` §6.4.

### Tuning headroom (deferred — workable WG=64 + BR=16 baseline shipped)

Things worth probing on Strix Halo before tile sizes lock in:

- **BR × BC tradeoff.** Current BR=16, BC=64. Larger BR amortises K reads more but bloats `outAccum` and `scoreMatrix` in LDS; current footprint is ~20 KB out of 64 KB, so BR=32 is room available (scoreMatrix grows to 8 KB, outAccum to 16 KB). Worth measuring whether BR=32 / BC=64 beats BR=16 / BC=64 on Strix Halo's specific occupancy / register-pressure profile.
- **Streamed vs shared K/V.** Current shader streams K and V directly from global in the inner loops — relies on L1/L2 cache locality across the 64 threads (all hit the same tile rows). On RDNA3.5 that should be fine. Loading the K-tile into LDS once per WG would trade LDS for global reads; might help on devices with weaker L1.
- **Per-row reduction granularity.** When `subgroupSize == WG_SIZE` (Strix Halo case), the `wgRowMax` / `wgRowSum` cross-subgroup combine is dead code — one `subgroupMax` / `subgroupAdd` is the whole reduce. Could specialise via a constant or a second SPV variant; not a meaningful speed win at the current per-tile barrier count, but cleaner.
- **Q-tile transpose for memory coalescing.** Currently `qTile[r * head_dim + d]` is row-major; threads access `qTile[r][d]` strided by `d` across rows during the score loop, which is irregular. A `[d][r]` layout might give better LDS bank-conflict behaviour. Trace it before changing.
- **Soft-cap fold into `scale`.** When `softCap > 0` and the raw score is far from saturation, the `tanh` is wasted work; could skip when score magnitude < softCap × 0.5. Marginal.

## IAttentionStrategy — Kernel Selection

```
IAttentionStrategy:
  ComputeAttention(Q, K, V, mask, scale) → output
  SupportsPagedKvCache → bool
  RequiredComputeCapability → int?
```

| Strategy | Memory | When |
|----------|--------|------|
| **Naive** | O(N²) | Reference, fallback, short sequences |
| **Flash Attention 2** | O(N) | GPU SM80+ (Ampere). Tiled in SRAM, online softmax. 2-7× speedup |
| **Flash Attention 3** | O(N) | GPU SM90+ (Hopper). Async TMA, FP8 |
| **Vulkan FA F32**     | O(N) | Vulkan GQA prefill. BR=16 Q-tile × BC=64 KV-tile, online softmax. 2.4-2.5× over naive at pp512/pp2048 on dev iGPU |
| **CPU Tiled** | O(N) | CPU. Tiles fit L2 cache |
| **Paged Flash** | O(N) | Flash + non-contiguous KV blocks (PagedAttention) |

Backend advertises capabilities; engine selects best strategy.
