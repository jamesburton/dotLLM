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

### Softmax exp precision (#501)

Step 7's exponential is **precise on every backend by default**. CPU uses
`TensorPrimitives.Exp` (through `FastMath.ExpSumAndStore`, which keeps the fused
shift+exp+store+sum structure and only changes the exp); CUDA's `attention_f32.cu` and
`attention_flash_mma_decode_gqa_split.cu` use `expf`; Vulkan shaders use GLSL `exp()`.
(`attention_flash_mma.cu` uses `__expf`, the ~2-ULP hardware intrinsic — a different and
accepted tradeoff.)

Until #501 the CPU path and its two CUDA twins used the Schraudolph 1999 bit trick (~1-2% max
relative error), on the premise that "errors in exp get normalized away when dividing by the
sum". That is true only of the *mean* of the attention weights: the error is relative and
per-element, so it survives normalization as a reweighting of the mixture each head computes,
and the cost scales with how close a model's attention scores already sit to each other.
Measured on Llama-3.2-1B-pure / wikitext-2 / ctx 512 / 40 chunks, paired per chunk:
**Q8_0 unmoved** (+0.0004 ± 0.0006 nats, t = +0.7) but **Q3_K −1.71% perplexity**
(−0.01724 ± 0.00185 nats, t = −9.3). Throughput was unchanged at ctx 512 — the accurate path
is extra vectorized passes over an attention tile sized to stay in L1.

> **The −1.71% is not a quality figure for any model anyone ships** (audit #527). Both arms are
> dotLLM, paired on the same tokens, so this is not a BOS-misalignment casualty — the *significance*
> (t = −9.3) is real. But the only model it was measured on is `Llama-3.2-1B-pure` **Q3_K**, a pure
> quant-ladder fixture, and a degraded model amplifies a fixed difference by one to two orders of
> magnitude (measured: the same engine delta reads +0.029% / +0.359% / +2.319% on Q8_0- / Q3_K- /
> Q2_K-derived weights — [PERPLEXITY.md](PERPLEXITY.md#how-to-measure-quality-against-llamacpp-then)).
> The nearest thing to a shipping-grade row here is the Q8_0 control, and it is **null**. No
> Q4_K_M / Q5_K_M / Q6_K row is recorded anywhere, so as far as the record goes the cost of the
> approximation on a shipping quant is unmeasured, not small.
>
> **The decision to default the approximation OFF still stands** — it rests on "no measured
> throughput benefit", which no amount of amplification touches. What must not be repeated is the
> quotation of 1.71% as the quality cost. Settling that needs a re-measurement on a shipping-grade
> quant with #516 in place; it was deliberately not run as part of this audit.

`DOTLLM_FAST_EXP=1` restores the bit trick as a benchmarking lever. **It is CPU-only**: CUDA
ships precompiled PTX with no equivalent switch, so setting it makes the CPU and CUDA backends
diverge by roughly the approximation's own error (~1%, ~5e-3 abs on attention output). That is
useful as a deliberate discriminator, but do not run cross-backend parity with it set. Several
CPU↔CPU and CPU↔CUDA attention tolerances were calibrated against the approximation and are now
looser than they need to be; each says so in place.

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
| `VulkanFlashAttentionF32Kernel` (wide) | `attention_flash_f32_hd256_br{4,8,16}.comp` | one (head, query-tile of BR rows) | Prefill (seq_q > 1), 128 < head_dim ≤ 256 |

The FA shader is Flash-Attention-v2 style: each workgroup holds BR=16 Q-rows in shared memory and walks the KV stream in BC=64 column tiles. Each KV row is read **once per Q-tile** (= BR× amortisation vs the per-token shader). Online softmax is maintained per Q-row (per-row running max + sum_exp), with one workgroup-wide tree reduce per (Q-row, KV-tile) pair using subgroupMax / subgroupAdd + cross-subgroup shared-memory combine — portable across subgroup widths.

Dispatch decision (`VulkanTransformerModel.RecordAttention` and analogous sites in `VulkanNemotronHTransformerModel` / `VulkanQwen3MoeHybridTransformerModel`):

```
if (_flashAttention != null && seqQ > 1 && headDim <= _flashAttention.SupportedMaxHeadDim) -> FA
else                                                                                      -> naive per-token
```

Gate on the **instance** property `SupportedMaxHeadDim`, never on the `MaxHeadDim` constant (128, the base shader's bound). A gate that reads the constant sends a 256-dim head to the per-token kernel even when the wide pipeline is loaded — that is issue #441 reintroduced.

Env-var opt-out: `DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1` forces every dispatch onto the legacy per-token kernel. The FA path is null when the SPV is missing (older builds) or when head_dim exceeds every loaded shader bound — both gates fall back automatically.

### Wide heads (head_dim > 128) and the silent-fallback diagnostic — issue #441

`attention_flash_f32.comp` bakes `MAX_HEAD_DIM = 128` into its `qTile` / `outAccum` shared-memory **declarations**, so it is a hard dispatch gate, not a slow path. Bonsai 2 (`qwen35`) declares `attention.key_length = value_length = 256`, so all 16 of its full-attention layers ran the per-token kernel on every prefill — **~20 % of the whole pass**, with nothing warning about it. It took a per-op profiling campaign to notice, which is the real defect.

Two things fix that:

1. **Wide SPV variants.** `attention_flash_f32_hd256_br{16,8,4}.comp` are byte-identical to the base shader apart from `MAX_HEAD_DIM` and `BR`; the compute loops already bound on `pc.headDim`. `FlashAttentionWideVariant` selects one, overridable with `DOTLLM_VK_FLASH_HD256=br4|br8|br16|off` (default `br4`).
2. **`VulkanAttentionFallbackDiagnostics`.** The single factory every Vulkan model uses to obtain its prefill FA kernel. It applies all the gates and prints a one-shot stderr warning naming the model, the head dim, the bound that rejected it and the remedy. Silenceable with `DOTLLM_VULKAN_QUIET_ATTENTION_FALLBACK=1`; the reported set stays queryable for tests.

**BR is the opposite of what amortisation predicts at this width.** Same-process, order-reversed, interleaved A/B on gfx1151 at Bonsai 2's real shape (24/4 heads, head_dim 256), min-ms over 5 rounds:

| seq | naive (per-token) | br16 | br8 | br4 |
|-----|-------------------|------|-----|-----|
| 512 | 45.33 ms | 10.72 ms (4.2×) | 4.59 ms (9.9×) | 2.95 ms (15.3×) |
| 2048 | 927.6 ms | 170.1 ms (5.5×) | 88.0 ms (10.5×) | 61.4 ms (15.1×) |

> **KV residency is part of the shape.** The harness re-touches K/V before each dispatch, outside the timed region, because the model's KV-cache update writes them in the dispatch immediately before attention. The first version did not, and its naive arm then read 84.11 ms at seq 512 against the model's ~38 ms per layer — while the flash arms agreed with the model all along. Only the per-token kernel is residency-sensitive: it re-reads each KV row 12,288 times (one workgroup per token × head) against flash's ~128. The effect is also size-gated — at seq 2048 the KV set is ~16 MB, too large to stay resident either way, and pre-touching moves the naive arm by only 5 %. With it, kernel bench and end-to-end profile agree (≈15× vs the bucket's 12–19×).

Per-arm ranges are disjoint at both lengths. A *smaller* tile reads each KV row *more* times, so KV amortisation is not the binding constraint here — LDS residency is: `qTile + outAccum` both scale with `BR × MAX_HEAD_DIM`, so at 256 dims BR=16 costs 36.2 KB and pins one workgroup (4 wave64) per CU, BR=8 costs 18.1 KB, BR=4 costs 9.2 KB (~6 workgroups / 24 waves of latency hiding). BR=4 is the **floor for this geometry**, not a measured optimum: `ROWS_PER_SLICE = BR / (WG_SIZE / BC) = BR / 4`, so BR=2 would leave a wave slice zero rows. Going lower needs a narrower workgroup — a separate change.

End-to-end pp512 on Bonsai 2 PQ2_0 (separate process launches, both orders, GDN scan pinned to `ldsfused` — #445 landed the variant but did **not** make it the default on this branch: unpinned, `gdn_scan_core` is 1394–1479 ms against 182–188 ms pinned, which puts attention at ~17–18.5 % of the pass instead of ~21 %): `attn_core` **604.1–810.9 ms → 42.2–51.0 ms** (12–19×) across 3 launches per arm; whole-pass **148.0–166.8 → 200.7–205.2 tok/s** (~1.25×). Both ranges disjoint, and the cleanest pair ran br4 FIRST — the unfavourable slot if GPU clock ramp were doing the work.

Wave-width safety: these shaders have no subgroup ops. `slice = tid >> 6` and `c = tid & 63` index the `BC = 64` KV-column tile — tile geometry, not hardware wave width — so they are correct at subgroupSize 32 and 64 alike and need no native-subgroup-size gate (unlike `PQ2_0GemmVariant.RequiresNativeSubgroupSize`).

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
