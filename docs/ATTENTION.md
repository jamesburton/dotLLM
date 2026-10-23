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

**Measured (#531, 2026-09-24).** That re-measurement was run: `dev`, CPU, wikitext-2 LF, ctx 512,
**64 windows**, plain `--corpus` (BOS-aligned by default since #516), paired per window, 63 df,
**fast minus accurate** so positive = the approximation costs perplexity. Full working:
`.docs/measurements/2026-09-24-531-fastexp-shipping-grade.md`.

| model | fast-exp | accurate | ΔNLL (nats) | PPL % | t | 95% CI on PPL % |
|---|---|---|---|---|---|---|
| **Q4_K_M** (shipping-grade) | 15.7620 | 15.7669 | −0.00031 ± 0.00066 | −0.031% | **−0.47** | **[−0.163%, +0.101%]** |
| Q4_K_M decoded to F32 (control) | 15.7378 | 15.7359 | +0.00012 ± 0.00037 | +0.012% | +0.32 | [−0.063%, +0.086%] |
| Q3_K (pure ladder) | 23.8620 | 23.4613 | +0.01693 ± 0.00178 | +1.708% | +9.50 | [+1.346%, +2.071%] |
| Q3_K decoded to F32 (control) | 23.6171 | 23.2145 | +0.01719 ± 0.00136 | +1.734% | +12.61 | [+1.457%, +2.012%] |

Two results, and the second is the one that changes how the paragraph above should be read.

1. **On a shipping-grade quant the cost is not resolvable at n = 64.** Q4_K_M and its F32 control
   are both consistent with zero and with each other; the honest statement is a bound, ≲0.16%, not
   a size. (Q4_K_M here is `llama-quantize --allow-requantize … Q4_K_M` of the healthy
   Q8_0-derived F32 model, so the row is on the same lineage as the amplification table.)
2. **The −1.71% on Q3_K reproduces (+1.708%) — and its F32-decoded control moves by the same
   +1.734%.** The control has no quantized matmul in it at all. So the penalty is **not** a
   property of the Q3_K path, and "the cost scales with how close a model's attention scores sit"
   is right while any reading of it as *a 3-bit format's* cost is wrong. It is a property of
   weights degraded to PPL ≈ 23, by whatever means.

**What was not wrong: the alignment.** #531 also settled the contradiction between the 40-chunk
figures quoted here and PERPLEXITY.md's — see
[PERPLEXITY.md](PERPLEXITY.md#501s-40-chunk-q3_k-rows-are-the-f32-decoded-path-531). The 2026-09-23
run did pass `--bos`; what it did not run is today's packed Q3_K × Q8_K kernel. Its numbers
reproduce digit-for-digit on the **`Q3_K-decoded-F32`** fixture (22.2692 / 21.8886, window 0
11.763354) and on neither packed row (22.5090 / 22.1019). The "+2.03% → +0.28% against llama.cpp"
line that accompanied them is therefore F32-decoded dotLLM against packed-Q3_K llama.cpp; the
like-for-like packed gap is +1.03% (64 chunks) / +1.325% (564).

`DOTLLM_FAST_EXP=1` restores the bit trick as a benchmarking lever. **It is CPU-only**: CUDA
ships precompiled PTX with no equivalent switch, so setting it makes the CPU and CUDA backends
diverge by roughly the approximation's own error (~1%, ~5e-3 abs on attention output). That is
useful as a deliberate discriminator, but do not run cross-backend parity with it set. Several
CPU↔CPU and CPU↔CUDA attention tolerances were calibrated against the approximation and are now
looser than they need to be; each says so in place.

### KV-cache-length invariance (#525) — CPU

A query position's attention output depends **only on the keys that position can see**. The CPU
kernel computes, softmaxes and accumulates over `[visibleStart, visibleEnd)` — bounds derived from
the query's own position, the mask mode and the sliding window — and never touches the rest of the
KV cache. The dense/tiled choice is made **per row on that visible length**, not per call on
`seqQ * seqKv`.

This is a correctness property, not a tidiness one. The previous implementation materialised the
full padded `seqKv` row, wrote `-inf` into the invisible entries, and reduced the *whole* row.
`-inf` padding contributes exactly `0.0` to the sum, so the result is mathematically identical —
but changing the span length changes SIMD lane assignment and the remainder tail, so the real
elements accumulate in a different order and the last ULP of the softmax denominator moves with the
cache length. Measured directly on the kernel: a 3-row causal chunk changed in 757 of 1728 outputs
when the declared cache grew, worst `1.19E-07`; the dense→tiled dispatch flip at `seqKv` 682→683
changed 855 of 1728, worst `1.27E-07`. Shorter visible prefixes are worse (98% of rows at 3 visible
keys vs 8% at 17).

On a quantized model that ULP is not absorbed. On SmolLM-135M Q8_0 it digitised at layer 21 into
exactly **one int8 value of 576, by one step**, which `o_proj` amplified ~14,600× and which by layer
29 changed the emitted token — so the same prompt produced different text depending only on how
prefill happened to be batched (chunked prefill, and by extension continuous batching, prefix-cache
hits and speculative verify widths). The F32-decoded control carried the same seed and never
crossed a threshold, which is why this was invisible outside quantized runs.

Guarded by `AttentionKvLengthInvarianceTests` (bit-exact, kernel level, incl. the head-parallel
worker, sliding window, soft-cap, ALiBi and sinks) and `ChunkedPrefillInvarianceTests` (logits, not
sampled tokens — argmax insensitivity is what hid this — swept over prompt length × chunk size on
Q8_0 with an F32-decoded sensitivity control).

Reduction order was not the whole mechanism. `FastMath.ExpSumAndStore` clamps its shifted input at
`MinClamp = -87.3` in **both** the precise and the approximate exp paths, so a masked `-inf` entry
did not become exactly `0` — it became `exp(-87.3) ≈ 1.3e-38`. Every padding column therefore added
a term to the softmax denominator, and the number of those terms was the KV-cache length. It also
makes the `w == 0f` short-circuit comment in `WeightedValues` ("exp(-inf) is exactly 0f") wrong on
the padded path. The visible-prefix form removes both: invisible keys are never scored, so nothing
is clamped and nothing is accumulated.

The work saved is confined to the dense path: prompts short enough to take it (`seqQ * seqKv * 4 <=
8192`, so ≈45 tokens single-pass, or any chunk of a chunked prefill) now score a triangle instead of
a square. The tiled path already walked only the visible range, so long-prompt prefill and decode
are neutral. Treat the change as correctness-motivated, not a perf win.

> The `#501` fast-exp figures above were measured against the padded reduction. They compare
> fast-exp on vs off, and both arms shifted by the same ULPs, so the comparison stands; the absolute
> attention outputs on either arm have changed by ~1e-7. A re-measurement (#531) will run against
> the new reduction.

Decode numerics are unchanged **without** a sliding window: a decode row's visible length equals the
declared `seqKv`, and the one-shot threshold is the same 8 KiB budget in floats. With a sliding
window, decode previously softmaxed over `seqKv` with the out-of-window entries masked and now
softmaxes over `window` entries — a genuine (ULP-scale) change on Gemma-2/3 and Mistral-style models.

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

**Vulkan (absorbed, #913)**: default when every layer ships the llama.cpp split `attn_k_b`/`attn_v_b` as Q8_0 with `kv_lora_rank` 512 and `qk_rope` 64 (GLM-4.7-Flash, DeepSeek-V2/V3 GGUFs from current converters). Projections (q_a, q_b, kv_a_mqa, attn_output) stay in their packed GGUF quantization and use the dense MMVQ/coopmat router; the GPU-only load skips the host F32 copies. The cache holds one `[rmsnorm(c_kv) | rope(k_pe)]` row (576 floats) per token per layer (`MlaVulkanKvCache` latent mode, 15.7x smaller than the expanded layout). Kernels: `mla_kv_prep_f32` (norm + rope + cache write), `mla_head_gemm_q8` (per-head W_UK query absorption with the q_pe rope pack, and the W_UV value up-projection), `attention_mla_absorbed_f32` (8 rows x 64 keys tile, split-KV for decode/tiny chunks) + `attention_mla_absorbed_merge_f32`. `DOTLLM_VK_MLA_LEGACY=1` forces the pre-#913 expanded F32 path (control arm; also the fallback for fused `attn_kv_b` files and LoRA on `kv_b_proj`).

**CUDA**: Phase A primitives landed (`CudaMlaAttention.Forward`, `attention_mla_f32` kernel, `mla_helpers.cu`, `CudaMlaWeights`, `CudaMlaKvCache`). F32 throughout for now, validated against the CPU oracle within FP16 noise. **Not yet wired into `CudaTransformerModel.Forward`** — that wiring blocks on the CUDA MoE FFN port (DeepSeek-V2/V3 layers are MLA + MoE FFN, and the FFN GPU path doesn't exist). Phase B/C and FP16/quantized weight paths are deferred follow-ups.

## Vulkan Flash Attention (prefill GQA path)

Vulkan attention has two F32 paths sharing one descriptor surface:

| Kernel | Shader | Workgroup unit | Use |
|--------|--------|----------------|-----|
| `AttentionF32Kernel` | `attention_f32.comp` (+ `_sg`, `_coopmat`) | one (query token, head) | Decode (seq_q = 1); fallback for FA-ineligible shapes |
| `VulkanFlashAttentionF32Kernel` | `attention_flash_f32.comp` | one (head, query-tile of BR=16 rows) | Prefill (seq_q > 1), head_dim ≤ 128 |
| `VulkanFlashAttentionF32Kernel` (wide) | `attention_flash_f32_hd256_br{4,8,16}.comp` | one (head, query-tile of BR rows) | Prefill (seq_q > 1), 128 < head_dim ≤ 256 |
| `VulkanFlashAttentionGqaKernel` | `attention_flash_gqa_cm.comp` (coopmat) / `attention_flash_gqa.comp` (scalar), `_kvf16` twins | one (KV head, 64 / 16 rows across its query heads) | Dense-hybrid (Bonsai 2) prefill, 128 < head_dim ≤ 256 — see "GQA-grouped tiled prefill" below (#441) |

The FA shader is Flash-Attention-v2 style: each workgroup holds BR=16 Q-rows in shared memory and walks the KV stream in BC=64 column tiles. Each KV row is read **once per Q-tile** (= BR× amortisation vs the per-token shader). Online softmax is maintained per Q-row (per-row running max + sum_exp), with one workgroup-wide tree reduce per (Q-row, KV-tile) pair using subgroupMax / subgroupAdd + cross-subgroup shared-memory combine — portable across subgroup widths.

Dispatch decision (`VulkanTransformerModel.RecordAttention` and analogous sites in `VulkanNemotronHTransformerModel` / `VulkanQwen3MoeHybridTransformerModel`):

```
if (_flashAttention != null && seqQ > 1 && headDim <= _flashAttention.SupportedMaxHeadDim) -> FA
else                                                                                      -> naive per-token
```

Gate on the **instance** property `SupportedMaxHeadDim`, never on the `MaxHeadDim` constant (128, the base shader's bound). A gate that reads the constant sends a 256-dim head to the per-token kernel even when the wide pipeline is loaded — that is issue #441 reintroduced.

Env-var opt-out: `DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1` forces every dispatch onto the legacy per-token kernel. The FA path is null when the SPV is missing (older builds) or when head_dim exceeds every loaded shader bound — both gates fall back automatically.

### Coopmat prefill numerics — what is f16, and the position-dependent bound (#543)

`attention_flash_f32_coopmat{,_hd64}.comp` stages its operands as f16 for the matrix cores.
That makes **multi-token prefill less accurate than a 1-token chunk**, which takes the dense
f32 path — so incremental and single-pass prefill of the same tokens wrote different KV
(#543, the same reachability class as #533: prefix-cache reuse and multi-turn continuation).

Two things worth knowing before touching this:

- **The P·V accumulator was always f32** (`coopmat<float, ...>`, stricter than llama.cpp cm1's
  f16 default). The error is from f16 **inputs**, so "accumulate in f32" is not an available
  fix — it is already done.
- **There is no f32 cooperative-matrix type on gfx1151.** The driver reports 11 tile shapes,
  all 16x16x16, every one f16 or 8-bit integer
  (`VulkanSubgroupProbeTests` prints the table). An f32-input coopmat attention is unavailable
  at any price on this hardware; do not plan around one.

What #543 changed: with #533's gate on (every non-NVIDIA device) the KV-length-**dependent**
tiles already bypass `coopMatMulAdd` and take a scalar tail, so their accuracy was set by
staging precision. That tail now reads full f32 — P from `sTile` (written back over the score
it came from, so no extra LDS), V staged f32 through `oStage` (idle on exactly those tiles),
and the mask pass writes 0.0 into `sTile`'s dead padding rows so the tail is branch-free.
It is both more accurate **and faster** (+6.5% p512, +4% hd128, neutral p2048).

**The bound is position-dependent and that matters more than the headline number.** A query row
at position p spans `ceil((p+1)/64)` KV tiles of which at most ~2 are scalar-tail tiles, so
the gain decays; maxAbs against the scalar f32 FA kernel, per 64-row block of a p512 prefill:

| rows | 0+ | 64+ | 128+ | 256+ | 448+ |
|---|---|---|---|---|---|
| before #543 | 3.296E-04 | 8.597E-05 | 6.918E-05 | 4.536E-05 | 3.520E-05 |
| after | 1.446E-04 | 8.469E-05 | 6.208E-05 | 4.402E-05 | 3.863E-05 |

At rows 0-63 (where every tile is a tail tile) that is 2.28x and the #543 acceptance
measurement — the per-layer KV bisect of a 1-token first chunk against a single-pass prefill,
F32-decoded Llama-3.2-1B — goes to **bit-identical across all 16 layers**. Past row ~256 it is
level: **QK^T is still f16-in on every tile and is the whole residual.**

Raising the coopmat tiles too (split-f16 V: `V_hi`, `V_lo = f16(v - V_hi)`, two `coopMatMulAdd`s)
was built and measured: 1.22-1.47x on the rows above for 2.7%, but only where `V_lo`'s storage
is free — `headDim <= 64` in the 128-dim shader, i.e. `seqKv < 640` before the #378 gate takes
over. Both routes out of that window cost 15-24%; see the hd64 shader's occupancy header for
the LDS reason and `.docs/ISSUE_543_MEASUREMENTS.md` for every arm.

**Gates.** `Probe533FixCandidateTests` holds an *absolute* accuracy bound (2.0E-04 at
seqQ=seqKv=64) plus `attention_flash_f32_coopmat_pre543`, a retained pre-fix shader that must
**exceed** it — a bound nothing in the tree can violate is not a gate. It also prints the
per-64-row-block decay, so the coverage ceiling stays measured rather than argued.

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

### GQA-grouped tiled prefill for head_dim 256 — issue #441 (2026-10)

**Why the wide variants above were not enough.** At long context the prefill bottleneck on Strix was attention, not the
projections: the #1100 growth run logged `[vk-prefill] … dense=11–15 TF/s attn=~1.0 TF/s`, and 128k-deep 8k-token turns took
395 s. On a coopmat device Bonsai 2 does reach a flash kernel (`attention_flash_f32_coopmat_hd256`, not the naive one: the
issue's original "silent naive fallback" only applies to non-coopmat devices without the wide variants). But that kernel is
slow by construction:

- one workgroup = **one query head** × 16 rows, so each K/V row is fetched once per query head (24× for Bonsai's 24 / 4 heads)
  and amortised over only 16 query rows;
- ~19 workgroup barriers per 32-column KV tile (K and V staged in four 64-wide d-chunks each);
- the online softmax is 16 threads scanning rows serially while 240 wait, and only 2 of 4 subgroups run QK^T.

Measured (`VulkanFlashAttentionGqaDepthBench`, one layer, Bonsai shape 24 / 4 / 256, F16 KV, rows appended at KV depth d):
**1.10–1.29 TFLOP/s at every depth** for the old coopmat kernel, 0.45–0.66 for the old scalar br4. It is a kernel-design
limit, not occupancy at 32 KiB LDS or the dispatch shape.

**`VulkanFlashAttentionGqaKernel`** (`attention_flash_gqa_cm{,_kvf16}.comp`, coopmat; `attention_flash_gqa{,_kvf16}.comp`,
scalar; bodies in the matching `.glsl`):

- **Row space per KV head.** The query side of KV head `hkv` is the matrix of its `G = numHeads / numKvHeads` query heads'
  rows: row `m` ↔ (token `m / G`, head `hkv·G + m % G`). One K/V tile load serves every head of the group. Grid =
  `(ceil(seqQ·G / rowsPerWG), numKvHeads)`, launched last-rows-first (they have the longest causal range).
- **Coopmat (Strix, NVIDIA):** 64 rows per workgroup, one 16-row strip per subgroup (subgroup size pinned; workgroup size =
  4 × subgroup via specialization constant 1). Flash-Attention-2 ownership: Q (hi and lo F16 halves) and O live in registers as
  cooperative matrices; the strip's lanes run its online softmax with shuffle reductions; the per-row O rescale is a
  component-wise multiply by a 16×16 matrix whose row r is `corr[r]`, skipped when every correction is exactly 1. Four
  workgroup barriers per 32-column tile, all around the shared K/V stage; the next tile's global fetch is held in registers
  while the current one is multiplied. LDS 30.5 KiB (F16 KV), under Strix's 32 KiB.
- **Scalar (Arc):** 16 rows per workgroup, 8 lanes per row each holding 32 of the row's Q and O dims in registers; QK^T
  partials are summed with a 3-step shuffle-xor butterfly (every lane of the row ends with the bit-identical score); K/V tiles
  are staged as F32 in LDS (converted once per element, not per reading lane). Two barriers per tile, 32 KiB LDS.
- **Numerics.** F32 accumulation for QK^T and P·V, F32 running max / sum / correction (the precision rules). Coopmat P is
  F16 and the denominator sums the same F16-rounded P. QK^T keeps the hd256 kernel's Q hi/lo split (specialization constant 2,
  `DOTLLM_VULKAN_FA_GQA_QLO=0` drops it — see the A/B below). The #533 rule carries over: a tile that is partial or holds a
  masked cell for any row of the workgroup takes a `precise` scalar P·V on AMD.
- **KV read in place.** F16 rows are read as packed words (`uvec2`, no 16-bit storage-buffer feature) and converted in the tile
  load; F32 rows split K and V into hi/lo halves so the F32 baseline is not rounded to F16. A q8_0 cache is one more tile-load
  body. Nothing widens the history.
- **Routing.** `VulkanQwen3MoeHybridKernels.EnsureGqaFlash(storage, headDim)` builds it per KV storage for 128 < head_dim ≤ 256;
  the dense-hybrid host dispatches it ahead of the per-head flash kernels (profiler notes `attn_flash_gqa_cm` /
  `attn_flash_gqa`; the per-head coopmat branch now notes `attn_flash_cm`). `DOTLLM_VULKAN_FA_GQA=0` restores the previous
  path, `DOTLLM_VULKAN_FA_GQA_SCALAR=1` forces the scalar implementation. A prefill at KV ≥ 1024 that still reaches the
  per-token kernel reports once per host and head width (`VulkanAttentionFallbackDiagnostics.ReportNaivePrefill`).
- **TDR.** A dispatch is one layer's attention for the forward's rows; #1100's submit budget (fitted per device, ~1 s) sizes the
  rows at depth, and the faster kernel simply lets it pack more rows per submission. No KV-axis split was needed.

**Kernel A/B** (min of 3 interleaved order-reversed rounds, attention TFLOP/s = causal QK^T + P·V work; rows halve at depth
where an arm would exceed ~0.8 s):

| device | arm | 8k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|
| Strix (8060S) | old coopmat hd256 | 1.24 | 1.14 | 1.10 | 1.11 |
| Strix | **GQA coopmat** | **3.99** | **3.92** | **3.90** | **3.92** |
| Strix | GQA coopmat, Q-lo off (A/B only) | 6.32 | – | – | 6.06 |
| Strix | old scalar br4 | 0.60 | 0.51 | 0.50 | 0.68 |
| Strix | GQA scalar | 1.89 | 1.80 | 1.72 | 1.77 |
| Arc (Meteor Lake) | old scalar br4 | 0.06 | 0.06 | 0.06 | 0.06 |
| Arc | **GQA scalar** | **0.22** | 0.20 | 0.19 | **0.19** |

**Q hi/lo split A/B.** Perplexity on Strix, wikitext-2 (`dotllm perplexity --device vulkan`, cacheless F32 K/V):

| model | windows | GQA, split (default) | GQA, no split | old kernel |
|---|---|---:|---:|---:|
| Bonsai 2 27B PQ2_0 | 10 × 4096 | 8.3260 (NLL 2.119382) | 8.3259 (2.119376) | 8.3260 (2.119378) |
| Tev1-4B IQ4_NL | 20 × 1024 | 9.4546 (2.246506) | 9.4546 (2.246500) | 9.4546 (2.246498) |

Perplexity cannot see the split on either hd256 model (the Tev1 NLL 2.272 → 3.509 that motivated it does not reproduce on this
kernel). But without it the kernel misses the coopmat tolerance on large scores (`Prefill_LargeScores_MatchesCpuReference`:
max abs 1.3e-2 against 1e-3), so it **stays on by default**; `DOTLLM_VULKAN_FA_GQA_QLO=0` (1.6× attention) is an opt-in pending
the user's call. (Tev1 at 4096-token windows lost the device on every arm, the old kernel included: a cacheless 4096-row
forward is not row-sliced by the submit budget.)

TFLOP/s is still a third of the dense projections; the remaining limit is the QK^T MMA count (the split doubles it) and the
B-fragment reload per MMA pair. Next levers: retire the split where measured safe, and a 128-row workgroup once LDS allows.

**End-to-end** (Strix, Bonsai 2 PQ2_0, F16 KV, `--no-mtp`, GDN scan default variant):

Session growth (`scripts/session-growth.py`, 16 turns of ~8k tokens to 128k, 63 generated tokens per turn). The old-kernel arm
is the same build with `DOTLLM_VULKAN_FA_GQA=0`, so the KV storage (F16) and everything else are identical; the #1100 column is
that issue's published baseline (F32 KV), for reference:

| turn | context | #1100 baseline TTFT (F32 KV) | old kernel, F16 KV | **GQA kernel, F16 KV** | speed-up (same build) | prefill tok/s old → new | decode tok/s old → new |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,823 | 45.7 s | 44.1 s | **38.2 s** | 1.15× | 178 → 206 | 20.4 → 20.8 |
| 2 | 17,037 | 83.7 s | 82.1 s | **52.2 s** | 1.57× | 112 → 177 | 18.1 → 18.3 |
| 3 | 25,060 | 98.8 s | 97.9 s | **52.8 s** | 1.85× | 82 → 152 | 16.4 → 16.5 |
| 4 | 32,934 | 119.5 s | 118.7 s | **58.0 s** | 2.05× | 66 → 136 | 15.0 → 14.9 |
| 5 | 41,026 | 146.4 s | 145.6 s | **66.9 s** | 2.18× | 56 → 121 | 13.7 → 13.5 |
| 6 | 48,783 | 163.4 s | 160.7 s | **70.9 s** | 2.27× | 48 → 110 | 12.7 → 12.5 |
| 7 | 56,411 | 181.3 s | 178.1 s | **76.7 s** | 2.32× | 43 → 100 | 11.9 → 11.6 |
| 8 | 64,320 | 212.0 s | 208.1 s | **86.4 s** | 2.41× | 38 → 92 | 11.1 → 10.8 |
| 9 | 72,287 | 235.8 s | 230.9 s | **94.7 s** | 2.44× | 35 → 84 | 10.4 → 10.2 |
| 10 | 80,915 | 281.6 s | 274.8 s | **110.0 s** | 2.50× | 31 → 79 | 9.8 → 9.6 |
| 11 | 89,112 | 309.4 s | 301.4 s | **112.7 s** | 2.67× | 27 → 73 | 9.2 → 9.1 |
| 12 | 96,656 | 307.4 s | 298.6 s | **110.0 s** | 2.71× | 25 → 69 | 8.8 → 8.6 |
| 13 | 104,149 | 327.1 s | 317.2 s | **116.4 s** | 2.73× | 24 → 65 | 8.4 → 8.1 |
| 14 | 112,017 | 352.1 s | 346.3 s | **129.1 s** | 2.68× | 23 → 61 | 8.0 → 7.8 |
| 15 | 120,386 | 392.6 s | 378.5 s | **145.0 s** | 2.61× | 22 → 58 | 7.6 → 7.5 |
| 16 | 128,254 | 394.6 s | 381.7 s | **143.9 s** | 2.65× | 21 → 55 | 7.3 → 7.2 |
| all 16 | | 60.9 min | 59.4 min | **24.4 min** | 2.44× | | |

Fresh single prompts (server restarted for each; #1100's baseline, F32 KV, vs this branch, F16 KV):

| prompt | #1100 TTFT | GQA kernel TTFT | prefill tok/s |
|---:|---:|---:|---:|
| 17,003 | 127.6 s | **90.1 s** | 133 → 189 |
| 32,831 | 344.6 s | **206.1 s** | 95 → 159 |
| 64,077 | 1,039.9 s | **500.5 s** | 62 → 128 |

The fitted `[vk-prefill]` attention rate at 128k is 3.6 TFLOP/s (was ~1.0); dense stays 10–12. Decode is not this change
(split-KV); it tracks the old arm within noise. The Arc is dense-bound for Bonsai 2 (projections 0.10–0.14 TFLOP/s against
attention 0.25), so its end-to-end prefill barely moves; its kernel numbers are above.

Tests: `VulkanFlashAttentionGqaKernelTests` (both implementations × F32 / F16 KV vs `Attention.ExecuteScalar` at the family
tolerances: GQA 6 and 8 with several KV heads, MHA, position offsets, ragged tiles, head_dim 256 and 128, large scores),
`VulkanQwen3HybridDenseGqaFlashTests` (whole model, 12 / 2 heads at hd256, both storages, vs the pre-#441 path). Mutants (wrong
head mapping, missing rescale, causal off by one; each on both implementations) fail every case of their implementation.

### Tuning headroom (deferred — workable WG=64 + BR=16 baseline shipped)

Things worth probing on Strix Halo before tile sizes lock in:

- **BR × BC tradeoff.** Current BR=16, BC=64. Larger BR amortises K reads more but bloats `outAccum` and `scoreMatrix` in LDS; current footprint is ~20 KB out of 64 KB, so BR=32 is room available (scoreMatrix grows to 8 KB, outAccum to 16 KB). Worth measuring whether BR=32 / BC=64 beats BR=16 / BC=64 on Strix Halo's specific occupancy / register-pressure profile.
- **Streamed vs shared K/V.** Current shader streams K and V directly from global in the inner loops — relies on L1/L2 cache locality across the 64 threads (all hit the same tile rows). On RDNA3.5 that should be fine. Loading the K-tile into LDS once per WG would trade LDS for global reads; might help on devices with weaker L1.
- **Per-row reduction granularity.** When `subgroupSize == WG_SIZE` (Strix Halo case), the `wgRowMax` / `wgRowSum` cross-subgroup combine is dead code — one `subgroupMax` / `subgroupAdd` is the whole reduce. Could specialise via a constant or a second SPV variant; not a meaningful speed win at the current per-tile barrier count, but cleaner.
- **Q-tile transpose for memory coalescing.** Currently `qTile[r * head_dim + d]` is row-major; threads access `qTile[r][d]` strided by `d` across rows during the score loop, which is irregular. A `[d][r]` layout might give better LDS bank-conflict behaviour. Trace it before changing.
- **Soft-cap fold into `scale`.** When `softCap > 0` and the raw score is far from saturation, the `tanh` is wasted work; could skip when score magnitude < softCap × 0.5. Marginal.

### Narrow (F16) KV on Vulkan — issue #1102

The dense hybrid host (`VulkanQwen3HybridDenseTransformerModel`, Bonsai 2) can store its KV as F16 (see
[KV_CACHE.md](KV_CACHE.md), "Vulkan dense-hybrid (Bonsai 2) KV"). Every attention kernel it can dispatch reads those halves
**in place**; nothing converts the history to an F32 scratch.

- **One body, two pipelines.** Each KV-reading shader is a `.glsl` body plus thin `.comp` wrappers. The body includes
  `kv_load.glsl`, which declares `BufK` / `BufV` (bindings 1 / 2) and reads elements through `KV_K(i)` / `KV_V(i)`. The F32
  wrapper gets `float k[]` (its SPIR-V is byte-identical to the pre-#1102 shader). The `_kvf16` wrapper defines `KV_F16`
  and gets uint-backed halves, two per word, with `unpackHalf2x16` in the load. No 16-bit-storage device feature is needed.

  | path | F32 shader | F16 twin |
  |---|---|---|
  | decode (split-KV, seqKv ≥ 17) | `attention_f32_splitkv` | `attention_f32_splitkv_kvf16` |
  | prefill, coopmat (Strix) | `attention_flash_f32_coopmat`, `_hd256` | `attention_flash_f32_coopmat_kvf16`, `_kvf16_hd256` |
  | prefill, scalar flash (Arc) | `attention_flash_f32`, `_hd256_br{4,8,16}` | `attention_flash_f32_kvf16`, `_hd256_br{4,8,16}_kvf16` |
  | per-token (short decode, fallback) | `attention_f32`, `_sg` | `attention_f32_kvf16`, `_sg_kvf16` (no coopmat twin) |

  The kernel classes take a `VulkanKvStorage` and size-check K / V with its element bytes. `VulkanNarrowKvAttentionKernels`
  holds the F16 set plus the row writer and is built with the same gates as the F32 set (env opt-outs, head_dim bounds).
  The forward picks the set from the cache's `Storage`, so F32 and F16 caches can coexist.
- **Numerics.** `unpackHalf2x16` is exact, so an F16 kernel computes on exactly the values its F32 twin reads from a
  cache holding the F16-rounded rows. `VulkanF16KvAttentionKernelTests` asserts that equality **bit for bit**, for every
  family, on Bonsai's GQA 24 / 4 and 8 / 4 at head_dim 256 (and 64 / 128), with position offsets and ragged tiles. It also
  checks the CPU reference at each family's existing tolerance. In the coopmat hd256 kernel's hi/lo split, F16 K makes
  `K_lo` exactly zero.
- **Bytes per decoded token** (Bonsai 2, attention only): F32 reads 128 KiB per cached position, F16 reads 64 KiB.
  At 128k that is 16 GiB vs 8 GiB per decoded token.
- **GQA-grouped decode pass** (`attention_f32_splitkv_gqa[_kvf16]`). The split-KV partial pass ran one workgroup per
  *query* head, so each of a KV head's G query heads streamed the same rows. For Bonsai 2 (24 / 4) that is 6 reads of every
  cached byte per decoded token. The grouped pass runs one workgroup per (KV head, split). Each K element feeds G dot
  products (one subgroup per position) and each V element feeds G register accumulators (thread per column). Each query
  head keeps its own online softmax, handled by one subgroup. The partial layout per (query head, split) and the merge pass
  are unchanged. It serves 2 ≤ G ≤ 8 and head_dim ≤ 256. The dense hybrid host opts in
  (`VulkanQwen3MoeHybridKernels.EnableGqaDecode`); `DOTLLM_VULKAN_SPLITKV_GQA=1` / `0` forces it on or off on any host.
  `VulkanSplitKvAttentionKernelTests` (ALiBi, soft-cap, sliding window, bidirectional, masked tails) and the F16 tests run
  both passes. `VulkanSplitKvDecodeBench` (`DOTLLM_SPLITKV_DECODE_AB=1`) times all four arms at the Bonsai shape.
- **Measured** (`VulkanSplitKvDecodeBench`, one layer's decode attention, Bonsai shape 24 / 4 / 256, minimum of 9 interleaved
  runs, ms; "unique GB/s" counts each cached K / V byte once):

  | Strix (8060S) | 8k | 32k | 64k | 128k | 128k unique GB/s |
  |---|---|---|---|---|---|
  | F32, per-query-head | 0.69 | 2.48 | 4.89 | 9.62 | 112 |
  | F32, GQA-grouped | 0.67 | 2.41 | 4.67 | 9.12 | 118 |
  | F16, per-query-head | 0.53 | 2.35 | 4.63 | 9.26 | 58 |
  | **F16, GQA-grouped** | **0.39** | **1.59** | **3.09** | **6.13** | 88 |

  F16 alone does not help the per-query-head pass: its six re-reads per KV head are served from cache, and the pass is
  bound by issue, not DRAM. F16 needs the grouped pass to pay off. Together they are 1.57x faster than F32 at 128k (88 GB/s of unique KV, against
  the ~115 GB/s the F32 arms reach). A variant loading element pairs (one F16 word per lane) reached 5.37 ms / 100 GB/s on
  Strix but computed wrong results on the Arc (Intel) and was withdrawn (#1113). Halving the bytes again is
  what q8_0 KV (#1112) would do. On the Arc (8k / 32k) the grouped pass is 1.55-1.7x faster for both formats, and F16
  adds little there: that decode is compute-bound.
- **Mutants** (shader edits, run on Strix): swapping the half select, reading word `i` instead of `i >> 1`, and a
  `hq % numKvHeads` head broadcast in the F16 split-KV body. Each fails every test of the family it touches, and the
  model-level F16-vs-F32 comparison as well.

## Qwen4-Exp QSA (query-sparse attention) on Vulkan — issue #819

The 12 full-attention layers of `qwen4exp` attend a SELECTED subset of the context: an indexer pools the keys of every block of 4 tokens, scores
the blocks against the query, keeps the top 512 (= `indexer.top_k` 2048 tokens) plus the tail of the incomplete block, and runs ordinary GQA over those
~2051 keys. Up to `top_k + block - 1` = 2051 tokens every complete block is selected, so QSA equals dense causal attention; beyond it the cost per decoded
token is bounded by 2051 keys regardless of context. The CPU oracle is `Qwen4ExpQsa`; `VulkanQwen4ExpQsa` (`src/DotLLM.Vulkan/VulkanQwen4ExpQsa.cs`,
`Kernels/Qwen4ExpQsaKernels.cs`) is the device implementation, hooked into the shared `RecordFullAttnLayer` through `IQ4AttentionHook`.

Per QSA layer, per forward (all in the forward's command buffer, nothing returns to the host):

1. `indexer.k_proj(x)` -> RAW (un-normed, un-rotated) keys, copied into a per-sequence, position-indexed store `[capacity, 128]`. Runs on EVERY forward,
   including dense-regime ones, so a sequence can cross the dense limit mid-stream; rollback is just a smaller sequence length (nothing to restore).
2. `qsa_pool_f32.comp`: block `b` = `rope(rmsnorm(mean(raw[4b..4b+3])), pos 4b)` (pool, then norm, then rotate), for every block this forward completes.
   A block is a pure function of its four raw rows, so a rewritten position simply re-derives it.
3. Only when the forward reaches positions whose complete-block count exceeds the budget: `indexer.q_proj(x)`, per-head RMSNorm, NeoX RoPE at the query
   position, then `qsa_score_f32.comp` (`sum_h relu(q_h . k_b) / sqrt(128)`, subgroup-per-block, lanes across the key dim) and `qsa_select_f32.comp`
   (exact top-512: 4-pass radix select on the float bit patterns, then an index-ordered compaction; ties go to the LOWER block index like the oracle and the
   ids come out ascending). Queries are processed in sub-chunks (<= 1024) so score / partial scratch is bounded independent of the prefill chunk.
4. `qsa_attention_f32.comp` + `qsa_merge_f32.comp`: split-KV online-softmax attention over the virtual key list (selected blocks' tokens, then the tail),
   so dense-regime and sparse queries share one kernel (identity selection when the query still has <= 512 blocks).

The dense attention of the shared layer is untouched and still serves forwards that end below the dense limit.

Validation (`VulkanQwen4ExpQsaTests`): exact set equality of the select kernel against the oracle incl. heavy ties / all-equal / 65536 blocks; pooled keys and
scores against `Qwen4ExpIndexerCache` / `ScoreBlocksScalar`; whole-model prefill, token-by-token decode through the dense limit, chunked == single-shot and the
released head_dim-256 geometry against the CPU oracle with a 16-token budget (sparse from the 20th token), with a sensitive control (the oracle's own dense vs
sparse outputs differ ~1000x more than Vulkan vs oracle). The GPU and the oracle sum scores in a different float order, so a near-tie at the 512th block can flip
one block (1 row in 648 in the synthetic sweep; `AMaterialRowMismatch_IsATopKBoundaryNearTie` shows its GPU rank-4 / rank-5 scores differ by 4.5e-5 relative, and
fails if a mismatching row is ever not such a near-tie); the kernel-level tests prove the selection logic itself is exact.

Measured on the real UD-Q4_K_XL file (Strix Halo, 512 MB BIOS split, 2026-10-10; harness `VulkanQwen4ExpRealQsaTests`, env-gated by `DOTLLM_QWEN4EXP_REAL_GGUF`):

| | result |
|---|---|
| needle-in-a-haystack (3 depths x 3 codes), 4046 and 7890 tokens | **6/6** (answer = the planted code, no distractor) |
| decode tok/s at depth 1K / 2K / 4K / 8K (same state, growing) | 22.5 / 21.2 / 21.6 / 21.3 (44-47 ms/token): flat, the cost is bounded by ~2051 keys |
| indexer cost at 1K (same-session A/B, 3 rounds, indexer on vs off) | 43.2-43.8 vs 44.4-45.7 ms/token: within noise |
| prefill, chunks of 1024 | 3.7 s per first 1K, 3.8 s for 1K-2K, then ~240-260 tok/s at 2K-8K depth (the sparse region costs ~15 % vs the dense one) |
| KL vs the CPU oracle, 4096-token held-out window, scored half | mean 0.054, median 0.026, top-1 88 %, PPL 9.358 (Vulkan) vs 9.374 (oracle); no step at the dense limit: rows [1800,2051) 0.051, [2051,2300) 0.040 |
| 16384-token capacity (`DOTLLM_VK_QWEN4EXP_CONTEXT=16384`, one live sequence): needles at 12,769 tokens | **3/3**; decode 21.3 / 21.2 tok/s at depth 12K / 16K (47 ms/token), prefill 230-240 tok/s at 12K. Before #898 each live state held the full K/V + indexer allocation (0.9 GiB at 16K), so two concurrent 16K states exceeded the resident wall (lazy growth, below, removed that) |
| device memory | heap1 69,780 MiB + heap0 9.8 GiB (was 12.1 GiB before the host embedding gather) at an 8192-token capacity |

The KL level is the pre-existing Vulkan-vs-oracle gap that grows with context (#873, ~0.03-0.05 at 512-2K positions on the same file), not a QSA effect: it has no
discontinuity where the sparse path starts.

Capacity: the K/V cache is F32 by default (48 KiB per token over the 12 layers) plus 7.5 KiB per token of indexer keys (6 KiB raw + 1.5 KiB pooled). The model is
sized for `min(context_length, 8192)` tokens by default (`DOTLLM_VK_QWEN4EXP_CONTEXT` overrides; halved automatically until the residency plan fits). On the
128 GiB Strix Halo box the OS keeps ~82 GiB resident per process and the UD-Q4_K_XL trunk is ~79 GiB, which is why the token-embedding table is gathered on the
HOST for qwen4exp (`DOTLLM_VK_QWEN4EXP_HOST_EMBED`, default on): the device-resident F32 copy was 2.4 GiB.

### Lazy growth and F16 K/V (issue #898, lifts the ~16K ceiling)

* **Lazy growth (default on, `DOTLLM_VK_QWEN4EXP_KV_LAZY=0` restores eager).** A sequence's K/V cache (`VulkanNemotronHKvCache`, `lazy: true`) and indexer state
  (`VulkanQwen4ExpIndexerState`) start empty and grow 25 % (>= 1024 positions) at a time: `ForwardCore` calls `EnsureContext` BEFORE recording, which reallocates layer by
  layer (one layer's K+V transient), copies the live prefix, and drops every descriptor cache that could name the freed handles (the aliasing trap of #437: a stale
  cached set silently reads zeros). `MaxLength` stays the logical limit. Output is **bit-identical** to eager (KL 0, 72 of 72 logit rows on the real file; synthetic
  test across 5 growths); a growth step stalls ~0.3 s (21 growths = 6.2 s over two 16K prefills, ~4 % each).
* **F16 K/V (opt-in: `DOTLLM_VK_QWEN4EXP_KV=f16`).** K/V rows are converted by `convert_f32_to_f16_range.comp` (round-to-nearest-even in the shader: AMD's `packHalf2x16`
  truncates) and read DIRECTLY by `qsa_attention_f16kv.comp`, a byte-for-byte twin of `qsa_attention_f32.comp` with only the K/V loads changed (no dequant pass; the
  gather touches ~2K keys per query, a full-history dequant would move ~190 MiB per layer per step at 32K). The gather kernel attends every block while a query's block count
  fits the budget, so below the dense limit an F16 cache is served by it too instead of the dense kernels (which cannot read F16). The hook therefore MUST handle every layer:
  `RecordFullAttnLayer` throws if an F16 cache reaches the dense path, and F16 with `DOTLLM_VK_QWEN4EXP_QSA=0` is refused. The MTP draft head keeps an F32 cache.
  `DOTLLM_VK_QWEN4EXP_IDX_F16=1` additionally stores the indexer's RAW keys as F16 (`qsa_pool_f16raw.comp`; pooled keys stay F32): -3 KiB/token, slightly noisier selection.

| per token, 12 layers | F32 (default) | F16 K/V | F16 K/V + F16 raw keys |
|---|---:|---:|---:|
| K/V | 48 KiB | 24 KiB | 24 KiB |
| indexer (raw + pooled) | 7.5 KiB | 7.5 KiB | 4.5 KiB |
| total; 32K tokens | 55.5 KiB; 1.73 GiB | 31.5 KiB; 0.98 GiB | 28.5 KiB; 0.89 GiB |

Measured on the real UD-Q4_K_XL file (same-session, `VulkanQwen4ExpRealQsaTests`; quiet machine unless noted):

| | F32 K/V (eager) | F32 lazy | F16 lazy | F16 + F16 raw keys |
|---|---|---|---|---|
| device heap0 after a 4096-token prefill (MiB; weights + scratch 9,813) | 10,870 | 10,260 | 10,140 | 10,125 |
| same at 16,000 tokens | 10,870 | 10,870 | 10,486 | 10,438 |
| KL vs CPU oracle, 4096-token held-out window, scored half (rope fix in) | 0.0549 | | 0.0558 | |
| held-out PPL, same window (CPU oracle 9.374) | 9.337 | | 9.356 | |
| KL(F32 K/V \|\| arm), same window | | 0 (bit-identical) | 0.0063 (top-1 95.8 %) | 0.0069 |
| KL(F32 \|\| arm), 72 decode/prefill-tail rows at 2K-16K | | 0 | 0.027 | 0.042 |
| control: KL(F32 \|\| F32 with the softmax merge in 1 split instead of <=16, a pure summation-order change), same 72 rows | | 0.017 | | |

The model is chaotic to any rounding change (quantised weights and activations digitise ULP drift), so an F16-vs-F32 KL of 0.006-0.03 is the same order as merely reordering float
sums (0.017) and is much smaller than the existing Vulkan-vs-oracle gap (0.054, #873); the distance to the CPU oracle and the PPL do not move. F16 raw keys flip a few more top-k blocks
(KL 0.042 on the decode rows), so they are a separate opt-in.

Long context with F16 K/V (capacity 34,000; needles from `scratchpad/needle2`): needles **6/6** at 23,783 and 31,011 tokens; prefill 229-238 tok/s; decode 20.5 tok/s at depth 32K
(19.5-20.7 from 4K to 32K: flat, as the gather is bounded by ~2K keys). Ceiling probe (`DOTLLM_VK_QWEN4EXP_CONTEXT=66000`, F16 K/V, one sequence of filler tokens, no needles):
the residency plan keeps the full 66,000 capacity, the sequence reaches 64,024 tokens without an allocation failure, prefill 210-220 tok/s over the last 16K, decode 20.1 tok/s at 48K and 19.4 at 64K
(quality beyond 32K is NOT validated: no needle files exist above 31K). Same-session interleaved A/B, F32 lazy vs F16 lazy: decode 45.9 / 44.4 ms at 1K, 49.6 / 46.5 at 4K, 48.0 / 46.9 at 8K,
52.4 / 47.9 at 16K; sparse prefill 202 / 221 tok/s: F16 is never slower (half the bytes per gathered key).

RoPE fp32 angle: the shaders computed `sin(pos * freq)` with the GPU's own sin/cos, whose internal fp32 argument reduction costs ~6e-8 x angle of extra phase error (1.3-2.7x the CPU
table's error from 8K positions: max |error| of (cos, sin) vs double truth at 32K/131K positions was 2.8e-3/1.1e-2 vs the CPU table's 1.0e-3/3.8e-3). `rope_f32`, `rope_kv_write_f32`,
`rope_invfreq_f32`, `rope_yarn_f32`, `rope_mla_f32` and `qsa_pool_*` now range-reduce the angle to [-pi, pi] with a Cody-Waite split of 2 pi before `sin`/`cos`, bringing the GPU to the CPU
table's error (1.07x at worst; identical for |angle| < pi, so short contexts are unchanged). The remaining loss is the fp32 product `pos * freq` itself, shared by the CPU table, HF and
llama.cpp (CUDA and the CPU use libm sin/cos, which reduce exactly). It did not move the Vulkan-vs-oracle KL at 4096 (0.0542 -> 0.0549): not a cause of #873.

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
