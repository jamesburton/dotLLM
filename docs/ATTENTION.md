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

### Tuning headroom (deferred — workable WG=64 + BR=16 baseline shipped)

Things worth probing on Strix Halo before tile sizes lock in:

- **BR × BC tradeoff.** Current BR=16, BC=64. Larger BR amortises K reads more but bloats `outAccum` and `scoreMatrix` in LDS; current footprint is ~20 KB out of 64 KB, so BR=32 is room available (scoreMatrix grows to 8 KB, outAccum to 16 KB). Worth measuring whether BR=32 / BC=64 beats BR=16 / BC=64 on Strix Halo's specific occupancy / register-pressure profile.
- **Streamed vs shared K/V.** Current shader streams K and V directly from global in the inner loops — relies on L1/L2 cache locality across the 64 threads (all hit the same tile rows). On RDNA3.5 that should be fine. Loading the K-tile into LDS once per WG would trade LDS for global reads; might help on devices with weaker L1.
- **Per-row reduction granularity.** When `subgroupSize == WG_SIZE` (Strix Halo case), the `wgRowMax` / `wgRowSum` cross-subgroup combine is dead code — one `subgroupMax` / `subgroupAdd` is the whole reduce. Could specialise via a constant or a second SPV variant; not a meaningful speed win at the current per-tile barrier count, but cleaner.
- **Q-tile transpose for memory coalescing.** Currently `qTile[r * head_dim + d]` is row-major; threads access `qTile[r][d]` strided by `d` across rows during the score loop, which is irregular. A `[d][r]` layout might give better LDS bank-conflict behaviour. Trace it before changing.
- **Soft-cap fold into `scale`.** When `softCap > 0` and the raw score is far from saturation, the `tanh` is wasted work; could skip when score magnitude < softCap × 0.5. Marginal.

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
one block (1 row in 648 in the synthetic sweep); the kernel-level tests prove the selection logic itself is exact.

Measured on the real UD-Q4_K_XL file (Strix Halo, 512 MB BIOS split, 2026-10-10; harness `VulkanQwen4ExpRealQsaTests`, env-gated by `DOTLLM_QWEN4EXP_REAL_GGUF`):

| | result |
|---|---|
| needle-in-a-haystack (3 depths x 3 codes), 4046 and 7890 tokens | **6/6** (answer = the planted code, no distractor) |
| decode tok/s at depth 1K / 2K / 4K / 8K (same state, growing) | 22.5 / 21.2 / 21.6 / 21.3 (44-47 ms/token): flat, the cost is bounded by ~2051 keys |
| indexer cost at 1K (same-session A/B, 3 rounds, indexer on vs off) | 43.2-43.8 vs 44.4-45.7 ms/token: within noise |
| prefill, chunks of 1024 | 3.7 s per first 1K, 3.8 s for 1K-2K, then ~240-260 tok/s at 2K-8K depth (the sparse region costs ~15 % vs the dense one) |
| KL vs the CPU oracle, 4096-token held-out window, scored half | mean 0.054, median 0.026, top-1 88 %, PPL 9.358 (Vulkan) vs 9.374 (oracle); no step at the dense limit: rows [1800,2051) 0.051, [2051,2300) 0.040 |
| 16384-token capacity (`DOTLLM_VK_QWEN4EXP_CONTEXT=16384`, one live sequence): needles at 12,769 tokens | **3/3**; decode 21.3 / 21.2 tok/s at depth 12K / 16K (47 ms/token), prefill 230-240 tok/s at 12K. Each live state holds the full K/V + indexer allocation (0.9 GiB at 16K), two concurrent 16K states exceed the resident wall |
| device memory | heap1 69,780 MiB + heap0 9.8 GiB (was 12.1 GiB before the host embedding gather) at an 8192-token capacity |

The KL level is the pre-existing Vulkan-vs-oracle gap that grows with context (#873, ~0.03-0.05 at 512-2K positions on the same file), not a QSA effect: it has no
discontinuity where the sparse path starts.

Capacity: the K/V cache is F32 (48 KiB per token over the 12 layers) plus 5 KiB per token of indexer keys. The model is sized for
`min(context_length, 8192)` tokens by default (`DOTLLM_VK_QWEN4EXP_CONTEXT` overrides; halved automatically until the residency plan fits). On the 128 GiB Strix Halo
box the OS keeps ~82 GiB resident per process and the UD-Q4_K_XL trunk is ~79 GiB, which is why the token-embedding table is gathered on the HOST for qwen4exp
(`DOTLLM_VK_QWEN4EXP_HOST_EMBED`, default on): the device-resident F32 copy was 2.4 GiB.

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
