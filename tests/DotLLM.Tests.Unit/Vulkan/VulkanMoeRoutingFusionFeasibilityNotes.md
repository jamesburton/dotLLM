# MoE routing/gather dispatch-count fusion — feasibility notes (2026-07-18)

Investigated whether llama.cpp's Vulkan `topk_moe.comp` "4-mode fusion" represents a dispatch-count
gap vs dotLLM's `RecordMoeLayer` MoE routing pipeline. **Conclusion: dotLLM already has the
equivalent fusion; the gap does not exist where the docs hinted it might.** Same shape as the
#377/#378 refutations this session.

## 1. dotLLM current dispatch list — `RecordMoeLayer` (`VulkanTransformerModel.cs:4550-4655`)

Non-grouped-F16 path (the one actually engaged for real quant-resident models, e.g. the validated
Qwen3.6-35B-A3B UD-Q4_K_XL run in #372 — banks resolve to Q4_K/Q5_K, not F16, so
`CanUseGroupedF16Moe` is false):

1. `_rmsnorm.Record` — pre-FFN RMSNorm (HiddenState → NormOutput).
2. `RecordMatmul` (router gate, Q8_0 or F32) — NormOutput → MoeRouterLogits, outputDim=numExperts.
3. `MoeTopKSoftmaxF32Kernel` (`moe_topk_softmax_f32.comp`) — **full softmax + top-K selection +
   optional renorm, ONE dispatch.**
4. `MoeBroadcastF32Kernel` — replicate each token's normed row topK× into `MoeExpandedInput`
   (physical row expansion so the indexed matmul below can read contiguous per-slot rows).
5. `RecordMoeIndexedMatmul` (W1/gate) — indexed expert matmul.
6. `RecordMoeIndexedMatmul` (W3/up) — indexed expert matmul.
   (+ optional `MoeIndexedLoraDeltaF32Kernel` ×2 if a LoRA adapter is active — not present in the
   baseline real-model run.)
7. `_swiglu.Record` — silu(gate) * up, pointwise.
8. `RecordMoeIndexedMatmul` (W2/down) — indexed expert matmul.
   (+ optional LoRA delta dispatch.)
9. `MoeWeightedScatterF32Kernel` — combine topK expert outputs × routing weights → NormOutput.
10. Optional shared-expert branch (`RecordMoeSharedExperts`) — DeepSeek ungated add / Qwen
    sigmoid-gated add (`moe_sigmoid_gated_add_f32.comp`) — skipped for models with no shared
    experts (Qwen3-MoE plain, Qwen3.6-35B-A3B, Gemma-4).

**9 dispatches per MoE layer in the common case** (no LoRA, no shared experts): 1 rmsnorm + 1 gate
matmul + 1 topk-softmax + 1 broadcast + 3 indexed matmuls (W1/W3/W2) + 1 swiglu + 1 weighted-scatter.
decode (seqLen==1) and prefill (seqLen>1) share this exact dispatch *sequence* — the difference is
only inside each kernel's launch parameters (`expandedRows = seqLen*topK`), not the dispatch count
or ordering. (Gemma-4 decode has a parallel MMVQ-specialized path in `RecordGemma4Ffn`, #137, with
the same dispatch shape — quantize-once + MMVQ indexed matmuls in place of steps 5/6/8 — not
examined line-by-line here since it doesn't change the routing-step conclusion below.)

The grouped-F16-coopmat path (`RecordMoeGroupedF16Layer`, only reachable when all three banks are
F16 and no LoRA) is *more* dispatch-heavy: expert-offsets + group-by-expert + 2 grouped matmuls +
2 ungroup-scatters (repeated once for W1/W3, once for W2) = 11 dispatches, plus the 5 routing/pre
dispatches above = 16 total. This path is not what real quant-resident models (#371/#372) use, so
it's not the representative case for this analysis.

**Routing-only overhead** (excluding the 3 expert matmuls + swiglu, which do the real FLOPs):
rmsnorm + gate matmul + topk-softmax + broadcast + weighted-scatter = **5 dispatches/layer**.

## 2. What llama.cpp's `topk_moe.comp` actually fuses

Read `ggml-vulkan.cpp:486-585` (the four fusable op-sequences) and the shader itself
(`vulkan-shaders/topk_moe.comp`, single WG, warp-level softmax/sigmoid + argmax-based top-K + gather
weights + optional norm, all in shared registers, no round-trip to global memory between stages).

The four fusion modes it recognizes and collapses into **one dispatch**:
- `TOPK_MOE_EARLY_SOFTMAX_NORM`: `SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS → RESHAPE →
  SUM_ROWS → CLAMP → DIV → RESHAPE` (10 ggml ops).
- `TOPK_MOE_SIGMOID_NORM_BIAS`: `UNARY(sigmoid) → RESHAPE → ADD(bias) → ARGSORT → VIEW → GET_ROWS →
  RESHAPE → SUM_ROWS → CLAMP → DIV → RESHAPE` (11 ops, DeepSeek-style sigmoid gate + bias + renorm).
- `TOPK_MOE_EARLY_SOFTMAX`: same as above truncated after `GET_ROWS` (no renorm).
- `TOPK_MOE_LATE_SOFTMAX`: `ARGSORT → VIEW → GET_ROWS → RESHAPE → SOFT_MAX → RESHAPE` (softmax
  applied to the already-selected top-K weights, not the full logit vector).

**All four modes cover exactly: gating activation (softmax/sigmoid) → optional bias → top-K
selection (argsort+view) → gather the selected weights (get_rows) → optional
renormalize/clamp/divide.** That is precisely the job of dotLLM's single `MoeTopKSoftmaxF32Kernel`
dispatch (step 3 above) — "full softmax + top-k + optional renorm" in one WG-64 dispatch. **dotLLM
already has this fusion; llama.cpp's `topk_moe.comp` is not fusing anything dotLLM currently keeps
split.** Neither implementation fuses the row-broadcast/gather-into-contiguous-expert-rows step
(dotLLM: `MoeBroadcastF32Kernel`; llama.cpp doesn't need an equivalent because its `mul_mat_id`
kernel indexes source rows directly rather than requiring a pre-broadcast contiguous buffer) or the
weighted-combine step (dotLLM: `MoeWeightedScatterF32Kernel`; llama.cpp: separate `mul` + reduce
ops, also un-fused) into `topk_moe.comp`.

## 3. Finding

**Gap is smaller than the doc's phrasing implied — already partially addressed.** dotLLM's
`MoeTopKSoftmaxF32Kernel` is functionally the same scope of fusion as llama.cpp's `topk_moe.comp`
(gating + top-K + gather-weights + renorm, one dispatch each). No fusion candidate work is
recommended on this specific step.

The two dispatches immediately adjacent to it — `MoeBroadcastF32Kernel` (step 4) and
`MoeWeightedScatterF32Kernel` (step 9) — are *not* addressed by llama.cpp's fusion either (llama.cpp
avoids the broadcast by having `mul_mat_id` gather-read directly, and keeps the weighted-combine
unfused). If dotLLM wanted to go further than llama.cpp here, the only structurally sound fusion
candidate left is **rmsnorm(step1) + gate-matmul(step2) into the existing dormant fused
rmsnorm+Q8_0-GEMV kernel** (KERNEL_MAP §"CPU/GPU MoE" already lists it as "effectively dormant" —
this MoE router call is a plausible use site if the router weights are quantized Q8_0, matching what
step 2 already does opportunistically via `RecordMatmul`). That is a *reuse* of prior fusion work,
not new shader design, and saves at most 1 dispatch/layer. No other candidate in this pipeline
matches the #145 fused-rmsnorm-quantize pattern (small fixed per-element work, disjoint memory
access) — broadcast and weighted-scatter both depend on the *variable* topK×seqLen shape and sit on
opposite sides of 3 large matmuls, so fusing across the matmuls isn't feasible without fusing the
matmuls themselves (out of scope, would duplicate `mul_mat_id`-style work already tracked as
SUSPECTED-SLOW item 2 in KERNEL_MAP §7).

## 4. Dispatch-count-reduction estimate for a real model

Using the validated real-model config from KERNEL_MAP §7 (Qwen3.6-35B-A3B UD-Q4_K_XL, topK=8, the
non-grouped-F16 indexed-matmul path, no LoRA, no shared experts) and typical Qwen3-30B-A3B-class
layer count (~48 transformer layers — dotLLM doesn't have this exact model's `NumLayers` recorded in
`.docs/`; treating this as an order-of-magnitude estimate from the public Qwen3-MoE family shape,
not a codebase-verified count):

- Routing-only dispatches/layer: 5 → **~240 dispatches/forward-token** for routing alone.
- Full MoE-block dispatches/layer (routing + 3 indexed matmuls + swiglu): 9 → **~432
  dispatches/forward-token** for the whole MoE block.
- Even eliminating the one plausible extra fusion (rmsnorm+gate-matmul → 1 dispatch) saves at most
  **~48 dispatches/forward-token** (1/layer), i.e. ~11% of the MoE-block dispatch count and ~0.2
  dispatches saved per matmul-FLOP-dominated layer — a rounding error next to the 3 expert matmuls
  that do the actual compute work, each of which is already the focus of #137/#372's real wins
  (220× decode speedup from resident-quant kernels).

This is a **dispatch-count** argument, not a **wall-clock** one: on Strix Halo the measured decode
bottleneck (per memory `vulkan-decode-barrier-bottleneck` and the #372 real-model numbers, 14.7-14.8
tok/s post-fix) is dominated by expert-matmul memory bandwidth, not command-buffer submission or
dispatch launch overhead — the earlier campaign (memory `vulkan-throughput-campaign-status`)
already ruled out barriers/launch overhead as a lever on this hardware for the *attention* path, and
nothing here suggests MoE routing's ~5 small, WG-64/128-sized dispatches would behave differently.

## 5. Priority recommendation

**Do not prioritize.** Three independent reasons converge:
1. The specific fusion the doc pointed at (`topk_moe.comp`) is a false gap — dotLLM already has it
   (`MoeTopKSoftmaxF32Kernel`).
2. The one remaining plausible candidate (rmsnorm+gate-matmul fusion) saves ≤1 dispatch/layer
   (~11% of MoE-block dispatch count) and requires no new shader work, but delivers a
   dispatch-count win in a regime (MoE decode) that #137/#372 already established is
   bandwidth-bound on the expert matmuls, not dispatch-count-bound — so the wall-clock payoff is
   expected to be near-zero.
3. MoE decode on Strix Halo was already heavily optimized this cycle (#137: Gemma-4 MMVQ +32%;
   #372: Qwen3.6 resident-quant +220×) — those wins came from *reducing bytes moved and avoiding
   F32 dequant*, not from *reducing dispatch count*. Routing-step fusion is optimizing a part of the
   pipeline that isn't the bottleneck.

If a future profiling pass on real hardware ever shows non-trivial time in the 5
routing-dispatch group specifically (vs. the 3 expert matmuls), revisit; no such evidence exists
today.
