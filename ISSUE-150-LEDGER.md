# Issue #150 — extend shared Q8_1 decode quant to all MMVQ weight quants: ledger

**FOR THE COORDINATOR: fold this into `.docs/KERNEL_MAP.md` §3/§4 (the "Q8_0-only" notes and
the §3 SUSPECTED-SLOW item 7 / §4 sharing bullet) and delete this file before the PR.** The
worktree cannot edit the git-ignored main-repo `.docs/`.

Branch `issue/150-q81-sharing` (base `dev` @ `76de82c3`). All numbers gfx1151 (Strix Halo,
Radeon 8060S), same-session back-to-back unless noted. GPU work done under `agent-150-q81`
gpu-lock.

## Why it was gated to Q8_0

`CanShareMmvqQuant` (pre-#150) hard-required every member of a same-input MMVQ decode group
(Q/K/V over post-attn-norm, gate/up over post-ffn-norm) to be `QuantType.Q8_0`. Investigated
whether this reflected a real kernel constraint or was conservatism carried over from the
original #46 spike (which only wired Q8_0 MMVQ):

- Every `MatMul*MmvqKernel.Record(cmdBuf, weights, xq, xds, y, m, k)` — Q8_0, Q2_K..Q6_K,
  IQ4_NL, IQ4_XS, IQ1_S..IQ3_S — has the **identical signature**, consuming the same packed
  int8 `xq` + per-32-block `(scale, sum)` `xds` pair produced by `QuantizeQ8_1Kernel`. The
  activation-quant format is weight-format-independent (matches llama.cpp: activations are
  Q8_1 for every integer-dot `mul_mat_vec` variant regardless of weight quant).
- Confirmed via `git log -S CanShareMmvqQuant`: the Q8_0 gate traces to the original #46/#52
  sharing spike (`00415b04`), before any non-Q8_0 MMVQ kernel existed; it was never revisited
  when Q2_K..Q6_K/IQ4_*/IQ1_S..IQ3_S MMVQ landed in #338/#339.
- **Conclusion: conservatism, not necessity.** The gate was extended.

## What changed

`src/DotLLM.Vulkan/VulkanTransformerModel.cs`:

- `CanShareMmvqQuant` now checks `HasMmvqDecodeKernel(quant, inputDim)` per member instead of
  `p.WeightQt != QuantType.Q8_0` — mirrors the existing prefill-side `HasMmqPrefillKernel`.
- New `HasMmvqDecodeKernel(QuantType, int inputDim)`: per-quant kernel-loaded + alignment check
  (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S).
- New `RecordMmvqGemvPreQuantized(qt, ...)`: dispatches the correct MMVQ kernel for `qt`
  against the already-populated shared `Q8_1Xq`/`Q8_1Xds` scratch. Replaces the previous
  hardcoded `_matmulQ8Mmvq!.Record(...)` call at both group-dispatch sites
  (`RecordSharedInputMmvqGroup` and the #145 fused-rmsnorm+quantize path in
  `RecordNormedSharedInputMmvqGroup`), so the #145 fused dispatch now also fires whenever the
  (possibly mixed-quant) group qualifies.
- Groups can now be **mixed-quant** (e.g. Q4_K q/k-proj + Q6_K v-proj) since the shared xq/xds
  scratch is quant-agnostic — each member still dispatches its own MMVQ kernel.

## Gates

### Parity

- **New** `VulkanMmvqMixedQuantSharedGroupTests` (unit, kernel-level): bit-identical
  shared-vs-per-projection GEMV outputs for 4 mixed Q4_K/Q6_K/IQ4_XS group shapes (incl.
  routing-order and minimum-superblock discriminators). PASS (4/4).
- **New** `VulkanMmvqShareGreedyParityHarness` (integration, env-gated
  `DOTLLM_VULKAN_SHARE_GREEDY=1`): exact-token greedy-128 decode, sharing (default) vs
  `DOTLLM_VULKAN_MMVQ_NO_SHARE=1` reference arm, on the two target real models:
  - `Llama-3.2-3B-Instruct-IQ4_XS.gguf`: **128/128 tokens identical.**
  - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`: **128/128 tokens identical**
    (`shared==unshared` full sequence: `578,3446,1288,387,279,1193,...,15746,433,11`).
- Existing `VulkanMmvqSharedQuantParityTests` (SmolLM-135M Q8_0, bit-identical logits,
  6 decode steps): still PASS — the Q8_0-only path is unaffected.
- Existing `VulkanMatMulQ8_0MmvqKernelTests.Mmvq_SharedQuant_BitIdenticalToPerProjection`:
  still PASS.

### Dispatch census (`DOTLLM_VULKAN_DECODE_PROFILE=1`, same-session A/B, `dotllm bench -p 64 -n 64 -r 1`)

| Model | dispatches/tok before→after | barriers/tok before→after |
|---|---|---|
| 3B IQ4_XS | 619 → 479 (**−140**) | 678 → 454 (**−224**) |
| 8B Q4_K_M | 707 → 547 (**−160**) | 774 → 518 (**−256**) |

(SmolLM-135M Q8_0 unaffected — was already sharing; census unchanged.)

### Same-session A/B decode (`dotllm bench --device vulkan -p 512 -n 128 -r 3`, median of 3)

| Model | decode tok/s before (NO_SHARE) | after (shared, default) | delta |
|---|---|---|---|
| SmolLM-135M Q8_0 | 483.30 | 552.79 | **+14.4%** |
| 3B IQ4_XS | 77.78 | 81.71 | **+5.1%** |
| 8B Q4_K_M | 28.11 | 30.06 | **+6.9%** |

(SmolLM's delta is larger than the #150 est. because the fused rmsnorm+quantize (#145) dispatch
now ALSO fires on 3B/8B for the first time — previously #145 could only fire on Q8_0 groups.
UMA back-to-back noise band is ~a few %, per LLMs.md; all three deltas exceed it comfortably
and are corroborated by the exact-token parity + dispatch-census evidence above.)

### Full Vulkan unit suite

`dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~Vulkan"`: **909 passed, 0
failed, 41 skipped** (clean rerun; an earlier interrupted run showed a single transient
failure that did not reproduce — not attributable to this change).

## Follow-ups

- Prefill-side (`CanShareMmqRowsQuant` / `HasMmqPrefillKernel`) was **already** quant-agnostic
  before #150 (issue #139's prefill analogue) — no further extension needed there.
- The fused rmsnorm+quantize (#145) kernel (`RmsNormQuantizeQ8_1FusedKernel`) required no
  changes: it only produces the shared `Q8_1Xq`/`Q8_1Xds` pair, which was already
  quant-agnostic; #150 just widened which GROUPS reach it.
- Not investigated: whether mixed-quant groups are common enough in practice to matter beyond
  the two target models (3B has an all-IQ4_XS attn block; 8B Q4_K_M has a Q4_K/Q6_K split on
  attn_v/ffn_down per llama.cpp's typical K-quant mix) — worth confirming against a
  quant-distribution audit of the GGUF llama.cpp uses for Q4_K_M/IQ4_XS if further wins are
  sought here.
