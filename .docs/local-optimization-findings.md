# Local-platform optimisation findings (2026-06-15)

Campaign on the local box: **Intel Core Ultra 7 155H (Meteor Lake)** CPU + **Intel Arc iGPU** + **NVIDIA
RTX 3060 (Ampere, eGPU)**. Branch `spike/local-platform-opt` (worktree `C:\Development\dotllm-bf16e2e`).
All numbers measured here; raw logs + per-spike detail in `local-optimization-campaign.md`.

> **Measurement discipline that mattered (this hardware is noisy):** a laptop CPU (turbo/thermal) and a
> consumer GPU (boost/thermal) both drift ~2× across consecutive runs. Reliable A/B requires **pairing the two
> variants per round and taking the median ratio** (CPU) or **interleaving within one warmed process + min**
> (GPU). Dividing separately-measured mins, or a fresh-process A/B, gave flatly wrong answers several times
> (e.g. a "VNNI 0.56×" and a "16F 2× slower" that were pure clock drift). Also: warm up GPU prefill (first
> forward pays one-time cuBLAS/PTX-JIT init) and pin CPU benches to a P-core.

---

## Implemented & committed

| # | Change | Target | Measured | Commit |
|---|--------|--------|----------|--------|
| **C1** | AVX-VNNI `VPDPBUSD` Q8_0 outer-product (port #321) | Meteor Lake CPU prefill | **1.55× on P-cores** (flat across K=2048/4096/8192), 1.2-1.35× E-cores; 14/14 parity | `70c85c0` |
| **C3** | Fix `NumaTopology` P/E inversion (`EfficiencyClass` min→max) | Meteor Lake hybrid scheduling | `--pcore-only` was pinning to the **slow** cores (~6×); now correct | `d17e860` |
| **V1** | Per-vendor Q8_0 GEMV variant (Intel→`sg`, NVIDIA→`wg64`, else wg128) | iGPU + eGPU decode | decode-GEMV min **Arc 1.17×, NVIDIA 1.13×** (sweep: 1.05×/1.07× weighted, up to 2.03× on a shape) | (pending) |
| infra | profiling harness, G1 toggle, paired microbench, Vulkan device override | — | — | `df6a3ba` |

C1/C3 are Meteor-Lake-specific by construction (C1 only dispatches on AVX2+VNNI-without-AVX512; C3 only changes
hybrid hosts) — **Strix/AMD/Zen5 are untouched.**

## Measured but minor / conditional

- **G1 — cuBLAS `COMPUTE_16F` prefill GEMM:** 1.06× median / 1.22× best whole-prefill (the GEMM is ~half of
  prefill and only it benefits). Real but small, and FP16 accumulation needs a perplexity gate before adoption.
  Wired behind `DOTLLM_CUDA_GEMM_16F` / `CudaGemm.Use16FCompute` — left OFF by default.

## Key findings (no code change, but they direct the big wins)

1. **🔑 Prefill attention is the #1 GPU prize and uses NO tensor cores.** Warmed Llama-3.2-1B pp256 on the 3060:
   **attention 43%**, GEMM 49%. `native/kernels/attention.cu` is a flash-style FP32 kernel on CUDA cores —
   the 3060's tensor cores sit idle for it. Attention is O(seq²) so it dominates further at longer prompts.
2. **🔑 Small-model decode is launch/dispatch-overhead-bound on ALL three backends** (Arc Vulkan 8 tok/s,
   3060 Vulkan 24, CUDA ~27 eager) — ~30 GB/s of 360 on CUDA. The lever is *fewer kernel launches per token*
   (CUDA graphs — already exist but profiling-disabled; Vulkan command-buffer batching; kernel fusion), not
   compute or bandwidth.
3. **Intel Arc cold start = 42 s** (no `VkPipelineCache` → every launch recompiles all pipelines; dGPU 7.8 s).
4. **Optimal GEMV workgroup is vendor-dependent** (sg on Intel, wg64 on NVIDIA) — and **per-shape** varies too
   (e.g. attn_kv → wg256 2.03× on NVIDIA). V1 captures the per-vendor part; per-shape is future work.

## Top remaining opportunities (ranked by value; not implemented — effort noted)

1. **Tensor-core prefill attention** (biggest GPU win, grows with prompt). Medium-large: either cuBLAS
   strided-batched QK^T/PV GEMM + a softmax pass (reuses existing cuBLAS P/Invoke), or a hand-written FP16
   `mma` flash kernel (needs a compute_80 PTX target — current floor is compute_75). The Vulkan side already
   has an `attention_f32_coopmat` variant to lean on.
2. **Per-token launch-overhead reduction for decode** (helps every model/backend). Ensure CUDA graphs are on in
   production decode; batch the Vulkan per-token command buffer; fuse small elementwise kernels.
3. **G2 — INT8 prefill GEMM (cuBLASLt IMMA)**: 2× FP16-TC rate + halves weight bytes (skips the FP16 dequant
   detour). Medium: needs new cuBLASLt P/Invoke + INT8 layout handling. Value capped by attention dominating.
4. **`VkPipelineCache`** persisted to disk — kills the 42 s (Arc) / 7.8 s (dGPU) cold start. Contained.
5. **Per-shape GEMV variant dispatch** — extend V1 from per-vendor to per-(vendor,shape) (attn_kv wg256 = 2.03×
   on NVIDIA). Small, but needs a small per-shape table.

## Explicitly de-prioritised
- **C2 — GFNI nibble unpack** for Q4/Q2 dequant: speculative; nibble unpack isn't the bottleneck (the dot
  product is), and AVX2 shift/mask is already cheap. Not pursued.
