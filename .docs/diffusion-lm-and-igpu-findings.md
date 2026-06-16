# iGPU optimization + diffusion-LM spike (2026-06-15)

Local box: **Intel Core Ultra 7 155H (Meteor Lake)** + **Intel Arc iGPU (Xe-LPG)** + **NVIDIA RTX 3060
(eGPU)**, 96 GB RAM. Branch `spike/local-platform-opt` (worktree `C:\Development\dotllm-bf16e2e`). Covers two
questions: (A) further iGPU/Vulkan optimization gaps, and (B) can DiffusionGemma run here.

---

## A. iGPU (Intel Arc / Vulkan) optimization gaps

Xe-LPG (Meteor Lake iGPU) reality: **DP4a yes, XMX no** (matrix engines dropped from Meteor Lake; only
discrete Arc Xe-HPG and Lunar Lake Xe2 have them), shared system RAM (no PCIe to iGPU), variable subgroup
(8/16/32, =32 here). Ranked gaps (all benefit every model on the iGPU, independent of diffusion):

| # | Gap | Status | Effort |
|---|-----|--------|--------|
| 1 | **DP4a INT8 GEMV** (`VK_KHR_shader_integer_dot_product`) — quant shaders unpack int8→FP32; Xe-LPG's DP4a hardware unused (~4–8× INT8 left on table) | **Absent** — headline compute win | ~2–3 d |
| 2 | **VkPipelineCache to disk** — ~95 SPIR-V pipelines recompiled every launch → ~42 s Arc cold start | **DONE** (commit `203faa8`) | — |
| 3 | Flash-attn WG=32 variant for Xe-LPG subgroup width (only WG=64 exists) | Absent | ~1 d |
| 4 | `copy_f32` dispatches → `vkCmdCopyBuffer` (KV scatter/gather; decode is launch-overhead-bound) | Absent | ~2 d |
| — | Zero-copy UMA flag `SupportsZeroCopyMmap=false` | **Cosmetic** — never read; real zero-copy gates on `HasExternalMemoryHost` (likely already active) | — |

`.NET-runtime` angle for the iGPU: it's the **interop/zero-copy** side, not `TensorPrimitives` — mmap'd weights →
`VkBuffer` via `VK_EXT_external_memory_host` (built in `HostVisibleBuffer.cs`); and the decode loop's per-step
**managed command-buffer construction** (~574 dispatches/step: descriptor sets, marshalling, P/Invoke) is worth
profiling. Coopmat is correctly gated off (Xe-LPG lacks it); per-vendor GEMV (Intel→`sg`) already in (V1).

## B. DiffusionGemma feasibility

DiffusionGemma (Google, 2026) is a **text-diffusion LM** — iterative full-sequence denoising, bidirectional
attention, large MoE (~26B, smallest quant ~18 GB). *Exact config is unverified web synthesis; confirm from HF
`config.json` before any build.*

- **Memory: fine here** (96 GB; iGPU addresses ~half → Q4/Q8 fit). *Throughput poor* though — Meteor Lake
  shared LPDDR5x is ~⅓ the bandwidth of the Strix-Halo reference (~14 tok/s), times many denoising passes →
  low-single-digit tok/s even with full support.
- **Doubly blocked in dotLLM today:** (1) no diffusion architecture family, and (2) cross-device offload
  (iGPU+eGPU layer-split / KV-on-other-device) is spec-only. So the "iGPU-only vs iGPU+eGPU" comparison isn't
  constructable yet. Not a near-term run-the-real-model exercise.
- **Decision:** validate the *mechanism* on a small text-diffusion model first (below), and keep DiffusionGemma
  itself as a later target once the diffusion family + offload exist.

---

## Diffusion-LM mechanism spike — COMPLETE ✅

Goal (advisor-scoped): prove dotLLM can be extended to text-diffusion LMs (bidirectional attention + iterative
denoising, no causal KV cache) on a small model, CPU-first, validated vs PyTorch.

> **Scope caveats (read before citing this as "runs on the iGPU"):**
> 1. **CPU only — NOT the iGPU.** The mechanism is proven on the CPU backend. The user's question was about the
>    *iGPU*; running diffusion on Arc still requires a **Vulkan bidirectional-attention shader**, which is
>    **unbuilt**. This spike removes the *architectural* blocker, not the iGPU-execution one.
> 2. **Only the non-tiled attention path is validated.** B0 added the `causal` flag to the straightforward
>    `ExecuteCore` path; the tiled/flash CPU attention path was not exercised bidirectionally and is unverified.

**Target: `open-dcoder-0.5B`** — chosen because its `config.json` is **plain `Qwen2ForCausalLM`** (hidden 896,
24L, 14/2 heads, vocab 151936, tied, bf16). The diffusion is entirely in the inference loop, not the weights,
so dotLLM's **existing Qwen + SafeTensors path loads it with zero new block/loader code.** (GPT-2-block models
like DiffuGPT-S were rejected — would need new LayerNorm/GELU/abs-pos kernels.)

What was built (all on `spike/local-platform-opt`, gated/additive — AR path untouched):

| Piece | What | Commit | Tests |
|-------|------|--------|-------|
| **B0** | Bidirectional attention: `causal` flag on CPU `ExecuteCore` + `Attention.ExecuteBidirectional` | `ed37e73` | 2/2 (vs naive full-softmax; query-0-attends-future discriminator) |
| **B1** | `MaskedDiffusionDenoiser` — absorbing-state mask-predict-unmask, MaskGIT cosine schedule, forbids the mask token | `b25b2c5` + fix | 3/3 (confidence-ordered commits, schedule, guards) |
| wire | `TransformerModel.BidirectionalAttention` flag → no-cache path uses `ExecuteBidirectional` | `039c81b` | — |
| gate | Numeric validation vs PyTorch on real weights | `039c81b` | see below |
| capstone | Full loop (B1 driving B0) generating on real weights | `c2f2b47` | 1/1 |

**Numeric gates vs a PyTorch/transformers reference (`def add(a, b):`, fp32):**
- Loads as Qwen (24L/896/14·2/151936/tied). ✅
- **CAUSAL** forward: argmax **6/6**, ~2.3% logit drift (bf16 + fast-softmax). ✅
- **BIDIRECTIONAL** forward: argmax **6/6** vs the reference's bidirectional logits, ~1.9% drift; position 0
  genuinely differs causal-vs-bidir (mask truly dropped — discriminating). ✅
- **End-to-end**: fully-masked 16-token canvas denoises to concrete non-mask tokens in 16 forward calls. ✅
  (Unconditional output is all-spaces — low quality, as expected without prompt conditioning + open-dcoder's
  temperature/remasking schedule. Mechanism proven; output quality is productionization.)

**Verdict:** the text-diffusion class is **tractable in dotLLM** — no architectural blocker. Reference env at
`C:\Development\dotllm-diffusion-ref\` (venv + `reference.py` + `dump_for_csharp.py` + the model).

**Remaining for a *productionised* diffusion-LM (not done — out of mechanism-spike scope):** prompt conditioning
(block-diffusion: prefill prompt to KV, canvas cross-attends), open-dcoder's temperature schedule (0.8→0.4) +
remasking for output quality, a first-class `IDiffusionScheduler` + `Architecture.DiffusionLM` + GGUF/loader
plumbing, and a Vulkan bidirectional-attention shader to run it on the iGPU. DiffusionGemma additionally needs
the Gemma-4 MoE backbone + cross-device offload.
