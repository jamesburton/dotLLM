# Gemma-4 / DiffusionGemma on the GPU backends — Gap Report

> **SUPERSEDED (2026-07-03).** This report described the state on 2026-06-16, hours before
> the gaps it scopes were closed on the same overnight: Vulkan gained the full gemma4
> forward (dense parity `1f53dca`, MoE fused gate_up Q4_K repack, per-layer-strided KV
> `14f7c61`, diffusion forward + self-conditioning + PKV `7a09756`) and CUDA gained the
> gemma4 AR MoE forward (`0d1f477`) — all merged to `dev` via `41621d5`. Real-weight
> validation: `DiffusionGemmaGgufForwardTests` (CPU) and
> `DiffusionGemmaVulkanRealGenerationTests` (Vulkan). Retained for the op-by-op mapping.

**Status (historical, 2026-06-16):** gemma4 (and diffusion-gemma) run on the **CPU** backend only. Neither the
**Vulkan** nor the **CUDA** backend has a gemma4 forward path. This document scopes exactly
what each GPU backend is missing, mapped to the CPU `RunGemma4Layer` / `Gemma4DenseFfn` /
`Gemma4Moe` ops, and notes for each op whether an existing GPU kernel already covers it or
whether it is new work.

It is the roadmap for bringing gemma4 to the GPU on the T5500 (CUDA) and the Strix Halo /
Radeon 8060S dev host (Vulkan).

---

## 0. What "no gemma4 on GPU" means today (measured)

The portable synthetic `gemma4` fixture (`SyntheticGemma4Gguf`, ~hundreds of KB tiny preset)
config-extracts correctly on every backend — `general.architecture = gemma4` maps to
`Architecture.Gemma4`, `Gemma4DualFfn = true`, a non-null `Moe` config, V-less global layers,
dual head dim, etc. The GGUF metadata mapping is backend-neutral and already done.

The gap is purely the forward/weight path. Probed by
`tests/DotLLM.Tests.Integration/Backends/Gemma4GpuGapProbeTests.cs`:

### Vulkan (measured on the Radeon 8060S, this host)

The probe loads the tiny gemma4 fixture and calls `VulkanTransformerModel.LoadFromGguf`.
It fails **at weight upload, before any forward / kernel runs**:

```
System.NullReferenceException: Object reference not set to an instance of an object.
  at VulkanWeights.UploadExpertBankSlot(... IntPtr srcPtr ...)        VulkanWeights.cs:1392
  at VulkanWeights.UploadMoeLayer(... MoeLayerWeights moe ...)        VulkanWeights.cs:1140
  at VulkanWeights.Upload(... TransformerWeights weights ...)         VulkanWeights.cs:488
  at VulkanTransformerModel.BuildModel(...)                           VulkanTransformerModel.cs:504
  at VulkanTransformerModel.LoadFromGguf(...)                         VulkanTransformerModel.cs:419
```

**Why:** `UploadMoeLayer` expects the standard Mixtral/Qwen-MoE expert layout — separate
per-expert gate (`moe.W1[e]`), up (`moe.W3[e]`), down (`moe.W2[e]`) slots (F32 pointers or
`*ExpsRaw` quantized banks). gemma4 instead stores a **fused `gate_up` expert tensor**
(split by row offset at runtime) plus a separate down bank and a per-expert down **scale**,
loaded into `TransformerWeights.Gemma4` / a gemma4-shaped `MoeLayerWeights`, so the standard
slots the Vulkan uploader copies from are null → NRE on the first `Memmove`.

`RejectUnsupportedArchitecture` does **not** currently reject gemma4 (it only guards Mamba/SSM
and MLA latent-cache), so the failure is an opaque NRE rather than a clear `NotSupported`.
**First recommended step (cheap):** add gemma4 (`config.Gemma4DualFfn`) to
`RejectUnsupportedArchitecture` so the failure is an actionable `NotSupportedException`
until the real path lands. (Owned file — can be done in the follow-up PR that starts the port.)

### CUDA (build-verify only on this host)

No NVIDIA device is present here, so `Cuda_Gemma4_Fixture_FailsWithCapturedGap` skips. The
managed CUDA assembly builds cleanly. On a CUDA box (T5500) the same fixture would fail in
`CudaWeights.LoadFromGguf` / the forward for the same reason — there is no gemma4 dispatch in
`CudaTransformerModel`, and the MoE path assumes the standard expert layout.

---

## 1. Op-by-op gap table

Legend — **Covered**: an existing GPU kernel does this as-is or with parameters already
present. **Adapt**: a near-miss kernel exists, needs a small change/new push-constant/new
glue. **New**: no kernel; net-new work.

| # | gemma4 op (CPU source) | Vulkan | CUDA | Notes / effort |
|---|------------------------|--------|------|----------------|
| 1 | **Attn scale = 1.0** (`QueryPreAttnScalar=1` → `1/sqrt(1)`) | **Covered** | **Covered** | Vulkan attention already takes a `scaleOverride` push-constant; CUDA attention takes a scale arg. Trivial wiring. |
| 2 | **Q-norm / K-norm** (per-head RMS×weight) | **Covered** | **Covered** | Gemma backbone parity (#41, commit `1f53dca`) brought per-head QK-norm to Vulkan; CUDA has it for Qwen3/Gemma. |
| 3 | **Weight-less V-norm** (per-kv-head RMS, **no** scale) | **Adapt** | **Adapt** | Both have RMSNorm kernels but always with a weight vector. Need a weight-less (unit-gamma) per-head variant, or pass a 1-vector. Small new shader / kernel param. |
| 4 | **V-from-K on V-less global layers** (copy raw K proj → V, skip `wv`) | **New (glue)** | **New (glue)** | No kernel needed — it is a buffer copy + skipping the V projection on global layers. Needs per-layer branching in the GPU forward (the V-less layer has a smaller KV-head count and no `attn_v` weight). Layout/scheduling work, not a kernel. |
| 5 | **Partial NeoX RoPE (0.25)** on global layers (rotate only leading `ropeDim` dims; rotate-half pairing at FULL head half-dim) | **Covered** | **Adapt** | Vulkan `rope_f32.comp` already has a `ropeDim` push-constant (rotates the first `ropeDim`, passes the rest through) — matches partial rope directly. Verify the rotate-half **pairing offset** (full head half-dim, not `ropeDim/2`) matches `RoPE.ExecutePartialNeoX`; if the shader pairs within `ropeDim` it needs a pairing-stride param. CUDA rope needs the same partial param checked. |
| 6 | **Dual RoPE tables** (sliding base 10k, global base 1M, dual head dim) | **Covered** | **Covered** | Per-layer rope table/dim selection already exists for Gemma's sliding/global split. |
| 7 | **Sliding-window + hybrid per-layer mask** (sliding layers windowed, global full) | **Covered** | **Covered** | Per-layer sliding-window already supported in both attention paths (Gemma-2/3). |
| 8 | **Non-causal (bidirectional / hybrid) attention mask** — required by diffusion-gemma canvas | **Partial** | **New** | **Vulkan:** the non-causal attention from **#41 (commit `7f8bf5b`)** that matches the CPU `Bidirectional` / `Hybrid(prefixLen)` modes is on the **`dg-pr12-vulkan-gemma`** branch, **NOT** on this branch. On this branch `VulkanTransformerModel` inherits the `IModel` default and **throws `NotSupportedException` for any non-causal mask**. So for diffusion-gemma we need to land `7f8bf5b` (or its equivalent). For **autoregressive gemma4**, attention stays causal and this is not blocking. **CUDA:** has no bidirectional/hybrid attention at all — net-new. |
| 9 | **Attention softcap = 0** (gemma4 attn is NOT softcapped; final-logit softcap IS) | **Covered** | **Covered** | Vulkan attention `softCap` push-constant + host final-logit softcap (`ApplyFinalLogitSoftcapHost`) already present; CUDA equivalent exists. |
| 10 | **Dual parallel dense + MoE FFN** (run BOTH branches, RMS-norm each, sum, RMS-norm, residual) | **New (graph)** | **New (graph)** | No backend runs a dense ("shared expert") FFN **in parallel with** a routed MoE and then sums two separately-normed branches. DeepSeek's shared-expert is **additive into** the MoE output, not a separately post-normed parallel branch. This is the core new forward-graph wiring: two FFN sub-graphs + `post_ffw_norm_1/_2/_(combined)` + `layer_output_scale`. Kernels (matmul, geglu-tanh, rmsnorm) all exist; the **graph** is new. |
| 11 | **GeGLU-tanh** dense activation | **Covered** | **Covered** | gelu/geglu activation kernels exist on both. |
| 12 | **Custom router** (`logits = ffn_gate_inp · (rms(x) · 1/sqrt(n_embd) · ffn_gate_inp_s)`) | **Adapt** | **Adapt** | The router input is a *scaled* RMS (`1/sqrt(hidden)` × per-channel `gate_inp_s`) feeding a GEMV → softmax → top-k. The GEMV (`matmul_*`/`moe_topk_softmax_f32`) and softmax/top-k kernels exist; the **input scaling** (`invSqrtH * RouterScale[j]`) is a new elementwise pre-step, and the top-k **renorm with the 6.1e-5 min-clamp** must match. Small new elementwise + a renorm tweak. |
| 13 | **Fused `gate_up` expert split** (one bank holds `[gate ; up]` interleaved by row; split at offset) + **down bank** | **New** | **Adapt** | The expert weight is a single fused `gate_up` tensor sliced into gate/up halves per expert, distinct from the separate W1/W3 banks both backends assume (this is the exact NRE in §0). Needs gemma4-aware weight upload (fused-bank slicing) **and** the indexed/grouped expert matmul to read gate+up from one bank at a row offset. CUDA's grouped GEMV is closer (it already handles per-expert banks); Vulkan needs both the upload fix and the kernel offsetting. |
| 14 | **Per-expert down scale** (`ffn_down_exps.scale[e]` folded into routing weight) | **Covered** | **Covered** | CPU folds it into the routing weight (`w[e] * DownExpertScale[e]`). On GPU it is one extra per-expert scalar multiply into the weighted scatter — both have weighted-scatter/`moe_axpy_scaled_row_f32`-style kernels. Trivial once the scale array is uploaded. |
| 15 | **MoE expert matmul in Q4_K (gate_up) + Q5_1 (down)** | **New (Q4_K + Q5_1)** | **Adapt** | **Vulkan:** `moe_indexed_matmul_*` exists only for **F32, Q8_0, Q6_K** — **no Q4_K and no Q5_1 indexed-MoE kernel**, and there is **no Q5_0/Q5_1/Q4_0/Q4_1 dequant shader at all** (only Q4_K/Q5_K/Q6_K/Q2_K/Q3_K + IQ*). So gemma4's default expert quant mix needs a new Q4_K MoE-indexed-matmul **and** Q5_1 support (dequant + indexed matmul), or up-convert experts to a supported type at load. **CUDA:** has quantized grouped-GEMV (MMQ) covering more types; verify Q4_K/Q5_1 grouped paths, otherwise dequant-on-load. |
| 16 | **`layer_output_scale`** (final per-layer multiply) | **Covered** | **Covered** | One elementwise scalar multiply at the end of the layer; both have scale kernels. For diffusion-gemma the **region split** (`enc_layer_output_scale` on prompt rows `[0,P)`, `layer_output_scale` on canvas rows `[P,seqLen)`) needs the prefix length P passed in — small per-row branch. |
| 17 | **Final-logit softcap (30)** | **Covered** | **Covered** | Already applied (Vulkan host-side `ApplyFinalLogitSoftcapHost`; CUDA tanh-cap). |
| 18 | **Region-aware diffusion embed** (canvas rows get an EXTRA weight-less RMS on the scaled embedding) — diffusion-gemma only | **New** | **New** | A region (prefix-length) split applied to the **input embedding** before layer 0: prompt rows keep `scaled_embed`, canvas rows get `rms_noscale(scaled_embed)`. New elementwise pre-pass parameterised by P; no heavy kernel. |
| 19 | **Self-conditioning** (canvas region: `rms_noscale(scaled_embed + sc_sig)`, `sc_sig` = gated GeGLU MLP over a soft-embedding of the previous step's logits) — diffusion-gemma only | **New** | **New** | A whole extra small MLP (soft token-embedding sweep over vocab + GeGLU + gate) feeding the canvas embedding, driven by `SetDiffusionSelfCond`. New sub-graph; matmul/geglu kernels exist, the **soft-embed sweep + gating + state plumbing** is new. Only needed for diffusion-gemma generation quality, not for AR gemma4. |
| 20 | **PKV decode mask** (canvas queries attend `[cached prompt K/V | fresh canvas K/V]` under a rectangular bidirectional mask, sliding-window clipped, `positionOffset = P`) — diffusion-gemma throughput opt | **New** | **New** | Depends on op #8 (non-causal attention) + a prompt-KV store on device. This is the GPU analogue of `DiffusionPrefillPromptKv` / `DiffusionDecodeWithPromptKv` (`SupportsDiffusionPromptKv`). Pure optimisation — the cacheless unified `[prompt|canvas]` forward (also needing #8) is the simpler first target. |

---

## 2. What #41's Vulkan work already gives us

- **`1f53dca` (Gemma backbone parity, #41/#35):** dense Gemma forward on Vulkan — per-head
  QK-norm (#2), per-layer sliding window (#7), attention `softCap` + `scaleOverride`
  push-constants (#1, #9), partial-`ropeDim` RoPE (#5), dual rope tables (#6). These are the
  *attention-side* gemma4 ops and are largely **already covered** on Vulkan.
- **`7f8bf5b` (non-causal attention, #41) — on `dg-pr12-vulkan-gemma`, NOT this branch:**
  the bidirectional / hybrid attention modes that match the CPU `AttentionMaskSpec`. This is
  the prerequisite for **diffusion-gemma** canvas attention (#8) and the PKV decode mask
  (#20). It must be merged forward before diffusion-gemma can run on Vulkan. Autoregressive
  gemma4 does **not** need it (stays causal).
- **Existing Vulkan MoE plumbing:** `moe_topk_softmax_f32`, `moe_indexed_matmul_{f32,q8_0,q6_k}`,
  `moe_weighted_scatter_f32`, `moe_sigmoid_gated_add_f32`, grouping/scatter shaders — the
  router→experts→scatter skeleton (#12, #14) is mostly present. The **missing pieces** are the
  Q4_K + Q5_1 indexed-expert matmul (#15), the fused `gate_up` bank slicing (#13), and the
  gemma4 weight-upload path (#0/§3).

---

## 3. Critical path / recommended sequencing

**Autoregressive gemma4 on Vulkan (does NOT need diffusion or non-causal attention):**

1. **Weight loading (blocker, #0/#13):** gemma4-aware `VulkanWeights` upload — fused `gate_up`
   bank slicing, separate down bank, per-expert down-scale array, V-less global layers (no
   `attn_v`). Until this lands, anything else NREs at load. Add gemma4 to
   `RejectUnsupportedArchitecture` first so the gap is a clean `NotSupported`.
2. **Forward graph (#10):** dual parallel dense + MoE FFN with the three `post_ffw_norm`s and
   `layer_output_scale`.
3. **Custom router (#12) + per-expert down scale (#14):** scaled-RMS router input + renorm
   clamp; fold down-scale into the scatter.
4. **Q4_K / Q5_1 expert matmul (#15):** the only genuinely new *kernel* work — a Q4_K
   MoE-indexed matmul and Q5_1 support, or up-convert experts at load (Q8_0/F32) as a first
   correctness milestone, then add the quantized kernels for memory/perf.
5. **V-from-K + weight-less V-norm (#3, #4):** per-layer branching for the global V-less layers.

Most attention-side ops (#1, #2, #5–#7, #9, #16, #17) are already covered by the #41 backbone.

**DiffusionGemma on Vulkan** additionally needs: forward-merge `7f8bf5b` (#8), region-aware
embed (#18), self-conditioning (#19), and finally the PKV decode opt (#20).

**CUDA (T5500):** same op list. CUDA's quantized grouped-GEMV MoE is a closer starting point
for #13/#15, but it has **no** bidirectional/hybrid attention (#8) at all, so diffusion-gemma
is further out on CUDA than on Vulkan. AR gemma4 on CUDA is the realistic first CUDA target.

---

## 4. Reproducing the probe

```
# Build (0 errors; CUDA managed side builds without an NVIDIA device):
dotnet build -c Release

# Vulkan gemma4 gap (runs on a Vulkan device; captures the load-time failure):
dotnet test tests/DotLLM.Tests.Integration -c Release \
  --filter "FullyQualifiedName~Gemma4GpuGapProbeTests"

# Prove the cross-backend timing harness on a SUPPORTED arch on the real GPU:
dotnet test tests/DotLLM.Tests.Integration -c Release \
  --filter "FullyQualifiedName~VulkanCrossBackendTimingDemoTests"
```

The harness (`tests/DotLLM.Tests.Integration/Backends/CrossBackendTimingHarness.cs`) loads any
GGUF on CPU / Vulkan / CUDA (graceful skip when a device is absent) and emits per-phase CSV
(`phase,name,ms,tokens_per_sec`) — the same shape as the CPU `SyntheticGemma4Harness` — so once
gemma4 runs on a GPU backend, the same harness times it head-to-head against the CPU path with
no new plumbing.
