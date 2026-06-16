# DiffusionGemma — overnight autonomous run progress

Durable state log for the overnight build (survives context resets). Updated as work lands.

## Environment constraints (discovered)
- **Disk: ~20 GB free on C:** (only drive). Hard limit on model downloads.
- No HF CLI / `huggingface_hub`; downloads via `curl` from `huggingface.co/.../resolve/main/...`.
- GPU: **AMD Radeon 8060S (Strix Halo, gfx1151, RDNA3.5)**. **Vulkan present, no ROCm** → Vulkan is the GPU path.
- Diffusion loaders so far are **safetensors-only**; affordable real models are **GGUF** → GGUF→diffusion wiring is the key enabler.

## Local models already present (no download)
- `unsloth/gemma-4-26B-A4B-it-GGUF`: **gemma-4-26B-A4B-it-UD-Q4_K_M.gguf (15.78 GB)** — the **AR backbone of DiffusionGemma** (Gemma-4 26B-A4B MoE). + mmproj 1.11 GB.
- gemma-4-E2B-it: metadata-only stubs (0 bytes) — not usable.
- No LLaDA, no DiffusionGemma weights locally.

## Download options (metadata-checked)
- **LLaDA-8B GGUF** (mradermacher): Q2_K 3.2G … **Q4_K_M 4.93G** … (fits). Llama backbone → loads via existing GGUF llama path + injected DiffusionConfig (mask token **126336**).
- **DiffusionGemma GGUF** (unsloth): smallest **Q4_K_M 16.81 GB** (fits only after freeing the LLaDA GGUF; needs GGUF Gemma-4 loading + #36).

## Plan
1. **LLaDA-8B GGUF real diffusion validation** (matches "validate first on LLaDA"). Download Q4_K_M (in progress) → GGUF llama load + DiffusionConfig inject + route to DiffusionTextGenerator → real masked-diffusion generation + coherence + throughput. (CPU then Vulkan.)
2. **#36** per-layer head_dim (CPU) → unblock Gemma-4 26B forward.
3. **GGUF Gemma-4 loader** → validate the local 26B-A4B Q4_K_M backbone (real weights, no download).
4. **DiffusionGemma**: swap LLaDA GGUF for the diffusiongemma Q4_K_M (disk), GGUF diffusion routing → real DiffusionGemma generation.
5. **Vulkan competitive** (#35): Gemma four-norm/GeGLU/embed-scale Vulkan kernels + GPU forward + throughput vs CPU.

## KEY FINDING — mask mode is model-specific (validated on real LLaDA-8B)
A fast single-forward diagnostic on real LLaDA-8B Q4 (prompt "The capital of France is" + 8 mask
positions) showed:
- **Bidirectional** mask → first masked position argmax = **" Paris"** → `Paris.` then EOS padding. ✓ correct.
- **Hybrid** (causal prompt) → all positions = EOS. ✗ degenerate (this caused the 0-token failure).

So **LLaDA needs fully Bidirectional attention** over [prompt|canvas]; **DiffusionGemma is block-AR
(Hybrid)**. Added `DiffusionConfig.CanvasAttentionMode` (default Hybrid; LLaDA sets Bidirectional);
`DiffusionTextGenerator.CanvasMaskSpec` picks the spec. The diffusion decode machinery is **confirmed
correct on real masked-diffusion weights**. (Perf: ~33 s / forward for 8B Q4 on CPU — recompute-per-step
is the bottleneck → #37; GPU diffusion forward is future Vulkan work.)

## ✅ LLaDA-8B REAL VALIDATION PASSED
`The capital of France is` → **`Paris.`** (decoded `Paris.<|eot_id|>`), 16 adaptive denoise steps,
CanvasAttentionMode=Bidirectional, canvas 16. Proves the full masked-diffusion decode (bidirectional
attention + entropy-bound unmask + scheduler + GGUF Llama load) is correct on real weights.
Perf: load 3.8 s; gen 1003 s (~16.7 min) — recompute-per-step 8B CPU is the bottleneck (~0.003 tok/s);
GPU/Vulkan diffusion forward + KV-prefix reuse (#37) are required for usable throughput.

## Status log
- [done] PR-1..PR-10 (issues #23-#34) implemented, CPU-validated, pushed (branches dg-pr1..dg-pr8). Synthetic end-to-end diffusion generation + server routing works.
- [done] Follow-ups filed: #35 (Vulkan Gemma parity), #36 (per-layer head_dim), #37 (KV-prefix reuse).
- [in progress] LLaDA-8B Q4_K_M download → C:\models\llada\.
- [in progress] GGUF→diffusion wiring on branch `dg-pr9-llada-validation`.

## Branch stack
dev-diffusiongemma → dg-pr1-gemma-backbone → dg-pr2-gemma4-moe → dg-pr3-bidirectional-attn →
dg-pr4-diffusion-config → dg-pr5-denoise-scheduler-sampler → dg-pr6-diffusion-text-generator →
dg-pr7-diffusiongemma-model-loader → dg-pr8-server-diffusion → dg-pr9-llada-validation (current)
