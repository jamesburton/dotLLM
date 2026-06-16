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

## Status log
- [done] PR-1..PR-10 (issues #23-#34) implemented, CPU-validated, pushed (branches dg-pr1..dg-pr8). Synthetic end-to-end diffusion generation + server routing works.
- [done] Follow-ups filed: #35 (Vulkan Gemma parity), #36 (per-layer head_dim), #37 (KV-prefix reuse).
- [in progress] LLaDA-8B Q4_K_M download → C:\models\llada\.
- [in progress] GGUF→diffusion wiring on branch `dg-pr9-llada-validation`.

## Branch stack
dev-diffusiongemma → dg-pr1-gemma-backbone → dg-pr2-gemma4-moe → dg-pr3-bidirectional-attn →
dg-pr4-diffusion-config → dg-pr5-denoise-scheduler-sampler → dg-pr6-diffusion-text-generator →
dg-pr7-diffusiongemma-model-loader → dg-pr8-server-diffusion → dg-pr9-llada-validation (current)
