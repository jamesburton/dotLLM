# 07 — DiffusionGemma model class + config extractor + loader dispatch

**Effort: L**

## Summary / Motivation
Tie the Gemma-4 MoE backbone (01–03) and the diffusion seam (04–06) into a concrete
`DiffusionGemmaModel` loadable from the real `google/diffusiongemma-26B-A4B-it` checkpoint
(`model_type=diffusion_gemma`, `architectures=["DiffusionGemmaForBlockDiffusion"]`).

## Scope
- `DiffusionGemmaConfigExtractor`: parse top-level `diffusion_gemma` config — hoist `text_config`
  (like the Gemma3 `text_config` hoist), read `canvas_length`, build the `DiffusionConfig` (issue 05),
  the Gemma-4 MoE `ModelConfig` (issue 02), and skip the `vision_config` (text-only first; multimodal
  out of scope).
- `Architecture.DiffusionGemma` enum entry.
- `DiffusionGemmaModel : IModel`: Gemma-4 MoE forward (reusing the dense/MoE `TransformerModel`
  internals) but exposing a forward that accepts the hybrid mask mode for the canvas.
- `ModelLoader` dispatch for `diffusion_gemma`/`diffusion_gemma_text` → extractor + model.
- Resolve mask token id from the checkpoint's tokenizer files.

## Acceptance Criteria
- [ ] `Architecture.DiffusionGemma` enum + XML doc.
- [ ] Config extractor produces a populated `ModelConfig` (Gemma-4 MoE + `DiffusionConfig`) from the
      real `config.json` (parsed offline, no download in CI).
- [ ] `ModelLoader.LoadFromSafetensors` returns a `DiffusionGemmaModel` for the diffusion model_type.
- [ ] Mask token id resolved from tokenizer metadata.
- [ ] Forward exposes the hybrid-mask canvas path used by `DiffusionTextGenerator`.
- [ ] Integration test loading a tiny synthetic diffusion-gemma fixture end-to-end.

## Dependencies
- Blocks on **01, 02, 03, 04, 05**. Consumed by **06** (for real weights) and **10/11** (validation).

## References (dev, file:line)
- `src/DotLLM.Models/ModelLoader.cs:77-89,136-151` (dispatch + Mamba3 model_type probe template)
- `src/DotLLM.Models/Architectures/Mamba3ConfigExtractor.cs`, `Mamba3TransformerModel.cs` (extractor+model template)
- `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs:58-69` (`text_config` hoist pattern)
- verified: `architectures=["DiffusionGemmaForBlockDiffusion"]`, `model_type=diffusion_gemma`
