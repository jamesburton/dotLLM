# 03 — SafeTensors loader: Gemma 3/4 tensor names + MoE + diffusion_gemma dispatch

**Effort: M**

## Summary / Motivation
The SafeTensors loader currently only loads the 2-norm Llama layout and `ModelLoader` throws
`NotSupportedException` for any architecture outside its dispatch list. To load real Gemma/
DiffusionGemma checkpoints we need the Gemma tensor-name set (4 norms, QK-norms) and dispatch for
`model_type=diffusion_gemma` / `diffusion_gemma_text`.

## Scope
- Add Gemma tensor names to the per-layer loader: `pre_feedforward_layernorm.weight`,
  `post_feedforward_layernorm.weight`, `self_attn.q_norm.weight`, `self_attn.k_norm.weight`
  (in addition to the existing `input_layernorm` / `post_attention_layernorm`).
- Load Gemma-4 MoE expert tensors (router gate + per-expert gate/up/down) via the existing MoE
  loader, honouring `moe_intermediate_size`.
- `ModelLoader.LoadFromSafetensors` dispatch: route `Architecture.Gemma3`/`Gemma4` to the dense/MoE
  `TransformerModel` path; reserve `diffusion_gemma` for the diffusion model (issue 07).
- `OpenSafetensorsAndConfig`: route `model_type=diffusion_gemma`/`diffusion_gemma_text` to the
  diffusion config extractor (issue 07) like the existing Mamba3 probe.

## Acceptance Criteria
- [ ] Loader resolves Gemma 4-norm + QK-norm tensors when present; falls back cleanly when absent.
- [ ] `ModelLoader` dispatches `Gemma3`/`Gemma4` without throwing.
- [ ] `diffusion_gemma`/`diffusion_gemma_text` routes to the diffusion extractor/model (stubbed until 07).
- [ ] Multi-shard index + tied-embedding handling verified for a Gemma-shaped fixture.
- [ ] Loader unit test on a synthetic Gemma-4 safetensors fixture.

## Dependencies
- Blocks on **01**, **02** (needs the new weight slots + config).

## References (dev, file:line)
- `src/DotLLM.Models/ModelLoader.cs:77-89` (dispatch; throws today)
- `src/DotLLM.Models/ModelLoader.cs:136-151` (Mamba3 probe pattern for model_type routing)
- `src/DotLLM.Models/Architectures/TransformerWeightsSafetensors.cs:116-120,230-239` (2-norm loader)
- `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs:605-655` (`ResolveArchitecture` switch)
