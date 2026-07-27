# 05 — Diffusion config + mask-token plumbing

**Effort: S**

## Summary / Motivation
The diffusion decode loop and sampler need diffusion-specific configuration that does not exist:
canvas length, max denoising steps, entropy bounds, temperature schedule, and the **mask token id**.
The mask token id is **not** in `config.json`/`generation_config.json` — it must be resolved from the
tokenizer / special-tokens map at load time and must not be hardcoded.

## Scope
- New `DiffusionConfig` record: `CanvasLength` (256), `MaxDenoisingSteps` (48), `EntropyBound` (0.1),
  `ConfidenceThreshold`/entropy stop (0.005), `StabilityThreshold` (1), temperature schedule
  (`TMax=0.8`, `TMin=0.4`), `MaskTokenId`.
- Resolve `MaskTokenId` from `tokenizer_config.json` / `special_tokens_map.json` / reference modelling
  code (fail loudly if unresolved). Surface on `ModelConfig.DiffusionConfig`.
- Read the diffusion fields from `generation_config.json` with the verified defaults above.

## Acceptance Criteria
- [ ] `DiffusionConfig` populated from `config.json` + `generation_config.json` with verified defaults.
- [ ] `MaskTokenId` resolved from tokenizer metadata, never hardcoded; load fails with a clear message
      if it cannot be resolved.
- [ ] Unit test parses a fixture config and asserts every field.

## Dependencies
- None to start; consumed by **06, 07, 08, 09**.

## References (dev, file:line)
- verified `generation_config.json`: `max_denoising_steps=48`, `EntropyBoundSamplerConfig`,
  `entropy_bound=0.1`, `confidence_threshold=0.005`, `stability_threshold=1`, `t_min=0.4`, `t_max=0.8`
- verified `config.json`: `canvas_length=256`
- `src/DotLLM.Core/Models/ModelConfig.cs` (add `DiffusionConfig?` slot)
- `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs` (extraction pattern)
