# 02 — Gemma 4 MoE architecture enum + config + RoPE-per-attention-type

**Effort: M**

## Summary / Motivation
DiffusionGemma's backbone is **Gemma 4 26B-A4B MoE**, which differs from the existing Gemma 3 path:
sparse MoE FFN (128 experts, top-8), `gelu_pytorch_tanh` **GeGLU experts** (current MoE experts are
SwiGLU-only), `num_global_key_value_heads` (a *different* KV-head count on full-attention layers),
and **partial-rotary RoPE** that differs per attention type. We add the Gemma-4 family
representation so the diffusion model (issue 07) can sit on a correct dense backbone.

## Scope
- Add `Architecture.Gemma4` (and treat `Gemma4Moe` as the same enum w/ `Moe != null`), documented
  like the other enum entries.
- Extend `ModelConfig`/`MoeConfig` consumption for: GeGLU experts, `moe_intermediate_size=704`,
  `num_experts=128`, `top_k_experts=8`.
- Per-attention-type RoPE: full-attn layers use theta 1e6 + `partial_rotary_factor=0.25` +
  `rope_type=proportional`; sliding layers use theta 1e4 default. Surface partial-rotary + per-type theta.
- `num_global_key_value_heads=2` for full-attention layers vs `num_key_value_heads=8` for sliding.
- Make the MoE expert MLP honour `ActivationFunction` (GeGLU) instead of hardcoded SwiGLU.

## Acceptance Criteria
- [ ] `Architecture.Gemma4` enum entry with XML doc covering the deltas vs Gemma 3.
- [ ] Config carries per-attention-type RoPE params (theta + partial-rotary-factor + type).
- [ ] Config carries the dual KV-head counts (global vs sliding).
- [ ] MoE expert path supports GeGLU activation (gated by `ActivationFunction`).
- [ ] Unit test: a synthetic Gemma-4-MoE forward produces finite, non-degenerate `[seq, vocab]` logits.

## Dependencies
- Blocks on **01** (Gemma backbone correctness — GeGLU/4-norm/embed-scale reused by experts/dense).

## References (dev, file:line)
- `src/DotLLM.Core/Configuration/Architecture.cs:147-194` (Gemma3 enum to extend)
- `src/DotLLM.Core/Models/MoeConfig.cs:36-141` (MoE config; experts SwiGLU today)
- `src/DotLLM.Models/Architectures/TransformerModel.cs:832-863` (MoE forward, SwiGLU experts)
- `src/DotLLM.Core/PositionEncoding/RoPEConfig.cs` (single-theta today; needs per-attn-type)
- verified config: `text_config.rope_parameters`, `num_global_key_value_heads`, `num_experts`, `top_k_experts`
