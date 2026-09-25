# 01 — Gemma backbone numerical correctness (embedding scale, 4-norm, GeGLU, QK-norm, (1+w))

**Effort: L**

## Summary / Motivation
A partial `Architecture.Gemma3` exists on dev but is *mechanism-only*: the attention-side features
(per-layer sliding window, softcaps, QPAS) are wired and tested, but the **FFN still runs SwiGLU**,
the loader uses the **2-norm Llama layout**, and there is **no embedding scaling, no Gemma 4-norm
layout, no per-head QK-norm loading, and no `(1+w)` RMSNorm absorption**. DiffusionGemma is a Gemma-4
derivative, so a numerically-correct Gemma forward is the foundation for everything else.

## Scope
- Embedding scaling: multiply input embeddings by `sqrt(hidden_size)` for Gemma architectures.
- Wire **GeGLU** (`FusedOps.GeGLUTanh`) into the dense FFN forward when
  `Config.ActivationFunction == GELUTanh` (kernel already exists, just not called).
- Gemma **4-norm layout** in the forward residual structure: `input_layernorm`,
  `post_attention_layernorm` (between sublayer out and residual add), `pre_feedforward_layernorm`,
  `post_feedforward_layernorm`.
- `(1+w)` RMSNorm absorption at load (pre-add 1.0 to every Gemma RMSNorm weight) so the existing
  kernel runs unchanged.
- Load + apply per-head Q/K RMSNorm (slots already exist).
- New `ModelConfig.EmbeddingScale` (or arch-derived) field + new norm-weight slots on `TransformerWeights`.

## Acceptance Criteria
- [ ] Input embeddings scaled by `sqrt(hidden_size)` for Gemma; unchanged for non-Gemma.
- [ ] GeGLU used in FFN for `GELUTanh` configs; SwiGLU path untouched for SiLU configs.
- [ ] 4 RMSNorms per layer applied in the correct residual positions.
- [ ] RMSNorm weights load with `(1+w)` absorption for Gemma.
- [ ] Per-head QK-norm loaded + applied.
- [ ] A synthetic Gemma-3 forward test extended to assert GeGLU + 4-norm + embed-scale behaviour
      (discriminative vs the SwiGLU/2-norm baseline), replacing the current mechanism-only fixture note.
- [ ] No regression in existing non-Gemma forward tests.

## Dependencies
None (foundational).

## References (dev, file:line)
- `src/DotLLM.Models/Architectures/TransformerModel.cs:974-997,1511-1539` (SwiGLU-only FFN today)
- `src/DotLLM.Models/Architectures/TransformerModel.cs:734-737,1428-1431` (QK-norm apply, not loaded for Gemma)
- `src/DotLLM.Cpu/Kernels/FusedOps.cs:95-211` (GeGLU kernel, unwired)
- `src/DotLLM.Models/Architectures/TransformerWeightsSafetensors.cs:116-120,234-239` (2-norm loader)
- `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelGemma3ForwardTests.cs:297-301,316-321,339` (notes FFN=SwiGLU, 2-norm)
- `src/DotLLM.Core/Models/ModelConfig.cs:48-49` (ActivationFunction default)
