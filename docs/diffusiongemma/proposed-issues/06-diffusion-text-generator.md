# 06 — DiffusionTextGenerator (iterative denoise decode loop)

**Effort: L**

## Summary / Motivation
`TextGenerator` is an autoregressive prefill+decode loop and cannot express masked-diffusion
generation (parallel multi-position canvas, iterative denoise, remask). We add a parallel
`DiffusionTextGenerator` that drives the encoder-prefill → canvas-init → iterative-denoise loop.

## Scope
- New `DiffusionTextGenerator`: encode prompt → AR prefill into KV cache (causal) → initialise a
  fully-masked canvas of `CanvasLength` → loop `MaxDenoisingSteps`: forward the canvas bidirectionally
  (hybrid mask, cross-attending the cached prompt), apply the unmasking sampler (issue 09) to unmask a
  subset, remask the rest (issue 08), early-stop on the entropy threshold.
- Block-autoregressive multi-canvas for sequences longer than one canvas (carry finished canvas into
  the cache, start a new one).
- Streaming hook analogous to HF `TextDiffusionStreamer` (emit intermediate canvas states).
- Reuse the existing KV-cache for the prompt prefix; canvas KV is recomputed per step (no causal cache
  for the bidirectional block) — document the memory/throughput tradeoff.

## Acceptance Criteria
- [ ] `DiffusionTextGenerator.Generate` runs prefill + N denoise steps and returns decoded text.
- [ ] Canvas starts all-mask; each step strictly reduces the masked count (monotone) until stop.
- [ ] Early stop fires when average canvas entropy < threshold.
- [ ] Multi-canvas: a >256-token request produces ≥2 canvases block-autoregressively.
- [ ] Streaming callback observes intermediate canvases.
- [ ] Unit test on the small validation model end-to-end on CPU (no GPU).

## Dependencies
- Blocks on **04** (hybrid mask), **05** (diffusion config + mask token), **08** (scheduler), **09** (sampler).
- For real weights also **07**.

## References (dev, file:line)
- `src/DotLLM.Engine/TextGenerator.cs:26-85` (AR generator to parallel)
- `src/DotLLM.Core/Models/IModel.cs:25,53` (all-position logits already returned)
- `src/DotLLM.Engine/KvCache/SimpleKvCache.cs` (prompt prefix cache reuse)
- DiffusionGemma docs: encoder-prefill + bidirectional canvas decode, multi-canvas sampling
