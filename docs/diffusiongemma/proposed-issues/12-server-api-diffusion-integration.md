# 12 — Server / API integration for the diffusion decode path

**Effort: M**

## Summary / Motivation
Expose DiffusionGemma through the existing OpenAI-compatible server so it is usable end-to-end. The
server's generation path assumes the AR `TextGenerator`; it needs to dispatch to
`DiffusionTextGenerator` for diffusion models and surface diffusion-specific options.

## Scope
- Detect diffusion models (`DiffusionConfig != null`) and route generation through
  `DiffusionTextGenerator` instead of `TextGenerator`.
- Map streaming to the canvas-streaming hook (emit intermediate canvases as SSE deltas, analogous to
  HF `TextDiffusionStreamer`).
- Surface diffusion options (canvas length, max denoise steps, temperature schedule) via request
  params with the verified defaults; keep AR requests unchanged.
- Warm-up: include a diffusion warm-up canvas in the server warm-up path.

## Acceptance Criteria
- [ ] Diffusion models route to `DiffusionTextGenerator` automatically.
- [ ] `/v1/chat/completions` (stream + non-stream) works against the small validation model on CPU.
- [ ] Diffusion request params plumbed with verified defaults; AR path untouched.
- [ ] Warm-up runs a diffusion canvas without GPU when none is available.
- [ ] Server integration test on the small model.

## Dependencies
- Blocks on **06, 07** (decode loop + model). Independent of **11**.

## References (dev, file:line)
- `src/DotLLM.Engine/TextGenerator.cs` (AR path the server uses today)
- `src/DotLLM.Server/` (endpoints, streaming, warm-up)
- DiffusionGemma docs: `TextDiffusionStreamer`, `model.generate(max_new_tokens=...)`
