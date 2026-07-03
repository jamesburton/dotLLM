# DiffusionGemma — Decode Loop Architecture Note

> **Status (2026-07-03): implemented as designed.** This note was written during the planning
> handoff (originally `docs/HANDOFF.md`, 2026-06-18) and its prescription landed as
> `DiffusionTextGenerator` (#28, commit `6230300`, merged to `dev` via `41621d5`), with the
> canvas/scheduler/sampler split it calls for (#27/#30/#31) and the mask-mode seam it warns
> about (`DiffusionConfig.CanvasAttentionMode`, `821a055`). Retained for the architectural
> rationale.

## Decode Loop Architecture Note

Do not route DiffusionGemma through the existing append-only `TextGenerator` as if it were a normal autoregressive model.

DiffusionGemma-style text generation is a regenerative masked-token process over a fixed-size canvas, not simple next-token append:

- maintain a mutable output token canvas, with prompt/context separated from masked generation positions;
- run iterative denoise passes over the canvas;
- update many token positions per step using confidence/entropy or scheduler policy;
- allow previously masked/generated positions to be replaced or remasked during denoising;
- use bidirectional or hybrid attention masks for the canvas, while still supporting AR-style prompt context where required;
- avoid assuming the standard autoregressive KV-cache contract, because generated canvas tokens are not immutable once produced.

Implementation implication: add a parallel generator abstraction, for example `DiffusionTextGenerator` / `MaskedDenoisingGenerator`, rather than stretching `TextGenerator.GenerateStreamingTokensAsync` around mutable-token regeneration. Shared pieces should remain tokenizer/model loading/logit processors where applicable, but the decode state, scheduler, canvas mutation, streaming semantics, and validation harness need to be separate.

CLI/API implication: expose this as a distinct mode or auto-select it from model metadata such as `model_type=diffusion_gemma` / `architectures=["DiffusionGemmaForBlockDiffusion"]`. A dedicated command or server path should report denoise steps, canvas latency, and effective tokens/sec separately from AR decode tokens/sec.
