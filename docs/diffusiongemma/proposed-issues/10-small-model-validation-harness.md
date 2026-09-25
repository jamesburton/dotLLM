# 10 — Small-model validation harness (DiffuGPT-S end-to-end + logit parity)

**Effort: M**

## Summary / Motivation
We need a fast, CPU-runnable harness that loads the smallest masked-diffusion LM, runs the dotLLM
diffusion decode path, and verifies numerical correctness against the HF reference — so every
milestone can be regression-checked without the 26B model or the GPU.

## Scope
- A loader path for the primary validation model
  [`diffusionfamily/diffugpt-s`](https://huggingface.co/diffusionfamily/diffugpt-s) (124M, GPT-2
  backbone, absorbing-state masked diffusion, safetensors F16, Apache-2.0). GPT-2 backbone reuses the
  dense `TransformerModel`; only the bidirectional mask + denoise loop are diffusion-specific.
- A harness that: (a) runs a single bidirectional forward over a fixed prompt+mask pattern and
  compares canvas logits to a captured HF-reference dump (cosine sim / max-abs-diff), and (b) runs the
  full denoise loop and compares decoded text to the HF reference on a fixed prompt set.
- Reference dumps captured offline (committed as small fixtures); CI runs CPU-only.

## Acceptance Criteria
- [ ] DiffuGPT-S loads and runs the dotLLM diffusion decode path on CPU.
- [ ] Single-forward canvas-logit parity vs HF reference: cosine ≥ 0.999 (document tolerance).
- [ ] Full-denoise decoded text matches the reference within a stated edit-distance/quality bound.
- [ ] Harness is a runnable sample/test, GPU-free.
- [ ] Documents how to obtain weights + capture the reference dump.

## Dependencies
- Blocks on **04, 06, 08, 09** (decode path). Independent of **01–03, 07** (GPT-2 backbone, not Gemma).

## References
- model: <https://huggingface.co/diffusionfamily/diffugpt-s> (DiffuLLaMA, arXiv 2410.17891)
- `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelGemma3ForwardTests.cs` (forward-test harness pattern)
- See [VALIDATION.md](../VALIDATION.md)
