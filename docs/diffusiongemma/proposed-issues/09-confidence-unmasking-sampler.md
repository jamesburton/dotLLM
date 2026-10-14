# 09 — Parallel confidence/entropy-based unmasking sampler

**Effort: M**

## Summary / Motivation
`ISamplerStep` operates per-position on a single token's logits. Diffusion decode needs a
**multi-position** sampler that, given logits for all masked canvas positions, selects the
lowest-entropy positions to unmask (entropy-bound selection) and samples their tokens. This does not
fit the existing single-token sampler contract.

## Scope
- New `IDiffusionUnmaskSampler` (canvas-level): input = logits for all masked canvas positions +
  current temperature; output = the set of positions to unmask this step + their sampled token ids.
- Implement `EntropyBoundSampler` matching DiffusionGemma's `EntropyBoundSamplerConfig`: select the
  lowest-entropy positions such that the mutual-information bound stays under `entropy_bound=0.1`.
- Compute per-position entropy (over the softmax of capped logits) for the scheduler's stop check.
- Reuse existing per-position sampler steps (temperature/top-k/top-p) to sample the chosen positions'
  tokens once selected.

## Acceptance Criteria
- [ ] Canvas-level sampler interface + `EntropyBoundSampler` implementation.
- [ ] Lowest-entropy positions selected under the `entropy_bound` constraint; deterministic at temp→0.
- [ ] Per-position entropy exposed for the early-stop decision.
- [ ] Composes with temperature from the scheduler (issue 08).
- [ ] Unit tests: entropy ordering, selection count, determinism.

## Dependencies
- Blocks on **05** (entropy bound config). Consumed by **06, 08**.

## References (dev, file:line)
- `src/DotLLM.Core/Sampling/ISamplerStep.cs:8-16` (per-position contract — insufficient)
- `src/DotLLM.Core/Sampling/SamplerContext.cs` (context to extend/parallel)
- verified: `EntropyBoundSamplerConfig`, `entropy_bound=0.1`, entropy stop `0.005`
