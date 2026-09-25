# 08 — Denoise / remask scheduler (step schedule + temperature)

**Effort: M**

## Summary / Motivation
The denoise loop needs a scheduler that, per step, decides the temperature, how many tokens to
unmask, and which stay masked — plus the adaptive early-stop. This is the policy layer between the
forward pass and the sampler.

## Scope
- `DenoiseScheduler`: produces per-step temperature from the linear schedule (`t_max=0.8 → t_min=0.4`
  over up to `max_denoising_steps=48`), the per-step unmask budget, and the stop decision.
- Adaptive early stop: stop when average canvas entropy < `confidence_threshold` (0.005) or
  `stability_threshold` (1) consecutive stable steps.
- Remask policy: tokens not selected for unmasking remain `MaskTokenId` for the next step; already
  unmasked tokens are frozen (absorbing-state) unless a remask-on-low-confidence policy is configured.
- Keep the schedule pluggable (the model ships one schedule; allow override for experiments).

## Acceptance Criteria
- [ ] Temperature follows the linear `0.8→0.4` schedule across steps.
- [ ] Per-step unmask budget + stop decision implemented; typical run lands in 12–16 steps with adaptive stop.
- [ ] Absorbing-state invariant: once unmasked, a position is not re-masked (unless remask policy on).
- [ ] Hard cap at `max_denoising_steps`.
- [ ] Unit tests for schedule values + stop conditions.

## Dependencies
- Blocks on **05** (diffusion config). Consumed by **06, 09**.

## References (dev, file:line)
- verified `generation_config.json`: `max_denoising_steps=48`, `t_min=0.4`, `t_max=0.8`,
  `confidence_threshold=0.005`, `stability_threshold=1`
- `src/DotLLM.Core/Sampling/IStopCondition.cs` (existing stop-condition pattern to mirror)
