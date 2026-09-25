# 11 — Throughput + capability benchmark harness (gated on GPU-free check)

**Effort: M**

## Summary / Motivation
The diffusion selling point is parallel-canvas throughput. We need a benchmark harness that measures
tokens/sec, denoise-steps/sec, and canvas latency for the dotLLM diffusion path, compares against an
AR baseline of comparable size, and reports capability (quality) on a fixed prompt set — all gated on
an explicit GPU-idle check because the dev box shares its iGPU with concurrent work.

## Scope
- BenchmarkDotNet (or the repo's existing bench harness) jobs for: canvas latency, denoise-steps/sec,
  effective tokens/sec, vs an AR model of similar size (e.g. a small Gemma/Llama).
- Capability scoring on the DiffusionGemma demo task set (code-gen, OCR-correction, short QA) against
  the HF reference output.
- **GPU-free guard**: a pre-run check that aborts the GPU-heavy jobs unless the AMD iGPU is idle (see
  [VALIDATION.md §GPU-free guard](../VALIDATION.md)). CPU correctness jobs always run.

## Acceptance Criteria
- [ ] Bench reports tokens/sec, denoise-steps/sec, canvas latency for the diffusion path.
- [ ] AR-baseline comparison at matched model size + matched quality.
- [ ] Capability scores on the fixed prompt set vs HF reference.
- [ ] GPU-heavy jobs refuse to start when the iGPU is busy; emit a clear skip message.
- [ ] Results written to a report artifact; methodology documented.

## Dependencies
- Blocks on **10** (validated decode path). For real-model numbers also **07**.

## References
- [VALIDATION.md](../VALIDATION.md) (metrics + GPU guard command)
- `benchmarks/DotLLM.Benchmarks/` (existing bench harness)
- DiffusionGemma perf claim: >1000 tok/s on a single H100, ~4× AR (context only — we measure on Strix Halo)
