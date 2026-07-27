# Spec — CPU BitNet (I2_S) GEMV decode: bandwidth-vs-compute profiling

**Status**: Approved (2026-07-27). Ready for plan generation.
**Owner**: dev branch, CPU backend (`src/DotLLM.Cpu/Kernels/MatMul.I2S.cs`).
**Phasing**: single small plan, single PR.

## Goal

Determine whether `GemvI2_S`'s decode path (single-token GEMV: `UnpackRowI8` + int8 VPDPBUSD dot),
after the #128 unpack-SIMD-vectorization fix already merged to `dev`, is now DRAM-bandwidth-bound
or still compute-bound on this Strix Halo box. This is a go/no-go gate for the AVX-512
activation-LUT dot-kernel optimization suggested upstream (issue #334, external contributor
`shifulegend`): if decode is already near the achievable memory-bandwidth ceiling for this access
pattern, a faster dot product would only convert to a modest end-to-end gain (their own BitNet
port saw 6-10% real tok/s from a 3.6x kernel speedup once bandwidth-bound); if decode is well
under the ceiling, the LUT kernel should convert its raw speedup more directly.

No new production code ships from this spec — it produces a measurement and a documented verdict
that decides the next issue (LUT kernel, or pivot to packing-density work).

## Non-goals

- Building the AVX-512 activation-LUT kernel itself — that's a follow-up issue, gated on this
  result.
- Prefill/GEMM path (`GemmI2_SW2A8Rows`) — unpack cost amortizes there already (#131 finding);
  this spec is decode/GEMV-only, where unpack was previously found unpack-bound.
- GPU (Vulkan/CUDA) BitNet paths — CPU only.
- Merging `dev-dotnet11` (`AvxVnni.V512`/`Avx512Bf16`) — not needed here; `Avx512BW`/`Avx512F`
  used for the streaming-ceiling probe are already mainstream-shipping intrinsics available on
  `dev`. Only revisit the preview-track merge if this profiling motivates a kernel that genuinely
  needs the unmerged VNNI-512/BF16 ops.

## Approach

Reuse the existing profiling hooks from #128/#131 (`BenchUnpackRowI8Only`, public forwarder in
`src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs`) plus the public `GemvI2_S` entry point, and add one
new probe: a **pure streaming-read kernel** over the same packed I2_S weight buffers (no unpack,
no dot — just touch every byte) to measure the empirical achievable bandwidth ceiling for this
exact access pattern (sequential 2-bit-packed byte reads) on this box, independent of compute.

This isolates three numbers per shape:
1. Streaming-only GB/s (the ceiling for this access pattern)
2. Full `GemvI2_S` decode GB/s (bytes of packed weight moved ÷ wall time)
3. Ratio of (2)/(1) — how close current decode already is to that ceiling

Unlike #131's throwaway "vendored copies in a standalone bench", this profiling harness is
committed to the repo as a normal BenchmarkDotNet benchmark class, since it's reusable —
future kernel changes (the LUT dot, packing-density changes) get the same before/after comparison
for free.

### Shapes

Same three shapes used in #131 for continuity with prior data: `attn_qproj` (2560×2560),
`ffn_gate` (6912×2560), `ffn_down` (2560×6912). Single-thread, matching #131's methodology (the
existing multi-thread dispatch in `GemvI2_SCore`/`GemvI2_SWorker` is orthogonal to this
bandwidth question and would confound the per-core-bandwidth reading).

### What gets added

| File | Change |
|---|---|
| `benchmarks/DotLLM.Benchmarks/I2SDecodeBandwidthBenchmarks.cs` | New BenchmarkDotNet class: `[Benchmark] StreamingOnly()`, `[Benchmark] UnpackOnly()` (via `BenchUnpackRowI8Only`), `[Benchmark] FullGemvDecode()` (via `GemvI2_S`), parameterized over the three shapes. Reports ns/row and derived GB/s via a custom column (reuse pattern from `Columns/DecodeTokPerSecColumn.cs`). |
| `src/DotLLM.Cpu/Kernels/MatMul.I2S.Bench.cs` | Add `BenchStreamingReadOnly(byte* weights, int m, int k)` — touches every packed byte (XOR-accumulate into a discarded local to prevent dead-code elimination) with no unpack/decode, single pass, same loop shape as `BenchUnpackRowI8Only` for a fair comparison. |

No changes to production kernels (`MatMul.I2S.cs`) — this is measurement-only.

## Error handling / edge cases

Not applicable — benchmark-only code, not a hot-path or user-facing change. Standard
`ArrayPool` rent/return in the new bench helper, mirroring the existing `BenchUnpackRowI8Only`
pattern.

## Testing

No new correctness tests needed (no new production logic). The benchmark class itself is
exercised by running it, not by the unit-test suite. Verification is running
`dotnet run -c Release --project benchmarks/DotLLM.Benchmarks -- --filter *I2SDecodeBandwidth*`
on this box and recording the three shapes' numbers.

## Output / next step

Post the resulting table (streaming-ceiling GB/s, full-decode GB/s, ratio) as a comment on
upstream issue #334, closing the loop with `shifulegend`. Then file a follow-up issue:
- if full-decode GB/s is well under the streaming ceiling (headroom exists): "AVX-512
  activation-LUT dot kernel for I2_S GEMV decode", scoped per shifulegend's suggested mapping
  (their N/K-groups axis = our M/output-rows axis, LUT built once per token and amortized across
  M rows).
- if full-decode GB/s is already close to the streaming ceiling (bandwidth-bound): pivot the
  follow-up issue to packing-density work instead (per shifulegend's fallback advice), scoped
  separately since it's a different code path (weight format, not just the decode kernel).
