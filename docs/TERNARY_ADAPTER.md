# Ternary residual adapters (`.nsadapter` v0) - PROPOSAL

> **Status: v0 proposal written in dotLLM (#1616).** The issue asks for the adapter spec and reference SIMD
> kernels "from NeuroSymbolicAI". A search of `C:\Development\neuro-symbolic-ai` (all branches, history,
> GitHub branches: only `main`), `OracleWrittenTernary` and `DotLLMExperiments` found **no** ternary adapter code, no
> `.nsadapter` format and no kernels (the neuro-symbolic repo is a Z3/reasoning pipeline; `OracleWrittenTernary` is
> an FFN-replacement oracle with its own 4-trits-per-byte format, unrelated). So this spec is **dotLLM-authored**.
> The "25,000+ tok/s at K=32" figure in the issue is **unverified**: no benchmark source was found and it is not
> reproduced here.

## Circuit

Applied to the residual stream after layer `l` (`HookPoint.PostLayer`), per token row `x` (width `d_model`):

```
h_j   = alpha_j * sigma( sum_c W_down[j,c] * x[c] )        j = 0..r-1
out_c = x[c] + sum_j W_up[c,j] * h_j
```

`W_down` (r x d) and `W_up` (d x r) are strictly {-1,0,+1}; `sigma` = ReLU (`s > 0 ? s : 0`, NaN -> 0) or identity;
`alpha` is a per-layer scalar (per-rank-row after composition). No multiplication touches a weight.

## Hook seam

`IInferenceHook` is observe-or-`Replace(float[])` (allocates per call) and was not wired into the CPU forward at all.
New `DotLLM.Core.Diagnostics.IResidualModifier` (`AppliesTo(layer)`, `ModifyResidual(Span<float> rows, layer, tokens)`)
is held as a nullable `TransformerModel.ResidualModifier`; the layer loop does one `is not null` test per layer and
nothing else when unset. With a modifier attached `ForwardBatch` falls back to the per-sequence core. CPU
`TransformerModel` only (Gemma-3n multi-stream layers and non-CPU backends are not covered).

## Packed weights

Row-major, rows = rank index. `W_up` is stored transposed (row j = column j of `W_up`). Each row is `ceil(d/64)`
blocks of two `ulong` words `[plus, minus]`; bit `b` of block `k` is column `64k+b`; bits at columns >= d must be 0;
a column with both bits set is invalid. 2 bits/weight, read in place (memory-mapped).

## Kernel summation order (canonical, identical on scalar / Vector256 / Vector512 -> bit-exact)

16-lane chunks (lane = column mod 16, tail lanes read +0). Down: per row, accumulators `P[16]`, `M[16]` start +0;
chunks ascending `P[l] += plus ? x : +0`, `M[l] += minus ? x : +0`; `s = fold(P) - fold(M)` with fold = pairwise
`a[i]+=a[i+8]; a[i]+=a[i+4]; a[i]+=a[i+2]; a[0]+a[1]`. `h_j = alpha_j * sigma(s)`. Up: per output chunk
`acc=+0`; j ascending `acc += plus ? h_j : +0` then `acc -= minus ? h_j : +0`; `out = x + acc`. F32 accumulation in
registers; only scratch is the rank-r intermediate (preallocated native buffer). Verified by
`TernaryAdapterTests` against an independent managed-matrix reference (bit-exact, 18 widths x 6 ranks x 3 alphas x 3 ISAs).

## `.nsadapter` file (little-endian)

| offset | field |
|---|---|
| 0 | magic `NSAD` |
| 4 | u16 version (0) |
| 6 | u16 flags (0) |
| 8 | u32 manifest length |
| 16 | u64 payload offset (64-aligned) |
| 24 | u64 payload length |
| 32 | SHA-256 of payload |
| 64 | manifest (YAML subset, UTF-8), then padding, then payload |

Manifest: `format: nsadapter`, `format_version: 0`, `name`, `d_model`, `sigma: relu|identity`,
`dtype: ternary-bitmask-2bit`, `block_cols: 64`, `mask_order: plus,minus`, and a `layers:` sequence of
`{layer, rank, alpha, down_offset, up_offset}` (payload-relative, 64-aligned). The loader verifies magic, version,
bounds, checksum, no `+1`/`-1` overlap and zero padding bits.

## Composition

`TernaryAdapterComposer.Compose` rank-concatenates adapters per layer (rows stacked, per-rank alpha vector carries each
adapter's alpha x optional weight): one pass computes `x + sum_i alpha_i W_up_i sigma(W_down_i x)` (parallel experts
reading the same `x`, not a chain). Summed rank capped (default 256).

## Use

`dotllm run ... --ternary-adapter a.nsadapter [--ternary-adapter b.nsadapter]` (CPU backend).

## Not done / follow-ups

Vulkan/CUDA adapter kernels; GGUF-native adapter metadata keys; server/engine-level `InferenceOptions`; NEON path
(Vector128 falls back to scalar); training/synthesis tooling (the companion repo's job).

---

# Zero-multiplication kernels for ternary MODEL weights (Bonsai-2 PQ2_0) - measurement (#1616 part 2)

Harness: `tests/DotLLM.Tests.Unit/Cpu/Kernels/TernaryZeroMulProbe.cs` (+ `TernaryMaskedProbeKernels.cs`), opt-in
(`Category=Benchmark`, `DOTLLM_TERNARY_PROBE=1`). Real weights from `Ternary-Bonsai-2-27B-PQ2_0.gguf` (all PQ2_0, zero
code-3 weights), 12-matrix rotation (>L3), Zen 5 (Strix Halo, 16 pool threads), 8 reps, both A/B orders alternating,
`cpu-bench` lock held, same session. Activations are synthetic (uniform / gaussian+outliers): the weights are real, the
activations are not (no model forward). Not re-run, per earlier findings: I2_S gather-LUT / dot-widen (refuted, #202) and
"512-bit width is not a lever on this part" (#415) - this compares different kernel *structures*.

Candidates (all int8 activations, int32 accumulation, no F32 weight expansion):

- **prod**: `GemmPQ2_0` (row unpack to int8 + sign-trick VNNI dot, Q8_0 activations).
- **K2 masked add/sub** (the issue's proposal): weights repacked to plus/minus bitmasks per 32 elements (same 34 B/group),
  masks expanded to byte masks (broadcast+pshufb+and+cmpeq; 512-bit variant uses `vpermb`), `sum(x&P)-sum(x&M)`.
- **K3 planar-offset** (new idea, not multiplication-free: it uses VPDPBUSD on the raw 2-bit codes): PQ2_0 bytes read IN PLACE
  with no unpack; activations re-laid into 4 planes per 128-group so `t_i=(byte>>2i)&3` lines up directly;
  `sum((t-1)q) = dpbusd(t,q) - sum(q)` with the `sum(q)` terms folded into one per-group scalar. Costs an O(k) per-call
  activation prep (shared by q/k/v/gate/up that read the same input).

Decode (n = 1), per-call median over 8 reps, weights-only GB/s; stream-read roofline = pure read of the same rotation:

| shape (m x k) | threads | prod | K2 256 | K2 512 | K3 planar | roofline |
|---|---|---|---|---|---|---|
| 17408x5120 (ffn gate/up) | 16 | 0.658 ms 36 GB/s | 0.98x | 1.21x | **2.16x** (77.7 GB/s) | 105 GB/s |
| 5120x17408 (ffn down) | 16 | 0.807 ms 29 GB/s | 1.18x | 1.46x | **2.33x** (68 GB/s) | 113 GB/s |
| 248320x5120 (lm_head) | 16 | 9.58 ms 35 GB/s | 1.06x | 1.47x | **2.56x** (90 GB/s) | 99 GB/s |
| same three | 1 | 4.2 / 3.9 / 4.3 GB/s | 1.02 / 1.06 / 1.03x | 1.30 / 1.31 / 1.28x | **2.73 / 2.76 / 2.69x** | 34-37 GB/s |

(per-rep min/max are within about +-8% of the median in the raw output; full tables are reproducible with the harness.)

Small-n / prefill (16 threads, median, ratio vs prod): n=4: K2 0.48-0.50x (512: 0.56-0.61x), K3 1.13-1.22x;
n=16: K2 0.36-0.39x (512: 0.45-0.48x), K3 0.87-0.94x. K2 re-expands the masks per token (about 8 extra ops per 32 elements);
K3 re-decodes per token (not token-blocked), which is why it fades at n=16 (production's unpack-once-per-row amortises).

Parity: K2 matches production to 3e-7..6e-6 (max |d| / output rms; same Q8_0 grid, different lane order). K3 uses a
different quantization grid, so vs production the max |d|/rms is 3e-2..2e-1 over millions of outputs, but vs an **F64
oracle on real weights** (`Parity_RealWeights_VsF64Truth`, 5 real tensors x {uniform, gaussian+outliers}) the rel-RMS error
is the same as production's (e.g. 3.94e-3 vs 3.44e-3 uniform, 9.87e-3 vs 1.05e-2 outliers). No end-to-end KL/perplexity was run.

## Verdict

- **Bitmask masked add/sub (the issue's proposal) does NOT win.** Decode: tie (256-bit) to +20-47% (512-bit `vpermb`
  expansion), but it loses 2-3x at n >= 4. Needing mask expansion (3-4 ops per 32 lanes, x2 masks) is more work than the
  existing multiply-add on unpacked int8. .NET exposes no mask registers, so AVX-512 `k`-mask merge-masking is not available.
  Not wired. (NEON: compile-checked only conceptually, `Vector128`/`vqtbl1q` would need the same expansion; no ARM box.)
- **K3 (planar, in-place 2-bit codes) wins decode clearly: 2.2-2.8x at n=1**, 70-90% of the streaming roofline (production is
  at ~30-35%), accuracy equal to production vs F64 truth. It is neutral at n=4 (1.1-1.2x) and loses slightly at n=16. It
  has been wired **default-off** behind `DOTLLM_PQ2_PLANAR=1` for the n=1 GEMV only (`MatMul.PQ2Planar.cs`); a follow-up
  issue tracks end-to-end validation (tok/s + PPL/greedy identity on Bonsai-2) before it can become default.
- Caveats: microkernel/UMA numbers (earlier notes: decode GEMV microkernel wins have been noise end-to-end when the path was
  already bandwidth-bound; here the production path is demonstrably compute-bound at ~35% of roofline, so a real gain is
  plausible but unproven end-to-end). No PMU counters on Windows without admin: cache/register behaviour is inferred
  from op counts (K3: 1 load + 3 shifts + 4 ands + 4 vpdpbusd + 4 cvt/fma per 128 elements, no 5 KB unpack round-trip through L1).
