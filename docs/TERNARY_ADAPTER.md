# Ternary residual adapters (`.nsadapter`) in dotLLM

dotLLM consumes the **NeuroSymbolicAI** ternary residual adapter packages (`jamesburton/neuro-symbolic-ai`, `Ternary.Core`,
commit `15104eb`) and reproduces the reference's numerics **bit for bit**. (An earlier dotLLM-authored v0 format in this PR was
dropped when the real reference was published; nothing of it remains.)

## The circuit (as the reference computes it)

Applied to one residual row `h` (width `D`) after transformer layer `L` (`SimulatedTransformerSpine` int8 step + `TernaryExpert.Forward`):

```
x8   = absmax_int8(h)          scale 127/max|h|, Math.Round (half-even), clamp [-128,127]; the x scale is NOT re-applied
d_j  = (float)( sum_c Wdown[j,c] * x8[c] ) * downScale[j]            j < rank      (integer sum, then one float multiply)
r_j  = max(0, d_j);   m = max_j r_j
q_j  = clamp(round(r_j * (127/m)), -128, 127)                         (only if the gate is open)
u_d  = (float)( sum_j Wup[d,j] * q_j ) * upScale[d]                   d < D
h'_d = h_d + ( 0 + u1_d*dq1 + u2_d*dq2 + ... ),   dq = (m/127) * Alpha
```

Down is `Rank x D`, Up is `D x Rank` (each row has its own F32 scale). All weight arithmetic is integer masked add/subtract; the
only float ops are single IEEE multiplies/adds in the order above (no FMA), so the scalar and AVX2 paths and the reference agree exactly.

**Alpha / ActivationThreshold.** The reference `TernaryExpert` has a `Gate` coefficient (`Gate <= 0` => exact identity, otherwise the
delta is multiplied by it) and the manifest carries `Alpha` and `ActivationThreshold`, but **no reference code reads the manifest
values** (the sample adapters have `Alpha = 1`, `ActivationThreshold = 0`). dotLLM maps `Alpha` to `Gate` (so `Alpha <= 0` disables the adapter)
and defines `ActivationThreshold` as: the circuit contributes only when `m > max(ActivationThreshold, 1e-6)` (the `1e-6` is the
reference's own existing guard; threshold 0 therefore reproduces the reference exactly). The threshold semantics is an
interpretation and needs sign-off from the NeuroSymbolicAI side.

## `.nsadapter` package

A ZIP archive (System.IO.Compression): `manifest.json`, `down.bin`, `up.bin` (legacy `weights_down.bin` / `weights_up.bin` accepted).

`manifest.json` (case-insensitive, unknown fields ignored): `TargetModel`, `TargetLayer`, `InputDimension` (D), `Rank`, `Alpha`,
`ActivationThreshold`, `Domain`, `Description`, `CreatedAt`, `DownChecksum`, `UpChecksum` (SHA-256 hex of the tensor's packed *data*
bytes, or of the whole entry stream).

Tensor binary (`PackedTernaryTensor.WriteTo`, little-endian): `int32 rows, int32 cols, float32 scale[rows], byte data[rows*ceil(cols/4)]`.
Weights are BitNet b1.58 2-bit codes, 4 per byte, column `c` at bits `2*(c&3)` of byte `c/4`: `00` = 0, `01` = +1, `11` = -1, `10` reserved
(read as 0). If a tensor entry is exactly `rows*ceil(cols/4)` bytes without a header it is read as raw data with unit scales (reference fallback).

dotLLM rejects (`InvalidDataException`): not a zip, missing entries, entries larger than the manifest's dimensions allow (checked before reading),
tensor dimensions that disagree with `Rank x D` / `D x Rank`, non-positive rank/D, negative layer, non-finite alpha/threshold, missing or mismatching checksums.
`ValidateFor(hidden, layers)` additionally rejects a D or layer that does not fit the model. `ToGgufMetadata()` returns the reference's key set
(`neuro_symbolic.{domain,target_model,layer,target_layer,input_dimension,rank,alpha,activation_threshold,description,created_at,down_checksum,up_checksum}`);
reading adapters back from GGUF is described in [GGUF form](#gguf-form-1619).

Files written by dotLLM load in the reference loader (checked) and the reference's sample adapters are test fixtures.

## How dotLLM runs them

- `TernaryResidualHook` (Models) loads packages, converts them once into an in-place bitmask layout (Down as-is, Up transposed to `Rank x D`;
  plus/minus 64-bit masks per 64 columns, 2 bits/weight) and applies them through `IResidualModifier` after each layer.
  Scratch (int8 activations, rank strip) is preallocated; nothing is allocated per token.
- **Why not `IInferenceHook`?** Its only way to change an activation is `HookResult.Replace(float[])`, a fresh allocation per call, and it was not wired into the CPU forward.
  `IResidualModifier` (Core) mutates the `[tokens x D]` residual span in place; `TransformerModel.ResidualModifier` is null by default (one branch per layer). `TernaryResidualHook`
  implements it. (Observing the delta norm through `IInferenceHook` was not added: it would need an allocation per call.)
- **Several adapters** (also at the same layer) are evaluated in one pass sharing one int8 quantization of the row, in registration order, deltas summed like the
  reference's delta buffer. This is the rank-concatenation idea of `LoraComposer`, but `LoraComposer` composes F32 `ILoraAdapter` weights into one concatenated matrix pair and
  cannot express per-adapter ReLU/int8 requantization or per-row scales, so the ternary composition lives in the hook. Summed rank per layer is capped at 256.
- **Scales are metadata** (CLAUDE.md #1107): `downScale[rank]` + `upScale[D]` per adapter (about 16 KB F32 at D=4096). Weights are never expanded; the masks are the packed weights.
  Default is F32, bit-identical to the reference. `TernaryResidualHook.ScaleFormat = TernaryScaleFormat.F16` stores them as binary16 (8 KB): see [F16 scales](#f16-per-row-scales-1620).
- CLI: `dotllm run ... --ternary-adapter a.nsadapter [--ternary-adapter b.gguf]` and `dotllm serve ... --ternary-adapter ...` (CPU backend; each path is a ZIP package or an adapter GGUF, detected by magic; repeatable). x86 AVX2 + scalar paths; other ISAs use the scalar path (NEON is a follow-up).

## Tests

`TernaryAdapterTests`: golden SHA-256 of the real reference's outputs (7 adapters x 4 rows, both ISAs: the 3 sample adapters, 2 reference-written synthetic packages, 2 regenerated deterministically),
an independent managed port of the reference pipeline over 252 shape/alpha/ISA combinations (bit-identical), multi-adapter order, alpha/threshold gating, format round trip and
rejection cases, engine wiring. The goldens come from building `Ternary.Core` + `Expert.Model` from the reference commit and running `TernaryExpert` on the same inputs.

## GGUF form (#1619)

`NsAdapterGguf` (DotLLM.Models) writes and reads the same circuits as the ZIP package inside a GGUF container, bit-identically (package -> GGUF -> package keeps every code,
scale, manifest field, the float bits of alpha/threshold and the timestamp offset; GGUF -> ZIP is accepted by the reference loader).

| what | where |
|---|---|
| adapter count | `nsadapter.count` (uint32; absent = 1), at most 256 |
| metadata of adapter 0 | the reference's `neuro_symbolic.*` key set, string values, verbatim (`ToGgufMetadata()`; checksums are recomputed from the data) |
| metadata of adapter i >= 1 | `neuro_symbolic.<i>.*` (same suffixes) |
| exact floats | `nsadapter.<i>.alpha`, `nsadapter.<i>.activation_threshold` (GGUF float32); preferred on read, because the reference's `G7` strings do not round-trip every binary32 |
| packed codes | tensors `nsadapter.<i>.down` (rank x D) and `nsadapter.<i>.up` (D x rank): the package's BitNet 2-bit bytes verbatim (`rows * ceil(cols/4)`, zero padded to a multiple of 4) as a **1-D `I32` tensor** |
| per-row scales | tensors `nsadapter.<i>.down_scale` (rank) and `nsadapter.<i>.up_scale` (D), 1-D `F32` |
| standalone file | `general.architecture = "nsadapter"`; the same keys and tensors may instead be appended to a model GGUF |

**Why a raw `I32` blob and not an existing 2-bit type.** dotLLM's GGUF reader only accepts ids in its `QuantizationType` enum (no `I8`/`U8`). The 2-bit types that exist (`I2_S`, `PQ2_0`,
`TQ2_0`) use different code maps (`I2_S`: 0/1/2 = -1/0/+1) and block layouts, so mapping onto one would be a conversion, not the reference's bytes, and could not be checksum-verified against
the package. `I32` is the only 4-byte integer type present and carries the packed bytes untouched (no new ggml type is introduced); shapes come from the `rank` / `input_dimension` metadata.
Validation matches the ZIP loader plus: tensor type and exact element count, zero padding, SHA-256 of both code tensors against the metadata, `rank`/`D` bounds before any allocation.

Wiring: `TernaryResidualHook.Load(paths)` accepts either format (magic `GGUF` or ZIP), so `dotllm run` and the new repeatable `dotllm serve --ternary-adapter` use both. `ServerOptions.TernaryAdapterPaths`
attaches the hook to every model the server loads (CPU transformer only; an explicit or auto GPU device is not used when adapters are requested, and a non-attachable model is an error, never a silent skip).
**Not done:** `InferenceOptions` per-request enable/disable. The residual hook sees `[tokens x D]` rows with no request identity and the scheduler batches requests, so a per-request switch
needs a per-row mask through `IResidualModifier` (filed as a follow-up); the adapters currently apply to every request of the server.

## F16 per-row scales (#1620)

Question: are F16 (or bf16) scales sufficient? Method: round the package scales RNE (F32 -> binary16 / bfloat16 -> F32 exact), keep weights exact and accumulation F32, and compare the adapter output delta against F32 scales
(`TernaryScalePrecisionTests`, managed port of the reference circuit; 30 residual rows per case, D = 4096, outlier channels). The noise floor is the adapter's own int8 requantisation (delta of the shipped circuit vs the same circuit with an unquantised bottleneck).

| scale distribution (median, sigma 0.3) | rank | int8 floor (relL2) | F16 relL2 | bf16 relL2 | F16 q_j flips | bf16 q_j flips | F16 / bf16 out-int8 flips |
|---|---|---|---|---|---|---|---|
| 0.02 | 8 / 32 / 128 | 3.2e-3 / 4.9e-3 / 6.8e-3 | 6.5e-4 / 2.0e-3 / 1.9e-3 | 3.7e-3 / 4.8e-3 / 5.7e-3 | 0.42 / 0.52 / 0.37 % | 2.5 / 4.3 / 2.8 % | 0.8-2.6 % / 5.8-10.3 % |
| 0.002 | 8 / 32 / 128 | 3.4e-3 / 5.6e-3 / 8.4e-3 | 9.1e-4 / 1.9e-3 / 2.3e-3 | 3.5e-3 / 4.9e-3 / 6.0e-3 | 0.3-0.4 % | 2.5-2.7 % | 0.5-1.6 % / 3.4-7.6 % |
| 1.0 | 8 / 32 / 128 | 3.3e-3 / 5.5e-3 / 6.8e-3 | 4.9e-4 / 1.7e-3 / 2.2e-3 | 2.4e-3 / 5.2e-3 / 5.5e-3 | 0.4-0.6 % | 0.8-3.9 % | 1.1-3.2 % / 5.2-10.4 % |
| 40 | 8 / 32 / 128 | 3.1e-3 / 5.1e-3 / 7.6e-3 | 1.9e-3 / 2.0e-3 / 2.3e-3 | 3.6e-3 / 4.4e-3 / 6.2e-3 | 0.4-0.8 % | 3.2-3.3 % | 1.1-3.3 % / 5.4-11.4 % |

("q_j flips": share of the int8-requantised bottleneck values that differ by one level; "out-int8 flips": share of output elements whose absmax-int8 value differs; a flip is +-1 of 127.)
Reference sample adapters (`bracket-parity`, `csharp14-syntax-attractor`, `modulo97-counter`) have only unit scales (1.0), so every format is exact for them (relL2 = 0). The reference's own scale rule is absmean (`PackedTernaryTensor.SetRow`), i.e. the mean |w| of the float row, typically 1e-3 .. a few.
End to end on the tiny synthetic llama (D = 64, adapter at layer 1, 24 positions): the adapter itself moves the logits by KL 2.6-3.5 nats (top-1 agrees 1/24, 0/24); switching its scales F32 -> F16 moves them by **KL 4.7e-6 / 6.1e-8** (top-1 24/24), F32 -> bf16 by 1.7e-5 / 2.4e-5.
Range: F16 has a normal range [6.1035156e-5, 65504]; at scales around 1e-6 it is subnormal (relL2 2.2e-2, 22 % of q_j flip) and above 65504 it overflows to infinity, bf16 (F32 exponent) does neither.

**Verdict.** *F16 is sufficient inside its normal range*: it adds at most 0.1-0.7x the adapter's own int8 requantisation noise, 0.3-0.8 % one-level flips, and an end-to-end KL six to eight orders below the adapter's effect. *bf16 is not worth it*:
2.5-4 % of the bottleneck values flip and its error is about equal to the int8 floor (it only buys range, which one F32 fallback also buys). *Keep F32 as the default*: F16 saves 8 KB per adapter at D = 4096 (the masks, 2 bits per weight for down and up, are 16 KB at rank 8 and 512 KB at rank 256), the adapter is not memory-bound,
and F32 keeps the bit-for-bit reference goldens. F16 is implemented as an **opt-in** in-memory format (`TernaryResidualHook.ScaleFormat = TernaryScaleFormat.F16`): scales are read as F32 from the package and rounded RNE to IEEE binary16 once at load; a circuit whose non-zero |scale| is outside
[`MinF16Scale`, `MaxF16Scale`] = [2^-14, 65504] silently stays F32 (`ScaleFormatAt(i)` reports what was chosen). Kernels widen F16 -> F32 exactly, so an F16 hook is bit-identical to an F32 hook built from RNE-rounded scales (tested on both ISAs, odd D, zero scales, mixed formats).

---

# Zero-multiplication kernels for ternary MODEL weights (Bonsai-2 PQ2_0) - measurement (#1616 part 2)

Harness: `tests/DotLLM.Tests.Unit/Cpu/Kernels/TernaryZeroMulProbe.cs` (+ `TernaryMaskedProbeKernels.cs`), opt-in
(`Category=Benchmark`, `DOTLLM_TERNARY_PROBE=1`). Real weights from `Ternary-Bonsai-2-27B-PQ2_0.gguf` (all PQ2_0, zero
code-3 weights), 12-matrix rotation (>L3), Zen 5 (Strix Halo, 16 pool threads), 6-8 reps, A/B orders alternating,
`cpu-bench` lock held, same session. Activations are synthetic (uniform / gaussian+outliers): the weights are real, the
activations are not (no model forward). Not re-run, per earlier findings: I2_S gather-LUT / dot-widen (refuted, #202) and
"512-bit width is not a lever on this part" (#415) - this compares different kernel *structures*.

Candidates (int8 activations, int32 accumulation, no F32 weight expansion unless stated):

- **prod**: `GemmPQ2_0` (row unpack to int8 + sign-trick VNNI dot, Q8_0 activations).
- **K2 masked add/sub** (the issue's idea): weights repacked to plus/minus bitmasks per 32 elements (same 34 B/group), masks expanded to byte
  masks (broadcast+pshufb+and+cmpeq; the 512-bit variant uses `vpermb`), `sum(x&P)-sum(x&M)`.
- **K3 planar-offset** (new, not multiplication-free: VPDPBUSD on the raw 2-bit codes): PQ2_0 bytes read IN PLACE, no unpack; activations re-laid into 4 planes
  per 128-group so `t_i=(byte>>2i)&3` lines up directly; `sum((t-1)q) = dpbusd(t,q) - sum(q)` with the `sum(q)` terms folded into one per-group scalar.
  Costs an O(k) per-call activation prep (shared by the matmuls that read the same input).
- **REF SimdTernaryKernel / REF UnpackedTernaryMatrix**: faithful ports of the NeuroSymbolicAI `SimdTernaryKernel.Forward` (packed BitNet codes, 256-entry LUT unpack,
  zero-word skip, `Vector256.Widen` multiply, long row sum) and `UnpackedTernaryMatrix.Forward` (int8 weights, 4x the memory). They are matvec kernels with ONE scale per
  row, so the PQ2_0 data is re-coded into their layout and per-group scales are dropped: a **throughput stand-in, not the same numerics**. For n > 1 they loop tokens
  (re-reading the weights per token, as the reference would). At 16 threads they are row-partitioned like the others (the reference itself is single-threaded).
  The ports are checked against an exact integer oracle in CI.

## Decode and small-n, real shapes (median per call; 16 threads unless noted; ratio vs prod, >1 is faster)

| shape (m x k) | n | prod | K2 256 | K2 512 | **K3 planar** | REF packed (LUT) | REF unpacked int8 |
|---|---|---|---|---|---|---|---|
| 17408x5120 (ffn gate/up) | 1 | 0.706 ms, 33.5 GB/s | 1.19x | 1.28x | **2.18x** (73 GB/s) | 0.13x | 0.56x |
| 5120x17408 (ffn down) | 1 | 0.758 ms, 31.2 GB/s | 1.34x | 1.28x | **2.02x** (63 GB/s) | 0.14x | 0.56x |
| 248320x5120 (lm_head, earlier run) | 1 | 9.58 ms, 35 GB/s | 1.06x | 1.47x | **2.56x** (90 GB/s) | - | - |
| 17408x5120 | 4 | 1.18 ms | 0.57x | 0.57x | 1.09x | 0.06x | 0.24x |
| 5120x17408 | 4 | 1.08 ms | 0.51x | 0.46x | 0.97x | 0.05x | 0.23x |
| 17408x5120 | 16 | 3.62 ms | 0.45x | 0.40x | 0.88x | 0.05x | 0.20x |
| 5120x17408 | 16 | 3.47 ms | 0.43x | 0.37x | 0.88x | 0.05x | 0.19x |
| same two, 1 thread, n=1 | 1 | 5.27 / 5.58 ms | 1.34 / 1.41x | 1.32 / 1.37x | 2.56 / 2.73x | 0.11x | 0.56x |

Read-only streaming roofline for the same rotation and threads: 101-105 GB/s (16 threads), 38-40 GB/s (1 thread). Per-rep min/max are within roughly +-8% of the medians in the raw
output (reproducible with the harness); activation prep is excluded from the table (planar prep 0.009-0.06 ms per input row vs 0.3 ms for the matmul).

Parity: K2 matches production to 3e-7..6e-6 (max |d| / output rms; same Q8_0 grid, different lane order). K3 uses a different quantization grid, so vs production the max |d|/rms over millions of
outputs is 3e-2..2e-1, but vs an **F64 oracle on real weights** (`Parity_RealWeights_VsF64Truth`) its rel-RMS error equals production's (3.9e-3 vs 3.4e-3 uniform; 9.9e-3 vs 1.05e-2 with outliers).
No end-to-end KL/perplexity was run.

## Verdict

- **At n = 1 (decode) the K3 planar in-place kernel wins: 2.0-2.6x over production, 63-90 GB/s = 60-90% of the read roofline (production sits at about 30-35%)**; it is neutral at n = 4 and
  loses 12% at n = 16 (it re-decodes the codes per token). It is wired **default-off** behind `DOTLLM_PQ2_PLANAR=1` for the n = 1 GEMV (`MatMul.PQ2Planar.cs`); follow-up #1620 covers end-to-end validation and token blocking.
- **The kernel from the issue as the NeuroSymbolicAI team wrote it loses badly on Bonsai-2 shapes**: the packed LUT version is 0.13x (decode) down to 0.03x (n=16), the int8-unpacked version
  0.56x down to 0.18x. Our own bitmask masked add/sub (K2) is 1.2-1.5x at decode but 0.4-0.57x at n >= 4. No ternary "masked add/sub" variant beats production at n >= 4.
- **Does the win shrink or grow with matrix size?** The reference's headline figure is a K x 4096 down-projection (128 KB of int8 weights, L2 resident). On this box its throughput is about 9.4 GB/s of int8 weights
  at 128 KB (13.9 us) and still 9.4 GB/s at 89 MB (9.4 ms for 17408x5120, single thread): it is compute-bound, not cache-bound, so its relative standing against a bandwidth-bound production kernel
  does not improve with size; the K3 in-place kernel is the opposite (bandwidth-limited at DRAM sizes, 11.5 GB/s single-thread, 63-90 GB/s with 16 threads). Cache residency does not rescue the reference approach on large matrices.
- Caveats: microkernel numbers on UMA; earlier notes record decode GEMV microkernel wins vanishing end-to-end when the path was already bandwidth-bound, but here production is demonstrably compute-bound at about a third of the roofline,
  so a real gain is plausible yet unproven end-to-end. No PMU counters on Windows without admin: cache/register behaviour is inferred from op counts (K3: 1 load + 3 shifts + 4 ands + 4 vpdpbusd + 4 cvt/fma per 128 elements, no 5 KB unpack round-trip through L1).

# Reproducing the adapter-kernel claim ("25,126 tok/s at rank 32")

The reference's `Ternary.Benchmark` "Experiment 1" times ONE `UnpackedTernaryMatrix.Forward`: a single int8 down-projection of K rows over D = 4096 (int8 activations in, K floats out). A "token" there is that one
matrix-vector product, not a full adapter application and not a model token. The committed report shows 50.70 us / 19,724 per s at K = 32 (on its machine; the 25,126 figure (39.80 us) is not in the repository).
Same-session measurements on this box (cpu-bench held; reference built from `15104eb`; order ref, dotLLM, dotLLM, ref; 15 reps x 2000 calls, min / median / max in microseconds):

| measurement (D = 4096) | K=32 | K=64 | K=128 |
|---|---|---|---|
| REF `UnpackedTernaryMatrix.Forward` (int8 weights; the benchmark's "token") | 13.64 / 13.84-14.18 / 16.3 (70.5-72.2k /s) | 27.5 / 27.7-28.2 / 28.8 | 54.8 / 55.1-55.4 / 61.4 |
| REF `SimdTernaryKernel.Forward` (packed, LUT) | 96.1 / 96.6-100.3 / 135.9 | 193.6 / 195-199 / 201.9 | 386 / 389-399 / 402 |
| dotLLM down-projection (packed masks, AVX2) | 4.57 / 5.82-5.86 / 6.43 (171k /s) | 9.8 / 10.25-10.31 / 11.2 | 20.1 / 20.3-20.9 / 21.2 |
| dotLLM down-projection, scalar path | 40.4 / 40.9-57.9 / 107.9 | 96.6 / 99.7-118.8 / 121 | 230 / 231-242 / 249 |
| REF full adapter `TernaryExpert.Forward` (int8 in, no quantize, no residual add) | 30.3 / 30.5-31.1 / 32.2 (32-33k /s) | | |
| dotLLM FULL adapter via the hook (float row in: absmax-int8 + down + ReLU/requant + up + residual add) | 29.2 / 36.1-36.9 / 37.7 (27-27.7k /s) | | |

So the claim **reproduces and is exceeded** on this machine (70k vs 25k per s for the benchmark's down-projection "token"; 32k per s for the whole adapter), but it measures a single down-projection, and it was obtained
with the int8-unpacked matrix (4x the memory of the 2-bit package). dotLLM's packed-mask down-projection is 2.4x faster than the reference's unpacked one at the same shape. The full adapter is about 20% slower than the reference's
`TernaryExpert.Forward` in dotLLM's current form (its up-projection, 4096 x rank masked add/sub, is the cost; the reference's own number excludes the activation quantization and the residual add): a follow-up optimisation, not a correctness issue.
