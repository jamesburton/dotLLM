# CPU feature gating (#1623)

How dotLLM decides which SIMD tier a CPU kernel runs, how to force a tier, and what the audit found.
Scope: every `System.Runtime.Intrinsics` user in `src/` (45 files: 40 in `DotLLM.Cpu/Kernels`, plus `TokenMaskApplier`,
`CategoricalSampler`, two Models loaders). Vulkan/CUDA/HIP are out of scope.

## The mechanism

`DotLLM.Cpu.CpuFeatures` (`src/DotLLM.Cpu/CpuFeatures.cs`) is the single place a kernel asks "may I use this instruction set?".

| Tier (`CpuIsaTier`) | Includes | Flags that become true |
|---|---|---|
| `Scalar` | nothing | none |
| `Vector128` | SSE2/SSE3/SSSE3/SSE4.1, portable `Vector128` | `Sse2`, `Sse3`, `Ssse3`, `Sse41`, `Vector128Accelerated` |
| `Avx2` | + AVX, AVX2, FMA, AVX-VNNI (if present), portable `Vector256` | `Avx2`, `Fma`, `Avx2Fma`, `AvxVnni`, `Avx2FmaVnni`, `Vector256Accelerated` |
| `Avx512` | + AVX-512 F/BW, portable `Vector512` | `Avx512F`, `Avx512Bw`, `Vector512Accelerated` |

* Every flag is `static readonly bool = (cap allows it) && X.IsSupported`, in a `beforefieldinit` class with no static constructor, so
  the JIT folds it to a constant exactly like `X.IsSupported` (tier-1, OSR and full-opt methods). **Do not turn them into properties.**
* A flag names the whole feature SET a kernel needs. `Avx2` does **not** imply `Fma`; a kernel that calls `Fma.MultiplyAdd` and is
  reached through an AVX2 gate must test `Avx2Fma`. `AvxVnni` implies `Avx2` (never the reverse). `Avx512F` implies `Avx2`.
* The tier chosen on supported hardware is unchanged by this work (AVX-512 on Zen 5 is 256-bit datapaths; see the memory note on
  not reordering tier dispatch). Only the correctness of the gating changed.

### Convention for kernel authors

1. Every intrinsic kernel is reached through a dispatcher that tests the matching `CpuFeatures` flag and ends in a scalar or
   `Vector128` fallback. A public/internal method must never call `Avx2.*` etc. without that guard.
2. A leaf kernel that uses an ISA class either is `private`, or says its tier in its name (`...Avx2`, `...Avx512`, `...Sse`,
   `...Vnni`, `...Fma`) so that a reader and the guard test (below) know a dispatcher must guard it.
3. Do not throw `PlatformNotSupportedException` from an entry point. Fall back. (A loader that cannot serve a model may refuse; a kernel may not.)
4. Never write a raw `Avx2.IsSupported` / `Vector256.IsHardwareAccelerated`: use `CpuFeatures` so the override reaches you.
5. Add the tier to `CpuTierMatrixTests` (scalar oracle on shapes that are not multiples of the vector width, plus the single-block case).

## Forcing a tier: `DOTLLM_CPU_ISA`

`DOTLLM_CPU_ISA=scalar|v128|avx2|avx512` (case-insensitive; unset, empty, `auto`, `best` = no cap). Read once at startup. It **caps** the
highest tier and never raises what the CPU lacks: `avx2` turns the AVX-512 flags off, `v128` turns AVX/AVX2/FMA/VNNI/AVX-512 off, `scalar`
turns every SIMD flag off (kernels run their plain C# loops). An unrecognised value is ignored (no cap) and reported through
`CpuFeatures.InvalidCapValue`/`Describe()`, so a typo cannot silently pin a slower tier.

`DOTNET_EnableAVX=0`, `DOTNET_EnableAVX2=0`, `DOTNET_EnableAVX512F=0`, `DOTNET_EnableAVXVNNI=0` ... keep working: they change what `IsSupported`
reports and every flag is derived from it. The difference: `DOTNET_Enable*` changes what the runtime reports (so also what BCL code such
as `TensorPrimitives` and `Vector256` do), `DOTLLM_CPU_ISA` changes only what dotLLM's own kernels pick. CI runs both: the
`DOTNET_Enable*` leg (as before, now the whole CPU-side unit suite instead of only `DotLLM.Tests.Unit.Cpu`) and a `DOTLLM_CPU_ISA` leg for
`scalar`, `v128` and `avx2`.

## Tests

| Test | Guards |
|---|---|
| `Cpu/CpuFeaturesTests` | cap parsing, flag lattice (`Avx512Bw` => `Avx512F` => `Avx2`, nothing above the hardware), the cap when the env var is set |
| `Cpu/Kernels/CpuTierMatrixTests` | every tier the host can run (Q8_0 Sse/Avx2/Avx512/Vnni; Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL and Q2_K..Q6_K Sse/Avx2; the Q8_0/Q8_1/Q8_K quantizers; LittleBit sign-dots; `LoraStage2`) against the scalar oracle; the Q8_0 `ComputeRows` dispatcher under the tier in force |
| `Cpu/CpuGatingConventionTests` | (1) no raw `IsSupported` outside `CpuFeatures`, (2) a non-private method that calls ISA intrinsics guards or names its tier, (3) a non-private caller of a tier-named kernel consults `CpuFeatures` (or a predicate built on it) |

The scan is a regex + brace-matching heuristic over `src/` (no Roslyn): cheap, reports `file:line`, and errs towards reporting. It cannot
prove a dispatcher's branch logic right; the dynamic backstop is the CI legs that run the whole suite on a lower tier.

## Audit (state of `origin/dev` at 7fdc1aa4, before this change)

Method: (a) enumerate every file with `using System.Runtime.Intrinsics` and every method that calls an ISA class
(`audit` of 154 intrinsic-calling methods; leaves are tier-named or private); (b) for every non-leaf caller check the guard; (c) run
the full CPU-side unit suite under `DOTNET_EnableAVX=0 DOTNET_EnableAVX2=0 DOTNET_EnableAVX512F=0` (and single-feature variants).

### Counts by class

| Class | Count | Sites |
|---|---|---|
| A. Guarded, with a scalar / lower-tier fallback (ok) | the other 39 of 45 files | the dispatch table below |
| B. Guarded, **no fallback** (throws `PlatformNotSupportedException`) | 4 | `LittleBitLayer.Gemv` (`Kernels/Experimental/LittleBit.cs:311-312`), NanoQuant loader (`Models/Architectures/TransformerWeights.cs:1448`), LittleBit loader (`TransformerWeightsSafetensors.cs:59`), `LoraStage2.ApplyF32_R16` (`Kernels/LoraStage2.cs:248`) |
| C. Unguarded (bug) | 1 found, already fixed on `dev` before the audit | `MatMul.PQ2Planar.cs` (#1617; scalar fallback in d246cf5b, line 100). No other unguarded intrinsic entry exists on `dev` |
| D. Tier-set inconsistency (guard names fewer features than the kernel uses) | 1 | `IsQ8_0VnniSupported` (`Kernels/MatMulVnni.cs:54`) tested `AvxVnni` alone, but `VecDotQ8_0Vnni`/`ProcessVnniBlock` (`:92`, `:162`) use `Fma.MultiplyAdd` |
| E. Raw `X.IsSupported` instead of one mechanism | 197 sites in 38 files (81 `Avx2`, 52 `Fma`, 25 `Ssse3`, 13 `AvxVnni`, 9 `Vector128/256.IsHardwareAccelerated`, 8 `Avx512`, 8 `Sse*`, 1 `Avx`) | no behavioural bug; no way to cap a tier. Now 0 outside `CpuFeatures.cs` (guard test 1) |
| F. Tests that threw or failed instead of skipping when an ISA is absent | 5 | `LittleBitRealCheckpointTests.RealLayers_MatchPyTorchReference` (`AVX2 required`), `LittleBitModelTests` x2 (`...require AVX2`), `FlintSyntheticGoldenTests.DenseQ4_1_Prefill_Matches_LlamaCpp_Q4_1_Golden` and `DenseQ4_1_Decode_Equals_Prefill` (dequantise-then-GEMM prefill on pre-AVX2 tiers is by design; see below) |
| G. Silent-pass tests (`if (!Avx2.IsSupported) return;`) | ~46 in `tests/.../Cpu/Kernels` (known gap documented in `SilentSkipConventionTests`) | not changed here; the LittleBit/NanoQuant ones were removed because those kernels now run on every tier |

The AVX-512 paths were checked for the "F checked, BW/VL used" inconsistency: the only AVX-512 intrinsics in `src/` are `Avx512F`
(`FastMath.ExpSumAndStoreAvx512`, `LoraStage2`) and `Avx512F`+`Avx512BW` (`MatMul` Q8_0 tiers, gated on `Avx512BW`, which implies F). No
`Avx512VL`/`DQ` is used. No AVX-512 gate was moved.

### Dispatch table (file : dispatcher line -> flag -> fallback), after this change

| Family | Dispatcher (file:line) | Tier flags | Fallback | Class |
|---|---|---|---|---|
| Q8_0 x Q8_0 GEMV | `MatMul.cs:135-188` | `Avx2FmaVnni` > `Avx512Bw` > `Avx2` > `Ssse3` | scalar (`:184`) | A (D fixed) |
| Q8_0 R4 / outer-product GEMM | `MatMul.cs:415,423,2020,2054,2174,2205,2375` | `Avx512Bw`, `Avx2`, `Ssse3` | scalar | A |
| Q8_0/Q8_1/Q8_K activation quantizers | `MatMul.cs:1050-1052,1243`, `MatMulKQuants.cs:46` | `Avx512Bw`, `Avx2` | scalar | A |
| Legacy Q4_0/Q4_1/Q5_1/IQ4_NL dots | `MatMulLegacyQuants.cs:393,401,436,453,523,558` | `Avx2`, `Ssse3` | scalar | A |
| Q5_0 | `MatMulQ5_0.cs:430,445`, `MatMul.Q5_0Sse.cs:195-206` | `Avx2`, `Ssse3` | scalar | A |
| K-quant dots Q2..Q6, IQ4_XS, IQ codebooks | `MatMulKQuants.cs:653,819,1009`, `.Sse.cs:400-437`, `.IQ4XS.cs:172,180`, `.IQCodebook.cs:457,467` | `Avx2`, `Ssse3` | scalar | A |
| K-quant fast tiled (Q4_K/Q5_K/Q6_K) | `MatMulQ4KFast.cs:34,141`, `MatMulQ5KFast.cs:76`, `MatMulQ6KFast.cs:118` | `Avx2`+`Sse41` | reference kernels | A |
| NVFP4 / MXFP4 | `MatMulNvfp4.cs:195`, `.Rows8.cs:75,89`, `MatMulMxfp4.cs:202,207` | `Avx2Fma`, `Ssse3` | scalar | A |
| I2_S / PQ2_0 / PTQ1 / W2A8 | `MatMul.I2S.cs:164,197,765`, `MatMul.PQ2S.cs:105,512,564`, `MatMul.W2A8Sse.cs` via those | `Avx2`, `AvxVnni`, `Ssse3`, `Vector128Accelerated` | scalar | A |
| PQ2_0 planar (opt-in) | `MatMul.PQ2Planar.cs:22,100` | `Avx2`, `AvxVnni`, `Fma` | `PQ2PlanarRowsScalar` | A (C fixed upstream) |
| F16 GEMM/GEMV | `MatMul.F16Tile.cs:32`, `MatMul.F16Sse.cs:102` | `Fma`; `Sse3` without AVX2 | scalar | A |
| Dequantize Q4_0..Q8_0 / K-quants | `Dequantize.cs:273,477`, `DequantizeKQuants.cs:109,394,522` | `Avx2` | scalar | A |
| KV quantize/dequantize | `KvQuantize.cs:56,125,252` | `Avx2` | scalar | A |
| FusedOps (RMSNorm+quantize ...) | `FusedOps.cs:327,443,579` | `Avx2` | scalar | A |
| RoPE | `RoPE.cs:377,406,468,484` | `Avx2` (+`Fma` inline) | scalar | A |
| FastMath exp | `FastMath.cs:165,168,270` | `Avx512F` > `Avx2` > `Fma` | portable | A |
| Hadamard, GatedDeltaNet, PortableMath | `Hadamard.cs:156-190`, `GatedDeltaNetScan.cs:166,472`, `PortableMath.cs:39` | `Vector256Accelerated`, `Vector128Accelerated` | scalar | A |
| LoRA stage 2 (rank 16) | `LoraStage2.cs:ApplyF32_R16` | `Avx512F` | `ApplyF32_R16Scalar` | B fixed |
| LittleBit / NanoQuant | `LittleBit.cs:SignedDot8FDispatch/IDispatch` | `Avx2`, `Avx2`+`AvxVnni` | `SignedDot8FScalar/IScalar` | B fixed |
| Ternary adapter | `TernaryAdapterKernel.cs:108,248` | `Avx2` | scalar | A |
| Token mask / sampler | `TokenMaskApplier.cs:20`, `CategoricalSampler.cs:30` | `Avx2`, `Vector256Accelerated` | scalar | A |

### Found by running the suite on a lower tier

Running the CPU-side unit suite with the wide ISAs off (the CI leg's environment, but not limited to `DotLLM.Tests.Unit.Cpu`) produced
five failures on `dev` (class F above). Two are not bugs in a kernel but tier semantics worth knowing: on tiers below AVX2 the Q4_0/Q4_1
GEMM for n > 1 is still dequantise-then-F32-GEMM (`MatMulLegacyQuants.cs:693`, `PackedAtAnyN`), so prefill does not reproduce llama.cpp's
Q4_1 arithmetic and disagrees with the packed decode dot by ~0.77 on the synthetic Flint model. That is #1159's AVX2-only multi-column kernel
(`ComputeRowsQ4MultiAvx2_8Rows`), not a wrong answer; those two tests now skip with a reason below AVX2. Extending the packed path to the SSE
tier is a performance item (see the issues listed in the PR).

The single-feature variants `DOTNET_EnableAVX512F=0`, `DOTNET_EnableAVXVNNI=0`, `DOTNET_EnableFMA=0` changed nothing on the Zen 5 test box
(identical pass counts to the default run), i.e. no kernel in the suite was reachable through an AVX-512-without-X or VNNI-without-FMA gap there.
