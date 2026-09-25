# Synthetic Gemma-4 / DiffusionGemma fixture — usage

Implementation of [SYNTHETIC-FIXTURE-DESIGN.md](SYNTHETIC-FIXTURE-DESIGN.md). A tiny,
deterministic, architecturally-complete `gemma4` (AR) + `diffusion-gemma` GGUF for fast
regression and cross-backend kernel work, generated entirely in-process (no checkpoint needed).

## Code map

| Piece | Location |
|---|---|
| Quantizers F32→{Q8_0,Q5_0,Q5_1,Q4_K} (inverse of `Dequantize`) | `src/DotLLM.Cpu/Kernels/Quantize.cs`, `QuantizeKQuants.cs` |
| Quantizer round-trip gate | `tests/DotLLM.Tests.Unit/Cpu/Kernels/QuantizeRoundTripTests.cs` |
| GGUF byte writer | `src/DotLLM.Models/Gguf/GgufWriter.cs` |
| Fixture builder + dims/quant config | `src/DotLLM.Models/Gguf/SyntheticGemma4Gguf.cs`, `SyntheticGemma4Config.cs` |
| Regression + all-features golden | `tests/DotLLM.Tests.Integration/Engine/SyntheticGemma4RegressionTests.cs` |
| Per-phase timing harness | `src/DotLLM.Models/Gguf/SyntheticGemma4Harness.cs` |
| Timing + emit-to-disk tests | `tests/DotLLM.Tests.Integration/Engine/SyntheticGemma4HarnessTests.cs` |

## Tiny config (block-valid)

`hidden=256, layers=6 (0-4 sliding, 5 global/V-less), head_count=4, sliding head_dim=16
(kv 2), global head_dim=32 (kv 1), dense ff=64, expert_ff=32, experts=8 top-2, vocab=256,
softcap=30, partial rope 0.25, attn scale 1.0`. Quant mix: **Q8_0** attn + dense FFN +
token_embd, **Q4_K** fused gate_up experts, **Q5_1** down experts, **F32** norms/scales/router.

> The design doc's illustrative `hidden=64` cannot carry a Q4_K row (64 % 256 ≠ 0). The tiny
> config bumps hidden to 256 so every quantized-row K dimension is block-valid for its format
> (Q4_K=256, Q5_x/Q8_0=32). `SyntheticGemma4Config.Validate()` enforces this.

## Emit the `.gguf` to disk (for the T5500 / other backends)

```csharp
using DotLLM.Models.Gguf;
SyntheticGemma4Gguf.WriteGemma4("synthetic-gemma4-tiny.gguf");               // AR
SyntheticGemma4Gguf.WriteDiffusionGemma("synthetic-diffusion-gemma-tiny.gguf"); // diffusion
// Larger compute-bound preset (still << 12 GB):
SyntheticGemma4Gguf.WriteGemma4("synthetic-gemma4-bench.gguf", SyntheticGemma4Gguf.Bench);
```

Or run the emit test, which writes both fixtures + prints their paths/sizes:

```
dotnet test tests/DotLLM.Tests.Integration -c Release \
  --filter "FullyQualifiedName~EmitFixturesToDisk"
# Override the output dir with DOTLLM_SYNTH_OUT_DIR=/path
```

## Cross-backend: same `.gguf`, any backend

The emitted file is a normal GGUF — every backend loads it through the SAME path:

```csharp
var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
using ITensor logits = model.Forward(ids, positions, deviceId, kvCache: null, adapter: null,
                                     AttentionMaskSpec.Causal);
```

- **CPU** — `deviceId: -1` (the default integration path).
- **Vulkan / CUDA / HIP** — load the same path and select the backend device id; the GGUF
  tensor layout (Q4_K/Q5_1/Q8_0 blocks) is exactly what those backends' dequant/GEMV kernels
  already consume. The tiny fixture is a fast kernel smoke; `Bench` is compute-bound enough to
  profile a GEMM/dequant kernel without the full 26B.

## Per-phase timing harness

```csharp
foreach (var row in SyntheticGemma4Harness.Run(SyntheticGemma4Gguf.Tiny, "tiny"))
    Console.WriteLine(row.ToCsv());   // phase,name,ms,tokens_per_sec
```

Phases: `gen` (fixture build), `load`, `warmup`, `prefill` (×N, with tokens/sec) + `prefill_avg`,
`diffusion_step` (×M, `[prompt|canvas]` under `Hybrid(promptLen)`). Sample (tiny, CPU dev host):

```
phase,name,ms,tokens_per_sec
gen,tiny,37.5,
load,tiny,24.3,
warmup,tiny,91.1,
prefill_avg,tiny,16.5,966.9
diffusion_step,tiny#0,16.4,
```

Timing is **coarse per-forward** (Stopwatch around `model.Forward`). Finer per-stage
(attention/FFN/MoE) timing would hang off the `IInferenceHook`/`HookPoint` diagnostics seam —
deferred to the BenchmarkDotNet project where the noise floor is controlled.

## BenchmarkDotNet

`benchmarks/DotLLM.Benchmarks` references `DotLLM.Models`, so a `[Benchmark]` can call
`SyntheticGemma4Harness.Run(...)` directly, or generate the fixture once
(`SyntheticGemma4Gguf.WriteGemma4`) in `[GlobalSetup]` and parameterise a `[Benchmark]` over
`{ CPU, Vulkan, CUDA }` × `{ Tiny, Bench }` for a rigorous cross-backend matrix on the T5500.
