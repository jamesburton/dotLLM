# LoRA Adapter Support — dotLLM

## Overview

LoRA (Low-Rank Adaptation) enables fine-tuned model behaviors without modifying base weights. Multiple adapters can coexist on the same base model, with per-request adapter selection.

## How LoRA Works at Inference

For each adapted linear layer:
```
y = x @ W + α × (x @ B) @ A
```
- `W`: frozen base weight [d_in × d_out]
- `B`: down-projection [d_in × r] (r = rank, typically 8-64)
- `A`: up-projection [r × d_out]
- `α`: scaling factor (usually `alpha / rank`)

The LoRA delta `α(xB)A` adds <5% compute overhead for typical ranks.

## Adapter Loading

### Format Support
- **SafeTensors**: Primary format. Adapter weights as `{layer_name}.lora_A.weight` and `{layer_name}.lora_B.weight`.
- **GGUF**: Possible future support for quantized adapters.

### Adapter Metadata
```
LoraAdapter:
  Name: string
  Rank: int
  Alpha: float
  TargetModules: string[]   (e.g., ["q_proj", "v_proj", "k_proj", "o_proj"])
  Layers: Dictionary<string, (A_tensor, B_tensor)>
```

## IAdapterManager Interface

```
IAdapterManager:
  LoadAdapter(name, path) → void
  UnloadAdapter(name) → void
  GetAdapter(name) → LoraAdapter?
  ListAdapters() → IReadOnlyList<string>
```

## Runtime Application

### Per-Request Adapter Selection
Each request specifies `lora_adapter: "adapter_name"` (or null for base model). The `RequestContext` carries the active adapter ID through the inference pipeline.

### Adapted Layer Forward Pass
```csharp
public Tensor Forward(Tensor input, RequestContext ctx)
{
    var output = input.MatMul(baseWeight);  // Always compute base

    if (ctx.AdapterId is not null &&
        adapterManager.GetAdapter(ctx.AdapterId) is { } adapter &&
        adapter.Layers.TryGetValue(layerName, out var lora))
    {
        var delta = input.MatMul(lora.B).MatMul(lora.A);
        output.AddInPlace(delta, scale: lora.Alpha / lora.Rank);
    }

    return output;
}
```

## Multi-Adapter Batching

In continuous batching, different sequences may use different adapters:

1. **Group by adapter**: Partition batch into groups sharing the same adapter (including "no adapter").
2. **Base matmul**: Batched across all sequences (same base weight).
3. **LoRA delta**: Computed per adapter group, added to corresponding outputs.

This is less efficient than uniform batching but the LoRA matmuls are small (low rank) so the overhead is modest.

## Design Decisions

- **No weight merging**: Adapters are never merged into base weights (`W' = W + αBA`). This enables instant switching and concurrent adapters. Trade-off: small per-layer overhead vs. large flexibility gain.
- **Adapter caching**: Loaded adapters kept in memory (GPU or CPU). Small footprint (10-100MB typical for 7B model adapter).
- **Hot loading**: Adapters can be loaded/unloaded at runtime without restarting the server.

## Performance — Macro-Bench (Phase 4d.3)

End-to-end forward-pass throughput with and without an active adapter,
measured via `benchmarks/DotLLM.Benchmarks/Lora/LoraMacroBenchmarks.cs`.
Sister bench to the kernel-level `LoraDeltaOverheadBenchmark` that reported
+9% prefill / +4% decode at TinyLlama `q_proj` shapes — this one closes the
loop at the system level.

### Methodology

- **Adapter**: deterministic synthetic, rank=16, alpha=32, covering every
  `(layer, projection)` site the standard `TransformerModel` dispatch looks
  up (q/k/v/o + gate/up/down per layer). Generated in-memory via
  `SyntheticLoraAdapter` so no real adapter checkpoint is shipped with the
  repo.
- **Scenarios**: `Prefill512` runs a ~512-token prompt + 1-token decode
  (prefill-dominated); `Decode128` runs a 32-token prompt + 128 decode
  steps (decode-dominated, batch=1).
- **Iterations**: BDN `SimpleJob(warmupCount: 1, iterationCount: 3)` —
  5 raw measurements per case (1 warmup + 1 pilot + 3 measured); the
  reported numbers are the **median** across all 5 to absorb the
  cold-cache warmup outlier without overfitting to a single best.
- **Sampling**: greedy (`Temperature = 0`) so the same prompt produces
  identical decoded tokens on every iteration, removing sampler variance.

### Results — 2026-05-13, Strix Halo (Ryzen AI Max+ 395)

| Component | Value |
|---|---|
| CPU | AMD Ryzen AI Max+ 395 (Strix Halo, 16C/32T, AVX-512F+CD+BW+DQ+VL+VBMI) |
| GPU | Radeon 8060S iGPU (not exercised — CPU backend only) |
| OS | Windows 11 Pro 10.0.26200.7019 |
| .NET | SDK 10.0.103 / Runtime 10.0.3 |
| BenchmarkDotNet | 0.14.0, InProcessEmitToolchain |
| Base model | `Llama-3.2-1B-Instruct.Q8_0.gguf` (16 layers, hidden=2048, 32 q-heads, 8 kv-heads, ffn=8192) — TinyLlama-1.1B GGUF not present in `~/.dotllm/test-cache/`, so the closest available stand-in was used. Adapter covers 16 layers × 7 projections = 112 sites. |
| Date | 2026-05-13 |

#### Prefill (512-token prompt, prefill-dominated)

| Variant | Median Prefill tok/s | Δ vs NoLora |
|---|---:|---:|
| NoLora | 115.02 | — |
| LoraF32 | 74.06 | **−35.6%** |
| LoraF16 | 74.64 | **−35.1%** |
| LoraBF16 | 73.57 | **−36.0%** |

#### Decode (32-token prompt, 128-step decode, batch=1)

| Variant | Median Decode tok/s | Δ vs NoLora |
|---|---:|---:|
| NoLora | 10.74 | — |
| LoraF32 | 11.17 | +4.0% (within noise) |
| LoraF16 | 10.57 | −1.6% (within noise) |
| LoraBF16 | 9.15 | **−14.8%** |

### Conclusion

The +9% kernel-level prefill regression is system-visible **and amplifies
sharply at the prefill scale** — observed real-checkpoint prefill drops
~36% with LoRA active across all three dtypes, vs the ~9% predicted by the
kernel bench at TinyLlama `q_proj` shapes. The disparity comes from
applying LoRA on top of a **quantised (Q8_0) base**: the base GEMM is now
memory-bandwidth-bound and very fast per FLOP, so the F32 LoRA delta —
which adds an unquantised `(seq × hidden × rank) + (seq × rank × out)`
matmul pair per projection — becomes a much larger relative share of the
forward-pass cost than the kernel bench (F32 base, F32 delta) predicted.

Decode is essentially noise-bounded: the per-step LoRA cost is small
enough relative to the per-step base forward (which is `seqLen=1` and
dominated by per-token KV-cache writes + attention) that the F32 / F16
variants land within ±5% of NoLora. The BF16 path lags by ~15%, traceable
to the scalar `ReadUInt16LittleEndian` per-element dequant loop in
`LoraDelta.DequantToF32` — F16 uses `TensorPrimitives.ConvertToSingle`
(SIMD) instead.

### Next Optimisation Opportunities

1. **Quantise the LoRA delta to match the base**. Today the base is Q8_0
   but `LoraDelta.Apply` dequantises to F32 and runs two F32 GEMMs. A
   Q8_0-LoRA path (or even an F16-LoRA path that fuses dequant into the
   GEMM inner loop) would put the delta on a roughly equal FLOP/byte
   ratio to the base — closing the bulk of the prefill regression.
2. **SIMD-vectorise the BF16 dequant**. The current scalar loop reads
   2 bytes and shifts left 16 to construct an F32; the equivalent F16
   path is ~8× faster via `TensorPrimitives.ConvertToSingle`. Either
   write a vectorised BF16→F32 routine or persist BF16 adapters as F16
   internally on load (small, one-time cost; halves CPU dequant work).
3. **Reuse adapter scratch across projections in a layer**. The current
   path rents `tmp[seq, rank]` and `delta[out]` per `Apply` call. Hoisting
   to a per-layer scratch pool would save 7 rent/return pairs per layer
   per forward — a small win individually but ~22 × 7 = 154 fewer pool
   round-trips per TinyLlama-class forward.
4. **Macro-bench the Vulkan path on Strix Halo's iGPU**. The 8060S
   integrated GPU is bandwidth-rich (UMA, 256 GB/s class) and may amortise
   the LoRA delta proportionally better than CPU — worth confirming
   before any kernel-side work lands.

### Reproducing

```pwsh
# Use a specific model checkpoint:
$env:DOTLLM_BENCH_MODEL_PATH = "C:\path\to\model.gguf"

dotnet run -c Release --project benchmarks/DotLLM.Benchmarks `
    -- --filter '*LoraMacroBenchmarks*' --invocationCount 1 --unrollFactor 1
```

Metrics are written to `%TEMP%/dotllm-bdn-metrics/Lora_*.json`; the BDN
summary table surfaces the median values via the custom `Prefill tok/s`
and `Decode tok/s` columns.
## Vulkan Backend — Fused Delta Path

The Vulkan backend ships a fused LoRA-delta GEMV (`LoraDeltaGemvFusedF32Kernel`,
shaders `lora_delta_b_reduce_f32.comp` + `lora_delta_gemv_fused_f32.comp`)
that replaces the original 4-dispatch chain
(`matmul B → matmul A → AddKernel → vkCmdCopyBuffer`) with **2 dispatches per delta site**:

1. **B-stage** — `tmp[t, r] = dot(B[r, :], x[t, :])`. One workgroup per `(t, r)` with WG=64
   threads doing a shared-memory tree reduction; same compute as the un-fused matmul step.
2. **A-stage in place** — `y[t, m] += sum_r A[m, r] * tmp[t, r]`. Each thread owns one output
   row in a WG-wide tile; writes accumulate directly into the base-projection output buffer,
   eliminating the AddKernel + vkCmdCopyBuffer tail.

`B` is pre-scaled by `alpha / rank` at upload time (`VulkanLoraAdapter.Upload`), so neither
shader carries the scale.

**Routing**: `MaybeApplyLoraDelta` selects the fused path automatically when the adapter
rank ≤ `LoraDeltaGemvFusedF32Kernel.MaxRank` (= 32, covering common PEFT defaults
4 / 8 / 16) and both `.spv` blobs are present. Larger ranks and older builds fall back
to the un-fused 4-dispatch chain. `DOTLLM_VULKAN_DISABLE_FUSED_LORA_DELTA=1` forces the
fallback path for A/B comparison.

**Bench (Strix Halo / Radeon 8060S iGPU)** — `VulkanLoraDeltaDispatchBenchmark`,
22 layers × 7 LoRA-adapted projections per token at TinyLlama-1.1B shapes
(hidden=2048, intermediate=5632), wall-clock for the full 154-site dispatch sequence:

| Rank | Un-fused (4 dispatches × 154) | Fused (2 dispatches × 154) | Speedup |
|-----:|------------------------------:|---------------------------:|--------:|
|    8 |                       17.16 ms |                     2.59 ms |   6.6×  |
|   16 |                       18.97 ms |                     3.42 ms |   5.5×  |
|   32 |                       19.87 ms |                     4.19 ms |   4.7×  |

Comfortably exceeds the ≥ 2× target on the LoRA-active decode path; the deltaSum-buffer
round-trip elimination dominates the win at decode (`seqLen=1`) shapes.
