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
## Performance — Macro-Bench (Phase 4d.3 + 4d.4)

End-to-end forward-pass throughput with and without an active adapter,
measured via `benchmarks/DotLLM.Benchmarks/Lora/LoraMacroBenchmarks.cs`.
Sister bench to the kernel-level `LoraDeltaOverheadBenchmark`.

### Methodology

- **Adapter**: deterministic synthetic, rank=16, alpha=32, covering every
  `(layer, projection)` site the standard `TransformerModel` dispatch looks
  up (q/k/v/o + gate/up/down per layer). Generated in-memory via
  `SyntheticLoraAdapter` so no real adapter checkpoint is shipped with the
  repo.
  up (q/k/v/o + gate/up/down per layer).
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
  5 raw measurements per case (1 warmup + 1 pilot + 3 measured); reported
  median + best-of-N across all 5.
- **Sampling**: greedy (`Temperature = 0`).
- **Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo, 16C/32T,
  AVX-512F+CD+BW+DQ+VL+VBMI), Windows 11, .NET SDK 10.0.103, BDN 0.14.0.
- **Base model**: `Llama-3.2-1B-Instruct.Q8_0.gguf` (16 layers,
  hidden=2048, 32 q-heads, 8 kv-heads, ffn=8192). Adapter covers
  16 layers × 7 projections = 112 sites.

### Phase 4d.4 — Q8_0 LoRA-B (2026-05-14)

Adds `LoraWeightDType.Q8_0` (B-only). The B factor is stored as Q8_0,
dequantised once per `Apply` call into a small F32 scratch (~128 KiB at
typical shapes), then the standard F32 stage-1 GEMM runs against it.
A stays F16 (its contracted axis is `rank` < 32-element block size).

| Variant | Median Prefill tok/s | Δ vs NoLora | Median Decode tok/s | Δ vs NoLora |
|---|---:|---:|---:|---:|
| NoLora | 147.83 | — | 33.78 | — |
| LoraF32 | 107.59 | −27.2% | 31.40 | −7.0% |
| LoraF16 | 108.95 | −26.3% | 38.79 | +14.8% |
| LoraBF16 | 123.27 | −16.6% | 35.04 | +3.7% |
| **LoraQ8_0** | **123.89** | **−16.2%** | **39.06** | **+15.6%** |

The Q8_0 path closes ~40% of the F32 LoRA prefill regression
(F32 −27.2% → Q8_0 −16.2%), bringing it level with BF16 and well above F16.
On decode-dominated workloads the Q8_0 LoRA actually outperforms NoLora —
the half-sized adapter weights reduce pressure on shared L2/L3 enough to
matter at decode batch=1.

The acceptance gate (≤ −10% prefill) was not fully met — the residual
−16% gap is dominated by stage-1 *activation* streaming cost (the
`(seqLen × inputDim) × 4` F32 reads), not by adapter weight bandwidth.
See "Spike notes" below for the negative result on the original Q8_0
activation-quantising path.

### Spike notes — why we didn't ship the activation-quantising Q8_0 path

The first Phase 4d.4 spike used `MatMul.GemmQ8_0` for stage 1 (mirroring
the base-model Q8_0 path). On the same hardware/fixture this measured
**~50% slower than F32 LoRA** (LoraQ8_0 prefill 73.96 vs LoraF32 105.95
tok/s, both medians). Root cause: `GemmQ8_0` quantises the entire
`(seqLen × inputDim)` activation tile per call, but stage 1 has M=rank=16
(very small) — the quantisation cost does not amortise across enough
output rows. The activation-quant overhead alone exceeded the F32 stage-1
compute. The base GEMV wins with Q8_0 because M is huge there
(per-projection M ≈ hidden = 2048+), so the activation quant is a small
share. For LoRA stage 1 the geometry is inverted.

The shipped path therefore uses Q8_0 *only as compressed weight storage*
and dequantises once per call into F32 — it gets the byte-volume win at
adapter memory residency without the activation-quant trap.

### Phase 4d.5 — Vulkan Q8_0 LoRA + CPU pre-quant-x plumbing (2026-05-14)

Two follow-ups against the Phase 4d.4 residual:

**Vulkan-side Q8_0 LoRA (Gap 1, shipped)**: `VulkanLoraAdapter.Upload`
now handles `LoraWeightDType.Q8_0` (and F16 / BF16) on the host side by
dequantising the source factor to F32 once at adapter-bind time, with
`alpha / rank` folded in. The device-side F32 fused-delta path (Agent 6's
two-dispatch B-reduce + A-accumulate kernel) consumes the resulting F32
device buffers without semantic changes, so the F32 LoRA shader stays
bit-identical for F32 adapters. Q8_0 adapters uploaded to Vulkan now
produce finite logits within Q8_0 round-trip tolerance (5e-2 abs / 5e-3
rel) vs the F32 LoRA Vulkan baseline; see
`VulkanLoraForwardParityTests.Forward_Q8_0Adapter_VulkanMatchesF32Vulkan_WithinQ8_0Tolerance`
for the parity gate.

A separate Q8_0-in-shader variant (dequant-on-the-fly in the WG-shared
reduce shader) would save ~50% of the on-device adapter byte footprint —
on Strix Halo's UMA (~256 GB/s) that adapter-class storage saving is
indistinguishable from F32 in inference throughput, so the host-dequant
path is what we ship. The shader variant remains an unexplored option
for discrete GPUs with constrained device memory.

**CPU pre-quant-x plumbing (Gap 2, plumbed but gated off)**:
`TransformerModel.ApplyLoraDelta` now accepts the pre-quantised
activation buffer + its quant type from each dispatch site (q/k/v/o +
gate/up/down). The new `LoraDelta.ApplyQ8_0BWithPreQuantX` overload
routes Q8_0-base + Q8_0-B stage 1 through `MatMul.GemmQ8_0(...,
preQuantizedInput=xQ8)` so the activation-quant cost (which killed the
original Phase 4d.4 spike at M=rank=16) is fully amortised across the
base projection.

The dispatch is gated behind `DOTLLM_LORA_FORCE_Q8_PREQUANT=1` because
direct kernel-level probing on Strix Halo (`benchmarks/LoraQ8Stage1Probe`,
not shipped — see `.continue-here-lora-final-mile.md`) showed that even
WITHOUT the activation-quant cost, `MatMul.GemmQ8_0` at M=rank=16,
K=hidden=2048, N=seqLen=512 is ~1.7× slower per call than Agent 7's
dequant-once F32 path. The Q8_0 integer-dot kernels are tuned for fat-M
projection geometries (base GEMM has M=hidden=2048+); the rank-tall LoRA
stage-1 shape doesn't amortise their per-block constant cost. Default
path therefore stays on Agent 7's shipped F32-dequant-once.

#### Macro-bench rerun — 2026-05-14, Strix Halo, same fixture

Same Llama-3.2-1B-Instruct.Q8_0 base + synthetic rank-16 adapter +
greedy sampling as Phase 4d.4. Default-path numbers (env var not set,
so the gated CPU fast path is OFF — i.e. Agent 7's shipped CPU
behaviour, with the dispatch seam refactored to carry the preQuant
buffer through):

| Variant | Median Prefill tok/s | Δ vs NoLora | Median Decode tok/s | Δ vs NoLora |
|---|---:|---:|---:|---:|
| NoLora | 153.20 | — | 40.18 | — |
| LoraF32 | 117.33 | −23.4% | 43.65 | +8.6% |
| LoraF16 | 115.35 | −24.7% | 40.28 | +0.2% |
| LoraBF16 | 114.72 | −25.1% | 36.71 | −8.7% |
| **LoraQ8_0** | **113.56** | **−25.9%** | **41.39** | **+3.0%** |

The acceptance gate (≤ −10% prefill) **was NOT met** on this run. The
prefill regression sits at −25.9% — wider than Agent 7's Phase 4d.4
measurement of −16.2%, but per-run variance on this fixture is
substantial (cross-run prefill stdev ~10 tok/s on the same code path,
visible in `allPrefillTokPerSec` per metrics JSON). Decode is +3% vs
NoLora — consistent with the Phase 4d.4 trend that decode-batch=1
benefits from the smaller adapter weight footprint.

Per the spike's blocker policy the partial improvement ships and
`.continue-here-lora-final-mile.md` documents the remaining lever: a
tiny-M-tuned Q8_0 stage-1 kernel (per-token GEMV across rank rows
rather than the existing row-tile-then-tokens pattern in
`MatMul.GemmQ8_0`).

### Phase 4d.6 — outer-product stage-2 fast path (2026-05-16)

The Phase 4d.5 macro-bench surfaced a ~−26% Q8_0 LoRA prefill regression
that the kernel-level probe pinned to `MatMul.GemmQ8_0`. The Phase 4d.6
investigation followed up with a rebuilt
`benchmarks/LoraQ8Stage1Probe` and discovered the diagnosis was
**off by one stage**: stage 1 cost ~440 µs / call (close to F32 GEMM
peak), but stage 2 cost ~4000 µs / call — ~85% of total LoRA-Apply
wall time at canonical (rank=16, K=2048, N=512, outputDim=2048)
shape.

Root cause: the production stage 2 looped tokens, then per token
called `MatMul.GemvF32(A, tmp_t, delta, outputDim, rank)` followed by
a per-token scaled `TensorPrimitives.MultiplyAdd` into `y`. The
inner `GemvF32` itself loops `outputDim` short length-rank Dot calls.
At outputDim=2048 / N=512 / rank=16 that's ~1 million Dot
invocations per LoRA-Apply call — function-entry-dominated even
though each individual Dot is just a 16-element dot product.

The fix (`src/DotLLM.Cpu/Kernels/LoraStage2.cs::ApplyF32_R16`):
when rank=16 + AVX-512 is available, route stage 2 through an
outer-product kernel that pre-broadcasts the 16 stage-1 scalars
once per token into 16 named `Vector512<float>` locals (RyuJIT
keeps them in ZMM registers across the inner loop), then sweeps
`outputDim` in tiles of 16 with a 16-FMA chain into one
tile-accumulator. Each tile reads 16 contiguous 16-float spans
from a `[rank=16, outputDim]` transposed view of A — built lazily
on first dispatch per `(layer, proj)` and cached on the
`LoraAdapter`. Per-call A dequant becomes dead work and is skipped.

Probe results (Strix Halo, µs/call, smaller=better):

| Shape (rank, K, N, outputDim) | S2 production | S2 outer-product | Speedup |
|---|---:|---:|---:|
| (16, 2048, 512, 512)  | ~1030 | ~78  | 13.2× |
| (16, 2048, 512, 2048) | ~4000 | ~580 | 6.9×  |
| (16, 2048, 512, 5632) | ~11000 | ~1580 | 7.0× |

End-to-end (stage 1 dequant-once F32 + stage 2):

| Shape | Production E2E | New (outer-product S2) | Speedup |
|---|---:|---:|---:|
| outputDim=512  | ~1420 µs | ~530 µs  | 2.7× |
| outputDim=2048 | ~4630 µs | ~1390 µs | 3.3× |
| outputDim=5632 | ~11430 µs | ~2540 µs | 4.5× |

#### Macro-bench rerun — 2026-05-16, Strix Halo, same fixture

Same Llama-3.2-1B-Instruct.Q8_0 base + synthetic rank-16 adapter +
greedy sampling. `[SimpleJob(warmupCount: 3, iterationCount: 5)]`
(per the Phase 4d.6 spec, replaces 4d.5's warmup=1 / iter=3).

| Variant | Median Prefill tok/s | Δ vs NoLora | Median Decode tok/s | Δ vs NoLora |
|---|---:|---:|---:|---:|
| NoLora | 150.03 | — | 37.57 | — |
| LoraF32 | **141.88** | **−5.4%** ✅ | 44.47 | **+18.4%** |
| LoraF16 | **138.24** | **−7.9%** ✅ | 43.09 | **+14.7%** |
| LoraBF16 | **136.45** | **−9.1%** ✅ | 40.25 | **+7.1%** |
| **LoraQ8_0** | **118.09** | **−21.3%** | **43.91** | **+16.9%** |

vs Phase 4d.5 published numbers (LoraF32 −23.4%, LoraF16 −24.7%,
LoraBF16 −25.1%, LoraQ8_0 −25.9%):

- LoraF32 prefill: −23.4% → **−5.4%** (improvement of 18 percentage points)
- LoraF16 prefill: −24.7% → **−7.9%** (improvement of 17 pp)
- LoraBF16 prefill: −25.1% → **−9.1%** (improvement of 16 pp)
- LoraQ8_0 prefill: −25.9% → **−21.3%** (improvement of 5 pp)

**Acceptance gate**: ≤−10% prefill regression vs NoLora.
- LoraF32, LoraF16, LoraBF16: all met. ✅
- LoraQ8_0: −21% prefill remains above the −10% gate. The Q8_0 path
  pays an extra per-call B dequant (Q8_0 → F32 staging, ~5–10 µs)
  that the F32/F16/BF16 paths don't, but that alone doesn't account
  for the 10-pt gap vs LoraBF16 — we suspect Q8_0 dequant SIMD
  throughput at the rank=16 × inputDim=2048 shape is the residual,
  to be confirmed with a stage-1 split bench in a follow-up.

**Decode wins outright**: every LoRA variant is now FASTER than
NoLora at decode (+7% to +18%). The smaller adapter weight footprint
relieves shared-cache pressure at decode batch=1; with the
outer-product stage 2 collapsing the per-token Dot fan-out, the
LoRA surface area is small enough that the cache-hit advantage
dominates.

#### What did NOT work (Phase 4d.6 negative results)

The original spec hypothesised that the residual −16% was a
**stage-1** problem and proposed two paths to fix it:
- **Path C** — manually-unrolled rank-specialised stage-1 kernel
  with 16 explicit `Vector256<float>` (or `Vector512<float>`) row
  accumulator locals, mirroring `MatMul.VecDotQ8_0Avx512_4Rows`'s
  technique extended to 16 rows. Both single-block and dual-block
  variants tested in `benchmarks/LoraQ8Stage1Probe`.
- **Path B** — R16 interleaved B layout (16 rows' Q8_0 blocks
  packed contiguously per K-step) with a kernel that reads
  sequentially.

Both paths produced kernels that were **2–4× SLOWER** than the
existing `MatMul.VecDotQ8_0Avx512_4Rows` wrapped in the
`ComputeRows` 4-row tile loop. Direct probe medians at canonical
shape:

| Stage-1 kernel | Mean | vs F32 dequant-once |
|---|---:|---:|
| F32 dequant-once + GemmF32 (production) | ~470 µs | 1.00× |
| GemmQ8_0 (preQuantizedInput)            | ~830 µs | 1.77× |
| Path C — explicit V256 single-block     | ~1700 µs | 3.62× |
| Path C2 — explicit V512 dual-block      | ~1620 µs | 3.45× |
| Path BC — Path C + R16 interleaved B    | ~1470 µs | 3.13× |

Why the explicit-locals approach lost: with 16 explicit
`Vector256<float>` accumulators + working-set registers (vx, absX,
ones, scratch) live across the inner `ProcessRow` calls, RyuJIT
spills despite the in-source unrolling. The 4-row pattern wins
because its working set comfortably fits in YMM registers; ramping
up to 16 simultaneous accumulators without a register window cracks
the model. R16 interleaving alone doesn't recover this — the
problem is computational throughput, not memory layout.

Why the Q8_0 stage 1 fundamentally can't beat dequant-once-F32 at
tiny-M: per dual-block, the integer-decode chain (`Sign`,
`MultiplyAddAdjacent` × 2, `ConvertToVector512Single`) competes for
the same AVX-512 ports as the productive FMA. Per-block constants
add ~50 µops vs F32's 1 FMA per K-step. F32 dequant-once moves
those constants out of the inner loop entirely. At M=rank=16 the
amortisation across rows is too thin to flip the trade.

#### What we shipped instead

The probe surfaced that the bench-published −16/−26% prefill
regression was **not** a stage-1 problem; stage 1 was within 30%
of FMA peak. The dominant cost was stage 2 (~85% of LoRA-Apply at
canonical shape) — and stage 2 had a clean restructuring win:
swap per-token GEMV for outer-product. That's the
`LoraStage2.ApplyF32_R16` kernel that ships under Phase 4d.6.

The `DOTLLM_LORA_FORCE_Q8_PREQUANT` env-var gate (Phase 4d.5)
stays in place — it's orthogonal to the stage-2 fix and gates the
unshipped `ApplyQ8_0BWithPreQuantX` path. A future stage-1 spike
could revisit it if a tiny-M Q8_0 kernel ever wins.

### Phase 4d.3 baseline (Agent 8, 2026-05-13)

Same fixture, run before Phase 4d.4 landed. NoLora baseline is lower
(115.02 vs 147.83) because of system load variance — the *deltas* vs
NoLora are the comparable signal:

- LoraF32 −35.6%, LoraF16 −35.1%, LoraBF16 −36.0% (prefill).

Both runs agree on the central finding: F32 LoRA on a Q8_0 base regresses
prefill by ~25-36%; quantising the LoRA weight storage to BF16 or Q8_0
recovers ~10pt of that.

### Reproducing

```pwsh
# Use a specific model checkpoint:
$env:DOTLLM_BENCH_MODEL_PATH = "C:\path\to\model.gguf"

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
Per-run metrics in `%TEMP%/dotllm-bdn-metrics/Lora_*.json`; the BDN
summary surfaces best-of-N values via the custom `Prefill tok/s` and
`Decode tok/s` columns.

### Remaining headroom

1. **Stage-1 activation reuse with the base projection**. The dominant
   residual cost is the `(seqLen × inputDim)` F32 activation re-read in
   stage 1. Pre-quantising x once per layer and sharing the buffer with
   both base GEMM and LoRA stage 1 would close this — requires changes at
   the `TransformerModel.ApplyLoraDelta` dispatch site (out of scope for
   Phase 4d.4 per the spike contract).
2. **Q8_0 A factor with rank-padded layout**. A's contracted axis (rank)
   is < 32 for typical PEFT, so naive Q8_0 needs 50%+ zero padding at
   rank=16. A custom rank-aware Q8_0 variant (e.g. one block holds two
   consecutive A rows packed) would recover the byte savings — but the
   spike result suggests A bandwidth is not the bottleneck, so unclear
   if the complexity pays.
3. **SIMD-vectorise the BF16 dequant** in `LoraDelta.DequantToF32`. The
   current scalar `ReadUInt16LittleEndian` per-element loop is ~8× slower
   than the F16 `TensorPrimitives.ConvertToSingle` SIMD path; this
   accounts for BF16 lagging F16 at decode (Agent 8's −15% finding).
4. **Reuse adapter scratch across projections in a layer**. The current
   path rents `tmp[seq, rank]`, `delta[out]`, and (for Q8_0/F16/BF16) the
   B/A dequant scratches per `Apply` call. Hoisting to a per-layer scratch
   pool would save ~9 rent/return pairs per layer per forward — small
   individually but compounding.
