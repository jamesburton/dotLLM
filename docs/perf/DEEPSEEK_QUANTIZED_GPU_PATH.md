# Quantized DeepSeek-V2 GPU Weight Paths (Tasks #9 + #10)

Design note. Maps the work needed to take real
`DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` from "metadata loads" to
"prefill + decode runs end-to-end on the RTX 3060".

## Status (2026-04-27)

- **#9-i, #9-ii, #9-iii (quantized MLA)** — COMPLETE.
- **#10-i, #10-ii, #10-iii (quantized MoE)** — COMPLETE.
- **#11 (real-checkpoint smoke)** — COMPLETE (1/2/4/8-layer V2-Lite Q4_K_M).
- **Phase A (direct quantized GEMV in CudaMoeFfn)** — COMPLETE in commit
  `4a09dbe`. Replaces the dequant→F16→F32→LinearF32 chain with
  `LaunchConvertF32ToF16` + `LaunchQuantizedGemv` + `LaunchConvertF16ToF32`
  for K-aligned projections (gate_proj + up_proj at hidden=2048).
  down_proj at K=intermediate=1408 stays on the dequant fallback (K not
  256-aligned). This is a perf-only improvement; doesn't move the GPU
  memory cap.
- **8-layer ceiling on RTX 3060 12 GB** — empirical, see Round 10.
  16+ layers GPU-OOM. The path to fitting more layers is **NOT this
  document's design**; it's separate (expert offloading or grouped-GEMM
  cross-expert batching).

## Constraint summary

- DeepSeek-V2-Lite has 16B params on disk at Q4_K_M (~10.4 GB).
- F32 dequant of the full MoE expert block: **~57 GB host RAM** (64 experts ×
  3 projections × 1408 × 2048 × 4 bytes × 26 MoE layers). Untenable.
- F16 dequant on GPU once at load: **~28 GB GPU RAM**. Doesn't fit in 12 GB.
- **Only viable path**: keep raw Q4_K bytes on GPU, dequant per call.

## What's already in place (from prior sessions)

- `quantized_gemv_*` kernels for Q4_K / Q5_K / Q6_K / Q8_0 (`native/kernels/quantized_gemv*.cu`)
- `LaunchDequantize` GPU kernel (Q4_K → F16 device scratch)
- The GQA `Project` helper in `CudaTransformerModel.cs` already routes
  decode through `LaunchQuantizedGemv` and prefill through
  `LaunchDequantize` + cuBLAS HGEMM
- Pre-Q8_1 + MMVQ-large infra for fastest decode-batch=1 path
- GGUF DeepSeek-V2/V3 loader (config + MLA tensors + MoE 3D-stacked
  experts), CPU-side, F32-dequant only

## The five sub-tasks (in implementation order)

### #9-i. CPU GGUF MLA loader: skip F32 dequant, store raw quant bytes

`TransformerWeights.LoadMlaLayer` currently dequants every MLA tensor to
an F32 host buffer via `DequantToF32`. The fix:

```csharp
// Today
nint qAProj = DequantToF32(dataBase, tensors[$"{prefix}.attn_q_a.weight"], ..., owned);

// Target
(nint qAProj, QuantizationType qAQt) = LoadLinearRaw(dataBase, tensors[$"{prefix}.attn_q_a.weight"]);
```

`LoadLinearRaw` returns the mmap pointer + quant type without copying
or dequanting. The `MlaLayerWeights` struct gains parallel `nint`+`QuantizationType`
fields per projection (or a discriminated wrapper struct). The CPU
`MlaAttention.Execute` path stays F32-only by keeping the existing
`DequantToF32` route gated on a flag — but the GPU path uses the raw
pointers directly.

**Cost: ~2h.** Touches `TransformerWeights.cs`, `MlaLayerWeights`,
the CPU `MlaAttention.Execute` if it needs an F32-dequant fallback.

### #9-ii. CUDA MLA loader: upload raw Q4_K + add per-projection dequant

`CudaMlaWeightsLoader.LoadLayerF16` today reads F32 host pointers via
`UploadF32AsF16`. The new path:

```csharp
public static CudaMlaLayerWeights LoadLayerQuant(
    in TransformerLayerWeights cpuLayer, int hiddenSize, List<nint> allocs)
{
    // Upload raw quantized bytes — same RowByteSize math as GQA path
    nint qAProjQuant = UploadQuant(cpuLayer.Mla.QAProjPtr, cpuLayer.Mla.QAProjQt, qLora, hidden, allocs);
    // ... etc for q_b, kv_a, kv_b, o_proj
    return new CudaMlaLayerWeights(..., MlaPrecision.Quantized);
}
```

Add `MlaPrecision.Quantized` enum variant. Per-projection scratch sized
for the largest projection (q_b or kv_b expansion = ~2048 × 1408 × 2 = 5.6 MB).
The scratch is reused across all 27 layers' projections — single allocation.

**Cost: ~2h.** Touches `CudaMlaWeights.cs` + `CudaMlaWeightsLoader.cs`.

### #9-iii. CudaMlaAttention.ForwardF16: branch on precision

```csharp
if (layer.Precision == MlaPrecision.Quantized)
{
    // Decode (seqLen=1): direct quantized GEMV
    if (seqLen == 1 && _kernels.HasQuantizedGemv(layer.QAProjQt))
    {
        _kernels.LaunchQuantizedGemv(
            layer.QAProj, layer.QAProjQt, scratch.NormHidden, scratch.QLatent,
            qLora, hiddenSize, stream);
    }
    else
    {
        // Prefill: dequant to scratch, then HGEMM
        _kernels.LaunchDequantize(layer.QAProj, layer.QAProjQt, dequantScratch, qLora, hiddenSize, stream);
        CudaGemm.LinearF16(cublasHandle, scratch.NormHidden, dequantScratch, scratch.QLatent,
            seqLen, hiddenSize, qLora, stream);
    }
}
else { /* existing HGEMM path */ }
```

Same pattern for q_b, kv_a_proj_with_mqa, kv_b, o_proj. The scratch
buffer is owned by the caller-passed `CudaMlaScratchF16`.

**Cost: ~3h.** Touches `CudaMlaAttention.cs` (mostly the F16 sibling — it
has the existing 5 cuBLAS LinearF16 calls; each gets a precision branch).

### #10-i. CPU GGUF MoE loader: skip F32 dequant, store raw 3D-stacked

`TransformerWeights.LoadDeepSeekMoeLayer` today dequants each expert's
slice via `SliceExpertsToF32`. New path: store the raw GGUF tensor
pointer + per-expert byte stride + quant type. The `MoeLayerWeights`
struct gains a parallel "quant" view.

**Cost: ~2h.** Touches `TransformerWeights.cs`, `MoeLayerWeights`.

### #10-ii. CUDA MoE loader + Forward: per-expert quantized GEMV

```csharp
// Per-expert in CudaMoeWeightsLoader.LoadLayer:
var w1Quant = new nint[numExperts];
var w1Qt = quantType;
for (int e = 0; e < numExperts; e++)
{
    w1Quant[e] = UploadQuantSlice(rawBase, e * perExpertBytes, perExpertElements, qt, allocs);
}
```

In `CudaMoeFfn.Forward`, the per-expert SwiGLU loop branches:

```csharp
if (weights.Precision == MoePrecision.Quantized)
{
    foreach (active expert e):
        _kernels.LaunchQuantizedGemv(weights.W1Quant[e], weights.W1QuantType,
            gatheredInput, gateBatch_e_slice, intermediate, hidden, stream);
        // similar for W3 (up), then SwiGLU, then W2 (down).
}
else { /* existing F32 cuBLAS path */ }
```

For DeepSeek-V2-Lite Q4_K_M:
- 64 routed experts × 3 projections × Q4_K decode = 192 kernel launches/layer/token
- @ 22 µs/launch on WDDM = 4.2 ms/layer just for FFN dispatch overhead — too slow.
- **Mitigation**: graph capture for the active-expert subset, OR grouped-GEMM (single
  kernel that walks the bucketed assignments). Stretch goal — sequential first.

**Cost: ~4h.** Touches `CudaMoeWeights.cs` + `CudaMoeFfn.cs`.

### Phase B: cross-expert grouped-GEMM kernel — NOT YET STARTED

The literal "grouped GEMM" — single CUDA kernel that walks K_active
experts in one launch instead of issuing one cuBLAS call per active
expert per projection. Combined with raw-quant-on-the-fly weight reads
(avoiding the per-expert resident GPU buffers), this is the path to:

1. **Reducing dispatch overhead** further: today (post-Phase A) we
   issue 3 launches per active expert per projection (F32→F16, GEMV,
   F16→F32). Grouped kernel handles all K active experts in 1 launch
   per projection. For K=6, V2-Lite, 26 MoE layers: ~40% of remaining
   dispatch overhead.
2. **(Stretch) Reducing GPU resident memory**: if the kernel streams
   raw quant blocks from a single contiguous per-tensor buffer (keeping
   one ffn_*_exps tensor on GPU rather than 64 per-expert allocations),
   bookkeeping shrinks ~10-15%. Doesn't unblock 16+ layers on its own.

Sketch:

```c
extern "C" __global__ void moe_grouped_gemv_q4_k_f16(
    const half* __restrict__ x,                  // [K] F16 input row, shared
    const uint8_t* const* __restrict__ weights,  // K_active per-expert weight ptrs
    half* const* __restrict__ outputs,           // K_active per-expert output ptrs
    int M, int K, int K_active)
{
    // Each block: one (expert, output_row) pair.
    int blocks_per_expert = M / 1;  // one block per output row
    int expert_idx = blockIdx.x / blocks_per_expert;
    int row_idx = blockIdx.x % blocks_per_expert;

    if (expert_idx >= K_active) return;

    // Standard Q4_K dot product over K elements...
    // Output one F16 to outputs[expert_idx][row_idx].
}
```

**Implementation cost:** ~1-2 days focused work — new CUDA kernel +
PTX module + per-quant-type variants (Q4_K, Q5_K, Q6_K, Q8_0) + tests.
Output goes through `LaunchMoeGroupedGemv` helper that mirrors
`LaunchQuantizedGemv` but accepts arrays of pointers.

The Phase A speedup was already meaningful — Phase B's incremental win
shrinks now that the dequant scratch overhead is eliminated. Phase B's
real value is opening the door to memory-streaming variants that fit
larger models, not the dispatch reduction itself.

### #11. Smoke test on cached real DeepSeek-V2-Lite GGUF

```csharp
[SkippableFact]
public void RealGguf_DeepSeekV2Lite_PrefillDecode_OnCuda()
{
    string path = "~/.dotllm/models/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/...Q4_K_M.gguf";
    Skip.If(!File.Exists(path));

    using var gguf = GgufFile.Open(path);
    var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
    using var gpu = CudaTransformerModel.LoadFromGguf(gguf, config);
    int[] tokenIds = gpu.Tokenizer.Encode("def fibonacci(n):");

    using var prefill = gpu.Forward(tokenIds, positions: [0,1,...], deviceId: 0, kvCache: null);
    AssertFiniteLogits(prefill);

    int curTok = ArgMax(prefill);
    for (int i = 0; i < 3; i++)
    {
        using var step = gpu.Forward([curTok], [tokenIds.Length + i], 0, null);
        AssertFiniteLogits(step);
        curTok = ArgMax(step);
    }
}
```

Doesn't compare against a reference (no CPU oracle at this scale on this
box) — just asserts the path runs without crashing or producing NaN.
Real correctness validation comes from a follow-up CPU↔CUDA parity run
on the Ryzen+iGPU box (where the safetensors HF model already loads).

**Cost: ~1h.**

## Total estimated cost

**14 hours of focused work.** Likely 2 sessions of 6-8h each. The first
session lands #9-i through #9-iii (MLA only) — that already gets the
attention block on real-quant weights, even though MoE layers fall back
to F32 dequant. Second session does #10 + #11.

## Risks

1. **Quantized GEMV kernels assume `K % 256 == 0`** for Q4_K/Q5_K/Q6_K
   block alignment. V2-Lite hidden=2048 = 8×256 ✓; intermediate=10944 = 42.75×256
   ✗ for the dense FFN at layer 0 (existing GQA path may have a fallback —
   need to verify). MoE intermediate=1408 = 5.5×256 ✗ — definitely needs
   the dequant-then-HGEMM fallback for some projections.

2. **Per-expert kernel launch count** at 192/layer/token: if WDDM overhead
   stays at 22 µs, that's 4.2 ms/layer × 26 MoE layers = 109 ms FFN overhead
   per token. Decode would be < 10 tok/s. Only acceptable if we accept it
   for the v1; otherwise grouped-GEMM is required.

3. **MoE 3D-stacked tensor on-device upload**: each layer's `ffn_gate_exps`
   is `[hidden, intermediate, num_experts]` raw bytes. We need to keep
   the 3D layout AND extract per-expert slice pointers. Probably easier to
   upload the whole tensor as one contiguous buffer and compute per-expert
   slice offsets at GEMV-launch time.

## What to do first next session

1. `git pull origin feature/mamba-3-cuda` — should match `9ebba26`.
2. Confirm the 10.4 GB GGUF is still cached at
   `~/.dotllm/models/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/...Q4_K_M.gguf`.
3. Run `RealGguf_ConfigExtractor_ParsesDeepSeekV2Lite` to confirm the
   metadata extractor still works.
4. Start on #9-i. Acceptance gate: existing CPU MLA tests still pass
   AND the F32-dequant fallback path produces bit-identical results.
5. Once #9 lands, attempt the smoke test #11. Even with MoE F32-host-OOM,
   you can `DebugMaxLayers = 2` on `CudaTransformerModel` to test only
   layers 0 (dense FFN, MLA attention) — that exercises the new
   quantized MLA path without hitting MoE.
