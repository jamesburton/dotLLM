using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Shared, full dtype-aware CPU LoRA delta projection. Extracted from
/// <see cref="TransformerModel"/>'s former private <c>ApplyLoraDelta</c> so that
/// both the full-CPU <see cref="TransformerModel"/> and the hybrid
/// <c>HybridTransformerModel</c> (DotLLM.Cuda) route their CPU-resident layers
/// through one implementation. Supports F32 / F16 / BF16 / Q8_0 adapter weights
/// (via <see cref="LoraDelta.Apply(float*, void*, void*, float*, int, int, int, int, float, LoraWeightDType, LoraWeightDType, nint)"/>)
/// and the Phase-4d.6 outer-product stage-2 fast path
/// (<see cref="LoraStage2.EnsureATransposedF32"/>).
/// </summary>
public static unsafe class LoraProjection
{
    /// <summary>
    /// Applies the LoRA delta for <paramref name="projName"/> at
    /// <paramref name="layer"/> if <paramref name="adapter"/> targets that site:
    /// <c>y += (alpha / rank) · (x · B) · A</c>. No-op when
    /// <paramref name="adapter"/> is null or has no entry for this projection.
    /// <para>
    /// Phase 4d.5 / Gap 2 — when the caller has already quantised
    /// <paramref name="x"/> for the base projection's GEMM and passes the
    /// resulting buffer as <paramref name="preQuantX"/>, AND
    /// <paramref name="preQuantXType"/> is <see cref="QuantizationType.Q8_0"/>,
    /// AND the adapter's B factor is <see cref="LoraWeightDType.Q8_0"/>, the LoRA
    /// stage-1 GEMM re-uses the pre-quantised buffer via
    /// <see cref="LoraDelta.ApplyQ8_0BWithPreQuantX"/> instead of dequanting B to
    /// F32 and running an F32 GEMM. This closes the residual −16% prefill
    /// regression the Phase 4d.4 dequant-once path left on the table on a Q8_0
    /// base (Strix Halo / Llama-3.2-1B). The default arguments give the legacy
    /// F32 / dequant-once behaviour.
    /// </para>
    /// </summary>
    public static void Apply(ILoraAdapter? adapter, int layer, string projName,
                             float* x, float* y, int seqLen, int inputDim, int outputDim,
                             ComputeThreadPool? threadPool,
                             byte* preQuantX = null,
                             QuantizationType preQuantXType = QuantizationType.F32,
                             LoraRegion region = LoraRegion.Any)
    {
        if (adapter is null) return;
        var lora = adapter.GetLayerWeights(layer, projName, region);
        if (lora is not { } w) return;

        // Defensive shape check — IsCompatible already validated dims, but
        // we re-check at the call site so a bug in dim plumbing surfaces
        // here rather than as a silent OOB into x/y buffers.
        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' layer={layer} proj='{projName}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match base projection "
                + $"({inputDim}x{outputDim}).");

        float scale = adapter.Alpha / adapter.Rank;

        // Phase 4d.6 — outer-product stage-2 fast path. At rank=16 with
        // AVX-512 present, the per-token GEMV-then-MultiplyAdd stage 2
        // (~outputDim short Dot calls per token, ~1M total at outputDim=2048
        // / seqLen=512) is replaced by an outer-product kernel that
        // collapses to ~seqLen × outputDim/16 tile FMAs (~3-4× faster on
        // Strix Halo). The kernel consumes a [rank, outputDim] transposed-A
        // buffer; we lazy-build + cache it on the adapter the first time we
        // dispatch a (layer, proj) pair through this path. The cache also
        // covers F16 / BF16 / Q8_0-B adapters — the dequant-and-transpose
        // happens once at first use.
        //
        // Region-tagged (Encoder/Decoder) entries skip this cache: InstallATransposedHandle
        // only ever writes back into the region-UNAWARE _layers dictionary, so caching a
        // region-tagged w's transposed-A there would either silently miss (leaking the
        // freshly-built buffer every call) or, worse, collide with a same-(layer,proj)
        // Any-region entry belonging to a different delta. Region-tagged adapters take the
        // slower per-token GEMV stage 2 instead — a documented perf-only gap, not a
        // correctness one.
        nint aTransposedHandle = region == LoraRegion.Any
            ? LoraStage2.EnsureATransposedF32(adapter as LoraAdapter, layer, projName, in w, adapter.Rank)
            : 0;

        // Phase 4d.5 / Gap 2 — fast-path plumbing: when both base and adapter
        // B are Q8_0 AND the caller pre-quantised x, we can route stage 1
        // through `MatMul.GemmQ8_0(preQuantizedInput=preQuantX)` and skip the
        // activation-quant cost. The original Phase 4d.5 spike gated this
        // behind `DOTLLM_LORA_FORCE_Q8_PREQUANT=1` because kernel-level
        // probing showed the Q8_0 GEMM at M=rank=16 was ~1.7× slower than
        // the dequant-once F32 path. Phase 4d.6 keeps the env-var gate —
        // independent of the stage-2 outer-product fix below — until a
        // tiny-M Q8_0 stage-1 kernel can win at this geometry.
        if (preQuantX is not null
            && preQuantXType == QuantizationType.Q8_0
            && w.WeightDType == LoraWeightDType.Q8_0
            && (inputDim & 31) == 0
            && Environment.GetEnvironmentVariable("DOTLLM_LORA_FORCE_Q8_PREQUANT") == "1")
        {
            LoraDelta.ApplyQ8_0BWithPreQuantX(
                preQuantX, (byte*)w.BHandle, (void*)w.AHandle, y,
                seqLen, inputDim, outputDim, adapter.Rank, scale,
                w.ResolvedAWeightDType, threadPool, aTransposedHandle);
            return;
        }

        LoraDelta.Apply((float*)x, (void*)w.BHandle, (void*)w.AHandle, (float*)y,
                        seqLen, inputDim, outputDim, adapter.Rank, scale,
                        w.WeightDType, w.ResolvedAWeightDType, aTransposedHandle);
    }
}
