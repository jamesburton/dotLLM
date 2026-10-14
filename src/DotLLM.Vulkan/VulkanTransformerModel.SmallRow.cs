using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using QuantType = DotLLM.Core.Configuration.QuantizationType;

namespace DotLLM.Vulkan;

public sealed partial class VulkanTransformerModel
{
    // Small-row dispatch (#996). A 2..4 row forward (speculative-decode verify, scheduler batched decode) used to send every dense projection to
    // the GEMM / MMQ kernels tuned for prefill: on MiMo-V2.6-Flash BPW2.0 the Q6_K qkv/o projections cost 3x their 1-row GEMV, the F32 router 6-12x.
    // The multi-column GEMVs below read each weight once for all N activation rows, with the per-column accumulation order of the 1-row GEMV.

    private static int EnvRows(string name, int fallback, int max)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= 0 ? Math.Min(v, max) : fallback;

    /// <summary>Largest row count served by the Q6_K multi-column GEMV (default 4; <c>DOTLLM_VK_SMALLROW_Q6K_MAX_ROWS</c>; 0 restores the MMQ / GEMM path). Runtime A/B switch.</summary>
    public int SmallRowQ6KMaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_Q6K_MAX_ROWS", 4, MatMulQuantGemvMultiKernel.MaxColumns);
    /// <summary>Largest row count served by the F32 multi-column GEMV (router, F32 projections; default 4; <c>DOTLLM_VK_SMALLROW_F32_MAX_ROWS</c>; 0 = off).</summary>
    public int SmallRowF32MaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_F32_MAX_ROWS", 4, MatMulF32GemvMultiKernel.MaxColumns);
    /// <summary>Largest row count served by the Q8_0 MMVQ multi-column GEMV (lm_head, Q8_0 projections; default 4; <c>DOTLLM_VK_SMALLROW_Q8_MAX_ROWS</c>; 0 = off).</summary>
    public int SmallRowQ8MaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_Q8_MAX_ROWS", 4, MatMulQ8_0MmvqMultiKernel.MaxColumns);

    /// <summary>Largest row count served by the small-row split-KV attention (MiMo-V2 and gpt-oss sink layers; default 4, hard cap <see cref="VulkanSplitKvRowsAttentionKernel.MaxRows"/>; <c>DOTLLM_VK_SMALLROW_ATTN_MAX_ROWS</c>; 0 = off).</summary>
    public int SmallRowAttnMaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_ATTN_MAX_ROWS", 4, VulkanSplitKvRowsAttentionKernel.MaxRows);
    /// <summary>Attention layers dispatched on the small-row split-KV kernel (test / diagnostic).</summary>
    public long SmallRowAttnCount { get; private set; }

    /// <summary>Dispatch counters of the small-row multi-column GEMVs (test / diagnostic).</summary>
    public long SmallRowQ6KCount { get; private set; }
    /// <inheritdoc cref="SmallRowQ6KCount"/>
    public long SmallRowF32Count { get; private set; }
    /// <inheritdoc cref="SmallRowQ6KCount"/>
    public long SmallRowQ8Count { get; private set; }

    private void CreateSmallRowKernels(VulkanDevice device, string spvDir)
    {
        _smallRowQ6K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K);
        _smallRowF32 = MatMulF32GemvMultiKernel.TryCreate(device, spvDir);
        _smallRowQ8 = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir);
        if (_splitKvAttention is not null) _smallRowAttn = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir);
    }

    private bool SmallRowQ8Fits(int inputDim, int seqLen)
        => _smallRowQ8 is not null && _quantizeQ8_1Rows is not null && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
           && (inputDim % QuantizeQ8_1RowsKernel.GroupSize) == 0
           && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
           && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size;

    /// <summary>True when <see cref="TryRecordSmallRowGemv"/> would serve a projection of <paramref name="qt"/> weights over <paramref name="seqLen"/> rows.</summary>
    private bool SmallRowGemvApplies(QuantType qt, int inputDim, int seqLen)
    {
        if (seqLen < 2) return false;
        return qt switch
        {
            QuantType.Q6_K => seqLen <= SmallRowQ6KMaxRows && _smallRowQ6K is not null && MatMulQuantGemvMultiKernel.Accepts(seqLen, inputDim),
            QuantType.F32 => seqLen <= SmallRowF32MaxRows && _smallRowF32 is not null && MatMulF32GemvMultiKernel.Accepts(seqLen, inputDim),
            QuantType.Q8_0 => seqLen <= SmallRowQ8MaxRows && MatMulQ8_0MmvqMultiKernel.Accepts(seqLen, inputDim) && SmallRowQ8Fits(inputDim, seqLen),
            _ => false,
        };
    }

    private bool TryRecordSmallRowGemv(
        nint cmdBuf, VulkanDevice.Buffer weights, QuantType qt, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (!SmallRowGemvApplies(qt, inputDim, seqLen)) return false;
        switch (qt)
        {
            case QuantType.Q6_K:
                SmallRowQ6KCount++;
                _smallRowQ6K!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("q6_k_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.F32:
                SmallRowF32Count++;
                _smallRowF32!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("f32_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.Q8_0:
                SmallRowQ8Count++;
                _quantizeQ8_1Rows!.Record(cmdBuf, input, _state.Q8_1XqRows!, _state.Q8_1XdsRows!, n: seqLen, k: inputDim);
                BarrierComputeToCompute(cmdBuf);
                _smallRowQ8!.Record(cmdBuf, weights, _state.Q8_1XqRows!, _state.Q8_1XdsRows!, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("q8_0_multi", outputDim, inputDim, seqLen);
                return true;
            default:
                return false;
        }
    }
}
