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
    public int SmallRowQ6KMaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_Q6K_MAX_ROWS", MatMulQuantGemvMultiKernel.WideMaxColumns, MatMulQuantGemvMultiKernel.WideMaxColumns);
    /// <summary>Smallest row count that takes the two-output-rows-per-workgroup Q6_K kernel (default 3: at 2 columns the single-row kernel is as fast; 3..4 columns are load-issue bound there; <c>DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS</c>; 99 = never).</summary>
    public int SmallRowQ6KR2MinRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS", 3, 99);
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

    /// <summary>
    /// <c>DOTLLM_VK_SMALLROW_K_GEMV=0</c> keeps Q4_K / Q5_K / IQ4_XS 2..16-row forwards on the MMQ / GEMM path (#917 A/B). Runtime switch.
    /// </summary>
    internal static bool SmallRowKQuantEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_K_GEMV") != "0";

    /// <summary>Largest row count served by the K-quant multi-column GEMVs here (#917; <c>DOTLLM_VK_SMALLROW_K_MAX_ROWS</c>, default 16, hard cap 16; 0 = off).</summary>
    public int SmallRowKMaxRows { get; set; } = EnvRows("DOTLLM_VK_SMALLROW_K_MAX_ROWS", MatMulQuantGemvMultiKernel.WideMaxColumns, MatMulQuantGemvMultiKernel.WideMaxColumns);

    /// <summary>Dispatch counter of the Q4_K / Q5_K / IQ4_XS multi-column GEMVs (test / diagnostic).</summary>
    public long SmallRowKQuantCount { get; private set; }

    /// <summary>
    /// <c>DOTLLM_VK_SMALLROW_LEGACYQ=0</c> keeps Q8_0 5..16-row and Q4_0 / Q5_0 2..16-row forwards on the MMQ / GEMM path (#1420 A/B). Runtime switch.
    /// </summary>
    internal static bool SmallRowLegacyQEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_LEGACYQ") != "0";

    /// <summary>Dispatch counter of the #1420 Q8_0 (5..16 rows) / Q4_0 / Q5_0 F32-activation multi-column GEMVs (test / diagnostic).</summary>
    public long SmallRowLegacyQCount { get; private set; }

    private MatMulQ8_0GemvMultiKernel? _smallRowQ8Wide;
    private MatMulExactGemvMultiKernel? _smallRowQ4_0, _smallRowQ5_0;

    /// <summary>A/B hook for VerifyRowCurve <c>ab</c>: true = K-quant multi-column GEMVs up to 16 rows (and the #1420 legacy-quant ones), false = the pre-#917 MMQ / GEMM path (and Q6_K capped at 4 rows).</summary>
    public static void SetSmallRowKQuantForBench(bool on) { SmallRowKQuantEnabled = on; SmallRowLegacyQEnabled = on; }

    private MatMulQuantGemvMultiKernel? _smallRowQ4K, _smallRowQ5K, _smallRowIq4Xs;

    private void CreateSmallRowKernels(VulkanDevice device, string spvDir)
    {
        // #917: NR rows per workgroup as on the qwen35 host (#1220): 2 for one column, 4 for 2..12 columns. #1516: 3 for 13..16 (the pipelined shader of #1511 is
        // smooth in the column count, so this is the only crossover left; 502-582 us against 570-1117 at NR 4 on the Qwen3.8-27B FFN shapes).
        _smallRowQ4K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q4_K, 2, 4, 4, new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3 });
        _smallRowQ5K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q5_K);
        _smallRowIq4Xs = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.IQ4_XS);
        _smallRowQ6K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K);
        _smallRowQ6KR2 = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_R2);
        _smallRowF32 = MatMulF32GemvMultiKernel.TryCreate(device, spvDir);
        _smallRowQ8 = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir);
        // #1420: FP32-activation kernels for what the Q8_1 MMVQ multi (<= 4 rows) and the missing Q4_0 / Q5_0 multi kernels left on the GEMM.
        _smallRowQ8Wide = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir, 2, 4, 4);
        _smallRowQ4_0 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Q4_0, 2, 4);
        _smallRowQ5_0 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Q5_0, 2, 4);
        if (_splitKvAttention is not null) _smallRowAttn = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir);
    }

    /// <summary>Test hook: whether a <c>[seqLen, inputDim]</c> activation currently fits the multi-row Q8_1 scratch (#1016).</summary>
    internal bool Q8RowsFitForTest(int seqLen, int inputDim) => _state.Q8_1RowsFit(seqLen, inputDim);

    private bool SmallRowQ8Fits(int inputDim, int seqLen)
        => _smallRowQ8 is not null && _quantizeQ8_1Rows is not null && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
           && (inputDim % QuantizeQ8_1RowsKernel.GroupSize) == 0
           && _state.Q8_1RowsFit(seqLen, inputDim);

    /// <summary>True when <see cref="TryRecordSmallRowGemv"/> would serve a projection of <paramref name="qt"/> weights over <paramref name="seqLen"/> rows.</summary>
    private bool SmallRowGemvApplies(QuantType qt, int inputDim, int seqLen)
    {
        if (seqLen < 2) return false;
        return qt switch
        {
            QuantType.Q4_K => seqLen <= SmallRowKMaxRows && KQuantFor(qt) is { } q4 && q4.AcceptsColumns(seqLen, inputDim),
            QuantType.Q5_K => seqLen <= SmallRowKMaxRows && KQuantFor(qt) is { } q5 && q5.AcceptsColumns(seqLen, inputDim),
            QuantType.IQ4_XS => seqLen <= SmallRowKMaxRows && KQuantFor(qt) is { } iq && iq.AcceptsColumns(seqLen, inputDim),
            QuantType.Q6_K => seqLen <= (SmallRowKQuantEnabled ? SmallRowQ6KMaxRows : Math.Min(4, SmallRowQ6KMaxRows)) && _smallRowQ6K is { } q6 && q6.AcceptsColumns(seqLen, inputDim),
            QuantType.F32 => seqLen <= SmallRowF32MaxRows && _smallRowF32 is not null && MatMulF32GemvMultiKernel.Accepts(seqLen, inputDim),
            QuantType.Q8_0 => (seqLen <= SmallRowQ8MaxRows && MatMulQ8_0MmvqMultiKernel.Accepts(seqLen, inputDim) && SmallRowQ8Fits(inputDim, seqLen))
                              || SmallRowQ8WideApplies(inputDim, seqLen),
            QuantType.Q4_0 => SmallRowLegacyQEnabled && seqLen <= MatMulExactGemvMultiKernel.MaxColumns && _smallRowQ4_0 is { } q40 && q40.Accepts(seqLen, inputDim),
            QuantType.Q5_0 => SmallRowLegacyQEnabled && seqLen <= MatMulExactGemvMultiKernel.MaxColumns && _smallRowQ5_0 is { } q50 && q50.Accepts(seqLen, inputDim),
            _ => false,
        };
    }

    /// <summary>True when rows beyond the Q8_1 MMVQ multi kernel's cap (and up to 16) run the FP32-activation Q8_0 multi GEMV (#1420).</summary>
    private bool SmallRowQ8WideApplies(int inputDim, int seqLen)
        => SmallRowLegacyQEnabled && Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_Q8WIDE") != "0" && seqLen > SmallRowQ8MaxRows && seqLen <= MatMulQ8_0GemvMultiKernel.WideMaxColumns
           && _smallRowQ8Wide is not null && MatMulQ8_0GemvMultiKernel.AcceptsWide(seqLen, inputDim);

    private MatMulQuantGemvMultiKernel? KQuantFor(QuantType qt)
        => !SmallRowKQuantEnabled ? null : qt switch
        {
            QuantType.Q4_K => _smallRowQ4K,
            QuantType.Q5_K => _smallRowQ5K,
            QuantType.IQ4_XS => _smallRowIq4Xs,
            _ => null,
        };

    private bool TryRecordSmallRowGemv(
        nint cmdBuf, VulkanDevice.Buffer weights, QuantType qt, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (!SmallRowGemvApplies(qt, inputDim, seqLen)) return false;
        switch (qt)
        {
            case QuantType.Q4_K:
            case QuantType.Q5_K:
            case QuantType.IQ4_XS:
                SmallRowKQuantCount++;
                KQuantFor(qt)!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote(qt == QuantType.Q4_K ? "q4_k_multi" : qt == QuantType.Q5_K ? "q5_k_multi" : "iq4_xs_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.Q6_K:
                SmallRowQ6KCount++;
                if (seqLen >= SmallRowQ6KR2MinRows && seqLen <= MatMulQuantGemvMultiKernel.MaxColumns && _smallRowQ6KR2 is { } r2)
                {
                    r2.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                    ProfNote("q6_k_multi_r2", outputDim, inputDim, seqLen);
                }
                else
                {
                    _smallRowQ6K!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                    ProfNote("q6_k_multi", outputDim, inputDim, seqLen);
                }
                return true;
            case QuantType.F32:
                SmallRowF32Count++;
                _smallRowF32!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("f32_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.Q4_0:
                SmallRowLegacyQCount++;
                _smallRowQ4_0!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("q4_0_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.Q5_0:
                SmallRowLegacyQCount++;
                _smallRowQ5_0!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("q5_0_multi", outputDim, inputDim, seqLen);
                return true;
            case QuantType.Q8_0 when SmallRowQ8WideApplies(inputDim, seqLen):
                SmallRowLegacyQCount++;
                _smallRowQ8Wide!.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                ProfNote("q8_0_multi_f32", outputDim, inputDim, seqLen);
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
