using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Nemotron-H MoE (<c>nemotron_h_moe</c>, Nemotron 3.5 Bolt / Lightning) routed-expert kernels from
/// <c>nemotron_moe.ptx</c> (issue #403): the expert-grouped GEMV and the slot-ordered weighted sum.
/// </summary>
public sealed unsafe partial class CudaKernels
{
    /// <summary>Weight rows per thread block of the expert-grouped GEMV (one warp per row); mirrors <c>MOE_EG_WARPS</c>.</summary>
    internal const int MoeExpertGroupedRowsPerBlock = 4;

    private readonly CudaModule? _nemotronMoeModule;
    private nint _moeEgGemvQ4KFunc;
    private nint _moeEgGemvQ5KFunc;
    private nint _moeEgGemvQ6KFunc;
    private nint _moeEgGemvQ8_0Func;
    private nint _moeEgGemvIQ2XxsFunc;
    private nint _moeWeightedSlotSumFunc;

    /// <summary>Resolves the <c>nemotron_moe.ptx</c> entry points; called once from the constructor.</summary>
    private void LoadNemotronMoeFunctions()
    {
        var m = _nemotronMoeModule!;
        _moeEgGemvQ4KFunc = m.GetFunction("moe_eg_gemv_q4_k_f32");
        _moeEgGemvQ5KFunc = m.GetFunction("moe_eg_gemv_q5_k_f32");
        _moeEgGemvQ6KFunc = m.GetFunction("moe_eg_gemv_q6_k_f32");
        _moeEgGemvQ8_0Func = m.GetFunction("moe_eg_gemv_q8_0_f32");
        _moeEgGemvIQ2XxsFunc = m.GetFunction("moe_eg_gemv_iq2_xxs_f32");
        _moeWeightedSlotSumFunc = m.GetFunction("moe_weighted_slot_sum_f32");
    }

    /// <summary>Whether <see cref="LaunchMoeExpertGroupedGemvF32"/> has a kernel for expert banks stored as <paramref name="qt"/>.</summary>
    /// <param name="qt">Expert bank quantization type.</param>
    /// <returns><see langword="true"/> for Q4_K, Q5_K, Q6_K, Q8_0 and IQ2_XXS.</returns>
    public static bool SupportsMoeExpertGroupedGemv(QuantizationType qt)
        => qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q8_0 or QuantizationType.IQ2_XXS;

    /// <summary>
    /// Contraction length granularity the expert-grouped GEMV needs for <paramref name="qt"/>: 32 for Q8_0
    /// (whole blocks), 256 for the K-quants and IQ2_XXS (whole super-blocks).
    /// </summary>
    /// <param name="qt">Expert bank quantization type.</param>
    /// <returns>The required divisor of K.</returns>
    public static int MoeExpertGroupedKMultiple(QuantizationType qt) => qt == QuantizationType.Q8_0 ? 32 : 256;

    /// <summary>
    /// Expert-grouped GEMV (issue #403): for every group <c>g</c> in <c>[0, groups)</c> and weight row <c>r</c> in
    /// <c>[0, m)</c>, <c>y[slot, r] = act(Σ_k W_g[r, k] · x[xRow[slot], k])</c> for each slot of the group, where
    /// <c>W_g</c> is the <c>m × k</c> quantized matrix at <c>groupWeights[g]</c>, the group's slots are
    /// <c>slotIds[groupStart[g] .. groupStart[g+1])</c>, and <c>act</c> is relu² when <paramref name="relu2"/> is set.
    /// </summary>
    /// <param name="qt">Bank quantization type (see <see cref="SupportsMoeExpertGroupedGemv"/>).</param>
    /// <param name="groupWeights">Device array of <paramref name="groups"/> device pointers, one expert matrix per group.</param>
    /// <param name="groupStart">Device <c>int[groups + 1]</c>: slot range of each group in <paramref name="slotIds"/>.</param>
    /// <param name="slotIds">Device <c>int[]</c>: slot ids ordered by group.</param>
    /// <param name="xRow">Device <c>int[]</c> indexed by slot id: the row of <paramref name="x"/> each slot reads; 0 means the slot's own id.</param>
    /// <param name="x">Device FP32 activations, row stride <paramref name="k"/>.</param>
    /// <param name="y">Device FP32 output, row = slot id, stride <paramref name="m"/>.</param>
    /// <param name="m">Weight rows (output width).</param>
    /// <param name="k">Contraction length; a multiple of <see cref="MoeExpertGroupedKMultiple"/>.</param>
    /// <param name="groups">Number of groups (used experts).</param>
    /// <param name="relu2">Fuse <c>max(0, v)²</c> on the output.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <exception cref="NotSupportedException"><paramref name="qt"/> has no kernel.</exception>
    /// <exception cref="ArgumentException"><paramref name="k"/> is not a multiple of the type's block.</exception>
    public void LaunchMoeExpertGroupedGemvF32(QuantizationType qt, nint groupWeights, nint groupStart, nint slotIds,
        nint xRow, nint x, nint y, int m, int k, int groups, bool relu2, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q4_K => _moeEgGemvQ4KFunc,
            QuantizationType.Q5_K => _moeEgGemvQ5KFunc,
            QuantizationType.Q6_K => _moeEgGemvQ6KFunc,
            QuantizationType.Q8_0 => _moeEgGemvQ8_0Func,
            QuantizationType.IQ2_XXS => _moeEgGemvIQ2XxsFunc,
            _ => throw new NotSupportedException(
                $"The CUDA Nemotron-H MoE expert kernel has no {qt} variant (supported: Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS)."),
        };
        int kMul = MoeExpertGroupedKMultiple(qt);
        if (k <= 0 || k % kMul != 0)
            throw new ArgumentException($"{qt} expert GEMV needs k to be a positive multiple of {kMul}, got {k}.", nameof(k));
        if (m <= 0 || groups <= 0) return;

        nint wArg = groupWeights, gsArg = groupStart, sArg = slotIds, xrArg = xRow, xArg = x, yArg = y;
        int mArg = m, kArg = k, reluArg = relu2 ? 1 : 0;
        void** args = stackalloc void*[] { &wArg, &gsArg, &sArg, &xrArg, &xArg, &yArg, &mArg, &kArg, &reluArg };
        uint gridX = (uint)((m + MoeExpertGroupedRowsPerBlock - 1) / MoeExpertGroupedRowsPerBlock);
        CudaDriverApi.cuLaunchKernel(func,
                gridX, (uint)groups, 1, (uint)(32 * MoeExpertGroupedRowsPerBlock), 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// <c>acc[t, d] = Σ_{k=0..topK-1} w[t·topK + k] · ySlot[t·topK + k, d]</c>, summed in slot order with separate
    /// multiply and add, as the CPU Nemotron-H MoE body does (issue #403).
    /// </summary>
    /// <param name="ySlot">Device FP32 per-slot outputs, <c>[nTok · topK, d]</c>.</param>
    /// <param name="weights">Device FP32 gating weights, <c>[nTok · topK]</c>.</param>
    /// <param name="acc">Device FP32 output, <c>[nTok, d]</c>.</param>
    /// <param name="nTok">Tokens.</param>
    /// <param name="topK">Slots per token.</param>
    /// <param name="d">Row width.</param>
    /// <param name="stream">CUDA stream.</param>
    public void LaunchMoeWeightedSlotSumF32(nint ySlot, nint weights, nint acc, int nTok, int topK, int d, nint stream)
    {
        if (nTok <= 0 || d <= 0) return;
        nint yArg = ySlot, wArg = weights, aArg = acc;
        int kArg = topK, dArg = d;
        void** args = stackalloc void*[] { &yArg, &wArg, &aArg, &kArg, &dArg };
        CudaDriverApi.cuLaunchKernel(_moeWeightedSlotSumFunc,
                (uint)((d + BlockSize - 1) / BlockSize), (uint)nTok, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
}
