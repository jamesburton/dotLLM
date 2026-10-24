using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// 2..8-row multi-column GEMVs for the dense <c>qwen35</c> projections (issue #1185). An MTP verify is K+1 rows; without these every
/// projection fell to the 128x128 coopmat GEMM tile, which on Strix made a 2-row forward cost 4.5 single-row forwards and turned a
/// 91 %-acceptance MTP run into a 2.5x slowdown.
/// </summary>
/// <remarks>
/// Only FP32-activation kernels that reduce each column in the 1-row decode GEMV's order are used (the #902 K-quant / IQ ones, F16 / F32,
/// and the Q8_0 one added here): the verify rows then reproduce plain greedy decode token for token. The Q8_1-activation MMVQ multi
/// kernel the MoE host uses for Q8_0 was measured flipping a near-tie argmax on Qwen3.8-27B, so it is not used here.
/// </remarks>
public sealed partial class VulkanQwen3HybridDenseTransformerModel
{
    /// <summary><c>DOTLLM_VK_SMALLROW_GEMV=0</c> disables the multi-column GEMVs here as on the MoE host (A/B and diagnostics).</summary>
    internal static bool SmallRowGemvEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_GEMV") != "0";

    /// <summary>Fewest rows that take the two-rows-per-workgroup Q6_K GEMV (#1193; <c>DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS</c>, as the dense host; 99 = off).</summary>
    private static readonly int SmallRowQ6KR2MinRows =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS"), out int r) && r >= 2 ? r : 3;

    /// <summary>Records <paramref name="n"/> (2..8) rows on a multi-column GEMV when one exists for the type and shape; false otherwise.</summary>
    private bool TryRecordSmallRowMatmul(nint cmdBuf, VulkanDevice.Buffer weights, QuantizationType qt,
                                         VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n)
    {
        if (!SmallRowGemvEnabled)
            return false;
        MatMulQuantGemvMultiKernel? quant = qt switch
        {
            QuantizationType.Q4_K => _kernels.MatMulQ4KMulti,
            QuantizationType.Q5_K => _kernels.MatMulQ5KMulti,
            QuantizationType.Q6_K => n >= SmallRowQ6KR2MinRows ? _kernels.MatMulQ6KMultiR2 ?? _kernels.MatMulQ6KMulti : _kernels.MatMulQ6KMulti,
            QuantizationType.IQ4_XS => _kernels.MatMulIq4XsMulti,
            _ => null,
        };
        if (quant is not null && MatMulQuantGemvMultiKernel.Accepts(n, k))
        {
            quant.Record(cmdBuf, weights, input, output, m: m, k: k, n: n);
            return true;
        }
        if (qt == QuantizationType.IQ3_S && _kernels.MatMulIq3SMulti is { } iq3s && MatMulQuantGemvMultiKernel.Accepts(n, k))
        {
            iq3s.Record(cmdBuf, weights, input, output, m: m, k: k, n: n, codebook: _kernels.MatMulIq3S.GridBuffer);
            return true;
        }
        // Mirrors the coalesced 1-row Q8_0 GEMV only; a device decoding on the block-per-lane variant keeps the GEMM.
        if (qt == QuantizationType.Q8_0 && _kernels.MatMulQ8.VariantName == "coalesced" && _kernels.MatMulQ8Multi is { } q8 && MatMulQ8_0GemvMultiKernel.Accepts(n, k))
        {
            q8.Record(cmdBuf, weights, input, output, m: m, k: k, n: n);
            return true;
        }
        if (qt == QuantizationType.F16 && _kernels.MatMulF16Multi is { } f16 && MatMulF16GemvMultiKernel.Accepts(n, k))
        {
            f16.Record(cmdBuf, weights, input, output, m: m, k: k, n: n);
            return true;
        }
        if (qt == QuantizationType.F32 && _kernels.MatMulF32Multi is { } f32 && MatMulF32GemvMultiKernel.Accepts(n, k))
        {
            f32.Record(cmdBuf, weights, input, output, m: m, k: k, n: n);
            return true;
        }
        return false;
    }
}
