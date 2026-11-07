using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan;

/// <summary>Which prefill (<c>seqLen &gt; 1</c>) GEMM a dense projection takes for a packed weight format.</summary>
internal enum DensePrefillGemmPath
{
    /// <summary>The f16 cooperative-matrix GEMM on F32 activations (128x128x4 blocked tile).</summary>
    Coopmat,
    /// <summary>The dp4a MMQ on Q8_1-quantized activation rows (integer dot, int32 accumulate).</summary>
    Mmq,
    /// <summary>The tiled F32-in GEMM: the correctness fallback, 10-30x slower than either of the above.</summary>
    Scalar,
}

/// <summary>
/// The dense prefill GEMM selection table, shared by the dense host (<see cref="VulkanTransformerModel"/>) and the hybrid-dense host
/// (<see cref="VulkanQwen3HybridDenseTransformerModel"/>) so both pick the same kernel for the same (format, token count) (#1149).
/// </summary>
/// <remarks>
/// <para>
/// Thresholds are the dense host's measured crossovers (8060S, Llama-3B / 8B K-quants): coopmat loses to the dp4a MMQ below ~100 tokens
/// (-24% at pp32) and wins from 128 (+10%) to 2048 (+26-29%); IQ4_XS's MMQ holds to 128 so coopmat starts at 256 (#600); Q2_K's coopmat
/// wins from 4 tokens (#596); IQ4_NL has no coopmat GEMM, so MMQ whenever the device has integer dot products (#344). Q8_0 is MMQ-first
/// on the dense host (#50) while the F32-accumulate coopmat is 1.10-1.25x faster per #1150 (open: needs an F32 reference arm), so the
/// hybrid host keeps its coopmat-first Q8_0 and this table is not consulted for Q8_0 there.
/// </para>
/// <para>
/// <c>DOTLLM_VULKAN_Q4K_COOPMAT_MIN_TOKENS</c> moves the K-quant crossover (default 128; IQ4_XS uses max(it, 256)).
/// </para>
/// </remarks>
internal static class VulkanDensePrefillGemmPolicy
{
    /// <summary>Env var moving the K-quant coopmat crossover (token count from which the coopmat GEMM replaces the dp4a MMQ).</summary>
    public const string KQuantCoopmatMinTokensEnvVar = "DOTLLM_VULKAN_Q4K_COOPMAT_MIN_TOKENS";

    /// <summary>Prefill length from which the blocked coopmat K-quant GEMM replaces the dp4a MMQ (Q3_K / Q4_K / Q5_K / Q6_K).</summary>
    public static readonly int KQuantCoopmatMinTokens =
        int.TryParse(Environment.GetEnvironmentVariable(KQuantCoopmatMinTokensEnvVar), out int t) ? t : 128;

    /// <summary>Prefill length from which the IQ4_XS coopmat GEMM replaces its MMQ (MMQ wins to 128, #600).</summary>
    public static int Iq4XsCoopmatMinTokens => Math.Max(KQuantCoopmatMinTokens, 256);

    /// <summary>
    /// True when, with both a coopmat GEMM and an MMQ available for <paramref name="qt"/>, the coopmat GEMM is the faster choice at
    /// <paramref name="seqLen"/> rows. False for formats without a coopmat GEMM, for <c>seqLen &lt;= 1</c>, and for Q8_0 (MMQ-first, #1150).
    /// </summary>
    public static bool CoopmatWins(QuantizationType qt, int seqLen)
    {
        if (seqLen <= 1) return false;
        return qt switch
        {
            QuantizationType.Q2_K => true,                                        // #596: wins from 4 tokens
            QuantizationType.Q5_0 => true,                                        // #592: the tiled F32 GEMM is 4-33x slower even at p=16
            QuantizationType.Q3_K or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K
                => seqLen >= KQuantCoopmatMinTokens,
            QuantizationType.IQ4_XS => seqLen >= Iq4XsCoopmatMinTokens,
            _ => false,                                                           // IQ4_NL (no coopmat), Q8_0 (MMQ-first), the rest
        };
    }

    /// <summary>
    /// The prefill GEMM for <paramref name="qt"/> at <paramref name="seqLen"/> rows given which kernels the host created for it.
    /// <paramref name="hasCoopmat"/> / <paramref name="hasMmq"/> must already include the format's block alignment of the input width.
    /// </summary>
    public static DensePrefillGemmPath Select(QuantizationType qt, int seqLen, bool hasCoopmat, bool hasMmq)
    {
        if (hasCoopmat && (!hasMmq || CoopmatWins(qt, seqLen))) return DensePrefillGemmPath.Coopmat;
        if (hasMmq) return DensePrefillGemmPath.Mmq;
        return hasCoopmat ? DensePrefillGemmPath.Coopmat : DensePrefillGemmPath.Scalar;
    }
}
