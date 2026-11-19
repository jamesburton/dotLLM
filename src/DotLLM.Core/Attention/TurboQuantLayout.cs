namespace DotLLM.Core.Attention;

/// <summary>
/// Storage format of the per-head-vector L2 norm that accompanies TurboQuant codes.
/// </summary>
/// <remarks>
/// <see cref="F32Reference"/> exists for oracles and A/B measurement only: FP32 storage on a hot path is
/// forbidden (CLAUDE.md precision rule 1). Every other member is a shipping candidate.
/// </remarks>
public enum TqNormFormat : byte
{
    /// <summary>FP32 (4 bytes). Oracle / A-B measurement only.</summary>
    F32Reference = 0,
    /// <summary>IEEE binary16 (2 bytes).</summary>
    F16 = 1,
    /// <summary>bfloat16 (2 bytes).</summary>
    Bf16 = 2,
    /// <summary>OCP FP8 E4M3 (1 byte; max 448, no infinity).</summary>
    Fp8E4M3 = 3,
    /// <summary>OCP FP8 E5M2 (1 byte; max 57344).</summary>
    Fp8E5M2 = 4,
    /// <summary>8-bit logarithmic grid over [<c>Log8Min</c>, <c>Log8Max</c>]; code 0 is exact zero.</summary>
    Log8 = 5,
}

/// <summary>
/// Shared in-place TurboQuant (MSE stage) storage layout for one KV side: per head vector,
/// <see cref="CodeUintsPerVec"/> packed 32-bit words of <see cref="MseBits"/>-bit centroid codes plus one
/// norm in <see cref="Norm"/> format. All backends (CPU, CUDA, Vulkan) use this one geometry.
/// </summary>
/// <param name="HeadDim">Head dimension (power of two, at most 256).</param>
/// <param name="NumKvHeads">KV heads per row.</param>
/// <param name="MseBits">Bits per coordinate of the MSE stage (1-8).</param>
/// <param name="Norm">Per-vector norm storage format.</param>
public readonly record struct TurboQuantLayout(int HeadDim, int NumKvHeads, int MseBits, TqNormFormat Norm)
{
    /// <summary>
    /// The shipping per-head-vector norm format for in-place TurboQuant: <see cref="TqNormFormat.F16"/> (2 bytes).
    /// Chosen by controller ruling on the data in <c>docs/PRECISION.md</c> E17: the 1-byte formats fail clearly, F16 is
    /// not separable from the null control at the 10% line, and has 8x less error than BF16. F16 saturates above 65504
    /// (counted by <c>TurboQuantNormCodec</c>, never Inf/NaN).
    /// </summary>
    public static TqNormFormat DefaultNorm => TqNormFormat.F16;

    /// <summary>32-bit words holding the packed codes of one head vector: <c>ceil(ceil(HeadDim*MseBits/8)/4)</c>.</summary>
    public int CodeUintsPerVec => (((HeadDim * MseBits) + 7) / 8 + 3) / 4;

    /// <summary>Bytes of one stored norm.</summary>
    public int NormBytes => Norm switch
    {
        TqNormFormat.F32Reference => 4,
        TqNormFormat.F16 or TqNormFormat.Bf16 => 2,
        TqNormFormat.Fp8E4M3 or TqNormFormat.Fp8E5M2 or TqNormFormat.Log8 => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(Norm), Norm, "unknown norm format"),
    };

    /// <summary>Code bytes for <paramref name="rows"/> positions of one layer, plus a 4-byte guard word.</summary>
    public long CodeBytesPerLayer(int rows) => (long)rows * NumKvHeads * CodeUintsPerVec * 4 + 4;

    /// <summary>Norm bytes for <paramref name="rows"/> positions of one layer.</summary>
    public long NormBytesPerLayer(int rows) => (long)rows * NumKvHeads * NormBytes;
}
