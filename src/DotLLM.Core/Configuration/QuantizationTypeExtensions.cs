namespace DotLLM.Core.Configuration;

/// <summary>
/// Extension methods for <see cref="QuantizationType"/> providing byte-size calculations.
/// Block sizes match the GGUF spec and <see cref="Tensors.DType"/> static instances.
/// </summary>
public static class QuantizationTypeExtensions
{
    /// <summary>
    /// Computes the total byte count for <paramref name="elementCount"/> elements stored
    /// in the given quantization format.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Unknown quantization type.</exception>
    public static long ComputeByteCount(this QuantizationType qt, long elementCount) => qt switch
    {
        QuantizationType.F32 => elementCount * 4,
        QuantizationType.F16 => elementCount * 2,
        QuantizationType.BF16 => elementCount * 2,
        QuantizationType.Q4_0 => elementCount / 32 * 18,
        QuantizationType.Q4_1 => elementCount / 32 * 20,
        QuantizationType.Q5_0 => elementCount / 32 * 22,
        QuantizationType.Q5_1 => elementCount / 32 * 24,
        QuantizationType.Q8_0 => elementCount / 32 * 34,
        QuantizationType.Q2_K => elementCount / 256 * 84,
        QuantizationType.Q3_K => elementCount / 256 * 110,
        QuantizationType.Q4_K => elementCount / 256 * 144,
        QuantizationType.Q5_K => elementCount / 256 * 176,
        QuantizationType.Q6_K => elementCount / 256 * 210,
        QuantizationType.IQ4_NL => elementCount / 32 * 18,
        QuantizationType.IQ4_XS => elementCount / 256 * 136,
        // IQ2_XXS:  d(2) + qs[QK_K/8](uint16) = 2 + 64 = 66 bytes / 256 elements (2.0625 bpw).
        QuantizationType.IQ2_XXS => elementCount / 256 * 66,
        // IQ2_XS:   d(2) + qs[QK_K/8](uint16) + scales[QK_K/32] = 2 + 64 + 8 = 74 bytes / 256 (2.3125 bpw).
        QuantizationType.IQ2_XS => elementCount / 256 * 74,
        // IQ2_S:    d(2) + qs[QK_K/4] + qh[QK_K/32] + scales[QK_K/32] = 2 + 64 + 8 + 8 = 82 bytes / 256.
        // Also the on-disk type for MOSTLY_IQ2_M file-type recipe (~2.5625 bpw).
        QuantizationType.IQ2_S => elementCount / 256 * 82,
        // IQ1_S:    d(2) + qs[QK_K/8] + qh[QK_K/32](uint16) = 2 + 32 + 16 = 50 bytes / 256 (~1.5625 bpw).
        QuantizationType.IQ1_S => elementCount / 256 * 50,
        _ => throw new ArgumentOutOfRangeException(nameof(qt), qt,
            $"Unknown quantization type: {qt}"),
    };
}
