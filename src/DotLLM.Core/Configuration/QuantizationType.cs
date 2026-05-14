namespace DotLLM.Core.Configuration;

/// <summary>
/// GGUF quantization type identifiers. Values match the GGUF spec.
/// </summary>
public enum QuantizationType
{
    /// <summary>32-bit IEEE float.</summary>
    F32 = 0,

    /// <summary>16-bit IEEE float.</summary>
    F16 = 1,

    /// <summary>4-bit quantization, group size 32, no min.</summary>
    Q4_0 = 2,

    /// <summary>4-bit quantization, group size 32, with min.</summary>
    Q4_1 = 3,

    /// <summary>5-bit quantization, group size 32, no min.</summary>
    Q5_0 = 6,

    /// <summary>5-bit quantization, group size 32, with min.</summary>
    Q5_1 = 7,

    /// <summary>8-bit quantization, group size 32.</summary>
    Q8_0 = 8,

    /// <summary>4-bit K-quant, super-block of 256.</summary>
    Q4_K = 12,

    /// <summary>5-bit K-quant, super-block of 256.</summary>
    Q5_K = 13,

    /// <summary>6-bit K-quant, super-block of 256.</summary>
    Q6_K = 14,

    /// <summary>
    /// 1.5-bit importance quantization (~1.5625 bpw — the smallest GGUF quant
    /// type). Super-block of 256 elements stored in 50 bytes:
    /// <c>d(Half@0) + qs[32]@2 + qh[8](uint16)@34</c>. Each 32-element sub-block
    /// uses one <c>qh</c> uint16 to encode a 3-bit per-block scale (top 3 bits),
    /// a sign-of-delta bit (bit 15), and four 3-bit grid-index high parts (bits
    /// 0..11). The 11-bit grid index per group of 8 elements selects from a
    /// 2048-entry signed-int8 codebook (each entry packs 8 ternary {-1, 0, +1}
    /// values into a uint64). Per-element decode:
    /// <c>y = dl * (grid[j] + delta)</c> with <c>delta = +/-0.125</c>.
    /// </summary>
    IQ1_S = 19
}
