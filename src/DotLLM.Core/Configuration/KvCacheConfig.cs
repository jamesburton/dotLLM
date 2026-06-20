namespace DotLLM.Core.Configuration;

/// <summary>
/// Configuration for KV-cache quantization. Allows separate quantization types for keys and values,
/// plus a mixed-precision window that keeps recent tokens in full precision.
/// </summary>
/// <param name="KeyDType">Quantization format for cached keys. <see cref="KvCacheDType.F32"/> = no quantization.</param>
/// <param name="ValueDType">Quantization format for cached values. <see cref="KvCacheDType.F32"/> = no quantization.</param>
/// <param name="MixedPrecisionWindowSize">
/// Number of recent tokens kept in full precision. Older tokens beyond this window
/// are stored in <paramref name="KeyDType"/>/<paramref name="ValueDType"/> format.
/// 0 = all tokens quantized immediately.
/// </param>
/// <param name="TurboQuantBits">Bits per coordinate when <see cref="KeyDType"/> is
/// <see cref="KvCacheDType.TurboQuant"/> (1–8; 4 ≈ quality-neutral). Ignored otherwise.</param>
/// <param name="TurboQuantSeed">Deterministic rotation seed for TurboQuant; persisted so
/// rollback / prefix-cache reuse stay valid. Ignored unless TurboQuant is selected.</param>
public readonly record struct KvCacheConfig(
    KvCacheDType KeyDType = KvCacheDType.F32,
    KvCacheDType ValueDType = KvCacheDType.F32,
    int MixedPrecisionWindowSize = 0,
    int TurboQuantBits = 4,
    ulong TurboQuantSeed = 0x5DEECE66DUL)
{
    /// <summary>Default config: full precision, no quantization.</summary>
    public static KvCacheConfig Default => new();

    /// <summary>Returns true if any block-quantization (Q8_0/Q4_0) is configured.</summary>
    public bool IsQuantized =>
        KeyDType is KvCacheDType.Q8_0 or KvCacheDType.Q4_0 ||
        ValueDType is KvCacheDType.Q8_0 or KvCacheDType.Q4_0;

    /// <summary>True when TurboQuant KV is selected (applies to both keys and values).</summary>
    public bool IsTurboQuant => KeyDType == KvCacheDType.TurboQuant;

    /// <summary>
    /// Parses a CLI string to <see cref="KvCacheDType"/>. <c>turboquant</c>/<c>tq</c> select
    /// TurboQuant at the default bit-width; <c>tq2</c>…<c>tq8</c> also set
    /// <paramref name="bits"/> to that width (else <paramref name="bits"/> is 0 = "use default").
    /// </summary>
    public static KvCacheDType ParseDType(string value, out int bits)
    {
        bits = 0;
        string v = value.ToLowerInvariant();
        if (v.Length is 3 or 4 && (v.StartsWith("tq") && int.TryParse(v.AsSpan(2), out int b) && b is >= 1 and <= 8))
        {
            bits = b;
            return KvCacheDType.TurboQuant;
        }
        return v switch
        {
            "f32" or "fp32" => KvCacheDType.F32,
            "q8_0" or "q8" => KvCacheDType.Q8_0,
            "q4_0" or "q4" => KvCacheDType.Q4_0,
            "turboquant" or "tq" => KvCacheDType.TurboQuant,
            _ => throw new ArgumentException(
                $"Unknown KV-cache type: '{value}'. Supported: f32, q8_0, q4_0, turboquant (tq2..tq8).")
        };
    }

    /// <summary>Back-compat overload: parses without surfacing an explicit TurboQuant bit-width.</summary>
    public static KvCacheDType ParseDType(string value) => ParseDType(value, out _);
}
