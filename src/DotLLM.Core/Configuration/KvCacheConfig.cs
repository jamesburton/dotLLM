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
/// <param name="TurboQuantUseQjl">Enable TurboQuant's QJL 1-bit residual stage for unbiased
/// attention scores (spends one of the <paramref name="TurboQuantBits"/> on the correction;
/// requires bits ≥ 2). Ignored unless TurboQuant is selected.</param>
public readonly record struct KvCacheConfig(
    KvCacheDType KeyDType = KvCacheDType.F32,
    KvCacheDType ValueDType = KvCacheDType.F32,
    int MixedPrecisionWindowSize = 0,
    int TurboQuantBits = 4,
    ulong TurboQuantSeed = 0x5DEECE66DUL,
    bool TurboQuantUseQjl = false)
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
    /// Parses a CLI string to <see cref="KvCacheDType"/>, also surfacing the TurboQuant bit-width
    /// and whether the QJL residual stage is requested. TurboQuant forms:
    /// <c>turboquant</c>/<c>tq</c> (default bits); <c>tq2</c>…<c>tq8</c> (explicit bits); a trailing
    /// <c>q</c> (<c>tqq</c>, <c>tq2q</c>…<c>tq8q</c>) enables QJL. <paramref name="bits"/> is 0 when
    /// not explicitly given ("use default").
    /// </summary>
    public static KvCacheDType ParseDType(string value, out int bits, out bool useQjl)
    {
        bits = 0;
        useQjl = false;
        string v = value.ToLowerInvariant();

        if (v.StartsWith("tq", StringComparison.Ordinal))
        {
            string rest = v[2..];
            bool qjl = rest.EndsWith("q", StringComparison.Ordinal);
            if (qjl) rest = rest[..^1];

            if (rest.Length == 0)                 // "tq" or "tqq"
            {
                useQjl = qjl;
                return KvCacheDType.TurboQuant;
            }
            if (int.TryParse(rest, out int b) && b is >= 1 and <= 8)
            {
                bits = b;
                useQjl = qjl;
                return KvCacheDType.TurboQuant;
            }
            // malformed tq* falls through to the error path below
        }

        return v switch
        {
            "f32" or "fp32" => KvCacheDType.F32,
            "q8_0" or "q8" => KvCacheDType.Q8_0,
            "q4_0" or "q4" => KvCacheDType.Q4_0,
            "turboquant" => KvCacheDType.TurboQuant,
            _ => throw new ArgumentException(
                $"Unknown KV-cache type: '{value}'. Supported: f32, q8_0, q4_0, turboquant, " +
                "tq[2..8][q]  (trailing q = QJL unbiased scores).")
        };
    }

    /// <summary>Overload that surfaces the TurboQuant bit-width but not the QJL flag.</summary>
    public static KvCacheDType ParseDType(string value, out int bits) => ParseDType(value, out bits, out _);

    /// <summary>Back-compat overload: parses without surfacing TurboQuant bit-width or QJL.</summary>
    public static KvCacheDType ParseDType(string value) => ParseDType(value, out _, out _);
}
