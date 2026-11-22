namespace DotLLM.Core.Attention;

/// <summary>
/// Storage format of ONE side (K or V) of a KV cache. A KV cache is described by a <see cref="KvFormatPair"/>:
/// the K side and the V side choose independently, so combinations (for example K in Q8_0 and V in TurboQuant) are
/// expressed as a pair rather than by adding one enum value per combination.
/// </summary>
public enum KvSideFormat
{
    /// <summary>IEEE binary16.</summary>
    F16 = 0,

    /// <summary>IEEE binary32 (reference only: CLAUDE.md precision rule 1 forbids it on hot paths).</summary>
    F32 = 1,

    /// <summary>Q8_0 blocks of 32 elements.</summary>
    Q8_0 = 2,

    /// <summary>Q4_0 blocks of 32 elements.</summary>
    Q4_0 = 3,

    /// <summary>
    /// TurboQuant: per-head-vector Lloyd-Max codes plus a norm. A side in this format is read in code space: the query is
    /// rotated by the K rotation before the dot product and the attention output is rotated back by the V rotation.
    /// </summary>
    Tq = 4,
}

/// <summary>The K and V storage formats of a KV cache.</summary>
/// <param name="K">Format of the key side.</param>
/// <param name="V">Format of the value side.</param>
public readonly record struct KvFormatPair(KvSideFormat K, KvSideFormat V)
{
    /// <summary>Both sides in the same format.</summary>
    /// <param name="format">The shared format.</param>
    /// <returns>A symmetric pair.</returns>
    public static KvFormatPair Both(KvSideFormat format) => new(format, format);

    /// <summary>True when K and V use different formats.</summary>
    public bool IsMixed => K != V;

    /// <summary>True when the query must be rotated into code space before scoring (K is TurboQuant).</summary>
    public bool RotatesQuery => K == KvSideFormat.Tq;

    /// <summary>True when the attention output must be rotated back out of code space (V is TurboQuant).</summary>
    public bool RotatesOutput => V == KvSideFormat.Tq;
}
