namespace DotLLM.Core.Attention;

/// <summary>
/// Describes the attention-mask pattern for a forward pass: the <see cref="Mode"/> plus the
/// prefix length used by <see cref="AttentionMaskMode.Hybrid"/>.
/// </summary>
/// <remarks>
/// <para>The default value (<c>default(AttentionMaskSpec)</c>) is <see cref="AttentionMaskMode.Causal"/>
/// with <see cref="PrefixLength"/> 0 — i.e. the existing autoregressive fast path. Every overload that
/// accepts a spec defaults it to <see cref="Causal"/>, so existing callers are unaffected.</para>
/// </remarks>
public readonly record struct AttentionMaskSpec
{
    /// <summary>Masking pattern. Defaults to <see cref="AttentionMaskMode.Causal"/>.</summary>
    public AttentionMaskMode Mode { get; init; }

    /// <summary>
    /// Length of the causal prompt prefix for <see cref="AttentionMaskMode.Hybrid"/>. Positions
    /// below this index stay causal; positions at or above it form the bidirectional canvas.
    /// Ignored for <see cref="AttentionMaskMode.Causal"/> and <see cref="AttentionMaskMode.Bidirectional"/>.
    /// </summary>
    public int PrefixLength { get; init; }

    /// <summary>Standard autoregressive mask. Equal to <c>default(AttentionMaskSpec)</c>.</summary>
    public static AttentionMaskSpec Causal => default;

    /// <summary>Fully bidirectional mask (no causal masking).</summary>
    public static AttentionMaskSpec Bidirectional => new() { Mode = AttentionMaskMode.Bidirectional };

    /// <summary>
    /// Hybrid mask with a causal prefix of <paramref name="prefixLength"/> positions followed by a
    /// bidirectional canvas.
    /// </summary>
    /// <param name="prefixLength">Number of leading causal prefix positions.</param>
    public static AttentionMaskSpec Hybrid(int prefixLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(prefixLength);
        return new() { Mode = AttentionMaskMode.Hybrid, PrefixLength = prefixLength };
    }

    /// <summary>True when this spec selects the default causal fast path.</summary>
    public bool IsCausal => Mode == AttentionMaskMode.Causal;
}
