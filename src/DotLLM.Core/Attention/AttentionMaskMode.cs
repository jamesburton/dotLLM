namespace DotLLM.Core.Attention;

/// <summary>
/// Selects the attention masking pattern applied inside the core attention kernel.
/// </summary>
/// <remarks>
/// <para>The default for every existing autoregressive path is <see cref="Causal"/>, which is the
/// byte-identical, zero-overhead fast path. The non-causal modes exist for diffusion-style /
/// block-denoising decoders (e.g. DiffusionGemma) where a "canvas" of positions is refined in
/// parallel and must cross-attend bidirectionally.</para>
/// </remarks>
public enum AttentionMaskMode
{
    /// <summary>
    /// Standard autoregressive mask: query at position <c>p</c> attends only to keys at positions
    /// <c>&lt;= p</c>. This is the default and the existing fast path.
    /// </summary>
    Causal = 0,

    /// <summary>
    /// Fully bidirectional: every query position attends to every key position in the block
    /// (no causal mask). Sliding-window limits, when configured, still apply.
    /// </summary>
    Bidirectional = 1,

    /// <summary>
    /// Hybrid prefix/canvas mask. Positions <c>&lt; prefixLen</c> form a causal prompt prefix and
    /// stay causal among themselves; positions <c>&gt;= prefixLen</c> form a bidirectional "canvas"
    /// that attends to the full prefix and the full canvas. Sliding-window limits, when configured,
    /// still apply.
    /// </summary>
    Hybrid = 2,
}
