using DotLLM.Core.Evaluation;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// The corpus-window geometry of a sliding-window perplexity run: which absolute token ranges are
/// forwarded, in which order, and which of their positions are scored.
/// </summary>
/// <remarks>
/// <para><b>Why this is a type and not a loop.</b> Layer-cycling (issue #395) runs the corpus more
/// than once — once per GPU layer window — and every pass must enumerate <em>exactly</em> the same
/// corpus windows in exactly the same order, because pass <c>p</c>'s boundary activations are
/// indexed by the position of the window in that enumeration. A second, independently written copy
/// of the <c>for (start = 0; start + L &lt;= n; start += stride)</c> loop is the obvious way for the
/// two to drift apart, and the failure mode is a plausible-looking wrong perplexity rather than an
/// exception (the same class of silent-wrong-number bug as issues #259 / #261). So
/// <see cref="PerplexityEvaluator"/> and <c>CyclingPerplexityEvaluator</c> share this one
/// enumeration.</para>
/// <para>Windowing rules are unchanged and are documented on <see cref="PerplexityOptions"/>:
/// advance (<c>Stride</c>) and scored span (<c>ContextLength - UnscoredPrefix</c>) are independent,
/// so llama.cpp's "advance a full window, score its second half" scheme is expressible.</para>
/// </remarks>
public readonly struct PerplexityWindowPlan
{
    private readonly int _tokenCount;

    private PerplexityWindowPlan(int tokenCount, int contextLength, int stride, int unscoredPrefix, int bosTokenId)
    {
        _tokenCount = tokenCount;
        ContextLength = contextLength;
        Stride = stride;
        UnscoredPrefix = unscoredPrefix;
        BosTokenId = bosTokenId;
        WindowCount = tokenCount < contextLength ? 0 : ((tokenCount - contextLength) / stride) + 1;
    }

    /// <summary>Window size in tokens, already clamped to the model's maximum context.</summary>
    public int ContextLength { get; }

    /// <summary>Tokens advanced between consecutive window starts.</summary>
    public int Stride { get; }

    /// <summary>Leading tokens of every window that serve as context only and are never scored.</summary>
    public int UnscoredPrefix { get; }

    /// <summary>BOS id substituted at each window's first slot, or <c>-1</c> when disabled.</summary>
    public int BosTokenId { get; }

    /// <summary>Number of windows this plan enumerates. Zero when the corpus is shorter than one window.</summary>
    public int WindowCount { get; }

    /// <summary>Absolute corpus index at which window <paramref name="windowIndex"/> starts.</summary>
    /// <param name="windowIndex">Zero-based window index, less than <see cref="WindowCount"/>.</param>
    /// <returns>The window's first absolute token index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside the plan.</exception>
    public int StartOf(int windowIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(windowIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(windowIndex, WindowCount);
        return windowIndex * Stride;
    }

    /// <summary>
    /// Validates the geometry in <paramref name="options"/> against a corpus length and a model
    /// context limit, and returns the resulting plan.
    /// </summary>
    /// <param name="options">Requested mode and window geometry.</param>
    /// <param name="tokenCount">Number of corpus tokens available.</param>
    /// <param name="maxContextLength">The model's maximum accepted window, used to clamp.</param>
    /// <returns>The validated plan.</returns>
    /// <exception cref="ArgumentException">The geometry cannot score any target.</exception>
    public static PerplexityWindowPlan Create(PerplexityOptions options, int tokenCount, int maxContextLength)
    {
        int context = Math.Min(options.ContextLength, maxContextLength);
        if (context < 2)
            throw new ArgumentException("Context length must be at least 2.", nameof(options));

        int stride = options.Stride;
        if (stride < 1 || stride > context)
            throw new ArgumentException(
                $"Stride must be in [1, {context}] for a context of {context}.", nameof(options));

        int prefix = options.UnscoredPrefix >= 0 ? options.UnscoredPrefix : context - stride;
        if (prefix < 1 || prefix >= context)
            throw new ArgumentException(
                $"Unscored prefix must be in [1, {context - 1}] for a context of {context}; " +
                "each scored token needs at least one token of context, and at least one token must be scored.",
                nameof(options));

        return new PerplexityWindowPlan(tokenCount, context, stride, prefix, options.BosTokenId);
    }

    /// <summary>
    /// Materialises window <paramref name="windowIndex"/>'s token ids into
    /// <paramref name="destination"/>, applying the BOS substitution when one is configured.
    /// </summary>
    /// <param name="tokens">The whole corpus.</param>
    /// <param name="windowIndex">Zero-based window index.</param>
    /// <param name="destination">Buffer of at least <see cref="ContextLength"/> elements.</param>
    /// <remarks>
    /// Copied rather than sliced because the BOS substitution must not mutate the corpus. Callers
    /// that do not need the substitution still go through here so that the two evaluation paths
    /// cannot disagree about whether it applied.
    /// </remarks>
    public void CopyWindow(ReadOnlySpan<int> tokens, int windowIndex, Span<int> destination)
    {
        int start = StartOf(windowIndex);
        tokens.Slice(start, ContextLength).CopyTo(destination);
        if (BosTokenId >= 0)
            destination[0] = BosTokenId;
    }
}
