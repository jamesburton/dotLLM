namespace DotLLM.Core.Evaluation;

/// <summary>
/// Scoring strategy. Chosen by the caller; the evaluator picks the execution path from
/// <see cref="IPerplexityModel.ReturnsAllRows"/>.
/// </summary>
public enum PerplexityMode
{
    /// <summary>
    /// Teacher-forced scoring over one window per stride step, each scored from a single forward
    /// pass where the backend permits it. The established in-tree methodology (the "G1 precedent"
    /// referenced by the CUDA prefill harnesses); preserved so existing quality gates keep their
    /// meaning after consolidation.
    /// </summary>
    /// <remarks>
    /// Ratio-oriented: the load-bearing signal is the OFF/ON perplexity ratio on identical tokens,
    /// not the absolute value. Not comparable to published figures.
    /// </remarks>
    TeacherForced,

    /// <summary>
    /// Sliding-window scoring. The corpus is walked in windows of
    /// <see cref="PerplexityOptions.ContextLength"/> advanced by
    /// <see cref="PerplexityOptions.Stride"/>, and each window scores only the targets beyond its
    /// <see cref="PerplexityOptions.UnscoredPrefix"/>, so every scored token carries that many
    /// tokens of context.
    /// </summary>
    /// <remarks>
    /// Absolute-value oriented: comparable to published llama.cpp figures when model, corpus,
    /// context, stride and unscored prefix all match. See
    /// <see cref="PerplexityOptions.LlamaCppDefault"/>.
    /// </remarks>
    SlidingWindow,
}

/// <summary>Configuration for a perplexity run.</summary>
/// <param name="Mode">Scoring strategy.</param>
/// <param name="ContextLength">
/// Window size <c>L</c> in tokens. Clamped to <see cref="IPerplexityModel.MaxContextLength"/>.
/// </param>
/// <param name="Stride">
/// Tokens advanced between window starts. <c>Stride == ContextLength</c> gives non-overlapping
/// windows; a smaller value overlaps them.
/// </param>
/// <param name="MaxTokens">
/// Upper bound on corpus tokens consumed; <c>0</c> means unbounded. Bounds runtime on large
/// corpora without truncating the corpus file itself.
/// </param>
/// <param name="UnscoredPrefix">
/// Leading tokens of each window used only as context and never scored. <c>-1</c> derives
/// <c>ContextLength - Stride</c>, which makes the scored ranges tile the corpus contiguously.
/// </param>
/// <param name="BosTokenId">
/// When non-negative, each window's first token is replaced by this id. llama.cpp does this for
/// every chunk, since each is evaluated as a fresh sequence; the substituted slot lies inside the
/// unscored prefix, so no scored target changes. <c>-1</c> disables the substitution.
/// </param>
/// <remarks>
/// <para><b>Advance and scored span are independent.</b> A single "stride" cannot express
/// llama.cpp's scheme: it advances by the full window yet scores only the second half, so its
/// scored ranges have gaps. Collapsing the two into one knob silently produces a different token
/// set — the same count, scored over different tokens — and therefore a figure that looks
/// comparable but is not.</para>
/// <para>Scored targets in a window starting at <c>s</c> are the absolute indices
/// <c>[s + UnscoredPrefix, s + ContextLength)</c>.</para>
/// </remarks>
public readonly record struct PerplexityOptions(
    PerplexityMode Mode,
    int ContextLength,
    int Stride,
    int MaxTokens = 0,
    int UnscoredPrefix = -1,
    int BosTokenId = -1)
{
    /// <summary>
    /// Options reproducing llama.cpp's <c>--perplexity</c> defaults for a given context:
    /// non-overlapping chunks of <paramref name="contextLength"/>, scoring the second half of each.
    /// </summary>
    /// <remarks>
    /// This is the configuration whose output is directly comparable to published llama.cpp
    /// figures. Verified against llama.cpp build 8683 (<c>d0a6dfeb2</c>).
    /// </remarks>
    /// <param name="contextLength">Window size.</param>
    /// <param name="maxTokens">Corpus token cap; <c>0</c> for unbounded.</param>
    /// <param name="bosTokenId">
    /// BOS id to substitute at the start of each chunk, mirroring llama.cpp; <c>-1</c> disables.
    /// </param>
    public static PerplexityOptions LlamaCppDefault(int contextLength, int maxTokens = 0, int bosTokenId = -1) =>
        new(PerplexityMode.SlidingWindow, contextLength, Stride: contextLength, maxTokens,
            UnscoredPrefix: contextLength / 2, BosTokenId: bosTokenId);
}

/// <summary>Outcome of a perplexity run.</summary>
/// <param name="Perplexity">
/// <c>exp(MeanNegativeLogLikelihood)</c> — the headline figure.
/// </param>
/// <param name="MeanNegativeLogLikelihood">
/// Mean NLL in nats over all scored tokens. Reported alongside perplexity because differences
/// between near-identical runs are easier to read here than through the exponential.
/// </param>
/// <param name="ScoredTokens">
/// Number of tokens that contributed. Comparisons across runs are meaningful only when this
/// matches: a perplexity computed over a different token count is a different measurement.
/// </param>
/// <param name="WindowCount">Number of forward windows evaluated.</param>
public readonly record struct PerplexityResult(
    double Perplexity,
    double MeanNegativeLogLikelihood,
    int ScoredTokens,
    int WindowCount);
