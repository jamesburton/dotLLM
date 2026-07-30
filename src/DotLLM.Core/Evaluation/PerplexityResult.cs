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
    /// Sliding-window scoring matching llama.cpp's <c>--perplexity</c> methodology: the corpus is
    /// walked in windows of <see cref="PerplexityOptions.ContextLength"/> advanced by
    /// <see cref="PerplexityOptions.Stride"/>, and only tokens beyond the carried-over prefix are
    /// scored, so every scored token has a full-length context.
    /// </summary>
    /// <remarks>
    /// Absolute-value oriented: comparable to published llama.cpp figures when context length,
    /// stride, corpus and model match.
    /// </remarks>
    SlidingWindow,
}

/// <summary>Configuration for a perplexity run.</summary>
/// <param name="Mode">Scoring strategy.</param>
/// <param name="ContextLength">
/// Window size in tokens. Clamped to <see cref="IPerplexityModel.MaxContextLength"/>.
/// </param>
/// <param name="Stride">
/// Tokens advanced between windows. Equal to <paramref name="ContextLength"/> gives
/// non-overlapping windows (llama.cpp's default); a smaller value overlaps, scoring each token
/// with more preceding context at proportionally higher cost.
/// </param>
/// <param name="MaxTokens">
/// Upper bound on corpus tokens consumed; <c>0</c> means unbounded. Bounds runtime on large
/// corpora without truncating the corpus file itself.
/// </param>
public readonly record struct PerplexityOptions(
    PerplexityMode Mode,
    int ContextLength,
    int Stride,
    int MaxTokens = 0);

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
