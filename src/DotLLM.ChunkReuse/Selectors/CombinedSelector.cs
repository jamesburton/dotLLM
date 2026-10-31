namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// The selection part of PSR ("prefix + scale + recompute", arXiv 2603.20218): chunk-start tokens first, then CacheBlend
/// deviation top-k for the rest of the budget. Only the token choice is implemented. PSR's other two components change the
/// encoding (Link0: chunks encoded behind a shared dummy prefix whose K/V is dropped) and the attention (APE scaling); neither
/// is a token selection, so neither is here. This is a selector-only approximation of PSR, not PSR: recomputing the first
/// <see cref="StartTokens"/> of each stale chunk stands in for the prefix's job of removing the per-chunk attention sinks.
/// </summary>
public sealed class CombinedSelector : RankedSelector
{
    private readonly ITokenScoreSource _deviation;

    /// <summary>Creates the selector.</summary>
    /// <param name="deviation">Deviation scores for the remainder (<see cref="Layer1KvDeviationSource"/>).</param>
    /// <param name="startTokens">Tokens taken at the start of each chunk after the first.</param>
    public CombinedSelector(ITokenScoreSource deviation, int startTokens = 16)
    {
        ArgumentNullException.ThrowIfNull(deviation);
        ArgumentOutOfRangeException.ThrowIfNegative(startTokens);
        _deviation = deviation;
        StartTokens = startTokens;
    }

    /// <summary>Tokens taken at the start of each chunk after the first.</summary>
    public int StartTokens { get; }

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        int[] head = ChunkPrefixRoundRobin(context, StartTokens).ToArray();
        int[] rest = ByScoreStaleFirst(context, _deviation.Score(context), head.ToHashSet());
        return WithFirstChunkLast(context, head.Concat(rest));
    }
}
