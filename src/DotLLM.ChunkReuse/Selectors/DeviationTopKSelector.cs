namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// CacheBlend-like (arXiv 2405.16444) top-k by K/V deviation between the reused (isolated-chunk) K/V and freshly computed
/// K/V. With <see cref="Layer1KvDeviationSource"/> this is CacheBlend's runnable proxy — layer-1 deviation, one fixed token
/// set for all layers (no gradual per-layer filtering). With <see cref="OracleKvDeviationSource"/> it is an upper-bound
/// variant that needs the full-prefill K/V and is therefore NOT runnable at serving time. Chunk-0 tokens (deviation 0) last.
/// </summary>
public sealed class DeviationTopKSelector : RankedSelector
{
    private readonly ITokenScoreSource _deviation;

    /// <summary>Creates the selector over a deviation score source.</summary>
    /// <param name="deviation">Per-token deviation scores.</param>
    public DeviationTopKSelector(ITokenScoreSource deviation)
    {
        ArgumentNullException.ThrowIfNull(deviation);
        _deviation = deviation;
    }

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        return WithFirstChunkLast(context, ByScoreStaleFirst(context, _deviation.Score(context)));
    }
}
