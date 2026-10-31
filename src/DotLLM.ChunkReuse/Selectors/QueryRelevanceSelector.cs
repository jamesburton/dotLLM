namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// ProphetKV-style (arXiv 2602.02579) query relevance: rank context tokens by the attention the query tokens pay them on the
/// reused K/V, averaged over heads, query tokens and layers (<see cref="QueryAttentionSource"/>), and recompute one fixed set in
/// every layer. Query-dependent: it cannot be precomputed per chunk. Chunk-0 tokens (exact; they also hold the position-0
/// attention sink, which would otherwise dominate) last.
/// </summary>
public sealed class QueryRelevanceSelector : RankedSelector
{
    private readonly ITokenScoreSource _relevance;

    /// <summary>Creates the selector over a query-attention score source.</summary>
    /// <param name="relevance">Per-token query-attention scores.</param>
    public QueryRelevanceSelector(ITokenScoreSource relevance)
    {
        ArgumentNullException.ThrowIfNull(relevance);
        _relevance = relevance;
    }

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        return WithFirstChunkLast(context, ByScoreStaleFirst(context, _relevance.Score(context)));
    }
}
