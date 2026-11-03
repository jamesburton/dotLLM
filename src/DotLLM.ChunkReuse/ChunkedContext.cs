namespace DotLLM.ChunkReuse;

/// <summary>
/// One request's context as a sequence of reused chunks, token-level only (no K/V, no model): the chunks in context order,
/// their concatenation at positions <c>[0, Length)</c>, each chunk's start offset, and the query tokens that follow the
/// context (<c>[Length, Length + Query.Length)</c>). Recompute selectors pick token indices in <c>[0, Length)</c> from it;
/// graph builders map chunk-local token offsets to context indices through <see cref="ChunkStarts"/>.
/// </summary>
/// <remarks>
/// Chunk 0 sits at offset 0, so its isolated encoding equals the full prefill there (its K/V are exact under naive
/// concatenation); every later chunk is stale. Instances are immutable; arrays are copied on construction.
/// </remarks>
public sealed class ChunkedContext
{
    private readonly int[][] _chunks;
    private readonly int[] _tokenIds;
    private readonly int[] _chunkStarts;
    private readonly int[] _query;

    /// <summary>Builds the context from chunk token ids in context order and the query that follows them.</summary>
    /// <param name="chunkTokenIds">Chunks in context order; none empty.</param>
    /// <param name="queryTokenIds">Tokens after the context known at request time (may be empty).</param>
    /// <exception cref="ArgumentException">There are no chunks, or a chunk is null or empty.</exception>
    public ChunkedContext(IReadOnlyList<int[]> chunkTokenIds, ReadOnlySpan<int> queryTokenIds)
    {
        ArgumentNullException.ThrowIfNull(chunkTokenIds);
        if (chunkTokenIds.Count == 0)
            throw new ArgumentException("A context needs at least one chunk.", nameof(chunkTokenIds));
        _chunks = new int[chunkTokenIds.Count][];
        _chunkStarts = new int[chunkTokenIds.Count];
        int total = 0;
        for (int c = 0; c < chunkTokenIds.Count; c++)
        {
            int[]? ids = chunkTokenIds[c];
            if (ids is null || ids.Length == 0)
                throw new ArgumentException($"Chunk {c} is null or empty.", nameof(chunkTokenIds));
            _chunks[c] = (int[])ids.Clone();
            _chunkStarts[c] = total;
            total += ids.Length;
        }
        _tokenIds = new int[total];
        for (int c = 0; c < _chunks.Length; c++) _chunks[c].CopyTo(_tokenIds, _chunkStarts[c]);
        _query = queryTokenIds.ToArray();
    }

    /// <summary>Number of context tokens (the sum of chunk lengths).</summary>
    public int Length => _tokenIds.Length;

    /// <summary>Number of chunks.</summary>
    public int ChunkCount => _chunks.Length;

    /// <summary>Token ids of chunk <paramref name="chunk"/>.</summary>
    public ReadOnlySpan<int> Chunk(int chunk) => _chunks[chunk];

    /// <summary>Length of chunk <paramref name="chunk"/>.</summary>
    public int ChunkLength(int chunk) => _chunks[chunk].Length;

    /// <summary>All context token ids, chunks concatenated in order (index = position).</summary>
    public int[] TokenIds => (int[])_tokenIds.Clone();

    /// <summary>Start offset of each chunk in <see cref="TokenIds"/>.</summary>
    public int[] ChunkStarts => (int[])_chunkStarts.Clone();

    /// <summary>The query tokens that follow the context.</summary>
    public int[] QueryTokenIds => (int[])_query.Clone();

    /// <summary>The chunk containing context token <paramref name="tokenIndex"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside <c>[0, Length)</c>.</exception>
    public int ChunkOf(int tokenIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokenIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(tokenIndex, Length);
        int c = Array.BinarySearch(_chunkStarts, tokenIndex);
        return c >= 0 ? c : ~c - 1;
    }
}
