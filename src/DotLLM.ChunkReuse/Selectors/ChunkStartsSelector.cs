namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// Chunk starts: spend the whole budget on equal-length prefixes of the chunks after the first (offset 0 of every stale chunk,
/// then offset 1, …). Motivated by the observation that CacheBlend-style early-layer picks cluster at the beginning of chunks
/// (arXiv 2603.20218; the plan's attribution to CacheClip was corrected in research/landscape.md). Equivalent to EPIC
/// first-k with <c>k</c> grown to fill the budget. Static; seed ignored.
/// </summary>
public sealed class ChunkStartsSelector : RankedSelector
{
    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        int longest = 0;
        for (int c = 1; c < context.ChunkCount; c++) longest = Math.Max(longest, context.ChunkLength(c));
        return WithFirstChunkLast(context, ChunkPrefixRoundRobin(context, longest));
    }
}
