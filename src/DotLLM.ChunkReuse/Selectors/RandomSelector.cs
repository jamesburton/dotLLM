namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// Control: a seeded uniform permutation. By default it draws from the stale tokens (chunks after the first) and puts chunk 0
/// last, like every other selector, so that it does not waste budget on tokens that are already exact; with
/// <c>includeFirstChunk</c> it draws from the whole context. Nested across budgets for a fixed seed.
/// </summary>
public sealed class RandomSelector : RankedSelector
{
    private readonly bool _includeFirstChunk;

    /// <summary>Creates the control.</summary>
    /// <param name="includeFirstChunk">Draw from all tokens instead of the stale ones first.</param>
    public RandomSelector(bool includeFirstChunk = false) => _includeFirstChunk = includeFirstChunk;

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        var rng = new Random(seed);
        if (_includeFirstChunk)
        {
            int[] all = Enumerable.Range(0, context.Length).ToArray();
            rng.Shuffle(all);
            return all;
        }
        int[] stale = StaleTokens(context).ToArray();
        rng.Shuffle(stale);
        return WithFirstChunkLast(context, stale);
    }
}
