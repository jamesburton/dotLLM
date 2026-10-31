namespace DotLLM.ChunkReuse.Selectors;

/// <summary>Chooses which context tokens of a naive-concatenation chunk cache to recompute (<see cref="SelectiveRecomputer"/>).</summary>
public interface IRecomputeSelector
{
    /// <summary>
    /// Picks <paramref name="budgetTokens"/> token indices of <paramref name="context"/> to recompute.
    /// </summary>
    /// <param name="context">The context.</param>
    /// <param name="budgetTokens">Number of tokens to pick; at least 0.</param>
    /// <param name="seed">Seed for randomised selectors; deterministic selectors ignore it.</param>
    /// <returns>Strictly ascending indices in <c>[0, context.Length)</c>, exactly <c>min(budgetTokens, context.Length)</c> of them.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="budgetTokens"/> is negative.</exception>
    int[] Select(ChunkedContext context, int budgetTokens, int seed);
}

/// <summary>
/// A selector defined by a priority order over all context tokens: the selection at budget <c>b</c> is the first <c>b</c>
/// tokens of <see cref="Rank"/>, sorted. Selections are therefore nested across budgets (a larger budget only adds tokens),
/// which keeps quality-versus-budget curves free of re-selection noise, and the order can be re-shaped by decorators
/// (<see cref="ContiguousRuns"/>).
/// </summary>
public interface IRankedSelector : IRecomputeSelector
{
    /// <summary>All context token indices, each once, most important first.</summary>
    /// <param name="context">The context.</param>
    /// <param name="seed">Seed for randomised selectors; deterministic selectors ignore it.</param>
    int[] Rank(ChunkedContext context, int seed);
}

/// <summary>Base for <see cref="IRankedSelector"/>s: <see cref="Select"/> = sorted prefix of <see cref="Rank"/>.</summary>
public abstract class RankedSelector : IRankedSelector
{
    /// <inheritdoc/>
    public abstract int[] Rank(ChunkedContext context, int seed);

    /// <inheritdoc/>
    public int[] Select(ChunkedContext context, int budgetTokens, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentOutOfRangeException.ThrowIfNegative(budgetTokens);
        int[] order = Rank(context, seed);
        if (order.Length != context.Length)
            throw new InvalidOperationException($"{GetType().Name}.Rank returned {order.Length} indices for a {context.Length}-token context.");
        int[] picked = order.AsSpan(0, Math.Min(budgetTokens, order.Length)).ToArray();
        Array.Sort(picked);
        return picked;
    }

    /// <summary>Appends the chunk-0 tokens (exact under naive concatenation) after <paramref name="staleOrder"/>, in index order.</summary>
    protected static int[] WithFirstChunkLast(ChunkedContext context, IEnumerable<int> staleOrder)
    {
        var order = new List<int>(context.Length);
        order.AddRange(staleOrder);
        for (int i = 0; i < context.ChunkLength(0); i++) order.Add(i);
        return order.ToArray();
    }

    /// <summary>Indices of every token after chunk 0 (the stale tokens), ascending.</summary>
    protected static IEnumerable<int> StaleTokens(ChunkedContext context)
        => Enumerable.Range(context.ChunkLength(0), context.Length - context.ChunkLength(0));

    /// <summary>
    /// Round-robin over the chunks after the first by within-chunk offset: offset 0 of every chunk, then offset 1, … up to
    /// <paramref name="maxOffset"/> (exclusive), skipping chunks that are shorter.
    /// </summary>
    protected static IEnumerable<int> ChunkPrefixRoundRobin(ChunkedContext context, int maxOffset)
    {
        int[] starts = context.ChunkStarts;
        for (int o = 0; o < maxOffset; o++)
            for (int c = 1; c < context.ChunkCount; c++)
                if (o < context.ChunkLength(c))
                    yield return starts[c] + o;
    }

    /// <summary>Indices ordered by score, highest first, ties to the lower index; chunk-0 tokens last.</summary>
    protected static int[] ByScoreStaleFirst(ChunkedContext context, float[] scores, ISet<int>? exclude = null)
    {
        if (scores.Length != context.Length)
            throw new InvalidOperationException($"Score source returned {scores.Length} scores for a {context.Length}-token context.");
        for (int i = 0; i < scores.Length; i++)
            if (!float.IsFinite(scores[i]))
                throw new InvalidOperationException($"Score source returned a non-finite score {scores[i]} for token {i}.");
        IEnumerable<int> stale = StaleTokens(context);
        if (exclude is not null) stale = stale.Where(i => !exclude.Contains(i));
        return stale.OrderByDescending(i => scores[i]).ThenBy(i => i).ToArray();
    }
}
