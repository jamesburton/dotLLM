namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// Decorator that turns any ranked selector's picks into contiguous runs within the same budget (CacheClip's window grouping,
/// arXiv 2510.10129, w = 8; "Contiguity, Not Importance", arXiv 2609.17983, shows scattered positions inherit nearby
/// staleness). Walking the inner ranking, each not-yet-covered anchor <c>a</c> contributes the run <c>a, a+1, …</c> of up to
/// <see cref="Window"/> tokens, stopping at the context end or at an already covered token. The selection at budget <c>b</c>
/// is the first <c>b</c> tokens of that order, so the last run is truncated to fit and selections stay nested.
/// </summary>
/// <remarks>
/// Runs extend forward from the anchor: the tokens right after a stale token attend to it most (local attention), and a
/// contiguous run is one <c>Forward</c> in <see cref="SelectiveRecomputer"/>. A selector that already picks contiguous chunk
/// prefixes (first-k, chunk starts) is returned unchanged while its prefixes are anchored at offset 0.
/// </remarks>
public sealed class ContiguousRuns : RankedSelector
{
    private readonly IRankedSelector _inner;

    /// <summary>Wraps <paramref name="inner"/>.</summary>
    /// <param name="inner">The selector whose ranking supplies the anchors.</param>
    /// <param name="window">Run length per anchor; at least 1.</param>
    public ContiguousRuns(IRankedSelector inner, int window = 8)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(window);
        _inner = inner;
        Window = window;
    }

    /// <summary>Run length per anchor.</summary>
    public int Window { get; }

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        int n = context.Length;
        var covered = new bool[n];
        var order = new List<int>(n);
        foreach (int a in _inner.Rank(context, seed))
        {
            for (int p = a; p < n && p < a + Window && !covered[p]; p++)
            {
                covered[p] = true;
                order.Add(p);
            }
        }
        return order.ToArray();
    }
}
