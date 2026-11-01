namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// Graph picks first (<see cref="GraphSignals"/>), then the rest of the budget in another ranked selector's order, skipping
/// tokens already picked. E4 runs it with <see cref="QueryRelevanceSelector"/> (the best E2 baseline) and with
/// <see cref="DeviationTopKSelector"/> (the plan's original pairing). Nested across budgets; deterministic for a seed when the
/// inner selector is.
/// </summary>
public sealed class HybridSelector : RankedSelector
{
    private readonly IRankedSelector _rest;

    /// <summary>Creates the selector.</summary>
    /// <param name="signals">The request's graph picks.</param>
    /// <param name="rest">Ranks the tokens after the graph picks.</param>
    public HybridSelector(GraphSignals signals, IRankedSelector rest)
    {
        ArgumentNullException.ThrowIfNull(signals);
        ArgumentNullException.ThrowIfNull(rest);
        Signals = signals;
        _rest = rest;
    }

    /// <summary>The request's graph picks.</summary>
    public GraphSignals Signals { get; }

    /// <summary>How many of the first <paramref name="budgetTokens"/> ranked tokens are graph picks.</summary>
    public int GraphPicks(int budgetTokens) => Math.Min(Math.Max(budgetTokens, 0), Signals.Count);

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        Signals.CheckContext(context);
        return GraphSelector.PicksThen(Signals, _rest.Rank(context, seed));
    }
}
