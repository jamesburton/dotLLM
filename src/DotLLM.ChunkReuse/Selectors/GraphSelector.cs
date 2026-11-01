using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;

namespace DotLLM.ChunkReuse.Selectors;

/// <summary>Which graph signal put a token into <see cref="GraphSignals.Picks"/> (highest priority first).</summary>
public enum GraphPickTier
{
    /// <summary>The definition window of a context symbol the query prefix references (Use/Call/Definition edge).</summary>
    QueryTarget,

    /// <summary>The first tokens of a chunk holding a query target (a cheap fallback: chunk starts carry most staleness, E2/E3).</summary>
    TargetChunkStart,

    /// <summary>An end of a Use/Call/Definition edge between two chunks of the context (definition windows, then use tokens).</summary>
    InContextEdge,

    /// <summary>An end of an Import edge (query prefix → context, or between context chunks).</summary>
    Import,
}

/// <summary>Knobs of <see cref="GraphSignals"/>.</summary>
public sealed record GraphSignalOptions
{
    /// <summary>Tokens taken from a definition token onwards (name, parameter list, …), clipped to its chunk. Default 16.</summary>
    public int DefinitionWindow { get; init; } = 16;

    /// <summary>Leading tokens of each query-target chunk (<see cref="GraphPickTier.TargetChunkStart"/>). Default 16.</summary>
    public int ChunkStartTokens { get; init; } = 16;
}

/// <summary>
/// The graph picks of one request: context tokens ranked by what the reference graph says the query needs, without any budget
/// fill. Built per request from a prebuilt <see cref="ReferenceGraph"/>, the context's chunks and the query's character range
/// in its file; nothing else about the request is read. In priority order:
/// <list type="number">
/// <item><see cref="GraphPickTier.QueryTarget"/>: for each context token that is the definition end of a query edge
/// (<see cref="ReferenceGraph.QueryEdges"/> over the query range, Use/Call/Definition kinds), its definition window, targets
/// ranked by the number of query edges into them (ties: context order).</item>
/// <item><see cref="GraphPickTier.TargetChunkStart"/>: the first tokens of each chunk holding a target (chunks by their total
/// query edges).</item>
/// <item><see cref="GraphPickTier.InContextEdge"/>: edges between context chunks (<see cref="ReferenceGraph.EdgesForContext(ChunkedContext, IReadOnlyList{Chunk})"/>):
/// definition windows ranked by in-context in-degree, then the use tokens in context order.</item>
/// <item><see cref="GraphPickTier.Import"/>: ends of Import edges (from the query, then between context chunks), context order.</item>
/// </list>
/// Chunk 0 is exact under naive concatenation, so its tokens are never picked; every token appears at most once.
/// </summary>
/// <remarks>
/// Leakage: the selector may read only what the model reads before generating. For a <see cref="CodeTask"/> use
/// <see cref="ForTask"/>, whose range ends at <see cref="CodeTask.MaskStartOffset"/>, so the masked call (the answer) never
/// contributes an edge. Query uses are not mapped to query tokens: only the context side of an edge is selected.
/// </remarks>
public sealed class GraphSignals
{
    private readonly int[] _picks;
    private readonly GraphPickTier[] _tiers;
    private readonly int[] _contextIds;

    private GraphSignals(int[] contextIds, int[] picks, GraphPickTier[] tiers, int queryEdges, int queryTargets, int queryTargetsInFirstChunk,
        int inContextEdges)
    {
        _contextIds = contextIds;
        _picks = picks;
        _tiers = tiers;
        QueryEdges = queryEdges;
        QueryTargets = queryTargets;
        QueryTargetsInFirstChunk = queryTargetsInFirstChunk;
        InContextEdges = inContextEdges;
    }

    /// <summary>Picked context token indices, most important first, distinct, none in chunk 0.</summary>
    public ReadOnlySpan<int> Picks => _picks;

    /// <summary>Number of picks.</summary>
    public int Count => _picks.Length;

    /// <summary>Length of the context the picks were built for.</summary>
    public int ContextLength => _contextIds.Length;

    /// <summary>Query edges (any kind) whose use lies in the query range, wherever their definition is.</summary>
    public int QueryEdges { get; }

    /// <summary>Distinct context tokens targeted by Use/Call/Definition query edges in stale chunks (the targets ranked).</summary>
    public int QueryTargets { get; }

    /// <summary>Distinct targeted tokens in chunk 0 (exact already, so skipped).</summary>
    public int QueryTargetsInFirstChunk { get; }

    /// <summary>Edges between different chunks of the context (any kind).</summary>
    public int InContextEdges { get; }

    /// <summary>Tier of the pick at rank <paramref name="rank"/>.</summary>
    public GraphPickTier TierOf(int rank) => _tiers[rank];

    /// <summary>Number of picks from <paramref name="tier"/>.</summary>
    public int CountIn(GraphPickTier tier) => _tiers.Count(t => t == tier);

    /// <summary>
    /// Signals for a cross-file task: the query is <see cref="CodeTask.Prefix"/>, i.e. the calling file's characters
    /// [<see cref="CodeTask.PrefixStartOffset"/>, <see cref="CodeTask.MaskStartOffset"/>), which excludes the masked call.
    /// </summary>
    public static GraphSignals ForTask(ReferenceGraph graph, ChunkedContext context, IReadOnlyList<Chunk> chunks, CodeTask task,
        GraphSignalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(task);
        return ForQuery(graph, context, chunks, task.CallSitePath, task.PrefixStartOffset, task.MaskStartOffset, options);
    }

    /// <summary>
    /// Signals for a query that is the character range [<paramref name="queryStart"/>, <paramref name="queryEnd"/>) of corpus
    /// file <paramref name="queryFile"/> (no query edges when it is null or the range is empty).
    /// </summary>
    /// <param name="graph">The prebuilt reference graph of the corpus the chunks come from.</param>
    /// <param name="context">The context.</param>
    /// <param name="chunks">The context's chunks, in context order (identified by <see cref="Chunk.Id"/>).</param>
    /// <param name="queryFile">Corpus path of the file the query text comes from, or null.</param>
    /// <param name="queryStart">First query character (file offset).</param>
    /// <param name="queryEnd">End of the query (exclusive). Must not reach into text the model has not seen.</param>
    /// <param name="options">Window sizes.</param>
    /// <exception cref="ArgumentException"><paramref name="chunks"/> does not describe the context, or a chunk is not in the graph.</exception>
    public static GraphSignals ForQuery(ReferenceGraph graph, ChunkedContext context, IReadOnlyList<Chunk> chunks, string? queryFile,
        int queryStart, int queryEnd, GraphSignalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(chunks);
        options ??= new GraphSignalOptions();
        ArgumentOutOfRangeException.ThrowIfNegative(options.DefinitionWindow);
        ArgumentOutOfRangeException.ThrowIfNegative(options.ChunkStartTokens);

        IReadOnlyList<SymbolEdge> inContext = graph.EdgesForContext(context, chunks); // validates chunks against the context
        int[] starts = context.ChunkStarts;
        int L = context.Length, first = context.ChunkLength(0);
        var placed = new Dictionary<int, int>(chunks.Count); // graph chunk -> first context chunk holding it
        for (int k = 0; k < chunks.Count; k++) placed.TryAdd(graph.IndexOf(chunks[k].Id), k);
        int ChunkEnd(int k) => k + 1 < starts.Length ? starts[k + 1] : L;

        var picks = new List<int>();
        var tiers = new List<GraphPickTier>();
        var taken = new HashSet<int>();
        void Add(int token, GraphPickTier tier)
        {
            if (token < first || !taken.Add(token)) return;
            picks.Add(token);
            tiers.Add(tier);
        }
        void AddWindow(int token, GraphPickTier tier)
        {
            int end = Math.Min(ChunkEnd(context.ChunkOf(token)), token + options.DefinitionWindow);
            for (int t = token; t < end; t++) Add(t, tier);
        }

        // 1. Query targets (Use/Call/Definition); imports are kept for tier 4.
        IReadOnlyList<QueryEdge> queryEdges = queryFile is null || queryEnd <= queryStart ? [] : graph.QueryEdges(queryFile, queryStart, queryEnd);
        var targetCount = new Dictionary<int, int>();
        var chunkCount = new Dictionary<int, int>();
        var queryImports = new List<int>();
        int inFirst = 0;
        var firstTargets = new HashSet<int>();
        foreach (QueryEdge q in queryEdges)
        {
            if (!placed.TryGetValue(q.DefinitionChunk, out int k)) continue; // definition not in this context
            int token = starts[k] + q.DefinitionToken;
            if (q.Kind == SymbolEdgeKind.Import)
            {
                queryImports.Add(token);
                continue;
            }
            if (k == 0 || token < first)
            {
                firstTargets.Add(token);
                continue;
            }
            targetCount[token] = targetCount.GetValueOrDefault(token) + 1;
            chunkCount[k] = chunkCount.GetValueOrDefault(k) + 1;
        }
        inFirst = firstTargets.Count;
        foreach (int token in targetCount.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key))
            AddWindow(token, GraphPickTier.QueryTarget);

        // 2. Starts of the chunks holding targets.
        foreach (int k in chunkCount.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key))
            for (int t = starts[k]; t < Math.Min(ChunkEnd(k), starts[k] + options.ChunkStartTokens); t++) Add(t, GraphPickTier.TargetChunkStart);

        // 3. Edges between context chunks (not imports): definition windows by in-degree, then use tokens.
        var inDegree = new Dictionary<int, int>();
        var uses = new SortedSet<int>();
        foreach (SymbolEdge e in inContext)
        {
            if (e.Kind == SymbolEdgeKind.Import) continue;
            inDegree[e.DefinitionTokenIndex] = inDegree.GetValueOrDefault(e.DefinitionTokenIndex) + 1;
            uses.Add(e.UseTokenIndex);
        }
        foreach (int token in inDegree.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key))
            AddWindow(token, GraphPickTier.InContextEdge);
        foreach (int token in uses) Add(token, GraphPickTier.InContextEdge);

        // 4. Imports: the query's, then those between context chunks (both ends).
        foreach (int token in queryImports.Order()) Add(token, GraphPickTier.Import);
        var importEnds = new SortedSet<int>();
        foreach (SymbolEdge e in inContext)
        {
            if (e.Kind != SymbolEdgeKind.Import) continue;
            importEnds.Add(e.UseTokenIndex);
            importEnds.Add(e.DefinitionTokenIndex);
        }
        foreach (int token in importEnds) Add(token, GraphPickTier.Import);

        return new GraphSignals(context.TokenIds, picks.ToArray(), tiers.ToArray(), queryEdges.Count, targetCount.Count, inFirst, inContext.Count);
    }

    /// <summary>Throws unless <paramref name="context"/> has the token ids these signals were built for.</summary>
    internal void CheckContext(ChunkedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.Length != _contextIds.Length || !context.TokenIds.AsSpan().SequenceEqual(_contextIds))
            throw new ArgumentException("These graph signals were built for another context.", nameof(context));
    }
}

/// <summary>
/// Structure-aware recompute selection: the graph picks of one request (<see cref="GraphSignals"/>) first, then, for the rest
/// of the budget, a deterministic fill in <see cref="ChunkStartsSelector"/> order (offset 0 of every stale chunk, then offset
/// 1, …), skipping tokens already picked. With no graph signal it therefore selects exactly what <see cref="ChunkStartsSelector"/>
/// does. <see cref="GraphPicks"/> reports how much of a budget the graph itself chose, because a large fill can hide a null
/// result. Static given the prebuilt graph (no model, no K/V); seed ignored.
/// </summary>
public sealed class GraphSelector : RankedSelector
{
    private static readonly ChunkStartsSelector Fill = new();

    /// <summary>Creates the selector for the request the signals were built for.</summary>
    public GraphSelector(GraphSignals signals)
    {
        ArgumentNullException.ThrowIfNull(signals);
        Signals = signals;
    }

    /// <summary>The request's graph picks.</summary>
    public GraphSignals Signals { get; }

    /// <summary>How many of the first <paramref name="budgetTokens"/> ranked tokens are graph picks (the rest is fill).</summary>
    public int GraphPicks(int budgetTokens) => Math.Min(Math.Max(budgetTokens, 0), Signals.Count);

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        Signals.CheckContext(context);
        return PicksThen(Signals, Fill.Rank(context, seed));
    }

    /// <summary>The picks, then <paramref name="rest"/> without them.</summary>
    internal static int[] PicksThen(GraphSignals signals, int[] rest)
    {
        var order = new int[rest.Length];
        var taken = new bool[rest.Length];
        int n = 0;
        foreach (int p in signals.Picks)
        {
            order[n++] = p;
            taken[p] = true;
        }
        foreach (int i in rest)
            if (!taken[i]) order[n++] = i;
        return order;
    }
}
