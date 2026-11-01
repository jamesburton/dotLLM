using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;

namespace ChunkKvLab;

/// <summary>Options of the <c>graph-info</c> command.</summary>
internal sealed record GraphOptions(string ModelPath, string CorpusDir, string Chunker, int ChunkTokens, int Repeat, int Plans, int Seed,
    string? ExpectHash);

/// <summary>
/// <c>graph-info</c> (Task 9, tokenizer only): builds the Roslyn reference graph of a code corpus <see cref="GraphOptions.Repeat"/>
/// times from scratch (fresh compilation each time) and prints build time per 100K tokens split into compile / bind / map,
/// then edge statistics: edges per 1K tokens, kind breakdown, share crossing chunks and files, and — over the Task 8 plans
/// (<see cref="CrossFileTaskGenerator.GeneratePlans"/>, default options) — cross-chunk edges per 1K context tokens,
/// unidentified context chunks and <see cref="ReferenceGraph.EdgesForContext(ChunkedContext)"/> time.
/// </summary>
internal static class GraphRunner
{
    public static int Run(GraphOptions o)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(o.ModelPath);
        var tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Corpus corpus = TasksRunner.LoadCorpus(o.CorpusDir, o.Chunker, o.ChunkTokens, tokenizer, o.ExpectHash);
        double per100K = 100_000.0 / corpus.TotalTokens;

        var builder = new RoslynGraphBuilder(tokenizer);
        ReferenceGraph graph = null!;
        var totals = new List<double>();
        for (int r = 0; r < o.Repeat; r++)
        {
            var sw = Stopwatch.StartNew();
            graph = builder.Build(corpus.Files, corpus.Chunks);
            double total = sw.Elapsed.TotalSeconds;
            totals.Add(total);
            var (compile, bind, map) = builder.LastTimings;
            Console.WriteLine($"build {r + 1}/{o.Repeat}: {total:F2} s (compile {compile.TotalSeconds:F2}, bind {bind.TotalSeconds:F2}, map {map.TotalSeconds:F2}); "
                + $"per 100K tokens {total * per100K:F3} s (compile {compile.TotalSeconds * per100K:F3}, bind {bind.TotalSeconds * per100K:F3}, map {map.TotalSeconds * per100K:F3})");
        }
        double[] sorted = totals.Order().ToArray();
        Console.WriteLine($"build time per 100K tokens: min {sorted[0] * per100K:F3} s, median {sorted[sorted.Length / 2] * per100K:F3} s over {o.Repeat} runs");

        IReadOnlyList<ChunkEdge> edges = graph.Edges;
        IReadOnlyList<Chunk> chunks = graph.Chunks;
        double kTokens = corpus.TotalTokens / 1000.0;
        int cross = edges.Count(e => e.UseChunk != e.DefinitionChunk);
        int crossFile = edges.Count(e => chunks[e.UseChunk].FilePath != chunks[e.DefinitionChunk].FilePath);
        Console.WriteLine($"edges {edges.Count} ({edges.Count / kTokens:F2} per 1K tokens); cross-chunk {cross} ({Pct(cross, edges.Count)}), "
            + $"cross-file {crossFile} ({Pct(crossFile, edges.Count)}); mapping-mismatch chunks {graph.MappingMismatchChunks}");
        foreach (SymbolEdgeKind kind in Enum.GetValues<SymbolEdgeKind>())
        {
            ChunkEdge[] k = edges.Where(e => e.Kind == kind).ToArray();
            int kc = k.Count(e => e.UseChunk != e.DefinitionChunk);
            Console.WriteLine($"  {kind,-10} {k.Length,9} ({Pct(k.Length, edges.Count)} of edges, {k.Length / kTokens:F2} per 1K tokens), cross-chunk {Pct(kc, k.Length)}");
        }
        int useTokens = edges.Select(e => (e.UseChunk, e.UseToken)).Distinct().Count();
        int defTokens = edges.Select(e => (e.DefinitionChunk, e.DefinitionToken)).Distinct().Count();
        Console.WriteLine($"distinct use-end tokens {useTokens} ({Pct(useTokens, corpus.TotalTokens)} of tokens), definition-end tokens {defTokens} ({Pct(defTokens, corpus.TotalTokens)})");
        // Chunks whose ids another chunk shares: EdgesForContext(ChunkedContext) cannot identify them.
        var dup = chunks.GroupBy(c => string.Join(',', c.TokenIds), StringComparer.Ordinal).Where(g => g.Count() > 1)
            .OrderByDescending(g => g.Count()).ToArray();
        Console.WriteLine($"chunks sharing their token content: {dup.Sum(g => g.Count())} in {dup.Length} groups; largest: "
            + string.Join("; ", dup.Take(3).Select(g => $"{g.Count()}x \"{Snippet(corpus.TextOf(g.First()))}\"")));

        if (o.Plans <= 0) return 0;
        var swPlans = Stopwatch.StartNew();
        IReadOnlyList<CodeContextPlan> plans = new CrossFileTaskGenerator().GeneratePlans(corpus, o.Plans, o.Seed);
        Console.WriteLine($"plans: {plans.Count} (task budget {o.Plans}, seed {o.Seed}) in {swPlans.Elapsed.TotalSeconds:F1} s");
        long ctxTokens = 0, ctxEdges = 0, ctxChunks = 0, unmatched = 0, exactEdges = 0, endTokens = 0;
        var byKind = new long[Enum.GetValues<SymbolEdgeKind>().Length];
        double queryMs = 0;
        var indexOf = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < chunks.Count; i++) indexOf.TryAdd(chunks[i].Id, i);
        long callFileEdges = 0, callFileEdgesToRequired = 0;
        foreach (CodeContextPlan plan in plans)
        {
            // Edges from the calling files (whole files: an upper bound for the query prefixes) into the context.
            var inContext = plan.Chunks.Select(c => indexOf[c.Id]).ToHashSet();
            var required = plan.RequiredIds.Select(id => indexOf[id]).ToHashSet();
            var callFiles = plan.Tasks.Select(t => t.CallSitePath).ToHashSet(StringComparer.Ordinal);
            foreach (ChunkEdge x in edges)
            {
                if (!callFiles.Contains(chunks[x.UseChunk].FilePath) || !inContext.Contains(x.DefinitionChunk)) continue;
                callFileEdges++;
                if (required.Contains(x.DefinitionChunk)) callFileEdgesToRequired++;
            }

            var ctx = new ChunkedContext(plan.Chunks.Select(c => c.TokenIds).ToArray(), []);
            var sw = Stopwatch.StartNew();
            IReadOnlyList<SymbolEdge> e = graph.EdgesForContext(ctx);
            queryMs += sw.Elapsed.TotalMilliseconds;
            exactEdges += graph.EdgesForContext(ctx, plan.Chunks).Count;
            unmatched += graph.MatchChunks(ctx).Count(m => m < 0);
            ctxTokens += ctx.Length;
            ctxChunks += ctx.ChunkCount;
            ctxEdges += e.Count;
            foreach (SymbolEdge x in e) byKind[(int)x.Kind]++;
            endTokens += e.SelectMany(x => new[] { x.UseTokenIndex, x.DefinitionTokenIndex }).Distinct().Count();
        }
        double ck = ctxTokens / 1000.0;
        Console.WriteLine($"contexts: {plans.Count}, mean {ctxTokens / (double)plans.Count:F0} tokens / {ctxChunks / (double)plans.Count:F1} chunks; "
            + $"unidentified chunks {unmatched}/{ctxChunks}; cross-chunk edges {ctxEdges} ({ctxEdges / ck:F2} per 1K context tokens, "
            + $"explicit-chunk overload {exactEdges}); tokens on an edge end {Pct(endTokens, ctxTokens)}");
        Console.WriteLine("  by kind: " + string.Join(", ", Enum.GetValues<SymbolEdgeKind>().Select(k => $"{k} {byKind[(int)k]} ({byKind[(int)k] / ck:F2}/1K)")));
        Console.WriteLine($"calling-file -> context edges (not in EdgesForContext: the query is not a chunk): {callFileEdges / (double)plans.Count:F1} per context, "
            + $"{Pct(callFileEdgesToRequired, callFileEdges)} into the required chunk(s)");
        Console.WriteLine($"EdgesForContext: {queryMs / plans.Count:F3} ms per context, {queryMs / (ctxTokens / 10_000.0):F3} ms per 10K context tokens");
        return 0;
    }

    private static string Snippet(string text) =>
        (text.Length > 60 ? text[..60] + "..." : text).Replace("\n", "\\n", StringComparison.Ordinal);

    private static string Pct(long part, long whole) => whole == 0 ? "n/a" : (100.0 * part / whole).ToString("F1", CultureInfo.InvariantCulture) + " %";
}
