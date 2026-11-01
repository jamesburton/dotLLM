using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Structure-aware recompute selection (Task 11): <see cref="GraphSignals"/> (graph picks of one request, from the reference
/// graph and the query PREFIX only), <see cref="GraphSelector"/> (graph picks, then chunk-start fill) and
/// <see cref="HybridSelector"/> (graph picks, then another selector's ranking). Fixtures are tiny C# repos tokenised one char
/// per token, so a context index is a character offset inside its chunk.
/// </summary>
public sealed class GraphSelectorTests(ITestOutputHelper output)
{
    private const string AText =
        "namespace Lib;\n\npublic static class A\n{\n    public static int Foo(int x) => x + 1;\n\n" +
        "    public static int Bar(int x) => x * 2;\n\n    public static int Baz() => 7;\n}\n";
    private const string BText = "namespace Lib;\n\npublic static class B\n{\n    public static int Qux() => A.Foo(1);\n}\n";
    private const string CText = "namespace Lib;\n\npublic static class C\n{\n    public static int Pad() => 0;\n}\n";
    private const string UseHead =
        "using Lib;\n\nnamespace App;\n\npublic class Use\n{\n    public int Run()\n    {\n        var a = A.Foo(1) + A.Foo(2);\n" +
        "        var b = C.Pad();\n        return A";
    private const string UseBar = ".Bar(a);\n    }\n}\n";
    private const string UseBaz = ".Baz() + b;\n    }\n}\n";

    private static (string Path, string Text)[] Repo(string maskedTail) =>
    [
        ("Lib/A.cs", AText), ("Lib/B.cs", BText), ("Lib/C.cs", CText), ("App/Use.cs", UseHead + maskedTail),
    ];

    private static TaskGeneratorOptions Options => new()
    {
        ContextTokens = 100_000, ContinuationTokens = 24, PrefixTokens = 400, TasksPerContext = 1,
    };

    private sealed record Fixture(Corpus Corpus, ReferenceGraph Graph, CodeTask Task, Chunk[] Order, ChunkedContext Context);

    /// <summary>Corpus + graph of <see cref="Repo"/>; the task masking the call after "return A"; context C | A.Foo | B | A.Bar | A.Baz.</summary>
    private static Fixture Build(string maskedTail)
    {
        ITokenizer tok = new CharTokenizer();
        var corpus = new Corpus(Repo(maskedTail).Select(f => new SourceFile(f.Path, f.Text)), new SymbolChunker(tok, 4096), tok);
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(corpus);
        CodeTask task = new CrossFileTaskGenerator(Options).Generate(corpus, 100, seed: 0)
            .Single(t => t.CallSitePath == "App/Use.cs" && t.MaskedText.TrimStart('.').StartsWith(maskedTail[1..4], StringComparison.Ordinal));
        Chunk Holding(string path, string needle)
        {
            string text = corpus.Files.Single(f => f.Path == path).Text;
            int off = text.IndexOf(needle, StringComparison.Ordinal);
            Assert.True(off >= 0, needle);
            return corpus.Chunks.Single(c => c.FilePath == path && off >= c.StartOffset && off < c.EndOffset);
        }
        Chunk[] order = [Holding("Lib/C.cs", "Pad"), Holding("Lib/A.cs", "Foo"), Holding("Lib/B.cs", "Qux"), Holding("Lib/A.cs", "Bar"), Holding("Lib/A.cs", "Baz")];
        Assert.Equal(5, order.Select(c => c.Id).Distinct().Count()); // every member is its own chunk
        return new Fixture(corpus, graph, task, order, new ChunkedContext(order.Select(c => c.TokenIds).ToArray(), task.PrefixTokenIds));
    }

    /// <summary>Context index of character <paramref name="offset"/> of <paramref name="path"/> (one char per token).</summary>
    private static int At(Chunk[] order, string path, int offset)
    {
        int start = 0;
        foreach (Chunk c in order)
        {
            if (c.FilePath == path && offset >= c.StartOffset && offset < c.EndOffset) return start + offset - c.StartOffset;
            start += c.TokenIds.Length;
        }
        throw new InvalidOperationException($"{path}:{offset} is not in the context");
    }

    private static int[] Window(int first, int count) => Enumerable.Range(first, count).ToArray();

    [Fact]
    public void GraphSelector_PrefersTokensWithCrossChunkEdges()
    {
        Fixture f = Build(UseBar);
        GraphSignals s = GraphSignals.ForTask(f.Graph, f.Context, f.Order, f.Task);
        int[] starts = f.Context.ChunkStarts;

        int aDecl = At(f.Order, "Lib/A.cs", AText.IndexOf("A\n{", StringComparison.Ordinal));
        int fooDecl = At(f.Order, "Lib/A.cs", AText.IndexOf("Foo(", StringComparison.Ordinal));
        int bA = At(f.Order, "Lib/B.cs", BText.IndexOf("A.Foo", StringComparison.Ordinal));
        int bFoo = bA + 2;
        // Query targets by query-edge count (type A: 3 uses before the mask; Foo: 2 calls), each with its 16-token definition
        // window; then the target chunk's first 16 tokens; then in-context cross-chunk edge ends not yet picked (B -> A, B -> Foo).
        // C.Pad's targets sit in chunk 0 (exact under naive concatenation) and are skipped; the masked call's callee (Bar) never
        // appears; the query's import (Lib, declared inside A.Foo's chunk start) adds nothing new.
        int[] expected = [.. Window(aDecl, 16), .. Window(fooDecl, 16), .. Window(starts[1], 16), bA, bFoo];
        Assert.Equal(expected, s.Picks.ToArray());
        Assert.Equal(32, s.CountIn(GraphPickTier.QueryTarget));
        Assert.Equal(16, s.CountIn(GraphPickTier.TargetChunkStart));
        Assert.Equal(2, s.CountIn(GraphPickTier.InContextEdge));
        Assert.Equal(0, s.CountIn(GraphPickTier.Import));
        Assert.Equal(GraphPickTier.QueryTarget, s.TierOf(0));
        Assert.Equal(GraphPickTier.InContextEdge, s.TierOf(expected.Length - 1));
        Assert.Equal(2, s.QueryTargets);
        Assert.Equal(2, s.QueryTargetsInFirstChunk); // C (type) and Pad are declared in chunk 0: skipped
        Assert.DoesNotContain(At(f.Order, "Lib/A.cs", AText.IndexOf("Bar", StringComparison.Ordinal)), s.Picks.ToArray());

        // The selector spends the budget on the picks first (nested), then on chunk starts, never on chunk 0 while stale tokens remain.
        var sel = new GraphSelector(s);
        Assert.Equal(expected.Order(), sel.Select(f.Context, expected.Length, seed: 0));
        Assert.Equal(expected[..20].Order(), sel.Select(f.Context, 20, seed: 0));
        // Fill = ChunkStarts round robin over the stale chunks, skipping picked tokens: offset 0 of B, A.Bar, A.Baz, then offset 1 of B.
        int[] more = sel.Select(f.Context, expected.Length + 4, seed: 0);
        Assert.Equal(expected.Concat([starts[2], starts[3], starts[4], starts[2] + 1]).Order(), more);
        Assert.Equal(20, sel.GraphPicks(20));
        Assert.Equal(expected.Length, sel.GraphPicks(500));
    }

    [Fact]
    public void GraphSelector_NoEdges_FillsBudgetDeterministically()
    {
        Fixture f = Build(UseBar);
        // A query with no reference edges (its range is empty) over a context whose chunks do not reference each other.
        Chunk[] order = [f.Order[0], f.Order[3], f.Order[4]]; // C | A.Bar | A.Baz
        var ctx = new ChunkedContext(order.Select(c => c.TokenIds).ToArray(), [1, 2, 3]);
        GraphSignals s = GraphSignals.ForQuery(f.Graph, ctx, order, "App/Use.cs", 0, 0);
        Assert.Equal(0, s.Count);
        var graph = new GraphSelector(s);
        var starts = new ChunkStartsSelector();
        for (int b = 0; b <= ctx.Length + 2; b++)
        {
            Assert.Equal(starts.Select(ctx, b, 0), graph.Select(ctx, b, 0));
            Assert.Equal(graph.Select(ctx, b, 0), graph.Select(ctx, b, 99));
            Assert.Equal(0, graph.GraphPicks(b));
        }
    }

    [Fact]
    public void HybridSelector_ReturnsExactBudget_NoDuplicates()
    {
        Fixture f = Build(UseBar);
        GraphSignals s = GraphSignals.ForTask(f.Graph, f.Context, f.Order, f.Task);
        int L = f.Context.Length;
        int[] picks = s.Picks.ToArray();
        foreach (IRankedSelector inner in new IRankedSelector[] { new RandomSelector(), new FirstKSelector(k: 3), new ChunkStartsSelector() })
        {
            var hybrid = new HybridSelector(s, inner);
            foreach (int seed in new[] { 0, 5 })
            {
                int[] rank = hybrid.Rank(f.Context, seed);
                Assert.Equal(Enumerable.Range(0, L), rank.Order());                       // a permutation
                Assert.Equal(picks, rank[..picks.Length]);                                // graph picks first
                Assert.Equal(inner.Rank(f.Context, seed).Where(i => !picks.Contains(i)), rank[picks.Length..]); // then the inner order
                foreach (int b in new[] { 0, 1, 7, picks.Length, picks.Length + 1, L / 2, L, L + 5 })
                {
                    int[] sel = hybrid.Select(f.Context, b, seed);
                    Assert.Equal(Math.Min(b, L), sel.Length);
                    Assert.Equal(sel.Length, sel.Distinct().Count());
                    for (int i = 1; i < sel.Length; i++) Assert.True(sel[i] > sel[i - 1]);
                    Assert.Equal(Math.Min(b, picks.Length), hybrid.GraphPicks(b));
                }
            }
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => new HybridSelector(s, new RandomSelector()).Select(f.Context, -1, 0));
    }

    [Fact]
    public void GraphSignals_MaskedCallEdge_NeverReachesTheSelector()
    {
        // End to end through the task generator: the masked call (A.Bar) has a Call edge into the context (chunk A.Bar), but it
        // starts at or after MaskStartOffset, so ForTask must not see it. Its definition is targeted by nothing in the prefix.
        Fixture f = Build(UseBar);
        IReadOnlyList<QueryEdge> answer = f.Graph.QueryEdges(f.Task.CallSitePath, f.Task.MaskStartOffset, f.Task.MaskStartOffset + f.Task.MaskedText.Length);
        QueryEdge call = Assert.Single(answer, e => e.Kind == SymbolEdgeKind.Call);
        int barDecl = At(f.Order, "Lib/A.cs", AText.IndexOf("Bar", StringComparison.Ordinal));
        Assert.Equal(f.Graph.Chunks[call.DefinitionChunk].Id, f.Order[3].Id); // the answer edge does point into the context
        Assert.Equal(barDecl - f.Context.ChunkStarts[3], call.DefinitionToken);

        GraphSignals s = GraphSignals.ForTask(f.Graph, f.Context, f.Order, f.Task);
        Assert.DoesNotContain(barDecl, s.Picks.ToArray());
        Assert.DoesNotContain(s.Picks.ToArray(), p => p >= f.Context.ChunkStarts[3] && p < f.Context.ChunkStarts[4]); // nothing in A.Bar's chunk
        // The leaky range (prefix + mask) would have picked it: the test can tell the difference.
        GraphSignals leaky = GraphSignals.ForQuery(f.Graph, f.Context, f.Order, f.Task.CallSitePath, f.Task.PrefixStartOffset,
            f.Task.MaskStartOffset + f.Task.MaskedText.Length);
        Assert.Contains(barDecl, leaky.Picks.ToArray());
    }

    [Fact]
    public void GraphSignals_SelectionIdentical_WhenMaskedTextChanges()
    {
        // Same repo up to the mask; the masked call is A.Bar(a) in one and A.Baz() + b in the other. The graphs differ (the
        // answer edge points at Bar vs Baz), the query prefix and the context do not, so every selection must be identical.
        Fixture bar = Build(UseBar), baz = Build(UseBaz);
        Assert.Equal(bar.Task.Prefix, baz.Task.Prefix);
        Assert.Equal(bar.Task.PrefixTokenIds, baz.Task.PrefixTokenIds);
        Assert.NotEqual(bar.Task.MaskedText, baz.Task.MaskedText);
        Assert.Equal(bar.Order.Select(c => c.Id), baz.Order.Select(c => c.Id));
        Assert.NotEqual(bar.Graph.Edges.Count(e => e.Kind == SymbolEdgeKind.Call && bar.Graph.Chunks[e.DefinitionChunk].Id == bar.Order[3].Id),
            baz.Graph.Edges.Count(e => e.Kind == SymbolEdgeKind.Call && baz.Graph.Chunks[e.DefinitionChunk].Id == bar.Order[3].Id));

        GraphSignals sBar = GraphSignals.ForTask(bar.Graph, bar.Context, bar.Order, bar.Task);
        GraphSignals sBaz = GraphSignals.ForTask(baz.Graph, baz.Context, baz.Order, baz.Task);
        Assert.True(sBar.Count > 0);
        Assert.Equal(sBar.Picks.ToArray(), sBaz.Picks.ToArray());
        var inner = new FirstKSelector(k: 2);
        for (int b = 0; b <= bar.Context.Length; b += 7)
        {
            Assert.Equal(new GraphSelector(sBar).Select(bar.Context, b, 1), new GraphSelector(sBaz).Select(baz.Context, b, 1));
            Assert.Equal(new HybridSelector(sBar, inner).Select(bar.Context, b, 1), new HybridSelector(sBaz, inner).Select(baz.Context, b, 1));
        }
    }

    [Fact]
    public void GraphSelector_SignalsOfAnotherContext_Throw()
    {
        Fixture f = Build(UseBar);
        GraphSignals s = GraphSignals.ForTask(f.Graph, f.Context, f.Order, f.Task);
        Chunk[] other = [f.Order[1], f.Order[0], f.Order[2], f.Order[3], f.Order[4]]; // same chunks and length, other order
        var otherCtx = new ChunkedContext(other.Select(c => c.TokenIds).ToArray(), []);
        Assert.Equal(f.Context.Length, otherCtx.Length);
        Assert.Throws<ArgumentException>(() => new GraphSelector(s).Select(otherCtx, 5, 0));
        Assert.Throws<ArgumentException>(() => new HybridSelector(s, new RandomSelector()).Select(otherCtx, 5, 0));
        // Chunks that do not describe the context are rejected when the signals are built.
        Assert.Throws<ArgumentException>(() => GraphSignals.ForTask(f.Graph, f.Context, other, f.Task));
    }

    [Fact]
    public void GraphSelector_SelectionTime_IsUnder1MsPer10KTokens()
    {
        // ~60 small classes that call each other (≈ 12K tokens of context, one char per token) and a query that calls 30 of
        // them. The graph is built once (offline cost, not timed here); the request-time work is GraphSignals.ForTask + Select.
        const int n = 60;
        var files = new List<SourceFile>();
        for (int i = 0; i < n; i++)
            files.Add(new SourceFile($"Lib/K{i:D2}.cs",
                $"namespace Lib;\n\npublic static class K{i:D2}\n{{\n    public static int Ma{i:D2}(int x) => K{(i + 1) % n:D2}.Mb{(i + 1) % n:D2}(x);\n\n" +
                $"    public static int Mb{i:D2}(int x) => x + {i};\n\n    public static string Name{i:D2} => \"K{i:D2}\";\n}}\n"));
        string query = "using Lib;\n\nnamespace App;\n\npublic class Q\n{\n    public int Run(int v)\n    {\n"
            + string.Concat(Enumerable.Range(0, 30).Select(i => $"        v += K{i * 2:D2}.Ma{i * 2:D2}(v);\n")) + "        return K00.Mb00(v);\n    }\n}\n";
        files.Add(new SourceFile("App/Q.cs", query));
        ITokenizer tok = new CharTokenizer();
        var corpus = new Corpus(files, new SymbolChunker(tok, 4096), tok);
        var sw = Stopwatch.StartNew();
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(corpus);
        output.WriteLine($"graph build (offline): {sw.Elapsed.TotalMilliseconds:F0} ms, {graph.Edges.Count} edges");
        CodeTask task = new CrossFileTaskGenerator(Options with { PrefixTokens = 4000 }).Generate(corpus, 1000, seed: 0)
            .Single(t => t.CallSitePath == "App/Q.cs" && t.MaskedText.TrimStart('.').StartsWith("Mb00", StringComparison.Ordinal));
        Chunk[] order = corpus.Chunks.Where(c => c.FilePath != "App/Q.cs").ToArray();
        var ctx = new ChunkedContext(order.Select(c => c.TokenIds).ToArray(), task.PrefixTokenIds);
        Assert.True(ctx.Length >= 10_000, $"context {ctx.Length}");
        int budget = ctx.Length / 10;

        double Once()
        {
            var t = Stopwatch.StartNew();
            GraphSignals s = GraphSignals.ForTask(graph, ctx, order, task);
            int[] picked = new GraphSelector(s).Select(ctx, budget, 0);
            t.Stop();
            Assert.Equal(budget, picked.Length);
            return t.Elapsed.TotalMilliseconds;
        }
        for (int i = 0; i < 10; i++) Once(); // JIT warm-up
        double[] times = Enumerable.Range(0, 41).Select(_ => Once()).Order().ToArray();
        double median = times[times.Length / 2], per10K = median * 10_000 / ctx.Length;
        GraphSignals sig = GraphSignals.ForTask(graph, ctx, order, task);
        output.WriteLine($"context {ctx.Length} tokens, budget {budget}, graph picks {sig.Count} (query targets {sig.QueryTargets}); "
            + $"selection median {median:F3} ms (min {times[0]:F3}, max {times[^1]:F3}) = {per10K:F3} ms per 10K tokens");
        Assert.True(sig.QueryTargets >= 30, $"query targets {sig.QueryTargets}");
        Assert.True(per10K < 1.0, $"selection {per10K:F3} ms per 10K tokens");
    }
}
