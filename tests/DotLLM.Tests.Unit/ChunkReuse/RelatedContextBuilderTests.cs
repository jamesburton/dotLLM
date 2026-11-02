using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// E4b related-chunk contexts (<see cref="RelatedContextBuilder"/>): the required chunk, then a <c>relatedFraction</c> share of
/// the other chunks drawn from files within two reference-graph hops of the calling file (use → definition edges, imports
/// excluded, the masked call's own edges excluded), the rest random distractors, with the task generator's leak filters.
/// Fixture (one char per token, every member its own chunk):
/// <list type="bullet">
/// <item><c>App/Use.cs</c> — the calling file: <c>H.H0(1) + Extra()</c> (a call into its own file) in the prefix, then the masked call <c>Go(a)</c> to <c>Lib.M.Go</c>
/// through <c>using static Lib.M</c> (so the masked call is the only non-import reference to <c>M.cs</c>).</item>
/// <item><c>Lib/H.cs</c> — hop 1 (its methods call <c>D</c>; it imports namespace <c>Other</c>, declared only in the unrelated
/// <c>Other/O.cs</c>); also declares a different member named <c>Go</c>.</item>
/// <item><c>Lib/D.cs</c> — hop 2 (calls <c>E</c> and, in <c>Leak</c>, the masked text <c>M.Go(a)</c>).</item>
/// <item><c>Lib/E.cs</c> — hop 3; <c>Lib/M.cs</c> — the callee's file (hop 3 via <c>D.Leak</c>); <c>Lib/U.cs</c>, <c>Lib/W.cs</c> — unrelated.</item>
/// <item><c>App/Use2.cs</c> — a second call site of <c>M.Go</c> whose prefix references <c>W</c>.</item>
/// </list>
/// </summary>
public sealed class RelatedContextBuilderTests
{
    private const int N = 10;

    private static string Members(Func<int, string> member) => string.Concat(Enumerable.Range(0, N).Select(i => "    " + member(i) + "\n\n"));

    private static readonly (string Path, string Text)[] Repo =
    [
        ("App/Use.cs",
            "using Lib;\nusing static Lib.M;\n\nnamespace App;\n\npublic class Use\n{\n    public int Extra() => 4;\n\n" +
            "    public int Run()\n    {\n        var a = H.H0(1) + Extra();\n        return Go(a);\n    }\n}\n"),
        ("App/Use2.cs",
            "using Lib;\nusing static Lib.M;\n\nnamespace App;\n\npublic class Use2\n{\n" +
            "    public int Run2()\n    {\n        var z = W.W0(2);\n        return Go(z);\n    }\n}\n"),
        ("Lib/D.cs", "namespace Lib;\n\npublic static class D\n{\n" + Members(i => $"public static int D{i}(int x) => E.E{i}(x);") +
            "    public static int Leak(int a) => M.Go(a);\n}\n"),
        ("Lib/E.cs", "namespace Lib;\n\npublic static class E\n{\n" + Members(i => $"public static int E{i}(int x) => x + {i};") + "}\n"),
        ("Lib/H.cs", "using Other;\n\nnamespace Lib;\n\npublic static class H\n{\n" + Members(i => $"public static int H{i}(int x) => D.D{i}(x);") +
            "    public static int Go() => H0(9);\n}\n"),
        ("Other/O.cs", "namespace Other;\n\npublic static class O\n{\n    public static int O1() => 1;\n}\n"),
        ("Lib/M.cs", "namespace Lib;\n\npublic static class M\n{\n    public static int Go(int x) => x + 1;\n\n    public static int Other() => 5;\n}\n"),
        ("Lib/U.cs", "namespace Lib;\n\npublic static class U\n{\n" + Members(i => $"public static int U{i}() => {i};") + "}\n"),
        ("Lib/W.cs", "namespace Lib;\n\npublic static class W\n{\n" + Members(i => $"public static int W{i}(int x) => x - {i};") + "}\n"),
    ];

    private sealed record Fixture(Corpus Corpus, ReferenceGraph Graph, CodeTask Task, CodeTask Task2, Chunk Required);

    private static readonly Lazy<Fixture> Shared = new(() =>
    {
        ITokenizer tok = new CharTokenizer();
        var corpus = new Corpus(Repo.Select(f => new SourceFile(f.Path, f.Text)), new SymbolChunker(tok, 4096), tok);
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(corpus);
        var gen = new CrossFileTaskGenerator(new TaskGeneratorOptions
        {
            ContextTokens = 100_000, ContinuationTokens = 24, PrefixTokens = 400, TasksPerContext = 1,
        });
        IReadOnlyList<CodeTask> tasks = gen.Generate(corpus, 1000, seed: 0);
        CodeTask Find(string path) => tasks.Single(t => t.CallSitePath == path && t.MaskedText.TrimStart('.', ' ').StartsWith("Go(", StringComparison.Ordinal));
        CodeTask task = Find("App/Use.cs"), task2 = Find("App/Use2.cs");
        Chunk required = Assert.Single(task.RequiredChunks);
        Assert.Equal("Lib/M.cs", required.FilePath);
        Assert.Equal(required.Id, Assert.Single(task2.RequiredChunks).Id);
        return new Fixture(corpus, graph, task, task2, required);
    });

    private static Fixture F => Shared.Value;

    private static int TokensOf(IEnumerable<Chunk> chunks) => chunks.Sum(c => c.TokenIds.Length);

    private static Chunk[] NonRequired(CodeContextPlan plan) => plan.Chunks.Where(c => c.Id != F.Required.Id).ToArray();

    private static bool IsRelated(Chunk c) => c.FilePath is "Lib/H.cs" or "Lib/D.cs";

    /// <summary>Related chunks that pass the leak filters (not D.Leak with the masked text, not H.Go with the callee's name).</summary>
    private static IEnumerable<Chunk> EligibleRelated() => F.Corpus.Chunks.Where(IsRelated)
        .Where(c => !F.Corpus.TextOf(c).Contains("Go(a)", StringComparison.Ordinal) && !F.Corpus.TextOf(c).Contains(" Go(", StringComparison.Ordinal));

    [Fact]
    public void Related_Neighbourhood_IsTwoHopsOfUseEdges_WithoutImportsOrTheMaskedCall()
    {
        IReadOnlyDictionary<string, int> hood = RelatedContextBuilder.Neighbourhood(F.Graph, [F.Task]);
        Assert.Equal(new Dictionary<string, int> { ["Lib/H.cs"] = 1, ["Lib/D.cs"] = 2 }, hood.OrderBy(kv => kv.Key).ToDictionary());

        // Two tasks of the same definition: the union of their calling files' neighbourhoods.
        IReadOnlyDictionary<string, int> both = RelatedContextBuilder.Neighbourhood(F.Graph, [F.Task, F.Task2]);
        Assert.Equal(new Dictionary<string, int> { ["Lib/H.cs"] = 1, ["Lib/W.cs"] = 1, ["Lib/D.cs"] = 2 }, both.OrderBy(kv => kv.Key).ToDictionary());
    }

    [Fact]
    public void Related_RequiredChunkAlwaysIncluded()
    {
        int req = F.Required.TokenIds.Length;
        foreach (double fraction in new[] { 0.0, 0.5, 1.0 })
            foreach (int budget in new[] { req, req + 30, req + 300, 100_000 })
                for (int seed = 0; seed < 4; seed++)
                {
                    CodeContextPlan plan = RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, budget, fraction, seed);
                    Assert.Single(plan.Chunks, c => c.Id == F.Required.Id);
                    Assert.True(TokensOf(plan.Chunks) <= budget, $"{TokensOf(plan.Chunks)} > {budget}");
                    Assert.Equal(plan.Chunks.Count, plan.Chunks.Select(c => c.Id).Distinct().Count());
                    CodeTask t = Assert.Single(plan.Tasks);
                    Assert.Equal(F.Task.Id, t.Id);
                    Assert.Equal(NonRequired(plan).Select(c => c.Id), t.DistractorChunks.Select(c => c.Id));
                }
        Assert.Throws<ArgumentException>(() => RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, req - 1, 0.5, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, 100_000, 1.5, 0));
    }

    [Fact]
    public void Related_FractionRespected_WithinOneChunk()
    {
        // Budget for about 8-12 member chunks: neither pool runs out (about 22 related, 30+ random).
        int budget = F.Required.TokenIds.Length + 12 * 50;
        foreach (double fraction in new[] { 0.0, 0.25, 0.5, 0.75, 1.0 })
            for (int seed = 0; seed < 6; seed++)
            {
                Chunk[] others = NonRequired(RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, budget, fraction, seed));
                int related = others.Count(IsRelated);
                Assert.True(others.Length >= 6, $"only {others.Length} chunks");
                Assert.True(Math.Abs(related - fraction * others.Length) <= 1.0,
                    $"fraction {fraction} seed {seed}: {related} of {others.Length} related");
            }
    }

    [Fact]
    public void Related_NoChunkFromOutsideTwoHopsUnlessFilling()
    {
        int req = F.Required.TokenIds.Length;
        // Few chunks fit: all related, and all from hop 1 (H) before any from hop 2 (D).
        for (int seed = 0; seed < 6; seed++)
        {
            Chunk[] small = NonRequired(RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, req + 4 * 45, 1.0, seed));
            Assert.Contains(small, IsRelated);
            Assert.All(small.Where(IsRelated), c => Assert.Equal("Lib/H.cs", c.FilePath));
            AssertOutsideChunksOnlyFill(small, req + 4 * 45, seed);

            int midBudget = req + 15 * 45;
            Chunk[] mid = NonRequired(RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, midBudget, 1.0, seed));
            Assert.Contains(mid, c => c.FilePath == "Lib/D.cs");
            AssertOutsideChunksOnlyFill(mid, midBudget, seed);
        }

        // Everything fits: every eligible related chunk is used, and only then outside chunks fill the budget.
        Chunk[] all = NonRequired(RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, 100_000, 1.0, 3));
        string[] eligibleRelated = EligibleRelated().Select(c => c.Id).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(eligibleRelated, all.Where(IsRelated).Select(c => c.Id).Order(StringComparer.Ordinal).ToArray());
        Assert.Contains(all, c => c.FilePath is "Lib/E.cs" or "Lib/U.cs" or "Lib/W.cs");
    }

    /// <summary>
    /// Chunks from outside the neighbourhood (with relatedFraction 1) may only fill: when the first was added, the budget left
    /// (at most the final slack plus the outside chunks) fitted no unused eligible related chunk.
    /// </summary>
    private static void AssertOutsideChunksOnlyFill(Chunk[] chosen, int budget, int seed)
    {
        Chunk[] outside = chosen.Where(c => !IsRelated(c)).ToArray();
        if (outside.Length == 0) return;
        int left = budget - F.Required.TokenIds.Length - TokensOf(chosen) + TokensOf(outside);
        var used = chosen.Select(c => c.Id).ToHashSet();
        int smallestUnused = EligibleRelated().Where(c => !used.Contains(c.Id)).Select(c => c.TokenIds.Length).DefaultIfEmpty(int.MaxValue).Min();
        Assert.True(smallestUnused > left, $"seed {seed}: outside chunk added while a related chunk of {smallestUnused} tokens fitted {left}");
    }

    [Fact]
    public void Related_SameSeed_SameContext()
    {
        int budget = F.Required.TokenIds.Length + 10 * 45;
        string Ids(int seed) => string.Join('|', RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, budget, 0.5, seed).Chunks.Select(c => c.Id));
        for (int seed = 0; seed < 4; seed++) Assert.Equal(Ids(seed), Ids(seed));
        Assert.True(Enumerable.Range(0, 6).Select(Ids).Distinct().Count() >= 4, "different seeds should give different contexts");
        // The chunk order is shuffled too: the required chunk is not always first (where naive reuse would already be exact).
        int[] positions = Enumerable.Range(0, 8).Select(seed => RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, budget, 0.5, seed)
            .Chunks.Select(c => c.Id).ToList().IndexOf(F.Required.Id)).ToArray();
        Assert.True(positions.Distinct().Count() >= 3, "required chunk positions " + string.Join(',', positions));
    }

    [Fact]
    public void Related_MaskedTextNeverInContext()
    {
        string masked = F.Task.MaskedText.TrimStart('.', '?', ' ');
        Assert.StartsWith("Go(a)", masked, StringComparison.Ordinal);
        // The masked call is the only non-import reference from Use.cs to M.cs, so M.cs is not reachable through it.
        Assert.DoesNotContain("Lib/M.cs", RelatedContextBuilder.Neighbourhood(F.Graph, [F.Task]).Keys);
        foreach (double fraction in new[] { 0.0, 0.5, 1.0 })
            for (int seed = 0; seed < 4; seed++)
            {
                CodeContextPlan plan = RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, 100_000, fraction, seed);
                Assert.All(plan.Chunks, c => Assert.DoesNotContain(masked, F.Corpus.TextOf(c), StringComparison.Ordinal));
                // D.Leak (hop 2) holds the masked call text: never used, even when every other chunk fits.
                Assert.DoesNotContain(plan.Chunks, c => F.Corpus.TextOf(c).Contains("Leak", StringComparison.Ordinal));
            }
    }

    [Fact]
    public void Related_LeakFiltersApplied_CallingFileAndSameNameDeclarations()
    {
        foreach (double fraction in new[] { 0.0, 0.5, 1.0 })
        {
            CodeContextPlan plan = RelatedContextBuilder.Build(F.Corpus, F.Task, F.Graph, 100_000, fraction, 1);
            Assert.DoesNotContain(plan.Chunks, c => c.FilePath == "App/Use.cs");
            // H.Go (hop 1) declares a member with the callee's name: excluded like the generator's distractors.
            Assert.DoesNotContain(NonRequired(plan), c => F.Corpus.TextOf(c).Contains("int Go(", StringComparison.Ordinal));
            // Everything else that passes the filters fits, so the context is the whole eligible corpus.
            Assert.Contains(plan.Chunks, c => c.FilePath == "App/Use2.cs");
            Assert.Contains(plan.Chunks, c => c.FilePath == "Lib/M.cs" && c.Id != F.Required.Id);
        }
        // A plan's tasks must share their required chunk(s); every calling file is excluded.
        CodeContextPlan two = RelatedContextBuilder.Build(F.Corpus, [F.Task, F.Task2], F.Graph, 100_000, 0.5, 0);
        Assert.Equal(2, two.Tasks.Count);
        Assert.DoesNotContain(two.Chunks, c => c.FilePath is "App/Use.cs" or "App/Use2.cs");
        CodeTask other = new CrossFileTaskGenerator(new TaskGeneratorOptions { ContextTokens = 100_000, ContinuationTokens = 24, PrefixTokens = 400, TasksPerContext = 1 })
            .Generate(F.Corpus, 1000, 0).First(t => t.RequiredChunks[0].Id != F.Required.Id);
        Assert.Throws<ArgumentException>(() => RelatedContextBuilder.Build(F.Corpus, [F.Task, other], F.Graph, 100_000, 0.5, 0));
    }
}
