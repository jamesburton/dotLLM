using System.Diagnostics;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase S Task 6: <see cref="ContextSelector"/> over a small corpus (symbol chunks, one member each, char tokenizer) with a
/// BCL-only reference graph. Call chain: <c>Top.Run → Calc.Sum → MathUtil.Twice → MathUtil.Add ← MathUtilTests.T, Info.V</c>;
/// <c>Unused.Lonely</c> and <c>Calc.Other</c> are unconnected. <c>../shared/Info.cs</c> lies outside the corpus root (as a
/// scan's linked file can before <c>ChunkIndex.RelativizeScan</c>).
/// </summary>
public sealed class ContextSelectorTests
{
    private static readonly string Root = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "ctxsel-fixture", "repo"));

    private static readonly (string Path, string Text)[] Repo =
    [
        ("src/Lib/MathUtil.cs",
            "namespace Lib;\n\npublic static class MathUtil\n{\n    public static int Add(int a, int b) => a + b;\n\n"
            + "    public static int Twice(int x) => Add(x, x);\n}\n"),
        ("src/Lib/Unused.cs", "namespace Lib;\n\npublic static class Unused\n{\n    public static int Lonely() => 7;\n}\n"),
        ("src/App/Calc.cs",
            "using Lib;\n\nnamespace App;\n\n// Calc adds numbers up with the MathUtil helpers; this comment makes its first chunk the largest.\npublic class Calc\n{\n    public int Sum() => MathUtil.Twice(3);\n\n    public int Other() => 1;\n}\n"),
        ("src/App/Top.cs", "namespace App;\n\npublic class Top\n{\n    public int Run() => new Calc().Sum();\n}\n"),
        ("tests/LibTests/MathUtilTests.cs",
            "using Lib;\n\nnamespace LibTests;\n\npublic class MathUtilTests\n{\n    public void T()\n    {\n        _ = MathUtil.Add(1, 2) + MathUtil.Add(3, 4);\n    }\n}\n"),
        ("../shared/Info.cs", "namespace Lib;\n\npublic static class Info\n{\n    public static int V() => MathUtil.Add(0, 0);\n}\n"),
    ];

    private sealed record Fixture(ReferenceGraph Graph, Chunk[] Chunks, SourceFile[] Files)
    {
        public Chunk With(string needle) =>
            Chunks.Single(c => Files.Single(f => f.Path == c.FilePath).Text[c.StartOffset..c.EndOffset].Contains(needle, StringComparison.Ordinal));

        public int Tokens(params string[] needles) => needles.Sum(n => With(n).TokenIds.Length);
    }

    private static Fixture Build(bool reversed = false)
    {
        var tok = new CharTokenizer();
        SourceFile[] files = Repo.Select(f => new SourceFile(f.Path, f.Text)).OrderBy(f => f.Path, StringComparer.Ordinal).ToArray();
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        if (reversed)
        {
            files = files.Reverse().ToArray();
            chunks = chunks.Reverse().ToArray();
        }
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(files, chunks);
        return new Fixture(graph, chunks, files);
    }

    private static string[] Ids(IEnumerable<SelectedChunk> s) => s.Select(c => c.Chunk.Id).ToArray();

    private static string[] Ids(params Chunk[] c) => c.Select(x => x.Id).ToArray();

    private static string[] Files(IEnumerable<SelectedChunk> s) => s.Select(c => c.Chunk.FilePath).Distinct().Order(StringComparer.Ordinal).ToArray();

    private sealed class FakeGit(params string[] changed) : IGitInfo
    {
        public List<string> Refs { get; } = [];

        public List<bool> MergeBase { get; } = [];

        public IReadOnlyList<string> ChangedFiles(string gitRef, bool sinceMergeBase = false)
        {
            Refs.Add(gitRef);
            MergeBase.Add(sinceMergeBase);
            return changed;
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    // Paths
    // -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Select_Glob_ReturnsMatchingFiles()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Paths = ["src/App/**"] }, fx.Files, fx.Graph);
        // Every chunk of the two App files.
        Assert.Equal(["src/App/Calc.cs", "src/App/Top.cs"], Files(r));
        Assert.Equal(fx.Chunks.Count(c => c.FilePath is "src/App/Calc.cs" or "src/App/Top.cs"), r.Count);
        Assert.All(r, s =>
        {
            Assert.Equal(SelectionReason.Path, s.Reason);
            Assert.Equal(SelectionPriority.Path, s.Priority);
            Assert.Contains("src/App/**", s.Detail, StringComparison.Ordinal);
            Assert.Equal(s.Chunk.EndOffset - s.Chunk.StartOffset, s.Tokens); // char tokenizer
        });

        // A pattern without '/' matches the file name at any depth; '!' removes; '*' stays inside a segment.
        r = ContextSelector.Select(new SelectionRequest { Paths = ["*.cs", "!src/App/*", "!tests/**"] }, fx.Files, fx.Graph);
        Assert.Equal(["src/Lib/MathUtil.cs", "src/Lib/Unused.cs"], Files(r));
        r = ContextSelector.Select(new SelectionRequest { Paths = ["src/*.cs"] }, fx.Files, fx.Graph);
        Assert.Empty(r);
        Assert.Equal(["src/*.cs"], r.UnmatchedPatterns);
    }

    [Fact]
    public void Select_Glob_WindowsSeparatorsAndDotSegmentsAreNormalised()
    {
        Fixture fx = Build();
        string[] Sel(params string[] p) => Files(ContextSelector.Select(new SelectionRequest { Paths = p }, fx.Files, fx.Graph));
        string[] expected = ["src/App/Calc.cs", "src/App/Top.cs"];
        Assert.Equal(expected, Sel("src\\App\\*.cs"));
        Assert.Equal(expected, Sel(".\\src\\App\\"));
        Assert.Equal(expected, Sel("./src//App/./*.cs"));
        Assert.Equal(expected, Sel("src/App/"));
        Assert.Equal(["src/App/Top.cs"], Sel("src/App/T?p.cs"));
        Assert.Equal(["src/App/Calc.cs"], Sel("**/Calc.cs"));
        Assert.Equal(["src/App/Calc.cs", "src/App/Top.cs", "src/Lib/MathUtil.cs", "src/Lib/Unused.cs"], Sel("src/**/*.cs"));
        if (OperatingSystem.IsWindows())
            Assert.Equal(expected, Sel("SRC\\app\\*.CS")); // case-insensitive on Windows, as RoslynGraphBuilder.PathComparer
        else
            Assert.Empty(Sel("SRC/app/*.CS"));
    }

    [Theory]
    [InlineData("**", "src/a.cs", true)]
    [InlineData("**", "../shared/a.cs", false)]
    [InlineData("**/*.cs", "a.cs", true)]
    [InlineData("**/*.cs", "x/y/z.cs", true)]
    [InlineData("*.cs", "x/y/z.cs", true)]
    [InlineData("*.cs", "x/y/z.csx", false)]
    [InlineData("x/*.cs", "x/y/z.cs", false)]
    [InlineData("x/**/z.cs", "x/z.cs", true)]
    [InlineData("x/**/z.cs", "x/y/w/z.cs", true)]
    [InlineData("x/**", "x", false)]
    [InlineData("x/**", "xy/z.cs", false)]
    [InlineData("a+b(1).cs", "a+b(1).cs", true)]
    [InlineData("a+b(1).cs", "aab(1).cs", false)]
    [InlineData("a.cs", "abcs", false)]
    [InlineData("[ab].cs", "[ab].cs", true)]
    [InlineData("[ab].cs", "a.cs", false)]
    [InlineData("x/?.cs", "x/ab.cs", false)]
    [InlineData("x?y.cs", "x/y.cs", false)]
    [InlineData("src/x.cs", "src\\x.cs", true)]
    [InlineData("src/x.cs", "./src/x.cs", true)]
    [InlineData("src/x.cs", ".\\src\\.\\x.cs", true)]
    [InlineData("[ab].cs", "{a,b}.cs", false)]
    [InlineData("{a,b}.cs", "{a,b}.cs", true)]
    [InlineData("{a,b}.cs", "a.cs", false)]
    public void PathGlob_Matches(string pattern, string path, bool expected) => Assert.Equal(expected, PathGlob.Parse(pattern).IsMatch(path));

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("!")]
    [InlineData("C:/repo/**")]
    [InlineData("C:\\repo\\*.cs")]
    [InlineData("c:x.cs")]
    [InlineData("/src/**")]
    [InlineData("\\\\server\\share\\*.cs")]
    [InlineData("../shared/*.cs")]
    [InlineData("src/../../x/*.cs")]
    [InlineData("src\\..\\x.cs")]
    public void PathGlob_RootedOrOutsideTheRoot_IsRejected(string pattern) =>
        Assert.Throws<ArgumentException>(() => PathGlob.Parse(pattern));

    [Fact]
    public void Select_PathsOutsideTheRoot_GlobsNeverReachThem_ButOtherCriteriaDo()
    {
        Fixture fx = Build();
        Assert.DoesNotContain("../shared/Info.cs", Files(ContextSelector.Select(new SelectionRequest { Paths = ["**"] }, fx.Files, fx.Graph)));
        Assert.Throws<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Paths = ["../shared/*.cs"] }, fx.Files, fx.Graph));
        Assert.Throws<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Paths = ["src/**"], Exclude = [Root + "/src/**"] }, fx.Files, fx.Graph));

        // The symbol walk does reach the file (it is in the scan), and so does a changed-files list.
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "Info.V", Hops = 0 }, fx.Files, fx.Graph);
        Assert.Equal(Ids(fx.With("int V(")), Ids(r));
        string outsideInScan = Path.GetFullPath(Path.Combine(Root, "..", "shared", "Info.cs"));
        string outsideNotInScan = Path.GetFullPath(Path.Combine(Root, "..", "other", "y.cs"));
        string otherVolume = OperatingSystem.IsWindows() ? (Root.StartsWith('Z') ? "Y:\\elsewhere\\x.cs" : "Z:\\elsewhere\\x.cs") : "/elsewhere/x.cs";
        var git = new FakeGit(outsideInScan, outsideNotInScan, otherVolume);
        r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph, git, Root);
        Assert.Equal(["../shared/Info.cs"], Files(r));
        Assert.Equal(2, r.UnmatchedChangedFiles.Count);
        Assert.Contains("../other/y.cs", r.UnmatchedChangedFiles);
    }

    // -------------------------------------------------------------------------------------------------------------
    // Symbol neighbourhood
    // -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Select_SymbolNeighbourhood_IncludesDefinitionAndCallersWithinHops()
    {
        Fixture fx = Build();
        Chunk twice = fx.With("int Twice("), add = fx.With("int Add("), sum = fx.With("int Sum("), run = fx.With("int Run("),
            t = fx.With("void T("), v = fx.With("int V("), lonely = fx.With("Lonely"), other = fx.With("int Other(");

        SelectionResult r0 = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 0 }, fx.Files, fx.Graph);
        Assert.Equal(Ids(twice), Ids(r0));
        Assert.Equal(SelectionReason.Definition, r0[0].Reason);
        Assert.Equal(0, r0[0].Hop);
        Assert.Equal(SelectionPriority.Definition, r0[0].Priority);
        Assert.Contains("Lib.MathUtil.Twice", r0[0].Detail, StringComparison.Ordinal);
        SymbolDefinition def = Assert.Single(r0.SymbolDefinitions);
        Assert.Equal(("Lib.MathUtil.Twice", "method", "src/Lib/MathUtil.cs", 7), (def.QualifiedName, def.Kind, def.FilePath, def.Line));

        // Hop 1: the callee (Add) and the caller (Sum). Ties on weight are ordered by path: src/App before src/Lib.
        SelectionResult r1 = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 1 }, fx.Files, fx.Graph);
        Assert.Equal(Ids(twice, sum, add), Ids(r1));
        SelectedChunk s = r1.Single(c => c.Chunk.Id == sum.Id), a = r1.Single(c => c.Chunk.Id == add.Id);
        Assert.Equal((SelectionReason.Caller, 1, SelectionPriority.Neighbour(1)), (s.Reason, s.Hop, s.Priority));
        Assert.Equal((SelectionReason.Callee, 1, SelectionPriority.Neighbour(1)), (a.Reason, a.Hop, a.Priority));
        Assert.True(s.Weight >= 1 && a.Weight >= 1);

        // Hop 2: the callers of Add (a test and the file outside the root) and of Sum (Run); never the unconnected members.
        SelectionResult r2 = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2 }, fx.Files, fx.Graph);
        Assert.Equal(Ids(twice, sum, add).Order(StringComparer.Ordinal), Ids(r2.Where(c => c.Hop <= 1)).Order(StringComparer.Ordinal));
        Assert.Equal(Ids(run, t, v).Order(StringComparer.Ordinal), Ids(r2.Where(c => c.Hop == 2)).Order(StringComparer.Ordinal));
        Assert.All(r2.Where(c => c.Hop == 2), c => Assert.Equal(SelectionReason.Caller, c.Reason));
        Assert.DoesNotContain(lonely.Id, Ids(r2));
        Assert.DoesNotContain(other.Id, Ids(r2));
        Assert.Equal(r2.Count, ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 9 }, fx.Files, fx.Graph).Count);

        // Within hop 2 (equal priority) the heavier neighbour ranks first: T references the hop-1 Add chunk four times
        // (MathUtil twice, Add twice), V and Run fewer; path order would put T last.
        SelectedChunk[] hop2 = r2.Where(c => c.Hop == 2).ToArray();
        Assert.Equal(t.Id, hop2[0].Chunk.Id);
        Assert.True(hop2[0].Weight > hop2[1].Weight);

        // No edge kinds: the walk stops at the definition.
        Assert.Equal(Ids(twice), Ids(ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2, EdgeKinds = [] }, fx.Files, fx.Graph)));
        // Only Use edges: Twice calls Add and is called by Sum (Call edges both), so nothing is a Use neighbour.
        Assert.Equal(Ids(twice), Ids(ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 1, EdgeKinds = [SymbolEdgeKind.Use] }, fx.Files, fx.Graph)));
        // Only Call edges: the same call chain as the default.
        Assert.Equal(Ids(r1), Ids(ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 1, EdgeKinds = [SymbolEdgeKind.Call] }, fx.Files, fx.Graph)));
    }

    [Theory]
    [InlineData("Twice")]
    [InlineData("Lib.MathUtil.Twice")]
    [InlineData("global::Lib.MathUtil.Twice")]
    [InlineData("MathUtil.Twice(int)")]
    [InlineData(" MathUtil.Twice ")]
    public void Select_Symbol_QualifiedForms(string symbol)
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = symbol, Hops = 0 }, fx.Files, fx.Graph);
        Assert.Equal(Ids(fx.With("int Twice(")), Ids(r));
    }

    [Fact]
    public void Select_TypeSymbol_DefinitionAndBodyChunks()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "App.Calc", Hops = 0 }, fx.Files, fx.Graph);
        // The class name is in the Sum chunk (the header glues forward); Other is the rest of the class.
        Assert.Equal(Ids(fx.With("int Sum("), fx.With("int Other(")), Ids(r));
        Assert.Equal([SelectionReason.Definition, SelectionReason.DefinitionBody], r.Select(c => c.Reason));
        Assert.Equal([SelectionPriority.Definition, SelectionPriority.DefinitionBody], r.Select(c => c.Priority));
        Assert.Equal("type", Assert.Single(r.SymbolDefinitions).Kind);
    }

    [Fact]
    public void Select_SymbolNotFound_ThrowsAClearError()
    {
        Fixture fx = Build();
        var ex = Assert.Throws<SymbolNotFoundException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "Twise" }, fx.Files, fx.Graph));
        Assert.Contains("'Twise'", ex.Message, StringComparison.Ordinal);
        Assert.Contains("not found", ex.Message, StringComparison.Ordinal);
        Assert.Equal("Twise", ex.Symbol);

        // Wrong case: suggested.
        ex = Assert.Throws<SymbolNotFoundException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "mathutil.twice" }, fx.Files, fx.Graph));
        Assert.Equal(["Lib.MathUtil.Twice"], ex.Suggestions);
        Assert.Contains("Lib.MathUtil.Twice", ex.Message, StringComparison.Ordinal);

        // Right name, wrong container: suggested.
        ex = Assert.Throws<SymbolNotFoundException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "Calc.Add" }, fx.Files, fx.Graph));
        Assert.Equal(["Lib.MathUtil.Add"], ex.Suggestions);

        // Declared, but only in a file the graph has no chunk of.
        Fixture partial = Build();
        Chunk[] noTop = partial.Chunks.Where(c => c.FilePath != "src/App/Top.cs").ToArray();
        ReferenceGraph g = new RoslynGraphBuilder(new CharTokenizer()).Build(partial.Files, noTop);
        ex = Assert.Throws<SymbolNotFoundException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "Top.Run" }, partial.Files, g));
        Assert.Contains("src/App/Top.cs", ex.Message, StringComparison.Ordinal);

        // Malformed symbols are argument errors.
        foreach (string bad in new[] { "", "  ", "A..B", ".A", "A.", "A(" })
            Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Symbol = bad }, fx.Files, fx.Graph));
    }

    [Fact]
    public void Select_Exclude_RemovesAndIsNotTraversed()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2, Exclude = ["tests/**"] }, fx.Files, fx.Graph);
        Assert.DoesNotContain("tests/LibTests/MathUtilTests.cs", Files(r));
        Assert.Contains(fx.With("int Run(").Id, Ids(r));

        // Run is reached only through Sum: excluding Calc.cs cuts it off.
        r = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2, Exclude = ["src/App/Calc.cs"] }, fx.Files, fx.Graph);
        Assert.DoesNotContain("src/App/Calc.cs", Files(r));
        Assert.DoesNotContain("src/App/Top.cs", Files(r));

        // An excluded definition is reported, not an error.
        r = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 1, Exclude = ["src/Lib/**"] }, fx.Files, fx.Graph);
        Assert.Empty(r);
        Assert.Contains(r.Notes, n => n.Contains("excluded", StringComparison.Ordinal));
    }

    [Fact]
    public void Select_ChunkMatchingSeveralCriteria_KeepsTheHighestPriorityAndAllReasons()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "Twice", Hops = 1, Paths = ["src/Lib/**"], ChunkKind = ChunkKind.Symbol },
            fx.Files, fx.Graph);
        SelectedChunk twice = r.Single(c => c.Chunk.Id == fx.With("int Twice(").Id);
        Assert.Equal(SelectionReason.Definition, twice.Reason);
        Assert.Equal(SelectionReason.Definition | SelectionReason.Path, twice.Reasons);
        SelectedChunk add = r.Single(c => c.Chunk.Id == fx.With("int Add(").Id);
        Assert.Equal((SelectionReason.Path, SelectionPriority.Path, 1), (add.Reason, add.Priority, add.Hop));
        Assert.Equal(SelectionReason.Path | SelectionReason.Callee, add.Reasons);
        Assert.All(r, c => Assert.Equal(ChunkKind.Symbol, c.Chunk.Kind));
        Assert.All(r, c => Assert.False(string.IsNullOrWhiteSpace(c.Detail)));
        // Each chunk once.
        Assert.Equal(r.Count, Ids(r).Distinct().Count());
    }

    // -------------------------------------------------------------------------------------------------------------
    // Changed since (git)
    // -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Select_ChangedSince_UsesGitDiff()
    {
        Fixture fx = Build();
        var git = new FakeGit(Path.Combine(Root, "src", "App", "Top.cs"), Path.Combine(Root, "README.md"), Path.Combine(Root, "src", "Gone.cs"));
        SelectionResult r = ContextSelector.Select(new SelectionRequest { ChangedSince = "origin/main" }, fx.Files, fx.Graph, git, Root);
        Assert.Equal(["origin/main"], git.Refs);
        Assert.Equal(Ids(fx.Chunks.Where(c => c.FilePath == "src/App/Top.cs").ToArray()), Ids(r));
        Assert.All(r, c =>
        {
            Assert.Equal(SelectionReason.Changed, c.Reason);
            Assert.Equal(SelectionPriority.Changed, c.Priority);
            Assert.Contains("origin/main", c.Detail, StringComparison.Ordinal);
        });
        Assert.Equal(["README.md", "src/Gone.cs"], r.UnmatchedChangedFiles);

        // An explicit path outranks a changed file under a budget (a git range can sweep in unrelated files).
        r = ContextSelector.Select(new SelectionRequest { ChangedSince = "origin/main", Paths = ["src/Lib/Unused.cs"], TokenBudget = fx.Tokens("Lonely") },
            fx.Files, fx.Graph, git, Root);
        Assert.Equal(Ids(fx.With("Lonely")), Ids(r));
        Assert.True(SelectionPriority.Path > SelectionPriority.Changed);
        Assert.Equal(["src/App/Top.cs"], Files(r.Trimmed));

        // A path from git that differs only in case is the same file on Windows.
        if (OperatingSystem.IsWindows())
        {
            r = ContextSelector.Select(new SelectionRequest { ChangedSince = "HEAD" }, fx.Files, fx.Graph, new FakeGit(Path.Combine(Root, "SRC", "app", "top.cs")), Root);
            Assert.Equal(["src/App/Top.cs"], Files(r));
        }
    }

    [Theory]
    [InlineData("--output=pwned.txt")]
    [InlineData("-p")]
    [InlineData("--exec=calc")]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("main\0--output=x")]
    [InlineData("main\n--output=x")]
    [InlineData(" -p")]
    public void Select_ChangedSince_OptionLikeOrMalformedRef_IsRejectedBeforeGitRuns(string gitRef)
    {
        Fixture fx = Build();
        var git = new FakeGit(Path.Combine(Root, "src", "App", "Top.cs"));
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { ChangedSince = gitRef }, fx.Files, fx.Graph, git, Root));
        Assert.Empty(git.Refs);
        Assert.ThrowsAny<ArgumentException>(() => GitCliInfo.ValidateRef(gitRef));
    }

    [Fact]
    public void Select_ChangedSince_ContractViolations()
    {
        Fixture fx = Build();
        // No IGitInfo.
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph));
        // IGitInfo must return absolute paths.
        Assert.Throws<InvalidOperationException>(() =>
            ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph, new FakeGit("src/App/Top.cs"), Root));
        // Without a root directory git paths cannot be mapped; a relative root is rejected too.
        foreach (string? root in new[] { null, "", "rel/dir" })
            Assert.ThrowsAny<ArgumentException>(() =>
                ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph, new FakeGit(Path.Combine(Root, "src", "App", "Top.cs")), root));
    }

    // -------------------------------------------------------------------------------------------------------------
    // Budget, determinism, empty selections, validation
    // -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Select_OverBudget_TrimsLowestPriorityFirst()
    {
        Fixture fx = Build();
        var request = new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2 };
        SelectionResult full = ContextSelector.Select(request, fx.Files, fx.Graph);
        Assert.Empty(full.Trimmed);
        Assert.Equal(full.SelectedTokens, full.CandidateTokens);

        // Budget = definition + hop 1: exactly hop 2 is trimmed.
        int budget = fx.Tokens("int Twice(", "int Sum(", "int Add(");
        SelectionResult r = ContextSelector.Select(request with { TokenBudget = budget }, fx.Files, fx.Graph);
        Assert.Equal(Ids(fx.With("int Twice("), fx.With("int Sum("), fx.With("int Add(")), Ids(r));
        Assert.Equal(budget, r.SelectedTokens);
        Assert.Equal(Ids(full.Where(c => c.Hop == 2)), Ids(r.Trimmed));
        Assert.Equal(full.CandidateTokens, r.CandidateTokens);

        // Any budget: the selection is the longest prefix of the full ranking that fits, and the trimmed rest is the remainder.
        for (int b = 0; b <= full.SelectedTokens + 1; b += 7)
        {
            SelectionResult rb = ContextSelector.Select(request with { TokenBudget = b }, fx.Files, fx.Graph);
            Assert.Equal(Ids(full), Ids(rb.Selected.Concat(rb.Trimmed)));
            Assert.True(rb.SelectedTokens <= b);
            if (rb.Trimmed.Count > 0) Assert.True(rb.SelectedTokens + rb.Trimmed[0].Tokens > b);
            Assert.Equal(rb.Selected.Sum(c => (long)c.Tokens), rb.SelectedTokens);
        }

        // Rank order: priority never increases; an explicit path (800) outranks a hop-1 neighbour (599).
        Assert.True(full.Zip(full.Skip(1)).All(p => p.First.Priority >= p.Second.Priority));
        int lonely = fx.Tokens("Lonely");
        r = ContextSelector.Select(request with { Paths = ["src/Lib/Unused.cs"], TokenBudget = fx.Tokens("int Twice(") + lonely }, fx.Files, fx.Graph);
        Assert.Equal(Ids(fx.With("int Twice("), fx.With("Lonely")), Ids(r));
        Assert.Equal([SelectionReason.Definition, SelectionReason.Path], r.Select(c => c.Reason));
    }

    [Fact]
    public void Select_BudgetSmallerThanTheFirstChunk_SelectsNothing()
    {
        Fixture fx = Build();
        var request = new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 1 };
        int first = fx.Tokens("int Twice(");
        foreach (int b in new[] { 0, 1, first - 1 })
        {
            SelectionResult r = ContextSelector.Select(request with { TokenBudget = b }, fx.Files, fx.Graph);
            Assert.Empty(r);
            Assert.Equal(0, r.SelectedTokens);
            Assert.Equal(3, r.Trimmed.Count);
            Assert.Equal(fx.With("int Twice(").Id, r.Trimmed[0].Chunk.Id);
        }
        // Strict prefix: a smaller lower-ranked chunk is not pulled in past the first chunk that does not fit.
        // Sum ranks before Add (equal weight, path order) and is the larger (Calc.cs has a long header comment): with room
        // for Add but not for Sum, only the definition is selected.
        int sum = fx.Tokens("int Sum("), add = fx.Tokens("int Add(");
        Assert.True(sum > add);
        SelectionResult strict = ContextSelector.Select(request with { TokenBudget = first + add }, fx.Files, fx.Graph);
        Assert.Equal(Ids(fx.With("int Twice(")), Ids(strict));
        Assert.Equal(Ids(fx.With("int Sum("), fx.With("int Add(")), Ids(strict.Trimmed));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContextSelector.Select(request with { TokenBudget = -1 }, fx.Files, fx.Graph));
    }

    [Fact]
    public void Select_Deterministic()
    {
        Fixture fx = Build(), rev = Build(reversed: true);
        var request = new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2, Paths = ["src/Lib/**"], TokenBudget = 260 };
        static string Sig(SelectionResult r) => string.Join("\n", r.Select(c => $"{c.Chunk.Id}|{c.Reason}|{c.Reasons}|{c.Priority}|{c.Hop}|{c.Weight}|{c.Detail}"))
            + "\n--\n" + string.Join("\n", r.Trimmed.Select(c => c.Chunk.Id));
        string a = Sig(ContextSelector.Select(request, fx.Files, fx.Graph));
        Assert.Equal(a, Sig(ContextSelector.Select(request, fx.Files, fx.Graph)));
        // Independent of the order of the scan's files and of the graph's chunks.
        Assert.Equal(a, Sig(ContextSelector.Select(request, rev.Files, rev.Graph)));
        Assert.Contains("--\n", a, StringComparison.Ordinal);
        Assert.False(a.EndsWith("--\n", StringComparison.Ordinal)); // the budget trims something, so the trimmed order is compared too
    }

    [Fact]
    public void Select_EmptySelection_IsAnEmptyResultNotAnError()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Paths = ["nothing/**", "*.vb"] }, fx.Files, fx.Graph);
        Assert.Empty(r);
        Assert.Empty(r.Trimmed);
        Assert.Equal(0, r.CandidateTokens);
        Assert.Equal(["nothing/**", "*.vb"], r.UnmatchedPatterns);
        Assert.Empty(r.InDocumentOrder());

        r = ContextSelector.Select(new SelectionRequest { ChangedSince = "HEAD" }, fx.Files, fx.Graph, new FakeGit(), Root);
        Assert.Empty(r);
    }

    [Fact]
    public void Select_InvalidRequests_AreRejected()
    {
        Fixture fx = Build();
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest(), fx.Files, fx.Graph));
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Exclude = ["src/**"] }, fx.Files, fx.Graph));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "Twice", Hops = -1 }, fx.Files, fx.Graph));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ContextSelector.Select(new SelectionRequest { Symbol = "Twice", Hops = ContextSelector.MaxHops + 1 }, fx.Files, fx.Graph));
        Assert.Throws<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Paths = ["src/**"], Exclude = ["!src/**"] }, fx.Files, fx.Graph));
        Assert.Throws<ArgumentNullException>(() => ContextSelector.Select(null!, fx.Files, fx.Graph));

        // The graph must be built over the given files.
        SourceFile[] other = fx.Files.Where(f => f.Path != "src/App/Top.cs").ToArray();
        Assert.Contains("src/App/Top.cs", Assert.Throws<ArgumentException>(() =>
            ContextSelector.Select(new SelectionRequest { Paths = ["**"] }, other, fx.Graph)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Select_InDocumentOrder_SortsByPathThenOffset()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2 }, fx.Files, fx.Graph);
        IReadOnlyList<SelectedChunk> doc = r.InDocumentOrder();
        Assert.Equal(r.Count, doc.Count);
        Assert.Equal(Ids(doc.OrderBy(c => c.Chunk.FilePath, StringComparer.Ordinal).ThenBy(c => c.Chunk.StartOffset)), Ids(doc));
        Assert.NotEqual(Ids(r), Ids(doc));
    }

    // -------------------------------------------------------------------------------------------------------------
    // Fix round (review-task6-report.md)
    // -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Select_Exclude_AppliesToPathsAndChangedFiles_AndReportsExcludedChangedFiles()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Paths = ["src/**"], Exclude = ["src/App/**"] }, fx.Files, fx.Graph);
        Assert.Equal(["src/Lib/MathUtil.cs", "src/Lib/Unused.cs"], Files(r));

        var git = new FakeGit(Path.Combine(Root, "src", "App", "Top.cs"), Path.Combine(Root, "src", "Lib", "Unused.cs"), Path.Combine(Root, "README.md"));
        r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main", Exclude = ["src/App/**"] }, fx.Files, fx.Graph, git, Root);
        Assert.Equal(["src/Lib/Unused.cs"], Files(r));
        Assert.Equal(["README.md"], r.UnmatchedChangedFiles);
        string note = Assert.Single(r.Notes, n => n.Contains("Exclude", StringComparison.Ordinal));
        Assert.Contains("src/App/Top.cs", note, StringComparison.Ordinal);
    }

    [Fact]
    public void Select_TwoPatternsMatchingOneFile_NeitherIsUnmatched()
    {
        Fixture fx = Build();
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Paths = ["src/Lib/*.cs", "**/MathUtil.cs", "nothing/**"] }, fx.Files, fx.Graph);
        Assert.Equal(["nothing/**"], r.UnmatchedPatterns);
        // The first matching pattern is the one named in the detail.
        Assert.All(r, c => Assert.Contains("'src/Lib/*.cs'", c.Detail, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("sv-SE")]
    [InlineData("tr-TR")]
    [InlineData("en-US")]
    public void Select_Order_IsOrdinal_UnderAnyCulture(string culture)
    {
        // Ordinal: "z/B.cs" < "z/a.b.cs" < "z/a.cs" < "z/a_b.cs"; culture-aware comparers put B last and may reorder '.' and '_'.
        (string Path, string Text)[] repo =
        [
            ("z/a_b.cs", "class A2 { }\n"), ("z/B.cs", "class B { }\n"), ("z/a.cs", "class A1 { }\n"), ("z/a.b.cs", "class A3 { }\n"),
            ("z/Ä.cs", "class Ae { }\n"), ("z/b-c.cs", "class Bc { }\n"),
        ];
        var tok = new CharTokenizer();
        SourceFile[] files = repo.Select(f => new SourceFile(f.Path, f.Text)).ToArray();
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(files, chunks);
        string[] ordinal = files.Select(f => f.Path).Order(StringComparer.Ordinal).ToArray();

        System.Globalization.CultureInfo saved = System.Globalization.CultureInfo.CurrentCulture;
        try
        {
            System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo(culture);
            SelectionResult r = ContextSelector.Select(new SelectionRequest { Paths = ["z/**"] }, files, graph);
            Assert.Equal(ordinal, r.Select(c => c.Chunk.FilePath).ToArray());
            Assert.Equal(ordinal, r.InDocumentOrder().Select(c => c.Chunk.FilePath).ToArray());
        }
        finally
        {
            System.Globalization.CultureInfo.CurrentCulture = saved;
        }
    }

    [Fact]
    public void Select_Symbol_PositionalRecordProperty_IsFound()
    {
        (string Path, string Text)[] repo =
        [
            ("r/Box.cs", "namespace N;\n\npublic sealed record Box(int Size, string Label)\n{\n    public int Twice => Size * 2;\n}\n"),
            ("r/Use.cs", "namespace N;\n\npublic static class Use\n{\n    public static int Get(Box b) => b.Size;\n}\n"),
        ];
        var tok = new CharTokenizer();
        SourceFile[] files = repo.Select(f => new SourceFile(f.Path, f.Text)).ToArray();
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(files, chunks);
        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "Box.Size", Hops = 0 }, files, graph);
        SymbolDefinition d = Assert.Single(r.SymbolDefinitions);
        Assert.Equal(("N.Box.Size", "property", "r/Box.cs", 3), (d.QualifiedName, d.Kind, d.FilePath, d.Line));
        Assert.Equal(SelectionReason.Definition, Assert.Single(r).Reason);
        // Unsupported declaration kinds (operators, indexers, finalizers) cannot be named: their "name" is not an identifier.
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Symbol = "Box.this" }, files, graph));
    }

    [Fact]
    public void Select_ChangedSinceMergeBase_IsPassedToGit_AndNamedInTheDetail()
    {
        Fixture fx = Build();
        var git = new FakeGit(Path.Combine(Root, "src", "App", "Top.cs"));
        SelectionResult r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main", ChangedSinceMergeBase = true }, fx.Files, fx.Graph, git, Root);
        Assert.Equal([true], git.MergeBase);
        Assert.All(r, c => Assert.Contains("merge base", c.Detail, StringComparison.Ordinal));
        ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph, git, Root);
        Assert.Equal([true, false], git.MergeBase);
        Assert.ThrowsAny<ArgumentException>(() => ContextSelector.Select(new SelectionRequest { Paths = ["**"], ChangedSinceMergeBase = true }, fx.Files, fx.Graph));
    }

    [SkippableFact]
    public void Select_ChangedSince_RootReachedThroughALink_MapsResolvedPaths()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ctxsel-link-" + Guid.NewGuid().ToString("N"));
        string real = Path.Combine(dir, "real"), link = Path.Combine(dir, "link");
        Directory.CreateDirectory(Path.Combine(real, "src", "App"));
        try
        {
            Skip.IfNot(TestLinks.TryCreateDirectoryLink(link, real), "Cannot create a directory junction or symlink here.");
            Fixture fx = Build();
            // An IGitInfo that reports resolved paths (as `git rev-parse --show-toplevel` does) while the root is the link.
            var git = new FakeGit(Path.Combine(real, "src", "App", "Top.cs"), Path.Combine(link, "src", "Lib", "Unused.cs"));
            SelectionResult r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, fx.Files, fx.Graph, git, link);
            Assert.Equal(["src/App/Top.cs", "src/Lib/Unused.cs"], Files(r));
            Assert.Empty(r.UnmatchedChangedFiles);
        }
        finally
        {
            TestLinks.DeleteTree(dir);
        }
    }

    [Fact]
    public void Select_MaxNeighboursPerHop_CapsEachHop_OffByDefault()
    {
        Fixture fx = Build();
        var request = new SelectionRequest { Symbol = "MathUtil.Twice", Hops = 2 };
        SelectionResult full = ContextSelector.Select(request, fx.Files, fx.Graph);
        Assert.Equal(6, full.Count);
        Assert.DoesNotContain(full.Notes, n => n.Contains("capped", StringComparison.Ordinal));

        // Cap 1: hop 1 keeps its best neighbour (Sum: equal weight, first by path), and hop 2 walks on from Sum only.
        SelectionResult capped = ContextSelector.Select(request with { MaxNeighboursPerHop = 1 }, fx.Files, fx.Graph);
        Assert.Equal(3, capped.Count);
        Assert.Equal(fx.With("int Sum(").Id, Assert.Single(capped, c => c.Hop == 1).Chunk.Id);
        Assert.Single(capped, c => c.Hop == 2);
        Assert.DoesNotContain(fx.With("int Add(").Id, Ids(capped.Where(c => c.Hop == 1)));
        Assert.Contains(capped.Notes, n => n.Contains("hop 1 capped", StringComparison.Ordinal) && n.Contains("1 of 2", StringComparison.Ordinal));

        Assert.Equal(Ids(fx.With("int Twice(")), Ids(ContextSelector.Select(request with { MaxNeighboursPerHop = 0 }, fx.Files, fx.Graph)));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContextSelector.Select(request with { MaxNeighboursPerHop = -1 }, fx.Files, fx.Graph));
    }
}
