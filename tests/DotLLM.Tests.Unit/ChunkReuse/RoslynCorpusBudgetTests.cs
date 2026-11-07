using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Whole-corpus budget and cancellation inside ONE slow bind (#1664, from the #1676 review). The zero-budget tests in
/// <see cref="RoslynBudgetTests"/> are caught by the top-of-loop token check, so they cannot tell whether the token reaches
/// <c>GetSymbolInfo</c>; a single invocation whose overload resolution is exponential can.
/// </summary>
public sealed class RoslynCorpusBudgetTests
{
    /// <summary>One statement whose binding is exponential in <paramref name="depth"/>: nested lambdas over two overloads.</summary>
    internal static string SlowBind(int depth)
    {
        string call = "0";
        for (int i = 0; i < depth; i++) call = $"M(x{i} => {call})";
        return "using System;\nstatic class Slow\n{\n    static int M(Func<int, int> f) => 0;\n    static int M(Func<string, int> f) => 0;\n    static int Go() => " + call + ";\n}\n";
    }

    private const string Ok = "class Lib\n{\n    public static int Add(int a, int b) => a + b;\n}\n\nclass Use\n{\n    int F() => Lib.Add(1, 2);\n}\n";

    // Depth 20 binds for about a minute (depth 18 took 12 s, each level doubles it). Measured: Roslyn does NOT poll the cancellation
    // token inside this overload resolution (a per-file budget of 300 ms let the bind run its full 2.5 minutes at depth 22), so only
    // the corpus budget, which abandons the thread after the grace, can bound it. The abandoned thread finishes in the background.
    private const int Depth = 20;
    private static readonly TimeSpan Prompt = TimeSpan.FromSeconds(20);
    private static readonly TimeSpan Grace = TimeSpan.FromSeconds(1);

    [Fact]
    public void CallResolution_CorpusBudget_StopsASlowBind_WithTimeoutException()
    {
        ProjectCompilation p = ProjectCompilation.BclOnly([new SourceFile("slow.cs", SlowBind(Depth))]);
        var sw = Stopwatch.StartNew();

        // The per-file budget is far longer than the corpus budget: only the corpus budget can stop this.
        Assert.Throws<TimeoutException>(() => CallResolution.SitesWithin([p], TimeSpan.FromMinutes(30), corpusBudget: TimeSpan.FromMilliseconds(300), grace: Grace));

        Assert.True(sw.Elapsed < Prompt, "took " + sw.Elapsed);
    }

    [Fact]
    public void GraphBuilder_CorpusBudget_StopsASlowBind_WithTimeoutException()
    {
        var tok = new CharTokenizer();
        SourceFile[] files = [new("slow.cs", SlowBind(Depth))];
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        var sw = Stopwatch.StartNew();

        Assert.Throws<TimeoutException>(() => new RoslynGraphBuilder(tok)
            { FileBudget = TimeSpan.FromMinutes(30), CorpusBudget = TimeSpan.FromMilliseconds(300), Grace = Grace }.Build(files, chunks));

        Assert.True(sw.Elapsed < Prompt, "took " + sw.Elapsed);
    }

    [Fact]
    public void CorpusBudget_ManyFastFiles_AreNotCutShort()
    {
        // Control: a generous corpus budget changes nothing for ordinary input.
        ProjectCompilation p = ProjectCompilation.BclOnly([new SourceFile("ok.cs", Ok)]);
        Assert.NotEmpty(CallResolution.SitesWithin([p], TimeSpan.FromSeconds(30), corpusBudget: TimeSpan.FromMinutes(5)));
    }
}
