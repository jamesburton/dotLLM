using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Hostile source text must not take the process down through the graph builder. The bracket-depth guard counts only
/// <c>( [ {</c>; these shapes nest without brackets (generic angles, '.' chains, ternaries, unary operators) and reach Roslyn's
/// binder, which recurses per level (a stack overflow cannot be caught). The builder skips a file whose syntax tree is deeper
/// than <see cref="RoslynParsing.MaxSyntaxDepth"/> and binds on the large-stack thread.
/// </summary>
public sealed class GraphBuilderHostileInputTests
{
    private const string Ok = "class Lib\n{\n    public static int Add(int a, int b) => a + b;\n}\n\nclass Use\n{\n    int F() => Lib.Add(1, 2);\n}\n";

    private static string Wrap(string body) => "class H\n{\n    void M(object o, bool b, int x)\n    {\n        " + body + "\n    }\n}\n";

    public static TheoryData<string, string> Shapes() => new()
    {
        { "generic angles", Wrap("var v = default(" + string.Concat(Enumerable.Repeat("System.Collections.Generic.List<", 4000)) + "int" + new string('>', 4000) + ");") },
        { "dot chain", Wrap("var s = o" + string.Concat(Enumerable.Repeat(".ToString()", 30_000)) + ";") },
        { "ternary chain", Wrap("int y = " + string.Concat(Enumerable.Repeat("b ? 1 : ", 20_000)) + "0;") },
        { "unary minus", Wrap("int y = " + new string('-', 50_000) + "x;") },
        { "else-if chain", Wrap(string.Concat(Enumerable.Repeat("if (b) x = 1; else ", 20_000)) + "x = 2;") },
    };

    private static IEnumerable<Microsoft.CodeAnalysis.MetadataReference> Bcl() =>
        ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator)
            .Where(p => Path.GetFileName(p) is "System.Private.CoreLib.dll" or "System.Runtime.dll" or "netstandard.dll")
            .Select(p => Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(p));

    private static ReferenceGraph Build(SourceFile hostile, out SourceFile ok)
    {
        ok = new SourceFile("ok.cs", Ok);
        SourceFile[] files = [ok, hostile];
        var tok = new CharTokenizer();
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        return new RoslynGraphBuilder(tok).Build(files, chunks);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Build_HostileNesting_DoesNotCrash_AndTheOtherFilesStillGetTheirEdges(string shape, string text)
    {
        // Control: the input really is hostile, and the bracket guard alone does not see it.
        Assert.False(RoslynParsing.TooDeep(text, out _), shape + ": the bracket guard must not be what stops this input");

        ReferenceGraph graph = Build(new SourceFile("hostile.cs", text), out SourceFile ok);

        // The ordinary file is still bound: 'Lib.Add(1, 2)' is a call edge.
        Assert.Contains(graph.QueryEdges(ok.Path, 0, ok.Text.Length), e => e.Kind == SymbolEdgeKind.Call);
        // The hostile file contributes no edge (its symbols are not resolved).
        Assert.Empty(graph.QueryEdges("hostile.cs", 0, text.Length));
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SymbolLocatorFind_HostileNesting_DoesNotCrash_AndTheOtherFilesAreSearched(string shape, string text)
    {
        Assert.False(RoslynParsing.TooDeep(text, out _), shape);
        var files = new[] { new SourceFile("hostile.cs", text + "\nclass Target { }\n"), new SourceFile("ok.cs", "class Target { }\n") };

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find(files, ["Target"]);

        Assert.Contains(matches, m => m.FilePath == "ok.cs");
    }

    /// <summary>The corpus' own compilation is bound as a whole by anything that asks for its diagnostics or symbols.</summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public void CorpusCompilation_HostileNesting_LeavesTheFileOut_AndBindsTheRest(string shape, string text)
    {
        var tok = new CharTokenizer();
        var corpus = new Corpus([new SourceFile("ok.cs", Ok), new SourceFile("hostile.cs", text)], new SymbolChunker(tok, 4096), tok);

        // Force binding of every tree in the compilation: an unguarded hostile tree overflows the stack here.
        var diagnostics = corpus.Compilation.GetDiagnostics();

        Assert.DoesNotContain(diagnostics, d => d.Severity == Microsoft.CodeAnalysis.DiagnosticSeverity.Error && d.Location.SourceTree?.FilePath == "ok.cs");
        Assert.Empty(corpus.TreeOf("hostile.cs").GetRoot().DescendantNodes()); // replaced by an empty tree: none of its symbols resolve
        Assert.NotEmpty(corpus.TreeOf("ok.cs").GetRoot().DescendantNodes());
        Assert.True(shape.Length > 0);
    }

    /// <summary>The projects overload takes compilations built elsewhere (a solution scan): the corpus' own guard has not run on them.</summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public void BuildFromProjects_HostileNesting_DoesNotCrash_AndTheOtherFilesStillGetTheirEdges(string shape, string text)
    {
        var ok = new SourceFile("ok.cs", Ok);
        // The hostile file also refers to a symbol of the ordinary file: leaving it unbound must not throw.
        var hostile = new SourceFile("hostile.cs", text + "\nclass Ref { int Q() => Lib.Add(1, 2); }\n");
        var trees = new Dictionary<string, Microsoft.CodeAnalysis.SyntaxTree>(StringComparer.Ordinal)
        {
            [ok.Path] = RoslynParsing.Parse(ok.Text, ok.Path),
            [hostile.Path] = RoslynParsing.Parse(hostile.Text, hostile.Path), // parsed as given: no corpus guard in between
        };
        Assert.True(RoslynParsing.SyntaxDepth(trees[hostile.Path].GetRoot()) > RoslynParsing.MaxSyntaxDepth, shape + ": control, the tree is hostile");
        var compilation = Microsoft.CodeAnalysis.CSharp.CSharpCompilation.Create("p", trees.Values, Bcl(),
            new Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions(Microsoft.CodeAnalysis.OutputKind.DynamicallyLinkedLibrary));
        var tok = new CharTokenizer();
        Chunk[] chunks = new[] { ok, hostile }.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();

        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build([new ProjectCompilation("p", compilation, trees)], chunks);

        Assert.Contains(graph.QueryEdges(ok.Path, 0, ok.Text.Length), e => e.Kind == SymbolEdgeKind.Call);
        Assert.Empty(graph.QueryEdges(hostile.Path, 0, text.Length));
    }

    /// <summary>Shapes that passed the previous 2048 cap and overflowed a caller's ordinary stack in <c>GetDiagnostics()</c>.</summary>
    public static TheoryData<string, string> CallerStackShapes() => new()
    {
        { "nested lambdas x1500", Wrap("System.Func<int, object> f = " + string.Concat(Enumerable.Repeat("v => ", 1500)) + "1;") },
        { "ternary chain x1100", Wrap("int y = " + string.Concat(Enumerable.Repeat("b ? 1 : ", 1100)) + "0;") },
        { "nested switch expressions x400", Wrap("var y = " + string.Concat(Enumerable.Repeat("x switch { _ => ", 400)) + "1" + new string('}', 400) + ";") },
        { "nested lambdas x2000", Wrap("System.Func<int, object> f = " + string.Concat(Enumerable.Repeat("v => ", 2000)) + "1;") },
    };

    /// <summary>The 1 MiB default of a Windows thread: what an ordinary caller binds on (the runtime may give more, never less).</summary>
    private static void OnOrdinaryStack(Action work)
    {
        Exception? error = null;
        var t = new Thread(() => { try { work(); } catch (Exception ex) { error = ex; } }, 1024 * 1024);
        t.Start();
        t.Join();
        if (error is not null) throw new Xunit.Sdk.XunitException("work failed: " + error);
    }

    [Theory]
    [MemberData(nameof(CallerStackShapes))]
    public void CorpusCompilation_BoundOnAnOrdinaryStack_DoesNotOverflow(string shape, string text)
    {
        var tok = new CharTokenizer();
        var corpus = new Corpus([new SourceFile("ok.cs", Ok), new SourceFile("hostile.cs", text)], new SymbolChunker(tok, 4096), tok);
        Assert.False(RoslynParsing.TooDeep(text, out _), shape + ": the bracket guard must not be what stops this input");

        OnOrdinaryStack(() => _ = corpus.Compilation.GetDiagnostics());

        Assert.Empty(corpus.TreeOf("hostile.cs").GetRoot().DescendantNodes());
    }

    [Theory]
    [MemberData(nameof(CallerStackShapes))]
    public void BclOnlyProjectCompilation_BoundOnAnOrdinaryStack_DoesNotOverflow(string shape, string text)
    {
        ProjectCompilation p = ProjectCompilation.BclOnly([new SourceFile("ok.cs", Ok), new SourceFile("hostile.cs", text)]);

        OnOrdinaryStack(() => _ = p.Compilation.GetDiagnostics());

        Assert.Empty(p.Trees["hostile.cs"].GetRoot().DescendantNodes());
        Assert.True(shape.Length > 0);
    }

    [Theory]
    [MemberData(nameof(CallerStackShapes))]
    public void CallResolutionSites_CalledOnAnOrdinaryStack_DoesNotOverflow_AndStillMeasuresTheOtherFiles(string shape, string text)
    {
        // Trees parsed as given (a solution scan): no corpus guard has run on them.
        var ok = new SourceFile("ok.cs", Ok);
        var hostile = new SourceFile("hostile.cs", text + "\nclass Ref { int Q() => Lib.Add(1, 2); }\n");
        var trees = new Dictionary<string, Microsoft.CodeAnalysis.SyntaxTree>(StringComparer.Ordinal)
        {
            [ok.Path] = RoslynParsing.Parse(ok.Text, ok.Path),
            [hostile.Path] = RoslynParsing.Parse(hostile.Text, hostile.Path),
        };
        var compilation = Microsoft.CodeAnalysis.CSharp.CSharpCompilation.Create("p", trees.Values, Bcl(),
            new Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions(Microsoft.CodeAnalysis.OutputKind.DynamicallyLinkedLibrary));
        var project = new ProjectCompilation("p", compilation, trees);
        IReadOnlyList<CallSite> sites = [];

        OnOrdinaryStack(() => sites = CallResolution.Sites([project]));

        Assert.Contains(sites, s => s.Path == "ok.cs" && s.Method is not null && s.Method.Contains("Add"));
        Assert.True(shape.Length > 0);
    }

    [Fact]
    public void SyntaxDepth_IsTheLongestParentChain_AndDoesNotRecurse()
    {
        Assert.True(RoslynParsing.SyntaxDepth(Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree.ParseText("class A { }").GetRoot()) < 10);
        string deep = Wrap("int y = " + string.Concat(Enumerable.Repeat("b ? 1 : ", 3000)) + "0;");
        Assert.True(RoslynParsing.SyntaxDepth(RoslynParsing.Parse(deep, "d.cs").GetRoot()) > 3000);
    }

    [Fact]
    public void Build_OrdinaryDeepButLegitimateFile_IsStillBound()
    {
        // 300 chained calls and a 300-term sum are far below the cap and must keep their edges.
        string body = "int y = Lib.Add(1, 2)" + string.Concat(Enumerable.Repeat(" + Lib.Add(1, 2)", 300)) + ";";
        var tok = new CharTokenizer();
        SourceFile lib = new("lib.cs", "class Lib\n{\n    public static int Add(int a, int b) => a + b;\n}\n");
        SourceFile use = new("use.cs", Wrap(body));
        SourceFile[] files = [lib, use];
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();

        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(files, chunks);

        Assert.True(graph.QueryEdges("use.cs", 0, use.Text.Length).Count(e => e.Kind == SymbolEdgeKind.Call) >= 300);
    }
}
