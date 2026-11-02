using System.Diagnostics;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.SolutionScan;
using DotLLM.Tokenizers;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="SolutionScanner"/> over <c>Fixtures/MiniSolution</c>:
/// <list type="bullet">
/// <item><c>Lib</c>: netstandard2.0 and net10.0; declares <c>Greeter.Greet</c> only under its own <c>MINI_GREETING</c> define;
/// <c>Polyfill.cs</c> only for netstandard2.0.</item>
/// <item><c>App</c>: references Lib; calls <c>Greeter.Greet</c> (and once with a wrong argument type); a hand-written
/// <c>Generated/Stamp.g.cs</c>, a <c>[GeneratedCode]</c> class (<c>Tooling.cs</c>) and an authored file that holds the marker
/// strings in literals and a later comment (<c>Heuristics.cs</c>).</item>
/// <item><c>Shared/BuildInfo.cs</c>: compiled by both (App links it in lower case on Windows).</item>
/// <item><c>Broken</c> (missing SDK; the workspace still adds it, with a failure), <c>UsesBroken</c> (references Broken; not
/// restored), <c>Missing</c> (listed, no file), <c>Tool</c> (Visual Basic).</item>
/// </list>
/// The fixture is copied to a temp directory, <c>Lib/Greeter.cs</c> is given CRLF line ends there, and <c>App</c> (with Lib)
/// is restored once per class.
/// </summary>
public sealed class SolutionScannerTests : IClassFixture<SolutionScannerTests.MiniSolution>
{
    private static readonly string[] Authored =
        ["App/Heuristics.cs", "App/Program.cs", "Lib/Greeter.cs", "Lib/LibInfo.cs", "Shared/BuildInfo.cs", "UsesBroken/Client.cs"];

    private readonly MiniSolution _mini;

    public SolutionScannerTests(MiniSolution mini) => _mini = mini;

    [Fact]
    public void Scan_FixtureSolution_ReturnsAllCSharpFiles()
    {
        ScanResult scan = _mini.Scan;
        // Every C# document of the scanned projects, with paths relative to the solution directory, each once (one casing);
        // generated files under obj/ (implicit global usings, assembly info) come too, flagged.
        string[] sourceTree = scan.Files.Select(f => f.Path).Where(p => !p.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.Equal(new[] { "App/Generated/Stamp.g.cs", "App/Heuristics.cs", "App/Program.cs", "App/Tooling.cs", "Lib/Greeter.cs",
            "Lib/LibInfo.cs", "Shared/BuildInfo.cs", "UsesBroken/Client.cs" }, sourceTree);
        Assert.Contains(scan.Files, f => f.Path.StartsWith("App/obj/", StringComparison.Ordinal) && f.Path.EndsWith("GlobalUsings.g.cs", StringComparison.Ordinal));
        Assert.Contains(scan.Files, f => f.Path.StartsWith("Lib/obj/", StringComparison.Ordinal) && f.Path.EndsWith("GlobalUsings.g.cs", StringComparison.Ordinal));
        // Lib's other target framework is not scanned (its obj/ files nor its own Polyfill.cs).
        Assert.DoesNotContain(scan.Files, f => f.Path.Contains("netstandard2.0", StringComparison.Ordinal) || f.Path == "Lib/Polyfill.cs");
        Assert.DoesNotContain(scan.Files, f => f.Path.EndsWith(".vb", StringComparison.Ordinal) || f.Path.StartsWith("Broken/", StringComparison.Ordinal));

        // Ordinal path order, unique paths (also ignoring case), text as on disk with CRLF → LF (as Corpus.FromDirectory).
        Assert.Equal(scan.Files.Select(f => f.Path).Order(StringComparer.Ordinal), scan.Files.Select(f => f.Path));
        Assert.Equal(scan.Files.Count, scan.Files.Select(f => f.Path).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        foreach (SourceFile f in scan.Files)
        {
            string disk = File.ReadAllText(Path.Combine(_mini.Root, f.Path)).Replace("\r\n", "\n", StringComparison.Ordinal);
            Assert.Equal(disk, f.Text);
        }
        Assert.Contains("\r\n", File.ReadAllText(Path.Combine(_mini.Root, "Lib", "Greeter.cs")), StringComparison.Ordinal);

        // Scanned projects (ordinal by project path). A file compiled by two projects belongs to the one whose directory holds
        // it (Lib/LibInfo.cs, also linked by App: Lib), else to the first by path (Shared/BuildInfo.cs: App).
        Assert.Equal(new[] { "App/App.csproj", "Lib/Lib.csproj", "UsesBroken/UsesBroken.csproj" }, scan.Projects.Select(p => p.ProjectPath));
        Assert.Equal(new[] { "App", "Lib(net10.0)", "UsesBroken" }, scan.Projects.Select(p => p.Name));
        Assert.Equal(new[] { null, "net10.0", null }, scan.Projects.Select(p => p.TargetFramework));
        Assert.Equal(new[] { "App/Generated/Stamp.g.cs", "App/Heuristics.cs", "App/Program.cs", "App/Tooling.cs", "Shared/BuildInfo.cs" },
            scan.Projects[0].Files.Where(p => !p.Contains("/obj/", StringComparison.Ordinal)));
        Assert.Equal(new[] { "Lib/Greeter.cs", "Lib/LibInfo.cs" }, scan.Projects[1].Files.Where(p => !p.Contains("/obj/", StringComparison.Ordinal)));
        Assert.Equal(scan.Files.Select(f => f.Path), scan.Projects.SelectMany(p => p.Files).Order(StringComparer.Ordinal));
        Assert.Equal(_mini.Root, scan.RootDirectory);
        // Project references (the dependency graph).
        Assert.Equal(new[] { "Lib/Lib.csproj" }, scan.Projects[0].ProjectReferences);
        Assert.Empty(scan.Projects[1].ProjectReferences);
        // Cached views.
        Assert.Same(scan.AuthoredFiles, scan.AuthoredFiles);
        Assert.Same(scan.Compilations, scan.Compilations);
    }

    [Fact]
    public void Scan_LinkedFileWithOtherCasing_IsOneFile()
    {
        // On Windows App links "..\shared\buildinfo.cs" and Lib "..\Shared\BuildInfo.cs": one file, on-disk casing.
        Assert.True(_mini.CaseVariantLink == OperatingSystem.IsWindows());
        ScanResult scan = _mini.Scan;
        Assert.Single(scan.Files, f => f.Path.Equals("Shared/BuildInfo.cs", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(scan.Files, f => f.Path == "Shared/BuildInfo.cs");
        // Lib's use of BuildInfo.Major binds in Lib's compilation, whose tree of the file has Lib's spelling; the owner (App)
        // has the other spelling. The location still maps to the owner's corpus path: the edge exists.
        ITokenizer tok = new CharTokenizer();
        Chunk[] chunks = scan.AuthoredFiles.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(scan.Compilations, chunks);
        SourceFile lib = scan.AuthoredFiles.Single(f => f.Path == "Lib/Greeter.cs"), info = scan.AuthoredFiles.Single(f => f.Path == "Shared/BuildInfo.cs");
        Assert.Contains(Edge(chunks, lib, Off(lib.Text, "Major +"), info, Off(info.Text, "Major ="), SymbolEdgeKind.Use), graph.Edges);
    }

    [Fact]
    public void Scan_ProjectFailsToLoad_IsReportedInSkipped()
    {
        ScanResult scan = _mini.Scan;
        Assert.Equal(new[] { "Broken/Broken.csproj", "Lib/Lib.csproj", "Missing/Missing.csproj", "Tool/Tool.vbproj" },
            scan.Skipped.Select(s => s.ProjectPath));
        SkippedProject broken = scan.Skipped[0], libExtra = scan.Skipped[1], missing = scan.Skipped[2], tool = scan.Skipped[3];
        Assert.Equal(SkippedProject.LoadFailed, broken.Reason);
        Assert.Contains("DotLLM.Missing.Sdk", broken.Message, StringComparison.Ordinal);
        Assert.Equal(SkippedProject.ExtraTargetFramework, libExtra.Reason);
        Assert.Contains("Lib(netstandard2.0)", libExtra.Message, StringComparison.Ordinal);
        // The files only the other target compiles are listed, not silently dropped.
        Assert.Equal(new[] { "Lib/Polyfill.cs" }, libExtra.Files);
        Assert.Contains("Lib/Polyfill.cs", libExtra.Message, StringComparison.Ordinal);
        Assert.Equal(SkippedProject.LoadFailed, missing.Reason);
        Assert.Contains("Missing.csproj", missing.Message, StringComparison.Ordinal);
        Assert.Equal(SkippedProject.UnsupportedLanguage, tool.Reason);
        Assert.Contains("Visual Basic", tool.Message, StringComparison.Ordinal);
        Assert.All(scan.Skipped.Where(s => s.Reason != SkippedProject.ExtraTargetFramework), s => Assert.Empty(s.Files));

        // UsesBroken references Broken. Loading the reference repeats Broken's SDK failure with an empty path ("processing the
        // file ''"), which names no project: it is reported top-level and flagged on every scanned project, never dropped.
        // UsesBroken itself is scanned.
        Assert.Contains(scan.Projects, p => p.ProjectPath == "UsesBroken/UsesBroken.csproj");
        string[] unattributedFailures = scan.Diagnostics.Where(d => d.StartsWith("Failure", StringComparison.Ordinal)).Select(d => "Unattributed failure: " + d).ToArray();
        Assert.NotEmpty(unattributedFailures);
        Assert.All(unattributedFailures, d => Assert.Contains("processing the file ''", d, StringComparison.Ordinal));
        Assert.All(scan.Projects, p => Assert.Equal(unattributedFailures, p.Diagnostics.Where(d => d.StartsWith("Unattributed", StringComparison.Ordinal))));
        // Attributed failures stay on their projects: no other failure on the scanned projects, and App and Lib have nothing else.
        Assert.All(scan.Projects, p => Assert.DoesNotContain(p.Diagnostics, d => d.StartsWith("Failure", StringComparison.Ordinal)));
        Assert.All(scan.Projects.Where(p => p.ProjectPath != "UsesBroken/UsesBroken.csproj"), p => Assert.Equal(unattributedFailures, p.Diagnostics));
        Assert.DoesNotContain(scan.Projects, p => p.ProjectPath.StartsWith("Broken/", StringComparison.Ordinal) || p.ProjectPath.StartsWith("Tool/", StringComparison.Ordinal));
    }

    [Fact]
    public void AttributeDiagnostics_ByQuotedProjectPath_NotBySubstring()
    {
        string root = Path.Combine(Path.GetTempPath(), "attr-sln");
        string a = Path.Combine(root, "A", "A.csproj"), b = Path.Combine(root, "B", "B.csproj");
        // Roslyn's spelling may differ from the solution's: case (Windows only) and separators.
        string aOther = (OperatingSystem.IsWindows() ? a.ToUpperInvariant() : a).Replace('\\', '/');
        string[] messages =
        [
            $"Msbuild failed when processing the file '{aOther}' with message: the project '{b}' could not be found.", // → A
            $"Found project reference without a matching metadata reference: {b}",                                   // unquoted → none
            $"Could not read '{b}.bak'.",                                                                              // not a project → none
            $"Error in '{b}': bad element.",                                                                            // → B
            $"Msbuild failed when processing the file '{Path.Combine(root, "X", "X.csproj")}' with message: x",         // unknown → none
            $"Projects '{a}' and '{b}' disagree.",                                                                      // two → none
            "Msbuild failed when processing the file '' with message: could not resolve SDK.",                         // empty path → none
        ];
        (Dictionary<string, List<string>> byProject, List<string> unattributed) =
            SolutionScanner.AttributeDiagnostics(messages, [a, b], m => m, p => p);
        Assert.Equal(new[] { messages[0] }, byProject[a]);
        Assert.Equal(new[] { messages[3] }, byProject[b]);
        Assert.Equal(new[] { messages[1], messages[2], messages[4], messages[5], messages[6] }, unattributed);
    }

    [Fact]
    public void TargetFramework_HighestDotNetPreferred_NotOrdinal()
    {
        string[] names = ["X(netstandard2.0)", "X(net48)", "X(net8.0)", "X(net10.0-windows)", "X(netcoreapp3.1)", "X(net10.0)", "X(net9.0)"];
        string[] preferredFirst = names.OrderByDescending(n => n, Comparer<string>.Create(SolutionScanner.TargetFrameworkPreference.CompareNames)).ToArray();
        Assert.Equal(new[] { "X(net10.0)", "X(net10.0-windows)", "X(net9.0)", "X(net8.0)", "X(netcoreapp3.1)", "X(netstandard2.0)", "X(net48)" }, preferredFirst);
        Assert.Equal("net10.0", SolutionScanner.TargetFrameworkPreference.FromName("Lib(net10.0)"));
        Assert.Null(SolutionScanner.TargetFrameworkPreference.FromName("Lib"));
    }

    [Fact]
    public async Task Scan_SingleProject_LoadsItAndItsReferences()
    {
        ScanResult scan = await SolutionScanner.LoadAsync(Path.Combine(_mini.Root, "App", "App.csproj"));
        // Paths are relative to the project's directory; the referenced Lib lies outside it.
        Assert.Equal(Path.Combine(_mini.Root, "App"), scan.RootDirectory);
        Assert.Equal(new[] { "../Lib/Lib.csproj", "App.csproj" }, scan.Projects.Select(p => p.ProjectPath));
        Assert.Contains(scan.Files, f => f.Path == "Program.cs");
        Assert.Contains(scan.Files, f => f.Path == "../Lib/Greeter.cs");
        Assert.Single(scan.Files, f => f.Path.Equals("../Shared/BuildInfo.cs", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(SkippedProject.ExtraTargetFramework, Assert.Single(scan.Skipped).Reason);

        ScanResult broken = await SolutionScanner.LoadAsync(Path.Combine(_mini.Root, "Broken", "Broken.csproj"));
        Assert.Empty(broken.Files);
        Assert.Empty(broken.Projects);
        SkippedProject skipped = Assert.Single(broken.Skipped);
        Assert.Equal(("Broken.csproj", SkippedProject.LoadFailed), (skipped.ProjectPath, skipped.Reason));
        Assert.Contains("DotLLM.Missing.Sdk", skipped.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Scan_Cancelled_Throws()
    {
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => SolutionScanner.LoadAsync(Path.Combine(_mini.Root, "MiniSolution.slnx"), cts.Token));
    }

    [Fact]
    public void Scan_GeneratedFiles_AreFlagged()
    {
        ScanResult scan = _mini.Scan;
        Assert.True(scan.IsGenerated("App/Generated/Stamp.g.cs"));   // file name
        Assert.True(scan.IsGenerated("App/Tooling.cs"));             // [GeneratedCode] attribute
        Assert.False(scan.IsGenerated("App/Heuristics.cs"));         // marker strings in literals and a later comment only
        Assert.False(scan.IsGenerated("App/Program.cs"));
        Assert.False(scan.IsGenerated("Lib/Greeter.cs"));
        // Every file under obj/ is generated (implicit usings, assembly info and attributes), whatever its header says.
        string[] obj = scan.Files.Select(f => f.Path).Where(p => p.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.True(obj.Length >= 4, string.Join(", ", obj));
        Assert.All(obj, p => Assert.True(scan.IsGenerated(p), p));
        Assert.Equal(scan.Files.Select(f => f.Path).Where(scan.IsGenerated).Order(StringComparer.Ordinal), scan.Generated.Order(StringComparer.Ordinal));
        // Authored files are the rest, in order.
        Assert.Equal(Authored, scan.AuthoredFiles.Select(f => f.Path));
    }

    [Theory]
    [InlineData("A.cs", "// <auto-generated/>\nclass A { }", true)]
    [InlineData("A.cs", "//------\n// <autogenerated />\n//------\nusing System;\nclass A { }", true)]
    [InlineData("A.cs", "/* <auto-generated> */ class A { }", true)]
    [InlineData("A.cs", "[System.CodeDom.Compiler.GeneratedCodeAttribute(\"t\", \"1\")]\nclass A { }", true)]
    [InlineData("A.cs", "class A { [GeneratedCode(\"t\", \"1\")] void M() { } }", true)]
    [InlineData("A.cs", "class A { const string S = \"[GeneratedCode(\\\"t\\\")]\"; }", false)]
    [InlineData("A.cs", "class A { const string S = \"<auto-generated>\"; }", false)]
    [InlineData("A.cs", "using System;\n// <auto-generated/> mentioned below the first token\nclass A { }", false)]
    [InlineData("A.cs", "class A { void M() { var r = new System.Text.RegularExpressions.Regex(@\"\\[GeneratedCode\\(\"); } }", false)]
    [InlineData("A.g.cs", "class A { }", true)]
    [InlineData("src/obj/Debug/A.cs", "class A { }", true)]
    [InlineData("src/Objects/A.cs", "class A { }", false)]
    public void GeneratedFiles_HeaderAndAttributeSyntax_NotRawText(string path, string text, bool generated)
    {
        var tree = CSharpSyntaxTree.ParseText(text, SymbolChunker.ParseOptions, path);
        Assert.Equal(generated, GeneratedFiles.IsGenerated(path, tree));
    }

    [Fact]
    public void Graph_WithProjectReferences_ResolvesCrossProjectCall()
    {
        ScanResult scan = _mini.Scan;
        ITokenizer tok = new CharTokenizer();
        var chunker = new SymbolChunker(tok, 4096);
        IReadOnlyList<SourceFile> authored = scan.AuthoredFiles;
        Chunk[] chunks = authored.SelectMany(chunker.Chunk).ToArray();
        SourceFile app = authored.Single(f => f.Path == "App/Program.cs"), lib = authored.Single(f => f.Path == "Lib/Greeter.cs");
        ChunkEdge greetCall = Edge(chunks, app, Off(app.Text, "Greet("), lib, Off(lib.Text, "Greet("), SymbolEdgeKind.Call);
        ChunkEdge greeterUse = Edge(chunks, app, Off(app.Text, "Greeter."), lib, Off(lib.Text, "Greeter\n"), SymbolEdgeKind.Use);

        // Per-project compilations (Lib's define, App's project reference): the call resolves across projects.
        ReferenceGraph projects = new RoslynGraphBuilder(tok).Build(scan.Compilations, chunks);
        Assert.Contains(greetCall, projects.Edges);
        Assert.Contains(greeterUse, projects.Edges);
        // The call with a wrong argument type binds to no symbol (a candidate only): no edge from its name.
        int wrongUse = Off(app.Text, "Greet(1)");
        int wrongChunk = ChunkAt(chunks, app.Path, wrongUse);
        Assert.DoesNotContain(projects.Edges, e => e.UseChunk == wrongChunk && e.UseToken == wrongUse - chunks[wrongChunk].StartOffset);
        // Import: `using Mini.Lib;` (use end `Mini`) → the last name of `namespace Mini.Lib;` in Lib.
        Assert.Contains(Edge(chunks, app, Off(app.Text, "Mini.Lib;"), lib, Off(lib.Text, "Lib;"), SymbolEdgeKind.Import), projects.Edges);
        // Stamp lives in a generated (unchunked) file: no edge end there, and nothing points outside the chunks.
        Assert.All(projects.Edges, e => Assert.NotEqual(e.UseChunk, e.DefinitionChunk));

        // BCL-only (one compilation over the same authored files, no defines): the type resolves, the call does not.
        ReferenceGraph bclOnly = new RoslynGraphBuilder(tok).Build(authored, chunks);
        Assert.Contains(greeterUse, bclOnly.Edges);
        Assert.DoesNotContain(greetCall, bclOnly.Edges);
        Assert.DoesNotContain(bclOnly.Edges, e => e.Kind == SymbolEdgeKind.Call);
    }

    [Fact]
    public void CallResolution_ProjectCompilations_BindTheCallBclOnlyCannot()
    {
        ScanResult scan = _mini.Scan;
        var authored = scan.AuthoredFiles.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);
        // Two invocations in the authored files, both in App/Program.cs: Greeter.Greet(Stamp.Value), declared in Lib/Greeter.cs,
        // and Greeter.Greet(1), an overload-resolution failure (a candidate, not a symbol: unresolved).
        Assert.Equal(new CallResolutionStats(2, 1, 1), CallResolution.Measure(scan.Compilations, authored));
        Assert.Equal(new CallResolutionStats(2, 0, 0), CallResolution.Measure([ProjectCompilation.BclOnly(scan.AuthoredFiles)], authored));
        // Restricted to Lib's file: no invocation; the declaring file must be among the measured ones to count as "to file".
        Assert.Equal(new CallResolutionStats(0, 0, 0), CallResolution.Measure(scan.Compilations, new HashSet<string>(["Lib/Greeter.cs"], StringComparer.Ordinal)));
        Assert.Equal(new CallResolutionStats(2, 1, 0), CallResolution.Measure(scan.Compilations, new HashSet<string>(["App/Program.cs"], StringComparer.Ordinal)));

        IReadOnlyList<CallSite> sites = CallResolution.Sites(scan.Compilations, authored);
        SourceFile app = scan.AuthoredFiles.Single(f => f.Path == "App/Program.cs"), lib = scan.AuthoredFiles.Single(f => f.Path == "Lib/Greeter.cs");
        Assert.Equal(new[]
        {
            new CallSite("App/Program.cs", Off(app.Text, "Greeter.Greet(Stamp"), "Greeter.Greet(Stamp.Value)".Length, CallTargetKind.File, "Mini.Lib.Greeter.Greet(string)", "Lib/Greeter.cs", Off(lib.Text, "Greet(")),
            new CallSite("App/Program.cs", Off(app.Text, "Greeter.Greet(1)"), "Greeter.Greet(1)".Length, CallTargetKind.Unresolved, null, null, -1),
        }, sites);
    }

    [Fact]
    public void Graph_FromCompilations_RejectsAFileOwnedTwice()
    {
        ScanResult scan = _mini.Scan;
        ITokenizer tok = new CharTokenizer();
        Chunk[] chunks = scan.AuthoredFiles.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ProjectCompilation app = scan.Compilations.Single(c => c.Name == "App"), lib = scan.Compilations.Single(c => c.Name == "Lib(net10.0)"),
            uses = scan.Compilations.Single(c => c.Name == "UsesBroken");
        Assert.Contains("more than one project", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app, lib, uses, app], chunks)).Message, StringComparison.Ordinal);
        // A tree must belong to its project's compilation.
        Assert.Contains("is not in project", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app with { Compilation = lib.Compilation }, lib, uses], chunks)).Message, StringComparison.Ordinal);
        // Every chunked file must be owned by a project.
        Assert.Contains("is not among the files", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app, uses], chunks)).Message, StringComparison.Ordinal);

        // Lib's compilation also holds a tree of Shared/BuildInfo.cs: claimed under another spelling, it is still the same file
        // on Windows (case-insensitive), and two files elsewhere.
        var libBuildInfo = lib.Compilation.SyntaxTrees.Single(t => t.FilePath.EndsWith("BuildInfo.cs", StringComparison.OrdinalIgnoreCase));
        var libClaims = lib with { Trees = new Dictionary<string, Microsoft.CodeAnalysis.SyntaxTree>(lib.Trees) { ["shared/buildinfo.cs"] = libBuildInfo } };
        if (OperatingSystem.IsWindows())
            Assert.Contains("more than one project", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app, libClaims, uses], chunks)).Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>The graph edge from char <paramref name="useOffset"/> of <paramref name="use"/> to <paramref name="defOffset"/> of <paramref name="def"/> (one token per char).</summary>
    private static ChunkEdge Edge(Chunk[] chunks, SourceFile use, int useOffset, SourceFile def, int defOffset, SymbolEdgeKind kind)
    {
        int uc = ChunkAt(chunks, use.Path, useOffset), dc = ChunkAt(chunks, def.Path, defOffset);
        return new ChunkEdge(uc, useOffset - chunks[uc].StartOffset, dc, defOffset - chunks[dc].StartOffset, kind);
    }

    private static int ChunkAt(Chunk[] chunks, string path, int offset) =>
        Array.FindIndex(chunks, c => c.FilePath == path && offset >= c.StartOffset && offset < c.EndOffset);

    private static int Off(string text, string needle)
    {
        int at = text.IndexOf(needle, StringComparison.Ordinal);
        Assert.True(at >= 0, $"'{needle}' not found");
        return at;
    }

    /// <summary>Temp copy of the fixture with App (and Lib) restored, scanned once.</summary>
    public sealed class MiniSolution : IAsyncLifetime
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(), "dotllm-minisolution-" + Guid.NewGuid().ToString("N")[..12]);

        public ScanResult Scan { get; private set; } = null!;

        /// <summary>True when App's link to Shared/BuildInfo.cs was rewritten in lower case (Windows only).</summary>
        public bool CaseVariantLink { get; private set; }

        public async Task InitializeAsync()
        {
            string source = Path.Combine(AppContext.BaseDirectory, "ChunkReuse", "Fixtures", "MiniSolution");
            foreach (string file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
            {
                string target = Path.Combine(Root, Path.GetRelativePath(source, file));
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                File.Copy(file, target);
            }
            // One file with CRLF line ends (whatever git checked out): scanned text and tree offsets must be LF.
            string greeter = Path.Combine(Root, "Lib", "Greeter.cs");
            File.WriteAllText(greeter, File.ReadAllText(greeter).Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\n", "\r\n", StringComparison.Ordinal));
            if (OperatingSystem.IsWindows())
            {
                string appProject = Path.Combine(Root, "App", "App.csproj");
                string text = File.ReadAllText(appProject);
                File.WriteAllText(appProject, text.Replace(@"..\Shared\BuildInfo.cs", @"..\shared\buildinfo.cs", StringComparison.Ordinal));
                CaseVariantLink = !string.Equals(text, File.ReadAllText(appProject), StringComparison.Ordinal);
            }
            await RunDotnetAsync("restore", Path.Combine("App", "App.csproj"), "--nologo", "-v", "q");
            Scan = await SolutionScanner.LoadAsync(Path.Combine(Root, "MiniSolution.slnx"));
        }

        public Task DisposeAsync()
        {
            try { Directory.Delete(Root, recursive: true); }
            catch (IOException) { /* best effort: a build host may still hold a file briefly */ }
            catch (UnauthorizedAccessException) { }
            return Task.CompletedTask;
        }

        private async Task RunDotnetAsync(params string[] args)
        {
            var psi = new ProcessStartInfo("dotnet") { WorkingDirectory = Root, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
            foreach (string a in args) psi.ArgumentList.Add(a);
            using Process p = Process.Start(psi)!;
            Task<string> stdout = p.StandardOutput.ReadToEndAsync(), stderr = p.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            await p.WaitForExitAsync(timeout.Token);
            if (p.ExitCode != 0)
                throw new InvalidOperationException($"dotnet {string.Join(' ', args)} exited {p.ExitCode}: {await stdout}\n{await stderr}");
        }
    }
}
