using System.Diagnostics;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="SolutionScanner"/> over <c>Fixtures/MiniSolution</c>: <c>Lib</c> (net10.0 and netstandard2.0; declares
/// <c>Greeter.Greet</c> only under its own <c>MINI_GREETING</c> define), <c>App</c> (references Lib, calls <c>Greeter.Greet</c>,
/// holds a hand-written <c>Generated/Stamp.g.cs</c>), <c>Shared/BuildInfo.cs</c> (compiled by both), <c>Broken</c> (its SDK
/// does not exist; the workspace still adds it, with a failure), <c>Missing</c> (listed, no file) and <c>Tool</c> (Visual
/// Basic). The fixture is copied to a temp directory, <c>Lib/Greeter.cs</c> is given CRLF line ends there, and <c>App</c>
/// (with Lib) is restored once per class.
/// </summary>
public sealed class SolutionScannerTests : IClassFixture<SolutionScannerTests.MiniSolution>
{
    private readonly MiniSolution _mini;

    public SolutionScannerTests(MiniSolution mini) => _mini = mini;

    [Fact]
    public void Scan_FixtureSolution_ReturnsAllCSharpFiles()
    {
        ScanResult scan = _mini.Scan;
        // Every C# document of the scanned projects, with paths relative to the solution directory, each once; generated files
        // under obj/ (implicit global usings, assembly info) come too, flagged.
        string[] sourceTree = scan.Files.Select(f => f.Path).Where(p => !p.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.Equal(new[] { "App/Generated/Stamp.g.cs", "App/Program.cs", "Lib/Greeter.cs", "Shared/BuildInfo.cs" }, sourceTree);
        Assert.Contains(scan.Files, f => f.Path.StartsWith("App/obj/", StringComparison.Ordinal) && f.Path.EndsWith("GlobalUsings.g.cs", StringComparison.Ordinal));
        Assert.Contains(scan.Files, f => f.Path.StartsWith("Lib/obj/", StringComparison.Ordinal) && f.Path.EndsWith("GlobalUsings.g.cs", StringComparison.Ordinal));
        // Lib's other target framework is not scanned.
        Assert.DoesNotContain(scan.Files, f => f.Path.Contains("netstandard2.0", StringComparison.Ordinal));
        Assert.DoesNotContain(scan.Files, f => f.Path.EndsWith(".vb", StringComparison.Ordinal) || f.Path.StartsWith("Broken/", StringComparison.Ordinal));

        // Ordinal path order, unique paths, text as on disk with CRLF → LF (as Corpus.FromDirectory).
        Assert.Equal(scan.Files.Select(f => f.Path).Order(StringComparer.Ordinal), scan.Files.Select(f => f.Path));
        Assert.Equal(scan.Files.Count, scan.Files.Select(f => f.Path).Distinct(StringComparer.Ordinal).Count());
        foreach (SourceFile f in scan.Files)
        {
            string disk = File.ReadAllText(Path.Combine(_mini.Root, f.Path)).Replace("\r\n", "\n", StringComparison.Ordinal);
            Assert.Equal(disk, f.Text);
        }
        Assert.Contains("\r\n", File.ReadAllText(Path.Combine(_mini.Root, "Lib", "Greeter.cs")), StringComparison.Ordinal);

        // Scanned projects (ordinal by project path); a file compiled by two projects belongs to the first.
        Assert.Equal(new[] { "App/App.csproj", "Lib/Lib.csproj" }, scan.Projects.Select(p => p.ProjectPath));
        Assert.Equal(new[] { "App", "Lib(net10.0)" }, scan.Projects.Select(p => p.Name));
        Assert.Equal(new[] { "App/Generated/Stamp.g.cs", "App/Program.cs", "Shared/BuildInfo.cs" },
            scan.Projects[0].Files.Where(p => !p.Contains("/obj/", StringComparison.Ordinal)));
        Assert.Equal(new[] { "Lib/Greeter.cs" }, scan.Projects[1].Files.Where(p => !p.Contains("/obj/", StringComparison.Ordinal)));
        Assert.Equal(scan.Files.Select(f => f.Path), scan.Projects.SelectMany(p => p.Files).Order(StringComparer.Ordinal));
        Assert.Equal(_mini.Root, scan.RootDirectory);
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
        Assert.Equal(SkippedProject.LoadFailed, missing.Reason);
        Assert.Contains("Missing.csproj", missing.Message, StringComparison.Ordinal);
        Assert.Equal(SkippedProject.UnsupportedLanguage, tool.Reason);
        Assert.Contains("Visual Basic", tool.Message, StringComparison.Ordinal);
        // Failures are attributed to their projects, not left over; the scanned projects loaded cleanly.
        Assert.Empty(scan.Diagnostics);
        Assert.All(scan.Projects, p => Assert.Empty(p.Diagnostics));
        Assert.DoesNotContain(scan.Projects, p => p.ProjectPath.StartsWith("Broken/", StringComparison.Ordinal) || p.ProjectPath.StartsWith("Tool/", StringComparison.Ordinal));
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
        Assert.Single(scan.Files, f => f.Path == "../Shared/BuildInfo.cs");
        Assert.Equal(SkippedProject.ExtraTargetFramework, Assert.Single(scan.Skipped).Reason);

        ScanResult broken = await SolutionScanner.LoadAsync(Path.Combine(_mini.Root, "Broken", "Broken.csproj"));
        Assert.Empty(broken.Files);
        Assert.Empty(broken.Projects);
        SkippedProject skipped = Assert.Single(broken.Skipped);
        Assert.Equal(("Broken.csproj", SkippedProject.LoadFailed), (skipped.ProjectPath, skipped.Reason));
        Assert.Contains("DotLLM.Missing.Sdk", skipped.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Scan_GeneratedFiles_AreFlagged()
    {
        ScanResult scan = _mini.Scan;
        Assert.True(scan.IsGenerated("App/Generated/Stamp.g.cs"));
        Assert.False(scan.IsGenerated("App/Program.cs"));
        Assert.False(scan.IsGenerated("Lib/Greeter.cs"));
        // Every file under obj/ is generated (implicit usings, assembly info and attributes), whatever its header says.
        string[] obj = scan.Files.Select(f => f.Path).Where(p => p.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.True(obj.Length >= 4, string.Join(", ", obj));
        Assert.All(obj, p => Assert.True(scan.IsGenerated(p), p));
        Assert.Equal(scan.Files.Select(f => f.Path).Where(scan.IsGenerated).Order(StringComparer.Ordinal), scan.Generated.Order(StringComparer.Ordinal));
        // Authored files are the rest, in order.
        Assert.Equal(new[] { "App/Program.cs", "Lib/Greeter.cs", "Shared/BuildInfo.cs" }, scan.AuthoredFiles.Select(f => f.Path));
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
    public void Graph_FromCompilations_RejectsAFileOwnedTwice()
    {
        ScanResult scan = _mini.Scan;
        ITokenizer tok = new CharTokenizer();
        Chunk[] chunks = scan.AuthoredFiles.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ProjectCompilation app = scan.Compilations.Single(c => c.Name == "App"), lib = scan.Compilations.Single(c => c.Name == "Lib(net10.0)");
        Assert.Contains("more than one project", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app, lib, app], chunks)).Message, StringComparison.Ordinal);
        // A tree must belong to its project's compilation.
        Assert.Contains("is not in project", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app with { Compilation = lib.Compilation }, lib], chunks)).Message, StringComparison.Ordinal);
        // Every chunked file must be owned by a project.
        Assert.Contains("is not among the files", Assert.Throws<ArgumentException>(() => new RoslynGraphBuilder(tok).Build([app], chunks)).Message, StringComparison.Ordinal);
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
