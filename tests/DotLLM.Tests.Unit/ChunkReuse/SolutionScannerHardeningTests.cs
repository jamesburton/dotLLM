using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.SolutionScan;
using Microsoft.CodeAnalysis;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="SolutionScanner"/> on hostile input. Each test writes a one-project solution to a temp directory, restores it and scans
/// it through the real MSBuildWorkspace: a file nested deeper than Roslyn's parser survives (the process dies on a stack overflow,
/// which no catch stops), a file too large to parse, a project that lists a non-C# file as a <c>Compile</c> item (its contents would
/// be returned as source), and a design-time build that runs longer than the caller allows (the token must end the call).
/// </summary>
public sealed class SolutionScannerHardeningTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-scanhard-" + Guid.NewGuid().ToString("N")[..12]);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort: a build host may still hold a file briefly */ }
        catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public async Task Scan_FileCompiledByANestedProject_BelongsToTheDeepestProject_WhateverTheOrder()
    {
        // The outer project's default globs take the inner project's files too. The inner project's directory holds the file, so it owns
        // it. Projects are visited in ordinal path order: Outer1 before its inner project, and Inner2 before its outer one.
        const string noGlobCollision = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework>"
            + "<EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include=\"**/*.cs\" Exclude=\"**/obj/**\" /></ItemGroup></Project>";
        string[] projects =
        [
            WriteAt("A", "A.csproj", noGlobCollision, ("OuterA.cs", "class OuterA { }\n")),
            WriteAt("A/Z", "Z.csproj", noGlobCollision, ("InnerZ.cs", "class InnerZ { }\n")),
            WriteAt("B/A", "Inner.csproj", noGlobCollision, ("InnerA.cs", "class InnerA { }\n")),
            WriteAt("B", "B.csproj", noGlobCollision, ("OuterB.cs", "class OuterB { }\n")),
        ];
        foreach (string p in projects) await DotnetAsync(Path.GetDirectoryName(p)!, "restore", p);
        string slnx = Path.Combine(_root, "S.slnx");
        File.WriteAllText(slnx, "<Solution><Project Path=\"A/A.csproj\" /><Project Path=\"A/Z/Z.csproj\" /><Project Path=\"B/A/Inner.csproj\" /><Project Path=\"B/B.csproj\" /></Solution>");

        ScanResult scan = await SolutionScanner.LoadAsync(slnx);

        string[] Files(string project) => scan.Projects.Single(p => p.ProjectPath == project).Files.Where(f => !f.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.Equal(new[] { "A/OuterA.cs" }, Files("A/A.csproj"));
        Assert.Equal(new[] { "A/Z/InnerZ.cs" }, Files("A/Z/Z.csproj"));
        Assert.Equal(new[] { "B/A/InnerA.cs" }, Files("B/A/Inner.csproj"));
        Assert.Equal(new[] { "B/OuterB.cs" }, Files("B/B.csproj"));
    }

    [Fact]
    public async Task Scan_SingleProject_ReferencedProjectIsScannedAsSourceEvenWhenItWasBuilt()
    {
        // A built referenced project has an output assembly; the scanner must still load it as a project (its files are wanted), not as metadata.
        WriteAt("Lib", "Lib.csproj", Plain, ("L.cs", "namespace Lib { public class L { public static int V => 1; } }\n"));
        string app = WriteAt("App", "App.csproj",
            "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>"
            + "<ItemGroup><ProjectReference Include=\"..\\Lib\\Lib.csproj\" /></ItemGroup></Project>", ("A.cs", "class A { int M() => Lib.L.V; }\n"));
        await DotnetAsync(Path.GetDirectoryName(app)!, "build", app);

        ScanResult scan = await SolutionScanner.LoadAsync(app);

        Assert.Equal(new[] { "App.csproj", "Lib/Lib.csproj" }, scan.Projects.Select(p => p.ProjectPath.Replace("../", "", StringComparison.Ordinal)).Order(StringComparer.Ordinal));
        Assert.Contains(scan.Files, f => f.Path.EndsWith("L.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Scan_ProjectFileThePathLoaderRejects_IsSkippedWithTheLoadersMessage()
    {
        // The workspace throws (rather than recording a diagnostic) for a file whose extension names no language: the exception text is the reason.
        string path = WriteAt("P", "Thing.txt", "<Project />");

        ScanResult scan = await SolutionScanner.LoadAsync(path);

        Assert.Empty(scan.Projects);
        SkippedProject skipped = Assert.Single(scan.Skipped);
        Assert.Equal("Thing.txt", skipped.ProjectPath);
        Assert.Contains("not associated with a language", skipped.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Options_DefaultTimeout_IsTheCorpusBudget() =>
        Assert.Equal(RoslynParsing.DefaultCorpusBudget, new SolutionScanOptions().Timeout);

    private const string Plain = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup></Project>";

    /// <summary>Writes <c>P/P.csproj</c> (the given text) and the files under <c>P</c>, restores it; returns the project path.</summary>
    private async Task<string> WriteProjectAsync(string project, params (string Name, string Text)[] files)
    {
        string csproj = WriteAt("P", "P.csproj", project, files);
        await DotnetAsync(Path.GetDirectoryName(csproj)!, "restore", csproj);
        return csproj;
    }

    /// <summary>Writes <paramref name="projectName"/> and <paramref name="files"/> under <paramref name="relDir"/> of the temp root; returns the project path.</summary>
    private string WriteAt(string relDir, string projectName, string project, params (string Name, string Text)[] files)
    {
        string dir = Path.Combine(_root, relDir);
        Directory.CreateDirectory(dir);
        foreach ((string name, string text) in files)
            File.WriteAllText(Path.Combine(dir, name), text);
        string csproj = Path.Combine(dir, projectName);
        File.WriteAllText(csproj, project);
        return csproj;
    }

    private static async Task DotnetAsync(string workingDirectory, params string[] args)
    {
        var psi = new ProcessStartInfo("dotnet") { WorkingDirectory = workingDirectory, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string a in args) psi.ArgumentList.Add(a);
        foreach (string a in new[] { "--nologo", "-v", "q" }) psi.ArgumentList.Add(a);
        using Process p = Process.Start(psi)!;
        Task<string> stdout = p.StandardOutput.ReadToEndAsync(), stderr = p.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
        try { await p.WaitForExitAsync(timeout.Token); }
        catch (OperationCanceledException)
        {
            p.Kill(entireProcessTree: true);
            await Task.WhenAll(stdout, stderr);
            throw;
        }
        string output = await stdout + await stderr;
        Assert.True(p.ExitCode == 0, $"dotnet {string.Join(' ', args)} exited {p.ExitCode}: {output}");
    }

    private static string Nested(string name, int depth) =>
        "class " + name + " { int M() { return " + new string('(', depth) + "1" + new string(')', depth) + "; } }\n";

    [Fact]
    public async Task Scan_FileNestedFarBeyondRoslynsStack_IsLeftOutOfTheCompilationAndTheProcessSurvives()
    {
        // 100,000 nested parentheses overflow the stack of whichever thread parses the file ("Stack overflow." and the test host dies);
        // 100 are ordinary code. The deep file is still a scanned file (its text is returned, a corpus chunks it by lines) but its tree is empty.
        string deep = Nested("Deep", 100_000), ok = Nested("Ok", 100);
        string csproj = await WriteProjectAsync(Plain, ("Deep.cs", deep), ("Ok.cs", ok), ("A.cs", "class A { }\n"));

        ScanResult scan = await SolutionScanner.LoadAsync(csproj);

        ScannedProject project = Assert.Single(scan.Projects);
        Assert.Equal(deep, scan.Files.Single(f => f.Path == "Deep.cs").Text);
        Assert.Equal(0, project.Compilation.Trees["Deep.cs"].GetRoot().FullSpan.Length);
        Assert.Equal(ok.Length, project.Compilation.Trees["Ok.cs"].GetRoot().FullSpan.Length);
        Assert.NotNull(project.Compilation.Compilation.GetTypeByMetadataName("A"));
        Assert.Null(project.Compilation.Compilation.GetTypeByMetadataName("Deep"));
        Assert.NotNull(project.Compilation.Compilation.GetTypeByMetadataName("Ok"));
        Assert.Contains(project.Diagnostics, d => d.Contains("Deep.cs", StringComparison.Ordinal) && d.Contains("nesting", StringComparison.Ordinal));
        Assert.DoesNotContain(project.Diagnostics, d => d.Contains("Ok.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Scan_FileTooLargeToParse_IsLeftOutOfTheCompilation_AndOneJustUnderTheLimitIsNot()
    {
        // Valid code: a class and a long comment. The limit is the corpus one (RoslynParsing.MaxFileChars = 2 Mi chars).
        const int limit = 2 * 1024 * 1024;
        string Pad(string cls, int total) => "class " + cls + " { }\n//" + new string('x', total - ("class " + cls + " { }\n//").Length - 1) + "\n";
        string big = Pad("Big", limit + 1000), nearly = Pad("Nearly", limit - 10);
        Assert.True(big.Length > limit && nearly.Length <= limit);
        string csproj = await WriteProjectAsync(Plain, ("Big.cs", big), ("Nearly.cs", nearly));

        ScanResult scan = await SolutionScanner.LoadAsync(csproj);

        ScannedProject project = Assert.Single(scan.Projects);
        Assert.Equal(big, scan.Files.Single(f => f.Path == "Big.cs").Text);
        Assert.Equal(0, project.Compilation.Trees["Big.cs"].GetRoot().FullSpan.Length);
        Assert.Null(project.Compilation.Compilation.GetTypeByMetadataName("Big"));
        Assert.NotNull(project.Compilation.Compilation.GetTypeByMetadataName("Nearly"));
        Assert.Contains(project.Diagnostics, d => d.Contains("Big.cs", StringComparison.Ordinal) && d.Contains("characters", StringComparison.Ordinal));
        Assert.DoesNotContain(project.Diagnostics, d => d.Contains("Nearly.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Scan_CompileItemThatIsNotACSharpFile_IsNeverReadIntoTheResult()
    {
        // A project chooses its own Compile items; one that names a data file (inside the root, or a secret outside it) must not
        // turn that file's bytes into "source". The .cs siblings are the control: they are returned.
        string secretOutside = Path.Combine(_root, "outside", "secret.txt");
        Directory.CreateDirectory(Path.GetDirectoryName(secretOutside)!);
        File.WriteAllText(secretOutside, "TOP-SECRET-OUTSIDE");
        // Default globbing is off: on Windows `**/*.cs` would also match Upper.CS and list it twice.
        string project = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup>"
            + "<ItemGroup><Compile Include=\"A.cs\" /><Compile Include=\"notes.txt\" /><Compile Include=\"" + secretOutside + "\" /><Compile Include=\"Upper.CS\" /></ItemGroup></Project>";
        string csproj = await WriteProjectAsync(project, ("A.cs", "class A { }\n"), ("notes.txt", "TOP-SECRET-INSIDE"), ("Upper.CS", "class Upper { }\n"));

        ScanResult scan = await SolutionScanner.LoadAsync(csproj);

        ScannedProject p = Assert.Single(scan.Projects);
        Assert.Contains(scan.Files, f => f.Path == "A.cs");
        Assert.Contains(scan.Files, f => f.Path == "Upper.CS"); // the extension test ignores case
        Assert.NotNull(p.Compilation.Compilation.GetTypeByMetadataName("Upper"));
        Assert.DoesNotContain(scan.Files, f => f.Text.Contains("TOP-SECRET", StringComparison.Ordinal));
        Assert.DoesNotContain(scan.Files, f => !f.Path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(p.Compilation.Compilation.SyntaxTrees, t => t.GetText().ToString().Contains("TOP-SECRET", StringComparison.Ordinal));
        Assert.Contains(p.Diagnostics, d => d.Contains("notes.txt", StringComparison.Ordinal));
        Assert.Contains(p.Diagnostics, d => d.Contains("secret.txt", StringComparison.Ordinal));
    }

    /// <summary>
    /// A project whose design-time build runs a command before the compile targets. The command writes the file
    /// <paramref name="started"/>, then waits for as long as the file <paramref name="hold"/> exists. The test creates the hold
    /// file, so the build cannot finish until the test deletes it (or the temp directory goes): there is no duration to outrun.
    /// The command polls every 100 ms.
    /// </summary>
    private static string HeldUntilFileRemoved(string hold, string started, string ended) =>
        "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>"
        + "<Target Name=\"Hang\" BeforeTargets=\"CoreCompile;ResolveAssemblyReferences\">"
        + "<Exec Condition=\"'$(OS)' == 'Windows_NT'\" Command=\"powershell -NoProfile -NonInteractive -Command &quot;Set-Content -LiteralPath '" + started + "' -Value 1; while (Test-Path -LiteralPath '" + hold + "') { Start-Sleep -Milliseconds 100 }; Set-Content -LiteralPath '" + ended + "' -Value 1&quot;\" />"
        + "<Exec Condition=\"'$(OS)' != 'Windows_NT'\" Command=\"echo 1 &gt; '" + started + "'; while [ -e '" + hold + "' ]; do sleep 0.1; done; echo 1 &gt; '" + ended + "'\" />"
        + "</Target></Project>";

    /// <summary>
    /// The project, or a solution file listing it (the two entry points wait on different workspace calls), with a design-time build
    /// that is held until <see cref="HeldBuild.Release"/> is called.
    /// </summary>
    private async Task<HeldBuild> HeldEntryAsync(bool solution)
    {
        string hold = Path.Combine(_root, "hold.flag"), started = Path.Combine(_root, "started.flag"), ended = Path.Combine(_root, "ended.flag");
        Directory.CreateDirectory(_root);
        File.WriteAllText(hold, "held");
        string csproj = await WriteProjectAsync(HeldUntilFileRemoved(hold, started, ended), ("A.cs", "class A { }\n"));
        string entry = csproj;
        if (solution)
        {
            entry = Path.Combine(_root, "S.slnx");
            File.WriteAllText(entry, "<Solution><Project Path=\"P/P.csproj\" /></Solution>");
        }
        return new HeldBuild(entry, hold, started, ended);
    }

    private sealed record HeldBuild(string Entry, string HoldFile, string StartedFile, string EndedFile)
    {
        /// <summary>True once the held design-time build command is running.</summary>
        public bool HasStarted => File.Exists(StartedFile);

        /// <summary>True once the held command has run to its end (only after <see cref="Release"/>).</summary>
        public bool HasEnded => File.Exists(EndedFile);

        /// <summary>Waits until the build command runs; fails the test when it never does (an empty project would then prove nothing).</summary>
        public async Task WaitForStartAsync()
        {
            using var limit = new CancellationTokenSource(HangGuard.Limit);
            try
            {
                while (!HasStarted) await Task.Delay(50, limit.Token);
            }
            catch (OperationCanceledException)
            {
                Assert.Fail("the held design-time build never started");
            }
        }

        public void Release()
        {
            try { File.Delete(HoldFile); }
            catch (IOException) { /* the directory is being removed: the build stops either way */ }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Scan_CancelledDuringTheDesignTimeBuild_ReturnsAtOnce(bool solution)
    {
        // The design-time build is held until the test releases it. The test cancels the token once the build command is running:
        // the call can only end through the token, while the build is still held. (No clock: a call that waits for the build
        // never returns, and HangGuard fails the test then.)
        HeldBuild build = await HeldEntryAsync(solution);
        try
        {
            using var cts = new CancellationTokenSource();
            Task<Exception?> call = Record.ExceptionAsync(() => SolutionScanner.LoadAsync(build.Entry, cts.Token));
            await build.WaitForStartAsync();
            await cts.CancelAsync();

            Exception? thrown = await HangGuard.ReturnsAsync(call, "LoadAsync");

            Assert.IsAssignableFrom<OperationCanceledException>(thrown);
            Assert.False(build.HasEnded, "the call returned while the design-time build was still held, so it ended before the build did");
        }
        finally
        {
            build.Release();
        }
    }

    /// <summary>
    /// Compiles a source generator whose <c>Initialize</c> sleeps for <paramref name="seconds"/> and returns the DLL path. Generators
    /// named by a project's <c>Analyzer</c> items are loaded and run in THIS process when the project's compilation is made.
    /// </summary>
    private string BuildSleepingGenerator(int seconds)
    {
        string source = "using Microsoft.CodeAnalysis; [Generator] public sealed class SleepGen : IIncrementalGenerator { "
            + "public void Initialize(IncrementalGeneratorInitializationContext context) { System.Threading.Thread.Sleep(" + seconds.ToString(CultureInfo.InvariantCulture) + "000); } }";
        string[] platform = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator);
        var compilation = Microsoft.CodeAnalysis.CSharp.CSharpCompilation.Create("SleepGen",
            [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree.ParseText(source)],
            platform.Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)),
            new Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Directory.CreateDirectory(_root);
        string dll = Path.Combine(_root, "SleepGen.dll");
        var emitted = compilation.Emit(dll);
        Assert.True(emitted.Success, string.Join("\n", emitted.Diagnostics));
        return dll;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Scan_CancelledWhileASourceGeneratorRuns_ReturnsAtOnce(bool solution)
    {
        // The design-time build is quick; the hold-up is the compilation, which runs the project's source generators.
        string project = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>"
            + "<ItemGroup><Analyzer Include=\"" + BuildSleepingGenerator(25) + "\" /></ItemGroup></Project>";
        string csproj = await WriteProjectAsync(project, ("A.cs", "class A { }\n"));
        string entry = csproj;
        if (solution)
        {
            entry = Path.Combine(_root, "S.slnx");
            File.WriteAllText(entry, "<Solution><Project Path=\"P/P.csproj\" /></Solution>");
        }
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
        var clock = Stopwatch.StartNew();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => SolutionScanner.LoadAsync(entry, cts.Token));

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(15), $"cancelled after 3 s, returned after {clock.Elapsed}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Scan_OptionTimeout_EndsAHungDesignTimeBuildWithATimeoutException(bool solution)
    {
        // The design-time build is held until the test releases it, so the call can only end through the option timeout. No clock
        // bound on how late it ends: a call that ignores the timeout waits for the held build and never returns (HangGuard).
        HeldBuild build = await HeldEntryAsync(solution);
        try
        {
            Exception? thrown = await HangGuard.ReturnsAsync(Record.ExceptionAsync(() =>
                SolutionScanner.LoadAsync(build.Entry, new SolutionScanOptions { Timeout = TimeSpan.FromSeconds(2) })), "LoadAsync");

            Assert.IsType<TimeoutException>(thrown);
            Assert.False(build.HasEnded, "the call returned while the design-time build was still held, so it ended before the build did");
        }
        finally
        {
            build.Release();
        }
    }
}
