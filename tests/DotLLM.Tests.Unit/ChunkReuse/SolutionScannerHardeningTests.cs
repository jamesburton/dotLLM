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

    private const string Plain = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup></Project>";

    /// <summary>Writes <c>P/P.csproj</c> (the given text) and the files under <c>P</c>, restores it; returns the project path.</summary>
    private async Task<string> WriteProjectAsync(string project, params (string Name, string Text)[] files)
    {
        string dir = Path.Combine(_root, "P");
        Directory.CreateDirectory(dir);
        foreach ((string name, string text) in files)
            File.WriteAllText(Path.Combine(dir, name), text);
        string csproj = Path.Combine(dir, "P.csproj");
        File.WriteAllText(csproj, project);
        var psi = new ProcessStartInfo("dotnet") { WorkingDirectory = dir, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string a in new[] { "restore", csproj, "--nologo", "-v", "q" }) psi.ArgumentList.Add(a);
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
        Assert.True(p.ExitCode == 0, $"dotnet restore exited {p.ExitCode}: {output}");
        return csproj;
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
        string project = "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>"
            + "<ItemGroup><Compile Include=\"notes.txt\" /><Compile Include=\"" + secretOutside + "\" /></ItemGroup></Project>";
        string csproj = await WriteProjectAsync(project, ("A.cs", "class A { }\n"), ("notes.txt", "TOP-SECRET-INSIDE"));

        ScanResult scan = await SolutionScanner.LoadAsync(csproj);

        ScannedProject p = Assert.Single(scan.Projects);
        Assert.Contains(scan.Files, f => f.Path == "A.cs");
        Assert.DoesNotContain(scan.Files, f => f.Text.Contains("TOP-SECRET", StringComparison.Ordinal));
        Assert.DoesNotContain(scan.Files, f => !f.Path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(p.Compilation.Compilation.SyntaxTrees, t => t.GetText().ToString().Contains("TOP-SECRET", StringComparison.Ordinal));
        Assert.Contains(p.Diagnostics, d => d.Contains("notes.txt", StringComparison.Ordinal));
        Assert.Contains(p.Diagnostics, d => d.Contains("secret.txt", StringComparison.Ordinal));
    }

    /// <summary>A project whose design-time build runs a command for <paramref name="seconds"/> before the compile targets.</summary>
    private static string Hanging(int seconds) =>
        "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>"
        + "<Target Name=\"Hang\" BeforeTargets=\"CoreCompile;ResolveAssemblyReferences\">"
        + "<Exec Condition=\"'$(OS)' == 'Windows_NT'\" Command=\"ping -n " + (seconds + 1).ToString(CultureInfo.InvariantCulture) + " 127.0.0.1 &gt; nul\" />"
        + "<Exec Condition=\"'$(OS)' != 'Windows_NT'\" Command=\"sleep " + seconds.ToString(CultureInfo.InvariantCulture) + "\" />"
        + "</Target></Project>";

    [Fact]
    public async Task Scan_CancelledDuringTheDesignTimeBuild_ReturnsAtOnce()
    {
        // The design-time build runs for 40 s. The caller's token fires after 2 s: the call must end then, not when the build does.
        string csproj = await WriteProjectAsync(Hanging(40), ("A.cs", "class A { }\n"));
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        var clock = Stopwatch.StartNew();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => SolutionScanner.LoadAsync(csproj, cts.Token));

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(20), $"cancelled after 2 s, returned after {clock.Elapsed}");
    }

    [Fact]
    public async Task Scan_OptionTimeout_EndsAHungDesignTimeBuildWithATimeoutException()
    {
        string csproj = await WriteProjectAsync(Hanging(40), ("A.cs", "class A { }\n"));
        var clock = Stopwatch.StartNew();

        await Assert.ThrowsAsync<TimeoutException>(() =>
            SolutionScanner.LoadAsync(csproj, new SolutionScanOptions { Timeout = TimeSpan.FromSeconds(2) }));

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(20), $"timeout 2 s, returned after {clock.Elapsed}");
    }
}
