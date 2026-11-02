using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="CliChunkIndexQueries"/> with a fake <see cref="IProcessRunner"/>.</summary>
public sealed class CliChunkIndexQueriesTests
{
    private const string SelectJson = """
        {"chunks":[{"id":"src/A.cs:1-9@0","path":"src/A.cs","start_line":1,"end_line":9,"tokens":42,
                    "hash":"sha256:00","reason":"path","detail":"src/*.cs","hop":0,"stored":true,"text":"class A {}\n"}],
         "selected_tokens":42,"candidate_tokens":100,"trimmed":1,"unmatched_patterns":["nope/*"],"notes":["n1"]}
        """;

    /// <summary>
    /// Every select input becomes a CLI option (spec 7.2 names mapped onto the <c>chunkcache</c> verbs' options), the
    /// call always asks for <c>--json</c>, and the spec 7.2 output object is parsed from stdout.
    /// </summary>
    [Fact]
    public async Task CliQueries_PassesArgsAndParsesJson()
    {
        var runner = new FakeRunner(new ProcessResult(0, SelectJson, "diagnostics on stderr\n"));
        var q = new CliChunkIndexQueries("C:/tools/chunkcache.exe", runner);

        var r = await q.SelectAsync(new SelectRequest
        {
            Store = "C:/store", Server = "http://localhost:8080", Path = "C:/src/App.slnx", Chunker = "symbol-512",
            Paths = ["src/*.cs", "lib/**"], Exclude = ["**/obj/**"], Symbol = "App.Main", Hops = 2,
            ChangedSince = "HEAD~3", TokenBudget = 8000, IncludeText = true, Order = "rank",
        }, CancellationToken.None);

        Assert.Equal("C:/tools/chunkcache.exe", runner.Executable);
        Assert.Equal(
        [
            "select", "--json", "--store", "C:/store", "--solution", "C:/src/App.slnx", "--chunker", "symbol-512",
            "--path", "src/*.cs", "--path", "lib/**", "--exclude", "**/obj/**", "--symbol", "App.Main", "--hops", "2",
            "--since", "HEAD~3", "--budget", "8000", "--include-text", "--order", "rank", "--server", "http://localhost:8080",
        ], runner.Arguments!);

        var c = Assert.Single(r.Chunks!);
        Assert.Equal("src/A.cs:1-9@0", c.Id);
        Assert.Equal(1, c.StartLine);
        Assert.Equal(9, c.EndLine);
        Assert.Equal(42, c.Tokens);
        Assert.True(c.Stored);
        Assert.Equal("class A {}\n", c.Text);
        Assert.Equal(42, r.SelectedTokens);
        Assert.Equal(100, r.CandidateTokens);
        Assert.Equal(1, r.Trimmed);
        Assert.Equal(["nope/*"], r.UnmatchedPatterns!);
    }

    /// <summary>A folder (not .sln/.slnx/.csproj) maps to <c>--directory</c>; unset inputs add no options.</summary>
    [Theory]
    [InlineData("C:/src/repo", "--directory")]
    [InlineData("C:/src/App.sln", "--solution")]
    [InlineData("C:/src/App.csproj", "--solution")]
    public async Task CliQueries_StatusMapsPathAndOmitsUnset(string path, string option)
    {
        var runner = new FakeRunner(new ProcessResult(0,
            """{"total_chunks":10,"missing_chunks":2,"missing_tokens":300,"up_to_date":false,"stale_files":["a.cs"],"new_files":[],"deleted_files":[],"missing":[{"path":"a.cs","start_line":1,"end_line":4,"tokens":150}],"store_bytes":1024,"budget_bytes":4096,"entries":8}""",
            ""));
        var r = await new CliChunkIndexQueries(runner: runner).StatusAsync(new StatusRequest { Path = path, ListMissing = true }, CancellationToken.None);

        Assert.Equal("chunkcache", runner.Executable);
        Assert.Equal(["status", "--json", option, path, "--list-missing"], runner.Arguments!);
        Assert.Equal(10, r.TotalChunks);
        Assert.False(r.UpToDate);
        Assert.Equal(["a.cs"], r.StaleFiles!);
        Assert.Equal(150, Assert.Single(r.Missing!).Tokens);
        Assert.Equal(4096, r.BudgetBytes);
    }

    /// <summary>A non-zero exit is an error carrying the exit code and stderr, even when stdout holds JSON.</summary>
    [Fact]
    public async Task CliQueries_NonZeroExit_ThrowsWithStderr()
    {
        var runner = new FakeRunner(new ProcessResult(1, SelectJson, "error: index unreadable: index-symbol-512.json\n"));

        var ex = await Assert.ThrowsAsync<ChunkCacheCliException>(() =>
            new CliChunkIndexQueries(runner: runner).SelectAsync(new SelectRequest(), CancellationToken.None).AsTask());

        Assert.Equal(1, ex.ExitCode);
        Assert.Contains("index unreadable", ex.Stderr, StringComparison.Ordinal);
        Assert.Contains("index unreadable", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    [InlineData("null")]
    public async Task CliQueries_StdoutNotJson_Throws(string stdout)
    {
        var runner = new FakeRunner(new ProcessResult(0, stdout, ""));
        var ex = await Assert.ThrowsAsync<ChunkCacheCliException>(() =>
            new CliChunkIndexQueries(runner: runner).StatusAsync(new StatusRequest(), CancellationToken.None).AsTask());
        Assert.Equal(0, ex.ExitCode);
    }

    /// <summary>The real runner passes arguments as a list (a value with spaces stays one argument) and captures both streams.</summary>
    [Fact]
    public async Task ProcessRunner_RunsRealProcess()
    {
        var result = await new ProcessRunner().RunAsync("dotnet", ["--version"], CancellationToken.None);
        Assert.Equal(0, result.ExitCode);
        Assert.Matches(@"^\d+\.\d+\.\d+", result.Stdout);

        var bad = await new ProcessRunner().RunAsync("dotnet", ["no such command with spaces"], CancellationToken.None);
        Assert.NotEqual(0, bad.ExitCode);
        Assert.Contains("no such command with spaces", bad.Stdout + bad.Stderr, StringComparison.Ordinal);
    }
}
