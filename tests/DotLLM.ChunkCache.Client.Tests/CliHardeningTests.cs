using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// Hardening of the <c>chunkcache</c> CLI wrapper (Task 6 review S1-S3): PATH-only resolution, a default timeout, and
/// rejecting list values the current CLI would split (contract ruling 18).
/// </summary>
public sealed class CliHardeningTests : IDisposable
{
    private static readonly string s_name = OperatingSystem.IsWindows() ? "chunkcache.exe" : "chunkcache";
    private readonly List<string> _dirs = [];

    public void Dispose()
    {
        foreach (string d in _dirs)
        {
            try { Directory.Delete(d, recursive: true); }
            catch (IOException) { /* best effort */ }
        }
    }

    private string DirWithCli(string root)
    {
        string dir = Path.Combine(root, "cc-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, s_name), "");
        _dirs.Add(dir);
        return dir;
    }

    private static FakeRunner OkRunner() => new(new ProcessResult(0, "{}", ""));

    // ---------------------------------------------------------------- S1: PATH only

    /// <summary>A bare name resolves to the absolute file on PATH, and that absolute path is what gets started.</summary>
    [Fact]
    public async Task BareName_ResolvedOnPath_RunnerGetsAbsolutePath()
    {
        string onPath = DirWithCli(Path.GetTempPath());
        var runner = OkRunner();
        var q = new CliChunkIndexQueries("chunkcache", runner) { PathVariable = onPath };

        await q.StatusAsync(new StatusRequest(), CancellationToken.None);

        Assert.Equal(Path.Combine(onPath, s_name), runner.Executable);
        Assert.True(Path.IsPathFullyQualified(runner.Executable!));
    }

    /// <summary>
    /// A relative PATH entry (".", "bin") would resolve against the working directory, which is usually the user's
    /// repository: a planted CLI there must never run. Such entries are skipped, so the name is "not found".
    /// </summary>
    [Fact]
    public async Task RelativePathEntries_AreIgnored_PlantedCliInWorkingDirectoryNeverRuns()
    {
        string planted = DirWithCli(Environment.CurrentDirectory);
        string relative = Path.GetFileName(planted);
        var runner = OkRunner();
        var q = new CliChunkIndexQueries("chunkcache", runner) { PathVariable = string.Join(Path.PathSeparator, ".", relative) };

        var ex = await Assert.ThrowsAsync<ChunkCacheCliException>(() => q.StatusAsync(new StatusRequest(), CancellationToken.None).AsTask());

        Assert.Contains("PATH", ex.Message, StringComparison.Ordinal);
        Assert.Null(runner.Executable);
    }

    /// <summary>The working directory itself is never searched, even when nothing is on PATH.</summary>
    [Fact]
    public async Task WorkingDirectory_NotSearched()
    {
        string planted = DirWithCli(Path.GetTempPath());
        string saved = Environment.CurrentDirectory;
        var runner = OkRunner();
        var q = new CliChunkIndexQueries("chunkcache", runner) { PathVariable = "" };
        try
        {
            Environment.CurrentDirectory = planted;
            await Assert.ThrowsAsync<ChunkCacheCliException>(() => q.StatusAsync(new StatusRequest(), CancellationToken.None).AsTask());
        }
        finally
        {
            Environment.CurrentDirectory = saved;
        }

        Assert.Null(runner.Executable);
    }

    /// <summary>A relative path with a directory part is ambiguous (working-directory relative): rejected up front.</summary>
    [Theory]
    [InlineData("tools/chunkcache")]
    [InlineData("./chunkcache")]
    [InlineData("..\\chunkcache.exe")]
    public void RelativePathWithDirectory_Rejected(string executable)
    {
        Assert.Throws<ArgumentException>(() => new CliChunkIndexQueries(executable, OkRunner()));
    }

    /// <summary>A fully qualified path is used as given.</summary>
    [Fact]
    public async Task FullyQualifiedPath_UsedAsGiven()
    {
        var runner = OkRunner();
        await new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, runner) { PathVariable = "" }
            .StatusAsync(new StatusRequest(), CancellationToken.None);
        Assert.Equal(CliChunkIndexQueriesTests.Exe, runner.Executable);
    }

    // ---------------------------------------------------------------- S2: timeout

    [Fact]
    public void DefaultTimeout_IsFiveMinutes()
    {
        Assert.Equal(TimeSpan.FromMinutes(5), CliChunkIndexQueries.DefaultTimeout);
        Assert.Equal(TimeSpan.FromMinutes(5), new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, OkRunner()).Timeout);
    }

    /// <summary>A CLI that never finishes (store lock, MSBuild) is cancelled at the timeout and reported as a CLI error.</summary>
    [Fact]
    public async Task Timeout_CancelsRunnerAndThrows()
    {
        var runner = new HangingRunner();
        var q = new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, runner, TimeSpan.FromMilliseconds(100));

        var ex = await Assert.ThrowsAsync<ChunkCacheCliException>(() => q.SelectAsync(new SelectRequest(), CancellationToken.None).AsTask());

        Assert.IsType<TimeoutException>(ex.InnerException);
        Assert.Contains("timed out", ex.Message, StringComparison.Ordinal);
        Assert.True(runner.SawCancellation);
    }

    /// <summary>The caller's own cancellation stays an <see cref="OperationCanceledException"/>.</summary>
    [Fact]
    public async Task CallerCancellation_Propagates()
    {
        var runner = new HangingRunner();
        var q = new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, runner, TimeSpan.FromMinutes(5));
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => q.SelectAsync(new SelectRequest(), cts.Token).AsTask());
    }

    [Fact]
    public void Timeout_NonPositive_Rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, OkRunner(), TimeSpan.Zero));
    }

    // ---------------------------------------------------------------- S3: values the CLI would split

    /// <summary>
    /// The current CLI splits every list option on commas and trims entries, so a brace glob would silently become two
    /// broken patterns. Until Task 17 adds a non-splitting form (ruling 18), such values are rejected before running.
    /// </summary>
    [Theory]
    [InlineData("src/**/*.{cs,razor}", null)]
    [InlineData(" src/*.cs", null)]
    [InlineData("src/*.cs ", null)]
    [InlineData(null, "a,b")]
    [InlineData(null, " obj/**")]
    public async Task ListValues_TheCliWouldSplitOrTrim_Rejected(string? path, string? exclude)
    {
        var runner = OkRunner();
        var q = new CliChunkIndexQueries(CliChunkIndexQueriesTests.Exe, runner);
        var r = new SelectRequest { Paths = path is null ? null : [path], Exclude = exclude is null ? null : [exclude] };

        await Assert.ThrowsAsync<ArgumentException>(() => q.SelectAsync(r, CancellationToken.None).AsTask());
        Assert.Null(runner.Executable);
    }

    // ---------------------------------------------------------------- runner nits

    /// <summary>A CLI that cannot be started is a <see cref="ChunkCacheCliException"/>, not a raw Win32 error.</summary>
    [Fact]
    public async Task ProcessRunner_MissingExecutable_ThrowsCliException()
    {
        string missing = Path.Combine(Path.GetTempPath(), "cc-missing-" + Guid.NewGuid().ToString("N"), s_name);
        var ex = await Assert.ThrowsAsync<ChunkCacheCliException>(() => new ProcessRunner().RunAsync(missing, [], CancellationToken.None));
        Assert.Contains(missing, ex.Message, StringComparison.Ordinal);
    }

    /// <summary>A runner that waits until cancelled.</summary>
    private sealed class HangingRunner : IProcessRunner
    {
        public bool SawCancellation { get; private set; }

        public async Task<ProcessResult> RunAsync(string executable, IReadOnlyList<string> arguments, CancellationToken ct)
        {
            try
            {
                await Task.Delay(Timeout.Infinite, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                SawCancellation = true;
                throw;
            }

            return new ProcessResult(0, "{}", "");
        }
    }
}
