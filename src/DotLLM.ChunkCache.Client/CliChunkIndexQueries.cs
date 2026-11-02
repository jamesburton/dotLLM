using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// <see cref="IChunkIndexQueries"/> over the <c>chunkcache</c> CLI: runs <c>&lt;exe&gt; select|status --json ...</c> and
/// parses the spec 7.2 output object from standard output (diagnostics go to standard error).
/// </summary>
/// <remarks>
/// Input mapping (spec 7.2 name to CLI option): <c>store</c> <c>--store</c>; <c>path</c> <c>--solution</c> for a
/// <c>.sln</c> / <c>.slnx</c> / <c>.csproj</c>, else <c>--directory</c>; <c>chunker</c> <c>--chunker</c>; <c>paths</c>
/// one <c>--path</c> each; <c>exclude</c> one <c>--exclude</c> each; <c>symbol</c> <c>--symbol</c>; <c>hops</c>
/// <c>--hops</c>; <c>changed_since</c> <c>--since</c>; <c>token_budget</c> <c>--budget</c>; <c>include_text</c>
/// <c>--include-text</c>; <c>order</c> <c>--order</c>; <c>list_missing</c> <c>--list-missing</c>; <c>server</c>
/// <c>--server</c>. Unset inputs add no option, so the CLI applies its own defaults.
/// </remarks>
public sealed class CliChunkIndexQueries : IChunkIndexQueries
{
    private readonly IProcessRunner _runner;

    /// <summary>Creates the queries.</summary>
    /// <param name="executable">The CLI to run (default <c>chunkcache</c>, resolved on <c>PATH</c>).</param>
    /// <param name="runner">The process runner; <see langword="null"/> = <see cref="ProcessRunner"/>.</param>
    public CliChunkIndexQueries(string executable = "chunkcache", IProcessRunner? runner = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(executable);
        Executable = executable;
        _runner = runner ?? new ProcessRunner();
    }

    /// <summary>The CLI executable.</summary>
    public string Executable { get; }

    /// <inheritdoc />
    public ValueTask<SelectResult> SelectAsync(SelectRequest r, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(r);
        var a = Begin("select", r.Store, r.Path, r.Chunker);
        foreach (string p in r.Paths ?? []) a.AddRange(["--path", p]);
        foreach (string p in r.Exclude ?? []) a.AddRange(["--exclude", p]);
        Add(a, "--symbol", r.Symbol);
        Add(a, "--hops", r.Hops);
        Add(a, "--since", r.ChangedSince);
        Add(a, "--budget", r.TokenBudget);
        if (r.IncludeText == true) a.Add("--include-text");
        Add(a, "--order", r.Order);
        Add(a, "--server", r.Server);
        return RunAsync(a, ChunkCacheJsonContext.Default.SelectResult, ct);
    }

    /// <inheritdoc />
    public ValueTask<StatusResult> StatusAsync(StatusRequest r, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(r);
        var a = Begin("status", r.Store, r.Path, r.Chunker);
        if (r.ListMissing == true) a.Add("--list-missing");
        Add(a, "--server", r.Server);
        return RunAsync(a, ChunkCacheJsonContext.Default.StatusResult, ct);
    }

    private static List<string> Begin(string verb, string? store, string? path, string? chunker)
    {
        var a = new List<string> { verb, "--json" };
        Add(a, "--store", store);
        if (path is not null) a.AddRange([IsProjectFile(path) ? "--solution" : "--directory", path]);
        Add(a, "--chunker", chunker);
        return a;
    }

    private static bool IsProjectFile(string path) =>
        path.EndsWith(".sln", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".slnx", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".csproj", StringComparison.OrdinalIgnoreCase);

    private static void Add(List<string> a, string option, string? value)
    {
        if (value is not null) a.AddRange([option, value]);
    }

    private static void Add(List<string> a, string option, int? value)
    {
        if (value is { } v) a.AddRange([option, v.ToString(CultureInfo.InvariantCulture)]);
    }

    private async ValueTask<T> RunAsync<T>(List<string> args, JsonTypeInfo<T> type, CancellationToken ct)
        where T : class
    {
        ProcessResult r = await _runner.RunAsync(Executable, args, ct).ConfigureAwait(false);
        string what = $"'{Executable} {args[0]}'";
        if (r.ExitCode != 0)
            throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} exited with code {r.ExitCode}: {r.Stderr.Trim()}");

        T? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize(r.Stdout, type);
        }
        catch (JsonException ex)
        {
            throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} printed output that is not its --json object: {ex.Message}");
        }

        return parsed ?? throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} printed JSON null instead of its --json object.");
    }
}

/// <summary>The default <see cref="IProcessRunner"/> over <see cref="Process"/>.</summary>
public sealed class ProcessRunner : IProcessRunner
{
    /// <inheritdoc />
    public async Task<ProcessResult> RunAsync(string executable, IReadOnlyList<string> arguments, CancellationToken ct)
    {
        ArgumentException.ThrowIfNullOrEmpty(executable);
        ArgumentNullException.ThrowIfNull(arguments);
        var psi = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (string a in arguments) psi.ArgumentList.Add(a);

        using var process = Process.Start(psi) ?? throw new InvalidOperationException($"Could not start '{executable}'.");
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(ct);
        Task<string> stderr = process.StandardError.ReadToEndAsync(ct);
        try
        {
            await process.WaitForExitAsync(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Only the process this call started (and its children), never anything matched by name.
            try { process.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) { /* already exited */ }
            throw;
        }

        return new ProcessResult(process.ExitCode, await stdout.ConfigureAwait(false), await stderr.ConfigureAwait(false));
    }
}
