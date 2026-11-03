// SPDX-License-Identifier: MIT
using System.ComponentModel;
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
/// <para>
/// Input mapping (spec 7.2 name to CLI option): <c>store</c> <c>--store</c>; <c>path</c> <c>--solution</c> for a
/// <c>.sln</c> / <c>.slnx</c> / <c>.csproj</c>, else <c>--directory</c>; <c>chunker</c> <c>--chunker</c>; <c>paths</c>
/// one <c>--path</c> each; <c>exclude</c> one <c>--exclude</c> each; <c>symbol</c> <c>--symbol</c>; <c>hops</c>
/// <c>--hops</c>; <c>changed_since</c> <c>--since</c>; <c>token_budget</c> <c>--budget</c>; <c>include_text</c>
/// <c>--include-text</c>; <c>order</c> <c>--order</c>; <c>list_missing</c> <c>--list-missing</c>; <c>server</c>
/// <c>--server</c>. Unset inputs add no option, so the CLI applies its own defaults.
/// </para>
/// <para>
/// <b>Executable resolution.</b> A bare name is resolved against <c>PATH</c> only, to an absolute path; relative
/// <c>PATH</c> entries (such as <c>.</c>) and the working directory are never searched, because a harness's working
/// directory is usually a repository it does not trust. A relative path with a directory part is rejected; a fully
/// qualified path is used as given.
/// </para>
/// <para>
/// <b>List values.</b> Today's CLI splits list options on commas and trims entries (contract ruling 18), so a
/// <c>paths</c> / <c>exclude</c> value containing a comma, or with leading or trailing whitespace, is rejected with
/// <see cref="ArgumentException"/> until the CLI gains a non-splitting form.
/// </para>
/// </remarks>
public sealed class CliChunkIndexQueries : IChunkIndexQueries
{
    private readonly IProcessRunner _runner;

    /// <summary>Creates the queries.</summary>
    /// <param name="executable">The CLI: a bare name resolved on <c>PATH</c> only (default <c>chunkcache</c>), or a fully qualified path.</param>
    /// <param name="runner">The process runner; <see langword="null"/> = <see cref="ProcessRunner"/>.</param>
    /// <param name="timeout">How long one CLI call may run before it is cancelled (default <see cref="DefaultTimeout"/>;
    /// <see cref="System.Threading.Timeout.InfiniteTimeSpan"/> disables it).</param>
    public CliChunkIndexQueries(string executable = "chunkcache", IProcessRunner? runner = null, TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(executable);
        if (!Path.IsPathFullyQualified(executable) && executable.IndexOfAny(['/', '\\']) >= 0)
            throw new ArgumentException(
                $"'{executable}' is a relative path; give a bare name (resolved on PATH) or a fully qualified path.", nameof(executable));
        TimeSpan t = timeout ?? DefaultTimeout;
        if (t <= TimeSpan.Zero && t != System.Threading.Timeout.InfiniteTimeSpan)
            throw new ArgumentOutOfRangeException(nameof(timeout), t, "The timeout must be positive or infinite.");

        Executable = executable;
        Timeout = t;
        _runner = runner ?? new ProcessRunner();
    }

    /// <summary>The default per-call timeout: five minutes (a cold solution load can take a while).</summary>
    public static TimeSpan DefaultTimeout { get; } = TimeSpan.FromMinutes(5);

    /// <summary>The per-call timeout.</summary>
    public TimeSpan Timeout { get; }

    /// <summary>The search path for a bare <see cref="Executable"/>; <see langword="null"/> = the <c>PATH</c> environment variable.</summary>
    public string? PathVariable { get; init; }

    /// <summary>The CLI executable as given.</summary>
    public string Executable { get; }

    /// <inheritdoc />
    public ValueTask<SelectResult> SelectAsync(SelectRequest r, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(r);
        var a = Begin("select", r.Store, r.Path, r.Chunker);
        foreach (string p in r.Paths ?? []) a.AddRange(["--path", ListValue(p, "paths")]);
        foreach (string p in r.Exclude ?? []) a.AddRange(["--exclude", ListValue(p, "exclude")]);
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

    /// <summary>
    /// Resolves a bare <paramref name="name"/> against the fully qualified entries of <paramref name="pathVariable"/>
    /// (on Windows, trying each <c>PATHEXT</c> extension when the name has none). Returns <see langword="null"/> when
    /// not found. Never consults the working directory.
    /// </summary>
    internal static string? ResolveOnPath(string name, string? pathVariable)
    {
        if (Path.IsPathFullyQualified(name))
            return name;
        string[] candidates = [name];
        if (OperatingSystem.IsWindows() && !Path.HasExtension(name))
        {
            string pathExt = Environment.GetEnvironmentVariable("PATHEXT") ?? ".COM;.EXE;.BAT;.CMD";
            candidates = pathExt.Split(';', StringSplitOptions.RemoveEmptyEntries).Select(e => name + e.ToLowerInvariant()).ToArray();
        }

        foreach (string dir in (pathVariable ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            string d = dir.Trim().Trim('"');
            if (!Path.IsPathFullyQualified(d))
                continue; // ".", "bin" and the like resolve against the working directory: never searched
            foreach (string c in candidates)
            {
                string full = Path.Combine(d, c);
                if (File.Exists(full))
                    return full;
            }
        }

        return null;
    }

    private static string ListValue(string value, string field)
    {
        if (value.Contains(',', StringComparison.Ordinal) || value.Length != value.Trim().Length || value.Length == 0)
            throw new ArgumentException(
                $"{field} value '{value}' contains a comma or leading/trailing whitespace, which the chunkcache CLI would split or trim; "
                + "use separate patterns without commas until the CLI has a non-splitting list option.", nameof(value));
        return value;
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
        string what = $"'{Executable} {args[0]}'";
        string exe = ResolveOnPath(Executable, PathVariable ?? Environment.GetEnvironmentVariable("PATH"))
            ?? throw new ChunkCacheCliException(-1, "", $"'{Executable}' was not found on PATH (the working directory is never searched).");

        using var timeout = new CancellationTokenSource();
        if (Timeout != System.Threading.Timeout.InfiniteTimeSpan) timeout.CancelAfter(Timeout);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeout.Token);

        ProcessResult r;
        try
        {
            r = await _runner.RunAsync(exe, args, linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException ex) when (timeout.IsCancellationRequested && !ct.IsCancellationRequested)
        {
            throw new ChunkCacheCliException(-1, "", $"{what} timed out after {Timeout}.", new TimeoutException(ex.Message, ex));
        }

        if (r.ExitCode != 0)
            throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} exited with code {r.ExitCode}: {r.Stderr.Trim()}");

        T? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize(r.Stdout, type);
        }
        catch (JsonException ex)
        {
            throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} printed output that is not its --json object: {ex.Message}", ex);
        }

        return parsed ?? throw new ChunkCacheCliException(r.ExitCode, r.Stderr, $"{what} printed JSON null instead of its --json object.");
    }
}

/// <summary>The default <see cref="IProcessRunner"/> over <see cref="Process"/>.</summary>
/// <remarks>
/// Standard output and error are read as UTF-8: the CLI must write UTF-8 JSON (contract ruling 18). A process that cannot
/// be started is a <see cref="ChunkCacheCliException"/>. Cancelling kills only the process this call started (and its
/// children), never anything matched by name.
/// </remarks>
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

        Process? started;
        try
        {
            started = Process.Start(psi);
        }
        catch (Win32Exception ex)
        {
            throw new ChunkCacheCliException(-1, "", $"Could not start '{executable}': {ex.Message}", ex);
        }

        using Process process = started ?? throw new ChunkCacheCliException(-1, "", $"Could not start '{executable}'.");
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(ct);
        Task<string> stderr = process.StandardError.ReadToEndAsync(ct);
        try
        {
            await process.WaitForExitAsync(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); }
            catch (Exception ex) when (ex is InvalidOperationException or Win32Exception) { /* exited, or not ours to kill */ }
            throw;
        }

        return new ProcessResult(process.ExitCode, await stdout.ConfigureAwait(false), await stderr.ConfigureAwait(false));
    }
}
