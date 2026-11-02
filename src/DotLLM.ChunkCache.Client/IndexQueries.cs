// SPDX-License-Identifier: MIT
namespace DotLLM.ChunkCache.Client;

/// <summary>
/// The control-plane operations that read a local index rather than the server: <c>select</c> and <c>status</c>
/// (spec 7.2). The server has no endpoint for them.
/// </summary>
public interface IChunkIndexQueries
{
    /// <summary>Runs <c>chunkcache_select</c>.</summary>
    ValueTask<SelectResult> SelectAsync(SelectRequest r, CancellationToken ct);

    /// <summary>Runs <c>chunkcache_status</c>.</summary>
    ValueTask<StatusResult> StatusAsync(StatusRequest r, CancellationToken ct);
}

/// <summary>The outcome of one process run.</summary>
/// <param name="ExitCode">The exit code.</param>
/// <param name="Stdout">Everything written to standard output.</param>
/// <param name="Stderr">Everything written to standard error.</param>
public sealed record ProcessResult(int ExitCode, string Stdout, string Stderr);

/// <summary>Runs a process to completion (a seam for tests).</summary>
public interface IProcessRunner
{
    /// <summary>
    /// Runs <paramref name="executable"/> with <paramref name="arguments"/> (passed as an argument list, never a joined
    /// command line) and returns its exit code and output. Cancelling kills the started process tree.
    /// </summary>
    Task<ProcessResult> RunAsync(string executable, IReadOnlyList<string> arguments, CancellationToken ct);
}
