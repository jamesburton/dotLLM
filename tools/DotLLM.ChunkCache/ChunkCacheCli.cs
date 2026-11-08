using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.ChunkCache;

/// <summary>
/// The <c>chunkcache</c> verbs and their dispatch. <see cref="Run"/> is the whole program (Program.Main calls it); tests call it
/// in process with their own writers and working directory.
/// </summary>
internal static class ChunkCacheCli
{
    public const string Usage = """
        chunkcache: chunk-level KV reuse for .NET solutions and source trees (dotLLM research tool; CPU encoder)

        usage: chunkcache <verb> [options]        chunkcache <verb> --help

          build     scan a solution/directory, update the store's index, encode and store missing chunks
          select    choose chunks for a context: paths/globs, a symbol's neighbourhood, git changes; under a token budget
          status    which chunks are missing or stale (by file and line range), and store statistics
          evict     list (default) or delete (--apply) entries no index references; --all-for-model removes one model's entries
          compare   quality vs prefill for a task: selectors x budgets x chunkers against full prefill

        exit codes: 0 ok; 1 error (I/O, busy store lock, model load, symbol not found, unreadable index);
                    2 usage (unknown verb/option, missing or bad value); 3 strict failure (build --strict: skipped
                    projects/load failures/excluded files; status --strict: not up to date)

        Store layout: <store>/<2 hex>/<key>.ckv entries (F32 K/V), one index-<chunker>.<model12>.<p8>.default.json per chunker, model and producer (p8: 8 hex of the producer identity) (older index names are migrated by the next build),
        and chunkcache.json (the store registry: each index's model, tree root and source). build holds index-<chunker>.<model12>.<p8>.default.lock while it updates its index and <store>/.chunkstore.lock only briefly (registry save, each put); evict --apply holds .chunkstore.lock for its pass.
        --json prints the result as one JSON document on stdout (snake_case names of the chunkcache_* MCP tool schemas);
        progress and warnings go to stderr.
        """;

    private static readonly Dictionary<string, (IReadOnlyDictionary<string, OptionKind> Spec, string Help, Func<CliContext, CliArgs, int> Run)> Verbs = new(StringComparer.Ordinal)
    {
        ["build"] = (BuildCommand.Spec, BuildCommand.Help, BuildCommand.Run),
        ["select"] = (SelectCommand.Spec, SelectCommand.Help, SelectCommand.Run),
        ["status"] = (StatusCommand.Spec, StatusCommand.Help, StatusCommand.Run),
        ["evict"] = (EvictCommand.Spec, EvictCommand.Help, EvictCommand.Run),
    };

    /// <summary>Runs one command line.</summary>
    /// <param name="args">Arguments (verb first). <c>--verbose</c> anywhere adds stack traces to error messages.</param>
    /// <param name="output">Standard output.</param>
    /// <param name="error">Standard error.</param>
    /// <param name="workingDirectory">Directory relative paths resolve against (default: the process's).</param>
    /// <returns>The exit code (<see cref="ExitCodes"/>).</returns>
    /// <remarks>
    /// With <c>--json</c>, a failure that produces no result writes <c>{"error": {"code", "exit_code", "message"}}</c> on stdout
    /// (codes: <c>usage</c>, <c>strict</c>, <c>store_busy</c>, <c>symbol_not_found</c>, <c>error</c>, <c>internal</c>); the
    /// human-readable message still goes to stderr.
    /// </remarks>
    public static int Run(string[] args, TextWriter output, TextWriter error, string? workingDirectory = null)
    {
        bool verbose = args.Contains("--verbose", StringComparer.Ordinal);
        bool json = args.Contains("--json", StringComparer.Ordinal);
        args = args.Where(a => !string.Equals(a, "--verbose", StringComparison.Ordinal)).ToArray();
        var ctx = new CliContext(output, error, workingDirectory ?? Environment.CurrentDirectory);

        int Fail(int exit, string code, string message, Exception? ex = null, string? extra = null)
        {
            error.WriteLine(message);
            if (extra is not null) error.WriteLine(extra);
            if (verbose && ex is not null) error.WriteLine(ex.ToString());
            if (json) ctx.WriteJson(new ErrorResult(new ErrorBody(code, exit, message)));
            return exit;
        }

        if (args.Length == 0) return Fail(ExitCodes.Usage, "usage", "usage error: no verb.", extra: Usage);
        if (args[0] is "--help" or "-h" or "help")
        {
            output.WriteLine(Usage);
            return ExitCodes.Ok;
        }
        if (!Verbs.TryGetValue(args[0], out var verb)) return Fail(ExitCodes.Usage, "usage", $"usage error: unknown verb '{args[0]}'.", extra: Usage);
        try
        {
            CliArgs parsed = CliArgs.Parse(args[0], args.Skip(1), verb.Spec);
            if (parsed.Flag("help"))
            {
                output.WriteLine(verb.Help);
                return ExitCodes.Ok;
            }
            ctx.Json = parsed.Flag("json");
            return verb.Run(ctx, parsed);
        }
        catch (UsageException ex)
        {
            return Fail(ExitCodes.Usage, "usage", $"usage error: {ex.Message}", ex, verb.Help);
        }
        catch (StrictFailureException ex)
        {
            return Fail(ExitCodes.StrictFailure, "strict", $"strict: {ex.Message}", ex);
        }
        catch (ChunkStoreLockTimeoutException ex)
        {
            return Fail(ExitCodes.Error, "store_busy", $"error: store busy: {ex.Message}", ex);
        }
        catch (LedgerUnreadableException ex)
        {
            return Fail(ExitCodes.Error, "ledger_unreadable", $"error: {ex.Message}", ex);
        }
        catch (SymbolNotFoundException ex)
        {
            return Fail(ExitCodes.Error, "symbol_not_found", $"error: {ex.Message}", ex,
                ex.Suggestions.Count > 0 ? $"  did you mean: {string.Join(", ", ex.Suggestions)}" : null);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or InvalidDataException
            or ArgumentException or NotSupportedException or GitException or KeyNotFoundException or FormatException)
        {
            return Fail(ExitCodes.Error, "error", $"error: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            // Anything else is a bug or an unexpected environment: still exit 1 with one clean line (stack trace with --verbose).
            return Fail(ExitCodes.Error, "internal", $"error: unexpected {ex.GetType().Name}: {ex.Message}" + (verbose ? "" : " (rerun with --verbose for details)"), ex);
        }
    }

    /// <summary>The JSON error document.</summary>
    internal sealed record ErrorResult(ErrorBody Error);

    /// <summary>The error object.</summary>
    internal sealed record ErrorBody(string Code, int ExitCode, string Message);
}
