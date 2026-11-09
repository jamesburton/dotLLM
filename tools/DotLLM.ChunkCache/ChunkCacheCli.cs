using System.Net.Http;
using DotLLM.ChunkCache.Client;
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

        usage: chunkcache <verb> [options]        chunkcache <verb> --help        chunkcache --help

          build     scan a solution/directory, update the store's index, encode and store (or queue on a server) the missing chunks
          select    choose chunks for a context: paths/globs, a symbol's neighbourhood, git changes; under a token budget
          status    which chunks are missing or stale (by file and line range), and store statistics
          list      read-only inventory of one or more stores: entries and sizes, owner-lock state, each index's model and chunk counts
          evict     --server: ask the server to evict (orphans | path | namespace | all; dry run unless --apply);
                    --offline: list or delete orphans (or, with --mode all --model, one model's entries) holding the store's owner lock
          prewarm   --server: queue chunk texts (a selection or a --texts file) for the server to encode (text only)
          explain   --server: a dry run of the chunk_cache plan for a request file (POST /v1/chunk-cache/explain)

        Every verb's --help lists all of its options (value form, default, whether repeatable, environment variable), its exit codes,
        its --json fields and examples. The server owns the model identity and every K/V write: build/prewarm/evict go through
        --server URL; --offline is the escape hatch for a store no server owns.

        global option: --verbose (any verb) adds the stack trace of an error to the message on stderr.
        environment:   CHUNKCACHE_API_KEY is the API key for --server. It is read from the environment only (there is no key option).
        output:        --json prints the result as one JSON document on stdout (snake_case names of the chunkcache_* MCP tool schemas);
                       progress and warnings go to stderr. A failure prints {"error": {"code", "message", "exit_code"}} instead.

        exit codes:
          0  ok
          1  error (I/O, busy store lock, store owned by a server or another writer, model load, symbol not found, unreadable index
             or ledger, server error or unreachable server)
          2  usage (unknown verb/option, missing or bad value)
          3  strict failure (build --strict: skipped projects/load failures/excluded files; status --strict: not up to date or
             ledger unreadable)

        Store layout: <store>/<2 hex>/<key>.ckv entries (F32 K/V), one index-<chunker>.<model12>.<p8>.default.json per chunker, model and producer (p8: 8 hex of the producer identity) (older index names are migrated by the next build),
        and chunkcache.json (the store registry: each index's model, tree root and source). build holds index-<chunker>.<model12>.<p8>.default.lock while it updates its index and <store>/.chunkstore.lock only briefly (registry save, each put); evict --apply holds .chunkstore.lock for its pass.
        """;

    /// <summary>The verbs: option registry and help (<see cref="VerbDef"/>); the parser and <c>--help</c> both read it.</summary>
    internal static IReadOnlyDictionary<string, VerbDef> Verbs => VerbDefs.All;

    private static readonly Dictionary<string, Func<CliContext, CliArgs, int>> Runners = new(StringComparer.Ordinal)
    {
        ["build"] = BuildCommand.Run,
        ["select"] = SelectCommand.Run,
        ["status"] = StatusCommand.Run,
        ["evict"] = EvictCommand.Run,
        ["prewarm"] = PrewarmCommand.Run,
        ["explain"] = ExplainCommand.Run,
        ["list"] = ListCommand.Run,
    };

    /// <summary>Runs one command line.</summary>
    /// <param name="args">Arguments (verb first). <c>--verbose</c> anywhere adds stack traces to error messages.</param>
    /// <param name="output">Standard output.</param>
    /// <param name="error">Standard error.</param>
    /// <param name="workingDirectory">Directory relative paths resolve against (default: the process's).</param>
    /// <param name="httpHandler">Handler of the calls to <c>--server</c> (null: the default); tests serve contract fixtures through it.</param>
    /// <param name="getEnv">Environment lookup (null: the process's); the API key is read through it.</param>
    /// <returns>The exit code (<see cref="ExitCodes"/>).</returns>
    /// <remarks>
    /// With <c>--json</c>, a failure that produces no result writes <c>{"error": {"code", "exit_code", "message"}}</c> on stdout
    /// (codes: <c>usage</c>, <c>strict</c>, <c>store_busy</c>, <c>store_owned</c>, <c>ledger_unreadable</c>, <c>server_error</c>,
    /// <c>server_unreachable</c>, <c>symbol_not_found</c>, <c>error</c>, <c>internal</c>), with a non-zero exit; the
    /// human-readable message still goes to stderr. No message or document ever contains the API key.
    /// </remarks>
    public static int Run(string[] args, TextWriter output, TextWriter error, string? workingDirectory = null,
        HttpMessageHandler? httpHandler = null, Func<string, string?>? getEnv = null)
    {
        bool verbose = args.Contains("--verbose", StringComparer.Ordinal);
        bool json = args.Contains("--json", StringComparer.Ordinal);
        args = args.Where(a => !string.Equals(a, "--verbose", StringComparison.Ordinal)).ToArray();
        var ctx = new CliContext(output, error, workingDirectory ?? Environment.CurrentDirectory)
        {
            HttpHandler = httpHandler, GetEnv = getEnv ?? Environment.GetEnvironmentVariable,
        };

        int Fail(int exit, string code, string message, Exception? ex = null, string? extra = null)
        {
            message = ctx.Redact(message);
            error.WriteLine(message);
            if (extra is not null) error.WriteLine(ctx.Redact(extra));
            if (verbose && ex is not null) error.WriteLine(ctx.Redact(ex.ToString()));
            if (json) ctx.WriteJson(new ErrorResult(new ErrorBody(code, exit, message)));
            return exit;
        }

        if (args.Length == 0) return Fail(ExitCodes.Usage, "usage", "usage error: no verb.", extra: Usage);
        if (args[0] is "--help" or "-h" or "help")
        {
            output.WriteLine(Usage);
            return ExitCodes.Ok;
        }
        if (!Verbs.TryGetValue(args[0], out VerbDef? verb) || !Runners.TryGetValue(args[0], out Func<CliContext, CliArgs, int>? run)) return Fail(ExitCodes.Usage, "usage", $"usage error: unknown verb '{args[0]}'.", extra: Usage);
        try
        {
            CliArgs parsed = CliArgs.Parse(args[0], args.Skip(1), verb.Spec);
            if (parsed.Flag("help"))
            {
                output.WriteLine(verb.Help);
                return ExitCodes.Ok;
            }
            ctx.Json = parsed.Flag("json");
            return run(ctx, parsed);
        }
        catch (UsageException ex)
        {
            return Fail(ExitCodes.Usage, "usage", $"usage error: {ex.Message}", ex, verb.UsageHint);
        }
        catch (StrictFailureException ex)
        {
            return Fail(ExitCodes.StrictFailure, "strict", $"strict: {ex.Message}", ex);
        }
        catch (ChunkStoreLockTimeoutException ex)
        {
            return Fail(ExitCodes.Error, "store_busy", $"error: store busy: {ex.Message}", ex);
        }
        catch (StoreOwnedException ex)
        {
            return Fail(ExitCodes.Error, "store_owned", $"error: {ex.Message}", ex);
        }
        catch (ChunkCacheRequestException ex)
        {
            return Fail(ExitCodes.Error, "server_error", $"error: the server answered {(int)ex.StatusCode}{(ex.Code is null ? "" : " " + ex.Code)}: {ex.Message}", ex);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or OperationCanceledException)
        {
            return Fail(ExitCodes.Error, "server_unreachable", $"error: the server could not be reached ({ex.GetType().Name}: {ex.Message})", ex);
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
