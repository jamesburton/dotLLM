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

        Store layout: <store>/<2 hex>/<key>.ckv entries (F32 K/V), one index-<chunker>.<model>.json per chunker and model,
        and chunkcache.json (each index's tree root and source). Writers (build, evict --apply) hold <store>/.chunkstore.lock.
        --json prints the result as one JSON document on stdout (snake_case names of the chunkcache_* MCP tool schemas);
        progress and warnings go to stderr.
        """;

    private static readonly Dictionary<string, (IReadOnlyDictionary<string, OptionKind> Spec, string Help, Func<CliContext, CliArgs, int> Run)> Verbs = new(StringComparer.Ordinal)
    {
        ["build"] = (BuildCommand.Spec, BuildCommand.Help, BuildCommand.Run),
        ["select"] = (SelectCommand.Spec, SelectCommand.Help, SelectCommand.Run),
        ["status"] = (StatusCommand.Spec, StatusCommand.Help, StatusCommand.Run),
        ["evict"] = (EvictCommand.Spec, EvictCommand.Help, EvictCommand.Run),
        ["compare"] = (CompareCommand.Spec, CompareCommand.Help, CompareCommand.Run),
    };

    /// <summary>Runs one command line.</summary>
    /// <param name="args">Arguments (verb first).</param>
    /// <param name="output">Standard output.</param>
    /// <param name="error">Standard error.</param>
    /// <param name="workingDirectory">Directory relative paths resolve against (default: the process's).</param>
    /// <returns>The exit code (<see cref="ExitCodes"/>).</returns>
    public static int Run(string[] args, TextWriter output, TextWriter error, string? workingDirectory = null)
    {
        var ctx = new CliContext(output, error, workingDirectory ?? Environment.CurrentDirectory);
        if (args.Length == 0)
        {
            error.WriteLine(Usage);
            return ExitCodes.Usage;
        }
        if (args[0] is "--help" or "-h" or "help")
        {
            output.WriteLine(Usage);
            return ExitCodes.Ok;
        }
        if (!Verbs.TryGetValue(args[0], out var verb))
        {
            error.WriteLine($"error: unknown verb '{args[0]}'.");
            error.WriteLine(Usage);
            return ExitCodes.Usage;
        }
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
            error.WriteLine($"usage error: {ex.Message}");
            error.WriteLine(verb.Help);
            return ExitCodes.Usage;
        }
        catch (StrictFailureException ex)
        {
            error.WriteLine($"strict: {ex.Message}");
            return ExitCodes.StrictFailure;
        }
        catch (TimeoutException ex)
        {
            error.WriteLine($"error: store busy: {ex.Message}");
            return ExitCodes.Error;
        }
        catch (SymbolNotFoundException ex)
        {
            error.WriteLine($"error: {ex.Message}");
            if (ex.Suggestions.Count > 0) error.WriteLine($"  did you mean: {string.Join(", ", ex.Suggestions)}");
            return ExitCodes.Error;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or InvalidDataException
            or ArgumentException or NotSupportedException or GitException or KeyNotFoundException or FormatException)
        {
            error.WriteLine($"error: {ex.Message}");
            return ExitCodes.Error;
        }
    }
}
