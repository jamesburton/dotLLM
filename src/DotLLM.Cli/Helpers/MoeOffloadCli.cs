using DotLLM.Core.Moe;
using Spectre.Console;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// CLI glue for routed-expert offload (issue #1070), shared by <c>run</c>, <c>chat</c>, <c>serve</c> and <c>bench</c>: the
/// option text, validation, the options record, and the one-line telemetry summary.
/// </summary>
internal static class MoeOffloadCli
{
    /// <summary>Option name of the mode.</summary>
    public const string ModeOption = "--moe-offload";

    /// <summary>Option name of the cache size.</summary>
    public const string CacheMibOption = "--moe-cache-mib";

    /// <summary>Option name of the policy.</summary>
    public const string PolicyOption = "--moe-cache-policy";

    /// <summary>Returns a user-facing error for invalid values, else null.</summary>
    /// <param name="mode"><c>--moe-offload</c>.</param>
    /// <param name="cacheMib"><c>--moe-cache-mib</c>.</param>
    /// <param name="policy"><c>--moe-cache-policy</c>.</param>
    /// <returns>The error, or null.</returns>
    public static string? Validate(string? mode, string? cacheMib, string? policy)
    {
        string? error = MoeOffloadOptions.Validate(mode, cacheMib, policy);
        if (error is not null) return error;
        // The environment fallbacks are parsed too, so a bad DOTLLM_MOE_* fails before the load rather than inside it.
        try
        {
            MoeOffloadOptions.FromCli(mode, cacheMib, policy);
            return null;
        }
        catch (ArgumentException ex)
        {
            return ex.Message.Split(" (Parameter", 2)[0];
        }
    }

    /// <summary>The options for the three CLI values (each unset one falls back to its <c>DOTLLM_MOE_*</c> variable).</summary>
    /// <param name="mode"><c>--moe-offload</c>.</param>
    /// <param name="cacheMib"><c>--moe-cache-mib</c>.</param>
    /// <param name="policy"><c>--moe-cache-policy</c>.</param>
    /// <returns>The options.</returns>
    public static MoeOffloadOptions Build(string? mode, string? cacheMib, string? policy)
        => MoeOffloadOptions.FromCli(mode, cacheMib, policy);

    /// <summary>
    /// Prints the offload summary for the work since <paramref name="start"/> (nothing when no offloaded model is active).
    /// </summary>
    /// <param name="start">Snapshot taken before the measured work.</param>
    /// <param name="json">Write to stderr (stdout stays pure JSON).</param>
    public static void PrintSummary(MoeTelemetrySnapshot start, bool json)
    {
        if (MoeTelemetry.Snapshot().Since(start).FormatSummary() is not { } line) return;
        if (json) Console.Error.WriteLine(line);
        else AnsiConsole.MarkupLine($"  [dim]{Markup.Escape(line)}[/]");
    }
}
