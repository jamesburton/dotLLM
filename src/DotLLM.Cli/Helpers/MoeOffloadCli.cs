using DotLLM.Core.Moe;
using Spectre.Console;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// CLI glue for routed-expert offload (issues #1070, #1071, #1083), shared by <c>run</c>, <c>chat</c>, <c>serve</c> and <c>bench</c>:
/// the option names, validation, the options record, and the one-line telemetry summary.
/// </summary>
internal static class MoeOffloadCli
{
    /// <summary>Option name of the mode.</summary>
    public const string ModeOption = "--moe-offload";

    /// <summary>Option name of the cache size.</summary>
    public const string CacheMibOption = "--moe-cache-mib";

    /// <summary>Option name of the policy.</summary>
    public const string PolicyOption = "--moe-cache-policy";

    /// <summary>Option name of the cold-expert mode (#1071).</summary>
    public const string ColdOption = "--moe-cold";

    /// <summary>Option name of the CPU cold-leg thread count (#1071).</summary>
    public const string CpuThreadsOption = "--moe-cpu-threads";

    /// <summary>Option name of llama.cpp's <c>--cpu-moe</c> (#1071).</summary>
    public const string CpuMoeOption = "--cpu-moe";

    /// <summary>Option name of llama.cpp's <c>--n-cpu-moe</c> (#1071; bench's #370 option).</summary>
    public const string NCpuMoeOption = "--n-cpu-moe";

    /// <summary>Option name of the iGPU cold leg's Vulkan device (#1083).</summary>
    public const string IgpuDeviceOption = "--moe-igpu-device";

    /// <summary>Option name of the forced re-calibration (#1071).</summary>
    public const string CalibrateOption = "--moe-calibrate";

    /// <summary>Option name of the context size the offload budget reserves KV cache for (#1071).</summary>
    public const string ContextOption = "--ctx-size|--ctx";

    /// <summary>Returns a user-facing error for invalid values (CLI or their <c>DOTLLM_MOE_*</c> fallbacks), else null.</summary>
    /// <param name="values">The raw option values.</param>
    /// <returns>The error, or null.</returns>
    public static string? Validate(MoeCliValues values) => MoeOffloadOptions.Validate(values);

    /// <summary>The options for the CLI values (each unset one falls back to its <c>DOTLLM_MOE_*</c> variable).</summary>
    /// <param name="values">The raw option values.</param>
    /// <returns>The options.</returns>
    public static MoeOffloadOptions Build(MoeCliValues values) => MoeOffloadOptions.FromCli(values);

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
