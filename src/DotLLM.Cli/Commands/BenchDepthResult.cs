namespace DotLLM.Cli.Commands;

/// <summary>
/// JSON output DTO for a single context-depth checkpoint measured by <c>dotllm bench</c>.
/// Compatible with <see cref="RunJsonResult"/> timings/usage DTOs so downstream scripts
/// (e.g., <c>bench_compare.py</c>) can consume both formats with minimal adaptation.
/// </summary>
internal sealed record BenchDepthResult
{
    /// <summary>Context depth (number of prompt tokens) at which decode was measured.</summary>
    public required int Depth { get; init; }

    /// <summary>Model filename.</summary>
    public required string Model { get; init; }

    /// <summary>Model architecture name.</summary>
    public required string Architecture { get; init; }

    /// <summary>Token usage for this checkpoint.</summary>
    public required RunUsageDto Usage { get; init; }

    /// <summary>Timing breakdown for this checkpoint.</summary>
    public required RunTimingsDto Timings { get; init; }
}
