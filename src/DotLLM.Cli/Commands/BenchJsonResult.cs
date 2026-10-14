using System.Text.Json.Serialization;

namespace DotLLM.Cli.Commands;

/// <summary>
/// JSON output DTO for <c>dotllm bench</c>. One row per depth checkpoint.
/// Fields mirror <see cref="RunJsonResult"/> so downstream tooling (bench_compare.py,
/// bench_trend.py) can consume both interchangeably.
/// </summary>
internal sealed record BenchJsonResult
{
    /// <summary>Depth (context length / prefill token count) of this checkpoint.</summary>
    public int Depth { get; init; }

    public required string Model { get; init; }
    public required string Architecture { get; init; }
    public required RunUsageDto Usage { get; init; }
    public required RunTimingsDto Timings { get; init; }
    public required RunMemoryDto Memory { get; init; }
}

/// <summary>
/// Envelope emitted by <c>dotllm bench --json</c> when using <c>--depth-list</c>.
/// Contains one <see cref="BenchJsonResult"/> per checkpoint, plus shared metadata.
/// </summary>
internal sealed record BenchJsonEnvelope
{
    public required string Model { get; init; }
    public required string Architecture { get; init; }
    public required int MaxTokens { get; init; }
    public required int Warmup { get; init; }
    public required int[] Depths { get; init; }
    public required BenchJsonResult[] Results { get; init; }
}
