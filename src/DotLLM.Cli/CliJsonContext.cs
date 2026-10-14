using System.Text.Json.Serialization;
using DotLLM.Cli.Benchmarking;
using DotLLM.Cli.Commands;

namespace DotLLM.Cli;

/// <summary>
/// Source-generated JSON serializer context for CLI output types.
/// Enables Native AOT compilation by eliminating reflection-based serialization.
/// </summary>
[JsonSerializable(typeof(RunJsonResult))]
[JsonSerializable(typeof(BenchJsonResult))]
[JsonSourceGenerationOptions(
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal partial class CliJsonContext : JsonSerializerContext;
