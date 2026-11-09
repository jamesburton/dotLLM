namespace DotLLM.Cli.Commands;

/// <summary>JSON output DTO for <c>dotllm list --json</c> / <c>dotllm model list --json</c>.</summary>
internal sealed record ModelListJsonResult
{
    public required int Version { get; init; }
    public required ModelListJsonModel[] Models { get; init; }
    public required ModelListJsonProfile[] Profiles { get; init; }
}

/// <summary>One locally available GGUF file.</summary>
internal sealed record ModelListJsonModel
{
    public required string Repo { get; init; }
    public required string Filename { get; init; }
    /// <summary>Quantization parsed from the file name, or <c>-</c> when it carries none.</summary>
    public required string Quant { get; init; }
    public required long SizeBytes { get; init; }
    public required string Path { get; init; }
    /// <summary>Last-write time of the file, ISO 8601.</summary>
    public required string DownloadedAt { get; init; }
}

/// <summary>One model profile (<c>dotllm model create</c>).</summary>
internal sealed record ModelListJsonProfile
{
    public required string Name { get; init; }
    public string? From { get; init; }
    public string? Device { get; init; }
}
