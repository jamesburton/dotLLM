using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /v1/models/inspect</c> — lightweight GGUF metadata.
/// </summary>
public sealed record ModelInspectResponse
{
    [JsonPropertyName("architecture")]
    public required string Architecture { get; init; }

    [JsonPropertyName("num_layers")]
    public int NumLayers { get; init; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    [JsonPropertyName("num_kv_heads")]
    public int NumKvHeads { get; init; }

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; init; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; init; }

    [JsonPropertyName("max_sequence_length")]
    public int MaxSequenceLength { get; init; }

    [JsonPropertyName("file_size_bytes")]
    public long FileSizeBytes { get; init; }

    /// <summary>
    /// False when a partial <c>gpu_layers</c> split is unsupported for this architecture (the server
    /// then loads it all-on-GPU or on the CPU). No property initializer: STJ source-gen drops them on <c>init</c>.
    /// </summary>
    [JsonPropertyName("supports_partial_offload")]
    public bool SupportsPartialOffload { get; init; }

    /// <summary>
    /// True when the GGUF carries an embedded MTP (Multi-Token Prediction) head that the engine can drive, so <c>serve</c>
    /// enables MTP self-speculation automatically (#757). No property initializer: STJ source-gen drops them on <c>init</c>.
    /// </summary>
    [JsonPropertyName("has_mtp")]
    public bool HasMtp { get; init; }
}
