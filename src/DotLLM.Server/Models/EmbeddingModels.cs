using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible <c>POST /v1/embeddings</c> request (issue #451).
/// </summary>
public sealed record EmbeddingRequest
{
    /// <summary>
    /// Input to embed. One of: a string, an array of strings, an array of token ids, or an
    /// array of token-id arrays. Kept as a raw <see cref="JsonElement"/> because the OpenAI
    /// schema is a union; see <c>EmbeddingInputParser</c>.
    /// </summary>
    [JsonPropertyName("input")]
    public JsonElement Input { get; init; }

    /// <summary>Model name. Informational — the server embeds with the currently loaded model.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary><c>float</c> (default) or <c>base64</c>.</summary>
    [JsonPropertyName("encoding_format")]
    public string? EncodingFormat { get; init; }

    /// <summary>Ignored — dotLLM does not support Matryoshka truncation. Rejected when set.</summary>
    [JsonPropertyName("dimensions")]
    public int? Dimensions { get; init; }

    /// <summary>Accepted and ignored (OpenAI abuse-tracking field).</summary>
    [JsonPropertyName("user")]
    public string? User { get; init; }

    /// <summary>
    /// dotLLM extension: pooling strategy override — <c>last</c>, <c>mean</c> or <c>cls</c>.
    /// When omitted the model's GGUF <c>{arch}.pooling_type</c> is used, falling back to
    /// <c>last</c>. Mirrors llama.cpp's <c>--pooling</c> flag.
    /// </summary>
    [JsonPropertyName("pooling")]
    public string? Pooling { get; init; }

    /// <summary>
    /// dotLLM extension: set to <c>false</c> to return un-normalised vectors. Defaults to
    /// <c>true</c> (L2 / Euclidean), matching llama.cpp's <c>--embd-normalize 2</c> default and
    /// OpenAI's unit-norm embeddings.
    /// </summary>
    [JsonPropertyName("normalize")]
    public bool? Normalize { get; init; }
}

/// <summary>One embedding in an <see cref="EmbeddingResponse"/>.</summary>
public sealed record EmbeddingData
{
    /// <summary>Always <c>embedding</c>.</summary>
    [JsonPropertyName("object")]
    public string Object { get; init; } = "embedding";

    /// <summary>Zero-based index of the corresponding input item.</summary>
    [JsonPropertyName("index")]
    public int Index { get; init; }

    /// <summary>
    /// The vector, as a float array (<c>encoding_format: float</c>) or a base64 string
    /// (<c>encoding_format: base64</c> — little-endian IEEE-754 float32, as OpenAI encodes it).
    /// </summary>
    [JsonPropertyName("embedding")]
    public JsonElement Embedding { get; init; }
}

/// <summary>Token accounting for an embeddings request.</summary>
public sealed record EmbeddingUsage
{
    /// <summary>Total tokens across every input item.</summary>
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    /// <summary>Equal to <see cref="PromptTokens"/> — embeddings generate no completion tokens.</summary>
    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

/// <summary>OpenAI-compatible <c>POST /v1/embeddings</c> response.</summary>
public sealed record EmbeddingResponse
{
    /// <summary>Always <c>list</c>.</summary>
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    /// <summary>One entry per input item, in input order.</summary>
    [JsonPropertyName("data")]
    public required IReadOnlyList<EmbeddingData> Data { get; init; }

    /// <summary>Name of the model that produced the embeddings.</summary>
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    /// <summary>Token accounting.</summary>
    [JsonPropertyName("usage")]
    public required EmbeddingUsage Usage { get; init; }
}
