using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible model list response for GET /v1/models.
/// </summary>
public sealed record ModelListResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("data")]
    public required ModelInfoDto[] Data { get; init; }
}

/// <summary>
/// Individual model information.
/// </summary>
public sealed record ModelInfoDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = "dotllm";

    /// <summary>
    /// Whether this is the model directly wired to <see cref="ServerState"/>'s live inference
    /// path right now, versus resident-but-inactive (#369). Always <c>true</c> for the single
    /// model reported by pre-#369 callers (default <c>MaxResidentModels = 1</c>).
    /// </summary>
    [JsonPropertyName("is_active")]
    public bool IsActive { get; init; } = true;

    /// <summary>Seconds since this model last served (or was activated for) a request (#369).</summary>
    [JsonPropertyName("idle_seconds")]
    public double IdleSeconds { get; init; }

    /// <summary>Effective keep-alive for this model, in seconds. Negative = never auto-unload (#369).</summary>
    [JsonPropertyName("keep_alive_seconds")]
    public double KeepAliveSeconds { get; init; }

    /// <summary>Seconds until auto-unload, or null when the keep-alive never expires (#369).</summary>
    [JsonPropertyName("expires_in_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ExpiresInSeconds { get; init; }

    /// <summary>Approximate resident footprint in bytes (#369).</summary>
    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }

    /// <summary>
    /// llama.cpp's per-model <c>meta</c> object (#1289), so clients that read the context window from
    /// <c>data[].meta</c> work unchanged. Null (omitted) when the model is not loaded.
    /// </summary>
    [JsonPropertyName("meta")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ModelMetaDto? Meta { get; init; }
}

/// <summary>
/// llama.cpp's <c>/v1/models</c> <c>meta</c> fields (<c>tools/server/server-context.cpp</c>), plus
/// <c>n_ctx</c> and <c>ftype</c> as published by xhinker/azllm for clients that need the served window (#1289).
/// </summary>
public sealed record ModelMetaDto
{
    /// <summary>llama.cpp <c>llama_vocab_type</c>: 0 none, 1 SPM, 2 BPE, 3 WPM, 4 UGM, 5 RWKV, 6 PLaMo2.</summary>
    [JsonPropertyName("vocab_type")]
    public int VocabType { get; init; }

    /// <summary>Vocabulary size.</summary>
    [JsonPropertyName("n_vocab")]
    public int NVocab { get; init; }

    /// <summary>Context window one request gets, after any memory fit or <c>--ctx-size</c>.</summary>
    [JsonPropertyName("n_ctx")]
    public int NCtx { get; init; }

    /// <summary>Context window the checkpoint was trained with.</summary>
    [JsonPropertyName("n_ctx_train")]
    public int NCtxTrain { get; init; }

    /// <summary>Embedding (hidden) width.</summary>
    [JsonPropertyName("n_embd")]
    public int NEmbd { get; init; }

    /// <summary>Total element count of the tensor directory (llama.cpp <c>llama_model_n_params</c>).</summary>
    [JsonPropertyName("n_params")]
    public long NParams { get; init; }

    /// <summary>Total tensor bytes (llama.cpp <c>llama_model_size</c>).</summary>
    [JsonPropertyName("size")]
    public long Size { get; init; }

    /// <summary>Dominant storage type and measured bits per weight, e.g. <c>"Q8_0 - 8.50 bpw"</c>.</summary>
    [JsonPropertyName("ftype")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Ftype { get; init; }
}
