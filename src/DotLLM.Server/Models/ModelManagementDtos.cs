using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /v1/models/available</c> — locally downloaded models.
/// </summary>
public sealed record AvailableModelsResponse
{
    [JsonPropertyName("models")]
    public required AvailableModelDto[] Models { get; init; }
}

/// <summary>
/// A locally downloaded GGUF model file.
/// </summary>
public sealed record AvailableModelDto
{
    [JsonPropertyName("repo_id")]
    public required string RepoId { get; init; }

    [JsonPropertyName("filename")]
    public required string Filename { get; init; }

    [JsonPropertyName("full_path")]
    public required string FullPath { get; init; }

    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }

    /// <summary>
    /// (#454) The model key a load of this file produces — the same id <c>GET /v1/models</c>
    /// reports and the enable/disable routes take. Lets a client correlate the two listings
    /// without re-deriving it from the filename.
    /// </summary>
    [JsonPropertyName("model_id")]
    public string ModelId { get; init; } = "";

    /// <summary>(#454) False when an operator has disabled this key via <c>POST /v1/models/disable</c>.</summary>
    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; } = true;
}

/// <summary>
/// Request for <c>POST /v1/models/load</c> — load/swap a model.
/// </summary>
public sealed record ModelLoadRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    /// <summary>
    /// Exact GGUF file to load (as listed by <c>/v1/models/available</c>). When set it wins over <c>model</c>/<c>quant</c> resolution,
    /// which can pick a different file from the same repo directory than the one the UI inspected.
    /// </summary>
    [JsonPropertyName("path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelPath { get; init; }

    [JsonPropertyName("quant")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Quant { get; init; }

    [JsonPropertyName("device")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Device { get; init; }

    /// <summary>Layers on GPU; a negative value (-1) means "all layers" and overrides profile/startup defaults.</summary>
    [JsonPropertyName("gpu_layers")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? GpuLayers { get; init; }

    [JsonPropertyName("cache_type_k")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CacheTypeK { get; init; }

    [JsonPropertyName("cache_type_v")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CacheTypeV { get; init; }

    [JsonPropertyName("cache_window")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? CacheWindow { get; init; }

    [JsonPropertyName("threads")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Threads { get; init; }

    [JsonPropertyName("decode_threads")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? DecodeThreads { get; init; }

    [JsonPropertyName("speculative_model")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SpeculativeModel { get; init; }

    [JsonPropertyName("speculative_k")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? SpeculativeK { get; init; }

    /// <summary>
    /// (#757) Embedded-MTP opt-out/in for this load. Null = keep the server setting (MTP is on by default for models that carry an MTP head);
    /// <c>false</c> disables it (like <c>--no-mtp</c>). Nullable on purpose: an <c>init</c> initializer would be dropped by STJ source-gen.
    /// </summary>
    [JsonPropertyName("mtp")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Mtp { get; init; }

    /// <summary>RoPE scaling override: "none", "linear", "yarn", "ntk", "dynamic". Overrides the GGUF-derived value.</summary>
    [JsonPropertyName("rope_scaling")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? RopeScaling { get; init; }

    /// <summary>RoPE base frequency (theta) override.</summary>
    [JsonPropertyName("rope_freq_base")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? RopeFreqBase { get; init; }

    /// <summary>RoPE scaling factor override (linear/YaRN/NTK).</summary>
    [JsonPropertyName("rope_scale")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? RopeScale { get; init; }

    /// <summary>YaRN original context length override.</summary>
    [JsonPropertyName("yarn_orig_ctx")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? YarnOrigCtx { get; init; }

    /// <summary>YaRN attention factor override.</summary>
    [JsonPropertyName("yarn_attn_factor")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? YarnAttnFactor { get; init; }

    /// <summary>YaRN beta-fast parameter override.</summary>
    [JsonPropertyName("yarn_beta_fast")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? YarnBetaFast { get; init; }

    /// <summary>YaRN beta-slow parameter override.</summary>
    [JsonPropertyName("yarn_beta_slow")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? YarnBetaSlow { get; init; }

    /// <summary>
    /// Idle-unload duration in seconds for this model (#369, ollama parity). Null = use the
    /// server-wide default (<see cref="ServerOptions.KeepAliveSeconds"/>, 5 min). 0 = unload
    /// immediately after each use. Negative = never auto-unload.
    /// </summary>
    [JsonPropertyName("keep_alive")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAlive { get; init; }
}

/// <summary>
/// Response for <c>POST /v1/models/load</c>.
/// </summary>
public sealed record ModelLoadResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }
}
