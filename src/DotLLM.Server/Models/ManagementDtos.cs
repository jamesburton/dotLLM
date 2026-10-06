using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

// ─────────────────────────────── settings (#454) ───────────────────────────────

/// <summary>
/// Runtime settings body, returned by <c>GET /v1/settings</c> and echoed by
/// <c>PUT /v1/settings</c>. Deliberately separate from <c>/v1/config</c>, which covers
/// <i>sampling</i> defaults only.
/// </summary>
public sealed record SettingsDto
{
    /// <summary>Server-wide default idle-unload duration. 0 = unload after each use. Negative = never.</summary>
    [JsonPropertyName("keep_alive_seconds")]
    public double KeepAliveSeconds { get; init; }

    /// <summary>Maximum models resident at once, counting the active one.</summary>
    [JsonPropertyName("max_resident_models")]
    public int MaxResidentModels { get; init; }

    /// <summary>Total byte budget across all resident models. 0 = unlimited.</summary>
    [JsonPropertyName("resident_memory_budget_bytes")]
    public long ResidentMemoryBudgetBytes { get; init; }

    /// <summary>Interval between idle-unload sweeps, in seconds.</summary>
    [JsonPropertyName("idle_sweep_interval_seconds")]
    public double IdleSweepIntervalSeconds { get; init; }

    /// <summary>Whether the #454 model-admin write routes are enabled. Read-only (startup flag).</summary>
    [JsonPropertyName("model_admin_api_enabled")]
    public bool ModelAdminApiEnabled { get; init; }

    /// <summary>Whether the LoRA admin write routes are enabled. Read-only (startup flag).</summary>
    [JsonPropertyName("lora_admin_api_enabled")]
    public bool LoraAdminApiEnabled { get; init; }

    /// <summary>Model keys currently disabled via <c>POST /v1/models/disable</c>.</summary>
    [JsonPropertyName("disabled_models")]
    public string[] DisabledModels { get; init; } = [];
}

/// <summary>
/// Partial update body for <c>PUT /v1/settings</c>. Every field is optional; only the fields
/// present in the JSON are applied.
/// </summary>
public sealed record SettingsUpdateRequest
{
    [JsonPropertyName("keep_alive_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAliveSeconds { get; init; }

    [JsonPropertyName("max_resident_models")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxResidentModels { get; init; }

    [JsonPropertyName("resident_memory_budget_bytes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? ResidentMemoryBudgetBytes { get; init; }

    [JsonPropertyName("idle_sweep_interval_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? IdleSweepIntervalSeconds { get; init; }
}

/// <summary>
/// Result of <c>PUT /v1/settings</c>: the settings after the update, which fields were applied
/// live, which would need a restart, and any models evicted as a side effect of a tightened
/// residency budget.
/// </summary>
public sealed record SettingsUpdateResponse
{
    [JsonPropertyName("settings")]
    public required SettingsDto Settings { get; init; }

    /// <summary>Field names that took effect immediately.</summary>
    [JsonPropertyName("applied")]
    public string[] Applied { get; init; } = [];

    /// <summary>Field names that were accepted but need a server restart to take effect.</summary>
    [JsonPropertyName("restart_required")]
    public string[] RestartRequired { get; init; } = [];

    /// <summary>Model keys evicted right away because the new budget no longer fits them.</summary>
    [JsonPropertyName("evicted")]
    public string[] Evicted { get; init; } = [];
}

// ─────────────────────────── unload / enable / disable ─────────────────────────

/// <summary>Request body for <c>POST /v1/models/unload</c>.</summary>
public sealed record ModelUnloadRequest
{
    /// <summary>Model key to unload. Omit to unload the currently-active model.</summary>
    [JsonPropertyName("model")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Model { get; init; }

    /// <summary>When true, unload every resident model (active + stashed). Overrides <see cref="Model"/>.</summary>
    [JsonPropertyName("all")]
    public bool All { get; init; }
}

/// <summary>Response for <c>POST /v1/models/unload</c>.</summary>
public sealed record ModelUnloadResponse
{
    /// <summary><c>unloaded</c>, or <c>not_resident</c> when nothing matched.</summary>
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    /// <summary>Model keys actually unloaded.</summary>
    [JsonPropertyName("unloaded")]
    public string[] Unloaded { get; init; } = [];
}

/// <summary>Request body for <c>POST /v1/models/enable</c> and <c>POST /v1/models/disable</c>.</summary>
public sealed record ModelEnableRequest
{
    /// <summary>Model key (the id reported by <c>GET /v1/models</c>).</summary>
    [JsonPropertyName("model")]
    public required string Model { get; init; }
}

/// <summary>Response for the enable/disable routes.</summary>
public sealed record ModelEnableResponse
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    /// <summary>Whether the model is loadable after this call.</summary>
    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; }

    /// <summary>
    /// True when this model is still the <i>active</i>, serving model even though it was just
    /// disabled. Disabling blocks the next activation; it does not interrupt a loaded model.
    /// </summary>
    [JsonPropertyName("still_loaded")]
    public bool StillLoaded { get; init; }
}

// ──────────────────────────────── devices (#454) ───────────────────────────────

/// <summary>Response for <c>GET /v1/devices</c>.</summary>
public sealed record DeviceListResponse
{
    [JsonPropertyName("backends")]
    public required BackendInfoDto[] Backends { get; init; }
}

/// <summary>One compute backend and the devices it can enumerate.</summary>
public sealed record BackendInfoDto
{
    /// <summary><c>cpu</c>, <c>cuda</c> or <c>vulkan</c>.</summary>
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    /// <summary>Whether the backend's runtime/driver is present on this machine.</summary>
    [JsonPropertyName("available")]
    public bool Available { get; init; }

    /// <summary>Number of devices this backend enumerates (0 when unavailable).</summary>
    [JsonPropertyName("device_count")]
    public int DeviceCount { get; init; }

    /// <summary>
    /// Whether <c>POST /v1/models/load</c> can actually place a model on this backend. The server's
    /// load path dispatches to CPU or CUDA only, so Vulkan reports <c>available</c> but not
    /// <c>servable</c>.
    /// </summary>
    [JsonPropertyName("servable")]
    public bool Servable { get; init; }

    /// <summary>Human-readable explanation, present when something is unavailable or unservable.</summary>
    [JsonPropertyName("note")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Note { get; init; }

    [JsonPropertyName("devices")]
    public DeviceInfoDto[] Devices { get; init; } = [];
}

/// <summary>One enumerated device.</summary>
public sealed record DeviceInfoDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("name")]
    public required string Name { get; init; }

    /// <summary>
    /// The exact string to pass as <c>device</c> on <c>POST /v1/models/load</c> to target this
    /// device, or null when the backend is not servable.
    /// </summary>
    [JsonPropertyName("device_string")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DeviceString { get; init; }

    /// <summary>Total device memory in bytes, when the backend reports it.</summary>
    [JsonPropertyName("total_memory_bytes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? TotalMemoryBytes { get; init; }

    /// <summary>CUDA compute capability ("8.6"), CUDA only.</summary>
    [JsonPropertyName("compute_capability")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ComputeCapability { get; init; }
}

// ───────────────────────────────── pull (#454) ─────────────────────────────────

/// <summary>Request body for <c>POST /v1/models/pull</c>.</summary>
public sealed record ModelPullRequest
{
    /// <summary>HuggingFace repo id, e.g. <c>bartowski/Qwen2.5-3B-Instruct-GGUF</c>.</summary>
    [JsonPropertyName("repo_id")]
    public required string RepoId { get; init; }

    /// <summary>File within the repo, e.g. <c>Qwen2.5-3B-Instruct-Q4_K_M.gguf</c>.</summary>
    [JsonPropertyName("filename")]
    public required string Filename { get; init; }

    /// <summary>Git revision. Defaults to <c>main</c>.</summary>
    [JsonPropertyName("revision")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Revision { get; init; }

    /// <summary>
    /// When true or omitted, the response is an SSE progress stream. When false the call returns
    /// <c>202 Accepted</c> with the job and the caller polls <c>GET /v1/models/pull/{id}</c>.
    /// </summary>
    /// <remarks>
    /// Nullable, with null meaning "stream", rather than a <c>bool</c> defaulting to <c>true</c>:
    /// <see cref="ServerJsonContext"/> is source-generated, and for a type carrying
    /// <c>required</c> members the generated deserializer does <b>not</b> run property
    /// initializers — an omitted <c>stream</c> would silently arrive as <c>false</c>, turning
    /// every default pull into a non-streaming one. Verified, not assumed: see
    /// <c>ModelManagementEndpointsTests.PullRequest_OmittedStream_IsTreatedAsStreaming</c>.
    /// </remarks>
    [JsonPropertyName("stream")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Stream { get; init; }
}

/// <summary>Observable state of a download job.</summary>
public sealed record ModelPullJobDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("repo_id")]
    public required string RepoId { get; init; }

    [JsonPropertyName("filename")]
    public required string Filename { get; init; }

    [JsonPropertyName("revision")]
    public required string Revision { get; init; }

    /// <summary><c>running</c>, <c>completed</c>, <c>failed</c> or <c>cancelled</c>.</summary>
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("bytes_downloaded")]
    public long BytesDownloaded { get; init; }

    /// <summary>Total size once known; null while the server has not reported a length.</summary>
    [JsonPropertyName("total_bytes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? TotalBytes { get; init; }

    /// <summary>0–100, null while <see cref="TotalBytes"/> is unknown.</summary>
    [JsonPropertyName("percent")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Percent { get; init; }

    /// <summary>Failure message, present only when <c>status == "failed"</c>.</summary>
    [JsonPropertyName("error")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Error { get; init; }

    /// <summary>Hub-cache blob path (the single physical copy), set on completion.</summary>
    [JsonPropertyName("blob_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? BlobPath { get; init; }

    /// <summary>Hub-cache snapshot path, set on completion.</summary>
    [JsonPropertyName("snapshot_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SnapshotPath { get; init; }

    /// <summary>
    /// Path under the models directory that <c>POST /v1/models/load</c> resolves, set on
    /// completion. A hardlink to <see cref="BlobPath"/>, not a second copy.
    /// </summary>
    [JsonPropertyName("model_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelPath { get; init; }

    [JsonPropertyName("started_at")]
    public long StartedAt { get; init; }

    [JsonPropertyName("completed_at")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? CompletedAt { get; init; }
}

/// <summary>Response for <c>GET /v1/models/pull</c>.</summary>
public sealed record ModelPullJobListResponse
{
    [JsonPropertyName("jobs")]
    public required ModelPullJobDto[] Jobs { get; init; }
}
