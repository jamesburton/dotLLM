using System.Text.Json.Serialization;

namespace DotLLM.Tray.Api;

// Client-side mirrors of the DotLLM.Server management contract (#454).
//
// Deliberately mirrored rather than shared: DotLLM.Server pulls in Microsoft.AspNetCore.App,
// DotLLM.Engine and every backend. A tray app is an HTTP *client* and must not link the engine —
// that is the same boundary the "#455 tray is a client" rule draws. The cost of mirroring is
// drift, and drift is guarded by TrayDtoContractTests, which round-trips every server DTO through
// this context field-for-field.

/// <summary>A single entry of <c>GET /v1/models</c> — a resident model and its lifecycle state.</summary>
public sealed record TrayModelInfo
{
    /// <summary>Model key. The same id the enable/disable/unload routes take.</summary>
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    /// <summary>Whether this model is wired to the live inference path right now.</summary>
    [JsonPropertyName("is_active")]
    public bool IsActive { get; init; }

    /// <summary>Seconds since this model last served a request.</summary>
    [JsonPropertyName("idle_seconds")]
    public double IdleSeconds { get; init; }

    /// <summary>Effective keep-alive in seconds. Negative = never auto-unload.</summary>
    [JsonPropertyName("keep_alive_seconds")]
    public double KeepAliveSeconds { get; init; }

    /// <summary>Seconds until auto-unload, or null when the keep-alive never expires.</summary>
    [JsonPropertyName("expires_in_seconds")]
    public double? ExpiresInSeconds { get; init; }

    /// <summary>Approximate resident footprint in bytes.</summary>
    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }
}

/// <summary>Response body of <c>GET /v1/models</c>.</summary>
public sealed record TrayModelList
{
    /// <summary>Resident models.</summary>
    [JsonPropertyName("data")]
    public TrayModelInfo[]? Data { get; init; }
}

/// <summary>A locally downloaded GGUF, from <c>GET /v1/models/available</c>.</summary>
public sealed record TrayAvailableModel
{
    /// <summary>HuggingFace repo id the file came from.</summary>
    [JsonPropertyName("repo_id")]
    public string? RepoId { get; init; }

    /// <summary>File name within the repo.</summary>
    [JsonPropertyName("filename")]
    public string? Filename { get; init; }

    /// <summary>Absolute path of the local file.</summary>
    [JsonPropertyName("full_path")]
    public string? FullPath { get; init; }

    /// <summary>On-disk size in bytes.</summary>
    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }

    /// <summary>(#454) The model key a load of this file produces — correlates with <see cref="TrayModelInfo.Id"/>.</summary>
    [JsonPropertyName("model_id")]
    public string? ModelId { get; init; }

    /// <summary>
    /// (#454) False when an operator has disabled this key. Nullable because the initializer form
    /// of this default was DROPPED by source generation, so an omitted <c>enabled</c> arrived as
    /// <c>false</c> and every available model showed as disabled. Read
    /// <see cref="IsEnabled"/>, never this.
    /// </summary>
    [JsonPropertyName("enabled")]
    public bool? Enabled { get; init; }

    /// <summary>Whether this model is enabled, treating an absent field as enabled.</summary>
    [JsonIgnore]
    public bool IsEnabled => Enabled ?? true;
}

/// <summary>Response body of <c>GET /v1/models/available</c>.</summary>
public sealed record TrayAvailableModelList
{
    /// <summary>Locally downloaded models.</summary>
    [JsonPropertyName("models")]
    public TrayAvailableModel[]? Models { get; init; }
}

/// <summary>Body of <c>GET /v1/settings</c> and the <c>settings</c> member of the PUT response.</summary>
public sealed record TraySettingsDto
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

    /// <summary>Whether the model-admin write routes are enabled. Read-only (startup flag).</summary>
    [JsonPropertyName("model_admin_api_enabled")]
    public bool ModelAdminApiEnabled { get; init; }

    /// <summary>Whether the LoRA admin write routes are enabled. Read-only (startup flag).</summary>
    [JsonPropertyName("lora_admin_api_enabled")]
    public bool LoraAdminApiEnabled { get; init; }

    /// <summary>Model keys currently disabled.</summary>
    [JsonPropertyName("disabled_models")]
    public string[]? DisabledModels { get; init; }
}

/// <summary>Partial update body for <c>PUT /v1/settings</c>. Null fields are omitted from the JSON.</summary>
public sealed record TraySettingsUpdate
{
    /// <summary>New server-wide default keep-alive, in seconds.</summary>
    [JsonPropertyName("keep_alive_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAliveSeconds { get; init; }

    /// <summary>New maximum resident model count. Must be &gt;= 1.</summary>
    [JsonPropertyName("max_resident_models")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxResidentModels { get; init; }

    /// <summary>New residency byte budget. 0 = unlimited.</summary>
    [JsonPropertyName("resident_memory_budget_bytes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? ResidentMemoryBudgetBytes { get; init; }

    /// <summary>New idle-sweep interval, in seconds. The server accepts 0.1 – 3600.</summary>
    [JsonPropertyName("idle_sweep_interval_seconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? IdleSweepIntervalSeconds { get; init; }
}

/// <summary>Response body of <c>PUT /v1/settings</c>.</summary>
public sealed record TraySettingsUpdateResult
{
    /// <summary>Settings as they stand after the update.</summary>
    [JsonPropertyName("settings")]
    public TraySettingsDto? Settings { get; init; }

    /// <summary>Field names that took effect immediately.</summary>
    [JsonPropertyName("applied")]
    public string[]? Applied { get; init; }

    /// <summary>Field names accepted but needing a server restart.</summary>
    [JsonPropertyName("restart_required")]
    public string[]? RestartRequired { get; init; }

    /// <summary>Model keys evicted as a side effect of a tightened budget.</summary>
    [JsonPropertyName("evicted")]
    public string[]? Evicted { get; init; }
}

/// <summary>Request body for <c>POST /v1/models/unload</c>.</summary>
public sealed record TrayUnloadRequest
{
    /// <summary>Model key to unload. Omit for the currently-active model.</summary>
    [JsonPropertyName("model")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Model { get; init; }

    /// <summary>Unload every resident model. Overrides <see cref="Model"/>.</summary>
    [JsonPropertyName("all")]
    public bool All { get; init; }
}

/// <summary>Response body of <c>POST /v1/models/unload</c>.</summary>
public sealed record TrayUnloadResult
{
    /// <summary><c>unloaded</c>, or <c>not_resident</c> when nothing matched.</summary>
    [JsonPropertyName("status")]
    public string? Status { get; init; }

    /// <summary>Model keys actually unloaded.</summary>
    [JsonPropertyName("unloaded")]
    public string[]? Unloaded { get; init; }
}

/// <summary>Request body for <c>POST /v1/models/enable</c> and <c>/v1/models/disable</c>.</summary>
public sealed record TrayEnableRequest
{
    /// <summary>Model key.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }
}

/// <summary>Response body of the enable/disable routes.</summary>
public sealed record TrayEnableResult
{
    /// <summary>The model key acted on.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>Whether the model is loadable after this call.</summary>
    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; }

    /// <summary>True when a just-disabled model is still the active, serving model.</summary>
    [JsonPropertyName("still_loaded")]
    public bool StillLoaded { get; init; }
}

/// <summary>Request body for <c>POST /v1/models/load</c>. Only the fields the tray exposes.</summary>
public sealed record TrayLoadRequest
{
    /// <summary>Model path, repo id, or key.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>Target device. Use a <see cref="TrayDeviceInfo.DeviceString"/> from <c>GET /v1/devices</c>.</summary>
    [JsonPropertyName("device")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Device { get; init; }

    /// <summary>Transformer layers to offload to the GPU.</summary>
    [JsonPropertyName("gpu_layers")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? GpuLayers { get; init; }

    /// <summary>KV-cache key quantization: <c>f32</c>, <c>q8_0</c>, <c>q4_0</c>.</summary>
    [JsonPropertyName("cache_type_k")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CacheTypeK { get; init; }

    /// <summary>KV-cache value quantization.</summary>
    [JsonPropertyName("cache_type_v")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CacheTypeV { get; init; }

    /// <summary>Per-model keep-alive override in seconds. Null = the server default.</summary>
    [JsonPropertyName("keep_alive")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KeepAlive { get; init; }
}

/// <summary>Response body of <c>POST /v1/models/load</c>.</summary>
public sealed record TrayLoadResult
{
    /// <summary>Always <c>loaded</c> on success.</summary>
    [JsonPropertyName("status")]
    public string? Status { get; init; }

    /// <summary>The model argument that was loaded.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }
}

/// <summary>One enumerated compute device, from <c>GET /v1/devices</c>.</summary>
public sealed record TrayDeviceInfo
{
    /// <summary>Index within its backend.</summary>
    [JsonPropertyName("index")]
    public int Index { get; init; }

    /// <summary>Human-readable device name.</summary>
    [JsonPropertyName("name")]
    public string? Name { get; init; }

    /// <summary>The exact string to pass as <c>device</c> on a load, or null when not servable.</summary>
    [JsonPropertyName("device_string")]
    public string? DeviceString { get; init; }

    /// <summary>Total device memory in bytes, when reported.</summary>
    [JsonPropertyName("total_memory_bytes")]
    public long? TotalMemoryBytes { get; init; }

    /// <summary>CUDA compute capability, CUDA only.</summary>
    [JsonPropertyName("compute_capability")]
    public string? ComputeCapability { get; init; }
}

/// <summary>One compute backend and the devices it enumerates.</summary>
public sealed record TrayBackendInfo
{
    /// <summary><c>cpu</c>, <c>cuda</c> or <c>vulkan</c>.</summary>
    [JsonPropertyName("name")]
    public string? Name { get; init; }

    /// <summary>Whether the backend's runtime is present on this machine.</summary>
    [JsonPropertyName("available")]
    public bool Available { get; init; }

    /// <summary>Number of devices enumerated.</summary>
    [JsonPropertyName("device_count")]
    public int DeviceCount { get; init; }

    /// <summary>
    /// Whether a load can actually place a model here. Vulkan reports available-but-not-servable,
    /// so the tray must gate its device picker on this flag rather than on <see cref="Available"/>.
    /// </summary>
    [JsonPropertyName("servable")]
    public bool Servable { get; init; }

    /// <summary>Explanation, present when something is unavailable or unservable.</summary>
    [JsonPropertyName("note")]
    public string? Note { get; init; }

    /// <summary>The enumerated devices.</summary>
    [JsonPropertyName("devices")]
    public TrayDeviceInfo[]? Devices { get; init; }
}

/// <summary>Response body of <c>GET /v1/devices</c>.</summary>
public sealed record TrayDeviceList
{
    /// <summary>The probed backends.</summary>
    [JsonPropertyName("backends")]
    public TrayBackendInfo[]? Backends { get; init; }
}

/// <summary>Request body for <c>POST /v1/models/pull</c>.</summary>
public sealed record TrayPullRequest
{
    /// <summary>HuggingFace repo id.</summary>
    [JsonPropertyName("repo_id")]
    public string? RepoId { get; init; }

    /// <summary>File within the repo.</summary>
    [JsonPropertyName("filename")]
    public string? Filename { get; init; }

    /// <summary>Git revision. Defaults to <c>main</c> server-side.</summary>
    [JsonPropertyName("revision")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Revision { get; init; }

    /// <summary>
    /// True/omitted streams SSE; false returns 202 and the caller polls. Note the server treats an
    /// <i>omitted</i> value as streaming, so the tray always sends it explicitly.
    /// </summary>
    [JsonPropertyName("stream")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Stream { get; init; }
}

/// <summary>Observable state of a download job.</summary>
public sealed record TrayPullJob
{
    /// <summary>Job id, used by <c>GET</c>/<c>DELETE /v1/models/pull/{id}</c>.</summary>
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    /// <summary>HuggingFace repo id.</summary>
    [JsonPropertyName("repo_id")]
    public string? RepoId { get; init; }

    /// <summary>File within the repo.</summary>
    [JsonPropertyName("filename")]
    public string? Filename { get; init; }

    /// <summary>Git revision.</summary>
    [JsonPropertyName("revision")]
    public string? Revision { get; init; }

    /// <summary><c>running</c>, <c>completed</c>, <c>failed</c> or <c>cancelled</c>.</summary>
    [JsonPropertyName("status")]
    public string? Status { get; init; }

    /// <summary>Bytes transferred so far.</summary>
    [JsonPropertyName("bytes_downloaded")]
    public long BytesDownloaded { get; init; }

    /// <summary>Total size once known.</summary>
    [JsonPropertyName("total_bytes")]
    public long? TotalBytes { get; init; }

    /// <summary>0–100, null while the total is unknown.</summary>
    [JsonPropertyName("percent")]
    public double? Percent { get; init; }

    /// <summary>Failure message, present only when the status is <c>failed</c>.</summary>
    [JsonPropertyName("error")]
    public string? Error { get; init; }

    /// <summary>Hub-cache blob path, set on completion.</summary>
    [JsonPropertyName("blob_path")]
    public string? BlobPath { get; init; }

    /// <summary>Hub-cache snapshot path, set on completion.</summary>
    [JsonPropertyName("snapshot_path")]
    public string? SnapshotPath { get; init; }

    /// <summary>Path under the models directory a load resolves, set on completion.</summary>
    [JsonPropertyName("model_path")]
    public string? ModelPath { get; init; }

    /// <summary>Unix seconds at which the job started.</summary>
    [JsonPropertyName("started_at")]
    public long StartedAt { get; init; }

    /// <summary>Unix seconds at which the job reached a terminal state.</summary>
    [JsonPropertyName("completed_at")]
    public long? CompletedAt { get; init; }

    /// <summary>True when the job has reached a terminal state and will produce no further updates.</summary>
    [JsonIgnore]
    public bool IsTerminal =>
        Status is "completed" or "failed" or "cancelled";
}

/// <summary>Response body of <c>GET /v1/models/pull</c>.</summary>
public sealed record TrayPullJobList
{
    /// <summary>All known jobs, running and terminal.</summary>
    [JsonPropertyName("jobs")]
    public TrayPullJob[]? Jobs { get; init; }
}

/// <summary>A <c>{ "status": ... }</c> body.</summary>
public sealed record TrayStatusResponse
{
    /// <summary>The status string.</summary>
    [JsonPropertyName("status")]
    public string? Status { get; init; }
}

/// <summary>A <c>{ "error": ... }</c> body. The gated routes return this with their 403.</summary>
public sealed record TrayErrorResponse
{
    /// <summary>
    /// The error object. #452 reshaped the server envelope from a flat
    /// <c>{"error":"&lt;string&gt;"}</c> to the SDK-shaped
    /// <c>{"type":"error","error":{message,type,param,code}}</c>; a client still reading a bare
    /// string gets nothing and shows a blank failure, so this follows the nested shape.
    /// </summary>
    [JsonPropertyName("error")]
    public TrayErrorDetail? Error { get; init; }

    /// <summary>The message, or empty when the body was not the expected envelope.</summary>
    [JsonIgnore]
    public string Message => Error?.Message ?? "";
}

/// <summary>The body of a <see cref="TrayErrorResponse"/> — mirrors the server's ErrorDetail.</summary>
public sealed record TrayErrorDetail
{
    /// <summary>Human-readable description. For a gate refusal it names the flag to set.</summary>
    [JsonPropertyName("message")]
    public string? Message { get; init; }

    /// <summary>Error class, e.g. <c>invalid_request_error</c>.</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    /// <summary>Machine-readable code, e.g. <c>admin_api_disabled</c>.</summary>
    [JsonPropertyName("code")]
    public string? Code { get; init; }
}
