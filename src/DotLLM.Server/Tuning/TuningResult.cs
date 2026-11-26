using System.Text.Json.Serialization;

namespace DotLLM.Server.Tuning;

// Saved tuning result (#1902, slice 1). JSON DTO rules (CLAUDE.md): every member is `init`-only and nullable with NO initializer,
// because the source-generated context drops initializers on init accessors. Defaults are resolved in code by the consumer.

/// <summary>The settings a tuning run chose. Null means "not tuned, keep the default".</summary>
public sealed record TunedSettings
{
    /// <summary>Canonical device string (<c>cpu</c>, <c>gpu:N</c>, <c>vulkan[:N]</c>); under <c>auto</c> scope it names the winning device.</summary>
    [JsonPropertyName("device")] public string? Device { get; init; }
    /// <summary>CUDA GPU layers.</summary>
    [JsonPropertyName("gpu_layers")] public int? GpuLayers { get; init; }
    /// <summary>MoE offload mode (the <c>--moe-offload</c> value).</summary>
    [JsonPropertyName("moe_offload")] public string? MoeOffload { get; init; }
    /// <summary><c>--n-cpu-moe</c>.</summary>
    [JsonPropertyName("n_cpu_moe")] public int? NCpuMoe { get; init; }
    /// <summary><c>--moe-cache-mib</c>.</summary>
    [JsonPropertyName("moe_cache_mib")] public int? MoeCacheMib { get; init; }
    /// <summary>KV key format (<c>f16</c>, <c>q8_0</c>, <c>q4_0</c>, <c>tq1..tq8</c>).</summary>
    [JsonPropertyName("cache_type_k")] public string? CacheTypeK { get; init; }
    /// <summary>KV value format.</summary>
    [JsonPropertyName("cache_type_v")] public string? CacheTypeV { get; init; }
    /// <summary>Paged KV cache on or off (TurboQuant needs off).</summary>
    [JsonPropertyName("paged")] public bool? Paged { get; init; }
    /// <summary><c>--prefill-chunk-size</c>.</summary>
    [JsonPropertyName("prefill_chunk_size")] public int? PrefillChunkSize { get; init; }
    /// <summary>MTP head on (true) or off (false).</summary>
    [JsonPropertyName("mtp_enabled")] public bool? MtpEnabled { get; init; }
    /// <summary><c>--speculative-k</c>.</summary>
    [JsonPropertyName("speculative_k")] public int? SpeculativeK { get; init; }
    /// <summary><c>--threads</c>.</summary>
    [JsonPropertyName("threads")] public int? Threads { get; init; }
    /// <summary><c>--decode-threads</c>.</summary>
    [JsonPropertyName("decode_threads")] public int? DecodeThreads { get; init; }
    /// <summary><c>--expected-concurrency</c> (serve).</summary>
    [JsonPropertyName("expected_concurrency")] public int? ExpectedConcurrency { get; init; }
    /// <summary><c>--ctx-size</c>.</summary>
    [JsonPropertyName("context_size")] public int? ContextSize { get; init; }
    /// <summary>The context the measurements were taken at (informational; drives the "not measured at that depth" warning).</summary>
    [JsonPropertyName("tuned_context")] public int? TunedContext { get; init; }
    /// <summary>Peak device bytes observed while tuning this placement; the apply-time admission compares it with free memory now.</summary>
    [JsonPropertyName("peak_device_bytes")] public long? PeakDeviceBytes { get; init; }
    /// <summary>Peak host bytes observed while tuning this placement.</summary>
    [JsonPropertyName("peak_host_bytes")] public long? PeakHostBytes { get; init; }
    /// <summary>Reserved for tier-B device crossovers; always null in slice 1.</summary>
    [JsonPropertyName("kernel_crossovers")] public IReadOnlyDictionary<string, long>? KernelCrossovers { get; init; }
}

/// <summary>Identity of one device. Driver fields are stale-level; the rest are miss-level.</summary>
public sealed record DeviceFingerprint
{
    /// <summary><c>cpu</c>, <c>cuda</c> or <c>vulkan</c>.</summary>
    [JsonPropertyName("backend")] public string? Backend { get; init; }
    /// <summary>GPU name.</summary>
    [JsonPropertyName("gpu_name")] public string? GpuName { get; init; }
    /// <summary>PCI vendor id.</summary>
    [JsonPropertyName("pci_vendor_id")] public uint? PciVendorId { get; init; }
    /// <summary>PCI device id.</summary>
    [JsonPropertyName("pci_device_id")] public uint? PciDeviceId { get; init; }
    /// <summary>Total device memory; null on UMA devices (the limit drifts and must not enter the key).</summary>
    [JsonPropertyName("device_memory_bytes")] public long? DeviceMemoryBytes { get; init; }
    /// <summary>CPU model string.</summary>
    [JsonPropertyName("cpu_model")] public string? CpuModel { get; init; }
    /// <summary>Physical cores, when known.</summary>
    [JsonPropertyName("physical_cores")] public int? PhysicalCores { get; init; }
    /// <summary>Logical cores.</summary>
    [JsonPropertyName("logical_cores")] public int? LogicalCores { get; init; }
    /// <summary>Installed RAM.</summary>
    [JsonPropertyName("ram_bytes")] public long? RamBytes { get; init; }
    /// <summary>Driver version (stale-level).</summary>
    [JsonPropertyName("driver_version")] public string? DriverVersion { get; init; }
    /// <summary>Vulkan driver id (stale-level).</summary>
    [JsonPropertyName("driver_id")] public string? DriverId { get; init; }
    /// <summary>Vulkan driver info string (stale-level).</summary>
    [JsonPropertyName("driver_info")] public string? DriverInfo { get; init; }
    /// <summary>Graphics API version (stale-level).</summary>
    [JsonPropertyName("api_version")] public string? ApiVersion { get; init; }
    /// <summary>OS build (stale-level).</summary>
    [JsonPropertyName("os_build")] public string? OsBuild { get; init; }
}

/// <summary>The dotLLM build's performance identity (stale-level): a change can move the optimum without changing the model or device.</summary>
public sealed record PerfIdentity
{
    /// <summary>Informational version without the <c>+commit</c> suffix.</summary>
    [JsonPropertyName("release_version")] public string? ReleaseVersion { get; init; }
    /// <summary>SHA-256 over the (relative path, SHA-256) of the shader and native files in the output directory.</summary>
    [JsonPropertyName("kernel_files_hash")] public string? KernelFilesHash { get; init; }
    /// <summary>SHA-256 over the (name, MVID) of the kernel-bearing managed assemblies.</summary>
    [JsonPropertyName("kernel_assemblies_hash")] public string? KernelAssembliesHash { get; init; }
    /// <summary>.NET runtime version.</summary>
    [JsonPropertyName("runtime_version")] public string? RuntimeVersion { get; init; }
}

/// <summary>The checksummed key of a saved result.</summary>
public sealed record TuningKey
{
    /// <summary>File layout version (miss).</summary>
    [JsonPropertyName("schema_version")] public int? SchemaVersion { get; init; }
    /// <summary>Model hash (miss).</summary>
    [JsonPropertyName("model_hash")] public string? ModelHash { get; init; }
    /// <summary>Digest of overlay, ternary adapters and separate MTP/DFlash heads (miss); null when none.</summary>
    [JsonPropertyName("model_modifiers")] public string? ModelModifiers { get; init; }
    /// <summary>Device scope, which is also part of the file name (miss).</summary>
    [JsonPropertyName("scope")] public string? Scope { get; init; }
    /// <summary>One device, or every device for the <c>auto</c> scope.</summary>
    [JsonPropertyName("devices")] public IReadOnlyList<DeviceFingerprint>? Devices { get; init; }
    /// <summary>Profile name (miss).</summary>
    [JsonPropertyName("profile")] public string? Profile { get; init; }
    /// <summary>Hash of the profile JSON (miss).</summary>
    [JsonPropertyName("profile_hash")] public string? ProfileHash { get; init; }
    /// <summary>KV quality budget in force (miss).</summary>
    [JsonPropertyName("kv_quality_budget")] public double? KvQualityBudget { get; init; }
    /// <summary>Build performance identity (stale).</summary>
    [JsonPropertyName("perf")] public PerfIdentity? Perf { get; init; }
    /// <summary>Hash of the set <c>DOTLLM_*</c> tuning and A/B variables (stale).</summary>
    [JsonPropertyName("env_overrides_hash")] public string? EnvOverridesHash { get; init; }
    /// <summary>The readable <c>name=value</c> list behind the hash, so a stale warning can name what differs.</summary>
    [JsonPropertyName("env_overrides")] public IReadOnlyList<string>? EnvOverrides { get; init; }
    /// <summary>Tuner protocol version (stale).</summary>
    [JsonPropertyName("protocol_version")] public int? ProtocolVersion { get; init; }
}

/// <summary>A saved tuning result file.</summary>
public sealed record TuningResult
{
    /// <summary>The key.</summary>
    [JsonPropertyName("key")] public TuningKey? Key { get; init; }
    /// <summary>The chosen settings.</summary>
    [JsonPropertyName("settings")] public TunedSettings? Settings { get; init; }
    /// <summary>True when the tuning run stopped before finishing.</summary>
    [JsonPropertyName("partial")] public bool? Partial { get; init; }
    /// <summary>UTC time of the run (round-trip ISO 8601).</summary>
    [JsonPropertyName("tuned_at_utc")] public string? TunedAtUtc { get; init; }
    /// <summary>Human-readable report.</summary>
    [JsonPropertyName("report")] public string? Report { get; init; }
    /// <summary>SHA-256 over the canonical JSON of the other members.</summary>
    [JsonPropertyName("checksum")] public string? Checksum { get; init; }
}
