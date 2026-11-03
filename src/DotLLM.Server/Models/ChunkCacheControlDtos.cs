using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

// JSON DTO rule (CLAUDE.md, #462): no `init` property here carries an initializer; every value is set where the DTO is
// built (ChunkCacheEndpoint). These types are in ServerJsonContext, so JsonContextDefaultsTests guards them.

/// <summary>
/// <c>GET /v1/chunk-cache</c> (spec §5.9; contract <c>capabilities.json</c>): what the convention supports on this server.
/// Never carries a namespace, a key or a key digest.
/// </summary>
public sealed record ChunkCacheCapabilitiesDto
{
    /// <summary>Convention versions spoken (<c>[1]</c>).</summary>
    [JsonPropertyName("versions")]
    public required int[] Versions { get; init; }

    /// <summary>Policies parsed (<c>exact</c> and <c>shadow</c> run; the rest degrade to <c>exact</c>).</summary>
    [JsonPropertyName("policies")]
    public required string[] Policies { get; init; }

    /// <summary>Recompute selectors that run on the active backend.</summary>
    [JsonPropertyName("selectors")]
    public required string[] Selectors { get; init; }

    /// <summary>Text-first chunkers a span may name.</summary>
    [JsonPropertyName("wire_chunkers")]
    public required string[] WireChunkers { get; init; }

    /// <summary>Validated <c>auto</c> combinations (empty until Phase I5).</summary>
    [JsonPropertyName("auto_validated")]
    public required string[] AutoValidated { get; init; }

    /// <summary>The loaded model and its chunk-key identity.</summary>
    [JsonPropertyName("model")]
    public required ChunkCacheModelDto Model { get; init; }

    /// <summary>Store figures for the caller.</summary>
    [JsonPropertyName("store")]
    public required ChunkCacheStoreDto Store { get; init; }

    /// <summary><c>--chunk-cache-max-chunks</c>.</summary>
    [JsonPropertyName("max_chunks_per_request")]
    public int MaxChunksPerRequest { get; init; }
}

/// <summary>
/// <c>capabilities.model</c>: the spec's <c>id</c> / <c>supported</c> / <c>reason</c> plus the chunk-key identity (plan Task
/// 14; Task 13 review S4), so a client can find its index (<c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;p8&gt;.&lt;ns&gt;.json</c>)
/// and diagnose a store that never hits. The identity fields are omitted when the model is not supported.
/// </summary>
public sealed record ChunkCacheModelDto
{
    /// <summary>Content model id (<c>GgufIdentity.ModelId</c>); null when unknown.</summary>
    [JsonPropertyName("id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Id { get; init; }

    /// <summary>Whether chunk reuse supports the model on this backend.</summary>
    [JsonPropertyName("supported")]
    public bool Supported { get; init; }

    /// <summary><c>chunk_model_unsupported</c> when not supported, else null (written).</summary>
    [JsonPropertyName("reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Reason { get; init; }

    /// <summary>How <see cref="Id"/> was computed: <c>sampled</c> (the server never uses the full-file hash).</summary>
    [JsonPropertyName("id_kind")]
    public string? IdKind { get; init; }

    /// <summary>Content tokenizer id (<c>GgufIdentity.TokenizerId</c>).</summary>
    [JsonPropertyName("tokenizer_id")]
    public string? TokenizerId { get; init; }

    /// <summary>
    /// The producer identity for the caller's namespace: <c>keyOf([]).Hex</c> (64 hex), which folds in the model
    /// fingerprint, tokenizer, RoPE, backend, K/V dtype, positioning and namespace. Its first 8 digits name the caller's index.
    /// </summary>
    [JsonPropertyName("producer")]
    public string? Producer { get; init; }

    /// <summary>Quantisation label in the fingerprint (<c>ftype-N</c>, or <c>unknown</c>).</summary>
    [JsonPropertyName("quantization")]
    public string? Quantization { get; init; }

    /// <summary>Backend tag in the key and the entry header (<c>cpu</c> / <c>vulkan</c>).</summary>
    [JsonPropertyName("backend")]
    public string? Backend { get; init; }

    /// <summary>Element type of stored K/V (<c>f32</c>).</summary>
    [JsonPropertyName("kv_dtype")]
    public string? KvDtype { get; init; }

    /// <summary>The RoPE parameters in the key.</summary>
    [JsonPropertyName("rope")]
    public ChunkCacheRopeDto? Rope { get; init; }
}

/// <summary>RoPE parameters hashed into every chunk key.</summary>
public sealed record ChunkCacheRopeDto
{
    /// <summary>Rotated dimensions per head.</summary>
    [JsonPropertyName("dim")]
    public int Dim { get; init; }

    /// <summary>Base frequency.</summary>
    [JsonPropertyName("theta")]
    public float Theta { get; init; }

    /// <summary>Pairing (<c>RoPEType</c>: <c>norm</c> adjacent pairs, <c>neox</c> split halves).</summary>
    [JsonPropertyName("type")]
    public required string Type { get; init; }
}

/// <summary>
/// <c>capabilities.store</c>: figures for the caller's own namespace (non-admin callers never see store-wide figures).
/// With <c>--chunk-store-write</c>: the caller namespace's ledger entries and the budget that applies to it; an admin sees
/// store-wide figures (every entry file) and the store budget. 0 without a writer for non-admins (no ledger is kept).
/// </summary>
public sealed record ChunkCacheStoreDto
{
    /// <summary><c>false</c> when the server has no <c>--chunk-store</c> (ruling 52); omitted otherwise (absent = configured).</summary>
    [JsonPropertyName("configured")]
    public bool? Configured { get; init; }

    /// <summary>Entries in the caller's namespace.</summary>
    [JsonPropertyName("entries")]
    public long Entries { get; init; }

    /// <summary>Their bytes on disk.</summary>
    [JsonPropertyName("bytes")]
    public long Bytes { get; init; }

    /// <summary>The disk budget that applies (<c>--chunk-store-budget</c>, or <c>--chunk-store-namespace-budget</c> for a non-admin when set); 0 = none.</summary>
    [JsonPropertyName("budget_bytes")]
    public long BudgetBytes { get; init; }

    /// <summary>Element type of stored K/V.</summary>
    [JsonPropertyName("dtype")]
    public required string Dtype { get; init; }
}

/// <summary>
/// <c>POST /v1/chunk-cache/explain</c> (spec §7.2): a dry run of the request body's annotations. Nothing is generated,
/// stored or counted in <c>/metrics</c>.
/// </summary>
public sealed record ChunkCacheExplainDto
{
    /// <summary>The policy the request would be served with (always <c>exact</c> in v1).</summary>
    [JsonPropertyName("policy_would_apply")]
    public required string PolicyWouldApply { get; init; }

    /// <summary>Why it differs from the requested one; null (written) when they agree.</summary>
    [JsonPropertyName("reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? Reason { get; init; }

    /// <summary>Per-annotation status: authenticated callers only, in their own namespace (spec §10); else omitted.</summary>
    [JsonPropertyName("chunks")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChunkCacheChunkStatusDto[]? Chunks { get; init; }

    /// <summary>The counts (<c>annotated = prefix + hit + miss + ignored</c>; <c>prefix</c> is 0: explain does not probe the prompt cache).</summary>
    [JsonPropertyName("counts")]
    public required ChunkCacheChunksDto Counts { get; init; }

    /// <summary>Predicted times.</summary>
    [JsonPropertyName("predicted_ms")]
    public required ChunkCacheExplainPredictedDto PredictedMs { get; init; }

    /// <summary>Warning codes (§5.8), in order of first occurrence.</summary>
    [JsonPropertyName("warnings")]
    public required string[] Warnings { get; init; }
}

/// <summary><c>explain</c>'s <c>predicted_ms</c> (cost model, the shadow plan's prediction).</summary>
public sealed record ChunkCacheExplainPredictedDto
{
    /// <summary>Predicted time to first token if the stored chunks were reused (the shadow plan).</summary>
    [JsonPropertyName("reuse")]
    public double Reuse { get; init; }

    /// <summary>Predicted prefill of the whole prompt without reuse.</summary>
    [JsonPropertyName("full_prefill")]
    public double FullPrefill { get; init; }
}

/// <summary>
/// <c>POST /v1/chunk-cache/prewarm</c> (spec §7.2): what one call queued. Counts only: no text, hash, key, namespace or path.
/// </summary>
public sealed record ChunkCachePrewarmDto
{
    /// <summary>Texts in the call.</summary>
    [JsonPropertyName("requested")]
    public int Requested { get; init; }

    /// <summary>Texts whose entry the caller's namespace already holds.</summary>
    [JsonPropertyName("already_stored")]
    public int AlreadyStored { get; init; }

    /// <summary>Texts queued for a background encode (or already queued).</summary>
    [JsonPropertyName("queued")]
    public int Queued { get; init; }

    /// <summary>Their tokens (each distinct text once).</summary>
    [JsonPropertyName("queued_tokens")]
    public int QueuedTokens { get; init; }

    /// <summary>Texts refused.</summary>
    [JsonPropertyName("refused")]
    public int Refused { get; init; }

    /// <summary>The distinct refusal reasons (admission reasons such as <c>denied_path</c>, <c>too_large_for_cpu</c>), ordinal order.</summary>
    [JsonPropertyName("refused_reasons")]
    public required string[] RefusedReasons { get; init; }
}

/// <summary>
/// <c>POST /v1/chunk-cache/evict</c> (spec §7.2): what the call removed, or with <c>dry_run</c> would remove. Counts only.
/// </summary>
public sealed record ChunkCacheEvictDto
{
    /// <summary>Entry files deleted (or that would be).</summary>
    [JsonPropertyName("evicted")]
    public int Evicted { get; init; }

    /// <summary>Their bytes.</summary>
    [JsonPropertyName("freed_bytes")]
    public long FreedBytes { get; init; }

    /// <summary>Entry files of forgotten keys kept because an index (or a remaining ledger row) still references them.</summary>
    [JsonPropertyName("kept_referenced")]
    public int KeptReferenced { get; init; }

    /// <summary>Whether this was a dry run (nothing changed).</summary>
    [JsonPropertyName("dry_run")]
    public bool DryRun { get; init; }

    /// <summary>What else the forget removed (additive to the spec's output).</summary>
    [JsonPropertyName("removed")]
    public required ChunkCacheEvictRemovedDto Removed { get; init; }

    /// <summary>Warning codes (<c>chunk_namespace_ignored</c>); omitted when none.</summary>
    [JsonPropertyName("warnings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string[]? Warnings { get; init; }
}

/// <summary><c>evict.removed</c>: the non-file kinds a forget removes (or would remove).</summary>
public sealed record ChunkCacheEvictRemovedDto
{
    /// <summary>Server-ledger rows.</summary>
    [JsonPropertyName("ledger_rows")]
    public int LedgerRows { get; init; }

    /// <summary>The namespace's index files.</summary>
    [JsonPropertyName("index_files")]
    public int IndexFiles { get; init; }

    /// <summary>Sighting rows.</summary>
    [JsonPropertyName("sightings")]
    public int Sightings { get; init; }

    /// <summary>Queued encodes dropped.</summary>
    [JsonPropertyName("queued")]
    public int Queued { get; init; }
}
