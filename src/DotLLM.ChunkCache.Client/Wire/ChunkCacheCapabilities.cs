using System.Text.Json;

namespace DotLLM.ChunkCache.Client;

// JSON DTO rule (dotLLM CLAUDE.md, #462): these types are deserialized through a source-generated context, so no
// `init` property carries an initializer. Absent fields arrive as null; defaults are resolved in code.

/// <summary>The body of <c>GET /v1/chunk-cache</c> (spec 5.9). Additive fields keep <c>v: 1</c>.</summary>
public sealed record ChunkCacheCapabilities
{
    /// <summary>Convention versions the server speaks; a client for v1 needs <c>1</c> here.</summary>
    public IReadOnlyList<int>? Versions { get; init; }

    /// <summary>Policies the server parses (<c>exact</c>, <c>shadow</c>, <c>balanced</c>, <c>fast</c>, <c>auto</c>).</summary>
    public IReadOnlyList<string>? Policies { get; init; }

    /// <summary>Selectors that run on the active backend (<c>query_relevance</c> appears once a probe exists).</summary>
    public IReadOnlyList<string>? Selectors { get; init; }

    /// <summary>Text-first chunkers the wire accepts (<c>symbol-512</c>; <c>lines-512</c> after E4c).</summary>
    public IReadOnlyList<string>? WireChunkers { get; init; }

    /// <summary>The <c>auto</c> policy's validated combinations; empty today. Element shape not yet fixed by the spec.</summary>
    public IReadOnlyList<JsonElement>? AutoValidated { get; init; }

    /// <summary>The loaded model and whether chunk reuse supports it.</summary>
    public ChunkCacheModelInfo? Model { get; init; }

    /// <summary>Store figures.</summary>
    public ChunkCacheStoreInfo? Store { get; init; }

    /// <summary>The most annotations one request may carry.</summary>
    public int? MaxChunksPerRequest { get; init; }

    /// <summary>Whether <see cref="Versions"/> contains <paramref name="version"/>.</summary>
    public bool SupportsVersion(int version) => Versions is { } v && v.Contains(version);
}

/// <summary><c>capabilities.model</c>.</summary>
public sealed record ChunkCacheModelInfo
{
    /// <summary>The served model id.</summary>
    public string? Id { get; init; }

    /// <summary>Whether chunk reuse supports this model; <see langword="null"/> when absent.</summary>
    public bool? Supported { get; init; }

    /// <summary>Why not, when <see cref="Supported"/> is false (e.g. <c>chunk_model_unsupported</c>).</summary>
    public string? Reason { get; init; }
}

/// <summary><c>capabilities.store</c>.</summary>
public sealed record ChunkCacheStoreInfo
{
    /// <summary>Stored entries.</summary>
    public long? Entries { get; init; }

    /// <summary>Bytes on disk.</summary>
    public long? Bytes { get; init; }

    /// <summary>The disk budget (<c>--chunk-store-budget</c>).</summary>
    public long? BudgetBytes { get; init; }

    /// <summary>K/V element type of stored entries (for example <c>f32</c>).</summary>
    public string? Dtype { get; init; }
}
