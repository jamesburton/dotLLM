// SPDX-License-Identifier: MIT
using System.Text.Json.Serialization;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// Source-generated JSON metadata for every chunk-cache wire DTO: snake_case names (spec 7.2), nulls omitted when
/// writing so the server applies its own defaults. Trim- and AOT-safe.
/// </summary>
/// <remarks>
/// Guarded by <c>ChunkCacheJsonContextDefaultsTests</c> (dotLLM CLAUDE.md: every new JSON context gets the
/// initializer guard).
/// </remarks>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ChunkCacheCapabilities))]
[JsonSerializable(typeof(SelectRequest))]
[JsonSerializable(typeof(SelectResult))]
[JsonSerializable(typeof(PrewarmRequest))]
[JsonSerializable(typeof(PrewarmResult))]
[JsonSerializable(typeof(StatusRequest))]
[JsonSerializable(typeof(StatusResult))]
[JsonSerializable(typeof(EvictRequest))]
[JsonSerializable(typeof(EvictResult))]
[JsonSerializable(typeof(ExplainResult))]
public sealed partial class ChunkCacheJsonContext : JsonSerializerContext;

/// <summary>The HTTP client's private request bodies and the server's error envelope (same options).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(PrewarmBody))]
[JsonSerializable(typeof(EvictBody))]
[JsonSerializable(typeof(ErrorEnvelope))]
internal sealed partial class ChunkCacheHttpJsonContext : JsonSerializerContext;
