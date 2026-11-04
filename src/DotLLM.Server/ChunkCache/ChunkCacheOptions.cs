namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Server switches for the <c>chunk_cache</c> wire convention (<c>--chunk-cache</c>, <c>--chunk-cache-max-chunks</c>).
/// Off by default: with <see cref="Enabled"/> false the server never parses an annotation, and request and response
/// bytes are what they were before the convention for any body.
/// </summary>
/// <param name="Enabled">True when <c>--chunk-cache</c> was given.</param>
/// <param name="MaxChunks">Most annotations accepted per request; more is <c>chunk_too_many</c>.</param>
public sealed record ChunkCacheOptions(bool Enabled, int MaxChunks)
{
    /// <summary>Default of <c>--chunk-cache-max-chunks</c> (advertised as <c>max_chunks_per_request</c>).</summary>
    public const int DefaultMaxChunks = 4096;

    /// <summary>The convention switched off (the default).</summary>
    public static ChunkCacheOptions Disabled { get; } = new(false, DefaultMaxChunks);

    /// <summary>
    /// Builds the options from the two CLI values, shared by <c>dotllm serve</c> and <see cref="ServerOptions.Parse"/>.
    /// </summary>
    /// <param name="enabled"><c>--chunk-cache</c>.</param>
    /// <param name="maxChunks"><c>--chunk-cache-max-chunks</c>; null = <see cref="DefaultMaxChunks"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxChunks"/> is below 1.</exception>
    public static ChunkCacheOptions Create(bool enabled, int? maxChunks)
    {
        int max = maxChunks ?? DefaultMaxChunks;
        if (max < 1)
            throw new ArgumentOutOfRangeException(nameof(maxChunks), max, "--chunk-cache-max-chunks must be at least 1.");
        return new ChunkCacheOptions(enabled, max);
    }
}
