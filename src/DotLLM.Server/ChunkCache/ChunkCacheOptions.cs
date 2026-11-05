namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Server switches for the <c>chunk_cache</c> wire convention (<c>--chunk-cache</c>, <c>--chunk-cache-max-chunks</c>,
/// <c>--chunk-store</c>, <c>--chunk-cache-api-keys</c>).
/// Off by default: with <see cref="Enabled"/> false the server never parses an annotation, maps no <c>/v1/chunk-cache</c>
/// route, and request and response bytes are what they were before the convention for any body.
/// </summary>
/// <param name="Enabled">True when <c>--chunk-cache</c> was given.</param>
/// <param name="MaxChunks">Most annotations accepted per request; more is <c>chunk_too_many</c>.</param>
/// <param name="StoreRoot">
/// <c>--chunk-store</c>: the chunk store the server looks chunks up in (read-only here: <c>exact</c> counts hits, <c>shadow</c>
/// plans with them). Null: no store, every located chunk is a miss. A configured root that does not exist is
/// <c>chunk_store_unavailable</c>; the server never creates it.
/// </param>
/// <param name="ApiKeys">
/// <c>--chunk-cache-api-keys</c>: the parsed keys file. Null: the control plane answers loopback callers only, all in the
/// default namespace (<see cref="ChunkCacheAuth"/>).
/// </param>
public sealed record ChunkCacheOptions(bool Enabled, int MaxChunks, string? StoreRoot = null, ChunkCacheKeys? ApiKeys = null)
{
    /// <summary>Default of <c>--chunk-cache-max-chunks</c> (advertised as <c>max_chunks_per_request</c>).</summary>
    public const int DefaultMaxChunks = 4096;

    /// <summary>Upper bound of <c>--chunk-cache-max-chunks</c>: keeps the per-request parse work bounded.</summary>
    public const int AbsoluteMaxChunks = 65536;

    /// <summary>The convention switched off (the default).</summary>
    public static ChunkCacheOptions Disabled { get; } = new(false, DefaultMaxChunks);

    /// <summary>
    /// Builds the options from the CLI values, shared by <c>dotllm serve</c> and <see cref="ServerOptions.Parse"/>.
    /// </summary>
    /// <param name="enabled"><c>--chunk-cache</c>.</param>
    /// <param name="maxChunks"><c>--chunk-cache-max-chunks</c>; null = <see cref="DefaultMaxChunks"/>.</param>
    /// <param name="storeRoot"><c>--chunk-store</c>; null or blank = no store.</param>
    /// <param name="allowShortKeys"><c>--chunk-cache-allow-short-keys</c> (needs a keys file).</param>
    /// <param name="apiKeysFile"><c>--chunk-cache-api-keys</c>; null or blank = none (loopback only). Read and parsed here.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxChunks"/> is below 1 or above <see cref="AbsoluteMaxChunks"/>.</exception>
    /// <exception cref="ArgumentException">A store or keys file is given without <c>--chunk-cache</c>, or the keys file is
    /// unreadable or invalid (<see cref="ChunkCacheKeys.Load"/>; the message names the line, never a key).</exception>
    public static ChunkCacheOptions Create(bool enabled, int? maxChunks, string? storeRoot = null, string? apiKeysFile = null, bool allowShortKeys = false)
    {
        int max = maxChunks ?? DefaultMaxChunks;
        if (max < 1 || max > AbsoluteMaxChunks)
            throw new ArgumentOutOfRangeException(nameof(maxChunks), max, $"--chunk-cache-max-chunks must be between 1 and {AbsoluteMaxChunks}.");
        ChunkCacheKeys? keys = null;
        if (!string.IsNullOrWhiteSpace(apiKeysFile))
        {
            if (!enabled)
                throw new ArgumentException("--chunk-cache-api-keys needs --chunk-cache (the keys guard only the chunk-cache surface).", nameof(apiKeysFile));
            string full;
            try
            {
                full = Path.GetFullPath(apiKeysFile);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or IOException or System.Security.SecurityException)
            {
                // A clean configuration error, never a startup stack trace (ServeCommand.Validate catches ArgumentException).
                throw new ArgumentException($"--chunk-cache-api-keys: not a usable path ({ex.GetType().Name}).", nameof(apiKeysFile), ex);
            }
            keys = ChunkCacheKeys.Load(full, allowShortKeys);
        }
        else if (allowShortKeys)
        {
            throw new ArgumentException("--chunk-cache-allow-short-keys needs --chunk-cache-api-keys.", nameof(allowShortKeys));
        }
        if (string.IsNullOrWhiteSpace(storeRoot))
            return new ChunkCacheOptions(enabled, max, null, keys);
        if (!enabled)
            throw new ArgumentException("--chunk-store needs --chunk-cache (the store is only read for chunk_cache requests).", nameof(storeRoot));
        // Resolved once, at startup, against the working directory of that moment (not lazily per request).
        return new ChunkCacheOptions(enabled, max, Path.GetFullPath(storeRoot), keys);
    }
}
