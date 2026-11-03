namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Server switches for the <c>chunk_cache</c> wire convention (<c>--chunk-cache</c>, <c>--chunk-cache-max-chunks</c>,
/// <c>--chunk-store</c>, <c>--chunk-cache-api-keys</c>, <c>--chunk-store-write</c>, <c>--chunk-store-budget</c>,
/// <c>--chunk-store-index-only</c>, <c>--chunk-store-namespace-budget</c>).
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
/// <param name="StoreWrite">
/// <c>--chunk-store-write</c>: the server becomes the store's sole writer (<see cref="ChunkStoreService"/>): it owns the root,
/// keeps the server ledger and encodes admitted misses in the background (spec §5.10). Off: the store is read-only.
/// </param>
/// <param name="StoreBudgetBytes"><c>--chunk-store-budget</c>: the store's byte budget; required (above 0) with
/// <see cref="StoreWrite"/>, 0 otherwise. No unlimited default.</param>
/// <param name="StoreIndexOnly"><c>--chunk-store-index-only</c>: store only chunks an index file references (spec §10).</param>
public sealed record ChunkCacheOptions(bool Enabled, int MaxChunks, string? StoreRoot = null, ChunkCacheKeys? ApiKeys = null,
    bool StoreWrite = false, long StoreBudgetBytes = 0, bool StoreIndexOnly = false)
{
    /// <summary>Default of <c>--chunk-cache-max-chunks</c> (advertised as <c>max_chunks_per_request</c>).</summary>
    public const int DefaultMaxChunks = 4096;

    /// <summary>Upper bound of <c>--chunk-cache-max-chunks</c>: keeps the per-request parse work bounded.</summary>
    public const int AbsoluteMaxChunks = 65536;
    /// <summary>
    /// <c>--chunk-store-namespace-budget</c>: the most bytes of ledger entries one namespace may hold (0 = off, the default).
    /// When set, budget eviction takes only the writing namespace's own rows, so one tenant's writes (prewarm or encode on
    /// miss) never evict another tenant's entries; a namespace over its share evicts its own least-recently-used rows, and an
    /// entry that cannot fit without evicting others is dropped. Off: one shared LRU (own rows first, then the others').
    /// </summary>
    public long StoreNamespaceBudgetBytes { get; init; }

    /// <summary>
    /// The startup warning for a per-namespace budget that is a cap, not a reservation: when the caps of all namespaces
    /// (the keys file's distinct namespaces, or 2 without one: the default namespace and one more writer) add up to more
    /// than the store budget, a namespace can find the store full of the others' entries; it is then blocked
    /// (<c>namespace_budget_blocked</c>) until space frees. Null when no warning applies.
    /// </summary>
    /// <returns>The warning text (no key or namespace in it), or null.</returns>
    public string? NamespaceBudgetWarning()
    {
        if (StoreNamespaceBudgetBytes <= 0 || StoreBudgetBytes <= 0)
            return null;
        int namespaces = ApiKeys?.NamespaceCount ?? 2;
        if ((double)StoreNamespaceBudgetBytes * namespaces <= StoreBudgetBytes)
            return null;
        return $"chunk-store: --chunk-store-namespace-budget is a cap, not a reservation: {namespaces} namespace(s) x {StoreNamespaceBudgetBytes} bytes "
            + $"exceed --chunk-store-budget ({StoreBudgetBytes} bytes), so a namespace can find the store full of the others' entries and is then "
            + "refused (namespace_budget_blocked) until space frees. Only namespaces x namespace budget <= store budget (minus CLI entries) is a real quota.";
    }

    /// <summary>
    /// With <c>--chunk-store-write</c>: takes ownership of the store before the model loads (<c>dotllm serve</c>), so a second
    /// writer fails fast; the writer started at map time uses the claim. No-op otherwise.
    /// </summary>
    /// <exception cref="InvalidServerOptionException">Another server owns the store, or it cannot be created or locked.</exception>
    public void ClaimStoreOwnership() => ChunkStoreService.ClaimOwnership(this);

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
    /// <param name="storeWrite"><c>--chunk-store-write</c> (needs <c>--chunk-store</c> and <c>--chunk-store-budget</c>).</param>
    /// <param name="storeBudget"><c>--chunk-store-budget</c>: a byte size such as <c>20GiB</c> (needs <c>--chunk-store-write</c>).</param>
    /// <param name="storeIndexOnly"><c>--chunk-store-index-only</c> (needs <c>--chunk-store-write</c>).</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxChunks"/> is below 1 or above <see cref="AbsoluteMaxChunks"/>.</exception>
    /// <param name="storeNamespaceBudget"><c>--chunk-store-namespace-budget</c>: a byte size (needs <c>--chunk-store-write</c>); null = off.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxChunks"/> is below 1.</exception>
    /// <exception cref="ArgumentException">A store or keys file is given without <c>--chunk-cache</c>, the keys file is
    /// unreadable or invalid (<see cref="ChunkCacheKeys.Load"/>; the message names the line, never a key), or a store-write
    /// option is given without what it needs (the budget is required with <c>--chunk-store-write</c>).</exception>
    public static ChunkCacheOptions Create(bool enabled, int? maxChunks, string? storeRoot = null, string? apiKeysFile = null, bool allowShortKeys = false,
        bool storeWrite = false, string? storeBudget = null, bool storeIndexOnly = false, string? storeNamespaceBudget = null)
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
        bool hasBudget = !string.IsNullOrWhiteSpace(storeBudget);
        bool hasNamespaceBudget = !string.IsNullOrWhiteSpace(storeNamespaceBudget);
        if (!storeWrite && (hasBudget || storeIndexOnly || hasNamespaceBudget))
            throw new ArgumentException((hasBudget ? "--chunk-store-budget" : storeIndexOnly ? "--chunk-store-index-only" : "--chunk-store-namespace-budget")
                + " needs --chunk-store-write (without it the store is read-only).", nameof(storeWrite));
        if (string.IsNullOrWhiteSpace(storeRoot))
        {
            if (storeWrite)
                throw new ArgumentException("--chunk-store-write needs --chunk-store <dir> (the store the server writes).", nameof(storeRoot));
            return new ChunkCacheOptions(enabled, max, null, keys);
        }
        if (!enabled)
            throw new ArgumentException("--chunk-store needs --chunk-cache (the store is only read for chunk_cache requests).", nameof(storeRoot));
        long budget = 0;
        if (storeWrite)
        {
            if (!hasBudget)
                throw new ArgumentException("--chunk-store-write needs --chunk-store-budget <bytes> (e.g. 20GiB): the store has no unlimited default.", nameof(storeBudget));
            budget = ServerOptions.ParseByteSize(storeBudget, "--chunk-store-budget");
            if (budget <= 0)
                throw new ArgumentException("--chunk-store-budget must be above 0 bytes.", nameof(storeBudget));
        }
        long namespaceBudget = 0;
        if (hasNamespaceBudget)
        {
            namespaceBudget = ServerOptions.ParseByteSize(storeNamespaceBudget, "--chunk-store-namespace-budget");
            if (namespaceBudget <= 0)
                throw new ArgumentException("--chunk-store-namespace-budget must be above 0 bytes.", nameof(storeNamespaceBudget));
            if (namespaceBudget >= budget)
                throw new ArgumentException("--chunk-store-namespace-budget must be below --chunk-store-budget (it is a per-namespace cap; "
                    + "at or above the store budget it would let one namespace fill the whole store).", nameof(storeNamespaceBudget));
        }
        string root;
        try
        {
            // Resolved once, at startup, against the working directory of that moment (not lazily per request).
            root = Path.GetFullPath(storeRoot);
        }
        catch (Exception ex) when (ex is NotSupportedException or IOException or System.Security.SecurityException)
        {
            throw new ArgumentException($"--chunk-store: not a usable path ({ex.GetType().Name}).", nameof(storeRoot), ex);
        }
        return new ChunkCacheOptions(enabled, max, root, keys, storeWrite, budget, storeIndexOnly) { StoreNamespaceBudgetBytes = namespaceBudget };
    }
}
