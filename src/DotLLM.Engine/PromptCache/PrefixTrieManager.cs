using DotLLM.Engine.KvCache;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// High-level orchestrator that wires the <see cref="PrefixTrie"/> in front of
/// <see cref="PagedKvCache"/> for cross-request prefix sharing (Step 37).
/// </summary>
/// <remarks>
/// <para>The manager is the single integration seam that engine and server
/// callers should use. It owns:</para>
/// <list type="bullet">
///   <item>The trie itself.</item>
///   <item>The paged cache factory used to mint per-sequence caches.</item>
///   <item>The mapping from KV-cache instances to their borrowed prefix blocks
///   so completion can release the right refs.</item>
///   <item>The named-prefix registry (<c>POST /v1/prompt-cache/{id}</c>).</item>
/// </list>
/// <para>Concurrency: per-sequence handles are not shared between threads; the
/// trie itself is thread-safe for concurrent <c>Lookup</c>/<c>Insert</c> calls,
/// and the per-handle bookkeeping uses a single lock.</para>
/// </remarks>
public sealed class PrefixTrieManager : IDisposable
{
    private readonly PrefixTrie _trie;
    private readonly PagedKvCacheFactory _factory;
    private readonly PrefixCacheConfig _config;
    private readonly Dictionary<PagedKvCache, AcquiredPrefix> _acquired = new(ReferenceEqualityComparer.Instance);
    private readonly object _acquiredLock = new();
    private bool _disposed;

    private long _hits;
    private long _misses;
    private long _evictionRefusals;

    /// <summary>The underlying trie.</summary>
    public PrefixTrie Trie => _trie;

    /// <summary>The cache factory the manager mints per-sequence caches from.</summary>
    public PagedKvCacheFactory Factory => _factory;

    /// <summary>The configuration this manager was constructed with.</summary>
    public PrefixCacheConfig Config => _config;

    /// <summary>Number of requests that observed a non-zero prefix-cache hit.</summary>
    public long Hits => Interlocked.Read(ref _hits);

    /// <summary>Number of requests that observed no prefix-cache hit.</summary>
    public long Misses => Interlocked.Read(ref _misses);

    /// <summary>
    /// Number of times admission ran into a full block pool that the manager
    /// could not relieve via eviction. The scheduler / engine surfaces this as
    /// a sequence-eviction warning.
    /// </summary>
    public long EvictionRefusals => Interlocked.Read(ref _evictionRefusals);

    /// <summary>
    /// Creates a manager backed by an existing paged factory. The trie is
    /// constructed over the factory's underlying pool.
    /// </summary>
    public PrefixTrieManager(PagedKvCacheFactory factory, PrefixCacheConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(factory);
        _factory = factory;
        _config = config ?? new PrefixCacheConfig();
        _trie = new PrefixTrie(factory.Pool, _config.MaxPrefixDepth, _config.EvictionPolicy);
    }

    /// <summary>
    /// Mints a new <see cref="PagedKvCache"/> for a sequence and seeds it with
    /// the longest matching prefix from the trie. The returned struct reports
    /// the number of tokens already covered by the seeded blocks — the caller
    /// only needs to prefill the suffix beyond that.
    /// </summary>
    /// <param name="promptTokens">Full prompt tokens.</param>
    /// <param name="maxSeqLen">Max sequence length to dimension the staging buffer.</param>
    /// <param name="allowApprox">Whether the request accepts approximate K/V (a request that applies chunk
    /// reuse). Without it the seeded prefix stops at the first approximate block, so exact and plain
    /// requests never restore approximate rows.</param>
    public PrefixAdmission Admit(ReadOnlySpan<int> promptTokens, int maxSeqLen, bool allowApprox = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var cache = _factory.Create(maxSeqLen);

        if (!_config.Enabled || promptTokens.Length < _trie.BlockSize)
        {
            Interlocked.Increment(ref _misses);
            return new PrefixAdmission(cache, cachedTokens: 0);
        }

        var matched = new List<int>(promptTokens.Length / _trie.BlockSize);
        int restoredTokens = TryLookupWithRetry(promptTokens, matched, maxSeqLen, allowApprox, out int exactTokens);

        if (restoredTokens > 0)
        {
            cache.SeedSharedPrefix(matched, restoredTokens);
            lock (_acquiredLock)
            {
                _acquired[cache] = new AcquiredPrefix(matched, firstApproxBlock: exactTokens < restoredTokens
                    ? exactTokens / _trie.BlockSize
                    : int.MaxValue);
            }
        }

        if (exactTokens > 0)
            Interlocked.Increment(ref _hits);
        else
            Interlocked.Increment(ref _misses);

        return new PrefixAdmission(cache, exactTokens, restoredTokens - exactTokens);
    }

    private int TryLookupWithRetry(ReadOnlySpan<int> promptTokens, List<int> matched, int maxSeqLen,
        bool allowApprox, out int exactTokens)
    {
        int cachedTokens = _trie.Lookup(promptTokens, matched, allowApprox, out exactTokens);
        int blockSize = _trie.BlockSize;
        // Cap seeded tokens to whole blocks the cache can hold.
        int maxSeededTokens = (maxSeqLen / blockSize) * blockSize;
        if (cachedTokens > maxSeededTokens && maxSeededTokens >= 0)
        {
            int desiredBlocks = maxSeededTokens / blockSize;
            var surplus = matched.GetRange(desiredBlocks, matched.Count - desiredBlocks);
            _trie.Release(surplus);
            matched.RemoveRange(desiredBlocks, matched.Count - desiredBlocks);
            cachedTokens = matched.Count * blockSize;
            exactTokens = Math.Min(exactTokens, cachedTokens);
        }
        return cachedTokens;
    }

    /// <summary>
    /// Records that a sequence has finished. Pushes its newly-computed blocks
    /// into the trie (for future reuse) and releases its prefix references.
    /// </summary>
    /// <param name="cache">The cache returned by <see cref="Admit"/>.</param>
    /// <param name="fullTokens">All tokens covered by <paramref name="cache"/>
    /// (prompt + generated). Used as the trie key for inserts.</param>
    /// <param name="firstApproxRow">The first row this sequence wrote with approximate K/V (chunk assembly
    /// or recompute), or <c>null</c> when it computed every row exactly. Approximate blocks restored at
    /// admission are tracked here already: every block from the earlier of the two is inserted as
    /// approximate (a block holding any approximate row is approximate).</param>
    public void RecordCompletion(PagedKvCache cache, ReadOnlySpan<int> fullTokens, int? firstApproxRow = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (firstApproxRow is < 0)
            throw new ArgumentOutOfRangeException(nameof(firstApproxRow));

        AcquiredPrefix? acquired;
        lock (_acquiredLock)
        {
            _acquired.Remove(cache, out acquired);
        }

        int blockSize = _trie.BlockSize;
        int prefixTokens = acquired is null ? 0 : acquired.Blocks.Count * blockSize;
        int totalTokens = Math.Min(fullTokens.Length, cache.CurrentLength);

        // Snapshot the full physical block IDs the cache currently holds.
        var allBlocks = new List<int>(totalTokens / blockSize);
        cache.SnapshotFullBlocks(totalTokens, allBlocks);

        // Determine the suffix blocks (those NOT already in the trie).
        int suffixStart = prefixTokens;
        int suffixBlocks = allBlocks.Count - (suffixStart / blockSize);

        if (_config.Enabled && suffixBlocks > 0 && suffixStart + suffixBlocks * blockSize <= fullTokens.Length)
        {
            // Bump refcounts on the new blocks first so they survive the
            // sequence's Dispose call below. The trie's Insert keeps these
            // refs ("trie ref"). The original allocation ref is still held
            // by the cache's block table and will be released by Dispose.
            var freshBlocks = allBlocks.GetRange(allBlocks.Count - suffixBlocks, suffixBlocks);
            for (int i = 0; i < freshBlocks.Count; i++)
                _trie.Pool.AddRef(freshBlocks[i]);
            int firstApproxBlock = Math.Min(
                acquired?.FirstApproxBlock ?? int.MaxValue,
                firstApproxRow is int row ? row / blockSize : int.MaxValue);
            _trie.Insert(fullTokens, suffixStart, freshBlocks, firstApproxBlock);
        }

        // The cache's upcoming Dispose() will release the pool refs that Admit
        // acquired via Lookup (seeded blocks are tracked in the cache's table and
        // get a single Release per block in Free()). We therefore only decrement
        // the trie's per-NODE refcount here — pool refs are handled by Dispose.
        if (acquired is not null)
            _trie.ReleaseNodeRefs(acquired.Blocks);
    }

    /// <summary>
    /// Gives back the admission of a sequence that ended without <see cref="RecordCompletion"/> (failed, cancelled): drops the
    /// bookkeeping entry and the per-node refcounts <see cref="Admit"/> took, so the matched prefix blocks become evictable
    /// again. Nothing is inserted into the trie. A no-op when the cache was never admitted here or was already recorded.
    /// Call before disposing the cache.
    /// </summary>
    public void Abandon(PagedKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (_disposed) return;
        AcquiredPrefix? acquired;
        lock (_acquiredLock)
        {
            _acquired.Remove(cache, out acquired);
        }
        if (acquired is not null)
            _trie.ReleaseNodeRefs(acquired.Blocks);
    }

    /// <summary>
    /// Registers a named prefix from an already-completed sequence. Currently the
    /// HTTP layer calls this with a synthetic single-shot generation that pre-warmed
    /// the trie; this helper just promotes the existing trie path to "pinned".
    /// </summary>
    public int RegisterNamedPrefix(string prefixId, ReadOnlySpan<int> tokens) =>
        _trie.RegisterNamedPrefix(prefixId, tokens);

    /// <summary>Unpins a previously named prefix.</summary>
    public bool UnpinNamedPrefix(string prefixId) => _trie.UnpinNamedPrefix(prefixId);

    /// <summary>Inspects a named prefix without acquiring a reference.</summary>
    public PrefixTrie.NamedPrefixInfo? InspectNamedPrefix(string prefixId) => _trie.InspectNamedPrefix(prefixId);

    /// <summary>
    /// Tries to free at least one block from the trie. Returns the number of
    /// blocks actually freed (>= 0). The scheduler calls this on block-pool
    /// pressure before refusing admission.
    /// </summary>
    public int TryEvict(int desiredBlocks = 1)
    {
        if (!_config.EvictionEnabled || desiredBlocks <= 0) return 0;
        int freed = 0;
        while (freed < desiredBlocks)
        {
            int blockId = _trie.EvictOneLru();
            if (blockId < 0)
            {
                Interlocked.Increment(ref _evictionRefusals);
                break;
            }
            freed++;
        }
        return freed;
    }

    /// <summary>Snapshot summary, useful for telemetry / admin endpoints.</summary>
    public PrefixCacheStats GetStats() => new(
        Enabled: _config.Enabled,
        BlockSize: _trie.BlockSize,
        NodeCount: _trie.NodeCount,
        HitTokens: _trie.HitTokens,
        MissTokens: _trie.MissTokens,
        Lookups: _trie.Lookups,
        Hits: Hits,
        Misses: Misses,
        EvictionRefusals: EvictionRefusals,
        FreeBlocks: _trie.Pool.FreeBlocks,
        TotalBlocks: _trie.Pool.TotalBlocks);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _trie.Clear();
    }

    private sealed class AcquiredPrefix
    {
        public List<int> Blocks { get; }

        /// <summary>Index of the first approximate block among <see cref="Blocks"/>; <c>int.MaxValue</c> if none.</summary>
        public int FirstApproxBlock { get; }

        public AcquiredPrefix(List<int> blocks, int firstApproxBlock)
        {
            Blocks = blocks;
            FirstApproxBlock = firstApproxBlock;
        }
    }
}

/// <summary>Result of <see cref="PrefixTrieManager.Admit"/>.</summary>
public sealed class PrefixAdmission
{
    /// <summary>The newly-minted paged KV-cache, seeded with prefix blocks where possible.</summary>
    public PagedKvCache Cache { get; }

    /// <summary>
    /// Number of prompt tokens already covered by <b>exact</b> trie-shared blocks: the value reported as
    /// <c>cached_tokens</c> / <c>cache_n</c>. Equals <see cref="RestoredTokens"/> unless the admission
    /// allowed approximate blocks.
    /// </summary>
    public int CachedTokens { get; }

    /// <summary>Approximate rows restored after the exact ones (only with <c>allowApprox</c>; otherwise 0).</summary>
    public int ApproxRowsReused { get; }

    /// <summary>All rows seeded into <see cref="Cache"/>: prefill starts here.</summary>
    public int RestoredTokens => CachedTokens + ApproxRowsReused;

    internal PrefixAdmission(PagedKvCache cache, int cachedTokens, int approxRowsReused = 0)
    {
        Cache = cache;
        CachedTokens = cachedTokens;
        ApproxRowsReused = approxRowsReused;
    }
}

/// <summary>Snapshot of prefix-cache statistics, suitable for /v1/prompt-cache responses.</summary>
public readonly record struct PrefixCacheStats(
    bool Enabled,
    int BlockSize,
    int NodeCount,
    long HitTokens,
    long MissTokens,
    long Lookups,
    long Hits,
    long Misses,
    long EvictionRefusals,
    int FreeBlocks,
    int TotalBlocks);
