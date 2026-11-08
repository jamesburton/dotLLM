using System.Diagnostics;
using DotLLM.Engine.KvCache;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// Block-granular radix trie of computed KV-cache blocks. Each path from the
/// root corresponds to a sequence of physical blocks in the shared
/// <see cref="KvBlockPool"/>; the tokens that filled each block are the
/// "label" on that edge.
/// </summary>
/// <remarks>
/// <para>The trie keys on per-block token-fingerprints so a lookup walks one
/// step per <see cref="KvBlockPool.BlockSize"/> tokens of prompt. Hash
/// collisions are resolved by comparing the full token sequence stored
/// on the node.</para>
/// <para>Each node tracks a reference count (the number of live sequences
/// that share the block at this node) plus an LRU timestamp. Nodes with
/// <see cref="PrefixTrieNode.RefCount"/> == 0 are candidates for
/// eviction — evicting one releases its block back to the pool.</para>
/// <para>Thread-safety: all mutating methods take a single lock. Lookup
/// + insert are not on the per-token decode hot path (they run once per
/// admission), so this is a deliberate simplicity choice over a lock-free
/// trie.</para>
/// </remarks>
public sealed class PrefixTrie
{
    private readonly KvBlockPool _pool;
    private readonly int _blockSize;
    private readonly int _maxPrefixDepth;
    private readonly PrefixCachePolicy _policy;
    private readonly PrefixTrieNode _root;
    private readonly Dictionary<string, PrefixTrieNode> _namedPrefixes = new(StringComparer.Ordinal);
    private readonly object _lock = new();

    private int _nodeCount;
    private long _hitTokens;
    private long _missTokens;
    private long _lookups;
    private long _tokenClock;   // prompt tokens looked up so far: the own-length rule's clock (#1654), guarded by _lock

    /// <summary>The KV block pool this trie hands out shared blocks from.</summary>
    public KvBlockPool Pool => _pool;

    /// <summary>Number of tokens per block (mirrors <see cref="KvBlockPool.BlockSize"/>).</summary>
    public int BlockSize => _blockSize;

    /// <summary>Maximum prefix depth (tokens) honoured during lookup; 0 = unbounded.</summary>
    public int MaxPrefixDepth => _maxPrefixDepth;

    /// <summary>Number of non-root nodes currently held in the trie.</summary>
    public int NodeCount { get { lock (_lock) return _nodeCount; } }

    /// <summary>Cumulative prompt tokens served from the trie since construction.</summary>
    public long HitTokens => Interlocked.Read(ref _hitTokens);

    /// <summary>Cumulative prompt tokens NOT found in the trie (had to be prefilled).</summary>
    public long MissTokens => Interlocked.Read(ref _missTokens);

    /// <summary>Total lookups performed.</summary>
    public long Lookups => Interlocked.Read(ref _lookups);

    /// <summary>The eviction policy (#1654).</summary>
    public PrefixCachePolicy Policy => _policy;

    /// <summary>Prompt tokens looked up so far: the clock the own-length rule is measured on (#1654).</summary>
    public long TokenClock { get { lock (_lock) return _tokenClock; } }

    /// <summary>
    /// Creates a new prefix trie over the given block pool.
    /// </summary>
    /// <param name="pool">The block pool.</param>
    /// <param name="maxPrefixDepth">Maximum prefix depth in tokens; 0 = unbounded.</param>
    /// <param name="policy">Which zero-refcount leaf <see cref="EvictOneLru"/> reclaims first (<see cref="PrefixCacheConfig.EvictionPolicy"/>).</param>
    public PrefixTrie(KvBlockPool pool, int maxPrefixDepth = 0, PrefixCachePolicy policy = PrefixCachePolicy.Lru)
    {
        ArgumentNullException.ThrowIfNull(pool);
        if (maxPrefixDepth < 0) throw new ArgumentOutOfRangeException(nameof(maxPrefixDepth));
        _pool = pool;
        _blockSize = pool.BlockSize;
        _maxPrefixDepth = maxPrefixDepth;
        _policy = policy;
        _root = new PrefixTrieNode(blockId: -1, parent: null);
    }

    /// <summary>
    /// Walks the trie matching <paramref name="promptTokens"/> in block-sized chunks
    /// and returns the longest matching path. The returned blocks have already had
    /// their refcounts bumped on the caller's behalf; <see cref="Release"/> must be
    /// called once the sequence no longer needs them.
    /// </summary>
    /// <param name="promptTokens">Full prompt tokens for the new sequence.</param>
    /// <param name="matchedBlocks">Receives the matched block IDs in order.</param>
    /// <returns>Number of tokens covered by <paramref name="matchedBlocks"/>.
    /// Always a multiple of <see cref="BlockSize"/> (the trie reuses full blocks only).</returns>
    public int Lookup(ReadOnlySpan<int> promptTokens, List<int> matchedBlocks)
        => Lookup(promptTokens, matchedBlocks, allowApprox: false, out _);

    /// <summary>
    /// Lineage-aware <see cref="Lookup(ReadOnlySpan{int}, List{int})"/>. An exact lookup
    /// (<paramref name="allowApprox"/> false) never walks into an approximate block. With
    /// <paramref name="allowApprox"/> the walk prefers the exact child at each step and falls back to an
    /// approximate one; every block from the first approximate block on counts as approximate.
    /// </summary>
    /// <param name="promptTokens">Full prompt tokens for the new sequence.</param>
    /// <param name="matchedBlocks">Receives the matched block IDs in order (refcounts bumped).</param>
    /// <param name="allowApprox">Whether approximate blocks may be matched.</param>
    /// <param name="exactTokens">Tokens covered by the blocks before the first approximate block
    /// (equals the return value when no approximate block was matched). Only these count as hits.</param>
    public int Lookup(ReadOnlySpan<int> promptTokens, List<int> matchedBlocks, bool allowApprox, out int exactTokens)
    {
        ArgumentNullException.ThrowIfNull(matchedBlocks);
        matchedBlocks.Clear();
        Interlocked.Increment(ref _lookups);
        exactTokens = 0;

        if (promptTokens.Length < _blockSize)
        {
            Interlocked.Add(ref _missTokens, promptTokens.Length);
            return 0;
        }

        long ts = Stopwatch.GetTimestamp();
        int matched = 0;
        int exact = -1;
        int cap = _maxPrefixDepth > 0 ? _maxPrefixDepth : int.MaxValue;
        int maxBlocks = Math.Min(promptTokens.Length / _blockSize, cap / _blockSize);

        lock (_lock)
        {
            // Every prompt looked up advances the clock, hit or miss: the own-length rule counts "other prompt tokens read" (#1654).
            _tokenClock += promptTokens.Length;
            PrefixTrieNode current = _root;
            for (int b = 0; b < maxBlocks; b++)
            {
                int offset = b * _blockSize;
                ReadOnlySpan<int> chunk = promptTokens.Slice(offset, _blockSize);

                var child = FindChild(current, chunk, approx: false);
                if (child is null && allowApprox)
                {
                    child = FindChild(current, chunk, approx: true);
                    if (child is not null && exact < 0)
                        exact = matched;
                }
                if (child is null)
                    break;

                _pool.AddRef(child.BlockId);
                child.RefCount++;
                child.LastTouchedTicks = ts;
                child.LastUseClock = _tokenClock;
                matchedBlocks.Add(child.BlockId);
                matched += _blockSize;
                current = child;
            }
        }

        exactTokens = exact < 0 ? matched : exact;
        Interlocked.Add(ref _hitTokens, exactTokens);
        Interlocked.Add(ref _missTokens, promptTokens.Length - matched);
        return matched;
    }

    /// <summary>
    /// Inserts a block range that completes a prefix in the trie. Inserts <paramref name="newBlocks"/>
    /// at <paramref name="startTokenIndex"/> of <paramref name="fullTokens"/>; each block covers
    /// exactly <see cref="BlockSize"/> tokens. Blocks already in the trie are skipped silently —
    /// the caller's blocks for those positions should already match.
    /// </summary>
    /// <remarks>
    /// The trie holds ONE reference per node (the "trie ref"). The owning
    /// sequence holds its own additional reference acquired during <see cref="Lookup(ReadOnlySpan{int}, List{int})"/>
    /// (or implicitly during insertion of fresh blocks — the caller increments before
    /// calling). The trie's own reference is released only when the node is evicted.
    /// </remarks>
    public int Insert(ReadOnlySpan<int> fullTokens, int startTokenIndex, IReadOnlyList<int> newBlocks)
        => Insert(fullTokens, startTokenIndex, newBlocks, firstApproxBlock: int.MaxValue);

    /// <summary>
    /// Lineage-aware <see cref="Insert(ReadOnlySpan{int}, int, IReadOnlyList{int})"/>: blocks at absolute
    /// block index <paramref name="firstApproxBlock"/> and later are linked as approximate nodes, which
    /// an exact <see cref="Lookup(ReadOnlySpan{int}, List{int}, bool, out int)"/> never matches. An
    /// approximate node and an exact node for the same tokens coexist as siblings, so an exact recompute
    /// after an approximate snapshot is cached exactly.
    /// </summary>
    public int Insert(ReadOnlySpan<int> fullTokens, int startTokenIndex, IReadOnlyList<int> newBlocks, int firstApproxBlock)
    {
        ArgumentNullException.ThrowIfNull(newBlocks);
        ArgumentOutOfRangeException.ThrowIfNegative(firstApproxBlock);
        if (newBlocks.Count == 0) return 0;
        if (startTokenIndex < 0 || startTokenIndex > fullTokens.Length)
            throw new ArgumentOutOfRangeException(nameof(startTokenIndex));
        if (startTokenIndex % _blockSize != 0)
            throw new ArgumentException("Start index must be aligned to block size.", nameof(startTokenIndex));

        long ts = Stopwatch.GetTimestamp();
        int blocksAvailable = (fullTokens.Length - startTokenIndex) / _blockSize;
        int blocksToInsert = Math.Min(newBlocks.Count, blocksAvailable);
        if (blocksToInsert == 0) return 0;
        int linked = 0;

        lock (_lock)
        {
            PrefixTrieNode current = _root;
            // Walk to the insertion point.
            int prefixBlocks = startTokenIndex / _blockSize;
            for (int b = 0; b < prefixBlocks; b++)
            {
                int offset = b * _blockSize;
                ReadOnlySpan<int> chunk = fullTokens.Slice(offset, _blockSize);
                // The sequence may have restored approximate blocks: follow the same preference as Lookup.
                var child = FindChild(current, chunk, approx: false) ?? FindChild(current, chunk, approx: true);
                if (child is null)
                {
                    // The caller is inserting blocks at a position whose prefix is not
                    // in the trie. The caller (PrefixTrieManager.RecordCompletion) only
                    // does this when fresh blocks correspond to suffix tokens past a
                    // matched-prefix boundary; if the prefix path is gone (e.g. evicted
                    // mid-flight) we can't link the suffix. Caller-supplied AddRef on
                    // those blocks is then redundant — release them.
                    for (int j = 0; j < blocksToInsert; j++)
                        _pool.Release(newBlocks[j]);
                    return 0;
                }
                current = child;
            }

            // Insert each new block as a child of the current node, walking down.
            for (int b = 0; b < blocksToInsert; b++)
            {
                int offset = startTokenIndex + b * _blockSize;
                ReadOnlySpan<int> chunk = fullTokens.Slice(offset, _blockSize);
                bool approx = prefixBlocks + b >= firstApproxBlock;
                long key = ChildKey(chunk, approx);
                int newBlockId = newBlocks[b];

                if (current.Children.TryGetValue(key, out var existing) && existing.IsApprox == approx
                    && existing.MatchesTokens(chunk))
                {
                    // Already in trie — the caller's AddRef'd block is a duplicate.
                    // Release the caller's spare ref (we keep the existing trie node's
                    // block and ref) and continue walking the trie.
                    _pool.Release(newBlockId);
                    existing.LastTouchedTicks = ts;
                    existing.LastUseClock = _tokenClock;
                    current = existing;
                    continue;
                }

                // New node: caller's pool ref becomes the trie's "trie ref" on the
                // block. The caller is no longer responsible for that ref.
                var node = new PrefixTrieNode(newBlockId, current)
                {
                    LastTouchedTicks = ts,
                    LastUseClock = _tokenClock,
                    RefCount = 0,
                    IsApprox = approx,
                };
                node.SetTokens(chunk);
                current.Children[key] = node;
                _nodeCount++;
                current = node;
                linked++;
            }
        }
        return linked;
    }

    /// <summary>
    /// Releases the reference acquired during <see cref="Lookup(ReadOnlySpan{int}, List{int})"/> for each block in
    /// <paramref name="blockIds"/>. After release the caller must not touch those blocks.
    /// </summary>
    public void Release(IReadOnlyList<int> blockIds)
    {
        ArgumentNullException.ThrowIfNull(blockIds);
        if (blockIds.Count == 0) return;

        lock (_lock)
        {
            for (int i = 0; i < blockIds.Count; i++)
            {
                int blockId = blockIds[i];
                var node = FindNodeByBlockId(blockId);
                if (node is not null)
                    node.RefCount = Math.Max(0, node.RefCount - 1);
            }
        }

        for (int i = 0; i < blockIds.Count; i++)
            _pool.Release(blockIds[i]);
    }

    /// <summary>
    /// Decrements only the <em>node</em> refcount for each block, leaving the
    /// pool refcount alone. Used when the pool refcount will be released through
    /// another path (e.g. <c>PagedKvCache.Dispose()</c> freeing seeded blocks).
    /// </summary>
    public void ReleaseNodeRefs(IReadOnlyList<int> blockIds)
    {
        ArgumentNullException.ThrowIfNull(blockIds);
        if (blockIds.Count == 0) return;

        lock (_lock)
        {
            for (int i = 0; i < blockIds.Count; i++)
            {
                var node = FindNodeByBlockId(blockIds[i]);
                if (node is not null)
                    node.RefCount = Math.Max(0, node.RefCount - 1);
            }
        }
    }

    /// <summary>
    /// Releases the trie's own reference on a named prefix, allowing it to be evicted.
    /// </summary>
    public bool UnpinNamedPrefix(string prefixId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefixId);
        lock (_lock)
        {
            if (!_namedPrefixes.Remove(prefixId, out var node)) return false;
            // Drop the pin we acquired in RegisterNamedPrefix.
            node.RefCount = Math.Max(0, node.RefCount - 1);
            // The named-prefix pin held a real block-pool ref on every block on the path;
            // walk from the node back to the root, releasing one ref per block.
            for (var cur = node; cur is not null && cur.BlockId >= 0; cur = cur.Parent)
                _pool.Release(cur.BlockId);
            return true;
        }
    }

    /// <summary>
    /// Looks up an existing named prefix without acquiring a new reference.
    /// </summary>
    public NamedPrefixInfo? InspectNamedPrefix(string prefixId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefixId);
        lock (_lock)
        {
            if (!_namedPrefixes.TryGetValue(prefixId, out var node)) return null;
            int depth = 0;
            for (var cur = node; cur is not null && cur.BlockId >= 0; cur = cur.Parent)
                depth++;
            return new NamedPrefixInfo(prefixId, depth * _blockSize, depth);
        }
    }

    /// <summary>
    /// Pins a named prefix at the deepest matching node for <paramref name="tokens"/>.
    /// The trie acquires an additional reference per block in the path, preventing
    /// eviction until <see cref="UnpinNamedPrefix"/> is called.
    /// </summary>
    /// <returns>The depth in tokens of the pinned prefix, or 0 if the prompt did
    /// not yield a matching trie path.</returns>
    public int RegisterNamedPrefix(string prefixId, ReadOnlySpan<int> tokens)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefixId);

        long ts = Stopwatch.GetTimestamp();
        int cap = _maxPrefixDepth > 0 ? _maxPrefixDepth : int.MaxValue;
        int maxBlocks = Math.Min(tokens.Length / _blockSize, cap / _blockSize);

        lock (_lock)
        {
            if (_namedPrefixes.ContainsKey(prefixId))
                throw new InvalidOperationException($"Prefix id '{prefixId}' is already registered.");

            PrefixTrieNode current = _root;
            int matched = 0;
            for (int b = 0; b < maxBlocks; b++)
            {
                int offset = b * _blockSize;
                ReadOnlySpan<int> chunk = tokens.Slice(offset, _blockSize);
                // Named prefixes pin exact paths only.
                var child = FindChild(current, chunk, approx: false);
                if (child is null)
                    break;
                current = child;
                matched += _blockSize;
            }

            if (current.BlockId < 0)
                return 0; // never reached past the root

            // Pin: acquire one extra block-pool ref per block in the path so eviction
            // cannot reclaim them. Also bump the deepest node's refcount so the LRU
            // sweep ignores the entire pinned path.
            for (var cur = current; cur is not null && cur.BlockId >= 0; cur = cur.Parent)
                _pool.AddRef(cur.BlockId);
            current.RefCount++;
            current.LastTouchedTicks = ts;
            _namedPrefixes[prefixId] = current;
            return matched;
        }
    }

    /// <summary>
    /// Evicts a single zero-refcount leaf and returns its block to the pool: the least recently touched under
    /// <see cref="PrefixCachePolicy.Lru"/>; under <see cref="PrefixCachePolicy.Agent"/> the leaf whose path is oldest relative
    /// to its own length (<see cref="PrefixRetention.Age"/>: tokens looked up since the leaf was last used, over the tokens on
    /// its path), so a long conversation's tail outlives shorter paths touched after it (#1654).
    /// </summary>
    /// <returns>The evicted block id, or -1 if no zero-refcount leaf was available.</returns>
    public int EvictOneLru()
    {
        lock (_lock)
        {
            PrefixTrieNode? victim = null;
            if (_policy == PrefixCachePolicy.Agent)
                FindOldestByOwnLengthZeroRefLeaf(_root, _tokenClock, _blockSize, ref victim);
            else
                FindLruZeroRefLeaf(_root, ref victim);
            if (victim is null) return -1;

            int blockId = victim.BlockId;
            UnlinkNode(victim);
            _pool.Release(blockId);
            _nodeCount--;
            return blockId;
        }
    }

    /// <summary>
    /// Drops every node in the trie, releasing the corresponding block-pool refs.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            ClearSubtree(_root);
            _root.Children.Clear();
            _namedPrefixes.Clear();
            _nodeCount = 0;
        }
    }

    private void ClearSubtree(PrefixTrieNode node)
    {
        foreach (var (_, child) in node.Children)
        {
            ClearSubtree(child);
            if (child.BlockId >= 0)
                _pool.Release(child.BlockId);
        }
        node.Children.Clear();
    }

    private static void FindLruZeroRefLeaf(PrefixTrieNode node, ref PrefixTrieNode? best)
    {
        if (node.Children.Count == 0 && node.BlockId >= 0 && node.RefCount == 0)
        {
            if (best is null || node.LastTouchedTicks < best.LastTouchedTicks)
                best = node;
            return;
        }
        foreach (var (_, child) in node.Children)
            FindLruZeroRefLeaf(child, ref best);
    }

    private static void FindOldestByOwnLengthZeroRefLeaf(PrefixTrieNode node, long clock, int blockSize, ref PrefixTrieNode? best)
    {
        if (node.Children.Count == 0 && node.BlockId >= 0 && node.RefCount == 0)
        {
            if (best is null) { best = node; return; }
            double age = PrefixRetention.Age(clock, node.LastUseClock, node.Depth * blockSize);
            double bestAge = PrefixRetention.Age(clock, best.LastUseClock, best.Depth * blockSize);
            if (age > bestAge || age == bestAge && node.LastTouchedTicks < best.LastTouchedTicks)
                best = node;
            return;
        }
        foreach (var (_, child) in node.Children)
            FindOldestByOwnLengthZeroRefLeaf(child, clock, blockSize, ref best);
    }

    private void UnlinkNode(PrefixTrieNode node)
    {
        var parent = node.Parent ?? throw new InvalidOperationException("Cannot unlink root.");
        // Remove the entry whose value == node. Children dict is keyed by chunk hash;
        // scan to find the right key (linear in sibling count, typically tiny).
        long? keyToRemove = null;
        foreach (var (k, v) in parent.Children)
        {
            if (ReferenceEquals(v, node)) { keyToRemove = k; break; }
        }
        if (keyToRemove.HasValue)
            parent.Children.Remove(keyToRemove.Value);
    }

    private PrefixTrieNode? FindNodeByBlockId(int blockId)
    {
        // Walk: we don't keep a reverse index because Release is rare relative to Lookup.
        // O(N) sweep is fine for the MVP; replace with a Dict<int,Node> if profiling shows it.
        return FindNodeByBlockIdRecursive(_root, blockId);
    }

    private static PrefixTrieNode? FindNodeByBlockIdRecursive(PrefixTrieNode node, int blockId)
    {
        if (node.BlockId == blockId) return node;
        foreach (var (_, child) in node.Children)
        {
            var found = FindNodeByBlockIdRecursive(child, blockId);
            if (found is not null) return found;
        }
        return null;
    }

    /// <summary>
    /// Salt that separates approximate children from exact ones in the child map, so both can exist
    /// for the same tokens. The node's <see cref="PrefixTrieNode.IsApprox"/> flag is checked as well.
    /// The salt adds a cross-class collision (<c>H(a) ^ salt == H(b)</c>) to the existing 64-bit token-hash
    /// collision case; both are negligible, and a collision only costs a miss or an overwritten sibling,
    /// never a wrong restore (tokens and the flag are compared on every match).
    /// </summary>
    private const long ApproxKeySalt = unchecked((long)0x9E3779B97F4A7C15UL);

    private static long ChildKey(ReadOnlySpan<int> chunk, bool approx)
        => approx ? HashChunk(chunk) ^ ApproxKeySalt : HashChunk(chunk);

    private static PrefixTrieNode? FindChild(PrefixTrieNode parent, ReadOnlySpan<int> chunk, bool approx)
        => parent.Children.TryGetValue(ChildKey(chunk, approx), out var child)
           && child.IsApprox == approx && child.MatchesTokens(chunk)
            ? child
            : null;

    /// <summary>
    /// FNV-1a 64-bit hash over a token block. The trie also stores the full token
    /// sequence per node so hash collisions are correctness-safe; the hash is purely
    /// a child-dictionary lookup key.
    /// </summary>
    private static long HashChunk(ReadOnlySpan<int> chunk)
    {
        const ulong fnvOffset = 14695981039346656037UL;
        const ulong fnvPrime = 1099511628211UL;
        ulong h = fnvOffset;
        for (int i = 0; i < chunk.Length; i++)
        {
            h ^= (uint)chunk[i];
            h *= fnvPrime;
        }
        return unchecked((long)h);
    }

    /// <summary>Public read-only summary of a registered named prefix.</summary>
    public readonly record struct NamedPrefixInfo(string PrefixId, int Tokens, int Blocks);
}

/// <summary>
/// A single node in the <see cref="PrefixTrie"/>. Internal: the public surface
/// of the trie does not leak node references.
/// </summary>
internal sealed class PrefixTrieNode
{
    public int BlockId { get; }
    public PrefixTrieNode? Parent { get; }
    public Dictionary<long, PrefixTrieNode> Children { get; } = new();

    /// <summary>External references (sequences) using this node's block.</summary>
    public int RefCount { get; set; }
    public long LastTouchedTicks { get; set; }

    /// <summary>The trie's token clock when this node was last inserted or matched (#1654).</summary>
    public long LastUseClock { get; set; }

    /// <summary>Blocks from the root to this node inclusive: the length of the path it ends, in blocks.</summary>
    public int Depth
    {
        get
        {
            int depth = 0;
            for (var cur = this; cur is not null && cur.BlockId >= 0; cur = cur.Parent) depth++;
            return depth;
        }
    }

    /// <summary>Snapshot lineage: the block's K/V is approximate (reuse / recompute, or computed after such a block).</summary>
    public bool IsApprox { get; init; }

    private int[]? _tokens;

    public PrefixTrieNode(int blockId, PrefixTrieNode? parent)
    {
        BlockId = blockId;
        Parent = parent;
    }

    public void SetTokens(ReadOnlySpan<int> tokens) => _tokens = tokens.ToArray();

    public bool MatchesTokens(ReadOnlySpan<int> tokens)
    {
        if (_tokens is null) return false;
        return tokens.SequenceEqual(_tokens);
    }
}
