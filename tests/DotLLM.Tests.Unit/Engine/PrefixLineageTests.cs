using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Snapshot lineage for the two KV prefix stores (<see cref="PrefixCache"/> and the
/// <see cref="PrefixTrieManager"/> trie): rows whose K/V came from approximate reuse
/// (chunk assembly / selective recompute) are tagged, an exact lookup never restores
/// them, and <c>cached_tokens</c> counts exact rows only. Nothing in the engine creates
/// approximate rows yet, so these tests create them directly.
/// </summary>
public sealed class PrefixLineageTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int TotalBlocks = 64;
    private const int MaxSeqLen = 64;

    private static int[] Seq(int seed, int count)
    {
        var p = new int[count];
        for (int i = 0; i < count; i++) p[i] = seed * 1000 + i + 1;
        return p;
    }

    private static int[] Extend(int[] prefix, int extra, int seed)
    {
        var p = new int[prefix.Length + extra];
        Array.Copy(prefix, p, prefix.Length);
        for (int i = 0; i < extra; i++) p[prefix.Length + i] = seed * 1000 + i + 1;
        return p;
    }

    private static SimpleKvCache NewSimple() => new(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

    private static PagedKvCacheFactory NewFactory() =>
        new(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: TotalBlocks * BlockSize);

    /// <summary>
    /// Admit, write fake K/V for the rows not restored, then record completion with the given
    /// lineage. Returns the admission so callers can inspect the counts.
    /// </summary>
    private static unsafe (int Cached, int Approx, int Restored) RunSequence(
        PrefixTrieManager mgr, int[] tokens, bool allowApprox = false, int? firstApproxRow = null)
    {
        var admission = mgr.Admit(tokens, MaxSeqLen, allowApprox);
        var cache = admission.Cache;
        int writeStart = admission.RestoredTokens;
        int writeCount = tokens.Length - writeStart;
        int kvStride = NumKvHeads * HeadDim;
        if (writeCount > 0)
        {
            var positions = new int[writeCount];
            for (int i = 0; i < writeCount; i++) positions[i] = writeStart + i;
            float[] k = new float[writeCount * kvStride];
            float[] v = new float[writeCount * kvStride];
            fixed (float* kp = k)
            fixed (float* vp = v)
            {
                var kRef = new DotLLM.Core.Tensors.TensorRef(writeCount, kvStride, DotLLM.Core.Tensors.DType.Float32, -1, (nint)kp);
                var vRef = new DotLLM.Core.Tensors.TensorRef(writeCount, kvStride, DotLLM.Core.Tensors.DType.Float32, -1, (nint)vp);
                for (int layer = 0; layer < NumLayers; layer++)
                    cache.Update(kRef, vRef, positions, layer);
            }
        }

        mgr.RecordCompletion(cache, tokens, firstApproxRow);
        cache.Dispose();
        return (admission.CachedTokens, admission.ApproxRowsReused, admission.RestoredTokens);
    }

    private static (int Cached, int Approx, int Restored) Probe(PrefixTrieManager mgr, int[] tokens, bool allowApprox)
    {
        var admission = mgr.Admit(tokens, MaxSeqLen, allowApprox);
        var result = (admission.CachedTokens, admission.ApproxRowsReused, admission.RestoredTokens);
        mgr.RecordCompletion(admission.Cache, tokens.AsSpan(0, admission.RestoredTokens));
        admission.Cache.Dispose();
        return result;
    }

    // ---------------- ExactLookup_StopsAtFirstApproxRow ----------------

    [Fact]
    public void ExactLookup_StopsAtFirstApproxRow_PrefixCache()
    {
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 10);
        cache.Store(tokens, NewSimple(), firstApproxRow: 6);

        var (entry, matched) = cache.FindMatch(tokens);
        Assert.NotNull(entry);
        Assert.Equal(6, matched);

        // A prompt that diverges before the first approximate row is unaffected.
        var (_, shorter) = cache.FindMatch(Extend(tokens[..4], 3, 9));
        Assert.Equal(4, shorter);

        // The explicit-flag overload agrees and reports no approximate rows.
        var (_, exactMatched) = cache.FindMatch(tokens, allowApprox: false, out int approx);
        Assert.Equal(6, exactMatched);
        Assert.Equal(0, approx);
    }

    [Fact]
    public void ExactLookup_StopsAtFirstApproxRow_PrefixCache_PicksBestCappedEntry()
    {
        // The approximate entry matches 10 rows but only 2 are exact; the exact entry matches 5.
        // An exact lookup must choose the exact entry (5), not the approximate one.
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 10);
        var approxKv = NewSimple();
        var exactKv = NewSimple();
        cache.Store(tokens, approxKv, firstApproxRow: 2);
        cache.Store(Extend(tokens[..5], 5, 7), exactKv);

        var (entry, matched) = cache.FindMatch(tokens);
        Assert.Same(exactKv, entry!.KvCache);
        Assert.Equal(5, matched);
    }

    [Fact]
    public void ExactLookup_StopsAtFirstApproxRow_Trie()
    {
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);

        RunSequence(mgr, tokens, firstApproxRow: 8);   // blocks 0-1 exact, 2-3 approximate

        var (cached, approx, restored) = Probe(mgr, tokens, allowApprox: false);
        Assert.Equal(8, cached);
        Assert.Equal(0, approx);
        Assert.Equal(8, restored);
    }

    [Fact]
    public void ExactLookup_StopsAtFirstApproxRow_Trie_MidBlockRowTaintsItsBlock()
    {
        // Row 6 lies inside block 1 (rows 4-7): the whole block is approximate, the exact match is 1 block.
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);

        RunSequence(mgr, tokens, firstApproxRow: 6);

        var (cached, _, restored) = Probe(mgr, tokens, allowApprox: false);
        Assert.Equal(4, cached);
        Assert.Equal(4, restored);
    }

    // ---------------- ApproxAllowed_RestoresFullMatch ----------------

    [Fact]
    public void ApproxAllowed_RestoresFullMatch_PrefixCache()
    {
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 10);
        cache.Store(tokens, NewSimple(), firstApproxRow: 6);

        var (entry, matched) = cache.FindMatch(tokens, allowApprox: true, out int approx);
        Assert.NotNull(entry);
        Assert.Equal(10, matched);
        Assert.Equal(4, approx);
    }

    [Fact]
    public void ApproxAllowed_RestoresFullMatch_Trie()
    {
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens, firstApproxRow: 8);

        var (cached, approx, restored) = Probe(mgr, tokens, allowApprox: true);
        Assert.Equal(16, restored);
        Assert.Equal(8, cached);
        Assert.Equal(8, approx);
    }

    // ---------------- CachedTokens_CountExactRowsOnly ----------------

    [Fact]
    public void CachedTokens_CountExactRowsOnly_PrefixCache()
    {
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 12);
        cache.Store(tokens, NewSimple(), firstApproxRow: 5);

        var (_, matched) = cache.FindMatch(tokens, allowApprox: true, out int approx);
        int cachedTokens = matched - approx;
        Assert.Equal(5, cachedTokens);
        Assert.Equal(7, approx);
    }

    [Fact]
    public void CachedTokens_CountExactRowsOnly_Trie_HitTokensExactOnly()
    {
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens, firstApproxRow: 8);
        long before = mgr.Trie.HitTokens;

        var (cached, approx, _) = Probe(mgr, tokens, allowApprox: true);

        Assert.Equal(8, cached);
        Assert.Equal(8, approx);
        Assert.Equal(8, mgr.Trie.HitTokens - before);
    }

    [Fact]
    public void CachedTokens_CountExactRowsOnly_Trie_LineagePropagatesThroughCompletion()
    {
        // A request that restored approximate rows computes everything after them against approximate K/V:
        // its new blocks are approximate even though the caller passed no lineage of its own.
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens, firstApproxRow: 8);

        var extended = Extend(tokens, 8, 5);
        var first = RunSequence(mgr, extended, allowApprox: true);
        Assert.Equal(16, first.Restored);

        var (cached, approx, restored) = Probe(mgr, extended, allowApprox: true);
        Assert.Equal(24, restored);
        Assert.Equal(8, cached);
        Assert.Equal(16, approx);

        var exact = Probe(mgr, extended, allowApprox: false);
        Assert.Equal(8, exact.Cached);
        Assert.Equal(8, exact.Restored);
    }

    [Fact]
    public void CachedTokens_CountExactRowsOnly_ExactRecomputeAfterApproxIsStoredExact_Trie()
    {
        // An exact request stops at the approximate block, recomputes the rest exactly, and stores it:
        // the next exact request restores everything, from exact blocks beside the approximate ones.
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens, firstApproxRow: 8);

        var recompute = RunSequence(mgr, tokens);
        Assert.Equal(8, recompute.Cached);

        var (cached, approx, restored) = Probe(mgr, tokens, allowApprox: false);
        Assert.Equal(16, cached);
        Assert.Equal(0, approx);
        Assert.Equal(16, restored);

        // Allowing approximation prefers the exact path, so nothing approximate is reported.
        var preferExact = Probe(mgr, tokens, allowApprox: true);
        Assert.Equal(16, preferExact.Cached);
        Assert.Equal(0, preferExact.Approx);
    }

    [Fact]
    public void CachedTokens_CountExactRowsOnly_ExactRecomputeAfterApproxIsStoredExact_PrefixCache()
    {
        // Same KV-cache instance re-stored (the live-reuse path): the caller's lineage replaces the old tag.
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 10);
        var kv = NewSimple();
        cache.Store(tokens, kv, firstApproxRow: 6);

        cache.Store(tokens, kv);                       // exact recompute from row 6
        var (_, matched) = cache.FindMatch(tokens);
        Assert.Equal(10, matched);

        cache.Store(tokens, kv, firstApproxRow: 3);    // and back to approximate
        (_, matched) = cache.FindMatch(tokens);
        Assert.Equal(3, matched);
        Assert.Equal(1, cache.EntryCount);
    }

    // ---------------- NoApproxRows_BehaviourUnchanged ----------------

    [Fact]
    public void NoApproxRows_BehaviourUnchanged_PrefixCache()
    {
        using var cache = new PrefixCache(4);
        var tokens = Seq(1, 10);
        cache.Store(tokens, NewSimple());

        var (_, plain) = cache.FindMatch(tokens);
        var (_, exact) = cache.FindMatch(tokens, allowApprox: false, out int exactApprox);
        var (_, loose) = cache.FindMatch(tokens, allowApprox: true, out int looseApprox);
        Assert.Equal(10, plain);
        Assert.Equal(10, exact);
        Assert.Equal(10, loose);
        Assert.Equal(0, exactApprox);
        Assert.Equal(0, looseApprox);
    }

    [Fact]
    public void NoApproxRows_BehaviourUnchanged_Trie()
    {
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens);

        foreach (bool allow in new[] { false, true })
        {
            var (cached, approx, restored) = Probe(mgr, tokens, allow);
            Assert.Equal(16, cached);
            Assert.Equal(0, approx);
            Assert.Equal(16, restored);
        }

        var stats = mgr.GetStats();
        Assert.Equal(stats.TotalBlocks, stats.FreeBlocks + stats.NodeCount);
    }

    // ---------------- Eviction_PreservesLineageTag ----------------

    [Fact]
    public void Eviction_PreservesLineageTag_PrefixCache()
    {
        using var cache = new PrefixCache(2);
        var a = Seq(1, 10);
        var b = Seq(2, 10);
        cache.Store(a, NewSimple(), firstApproxRow: 3);
        cache.Store(b, NewSimple());
        cache.FindMatch(a);                             // touch A: B becomes LRU
        cache.Store(Seq(3, 10), NewSimple());           // evicts B

        Assert.Equal(2, cache.EntryCount);
        Assert.Equal(0, cache.FindMatch(b).MatchedTokens);
        Assert.Equal(3, cache.FindMatch(a).MatchedTokens);
        var (_, loose) = cache.FindMatch(a, allowApprox: true, out int approx);
        Assert.Equal(10, loose);
        Assert.Equal(7, approx);
    }

    [Fact]
    public void Eviction_PreservesLineageTag_Trie()
    {
        using var factory = NewFactory();
        using var mgr = new PrefixTrieManager(factory);
        var tokens = Seq(1, 16);
        RunSequence(mgr, tokens, firstApproxRow: 8);    // older: its approximate leaf (block 3) is the LRU leaf
        RunSequence(mgr, Seq(2, 16));                   // unrelated exact path

        Assert.Equal(1, mgr.TryEvict(1));

        var exact = Probe(mgr, tokens, allowApprox: false);
        Assert.Equal(8, exact.Cached);
        Assert.Equal(8, exact.Restored);

        var loose = Probe(mgr, tokens, allowApprox: true);
        Assert.Equal(12, loose.Restored);              // block 3 evicted, block 2 still approximate
        Assert.Equal(8, loose.Cached);
        Assert.Equal(4, loose.Approx);

        var stats = mgr.GetStats();
        Assert.Equal(stats.TotalBlocks, stats.FreeBlocks + stats.NodeCount);
    }
}
