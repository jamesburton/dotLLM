using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;
using Xunit;
using static DotLLM.Tests.Unit.Engine.PrefixCacheFailedRequestTests;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1591: <see cref="IPrefixReusableKvCache"/> (the Vulkan single-request caches) reuses the longest common prefix of two
/// sequential prompts through <see cref="PrefixCache"/>. Runs the deterministic CPU qwen3 behind a delegating cache that exposes
/// only the interface, so the generic path is exercised without a GPU; a wrong reused row changes the greedy continuation.
/// </summary>
public sealed class PrefixCacheReusableKvTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private const int MaxTokens = 6;
    private readonly TinyQwen _tiny;

    public PrefixCacheReusableKvTests(TinyQwen tiny) => _tiny = tiny;

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13) % (TinyQwen.Vocab - 2);
        return ids;
    }

    private static readonly int[] A = Ids(24, 3);
    private static int[] Fork(int shared) => [.. A[..shared], .. Ids(12, 11)];

    private static string Text(int[] ids) => string.Join(",", ids);
    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = MaxTokens };

    private sealed class Tok(int v) : ITokenizer
    {
        public int VocabSize => v;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(int.Parse).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString();
    }

    /// <summary>
    /// Exposes a SimpleKvCache only through the interfaces a device cache offers (never as a SimpleKvCache). <paramref name="growLimit"/>:
    /// the largest <see cref="IKvCache.MaxLength"/> a <see cref="IKvCache.TryGrowTo"/> may reach (0 = never grows, the interface default),
    /// standing in for a device cache whose admission refuses a growth that would not fit in free device memory (issue #1781).
    /// </summary>
    private sealed class ReusableKv(SimpleKvCache inner, bool exclusive, bool canTruncate, int growLimit = 0, Func<bool>? roomToGrow = null)
        : IPrefixReusableKvCache, IPerLayerKvCache, IDeviceResidentKvCache
    {
        public bool Disposed;
        public long DeviceBytes { get; set; }
        public bool Exclusive = exclusive;
        public List<int> GrowthRequests { get; } = [];
        public bool TryGrowTo(int maxLength)
        {
            if (maxLength <= inner.MaxLength) return true;
            GrowthRequests.Add(maxLength);
            return (roomToGrow?.Invoke() ?? true) && maxLength <= growLimit && inner.TryGrowTo(maxLength);
        }
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public int LayerCount => ((IPerLayerKvCache)inner).LayerCount;
        public int KvStrideOf(int l) => ((IPerLayerKvCache)inner).KvStrideOf(l);
        public bool HasExclusiveStorage => Exclusive;
        public bool CanTruncateTo(int length) => canTruncate && inner.CanTruncateTo(length);
        public void Rollback(int length) => inner.Rollback(length);
        public void Update(ITensor k, ITensor v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public void Update(TensorRef k, TensorRef v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public ITensor GetKeys(int l) => inner.GetKeys(l);
        public ITensor GetValues(int l) => inner.GetValues(l);
        public TensorRef GetKeysRef(int l) => inner.GetKeysRef(l);
        public TensorRef GetValuesRef(int l) => inner.GetValuesRef(l);
        public void Dispose() { Disposed = true; inner.Dispose(); }
    }

    /// <summary>A cache type the engine can never adopt again (no <see cref="IPrefixReusableKvCache"/>).</summary>
    private sealed class OpaqueKv(SimpleKvCache inner) : IKvCache, IPerLayerKvCache
    {
        public bool Disposed;
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public int LayerCount => ((IPerLayerKvCache)inner).LayerCount;
        public int KvStrideOf(int l) => ((IPerLayerKvCache)inner).KvStrideOf(l);
        public void Rollback(int length) => inner.Rollback(length);
        public void Update(ITensor k, ITensor v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public void Update(TensorRef k, TensorRef v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public ITensor GetKeys(int l) => inner.GetKeys(l);
        public ITensor GetValues(int l) => inner.GetValues(l);
        public TensorRef GetKeysRef(int l) => inner.GetKeysRef(l);
        public TensorRef GetValuesRef(int l) => inner.GetValuesRef(l);
        public void Dispose() { Disposed = true; inner.Dispose(); }
    }

    private bool _firstDisposedBeforeSecondAlloc;

    private (TextGenerator Gen, PrefixCache Prefix, List<ReusableKv> Caches) Make(bool exclusive = true, bool canTruncate = true, int headroom = 0, int entries = 2,
        int growLimit = 0, PrefixCachePolicy policy = PrefixCachePolicy.Lru, Func<List<ReusableKv>, bool>? roomToGrow = null)
    {
        var caches = new List<ReusableKv>();
        var prefix = new PrefixCache(entries, policy);
        Func<bool>? room = roomToGrow is null ? null : () => roomToGrow(caches);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab),
            kvCacheFactory: (c, size) =>
            {
                var k = new ReusableKv(new SimpleKvCache(KvGeometry.FromConfig(c, false), size), exclusive, canTruncate, growLimit, room);
                if (caches.Count == 1) _firstDisposedBeforeSecondAlloc = caches[0].Disposed;
                caches.Add(k);
                return k;
            },
            prefixCache: prefix, prefixCacheHeadroomTokens: headroom, prefillChunkSize: 4);
        return (gen, prefix, caches);
    }

    private InferenceResponse Cold(int[] prompt)
        => new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab), prefillChunkSize: 4).Generate(Text(prompt), Opts());

    [Theory]
    [InlineData(12)]
    [InlineData(5)]
    [InlineData(23)]
    public void SharedPrefix_IsReused_AndMatchesColdPrefill(int shared)
    {
        var (gen, prefix, caches) = Make(headroom: 16);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        var warm = gen.Generate(Text(Fork(shared)), Opts());
        var cold = Cold(Fork(shared));

        Assert.Equal(shared, warm.Timings.CachedTokenCount);
        Assert.Equal(cold.GeneratedTokenIds, warm.GeneratedTokenIds);
        Assert.Single(caches);                                  // the second request adopted the first request's cache
    }

    [Fact]
    public void IdenticalPrompt_ReusesAllButOneToken()
    {
        var (gen, prefix, _) = Make();
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        var warm = gen.Generate(Text(A), Opts());
        Assert.Equal(A.Length - 1, warm.Timings.CachedTokenCount);   // one row stays to produce the first token's logits
        Assert.Equal(Cold(A).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    [Fact]
    public void PromptExtendingTheWholeStoredSequence_ReusesOnlyRowsThatWereWritten()
    {
        var (gen, prefix, _) = Make(headroom: 16);
        using var _ = prefix;
        var first = gen.Generate(Text(A), Opts());
        // The stored sequence is prompt + generated, but the last generated token was never forwarded: its row does not exist.
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(5, 19)];
        var warm = gen.Generate(Text(next), Opts());
        Assert.Equal(A.Length + MaxTokens - 1, warm.Timings.CachedTokenCount);
        Assert.Equal(Cold(next).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    [Fact]
    public void LongerNextTurn_MissesWithoutHeadroom_HitsWithIt()
    {
        int[] longer = [.. A, .. Ids(10, 5)];
        var (g0, p0, _) = Make(headroom: 0);
        using (p0)
        {
            g0.Generate(Text(A), Opts());
            Assert.Equal(0, g0.Generate(Text(longer), Opts()).Timings.CachedTokenCount);
        }

        var (g1, p1, c1) = Make(headroom: 16);
        using (p1)
        {
            g1.Generate(Text(A), Opts());
            var warm = g1.Generate(Text(longer), Opts());
            Assert.Equal(A.Length, warm.Timings.CachedTokenCount);
            Assert.Equal(Cold(longer).GeneratedTokenIds, warm.GeneratedTokenIds);
            Assert.Single(c1);
        }
    }

    /// <summary>
    /// Issue #1781: the next turn of the conversation adds more than the headroom (an agent's large tool result). The retained cache
    /// grows, keeps the reused rows, and the request prefills only the new suffix; the output equals a cold prefill.
    /// </summary>
    [Theory]
    [InlineData(PrefixCachePolicy.Lru)]
    [InlineData(PrefixCachePolicy.Agent)]
    public void TurnGrowingPastTheHeadroom_GrowsTheRetainedCache_AndReusesThePrefix(PrefixCachePolicy policy)
    {
        var (gen, prefix, caches) = Make(headroom: 4, growLimit: TinyQwen.Context, policy: policy);
        using var _ = prefix;
        var first = gen.Generate(Text(A), Opts());
        int retained = caches[0].MaxLength;                                 // A + MaxTokens + 4 = 34
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(16, 23)];    // 46 tokens: 46 + 6 > 34
        Assert.True(next.Length + MaxTokens > retained);

        var warm = gen.Generate(Text(next), Opts());

        Assert.Equal(A.Length + MaxTokens - 1, warm.Timings.CachedTokenCount);  // every row the first request wrote
        Assert.Equal(Cold(next).GeneratedTokenIds, warm.GeneratedTokenIds);
        Assert.Single(caches);                                              // grown in place, not replaced
        Assert.True(caches[0].MaxLength >= next.Length + MaxTokens);
        Assert.Equal(1, prefix.EntryCount);
    }

    /// <summary>
    /// Issue #1781 fallback: a growth the cache refuses (a device cache whose admission finds no room) leaves the request a miss with
    /// a fresh cache, as before, and the output is still a cold prefill's. The growth was asked for: with headroom first, then exactly.
    /// </summary>
    [Fact]
    public void RefusedGrowth_FallsBackToAMiss_AndIsStillCorrect()
    {
        var (gen, prefix, caches) = Make(headroom: 4, growLimit: 40);
        using var _ = prefix;
        var first = gen.Generate(Text(A), Opts());
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(16, 23)];    // needs 52 > 40

        var warm = gen.Generate(Text(next), Opts());

        Assert.Equal(0, warm.Timings.CachedTokenCount);
        Assert.Equal(Cold(next).GeneratedTokenIds, warm.GeneratedTokenIds);
        Assert.Equal([56, 52], caches[0].GrowthRequests);                   // headroom target, then the exact size
        Assert.Equal(2, caches.Count);
        // The refused device cache is freed BEFORE the miss allocates its replacement (#1594 review): holding both would spill a
        // card whose free memory just refused the growth. Only the new cache is retained.
        Assert.True(_firstDisposedBeforeSecondAlloc, "the refused entry was still resident when the miss allocated");
        Assert.Equal(1, prefix.EntryCount);
        Assert.False(caches[1].Disposed);
    }

    /// <summary>
    /// Issue #1843: a hit whose growth is refused only because ANOTHER retained entry holds the device memory (or, on the CUDA paged
    /// pool, the blocks) evicts that entry and retries the growth, instead of giving up the hit and re-prefilling the conversation.
    /// Here a growth fits only while one cache is alive.
    /// </summary>
    [Fact]
    public void RefusedGrowth_EvictsAnotherRetainedEntry_AndKeepsTheHit()
    {
        var (gen, prefix, caches) = Make(headroom: 4, growLimit: TinyQwen.Context,
            roomToGrow: all => all.Count(c => !c.Disposed) <= 1);
        using var _ = prefix;
        var first = gen.Generate(Text(A), Opts());
        int[] unrelated = Ids(20, 41);
        gen.Generate(Text(unrelated), Opts());                              // a second, unrelated entry is retained beside A's
        Assert.Equal(2, prefix.EntryCount);
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(16, 23)];    // outgrows A's cache

        var warm = gen.Generate(Text(next), Opts());

        Assert.Equal(A.Length + MaxTokens - 1, warm.Timings.CachedTokenCount);   // the hit was kept
        Assert.True(caches[1].Disposed, "the other entry was not evicted");
        Assert.False(caches[0].Disposed);
        Assert.Equal(2, caches.Count);                                           // grown in place, no miss allocation
        Assert.Equal(1, prefix.AdmissionEvictions);
        Assert.Equal(Cold(next).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    /// <summary>
    /// Issue #1781 on the CPU <c>--no-paged</c> path: the server passes no headroom off Vulkan, so before the fix any next turn longer
    /// than the first request's prompt + max tokens missed. <see cref="SimpleKvCache"/> grows, and the next turn reuses the prefix.
    /// </summary>
    [Fact]
    public void HostSimpleKvCache_NextTurn_GrowsAndReusesThePrefix()
    {
        using var prefix = new PrefixCache(2, PrefixCachePolicy.Agent);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab), prefixCache: prefix, prefillChunkSize: 4);
        var first = gen.Generate(Text(A), Opts());
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(16, 23)];

        var warm = gen.Generate(Text(next), Opts());

        Assert.Equal(A.Length + MaxTokens - 1, warm.Timings.CachedTokenCount);
        Assert.Equal(Cold(next).GeneratedTokenIds, warm.GeneratedTokenIds);
        Assert.Equal(1, prefix.EntryCount);
    }

    [Fact]
    public void NonExclusiveStorage_NeverReused()
    {
        var (gen, prefix, _) = Make(exclusive: false);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        Assert.Equal(0, gen.Generate(Text(Fork(12)), Opts()).Timings.CachedTokenCount);
        Assert.Equal(0, prefix.EntryCount);                         // not even retained
    }

    [Fact]
    public void CacheThatCannotTruncate_IsAMiss_AndStillCorrect()
    {
        var (gen, prefix, _) = Make(canTruncate: false);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        var warm = gen.Generate(Text(Fork(12)), Opts());
        Assert.Equal(0, warm.Timings.CachedTokenCount);
        Assert.Equal(Cold(Fork(12)).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    [Fact]
    public void ReusedCacheIsNotDisposed_AndEvictedOneIs()
    {
        var (gen, prefix, caches) = Make(entries: 1);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        gen.Generate(Text(Fork(12)), Opts());                      // reuses
        Assert.False(caches[0].Disposed);
        gen.Generate(Text(Ids(40, 29)), Opts());                   // too long for the retained cache and unrelated: new cache, old evicted
        Assert.Equal(2, caches.Count);
        Assert.True(caches[0].Disposed);
    }

    [Fact]
    public void CacheTypeThatCanNeverBeReused_IsNotRetained()
    {
        var caches = new List<OpaqueKv>();
        using var prefix = new PrefixCache(2);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab),
            kvCacheFactory: (c, size) =>
            {
                var k = new OpaqueKv(new SimpleKvCache(KvGeometry.FromConfig(c, false), size));
                caches.Add(k);
                return k;
            },
            prefixCache: prefix, prefillChunkSize: 4);
        gen.Generate(Text(A), Opts());
        Assert.Equal(0, prefix.EntryCount);
        Assert.True(caches[0].Disposed);
    }

    [Fact]
    public void StorageThatStopsBeingExclusiveAfterRetention_IsNotReused()
    {
        var (gen, prefix, caches) = Make();
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        caches[0].Exclusive = false;                                // a forward on another cache has since overwritten the rows (#1025)
        var warm = gen.Generate(Text(Fork(12)), Opts());
        Assert.Equal(0, warm.Timings.CachedTokenCount);
        Assert.Equal(Cold(Fork(12)).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    /// <summary>
    /// A factory standing in for a device whose free memory holds <paramref name="room"/> live caches (#1848): one more is refused
    /// with <see cref="DeviceMemoryAdmissionException"/>, as the CUDA KV cache constructors refuse past free VRAM.
    /// </summary>
    private (TextGenerator Gen, PrefixCache Prefix, List<ReusableKv> Caches) MakeAdmitting(int room, int entries = 2,
        long? requiredBytes = null, long entryDeviceBytes = 0)
    {
        var caches = new List<ReusableKv>();
        var prefix = new PrefixCache(entries);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab),
            kvCacheFactory: (c, size) =>
            {
                if (caches.Count(k => !k.Disposed) >= room)
                    throw requiredBytes is long req
                        ? new DeviceMemoryAdmissionException("Not enough free GPU memory for the KV cache (test)", req, freeBytes: 0, reserveBytes: 0)
                        : new DeviceMemoryAdmissionException("Not enough free GPU memory for the KV cache (test)");
                var k = new ReusableKv(new SimpleKvCache(KvGeometry.FromConfig(c, false), size), exclusive: true, canTruncate: true)
                {
                    DeviceBytes = entryDeviceBytes,
                };
                caches.Add(k);
                return k;
            },
            prefixCache: prefix, prefillChunkSize: 4);
        return (gen, prefix, caches);
    }

    [Fact]
    public void RefusedMissAllocation_EvictsTheRetainedEntry_AndRetries()
    {
        // #1848: a miss allocates a whole new cache beside the retained one; on a card with room for one, the retained entry is
        // evicted and the allocation retried, instead of the refusal failing the request (or the cache spilling).
        var (gen, prefix, caches) = MakeAdmitting(room: 1);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        Assert.Equal(1, prefix.EntryCount);

        int[] unrelated = Ids(20, 41);
        var warm = gen.Generate(Text(unrelated), Opts());

        Assert.True(caches[0].Disposed);
        Assert.Equal(1, prefix.AdmissionEvictions);
        Assert.Equal(Cold(unrelated).GeneratedTokenIds, warm.GeneratedTokenIds);
        Assert.Equal(1, prefix.EntryCount);   // the new request's cache is retained in its place
    }

    [Fact]
    public void RefusedMissAllocation_WithNothingLeftToEvict_ReachesTheCaller()
    {
        var (gen, prefix, _) = MakeAdmitting(room: 0);
        using var __ = prefix;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => gen.Generate(Text(A), Opts()));
        Assert.Contains("Not enough free GPU memory", ex.Message);
        Assert.Equal(0, prefix.AdmissionEvictions);
    }

    [Fact]
    public void RefusalThatEvictionCannotFix_KeepsTheRetainedEntries()
    {
        // #1848 review: a request that needs more than free memory plus everything retained (an oversized request, e.g. --ctx-size
        // above the fit) must not empty the prefix cache on its way to a 400. 1,000 bytes needed, 0 free, the entry holds 400.
        var (gen, prefix, caches) = MakeAdmitting(room: 1, requiredBytes: 1000, entryDeviceBytes: 400);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        Assert.Equal(400, prefix.DeviceBytesHeld);

        Assert.Throws<DeviceMemoryAdmissionException>(() => gen.Generate(Text(Ids(20, 41)), Opts()));
        Assert.False(caches[0].Disposed);
        Assert.Equal(1, prefix.EntryCount);
        Assert.Equal(0, prefix.AdmissionEvictions);
    }

    [Fact]
    public void RefusalThatEvictionCanFix_Evicts()
    {
        // The same refusal, but the retained entry holds enough to make the allocation fit: it is evicted and the retry succeeds.
        var (gen, prefix, caches) = MakeAdmitting(room: 1, requiredBytes: 1000, entryDeviceBytes: 1000);
        using var _ = prefix;
        gen.Generate(Text(A), Opts());
        int[] unrelated = Ids(20, 41);
        var warm = gen.Generate(Text(unrelated), Opts());
        Assert.True(caches[0].Disposed);
        Assert.Equal(1, prefix.AdmissionEvictions);
        Assert.Equal(Cold(unrelated).GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    [Theory]
    [InlineData(1000, 0, 0, 999, false)]
    [InlineData(1000, 0, 0, 1000, true)]
    [InlineData(1000, 300, 100, 799, false)]
    [InlineData(1000, 300, 100, 800, true)]
    public void CouldFitAfterReleasing_CountsFreeReclaimableAndReserve(long required, long free, long reserve, long reclaimable, bool expected)
        => Assert.Equal(expected, new DeviceMemoryAdmissionException("x", required, free, reserve).CouldFitAfterReleasing(reclaimable));

    [Fact]
    public void CouldFitAfterReleasing_WithoutNumbers_CannotRuleItOut()
        => Assert.True(new DeviceMemoryAdmissionException("x").CouldFitAfterReleasing(0));

    [Fact]
    public void EvictForAdmission_TakesTheProtectedEntryLast()
    {
        // #1848: the protected entry a miss seeds from (#1780) goes only when nothing else is left. k1 is stored first, so it is the
        // LRU victim a keep-blind eviction would pick.
        using var prefix = new PrefixCache(4);
        var k1 = new ReusableKv(new SimpleKvCache(KvGeometry.FromConfig(_tiny.Model.Config, false), 32), true, true);
        var k2 = new ReusableKv(new SimpleKvCache(KvGeometry.FromConfig(_tiny.Model.Config, false), 32), true, true);
        prefix.Store(Ids(8, 1), k1);
        prefix.Store(Ids(8, 2), k2);
        var keep = prefix.Entries.Single(e => ReferenceEquals(e.KvCache, k1));

        Assert.Same(k2, prefix.EvictForAdmission(keep)!.KvCache);
        Assert.True(k2.Disposed);
        Assert.False(k1.Disposed);
        Assert.Same(k1, prefix.EvictForAdmission(keep)!.KvCache);
        Assert.True(k1.Disposed);
        Assert.Null(prefix.EvictForAdmission(keep));
        Assert.Equal(2, prefix.AdmissionEvictions);
    }
}
