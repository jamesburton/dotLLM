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

    /// <summary>Exposes a SimpleKvCache only through the interfaces a device cache offers (never as a SimpleKvCache).</summary>
    private sealed class ReusableKv(SimpleKvCache inner, bool exclusive, bool canTruncate) : IPrefixReusableKvCache, IPerLayerKvCache
    {
        public bool Disposed;
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public int LayerCount => ((IPerLayerKvCache)inner).LayerCount;
        public int KvStrideOf(int l) => ((IPerLayerKvCache)inner).KvStrideOf(l);
        public bool HasExclusiveStorage => exclusive;
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

    private (TextGenerator Gen, PrefixCache Prefix, List<ReusableKv> Caches) Make(bool exclusive = true, bool canTruncate = true, int headroom = 0, int entries = 2)
    {
        var caches = new List<ReusableKv>();
        var prefix = new PrefixCache(entries);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab),
            kvCacheFactory: (c, size) =>
            {
                var k = new ReusableKv(new SimpleKvCache(KvGeometry.FromConfig(c, false), size), exclusive, canTruncate);
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
}
