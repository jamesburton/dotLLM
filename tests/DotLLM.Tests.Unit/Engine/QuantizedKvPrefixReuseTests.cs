using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;
using Xunit;
using static DotLLM.Tests.Unit.Engine.PrefixCacheFailedRequestTests;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1843, CPU side: <see cref="QuantizedKvCache"/> (q8_0 / q4_0 KV, <c>--no-paged</c> or a paged request the block pool cannot
/// serve) is prefix-reusable like CUDA's and Vulkan's quantized caches: a later request cuts it back, grows it when the next turn is
/// longer, and the continuation equals a cold run on the same cache type. A full-precision window may only be cut before its first
/// eviction. Runs the deterministic CPU qwen3 at head dim 32: CPU quantized attention dequantizes per head and throws for a head dim
/// that is not a multiple of 32 (the head-dim-16 fixture), a cross-backend gap tracked in #1883.
/// </summary>
public sealed class QuantizedKvPrefixReuseTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private const int MaxTokens = 6;
    private readonly TinyQwen _tiny;

    public QuantizedKvPrefixReuseTests(TinyQwen tiny) => _tiny = tiny;

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13) % (TinyQwen.Vocab - 2);
        return ids;
    }

    private static readonly int[] A = Ids(24, 3);
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

    private Func<Core.Models.ModelConfig, int, IKvCache> Factory(KvCacheDType k, KvCacheDType v, int window, List<QuantizedKvCache>? made = null)
        => (c, size) =>
        {
            var cache = new QuantizedKvCache(KvGeometry.FromConfig(c), size, k, v, window);
            made?.Add(cache);
            return cache;
        };

    [Theory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q4_0)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0)]
    public void SecondTurn_ReusesPrefix_GrowsTheCache_AndMatchesCold(KvCacheDType k, KvCacheDType v)
    {
        var tok = new Tok(TinyQwen.Vocab);
        var made = new List<QuantizedKvCache>();
        using var prefix = new PrefixCache(2);
        var warm = new TextGenerator(_tiny.Model32, tok, Factory(k, v, 0, made), prefix, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
        var cold = new TextGenerator(_tiny.Model32, tok, Factory(k, v, 0), mtpEnabled: false);

        var first = warm.Generate(Text(A), Opts());
        Assert.Equal(1, prefix.EntryCount);
        int[] next = [.. A, .. first.GeneratedTokenIds, .. Ids(16, 23)];   // outgrows the (prompt + MaxTokens) cache

        var actual = warm.Generate(Text(next), Opts());

        Assert.Equal(A.Length + MaxTokens - 1, actual.Timings.CachedTokenCount);
        Assert.Single(made);                                        // the first cache was adopted and grown, not replaced
        Assert.True(made[0].MaxLength >= next.Length + MaxTokens);
        Assert.Equal(cold.Generate(Text(next), Opts()).GeneratedTokenIds, actual.GeneratedTokenIds);
    }

    [Fact]
    public void Window_AfterItsFirstEviction_CannotBeCut_SoTheHitMisses_AndIsStillCorrect()
    {
        var tok = new Tok(TinyQwen.Vocab);
        using var prefix = new PrefixCache(2);
        var made = new List<QuantizedKvCache>();
        var warm = new TextGenerator(_tiny.Model32, tok, Factory(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 4, made), prefix, mtpEnabled: false);
        var cold = new TextGenerator(_tiny.Model32, tok, Factory(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 4), mtpEnabled: false);

        var first = warm.Generate(Text(A), Opts());
        Assert.True(made[0].QuantizedLength > 0, "the window should have evicted rows");
        Assert.False(made[0].CanTruncateTo(A.Length - 2));
        int[] fork = [.. A[..^2], .. Ids(5, 31)];

        var actual = warm.Generate(Text(fork), Opts());

        Assert.Equal(0, actual.Timings.CachedTokenCount);
        Assert.Equal(cold.Generate(Text(fork), Opts()).GeneratedTokenIds, actual.GeneratedTokenIds);
    }

    [Fact]
    public void Window_BeforeAnyEviction_CanBeCut()
    {
        using var kv = new QuantizedKvCache(KvGeometry.FromConfig(_tiny.Model32.Config), 32, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 16);
        var positions = Enumerable.Range(0, 10).ToArray();
        _tiny.Model32.Forward(A.AsSpan(0, 10), positions, 0, kv).Dispose();
        Assert.Equal(0, kv.QuantizedLength);
        Assert.True(kv.CanTruncateTo(4));
        Assert.False(kv.CanTruncateTo(11));
    }

    [Theory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q8_0)]
    public unsafe void Seed_CopiesTheRowsAsStored_SoTheSuffixLogitsEqualTheCutBackSource(KvCacheDType k, KvCacheDType v)
    {
        int n = A.Length, cut = 13;
        var geom = KvGeometry.FromConfig(_tiny.Model32.Config);
        int[] positions = Enumerable.Range(0, n).ToArray();
        float[] Suffix(IKvCache kv)
        {
            using var t = _tiny.Model32.Forward(A.AsSpan(cut), positions.AsSpan(cut), 0, kv);
            int vocab = _tiny.Model32.Config.VocabSize, rows = t.Shape[0];
            var row = new float[vocab];
            new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
            return row;
        }

        using var src = new QuantizedKvCache(geom, n, k, v, 0);
        _tiny.Model32.Forward(A, positions, 0, src).Dispose();
        using var seeded = new QuantizedKvCache(geom, n, k, v, 0);
        Assert.True(seeded.TrySeedPrefixFrom(src, cut));
        Assert.Equal(cut, seeded.CurrentLength);
        float[] fromSeed = Suffix(seeded);
        src.Rollback(cut);
        Assert.Equal(Suffix(src), fromSeed);

        using var otherType = new QuantizedKvCache(geom, n, KvCacheDType.Q4_0, KvCacheDType.Q4_0, 0);
        if (k != KvCacheDType.Q4_0 || v != KvCacheDType.Q4_0)
            Assert.False(otherType.TrySeedPrefixFrom(src, cut));
        using var windowed = new QuantizedKvCache(geom, n, k, v, 4);
        Assert.False(windowed.TrySeedPrefixFrom(src, cut));
        Assert.Equal(0, windowed.CurrentLength);
    }

    [Fact]
    public unsafe void TryGrowTo_KeepsTheRows()
    {
        var geom = KvGeometry.FromConfig(_tiny.Model32.Config);
        int[] positions = Enumerable.Range(0, A.Length).ToArray();
        using var reference = new QuantizedKvCache(geom, 64, KvCacheDType.Q4_0, KvCacheDType.Q8_0, 0);
        using var grown = new QuantizedKvCache(geom, A.Length + 1, KvCacheDType.Q4_0, KvCacheDType.Q8_0, 0);
        _tiny.Model32.Forward(A, positions, 0, reference).Dispose();
        _tiny.Model32.Forward(A, positions, 0, grown).Dispose();
        Assert.True(grown.TryGrowTo(64));
        Assert.Equal(64, grown.MaxLength);
        for (int l = 0; l < _tiny.Model32.Config.NumLayers; l++)
        {
            int bytes = A.Length * grown.KeyQuantizedRowBytesOf(l);
            Assert.True(new ReadOnlySpan<byte>((void*)reference.GetQuantizedKeysPtr(l), bytes).SequenceEqual(
                new ReadOnlySpan<byte>((void*)grown.GetQuantizedKeysPtr(l), bytes)), $"layer {l} keys moved");
        }
    }

}
