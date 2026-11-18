using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Configuration;

/// <summary>#1888: TurboQuant on either side is TurboQuant; K-only <see cref="KvCacheConfig.IsTurboQuant"/> let V-only tq slip through.</summary>
public class KvCacheConfigTurboQuantTests
{
    [Fact]
    public void VOnlyTurboQuant_IsAny_AndMixed()
    {
        var kv = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.TurboQuant);
        Assert.True(kv.IsTurboQuantAny);
        Assert.True(kv.IsTurboQuantMixed);
        Assert.False(kv.IsTurboQuant);
    }

    [Fact]
    public void KOnlyTurboQuant_IsAny_AndMixed()
    {
        var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.Q8_0);
        Assert.True(kv.IsTurboQuantAny);
        Assert.True(kv.IsTurboQuantMixed);
    }

    [Fact]
    public void BothTurboQuant_IsAny_NotMixed()
    {
        var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant);
        Assert.True(kv.IsTurboQuantAny);
        Assert.False(kv.IsTurboQuantMixed);
    }

    [Theory]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q8_0)]
    public void NeitherTurboQuant_IsNotAny(KvCacheDType k, KvCacheDType v)
    {
        var kv = new KvCacheConfig(k, v);
        Assert.False(kv.IsTurboQuantAny);
        Assert.False(kv.IsTurboQuantMixed);
    }

    [Fact]
    public void G3_DoesNotServe_TurboQuant()
    {
        // G3 reads an f16 cache; TurboQuant KV is codes, so G3 must never be planned over it (not even V-only).
        Assert.False(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant)));
        Assert.False(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.F32, KvCacheDType.TurboQuant)));
        Assert.True(CudaTransformerModel.G3ServesKvCache(KvCacheConfig.Default));
    }
}
