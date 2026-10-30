using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#1284: the CPU paged block type when no KV cache type is specified.</summary>
public class PagedKvBlockTypeDefaultTests
{
    private static ModelConfig Dense(int headDim = 64) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 1000, HiddenSize = 1024, IntermediateSize = 2048,
        NumLayers = 4, NumAttentionHeads = 16, NumKvHeads = 4, HeadDim = headDim,
        MaxSequenceLength = 2048,
    };

    private static KvCacheConfig Kv(KvCacheDType k, KvCacheDType v) => new(k, v);

    [Fact]
    public void Unspecified_DefaultsToQ8_WithInfo()
    {
        var t = ServerStartup.ResolvePagedBlockType(false, KvCacheConfig.Default, Dense(), null, out var info);
        Assert.Equal(DType.Q8_0, t);
        Assert.Contains("Q8_0", info);
        Assert.Contains("--cache-type-k f16 --cache-type-v f16", info);
    }

    [Fact]
    public void ExplicitF16_StaysF16_NoInfo()
    {
        var t = ServerStartup.ResolvePagedBlockType(true, Kv(KvCacheDType.F32, KvCacheDType.F32), Dense(), null, out var info);
        Assert.Equal(DType.Float16, t);
        Assert.Null(info);
    }

    [Fact]
    public void ExplicitQ8_IsQ8_NoInfo()
    {
        var t = ServerStartup.ResolvePagedBlockType(true, Kv(KvCacheDType.Q8_0, KvCacheDType.Q8_0), Dense(), null, out var info);
        Assert.Equal(DType.Q8_0, t);
        Assert.Null(info);
    }

    [Fact]
    public void UnsupportedModel_KeepsF16_NoInfo()
    {
        // Head dim not a multiple of 32: not Q8_0-eligible.
        var odd = ServerStartup.ResolvePagedBlockType(false, KvCacheConfig.Default, Dense(48), null, out var info1);
        Assert.Equal(DType.Float16, odd);
        Assert.Null(info1);

        // Hybrid (GDN) model.
        var hybrid = Dense() with { GdnConfig = new GatedDeltaNetConfig(4, 48, 16, 128, 6144, 4) };
        var t = ServerStartup.ResolvePagedBlockType(false, KvCacheConfig.Default, hybrid, null, out var info2);
        Assert.Equal(DType.Float16, t);
        Assert.Null(info2);
    }

    [Fact]
    public void Mixed_OnlyOneSideGiven_Unchanged()
    {
        // Lone f16 side: F16 blocks, no default message.
        var t = ServerStartup.ResolvePagedBlockType(true, KvCacheConfig.Default, Dense(), null, out var info);
        Assert.Equal(DType.Float16, t);
        Assert.Null(info);

        // A lone q8_0 side is not paged-eligible (the caller falls back to the quantized simple cache).
        Assert.False(ServerStartup.IsPagedQ8Eligible(Kv(KvCacheDType.Q8_0, KvCacheDType.F32), Dense(), null));
    }

    [Fact]
    public void ServerOptions_Parse_UnspecifiedIsNull_ExplicitIsKept()
    {
        var none = ServerOptions.Parse(["--model", "x"]);
        Assert.Null(none.CacheTypeK);
        Assert.Null(none.CacheTypeV);
        var f16 = ServerOptions.Parse(["--model", "x", "--cache-type-k", "f16", "--cache-type-v", "f16"]);
        Assert.Equal("f16", f16.CacheTypeK);
        Assert.Equal("f16", f16.CacheTypeV);
    }
}
