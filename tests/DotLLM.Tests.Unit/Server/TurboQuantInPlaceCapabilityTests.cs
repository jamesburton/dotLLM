using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1892 Task 8: the capability query, the shared KV factory and the sizing / bound arithmetic, decided without a GPU. Each test names the
/// mutant it kills in its comment (CLAUDE.md "Prove the Test Can Fail").
/// </summary>
public sealed class TurboQuantInPlaceCapabilityTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public TurboQuantInPlaceCapabilityTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    private static KvCacheConfig Tq(int bits = 4, bool qjl = false, int window = 0)
        => new(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, window, bits, TurboQuantUseQjl: qjl);

    private static ModelConfig Dense(int heads = 16, int kvHeads = 4, int headDim = 64, int layers = 4) => new()
    {
        Architecture = Architecture.Llama, VocabSize = 1000, HiddenSize = heads * headDim, IntermediateSize = 2048,
        NumLayers = layers, NumAttentionHeads = heads, NumKvHeads = kvHeads, HeadDim = headDim, MaxSequenceLength = 2048,
    };

    // ---------------------------------------------------------------- the host-independent capability

    [Theory]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 4, false, 0, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 8, false, 0, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 1, false, 0, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 4, true, 0, false)]    // tq*q: step 5
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 4, false, 32, false)]  // no original-space window
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.Q8_0, 4, false, 0, false)]         // mixed: step 6
    [InlineData(KvCacheDType.F32, KvCacheDType.TurboQuant, 4, false, 0, false)]          // V-only: step 6
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 4, false, 0, false)]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32, 4, false, 0, false)]
    public void Eligible_PerRequest(KvCacheDType k, KvCacheDType v, int bits, bool qjl, int window, bool expected)
        => Assert.Equal(expected, TurboQuantInPlaceSupport.Eligible(new KvCacheConfig(k, v, window, bits, TurboQuantUseQjl: qjl), Dense()));

    [Fact]
    public void Eligible_PerGeometry()
    {
        Assert.True(TurboQuantInPlaceSupport.Eligible(Tq(), Dense(headDim: 128)));
        Assert.True(TurboQuantInPlaceSupport.Eligible(Tq(), Dense(headDim: 256)));
        Assert.False(TurboQuantInPlaceSupport.Eligible(Tq(), Dense(headDim: 80)));    // #1897
        Assert.False(TurboQuantInPlaceSupport.Eligible(Tq(), Dense(headDim: 512)));   // #1897
        Assert.False(TurboQuantInPlaceSupport.Eligible(Tq(), Dense() with { ValueHeadDim = 32 }));   // #1898
        Assert.False(TurboQuantInPlaceSupport.Eligible(Tq(), Dense() with { MlaConfig = new MlaConfig { KvLoraRank = 512, QkNopeHeadDim = 48, QkRopeHeadDim = 16, VHeadDim = 64 } }));   // #1899
    }

    [Fact]
    public void ModelsThatDoNotImplementIt_RefuseByDefault()
    {
        // The default member is false: hybrids, recurrent, Qwen4Exp and pipeline hosts keep it, so the resolver refuses them up front (#1900).
        Assert.False(new DotLLM.Vulkan.TestDoubles.FakeVulkanHost(Dense(), supportsInPlace: false).SupportsInPlaceTurboQuant(Tq()));
        IModel plain = new NoCapabilityModel(Dense());
        Assert.False(plain.SupportsInPlaceTurboQuant(Tq()));
        string? reason = TurboQuantKv.PostLoadRefusal(Tq(), null, KvCommand.Run, paged: false, plain.Config, plain, false, false);
        Assert.Contains("#1900", reason);
    }

    private sealed class NoCapabilityModel(ModelConfig config) : IModel
    {
        public ModelConfig Config { get; } = config;
        public long ComputeMemoryBytes => 0;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    [Fact]
    public void CpuTransformerModel_SaysYesForTheMseStage_NoForTheRest()
    {
        var m = _tiny.Model;
        Assert.True(m.SupportsInPlaceTurboQuant(Tq(4)));
        Assert.True(m.SupportsInPlaceTurboQuant(Tq(2)));
        Assert.False(m.SupportsInPlaceTurboQuant(Tq(4, qjl: true)));
        Assert.False(m.SupportsInPlaceTurboQuant(Tq(4, window: 16)));
        Assert.False(m.SupportsInPlaceTurboQuant(new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0)));
        Assert.False(m.SupportsInPlaceTurboQuant(KvCacheConfig.Default));
    }

    // ---------------------------------------------------------------- the shared factory (serve / run / chat / bench)

    [Fact]
    public void CreateFactory_Cpu_Mse_BuildsInPlace_Qjl_BuildsTheScratchCache()
    {
        using (var c = TurboQuantKv.CreateFactory(_tiny.Model, Tq(3))(_tiny.Config, 32))
            Assert.IsType<TurboQuantInPlaceKvCache>(c);
        using (var c = TurboQuantKv.CreateFactory(_tiny.Model, Tq(4, qjl: true))(_tiny.Config, 32))
            Assert.IsType<TurboQuantKvCache>(c);
    }

    [Fact]
    public void CreateFactory_ForAHostThatCannot_Throws_NeverAnotherCache()
    {
        var host = new NoCapabilityModel(Dense());
        var e = Assert.Throws<NotSupportedException>(() => TurboQuantKv.CreateFactory(host, Tq()));
        Assert.Contains("#1900", e.Message);
        // A fake Vulkan host (by namespace) with the capability but not the real model type is still not a cache source.
        Assert.Throws<NotSupportedException>(() => TurboQuantKv.CreateFactory(new DotLLM.Vulkan.TestDoubles.FakeVulkanHost(Dense(), true), Tq()));
    }

    // ---------------------------------------------------------------- sizing charges the real TurboQuant bytes

    /// <summary>
    /// Kills mutant (c), "sizing charges F16 for tq": 4 layers, 4 KV heads, head dim 64, 4 bits = 32 code bytes + a 2-byte F16 norm per
    /// vector = 34 B, times 4 heads, 2 sides, 4 layers = 1088 B per token; F16 is 4096.
    /// </summary>
    [Fact]
    public void KvBytesPerToken_IsCodesPlusNorms_NotF16()
    {
        var config = Dense(heads: 16, kvHeads: 4, headDim: 64, layers: 4);
        Assert.Equal(1088, TurboQuantInPlaceSupport.KvBytesPerToken(config, Tq(4)));
        long f16 = 2L * 2 * 4 * 64 * 4;
        Assert.Equal(4096, f16);
        Assert.True(TurboQuantInPlaceSupport.KvBytesPerToken(config, Tq(4)) < f16);
        // 8 bits: 64 code bytes + 2 = 66 per vector.
        Assert.Equal(66L * 4 * 2 * 4, TurboQuantInPlaceSupport.KvBytesPerToken(config, Tq(8)));
        // It is the layout's own figure, and it matches what the CPU cache really allocates (guard words aside).
        var layout = new TurboQuantLayout(64, 4, 4, TurboQuantLayout.DefaultNorm);
        Assert.Equal(layout.KvBytesPerPosition(4), TurboQuantInPlaceSupport.KvBytesPerToken(config, Tq(4)));
    }

    [Fact]
    public void CpuCache_AllocatedBytes_MatchesTheSizing()
    {
        using var cache = new TurboQuantInPlaceKvCache(_tiny.Config.NumLayers, _tiny.Config.NumKvHeads, _tiny.Config.HeadDim, 128, 4, 1UL);
        long perToken = TurboQuantInPlaceSupport.KvBytesPerToken(_tiny.Config, Tq(4));
        long guardAndRounding = cache.AllocatedBytes - perToken * 128;
        Assert.InRange(guardAndRounding, 0, 2L * _tiny.Config.NumLayers * 8);   // only the 4-byte guard word per side per layer is not per-token
    }

    [Fact]
    public void CudaDenseContextBytesPerToken_ChargesTheRealTurboQuantBytes()
    {
        var config = Dense(heads: 16, kvHeads: 4, headDim: 64, layers: 4);
        var geom = KvGeometry.FromConfig(config);
        long f16 = CudaTransformerModel.DenseContextBytesPerToken(geom, KvCacheConfig.Default, activationBytesPerToken: 0, chunkedPrefill: true);
        long tq4 = CudaTransformerModel.DenseContextBytesPerToken(geom, Tq(4), 0, chunkedPrefill: true, config.HeadDim);
        long tq8 = CudaTransformerModel.DenseContextBytesPerToken(geom, Tq(8), 0, chunkedPrefill: true, config.HeadDim);
        Assert.Equal(4096, f16);
        Assert.Equal(1088, tq4);          // codes + F16 norms: not the F16 figure
        Assert.Equal(66L * 4 * 2 * 4, tq8);
        Assert.True(tq4 < tq8 && tq8 < f16);
        // Activation scratch is added the same way it is for every other KV type.
        Assert.Equal(tq4 + 500, CudaTransformerModel.DenseContextBytesPerToken(geom, Tq(4), 500, chunkedPrefill: false, config.HeadDim));
        Assert.Throws<ArgumentException>(() => CudaTransformerModel.DenseContextBytesPerToken(geom, Tq(4), 0, true));   // no head dim: refuse, never guess F16
    }

    [Fact]
    public void CudaAutoSplit_PlansTheRealTurboQuantBytesPerLayer()
    {
        var config = Dense(heads: 16, kvHeads: 4, headDim: 64, layers: 4);
        long f16PerLayer = CudaAutoSplit.KvBytesPerTokenPerLayer(config, null);
        long tqPerLayer = CudaAutoSplit.KvBytesPerTokenPerLayer(config, Tq(4));
        Assert.Equal(1024, f16PerLayer);
        Assert.Equal(1088 / 4, tqPerLayer);
        Assert.True(tqPerLayer < f16PerLayer);
        Assert.Equal(f16PerLayer, CudaAutoSplit.KvBytesPerTokenPerLayer(config, new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0)));   // others keep their figure
    }

    // ---------------------------------------------------------------- CUDA capability inputs

    [Theory]
    [InlineData(true, true, true, true, true)]
    [InlineData(false, true, true, true, false)]   // encode PTX missing / stale
    [InlineData(true, false, true, true, false)]   // Q rotation missing
    [InlineData(true, true, false, true, false)]   // O rotation missing
    [InlineData(true, true, true, false, false)]   // attention_kvtq.ptx missing
    public void CudaKernelsLoaded_NeedsEveryKernel(bool encode, bool rotQ, bool rotO, bool attn, bool expected)
        => Assert.Equal(expected, CudaTransformerModel.InPlaceTurboQuantKernelsLoaded(encode, rotQ, rotO, attn));

    /// <summary>
    /// #1892 final review I1: the F32 dense forward (<c>ForwardGemmaF32</c>) accepts only a <c>CudaKvCache</c>, so a host that runs it
    /// must not claim in-place TurboQuant (the first forward would throw mid-request). Kills the mutant "drop the _isGemmaFamily clause".
    /// Compile-checked plus this predicate test only: the host-level wiring needs a GPU.
    /// </summary>
    [Theory]
    [InlineData(false, false, false, true)]
    [InlineData(true, false, false, false)]    // MLA keeps its own cache
    [InlineData(false, true, false, false)]    // Gemma-4 F32 forward
    [InlineData(false, false, true, false)]    // Gemma 1/2, Olmo2, SmolLM3, non-fold Granite
    public void CudaHostServed_RefusesTheF32DenseFamily(bool mla, bool gemma4, bool gemmaFamily, bool expected)
        => Assert.Equal(expected, CudaTransformerModel.InPlaceTurboQuantHostServed(mla, gemma4, gemmaFamily));

    [Theory]
    [InlineData(Architecture.Olmo2, true)]
    [InlineData(Architecture.SmolLM3, true)]
    [InlineData(Architecture.Llama, false)]
    public void CudaUsesF32DenseForward_ForTheArchitecturesTheCapabilityMustRefuse(Architecture arch, bool expected)
        => Assert.Equal(expected, CudaTransformerModel.UsesF32DenseForward(Dense() with { Architecture = arch }));

    // ---------------------------------------------------------------- Vulkan: the row bound and the arm completeness

    /// <summary>
    /// Kills mutant (b), "the Vulkan prefill bound ignores numHeads": at GQA ratio 32 the flash twin's ceil(rows / 16) x 32 groups bind at
    /// 16 x 2047 = 32752, not the encode's 65535 / 1.
    /// </summary>
    [Theory]
    [InlineData(32, 32, 2047)]    // encode binds
    [InlineData(4, 2, 32767)]
    [InlineData(32, 1, 32752)]    // flash grid binds
    [InlineData(64, 1, 16368)]    // 16 * floor(65535 / 64)
    [InlineData(16, 1, 65520)]    // ratio 16: 16 * 4095 = 65520 < 65535
    [InlineData(8, 8, 8191)]
    public void VulkanTurboQuantRowBound(int heads, int kvHeads, int expected)
    {
        Assert.Equal(expected, VulkanTransformerModel.MaxTurboQuantRowsPerForward(heads, kvHeads));
        int rows = expected;
        Assert.True((long)rows * kvHeads <= 65535, "encode grid");
        Assert.True((long)((rows + 15) / 16) * heads <= 65535, "flash grid");
    }

    private static VulkanTransformerModel.TqArms Arms(bool attention = true, bool splitKv = true, bool flash = true, bool coopmat = true,
        bool prefillGqa = true, bool qpos = true, bool qposTiled = true, bool sinkFlash = true, bool sinkRef = true)
        => new(attention, splitKv, flash, coopmat, prefillGqa, qpos, qposTiled, sinkFlash, sinkRef);

    /// <summary>
    /// Kills the plan's mutant "Vulkan's capability skips one arm (Coopmat)": a host whose F32 chain loaded the coopmat flash must not be
    /// accepted with a TurboQuant set that lacks its twin, and likewise for every other arm; an arm the device never had (null in the
    /// baseline) needs no twin.
    /// </summary>
    [Theory]
    [InlineData("splitKv")]
    [InlineData("flash")]
    [InlineData("coopmat")]
    [InlineData("prefillGqa")]
    [InlineData("qpos")]
    [InlineData("qposTiled")]
    public void TqArmsComplete_Dense_EveryBaselineArmNeedsItsTwin(string missing)
    {
        var full = Arms();
        Assert.True(VulkanTransformerModel.TqArmsComplete(false, true, full, full));
        var tq = missing switch
        {
            "splitKv" => Arms(splitKv: false), "flash" => Arms(flash: false), "coopmat" => Arms(coopmat: false),
            "prefillGqa" => Arms(prefillGqa: false), "qpos" => Arms(qpos: false), "qposTiled" => Arms(qposTiled: false),
            _ => throw new ArgumentException(missing),
        };
        Assert.False(VulkanTransformerModel.TqArmsComplete(false, true, tq, full), $"accepted a set without the {missing} twin");
        // The device never had that arm: nothing to twin.
        var baseline = missing switch
        {
            "splitKv" => Arms(splitKv: false), "flash" => Arms(flash: false), "coopmat" => Arms(coopmat: false),
            "prefillGqa" => Arms(prefillGqa: false), "qpos" => Arms(qpos: false), "qposTiled" => Arms(qposTiled: false),
            _ => throw new ArgumentException(missing),
        };
        Assert.True(VulkanTransformerModel.TqArmsComplete(false, true, tq, baseline));
    }

    [Fact]
    public void TqArmsComplete_Dense_NeedsThePerTokenKernelAndTheWriter()
    {
        Assert.False(VulkanTransformerModel.TqArmsComplete(false, true, Arms(attention: false), Arms()));   // the arm of last resort
        Assert.False(VulkanTransformerModel.TqArmsComplete(false, writerAndRotation: false, Arms(), Arms()));
    }

    [Theory]
    [InlineData("splitKv")]
    [InlineData("sinkFlash")]
    [InlineData("sinkRef")]
    public void TqArmsComplete_Flint_NeedsItsThreeArms(string missing)
    {
        Assert.True(VulkanTransformerModel.TqArmsComplete(true, true, Arms(), Arms()));
        var tq = missing switch
        {
            "splitKv" => Arms(splitKv: false), "sinkFlash" => Arms(sinkFlash: false), "sinkRef" => Arms(sinkRef: false),
            _ => throw new ArgumentException(missing),
        };
        Assert.False(VulkanTransformerModel.TqArmsComplete(true, true, tq, Arms()));
        Assert.False(VulkanTransformerModel.TqArmsComplete(true, writerAndRotation: false, Arms(), Arms()));
    }

    // ---------------------------------------------------------------- serve builds the in-place cache (CPU, TinyQwen)

    [Fact]
    public void Serve_BuildsTheInPlaceCache()
    {
        var options = new ServerOptions
        {
            Model = _tiny.GgufPath, ModelId = "tiny-1892", Device = "cpu", UsePaged = false, Warmup = WarmupOptions.Disabled,
            CacheTypeK = "tq3", CacheTypeV = "tq3",
        };
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, options);
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        Assert.IsType<TurboQuantInPlaceKvCache>(cache);
    }
}
