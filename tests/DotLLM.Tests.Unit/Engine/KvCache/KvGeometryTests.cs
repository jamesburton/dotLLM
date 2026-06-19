using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Unit tests for <see cref="KvGeometry"/> — the Core per-layer KV-stride
/// descriptor (KV Phase 0). Verifies the uniform fast path stays byte-identical
/// for non-Gemma models and that <see cref="KvGeometry.FromConfig"/> produces the
/// distinct sliding-vs-global strides for Gemma-4.
/// </summary>
public sealed class KvGeometryTests
{
    private static ModelConfig BaseConfig(int numLayers, int numKvHeads, int headDim) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = 64,
        IntermediateSize = 128,
        NumLayers = numLayers,
        NumAttentionHeads = 4,
        NumKvHeads = numKvHeads,
        HeadDim = headDim,
        MaxSequenceLength = 16,
    };

    [Fact]
    public void Uniform_RepeatsStrideAcrossLayers()
    {
        var geom = KvGeometry.Uniform(numLayers: 4, numKvHeads: 2, headDim: 16);

        Assert.True(geom.IsUniform);
        Assert.Equal(32, geom.UniformStride);
        Assert.Equal(4, geom.LayerCount);
        for (int l = 0; l < geom.LayerCount; l++)
            Assert.Equal(32, geom.KvStrideOf(l));
    }

    [Fact]
    public void PerLayer_DistinctStrides_IsNonUniform()
    {
        var geom = KvGeometry.PerLayer([32, 64]);

        Assert.False(geom.IsUniform);
        Assert.Equal(0, geom.UniformStride);
        Assert.Equal(2, geom.LayerCount);
        Assert.Equal(32, geom.KvStrideOf(0));
        Assert.Equal(64, geom.KvStrideOf(1));
    }

    [Fact]
    public void PerLayer_AllEqual_CollapsesToUniformFastPath()
    {
        var geom = KvGeometry.PerLayer([48, 48, 48]);

        Assert.True(geom.IsUniform);
        Assert.Equal(48, geom.UniformStride);
        Assert.Equal(3, geom.LayerCount);
    }

    [Fact]
    public void FromConfig_NonGemma_IsUniform()
    {
        var config = BaseConfig(numLayers: 6, numKvHeads: 4, headDim: 64);

        var geom = KvGeometry.FromConfig(config);

        Assert.True(geom.IsUniform);
        Assert.Equal(4 * 64, geom.UniformStride);
        Assert.Equal(6, geom.LayerCount);
    }

    [Fact]
    public void FromConfig_Gemma4_ProducesDistinctSlidingAndGlobalStrides()
    {
        // Sliding layers: NumKvHeads(2) * HeadDim(16)            = 32
        // Global  layers: NumGlobalKvHeads(2) * GlobalHeadDim(32) = 64
        // PerLayerSlidingWindow: a window size => sliding; null => full-attention (global).
        var config = BaseConfig(numLayers: 4, numKvHeads: 2, headDim: 16) with
        {
            Architecture = Architecture.Gemma4,
            NumGlobalKvHeads = 2,
            GlobalHeadDim = 32,
            PerLayerSlidingWindow = new int?[] { 256, null, 256, null },
        };

        var geom = KvGeometry.FromConfig(config);

        Assert.False(geom.IsUniform);
        Assert.Equal(4, geom.LayerCount);
        Assert.Equal(32, geom.KvStrideOf(0)); // sliding
        Assert.Equal(64, geom.KvStrideOf(1)); // global
        Assert.Equal(32, geom.KvStrideOf(2)); // sliding
        Assert.Equal(64, geom.KvStrideOf(3)); // global
    }

    [Fact]
    public void FromConfig_MatchesPerLayerGetters()
    {
        var config = BaseConfig(numLayers: 4, numKvHeads: 2, headDim: 16) with
        {
            Architecture = Architecture.Gemma4,
            NumGlobalKvHeads = 2,
            GlobalHeadDim = 32,
            PerLayerSlidingWindow = new int?[] { 256, null, 256, null },
        };

        var geom = KvGeometry.FromConfig(config);

        for (int l = 0; l < config.NumLayers; l++)
            Assert.Equal(config.GetLayerKvHeads(l) * config.GetLayerHeadDim(l), geom.KvStrideOf(l));
    }

    [Fact]
    public void PerLayer_RejectsEmpty() =>
        Assert.Throws<System.ArgumentException>(() => KvGeometry.PerLayer([]));

    [Fact]
    public void Uniform_RejectsNonPositive() =>
        Assert.Throws<System.ArgumentOutOfRangeException>(() => KvGeometry.Uniform(0, 2, 16));
}
