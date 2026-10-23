using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public class GgufModelConfigExtractorRoPEOverrideTests
{
    private static ModelConfig BaseConfig(RoPEConfig? rope, RoPEConfig? globalRope = null) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32000,
        HiddenSize = 4096,
        IntermediateSize = 11008,
        NumLayers = 32,
        NumAttentionHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        MaxSequenceLength = 4096,
        NormEpsilon = 1e-5f,
        RoPEConfig = rope,
        GlobalRoPEConfig = globalRope,
    };

    [Fact]
    public void ApplyRoPEOverride_NullOverrides_ReturnsConfigUnchanged()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128);
        var config = BaseConfig(rope);

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, null);

        Assert.Equal(config, result);
    }

    [Fact]
    public void ApplyRoPEOverride_EmptyOverrides_ReturnsConfigUnchanged()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128);
        var config = BaseConfig(rope);

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, new RoPEOverrideOptions());

        Assert.Equal(config, result);
    }

    [Fact]
    public void ApplyRoPEOverride_NoRopeConfig_IsNoOp()
    {
        var config = BaseConfig(rope: null);
        var overrides = new RoPEOverrideOptions { FreqBase = 500000f };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.Null(result.RoPEConfig);
    }

    [Fact]
    public void ApplyRoPEOverride_FreqBase_OverridesThetaOnly()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128, ScalingType: RoPEScalingType.YaRN,
            ScalingFactor: 2.0f, OrigMaxSeqLen: 4096);
        var config = BaseConfig(rope);
        var overrides = new RoPEOverrideOptions { FreqBase = 500000f };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.NotNull(result.RoPEConfig);
        Assert.Equal(500000f, result.RoPEConfig!.Value.Theta);
        // Untouched fields carry over unchanged.
        Assert.Equal(RoPEScalingType.YaRN, result.RoPEConfig!.Value.ScalingType);
        Assert.Equal(2.0f, result.RoPEConfig!.Value.ScalingFactor);
        Assert.Equal(4096, result.RoPEConfig!.Value.OrigMaxSeqLen);
    }

    [Fact]
    public void ApplyRoPEOverride_ScalingTypeAndFactor_BothApplied()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128, ScalingType: RoPEScalingType.None);
        var config = BaseConfig(rope);
        var overrides = new RoPEOverrideOptions { ScalingType = RoPEScalingType.Linear, ScalingFactor = 4.0f };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.Equal(RoPEScalingType.Linear, result.RoPEConfig!.Value.ScalingType);
        Assert.Equal(4.0f, result.RoPEConfig!.Value.ScalingFactor);
        Assert.Equal(10000f, result.RoPEConfig!.Value.Theta); // untouched
    }

    [Fact]
    public void ApplyRoPEOverride_YarnParameters_AllOverridable()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128);
        var config = BaseConfig(rope);
        var overrides = new RoPEOverrideOptions
        {
            OrigMaxSeqLen = 8192,
            AttnFactor = 0.5f,
            BetaFast = 16.0f,
            BetaSlow = 2.0f,
        };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.Equal(8192, result.RoPEConfig!.Value.OrigMaxSeqLen);
        Assert.Equal(0.5f, result.RoPEConfig!.Value.AttnFactor);
        Assert.Equal(16.0f, result.RoPEConfig!.Value.BetaFast);
        Assert.Equal(2.0f, result.RoPEConfig!.Value.BetaSlow);
    }

    [Fact]
    public void ApplyRoPEOverride_AppliesToBothRopeConfigAndGlobalRopeConfig()
    {
        var rope = new RoPEConfig(Theta: 10000f, DimensionCount: 128);
        var globalRope = new RoPEConfig(Theta: 10000f, DimensionCount: 128);
        var config = BaseConfig(rope, globalRope);
        var overrides = new RoPEOverrideOptions { FreqBase = 1000000f };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.Equal(1000000f, result.RoPEConfig!.Value.Theta);
        Assert.Equal(1000000f, result.GlobalRoPEConfig!.Value.Theta);
    }

    /// <summary>
    /// Issue #1101: Qwen3.8's opt-in YaRN (factor 4 over 262,144) raises the window to 1,048,576, so the server fit,
    /// <c>--ctx-size</c> and the KV growth limits (all read <c>MaxSequenceLength</c>) allow it.
    /// </summary>
    [Fact]
    public void ApplyRoPEOverride_Yarn_RaisesWindowToFactorTimesOriginal()
    {
        var config = BaseConfig(new RoPEConfig(Theta: 1e7f, DimensionCount: 64, Type: RoPEType.NeoX)) with
        {
            Architecture = Architecture.Qwen3HybridDense,
            MaxSequenceLength = 262144,
        };
        var overrides = new RoPEOverrideOptions
        {
            ScalingType = RoPEScalingType.YaRN,
            ScalingFactor = 4.0f,
            OrigMaxSeqLen = 262144,
        };

        var result = GgufModelConfigExtractor.ApplyRoPEOverride(config, overrides);

        Assert.True(result.RoPEConfig!.Value.IsDenseYarnActive);
        Assert.Equal(1_048_576, result.MaxSequenceLength);
    }

    /// <summary>A YaRN window below the checkpoint's own never lowers it; a non-YaRN override leaves it alone.</summary>
    [Fact]
    public void ApplyRoPEOverride_NeverLowersWindow_AndIgnoresNonYarn()
    {
        var config = BaseConfig(new RoPEConfig(Theta: 1e7f, DimensionCount: 64)) with { MaxSequenceLength = 262144 };

        var smallYarn = GgufModelConfigExtractor.ApplyRoPEOverride(config, new RoPEOverrideOptions
        {
            ScalingType = RoPEScalingType.YaRN, ScalingFactor = 2.0f, OrigMaxSeqLen = 4096,
        });
        Assert.Equal(262144, smallYarn.MaxSequenceLength);

        var linear = GgufModelConfigExtractor.ApplyRoPEOverride(config, new RoPEOverrideOptions
        {
            ScalingType = RoPEScalingType.Linear, ScalingFactor = 4.0f, OrigMaxSeqLen = 262144,
        });
        Assert.Equal(262144, linear.MaxSequenceLength);

        var freqOnly = GgufModelConfigExtractor.ApplyRoPEOverride(config, new RoPEOverrideOptions { FreqBase = 5e6f });
        Assert.Equal(262144, freqOnly.MaxSequenceLength);
    }

    [Fact]
    public void RoPEOverrideOptions_HasAnyOverride_FalseWhenAllNull()
    {
        Assert.False(new RoPEOverrideOptions().HasAnyOverride);
    }

    [Fact]
    public void RoPEOverrideOptions_HasAnyOverride_TrueWhenOneFieldSet()
    {
        Assert.True(new RoPEOverrideOptions { BetaSlow = 1.5f }.HasAnyOverride);
    }
}
