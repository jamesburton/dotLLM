using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #743 — llama3-style <c>rope_freqs.weight</c> must reach the MAIN RoPE table of dense models.
/// The expected values below are computed from the analytic ggml formula
/// (<c>angle = pos * theta^(-2i/d) / factor[i]</c>), never from another dotLLM backend.
/// </summary>
public sealed class DenseRopeFreqFactorsTests
{
    private const int HeadDim = 64;
    private const float Theta = 500000f;

    /// <summary>Llama-3.1 recipe: factor 1 for high-frequency pairs, 8 for low, linear blend between.</summary>
    private static float[] Llama3Factors()
    {
        var f = new float[HeadDim / 2];
        for (int i = 0; i < f.Length; i++)
        {
            double wavelen = 2 * Math.PI * Math.Pow(Theta, 2.0 * i / HeadDim);
            double lowWl = 8192.0 / 1.0, highWl = 8192.0 / 4.0;
            if (wavelen < highWl) f[i] = 1f;
            else if (wavelen > lowWl) f[i] = 8f;
            else
            {
                double smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
                f[i] = (float)(1.0 / ((1 - smooth) / 8.0 + smooth));
            }
        }
        return f;
    }

    [Fact]
    public void ForwardState_MainTable_AppliesFactors_MatchingAnalyticFormula()
    {
        float[] factors = Llama3Factors();
        Assert.Contains(factors, v => v > 1.5f); // non-degenerate: some pairs really are rescaled
        Assert.Contains(factors, v => v == 1f);

        const int maxSeq = 300;
        using var withF = new TransformerForwardState(
            256, 8, 2, HeadDim, 512, 1000, maxSeq, HeadDim, Theta, ropeFreqFactors: factors);
        using var plain = new TransformerForwardState(
            256, 8, 2, HeadDim, 512, 1000, maxSeq, HeadDim, Theta);

        int half = HeadDim / 2;
        bool anyDiffers = false;
        for (int pos = 0; pos < maxSeq; pos += 7)
        for (int i = 0; i < half; i++)
        {
            double angle = pos * Math.Pow(Theta, -2.0 * i / HeadDim) / factors[i];
            Assert.InRange(withF.CosTable[pos * half + i], Math.Cos(angle) - 2e-4, Math.Cos(angle) + 2e-4);
            Assert.InRange(withF.SinTable[pos * half + i], Math.Sin(angle) - 2e-4, Math.Sin(angle) + 2e-4);
            if (Math.Abs(withF.SinTable[pos * half + i] - plain.SinTable[pos * half + i]) > 1e-3) anyDiffers = true;
        }
        Assert.True(anyDiffers, "factors must change the table; otherwise this test cannot detect an ignored tensor");
    }

    private static ModelConfig Config(Architecture arch, RoPEConfig rope, RoPEConfig? global = null) => new()
    {
        Architecture = arch,
        VocabSize = 1000,
        HiddenSize = 256,
        IntermediateSize = 512,
        NumLayers = 2,
        NumAttentionHeads = 8,
        NumKvHeads = 2,
        HeadDim = HeadDim,
        MaxSequenceLength = 512,
        RoPEConfig = rope,
        GlobalRoPEConfig = global,
    };

    [Fact]
    public void Select_DenseLlama_ReturnsFactors()
    {
        float[] f = Llama3Factors();
        var cfg = Config(Architecture.Llama, new RoPEConfig(Theta, HeadDim));
        Assert.Same(f, DenseRopeFreqFactors.Select(cfg, f, HeadDim));
    }

    [Fact]
    public void Select_NoTensor_ReturnsNull()
        => Assert.Null(DenseRopeFreqFactors.Select(Config(Architecture.Llama, new RoPEConfig(Theta, HeadDim)), null, HeadDim));

    [Fact]
    public void Select_Gemma4WithGlobalTable_KeepsLocalLayersFactorFree()
    {
        float[] f = Llama3Factors();
        var cfg = Config(Architecture.Gemma4, new RoPEConfig(10000f, HeadDim),
            global: new RoPEConfig(1000000f, HeadDim));
        Assert.Null(DenseRopeFreqFactors.Select(cfg, f, HeadDim));
    }

    [Fact]
    public void Select_FactorsPlusYarn_ThrowsInsteadOfSilentlyDroppingOne()
    {
        var rope = new RoPEConfig(Theta, HeadDim, ScalingType: RoPEScalingType.YaRN, ScalingFactor: 8f, OrigMaxSeqLen: 8192);
        Assert.Throws<NotSupportedException>(
            () => DenseRopeFreqFactors.Select(Config(Architecture.Llama, rope), Llama3Factors(), HeadDim));
    }

    [Fact]
    public void Select_WrongLength_Throws()
        => Assert.Throws<NotSupportedException>(
            () => DenseRopeFreqFactors.Select(Config(Architecture.Llama, new RoPEConfig(Theta, HeadDim)), new float[3], HeadDim));
}
