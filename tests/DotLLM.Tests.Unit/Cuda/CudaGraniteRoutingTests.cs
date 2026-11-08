using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Which CUDA forward serves dense Granite (#1669). A Granite model whose only scalar is the attention multiplier
/// (TwIL-8B / Granite-4.2) runs the generic FP16 loop with the score scale folded into the RoPE multiplier; a model with
/// any other non-trivial scalar (Granite-3.x: embedding 12, residual 0.22, logit 8) stays on the FP32 dense forward that
/// implements them. No GPU needed: both decisions are config predicates.
/// </summary>
public sealed class CudaGraniteRoutingTests
{
    private static ModelConfig Config(float? embed, float? residual, float? logit, float? attn, int headDim = 128, int ropeDim = 0) => new()
    {
        Architecture = Architecture.Granite,
        VocabSize = 100, HiddenSize = headDim * 4, IntermediateSize = 256, NumLayers = 2,
        NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = headDim, MaxSequenceLength = 128,
        EmbeddingScale = embed, ResidualScale = residual, LogitScale = logit, AttentionScale = attn,
        RoPEConfig = new RoPEConfig(Theta: 1e7f, DimensionCount: ropeDim),
    };

    [Fact]
    public void TwilShape_FoldsAttentionScaleIntoRope_AndSkipsF32Forward()
    {
        var c = Config(1f, 1f, 1f, 0.0078125f);
        Assert.True(c.GraniteScalarsFoldIntoRope);
        Assert.False(CudaTransformerModel.UsesF32DenseForward(c));
        // mscale^2 turns the kernels' fixed 1/sqrt(128) into 1/128.
        float m = c.GraniteAttentionRopeMscale;
        Assert.Equal(0.0078125, m * m / MathF.Sqrt(128), 7);
        Assert.NotEqual(1f, m);
    }

    [Fact]
    public void NullScalarsEqualOne()
    {
        Assert.True(Config(null, null, null, 0.0078125f).GraniteScalarsFoldIntoRope);
    }

    [Theory]
    [InlineData(12f, 0.22f, 8f, 0.0078125f)]   // Granite-3.3-2b
    [InlineData(1f, 0.22f, 1f, 0.0078125f)]    // residual only
    [InlineData(1f, 1f, 8f, 0.0078125f)]       // logit only
    [InlineData(12f, 1f, 1f, 0.0078125f)]      // embedding only
    public void AnyOtherScalar_StaysOnF32Forward(float embed, float residual, float logit, float attn)
    {
        var c = Config(embed, residual, logit, attn);
        Assert.False(c.GraniteScalarsFoldIntoRope);
        Assert.True(CudaTransformerModel.UsesF32DenseForward(c));
    }

    [Fact]
    public void PartialRotary_StaysOnF32Forward_BecauseTheTailIsNotRotated()
    {
        var c = Config(1f, 1f, 1f, 0.0078125f, headDim: 128, ropeDim: 64);
        Assert.False(c.GraniteScalarsFoldIntoRope);
        Assert.True(CudaTransformerModel.UsesF32DenseForward(c));
    }

    [Fact]
    public void FoldedModelUsesTheModelsOwnScoreScale_NotAHardCodedConstant()
    {
        var a = Config(1f, 1f, 1f, 1f / 64, headDim: 64);
        var b = Config(1f, 1f, 1f, 1f / 128, headDim: 64);
        Assert.NotEqual(a.GraniteAttentionRopeMscale, b.GraniteAttentionRopeMscale);
        Assert.Equal(1f / 64, a.GraniteAttentionRopeMscale * a.GraniteAttentionRopeMscale / MathF.Sqrt(64), 6);
    }
}
