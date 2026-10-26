using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1063 addendum to #403: the pure device-memory estimate the CUDA Nemotron-H loader checks against free VRAM before
/// uploading a <c>nemotron_h_moe</c> model. CPU-only (no device work): the weights only need shapes and types.
/// </summary>
public sealed class CudaNemotronHMoeDeviceBudgetTests
{
    private const int Hidden = 512, Vocab = 16, Latent = 256, Inter = 256, Shared = 64, Experts = 4, TopK = 2;

    [Fact]
    public void Weights_CountEveryUploadIncludingAllExpertBanks()
    {
        var b = Estimate(withMoe: true, planTokens: 1, block: 512);

        long outputAndNorm = Hidden * 4 + (long)Vocab * Hidden * 4;
        long ssmLayer = Hidden * 4                      // attn_norm
            + 25L * Hidden * 4 + Hidden * 8L * 4        // ssm_in [512 -> 25], ssm_out [8 -> 512], F32
            + (4 * 16 + 16 + 1 + 1 + 1 + 8) * 4;        // conv1d w/b, A, D, dt bias, norm
        long moeLayer = Hidden * 4                      // attn_norm
            + (long)Experts * Hidden * 4                // router F32
            + 2L * Shared * Hidden * 4                  // shared up/down F32
            + Experts * (Inter * 144L)                  // up bank: Q4_K rows of K=256 (144 B) x 256 rows
            + Experts * (Latent * 272L)                 // down bank: Q8_0 rows of K=256 (272 B) x 256 rows
            + Latent * 544L + Hidden * 272L;            // latent_down Q8_0 [512 -> 256], latent_up Q8_0 [256 -> 512]
        Assert.Equal(outputAndNorm + ssmLayer + moeLayer, b.Weights);
    }

    [Theory]
    [InlineData(1, 512, 1)]
    [InlineData(7, 512, 7)]
    [InlineData(2048, 512, 512)]   // capped at the MoE block: scratch is sized per block, not per prompt
    [InlineData(100, 3, 3)]
    public void MoeScratch_IsSizedFromPerLayerMaximaForOneBlock(int planTokens, int block, int tokens)
    {
        var b = Estimate(withMoe: true, planTokens, block);
        long slots = (long)tokens * TopK;
        long floats = tokens * (long)Experts + tokens * (long)Latent + tokens * (long)Shared + tokens * (long)Hidden
                    + slots * Inter + slots * Latent + tokens * (long)Latent;
        long schedule = 2L * Experts * 8 + (Experts + 1) * 4L + 3 * slots * 4;
        Assert.Equal(floats * 4 + schedule, b.MoeScratch);
    }

    [Fact]
    public void NoMoeLayer_HasNoMoeScratch_AndStateGrowsWithPlannedTokens()
    {
        Assert.Equal(0, Estimate(withMoe: false, planTokens: 512, block: 512).MoeScratch);
        var small = Estimate(withMoe: true, planTokens: 1, block: 512);
        var large = Estimate(withMoe: true, planTokens: 512, block: 512);
        Assert.True(large.ForwardState > small.ForwardState);
        Assert.True(large.GemmScratch > small.GemmScratch);
        Assert.Equal(large.Weights, small.Weights);
        Assert.Equal(large.Weights + large.ForwardState + large.MoeScratch + large.GemmScratch, large.Total);
    }

    private static CudaNemotronHTransformerModel.DeviceBudget Estimate(bool withMoe, int planTokens, int block)
    {
        var ssmLayer = new NemotronHLayerWeights
        {
            AttnNormWeight = new float[Hidden],
            Ssm = new NemotronHSsmWeights
            {
                InWeight = 1, InQuantType = QuantizationType.F32, InInputDim = Hidden, InOutputDim = 25,
                Conv1dWeight = new float[4 * 16], Conv1dBias = new float[16], A = new float[1], D = new float[1],
                DtBias = new float[1], NormWeight = new float[8],
                OutWeight = 1, OutQuantType = QuantizationType.F32, OutInputDim = 8, OutOutputDim = Hidden,
            },
        };
        var moeLayer = new NemotronHLayerWeights
        {
            AttnNormWeight = new float[Hidden],
            Moe = new NemotronHMoeWeights
            {
                GateInpWeight = 1, GateInpQuantType = QuantizationType.F32, SelectionBias = new float[Experts],
                UpExpsWeight = 1, UpExpsQuantType = QuantizationType.Q4_K, UpPerExpertBytes = Inter * 144L,
                DownExpsWeight = 1, DownExpsQuantType = QuantizationType.Q8_0, DownPerExpertBytes = Latent * 272L,
                UpShexpWeight = 1, UpShexpQuantType = QuantizationType.F32,
                DownShexpWeight = 1, DownShexpQuantType = QuantizationType.F32,
                NumExperts = Experts, NumExpertsPerTok = TopK, MoeIntermediateSize = Inter, SharedIntermediateSize = Shared,
                NormalizeWeights = true, WeightsScale = 5f,
                LatentSize = Latent,
                LatentDownWeight = 1, LatentDownQuantType = QuantizationType.Q8_0,
                LatentUpWeight = 1, LatentUpQuantType = QuantizationType.Q8_0,
            },
        };
        NemotronHLayerWeights[] layers = withMoe ? [ssmLayer, moeLayer] : [ssmLayer];
        var config = new ModelConfig
        {
            Architecture = Architecture.NemotronHMoe,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Inter,
            NumLayers = layers.Length,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HeadDim = 64,
            MaxSequenceLength = 4096,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            ActivationFunction = ActivationFunction.ReluSquared,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            HybridLayout = new HybridLayerLayout
            {
                LayerKind = withMoe ? [HybridLayerKind.Ssm, HybridLayerKind.Ffn] : [HybridLayerKind.Ssm],
                HeadCountKv = withMoe ? [0, 0] : [0],
                FeedForwardLength = withMoe ? [0, Inter] : [0],
            },
            SsmConfig = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 1),
            ChatTemplate = null,
        };
        return CudaNemotronHTransformerModel.EstimateDeviceBytes(
            config, layers, QuantizationType.F32, Vocab, Hidden, planTokens, block);
    }
}
