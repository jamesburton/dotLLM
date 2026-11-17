using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1851: the sizes <see cref="CudaAutoSplit"/> feeds the planner, and which models it refuses to split. Pure: no GPU.
/// </summary>
public sealed class CudaAutoSplitTests
{
    private static ModelConfig Config(int layers = 2) => new()
    {
        Architecture = Architecture.Llama, VocabSize = 1000, HiddenSize = 256, IntermediateSize = 512, NumLayers = layers,
        NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 64, MaxSequenceLength = 32768, NormEpsilon = 1e-5f,
    };

    // Q8_0: 34 bytes per 32 elements. A [256 x 256] projection = 2,048 blocks = 69,632 bytes raw.
    private const long Q8Proj = 256L * 256 / 32 * 34;

    private static (string, int, long, QuantizationType) T(string name, long elements, QuantizationType qt = QuantizationType.Q8_0, int inner = 256)
        => (name, inner, elements, qt);

    [Fact]
    public void LayerBytes_AreRawBlocks_PlusAnFp16CopyOnlyForProjectionsWithoutAGemv()
    {
        var tensors = new[]
        {
            T("blk.0.attn_q.weight", 256 * 256),
            T("blk.0.attn_norm.weight", 256, QuantizationType.F32),
            T("blk.1.attn_q.weight", 256 * 256, QuantizationType.Q4_0),
            T("token_embd.weight", 1000 * 256),
            T("output.weight", 1000 * 256),
        };
        var (layers, _, _) = CudaAutoSplit.EstimateWeightBytes(tensors, Config(),
            runsQuantized: qt => qt != QuantizationType.Q4_0, hasEmbeddingLookup: _ => true);
        Assert.Equal(Q8Proj + 256 * 4, layers[0]);                          // no FP16 copy; the F32 norm counted raw
        Assert.Equal(256L * 256 / 32 * 18 + 256L * 256 * 2, layers[1]);    // Q4_0 has no GEMV here: raw + FP16 copy
    }

    [Fact]
    public void Embedding_IsFp16Table_WhenNoRowLookupKernel()
    {
        var tensors = new[] { T("token_embd.weight", 1000 * 256), T("output.weight", 1000 * 256) };
        var (_, withLookup, _) = CudaAutoSplit.EstimateWeightBytes(tensors, Config(), _ => true, _ => true);
        var (_, without, _) = CudaAutoSplit.EstimateWeightBytes(tensors, Config(), _ => true, _ => false);
        Assert.Equal(1000L * 256 / 32 * 34, withLookup);
        Assert.Equal(1000L * 256 * 2, without);
    }

    [Fact]
    public void Head_IsTheOutputTensor_OrTheEmbeddingAgainWhenTied()
    {
        long embRaw = 1000L * 256 / 32 * 34;
        var untied = new[] { T("token_embd.weight", 1000 * 256), T("output.weight", 1000 * 256, QuantizationType.Q4_0), T("output_norm.weight", 256, QuantizationType.F32) };
        var (_, _, head) = CudaAutoSplit.EstimateWeightBytes(untied, Config(), _ => true, _ => true);
        Assert.Equal(1000L * 256 / 32 * 18 + 256 * 4, head);

        var tied = new[] { T("token_embd.weight", 1000 * 256), T("output_norm.weight", 256, QuantizationType.F32) };
        var (_, _, tiedHead) = CudaAutoSplit.EstimateWeightBytes(tied, Config(), _ => true, _ => true);
        Assert.Equal(embRaw + 256 * 4, tiedHead);
    }

    [Fact]
    public void PlannerInput_ChargesF16KvPerLayer_AndPlansTheContextDefault()
    {
        var config = Config(layers: 4);
        var m = new CudaAutoSplit.Measurement(1L << 30, ActivationBytesPerToken: 100, PendingLazyBytes: 7, LayerBytes: [1, 2, 3, 4],
            EmbeddingBytes: 1000, HeadBytes: 50);
        var input = CudaAutoSplit.BuildInput(config, m, 1L << 30, 256L << 20, ctx: 4096, igpu: null);
        Assert.Equal(2L * 2 * 64 * 2, input.KvBytesPerTokenPerLayer);   // K + V, 2 KV heads x 64, F16
        Assert.Equal(1007, input.DgpuFixedBytes);
        Assert.Equal(4096, input.ActivationTokens);
        Assert.Equal(CudaAutoSplit.DefaultPlanContextTokens, CudaAutoSplit.PlanContext(null, 32768));
        Assert.Equal(2048, CudaAutoSplit.PlanContext(null, 2048));
    }

    [Fact]
    public void Blocker_RefusesWhatTheCpuTailCannotRun()
    {
        Assert.Null(GpuOffloadPlanner.AutoSplitBlocker(Config()));
        Assert.Contains("MoE", GpuOffloadPlanner.AutoSplitBlocker(Config() with { Moe = new MoeConfig { NumExperts = 8, NumExpertsPerTok = 2, MoeIntermediateSize = 64 } }));
        Assert.Contains("#432", GpuOffloadPlanner.AutoSplitBlocker(Config() with { SlidingWindowSize = 128, SlidingWindowPattern = 2 }));
        Assert.Contains("sinks", GpuOffloadPlanner.AutoSplitBlocker(Config(), hasAttentionSinks: true));
        Assert.Contains("cannot be split", GpuOffloadPlanner.AutoSplitBlocker(Config() with { Architecture = Architecture.NemotronH }));
        // A uniform window (Mistral) is what the tail already honours.
        Assert.Null(GpuOffloadPlanner.AutoSplitBlocker(Config() with { SlidingWindowSize = 4096 }));
    }
}
