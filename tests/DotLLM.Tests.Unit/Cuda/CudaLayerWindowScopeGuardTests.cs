using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cuda.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Proves that <c>CudaLayerWindowModel</c>'s architecture scope guard actually fires for every
/// feature its CUDA layer-window forward pass does not implement.
/// </summary>
/// <remarks>
/// <para><b>Why this is worth its own file (issue #395).</b> The guard is the only thing standing
/// between an unsupported checkpoint and a <em>plausible, authoritative, wrong</em> perplexity —
/// the window loop runs happily to completion on a model whose architecture it half-implements, and
/// nothing downstream can tell. A rejection that quietly stopped firing would look exactly like a
/// passing build.</para>
/// <para>These run without a GPU and without a checkpoint on disk: the guard is a pure function of
/// <see cref="ModelConfig"/>, and the alternative — a real GGUF per rejected feature — is not
/// something the suite can carry. That is why <c>ValidateSupported</c> is internal rather than
/// private.</para>
/// <para><b>The scaled-RoPE case is the one that was missing.</b> Every other rejection here covers
/// a structural feature that changes the layer graph, and those were caught by inspection. Scaled
/// RoPE does not change the graph at all — it changes the frequency table — so a YaRN model cleared
/// every structural check and would then have been scored with the <em>unscaled</em> table, because
/// <c>CudaPipelineStage</c> passes only theta / dim / type to <c>LaunchRoPE</c> while the CPU
/// reference applies the full YaRN correction. Note this is a pre-existing backend-wide gap
/// (whole-device <c>--device cuda</c> is equally unscaled); the rejection keeps the layer-window
/// guard's own promise, it does not fix CUDA RoPE scaling.</para>
/// </remarks>
public sealed class CudaLayerWindowScopeGuardTests
{
    /// <summary>A minimal dense/GQA Llama-shaped config: the shape the window loop does implement.</summary>
    private static ModelConfig DenseConfig(RoPEConfig? rope = null) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 128,
        HiddenSize = 64,
        IntermediateSize = 128,
        NumLayers = 4,
        NumAttentionHeads = 4,
        NumKvHeads = 2,
        HeadDim = 16,
        MaxSequenceLength = 128,
        RoPEConfig = rope ?? new RoPEConfig(Theta: 10000.0f),
    };

    /// <summary>The supported shape must pass, or every rejection below proves nothing.</summary>
    /// <remarks>
    /// The control. Without it a guard that rejected <em>everything</em> would satisfy all the
    /// negative cases and still be useless.
    /// </remarks>
    [Fact]
    public void PlainDenseGqaConfig_IsAccepted()
    {
        CudaLayerWindowModel.ValidateSupported(DenseConfig());
    }

    /// <summary>Unscaled RoPE is the only frequency-table configuration the window loop implements.</summary>
    [Theory]
    [InlineData(RoPEScalingType.YaRN)]
    [InlineData(RoPEScalingType.Linear)]
    [InlineData(RoPEScalingType.NTK)]
    [InlineData(RoPEScalingType.DynamicNTK)]
    [InlineData(RoPEScalingType.Su)]
    public void ScaledRoPE_IsRejected(RoPEScalingType scaling)
    {
        ModelConfig config = DenseConfig(new RoPEConfig(
            Theta: 10000.0f, ScalingType: scaling, ScalingFactor: 8.0f));

        var ex = Assert.Throws<NotSupportedException>(() => CudaLayerWindowModel.ValidateSupported(config));
        Assert.Contains("scaled RoPE", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// Llama-3's <c>llama3</c> rope scaling maps to <see cref="RoPEScalingType.None"/> in
    /// <c>GgufModelConfigExtractor</c>, so the Llama-3.x family must keep working.
    /// </summary>
    /// <remarks>
    /// Pins the blast radius of the rejection above. Llama-3.2-1B is the fixture the CUDA cycling
    /// integration tests run on, so a guard that swept it up would silently skip them.
    /// </remarks>
    [Fact]
    public void UnscaledRoPE_IsStillAccepted()
    {
        CudaLayerWindowModel.ValidateSupported(
            DenseConfig(new RoPEConfig(Theta: 500000.0f, ScalingType: RoPEScalingType.None)));
    }

    /// <summary>A hybrid SSM+Transformer layout must be refused, not scored.</summary>
    /// <remarks>
    /// This is the path a Nemotron-H or Qwen3-hybrid checkpoint takes, and the rejection is what the
    /// CLI turns into a clean one-line error rather than a stack trace.
    /// </remarks>
    [Fact]
    public void HybridLayerLayout_IsRejected()
    {
        ModelConfig config = DenseConfig() with
        {
            HybridLayout = new HybridLayerLayout
            {
                LayerKind =
                [
                    HybridLayerKind.Ssm, HybridLayerKind.Attention,
                    HybridLayerKind.Ssm, HybridLayerKind.Attention,
                ],
                HeadCountKv = [0, 2, 0, 2],
                FeedForwardLength = [128, 128, 128, 128],
            },
        };

        var ex = Assert.Throws<NotSupportedException>(() => CudaLayerWindowModel.ValidateSupported(config));
        Assert.Contains("hybrid", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Partial rotary embeddings change what RoPE covers, so they are refused.</summary>
    [Fact]
    public void PartialRotary_IsRejected()
    {
        ModelConfig config = DenseConfig() with { PartialRotaryFactor = 0.5f };

        Assert.Throws<NotSupportedException>(() => CudaLayerWindowModel.ValidateSupported(config));
    }
}
