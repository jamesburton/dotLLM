using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;
using CloverLayout = DotLLM.Models.Gguf.SyntheticQwen35MoeGguf.CloverLayout;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// DiffuSpace Clover (<c>clover_moe</c>, issue #1794) causal path on the CPU Qwen3MoeHybrid host. The synthetic Clover fixtures carry
/// the SAME trunk weights as the plain qwen35moe fixture (same PRNG stream), so the causal logits must agree. The official layout stores
/// <c>ssm_a</c> as the raw <c>A_log</c> and adds four inert preview tensors per GDN layer; both must be handled for the logits to match.
/// </summary>
public sealed class CloverCausalTests : IDisposable
{
    private static readonly int[] Tokens = [1, 3, 0, 2, 5, 4];
    private readonly string _scratch;

    public CloverCausalTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-clover-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Write(CloverLayout layout, int blockCount = 4) =>
        SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, $"{layout}-{blockCount}.gguf"), blockCount: blockCount, clover: layout);

    private static float[] Logits(string path, Func<ModelConfig, ModelConfig>? edit = null)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (edit is not null) config = edit(config);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        int[] positions = Enumerable.Range(0, Tokens.Length).ToArray();
        using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using ITensor logits = model.Forward(Tokens, positions, deviceId: -1, kv);
        var result = new float[Tokens.Length * config.VocabSize];
        unsafe
        {
            new ReadOnlySpan<float>((void*)logits.DataPointer, result.Length).CopyTo(result);
        }
        return result;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Extract_CloverMoe_MapsToQwen3MoeHybridWithCloverSidecar()
    {
        using var gguf = GgufFile.Open(Write(CloverLayout.Official));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        var clover = Assert.IsType<CloverConfig>(config.Clover);
        Assert.Equal(32, clover.BlockSize);
        Assert.Equal(SyntheticQwen35MoeGguf.VocabSize - 1, clover.MaskTokenId);
        Assert.True(clover.MoeBlockCoresetEnabled);
        Assert.Equal(0.9f, clover.MoeBlockCoresetTopP);
        // The shared expert keeps its sigmoid gate (qwen35moe convention: implicit count 1 from expert_shared_feed_forward_length).
        Assert.Equal(1, config.Moe!.NumSharedExperts);
        Assert.True(config.Moe.HasSharedExpertGate);

        using var plain = GgufFile.Open(Write(CloverLayout.None));
        Assert.Null(GgufModelConfigExtractor.Extract(plain.Metadata).Clover);
    }

    /// <summary>
    /// The official layout (A_log under <c>ssm_a</c> + preview tensors) gives the qwen35moe logits. A_log = ln(-a) is rounded once on
    /// write and -exp once on load, so the tolerance is a few float ULPs of the decay, far below the negative control's gap.
    /// </summary>
    [Fact]
    public void OfficialLayout_CausalLogitsMatchQwen35Moe()
    {
        float[] reference = Logits(Write(CloverLayout.None));
        float[] clover = Logits(Write(CloverLayout.Official));
        float diff = MaxAbsDiff(reference, clover);
        Assert.True(diff < 1e-5f, $"Clover official-layout logits differ from qwen35moe by {diff:E3}");
    }

    /// <summary>llama.cpp PR #31306's converter writes qwen35moe-form <c>ssm_a</c>: it must be left alone (bit-identical logits).</summary>
    [Fact]
    public void LlamaCppConverterLayout_CausalLogitsBitIdenticalToQwen35Moe()
    {
        float[] reference = Logits(Write(CloverLayout.None));
        float[] clover = Logits(Write(CloverLayout.LlamaCppConverter));
        Assert.Equal(0f, MaxAbsDiff(reference, clover));
    }

    /// <summary>
    /// Sensitivity arm for <see cref="OfficialLayout_CausalLogitsMatchQwen35Moe"/>: reading the official file WITHOUT the Clover
    /// sidecar (so <c>ssm_a</c> is taken as qwen35moe's <c>-exp(A_log)</c>) must move the logits far beyond that test's tolerance,
    /// otherwise the fixture could not tell the transform from its absence.
    /// </summary>
    [Fact]
    public void OfficialLayout_WithoutALogTransform_Diverges()
    {
        float[] reference = Logits(Write(CloverLayout.None));
        float[] untransformed = Logits(Write(CloverLayout.Official), c => c with { Clover = null });
        float diff = MaxAbsDiff(reference, untransformed);
        Assert.True(diff > 1e-3f, $"untransformed A_log only moved the logits by {diff:E3}: the fixture cannot see the transform");
    }

    [Fact]
    public void SsmAIsALog_PositiveEntryProvesALog_EvenWithoutPreviewTensors()
    {
        using var gguf = GgufFile.Open(Write(CloverLayout.LlamaCppConverter));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tensors = gguf.TensorsByName;

        Assert.False(CloverGgufConventions.SsmAIsALog(config, tensors, "blk.0", [-0.5f, -0.25f]));
        Assert.True(CloverGgufConventions.SsmAIsALog(config, tensors, "blk.0", [-0.5f, 0.25f]));
        // Never for a non-Clover model, whatever the values.
        Assert.False(CloverGgufConventions.SsmAIsALog(config with { Clover = null }, tensors, "blk.0", [4f]));

        float[] a = [0f, MathF.Log(0.5f)];
        CloverGgufConventions.ALogToSsmA(a);
        Assert.Equal(-1f, a[0]);
        Assert.Equal(-0.5f, a[1], 6);
    }
}
