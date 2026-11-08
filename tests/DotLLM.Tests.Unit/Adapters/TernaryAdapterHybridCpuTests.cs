using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Tests.Unit.Vulkan.Ternary;
using Xunit;
using Fixture = DotLLM.Tests.Unit.Vulkan.VulkanHybridLegacyQuantForwardTests.LegacyQuantFixture;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>
/// Ternary residual adapters on the CPU qwen35 hybrid-dense model (Bonsai-2, Qwen3.5/3.6 dense hybrids; #1711): the PostLayer hook must follow
/// the same convention as <see cref="TransformerModel"/> (applied after layer L's output, GDN and attention layers alike, final layer
/// included, every row, every forward shape). Uses the 2-layer synthetic hybrid of <see cref="VulkanHybridLegacyQuantForwardTests"/> (layer 0
/// Gated-DeltaNet, layer 1 full attention).
/// </summary>
public sealed unsafe class TernaryAdapterHybridCpuTests
{
    private const int Vocab = 8, KvHeads = 2, HeadDim = 64, MaxSeq = 16;
    private const QuantizationType Qt = QuantizationType.Q4_1;
    private static readonly int[] Ids = [1, 5, 2, 7, 3, 0];

    private static Qwen3HybridDenseTransformerModel Build(Fixture fx)
        => Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32,
            fx.OutputWeightPtr, Qt, Vocab, fx.DenseConfig.HiddenSize);

    private static float[] Run(Qwen3HybridDenseTransformerModel m, int[] ids, int firstPos = 0, SimpleKvCache? kv = null)
    {
        bool own = kv is null;
        if (own) m.ResetSequenceState();   // the model-owned GDN state persists across forwards; a fresh sequence starts from zero
        kv ??= new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
        try
        {
            using ITensor t = m.Forward(ids, Enumerable.Range(firstPos, ids.Length).ToArray(), -1, kv);
            return new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        }
        finally { if (own) kv!.Dispose(); }
    }

    private static NsAdapterPackage Adapter(int d, int layer, float alpha, ulong seed = 77)
        => VulkanTernaryAdapterKernelTests.Synth(8, d, seed, alpha, 0f, layer);

    private static double MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double m = 0; for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m;
    }

    private static void AssertBits(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]), $"{what}: element {i} {a[i]} != {b[i]}");
    }

    private sealed class Recorder : IResidualModifier
    {
        public readonly List<(int layer, int tokens)> Calls = new();
        public bool AppliesTo(int layerIndex) => true;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) => Calls.Add((layerIndex, tokenCount));
    }

    /// <summary>Overwrites the residual rows at one layer with fixed rows (independent of everything upstream).</summary>
    private sealed class Overwrite(int layer, float[] rows, int d) : IResidualModifier
    {
        public bool AppliesTo(int layerIndex) => layerIndex == layer;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) => rows.AsSpan(0, tokenCount * d).CopyTo(residual);
    }

    [Fact]
    public void AdapterChangesLogits_DetachAndAlphaZeroAreBitExact_AtEveryLayerKind()
    {
        using var fx = Fixture.Build(1711, Qt);
        using var m = Build(fx);
        int d = fx.DenseConfig.HiddenSize;
        Assert.Null(m.ResidualModifier);
        float[] baseline = Run(m, Ids);
        AssertBits(baseline, Run(m, Ids), "determinism");

        foreach (int layer in new[] { 0, 1 })   // 0 = Gated-DeltaNet layer, 1 = full-attention (= final) layer
        {
            using var hook = new TernaryResidualHook(d);
            hook.Add(Adapter(d, layer, 3f));
            hook.ValidateFor(d, fx.DenseConfig.NumLayers);
            m.ResidualModifier = hook;
            double shift = MaxDiff(baseline, Run(m, Ids));
            Assert.True(shift > 1e-3, $"layer {layer}: adapter did not change logits (max diff {shift})");
            m.ResidualModifier = null;
            AssertBits(baseline, Run(m, Ids), $"layer {layer}: after detach");

            using var zero = new TernaryResidualHook(d);
            zero.Add(Adapter(d, layer, 0f));
            m.ResidualModifier = zero;
            AssertBits(baseline, Run(m, Ids), $"layer {layer}: alpha=0 identity");
            m.ResidualModifier = null;
        }
    }

    [Fact]
    public void Modifier_RunsAfterEachLayer_InOrder_WithAllRows_OnPrefillAndDecode()
    {
        using var fx = Fixture.Build(1712, Qt);
        using var m = Build(fx);
        var rec = new Recorder();
        m.ResidualModifier = rec;
        using var kv = new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
        _ = Run(m, Ids, 0, kv);
        _ = Run(m, [4], Ids.Length, kv);
        Assert.Equal(new[] { (0, Ids.Length), (1, Ids.Length), (0, 1), (1, 1) }, rec.Calls.ToArray());
    }

    /// <summary>
    /// The layer-index convention, proved structurally (a one-layer-off mutant fails both halves): a modifier that OVERWRITES the residual after layer 0
    /// with the raw embedding rows must make the 2-layer model bit-identical to a 1-layer model made of layer 1 alone, i.e. the hook runs after layer 0
    /// and before layer 1. And overwriting after the LAST layer must make the logits independent of the tokens (hook runs after the final layer, before
    /// the final norm / head), the same convention as <see cref="TransformerModel"/>.
    /// </summary>
    [Fact]
    public void LayerIndexConvention_MatchesTransformerModel_AfterLayerL_FinalLayerIncluded()
    {
        using var fx = Fixture.Build(1713, Qt);
        int d = fx.DenseConfig.HiddenSize;
        var emb = new float[Ids.Length * d];   // the F32 embedding rows of Ids: what the layer stack receives
        for (int t = 0; t < Ids.Length; t++)
            new ReadOnlySpan<float>((float*)fx.TokenEmbedPtr + (long)Ids[t] * d, d).CopyTo(emb.AsSpan(t * d, d));

        var cfg1 = fx.DenseConfig with
        {
            NumLayers = 1,
            HybridLayout = new HybridLayerLayout
            {
                LayerKind = [HybridLayerKind.Attention], HeadCountKv = [KvHeads], FeedForwardLength = [fx.DenseConfig.IntermediateSize],
            },
        };
        float[] expected;
        using (var one = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            cfg1, [fx.DenseLayers[1]], fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32, fx.OutputWeightPtr, Qt, Vocab, d))
            expected = Run(one, Ids);

        using var m = Build(fx);
        float[] baseline = Run(m, Ids);
        m.ResidualModifier = new Overwrite(0, emb, d);
        float[] got = Run(m, Ids);
        AssertBits(expected, got, "overwrite-after-layer-0 == layer-1-only model");
        Assert.True(MaxDiff(baseline, got) > 1e-3, "control: the 2-layer model must differ from the layer-1-only result without the overwrite");

        var fixedRows = new float[Ids.Length * d];
        for (int i = 0; i < fixedRows.Length; i++) fixedRows[i] = 0.01f * ((i % 13) - 6);
        m.ResidualModifier = new Overwrite(1, fixedRows, d);
        float[] a = Run(m, Ids);
        float[] b = Run(m, [6, 6, 4, 4, 1, 2]);
        AssertBits(a, b, "overwrite-after-final-layer is token independent");
        Assert.True(MaxDiff(a, baseline) > 1e-3);
    }

    [Fact]
    public void ChunkedPrefill_Decode_AndBatch_WithAdapter_AgreeWithOneShotWithinBaselineNoise()
    {
        using var fx = Fixture.Build(1714, Qt);
        using var m = Build(fx);
        int d = fx.DenseConfig.HiddenSize;
        int last = (Ids.Length - 1) * Vocab;

        float[] Chunked(int[] sizes)
        {
            using var kv = new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
            m.ResetSequenceState();
            int pos = 0; float[] lastLogits = [];
            foreach (int n in sizes)
            {
                float[] l = Run(m, Ids[pos..(pos + n)], pos, kv);
                lastLogits = l[^Vocab..];
                pos += n;
            }
            return lastLogits;
        }
        float[] OneShot() { m.ResetSequenceState(); return Run(m, Ids)[last..]; }

        float[] baseOne = OneShot();
        double baseNoise = Math.Max(MaxDiff(baseOne, Chunked([3, 3])), MaxDiff(baseOne, Chunked([1, 1, 1, 1, 1, 1])));

        using var hook = new TernaryResidualHook(d);
        hook.Add(Adapter(d, 0, 3f));
        hook.Add(Adapter(d, 1, 3f, seed: 78));
        m.ResidualModifier = hook;
        float[] one = OneShot();
        double effect = MaxDiff(baseOne, one);
        double noise = Math.Max(MaxDiff(one, Chunked([3, 3])), MaxDiff(one, Chunked([1, 1, 1, 1, 1, 1])));
        Assert.True(effect > 1e-3, $"adapter effect {effect}");
        Assert.True(noise <= Math.Max(20 * baseNoise, 1e-3) && noise < effect / 10,
            $"chunked/decode with adapter diverged by {noise}, baseline chunk noise {baseNoise}, adapter effect {effect}");

        // ForwardBatch (per-sequence loop): same logits as an individual forward, adapter included.
        using var kvA = new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
        using var kvB = new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
        using var gA = (GdnStateCache)m.CreateSequenceState()!;
        using var gB = (GdnStateCache)m.CreateSequenceState()!;
        var reqs = new[]
        {
            new SequenceForwardRequest { TokenIds = Ids, Positions = Enumerable.Range(0, Ids.Length).ToArray(), KvCache = kvA, GdnState = gA },
            new SequenceForwardRequest { TokenIds = Ids[..3], Positions = new[] { 0, 1, 2 }, KvCache = kvB, GdnState = gB },
        };
        var outs = m.ForwardBatch(reqs, -1);
        try
        {
            float[] b0 = new ReadOnlySpan<float>((float*)outs[0].DataPointer, (int)outs[0].Shape.ElementCount).ToArray();
            m.ResetSequenceState();
            AssertBits(Run(m, Ids), b0, "ForwardBatch seq0 == single forward");
        }
        finally { foreach (var o in outs) o.Dispose(); }
    }
}
