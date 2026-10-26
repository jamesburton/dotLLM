using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>#839: KV geometry holds slots for attention layers only on hybrid models (layer -> slot map).</summary>
public sealed class HybridKvGeometryTests
{
    private static ModelConfig HybridConfig(HybridLayerKind[] kinds, int kvHeads = 2, int headDim = 8) => new()
    {
        Architecture = Architecture.NemotronH,
        VocabSize = 8, HiddenSize = 16, IntermediateSize = 16, NumLayers = kinds.Length,
        NumAttentionHeads = 4, NumKvHeads = kvHeads, HeadDim = headDim, MaxSequenceLength = 32,
        AttentionType = AttentionType.GQA, PositionEncodingType = PositionEncodingType.None,
        ActivationFunction = ActivationFunction.SiLU, NormType = NormType.RMSNorm, NormEpsilon = 1e-5f,
        HybridLayout = new HybridLayerLayout
        {
            LayerKind = kinds,
            HeadCountKv = kinds.Select(k => k == HybridLayerKind.Attention ? kvHeads : 0).ToArray(),
            FeedForwardLength = new int[kinds.Length],
        },
    };

    [Fact]
    public void IrregularLayout_SlotsOnlyForAttentionLayers_InLayerOrder()
    {
        var A = HybridLayerKind.Attention; var G = HybridLayerKind.GatedDeltaNet; var F = HybridLayerKind.Ffn;
        var cfg = HybridConfig([G, G, A, F, G, A, A, G, G, G]);   // attention at 2, 5, 6 (not periodic)
        var geom = KvGeometry.FromConfig(cfg);

        Assert.Equal(3, geom.LayerCount);
        Assert.Equal(3, KvGeometry.SlotCount(cfg));
        Assert.True(geom.HasLayerSlotMap);
        Assert.Equal(10, geom.ModelLayerCount);
        int[] expected = [-1, -1, 0, -1, -1, 1, 2, -1, -1, -1];
        for (int l = 0; l < 10; l++) Assert.Equal(expected[l], geom.SlotOfLayer(l));
        for (int s = 0; s < 3; s++) Assert.Equal(16, geom.KvStrideOf(s));

        using var hybrid = new SimpleKvCache(geom, 64);
        using var legacy = new SimpleKvCache(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, 64);
        Assert.Equal(legacy.AllocatedBytes * 3 / 10, hybrid.AllocatedBytes);
        Assert.Equal(3, hybrid.NumLayers);
    }

    [Fact]
    public void OneInFourHybrid_Allocates4xLess()
    {
        var kinds = Enumerable.Range(0, 48).Select(i => i % 4 == 3 ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet).ToArray();
        var cfg = HybridConfig(kinds);
        using var hybrid = new SimpleKvCache(KvGeometry.FromConfig(cfg), 128);
        using var legacy = new SimpleKvCache(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, 128);
        Assert.Equal(4, legacy.AllocatedBytes / hybrid.AllocatedBytes);
        Assert.Equal(12, hybrid.NumLayers);
    }

    [Fact]
    public void DenseModel_HasNoSlotMap_AndIsUnchanged()
    {
        var cfg = HybridConfig([HybridLayerKind.Attention, HybridLayerKind.Attention, HybridLayerKind.Attention]) with { HybridLayout = null };
        var geom = KvGeometry.FromConfig(cfg);
        Assert.False(geom.HasLayerSlotMap);
        Assert.Equal(3, geom.LayerCount);
        Assert.Equal(3, KvGeometry.SlotCount(cfg));
        for (int l = 0; l < 3; l++) Assert.Equal(l, geom.SlotOfLayer(l));
    }

    [Fact]
    public void NoAttentionLayer_KeepsOneMinimalSlot()
    {
        var cfg = HybridConfig([HybridLayerKind.GatedDeltaNet, HybridLayerKind.GatedDeltaNet]);
        Assert.Equal(1, KvGeometry.SlotCount(cfg));
        Assert.Equal(1, KvGeometry.FromConfig(cfg).LayerCount);
    }

    [Fact]
    public void QuantizedCache_OnHybridGeometry_AllocatesSlotsOnly()
    {
        var kinds = Enumerable.Range(0, 8).Select(i => i % 4 == 3 ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet).ToArray();
        var cfg = HybridConfig(kinds, kvHeads: 1, headDim: 32);
        using var q = new QuantizedKvCache(KvGeometry.FromConfig(cfg), 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        Assert.Equal(2, ((IPerLayerKvCache)q).LayerCount);
    }

    [Fact]
    public void Qwen4Exp_DefaultCache_IsQuarterSize_AndLogitsBitIdentical()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-839-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = SyntheticQwen4ExpGguf.Write(Path.Combine(dir, "syn.gguf"));
            var (m, g, cfg) = ModelLoader.LoadFromGguf(path);
            using var gg = g; using var mm = m;
            var model = (Qwen4ExpTransformerModel)m;
            int[] ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 6, 7];      // > the 11-token QSA budget: sparse path
            int[] pos = Enumerable.Range(0, ids.Length).ToArray();

            float[] Run(SimpleKvCache kv, out long bytes)
            {
                using (kv)
                {
                    bytes = kv.AllocatedBytes;
                    using var t = model.Forward(ids, pos, -1, kv);
                    unsafe { return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray(); }
                }
            }
            var legacyGeom = KvGeometry.Uniform(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim);
            var a = Run(new SimpleKvCache(legacyGeom, 64), out long legacyBytes);
            var b = Run(new SimpleKvCache(KvGeometry.FromConfig(cfg), 64), out long hybridBytes);
            Assert.Equal(a, b);
            Assert.Equal(cfg.NumLayers, (int)(legacyBytes / hybridBytes));      // 1 QSA layer in 4
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }
}
