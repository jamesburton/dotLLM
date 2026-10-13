using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// Hermetic MiMo-V2 CPU tests: a random-weight <c>mimo2</c> GGUF is written by <see cref="SyntheticMiMoV2Gguf"/> (no fixture
/// files) and checked against golden logits produced by llama.cpp CPU (F32, no flash attention, F32 KV) on the SAME bytes. The
/// tools/mimo2-tiny-oracle HF model establishes that this llama.cpp build agrees with the HF reference implementation to 2e-5
/// (see <see cref="MiMoV2TinyOracleParityTests"/>), so the golden is the HF semantics transitively.
/// </summary>
/// <remarks>
/// Regenerating <see cref="Golden"/>: <c>DOTLLM_MIMO2_SYNTH_WRITE=&lt;dir&gt; dotnet test --filter WriteSyntheticFixture</c>, then
/// <c>tools/mimo2-tiny-oracle/dumplogits.exe &lt;dir&gt;/synthetic.gguf ids.txt out.f32</c> and paste the values.
/// </remarks>
public sealed class MiMoV2SyntheticGoldenTests : IDisposable
{
    /// <summary>16 tokens: longer than the sliding window (4) so the window really clips on 4 of 6 layers.</summary>
    internal static readonly int[] Ids = [3, 17, 8, 22, 5, 39, 1, 14, 30, 9, 26, 11, 2, 35, 20, 6];
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mimo2-{Guid.NewGuid():N}");

    public MiMoV2SyntheticGoldenTests() => Directory.CreateDirectory(_scratch);
    public void Dispose() { try { Directory.Delete(_scratch, true); } catch { /* best effort */ } }

    private string Write(SyntheticMiMoV2Gguf.Options? o = null, string name = "m")
    {
        string p = Path.Combine(_scratch, name + ".gguf");
        File.WriteAllBytes(p, SyntheticMiMoV2Gguf.Build(o));
        return p;
    }

    private static unsafe float[] Run(GgufFile gguf, ModelConfig cfg, bool decode = false)
    {
        int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        if (!decode)
        {
            using ITensor t = model.Forward(Ids, pos, deviceId: -1);
            return Copy(t);
        }
        var cache = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), 64);
        var all = new List<float>();
        for (int i = 0; i < Ids.Length; i++)
        {
            using ITensor t = model.Forward(Ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, cache);
            all.AddRange(Copy(t));
        }
        return all.ToArray();
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Config_Maps_Per_Layer_Arrays_And_Trunk_Excludes_NextN()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig c = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.MiMoV2, c.Architecture);
        Assert.Equal(SyntheticMiMoV2Gguf.Layers, c.NumLayers);
        Assert.Equal(SyntheticMiMoV2Gguf.NextN, c.NextnPredictLayers);
        Assert.Equal(48, c.HeadDim);
        Assert.Equal(32, c.GetValueHeadDim());
        Assert.True(c.HasDistinctValueHeadDim);
        Assert.Equal(SyntheticMiMoV2Gguf.SlidingKv, c.NumKvHeads);
        Assert.Equal(SyntheticMiMoV2Gguf.GlobalKv, c.NumGlobalKvHeads);
        Assert.Equal([2, 4, 4, 4, 2, 4], Enumerable.Range(0, c.NumLayers).Select(c.GetLayerKvHeads).ToArray());
        Assert.Equal([null, 4, 4, 4, null, 4], c.PerLayerSlidingWindow!.ToArray());
        Assert.Equal(16, c.RoPEConfig!.Value.DimensionCount);
        Assert.Equal(1e4f, c.RoPEConfig.Value.Theta);
        Assert.Equal(1e7f, c.GlobalRoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.NeoX, c.RoPEConfig.Value.Type);
        Assert.Equal(0.707f, c.AttentionValueScale);
        Assert.Equal(1f / MathF.Sqrt(48), c.AttentionScoreScale(c.HeadDim));   // 1/sqrt(K head dim), NOT of V's
        var moe = c.Moe!;
        Assert.True(moe.SigmoidGating); Assert.True(moe.HasSelectionBias); Assert.True(moe.NormalizeExpertWeights);
        Assert.Null(moe.SharedExpertIntermediateSize);
        Assert.Equal(8, moe.NumExperts); Assert.Equal(2, moe.NumExpertsPerTok);
        Assert.False(moe.IsMoeLayer(0)); Assert.True(moe.IsMoeLayer(1)); Assert.True(moe.IsMoeLayer(5));
    }

    [Fact]
    public void Forward_Matches_LlamaCpp_Golden()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] ours = Run(gguf, cfg);
        Assert.Equal(Golden.Length, ours.Length);
        float diff = MaxAbs(ours, Golden);
        Assert.True(diff < 1e-4f, $"max |dotLLM - llama.cpp| = {diff}");
    }

    [Fact]
    public void KvCacheDecode_Equals_BatchedForward()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float diff = MaxAbs(Run(gguf, cfg, decode: true), Run(gguf, cfg));
        Assert.True(diff < 1e-4f, $"decode vs batched {diff}");
    }

    // ---- each MiMo mechanism must be observable: a mutant of it has to move the logits far beyond the 1e-4 parity tolerance ----

    [Theory]
    [InlineData("value-scale")]
    [InlineData("window")]
    [InlineData("rope-theta-swapped")]
    [InlineData("rope-full-dim")]
    [InlineData("value-dim-padded-as-key")]
    [InlineData("kv-heads-uniform")]
    public void Mutant_Moves_Logits_Or_Fails_Loudly(string mutant)
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ModelConfig m = mutant switch
        {
            "value-scale" => cfg with { AttentionValueScale = null },
            "window" => cfg with { PerLayerSlidingWindow = cfg.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : 4096).ToArray() },
            "rope-theta-swapped" => cfg with { RoPEConfig = cfg.GlobalRoPEConfig, GlobalRoPEConfig = cfg.RoPEConfig },
            "rope-full-dim" => cfg with
            {
                RoPEConfig = cfg.RoPEConfig!.Value with { DimensionCount = 48 },
                GlobalRoPEConfig = cfg.GlobalRoPEConfig!.Value with { DimensionCount = 48 },
            },
            "value-dim-padded-as-key" => cfg with { ValueHeadDim = null },
            _ => cfg with { NumGlobalKvHeads = null },
        };
        if (mutant is "value-dim-padded-as-key" or "kv-heads-uniform")
        {
            Assert.ThrowsAny<Exception>(() => Run(gguf, m));   // tensor geometry no longer matches: must be loud
            return;
        }
        float diff = MaxAbs(Run(gguf, m), Golden);
        Assert.True(diff > 0.02f, $"mutant '{mutant}' is invisible to the golden (max diff {diff})");
    }

    [Fact]
    public void Sinks_Are_Read_And_Change_The_Output()
    {
        using var withSinks = GgufFile.Open(Write(name: "sinks"));
        using var noSinks = GgufFile.Open(Write(new SyntheticMiMoV2Gguf.Options { Sinks = false }, "nosinks"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(withSinks.Metadata);
        float diff = MaxAbs(Run(withSinks, cfg), Run(noSinks, cfg));
        Assert.True(diff > 0.02f, $"attention sinks have no effect on the CPU forward (max diff {diff})");
    }

    [Fact]
    public unsafe void ExpandThenCompact_Value_Heads_Is_Exact_And_Zero_Pads()
    {
        const int T = 3, H = 2, V = 5, K = 8;
        var v = new float[T * H * K];
        for (int t = 0; t < T; t++) for (int i = 0; i < H * V; i++) v[t * H * V + i] = 100 * t + i + 1;   // compact layout at the front
        fixed (float* p = v)
        {
            TransformerModel.ExpandValueHeads(p, T, H, V, K);
            for (int t = 0; t < T; t++)
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < K; d++)
                        Assert.Equal(d < V ? 100 * t + h * V + d + 1 : 0f, v[(t * H + h) * K + d]);
            // compaction of a padded attention output restores the compact layout
            TransformerModel.CompactAttentionHeads(p, T, H, K, V);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < H * V; i++)
                    Assert.Equal(100 * t + i + 1f, v[t * H * V + i]);
        }
    }

    [Fact]
    public void Gpu_Backends_Refuse_Loudly()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var vk = Assert.Throws<NotSupportedException>(() => DotLLM.Vulkan.VulkanTransformerModel.RejectUnsupportedArchitecture(cfg));
        Assert.Contains("MiMo-V2", vk.Message);
        Assert.Contains("issues/935", vk.Message);
        var cu = Assert.Throws<NotSupportedException>(() => DotLLM.Cuda.CudaTransformerModel.RejectUnsupportedArchitecture(cfg));
        Assert.Contains("MiMo-V2", cu.Message);
    }

    [SkippableFact]
    public void WriteSyntheticFixture()
    {
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_MIMO2_SYNTH_WRITE");
        Skip.If(string.IsNullOrEmpty(dir), "generator for Golden - run by hand with DOTLLM_MIMO2_SYNTH_WRITE=<dir>");
        Directory.CreateDirectory(dir!);
        File.WriteAllBytes(Path.Combine(dir!, "synthetic.gguf"), SyntheticMiMoV2Gguf.Build());
        File.WriteAllText(Path.Combine(dir!, "ids.txt"), string.Join(' ', Ids));
    }

    /// <summary>llama.cpp CPU F32 logits (16 x 40) for <see cref="Ids"/> on <c>SyntheticMiMoV2Gguf.Build()</c> default options.</summary>
    private static readonly float[] Golden = DecodeGolden(
        "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");

    private static float[] DecodeGolden(string b64)
    {
        byte[] b = Convert.FromBase64String(b64);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
}
