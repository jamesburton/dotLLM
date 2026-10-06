using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #734: Gemma-4 dense-PLE (E2B/E4B) on CUDA vs the CPU oracle on the synthetic
/// <see cref="SyntheticGemma4Gguf.E4BLike"/> fixture (PLE + 4 shared-KV layers + proportional rope_freqs + dual head dim
/// 16/32 + layer_output_scale + final soft-cap), cacheless and through the FP16 <see cref="CudaKvCache"/>. Each component
/// is also toggled off on BOTH backends in isolation. Multi-position sequences are mandatory (position 0 hides rope).
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma4DensePleCudaParityTests
{
    private static readonly int[] Ids = [2, 7, 8, 9, 5, 6, 3];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5, 6];

    private readonly ITestOutputHelper _output;
    public Gemma4DensePleCudaParityTests(ITestOutputHelper output) => _output = output;

    // Same fixture knobs as the Vulkan twin (low global rope base so every rotated pair moves; proportional factors).
    private static SyntheticGemma4Config Cfg() =>
        SyntheticGemma4Gguf.E4BLike with { RopeFreqsProportionalPairs = 2, GlobalRopeFreqBase = 20f };

    private static void RequireCuda() =>
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA device/driver on this host.");

    public static IEnumerable<object[]> Variants() =>
    [
        ["full", 0],
        ["no-ple", 1],
        ["no-shared-kv", 2],
        ["no-rope-freqs", 3],
    ];

    private static ModelConfig Mutate(ModelConfig c, int variant) => variant switch
    {
        1 => c with { PerLayerEmbedding = null },
        2 => c with { NumSharedKvLayers = 0 },
        _ => c,
    };

    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void Cacheless_Logits_MatchCpu(string name, int variant)
    {
        RequireCuda();
        var syn = variant == 3 ? Cfg() with { EmitRopeFreqs = false } : Cfg();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_cuda_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, syn);
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = Mutate(GgufModelConfigExtractor.Extract(gguf.Metadata), variant);
            using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using var cuda = CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0);
            float[] a = LastRow(cpu.Forward(Ids, Pos, -1), cfg.VocabSize);
            float[] b = LastRow(cuda.Forward(Ids, Pos, -1, kvCache: null), cfg.VocabSize);
            AssertClose(name, a, b);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [SkippableFact]
    public void CachedPrefillThenDecode_MatchesCpu()
    {
        RequireCuda();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_cuda_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, Cfg());
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using var cuda = CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0);
            int last = Ids.Length - 1;
            float[] oracle = LastRow(cpu.Forward(Ids, Pos, -1), cfg.VocabSize);

            using var kv = cuda.CreateKvCache(maxSeqLen: 16);
            using (cuda.Forward(Ids.AsSpan(0, last), Pos.AsSpan(0, last), -1, kv)) { }
            float[] decode = LastRow(cuda.Forward(Ids.AsSpan(last, 1), Pos.AsSpan(last, 1), -1, kv), cfg.VocabSize);
            AssertClose("cached-decode", oracle, decode);

            int[] more = [11, 12, 13];
            var allIds = Ids.ToList();
            var allPos = Pos.ToList();
            foreach (int tok in more)
            {
                allIds.Add(tok);
                allPos.Add(allPos.Count);
                float[] o = LastRow(cpu.Forward(allIds.ToArray(), allPos.ToArray(), -1), cfg.VocabSize);
                float[] d = LastRow(cuda.Forward([tok], [allPos[^1]], -1, kv), cfg.VocabSize);
                AssertClose($"decode+{allIds.Count}", o, d);
            }
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [SkippableFact]
    public void CudaDrift_IsFarSmallerThanEveryComponentsEffect()
    {
        // The absolute envelope (6e-2) exceeds these fixtures' whole logit range (~1), so make the check SENSITIVE: the
        // CUDA-vs-CPU drift must be well below the effect of removing each component from the CPU oracle.
        RequireCuda();
        string p1 = Path.Combine(Path.GetTempPath(), $"syn_g4ple_cuda_{Guid.NewGuid():N}.gguf");
        string p2 = Path.Combine(Path.GetTempPath(), $"syn_g4ple_cuda_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(p1, Cfg());
        SyntheticGemma4Gguf.WriteGemma4(p2, Cfg() with { EmitRopeFreqs = false });
        try
        {
            using var g1 = GgufFile.Open(p1);
            using var g2 = GgufFile.Open(p2);
            var cfg = GgufModelConfigExtractor.Extract(g1.Metadata);
            var cfg2 = GgufModelConfigExtractor.Extract(g2.Metadata);
            float[] Cpu(GgufFile g, ModelConfig c)
            {
                using var m = TransformerModel.LoadFromGguf(g, c, ThreadingConfig.SingleThreaded);
                return LastRow(m.Forward(Ids, Pos, -1), c.VocabSize);
            }
            float[] full = Cpu(g1, cfg);
            static float Max(float[] x, float[] y) { float w = 0; for (int i = 0; i < x.Length; i++) w = MathF.Max(w, MathF.Abs(x[i] - y[i])); return w; }
            float ple = Max(full, Cpu(g1, cfg with { PerLayerEmbedding = null }));
            float shared = Max(full, Cpu(g1, cfg with { NumSharedKvLayers = 0 }));
            float rope = Max(full, Cpu(g2, cfg2));
            using var cuda = CudaTransformerModel.LoadFromGguf(g1, cfg, deviceId: 0);
            float drift = Max(full, LastRow(cuda.Forward(Ids, Pos, -1, kvCache: null), cfg.VocabSize));
            _output.WriteLine($"effects (CPU full vs ablated): ple={ple:E3} sharedKv={shared:E3} ropeFreqs={rope:E3}; CUDA-vs-CPU drift={drift:E3}");
            float min = MathF.Min(ple, MathF.Min(shared, rope));
            Assert.True(drift < min / 4, $"drift {drift:E3} is not << smallest component effect {min:E3} (ple={ple:E3} shared={shared:E3} rope={rope:E3})");
        }
        finally { try { File.Delete(p1); File.Delete(p2); } catch { } }
    }

    [SkippableFact]
    public void GeneralRopeFreqFactors_AreRejectedAtLoad_NotSilentlyMisrotated()
    {
        RequireCuda();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_cuda_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, SyntheticGemma4Gguf.E4BLike);   // general factors in [1,1.5)
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Throws<NotSupportedException>(() => CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0));
        }
        finally { try { File.Delete(path); } catch { } }
    }

    private void AssertClose(string name, float[] cpu, float[] gpu)
    {
        Assert.All(gpu, v => Assert.True(float.IsFinite(v), $"{name}: non-finite CUDA logit"));
        int ac = ArgMax(cpu), ag = ArgMax(gpu);
        float worst = 0; int worstCol = -1;
        for (int i = 0; i < cpu.Length; i++)
        {
            float d = MathF.Abs(cpu[i] - gpu[i]);
            if (d > worst) { worst = d; worstCol = i; }
        }
        _output.WriteLine($"[{name}] argmax cpu={ac} cuda={ag} worst|diff|={worst:E3} @ {worstCol} (logit range {cpu.Min():F2}..{cpu.Max():F2})");
        const float absTol = 6.0e-2f, relTol = 5.0e-3f;
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) <= absTol + relTol * MathF.Abs(cpu[i]),
                $"{name} col {i}: cpu={cpu[i]:F6} cuda={gpu[i]:F6}; worst={worst:E3}@{worstCol}; argmax cpu={ac} cuda={ag}");
        Assert.Equal(ac, ag);
        Assert.True(cpu.Max() - cpu.Min() > 0.5f, $"{name}: CPU logits degenerate");
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        using (logits)
        {
            int total = 1;
            for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
            var all = new float[total];
            new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(all);
            var row = new float[vocab];
            Array.Copy(all, total - vocab, row, 0, vocab);
            return row;
        }
    }
}
