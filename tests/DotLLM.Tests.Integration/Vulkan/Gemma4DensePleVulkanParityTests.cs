using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #734: Gemma-4 dense-PLE (E2B/E4B) on Vulkan vs the CPU oracle, on the synthetic
/// <see cref="SyntheticGemma4Gguf.E4BLike"/> fixture (PLE + 4 shared-KV layers + proportional rope_freqs +
/// dual head dim 16/32 + layer_output_scale + final soft-cap). Every component is also toggled off on BOTH
/// backends in isolation. Multi-position sequences are mandatory: at position 0 every rope angle is 0, so a
/// position-0-only test cannot see a missing rope_freqs mapping.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma4DensePleVulkanParityTests
{
    private static readonly int[] Ids = [2, 7, 8, 9, 5, 6, 3];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5, 6];

    private readonly ITestOutputHelper _output;
    public Gemma4DensePleVulkanParityTests(ITestOutputHelper output) => _output = output;

    // E4BLike with the released-checkpoint rope_freqs form (first 2 of 16 pairs rotate on global layers). The global
    // rope base is lowered from 1e6 to 20 so EVERY pair rotates visibly within a 7-token sequence - at 1e6 the high
    // pairs barely move and the rope_freqs effect drowns in noise (measured: 6.7e-3 vs 6.3e-1 for PLE).
    private static SyntheticGemma4Config Cfg() =>
        SyntheticGemma4Gguf.E4BLike with { RopeFreqsProportionalPairs = 2, GlobalRopeFreqBase = 20f };

    private static void RequireVulkan()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
    }

    private static string SpvDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "spv");
        Skip.IfNot(Directory.Exists(dir), "spv dir missing: " + dir);
        return dir;
    }

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
    public void Cacheless_PrefillLogits_MatchCpu(string name, int variant)
    {
        RequireVulkan();
        string spv = SpvDir();
        var syn = variant == 3 ? Cfg() with { EmitRopeFreqs = false } : Cfg();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, syn);
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = Mutate(GgufModelConfigExtractor.Extract(gguf.Metadata), variant);
            using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spv);
            float[] a = LastRow(cpu.Forward(Ids, Pos, -1), cfg.VocabSize);
            float[] b = LastRow(vk.Forward(Ids, Pos, -1, kvCache: null), cfg.VocabSize);
            AssertClose(name, a, b);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [SkippableFact]
    public void CachedPrefillThenDecode_MatchesCpu()
    {
        RequireVulkan();
        string spv = SpvDir();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, Cfg());
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spv);
            int last = Ids.Length - 1;
            float[] oracle = LastRow(cpu.Forward(Ids, Pos, -1), cfg.VocabSize);

            // Prefill [0,last) then decode the final token: shared layers read the donor's CACHE lines.
            using var kv = vk.CreateKvCache(maxSeqLen: 16);
            using (vk.Forward(Ids.AsSpan(0, last), Pos.AsSpan(0, last), -1, kv)) { }
            float[] decode = LastRow(vk.Forward(Ids.AsSpan(last, 1), Pos.AsSpan(last, 1), -1, kv), cfg.VocabSize);
            AssertClose("cached-decode", oracle, decode);

            // Several single-token decodes past the prefill, vs the CPU cacheless oracle at each length.
            int[] more = [11, 12, 13];
            var allIds = Ids.ToList();
            var allPos = Pos.ToList();
            foreach (int tok in more)
            {
                allIds.Add(tok);
                allPos.Add(allPos.Count);
                float[] o = LastRow(cpu.Forward(allIds.ToArray(), allPos.ToArray(), -1), cfg.VocabSize);
                float[] d = LastRow(vk.Forward([tok], [allPos[^1]], -1, kv), cfg.VocabSize);
                AssertClose($"decode+{allIds.Count}", o, d);
            }
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [SkippableFact]
    public void VulkanDrift_IsFarSmallerThanEveryComponentsEffect()
    {
        // The absolute envelope above (6e-2) is wider than these fixtures' whole logit range (~1), so on its own it
        // could not see a missing component. This test makes the check SENSITIVE: the Vulkan-vs-CPU drift must be
        // well below the effect of removing each component from the CPU oracle (PLE, shared KV, rope_freqs).
        RequireVulkan();
        string spv = SpvDir();
        string p1 = Path.Combine(Path.GetTempPath(), $"syn_g4ple_{Guid.NewGuid():N}.gguf");
        string p2 = Path.Combine(Path.GetTempPath(), $"syn_g4ple_{Guid.NewGuid():N}.gguf");
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
            float Max(float[] x, float[] y) { float w = 0; for (int i = 0; i < x.Length; i++) w = MathF.Max(w, MathF.Abs(x[i] - y[i])); return w; }
            float ple = Max(full, Cpu(g1, cfg with { PerLayerEmbedding = null }));
            float shared = Max(full, Cpu(g1, cfg with { NumSharedKvLayers = 0 }));
            float rope = Max(full, Cpu(g2, cfg2));
            using var vk = VulkanTransformerModel.LoadFromGguf(g1, cfg, spv);
            float drift = Max(full, LastRow(vk.Forward(Ids, Pos, -1, kvCache: null), cfg.VocabSize));
            _output.WriteLine($"effects (CPU full vs ablated): ple={ple:E3} sharedKv={shared:E3} ropeFreqs={rope:E3}; Vulkan-vs-CPU drift={drift:E3}");
            float min = MathF.Min(ple, MathF.Min(shared, rope));
            Assert.True(drift < min / 4, $"drift {drift:E3} is not << smallest component effect {min:E3} (ple={ple:E3} shared={shared:E3} rope={rope:E3})");
        }
        finally { try { File.Delete(p1); File.Delete(p2); } catch { } }
    }

    [SkippableFact]
    public void Baseline_ValidatedMoeTiny_DriftReference()
    {
        // Reference for the noise floor of the already-validated 26B-shaped path on the same hardware.
        RequireVulkan();
        string spv = SpvDir();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4moe_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, SyntheticGemma4Gguf.Tiny);
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spv);
            float[] a = LastRow(cpu.Forward(Ids, Pos, -1), cfg.VocabSize);
            float[] b = LastRow(vk.Forward(Ids, Pos, -1, kvCache: null), cfg.VocabSize);
            float worst = 0;
            for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
            _output.WriteLine($"[baseline tiny-moe] worst|diff|={worst:E3} logit range {a.Min():F2}..{a.Max():F2}");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [SkippableFact]
    public void GeneralRopeFreqFactors_AreRejectedAtLoad_NotSilentlyMisrotated()
    {
        RequireVulkan();
        string spv = SpvDir();
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4ple_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, SyntheticGemma4Gguf.E4BLike);   // general factors in [1,1.5)
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Throws<NotSupportedException>(() => VulkanTransformerModel.LoadFromGguf(gguf, cfg, spv));
        }
        finally { try { File.Delete(path); } catch { } }
    }

    private void AssertClose(string name, float[] cpu, float[] vk)
    {
        Assert.All(vk, v => Assert.True(float.IsFinite(v), $"{name}: non-finite Vulkan logit"));
        int ac = ArgMax(cpu), av = ArgMax(vk);
        float worst = 0; int worstCol = -1;
        for (int i = 0; i < cpu.Length; i++)
        {
            float d = MathF.Abs(cpu[i] - vk[i]);
            if (d > worst) { worst = d; worstCol = i; }
        }
        _output.WriteLine($"[{name}] argmax cpu={ac} vulkan={av} worst|diff|={worst:E3} @ {worstCol} (logit range {cpu.Min():F2}..{cpu.Max():F2})");
        const float absTol = 6.0e-2f, relTol = 5.0e-3f;
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= absTol + relTol * MathF.Abs(cpu[i]),
                $"{name} col {i}: cpu={cpu[i]:F6} vulkan={vk[i]:F6}; worst={worst:E3}@{worstCol}; argmax cpu={ac} vk={av}");
        Assert.Equal(ac, av);
        // Discrimination: the logits must not be degenerate (else parity is vacuous).
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
