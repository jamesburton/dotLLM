using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan <c>granite</c> / <c>granitemoe</c> (GGUF) against the independent double-precision
/// <see cref="GraniteGgufReference"/> (#764, #313) — all four Granite scalars != 1 and mutually distinct. The reference
/// (not the CPU backend) is the oracle for the MoE fixture: the CPU routed-MoE path uses Q8-quantized activations like
/// llama.cpp, so on Q8_0 experts it sits ~1% from the exact value, whereas Vulkan keeps F32 activations (measured on
/// gfx1151: Vulkan vs exact reference 4.5e-3, CPU vs exact reference 7e-2 on this fixture).
/// </summary>
[Trait("Category", "GPU")]
public sealed class GraniteVulkanReferenceTests
{
    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3];

    public static IEnumerable<object[]> Fixtures()
    {
        yield return new object[] { new SyntheticGraniteConfig() };
        yield return new object[] { new SyntheticGraniteConfig { Hidden = 256, Heads = 4, KvHeads = 4, Tied = true, Layers = 2 } };
        yield return new object[] { new SyntheticGraniteConfig { Arch = "granitemoe", ExpertsQ8_0 = true } };
    }

    private static unsafe float[] Last(ITensor t)
    {
        int rows = t.Shape[0], v = t.Shape[1];
        var r = new float[v];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * v, v).CopyTo(r);
        return r;
    }

    private static double Max(float[] a, double[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    [SkippableTheory]
    [MemberData(nameof(Fixtures))]
    public void Vulkan_MatchesIndependentReference_PrefillCachedDecodeAndAblations(SyntheticGraniteConfig cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        var wts = SyntheticGraniteGguf.BuildWeights(cfg);
        string path = Path.Combine(Path.GetTempPath(), $"granite_vk_ref_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.True(config.HasGraniteScalars);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, SpvDir());

            // Cacheless full forward -> last-row logits.
            double[] reference = GraniteGgufReference.LastTokenLogits(wts, Prompt);
            double dCacheless;
            using (ITensor l = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), -1, kvCache: null))
                dCacheless = Max(Last(l), reference);
            Assert.True(dCacheless < 0.03, $"cacheless Vulkan vs reference = {dCacheless:E3}");

            // KV-cache prefill + single-token decode of the final position.
            int split = Prompt.Length - 1;
            using (var cache = model.CreateKvCache(maxSeqLen: Prompt.Length + 1))
            {
                using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, cache)) { }
                using ITensor l = model.Forward([Prompt[split]], [split], -1, cache);
                double d = Max(Last(l), reference);
                Assert.True(d < 0.03, $"cached decode Vulkan vs reference = {d:E3}");
            }

            // Each scalar must matter: the ablated reference is farther from the Vulkan result than the tolerance.
            var ablations = new (string Name, GraniteGgufReference.Options Opt)[]
            {
                ("embedding scale", new() { EmbeddingScale = false }),
                ("attention scale", new() { AttentionScale = false }),
                ("residual scale", new() { ResidualScale = false }),
                ("logit scale", new() { LogitScale = false }),
            };
            using (ITensor l = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), -1, kvCache: null))
            {
                float[] gpu = Last(l);
                foreach (var (name, opt) in ablations)
                {
                    double d = Max(gpu, GraniteGgufReference.LastTokenLogits(wts, Prompt, opt));
                    Assert.True(d > 0.1, $"{cfg.Arch}: Vulkan result is within {d:E3} of the reference WITHOUT the {name} (not sensitive enough).");
                }
            }
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }
}
