using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1548: ordinary dense prefill attention on the GQA-grouped tiled flash kernel versus the per-head coopmat flash kernel it replaced
/// (kept as the oracle behind <see cref="VulkanTransformerModel.PrefillGqaFlashEnabled"/>). A 2K prompt is prefilled in two 1024-row
/// chunks (the second at positionOffset 1024, so a wrong offset or causal span is observable) under both arms; the K/V cache of every
/// layer and the final-row logits must agree within the numerics floor, and the dispatch counter proves which kernel ran.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanPrefillGqaFlashTests
{
    private readonly ITestOutputHelper _out;
    public VulkanPrefillGqaFlashTests(ITestOutputHelper output) => _out = output;

    private static unsafe float[] Copy(ITensor t)
    {
        var r = new float[t.ElementCount];
        new ReadOnlySpan<float>((void*)t.DataPointer, r.Length).CopyTo(r);
        return r;
    }

    [SkippableFact]
    public void Qwen3_4B_ChunkedPrefill_GqaFlash_MatchesPerHeadFlash()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fx.Path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, VulkanModelLoader.ResolveSpvDir());
        Skip.IfNot(vk.SupportsPrefillGqaFlash, "GQA flash kernel not loaded");

        const int N = 2048, Chunk = 1024;
        int[] ids = Enumerable.Range(0, N).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
        int[] pos = Enumerable.Range(0, N).ToArray();
        int layers = cfg.NumLayers;

        (VulkanKvCache cache, float[] logits, long gqa) Run(bool gqaOn)
        {
            vk.PrefillGqaFlashEnabled = gqaOn;
            long before = vk.PrefillGqaFlashCount;
            var c = (VulkanKvCache)vk.CreateKvCache(N + 32);
            float[] last = Array.Empty<float>();
            for (int s = 0; s < N; s += Chunk)
            {
                using ITensor t = vk.Forward(ids.AsSpan(s, Chunk), pos.AsSpan(s, Chunk), 0, c);
                last = Copy(t);
            }
            return (c, last, vk.PrefillGqaFlashCount - before);
        }

        var a = Run(true);
        var b = Run(false);
        vk.PrefillGqaFlashEnabled = true;
        using var ca = a.cache; using var cb = b.cache;
        Assert.Equal(layers * (N / Chunk), (int)a.gqa);   // the new kernel served every layer of both chunks
        Assert.Equal(0L, b.gqa);                          // and the oracle arm never touched it

        int stride = ca.KvStride;
        double worstK = 0, worstV = 0, sumK = 0, sumV = 0; int n = 0;
        var ka = new float[N * stride]; var va = new float[N * stride]; var kb = new float[N * stride]; var vb = new float[N * stride];
        for (int l = 0; l < layers; l++)
        {
            ca.ReadRows(l, 0, N, ka, va);
            cb.ReadRows(l, 0, N, kb, vb);
            for (int p = 0; p < N; p++)
            {
                double dk = 0, nk = 0, dv = 0, nv = 0;
                for (int i = 0; i < stride; i++)
                {
                    double x = ka[p * stride + i], y = kb[p * stride + i]; dk += (x - y) * (x - y); nk += x * x;
                    x = va[p * stride + i]; y = vb[p * stride + i]; dv += (x - y) * (x - y); nv += x * x;
                }
                double rk = Math.Sqrt(dk / nk), rv = Math.Sqrt(dv / nv);
                worstK = Math.Max(worstK, rk); worstV = Math.Max(worstV, rv); sumK += rk; sumV += rv; n++;
            }
        }
        // KL(old || new) of the final row's next-token distribution.
        int V = cfg.VocabSize; int row = a.logits.Length / V - 1;
        double ma = double.MinValue, mb = double.MinValue;
        for (int i = 0; i < V; i++) { ma = Math.Max(ma, b.logits[row * V + i]); mb = Math.Max(mb, a.logits[row * V + i]); }
        double za = 0, zb = 0;
        for (int i = 0; i < V; i++) { za += Math.Exp(b.logits[row * V + i] - ma); zb += Math.Exp(a.logits[row * V + i] - mb); }
        double kl = 0; int am = 0, bm = 0;
        for (int i = 0; i < V; i++)
        {
            double p = Math.Exp(b.logits[row * V + i] - ma) / za, q = Math.Exp(a.logits[row * V + i] - mb) / zb;
            if (p > 1e-12) kl += p * Math.Log(p / Math.Max(q, 1e-30));
            if (b.logits[row * V + i] > b.logits[row * V + am]) am = i;
            if (a.logits[row * V + i] > a.logits[row * V + bm]) bm = i;
        }
        _out.WriteLine($"[1548] K/V rel L2 old-vs-new: mean {sumK / n:E2}/{sumV / n:E2} worst {worstK:E2}/{worstV:E2}; last-row KL {kl:E2}; argmax {am}/{bm}");

        Assert.True(sumK / n > 0 && sumV / n > 0 || worstK > 0, "arms are bit-identical: the toggle changed nothing (control insensitive)");
        Assert.True(sumK / n < 3e-3 && sumV / n < 3e-3, $"mean rel L2 K {sumK / n:E2} V {sumV / n:E2}");
        Assert.True(kl < 1e-2, $"KL {kl:E2}");
        Assert.Equal(am, bm);
    }
}
