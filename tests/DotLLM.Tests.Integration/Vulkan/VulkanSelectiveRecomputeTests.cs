using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1531 step 3 on the real model: after a full prefill, re-running a NON-CONTIGUOUS subset of the prompt through
/// <c>Forward(subsetIds, subsetPositions, cache)</c> must, in one pass, rewrite each selected cache row to what the full prefill
/// wrote (same tokens, same context, so the recompute is the identity up to numerics), at every layer. A kernel that derives row
/// positions as <c>positions[0] + row</c> masks keys the later rows must see and changes layer-1+ K/V of those rows by O(1).
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanSelectiveRecomputeTests
{
    private readonly ITestOutputHelper _out;
    public VulkanSelectiveRecomputeTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void Qwen3_4B_NonContiguousSubsetRecompute_ReproducesFullPrefillRows()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fx.Path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, VulkanModelLoader.ResolveSpvDir());

        const int N = 96;
        int[] ids = Enumerable.Range(0, N).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
        int[] pos = Enumerable.Range(0, N).ToArray();
        int[] subset = { 3, 4, 5, 20, 33, 34, 70, 95 };   // scattered, with runs, includes the last position
        int layers = cfg.NumLayers;

        using var reference = (VulkanKvCache)vk.CreateKvCache(128);
        using (vk.Forward(ids, pos, 0, reference)) { }

        using var recomputed = (VulkanKvCache)vk.CreateKvCache(128);
        using (vk.Forward(ids, pos, 0, recomputed)) { }
        // Poison the selected rows (a stale, wrong K/V) so a recompute that did not run, or wrote elsewhere, cannot pass.
        int stride = reference.KvStride;
        var junk = new float[stride];
        Array.Fill(junk, 7.5f);
        for (int l = 0; l < layers; l++)
            foreach (int p in subset) recomputed.WriteRows(l, p, 1, junk, junk);

        long before = vk.AttentionQueryPositionsCount;
        int[] subIds = subset.Select(p => ids[p]).ToArray();
        using (vk.Forward(subIds, subset, 0, recomputed)) { }
        long ran = vk.AttentionQueryPositionsCount - before;
        Assert.Equal(layers, (int)ran);   // the per-query-position kernel ran once per layer

        Assert.Equal(N, recomputed.CurrentLength);
        double worstSelK = 0, worstSelV = 0, worstOther = 0;
        var rk = new float[N * stride]; var rv = new float[N * stride];
        var ck = new float[N * stride]; var cv = new float[N * stride];
        for (int l = 0; l < layers; l++)
        {
            reference.ReadRows(l, 0, N, rk, rv);
            recomputed.ReadRows(l, 0, N, ck, cv);
            for (int p = 0; p < N; p++)
            {
                bool sel = Array.IndexOf(subset, p) >= 0;
                double dk = 0, nk = 0, dv = 0, nv = 0;
                for (int i = 0; i < stride; i++)
                {
                    double a = rk[p * stride + i], b = ck[p * stride + i];
                    dk += (a - b) * (a - b); nk += a * a;
                    a = rv[p * stride + i]; b = cv[p * stride + i];
                    dv += (a - b) * (a - b); nv += a * a;
                }
                double relK = Math.Sqrt(dk / nk), relV = Math.Sqrt(dv / nv);
                if (sel) { worstSelK = Math.Max(worstSelK, relK); worstSelV = Math.Max(worstSelV, relV); }
                else worstOther = Math.Max(worstOther, Math.Max(relK, relV));
            }
        }
        _out.WriteLine($"[1531 step3] layers={layers} selected rows: worst rel L2 K {worstSelK:E2} V {worstSelV:E2}; untouched rows worst {worstOther:E2}");
        Assert.Equal(0.0, worstOther);       // rows outside the subset are byte-identical
        Assert.True(worstSelK < 5e-3 && worstSelV < 5e-3, $"recomputed rows drift from the full prefill: K {worstSelK:E2} V {worstSelV:E2}");
    }
}
