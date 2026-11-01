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
        int stride = reference.KvStride;

        // Recomputes `rows` (positions) on a fresh full-prefill cache whose selected rows were first poisoned, so a recompute that did
        // not run, or wrote elsewhere, cannot pass. Returns (worst rel L2 K, V over the rows, untouched-row worst, per-layer worst K, kernel count).
        (double k, double v, double other, double[] perLayerK, long qposRuns) Recompute(int[] rows)
        {
            using var c = (VulkanKvCache)vk.CreateKvCache(128);
            using (vk.Forward(ids, pos, 0, c)) { }
            var junk = new float[stride];
            Array.Fill(junk, 7.5f);
            for (int l = 0; l < layers; l++)
                foreach (int p in rows) c.WriteRows(l, p, 1, junk, junk);
            long before = vk.AttentionQueryPositionsCount;
            using (vk.Forward(rows.Select(p => ids[p]).ToArray(), rows, 0, c)) { }
            long ran = vk.AttentionQueryPositionsCount - before;
            Assert.Equal(N, c.CurrentLength);
            double wk = 0, wv = 0, wo = 0;
            var perLayer = new double[layers];
            var rk = new float[N * stride]; var rv = new float[N * stride];
            var ck = new float[N * stride]; var cv = new float[N * stride];
            for (int l = 0; l < layers; l++)
            {
                reference.ReadRows(l, 0, N, rk, rv);
                c.ReadRows(l, 0, N, ck, cv);
                for (int p = 0; p < N; p++)
                {
                    bool sel = Array.IndexOf(rows, p) >= 0;
                    double dk = 0, nk = 0, dv = 0, nv = 0;
                    for (int i = 0; i < stride; i++)
                    {
                        double a = rk[p * stride + i], b = ck[p * stride + i];
                        dk += (a - b) * (a - b); nk += a * a;
                        a = rv[p * stride + i]; b = cv[p * stride + i];
                        dv += (a - b) * (a - b); nv += a * a;
                    }
                    double relK = Math.Sqrt(dk / nk), relV = Math.Sqrt(dv / nv);
                    if (sel) { wk = Math.Max(wk, relK); wv = Math.Max(wv, relV); perLayer[l] = Math.Max(perLayer[l], relK); }
                    else wo = Math.Max(wo, Math.Max(relK, relV));
                }
            }
            return (wk, wv, wo, perLayer, ran);
        }

        var sub = Recompute(subset);
        // Noise floor: an 8-row contiguous tail run on the same path the model always used (flash/coopmat attention, tight span). It differs
        // from the 96-row prefill only through the numerics of an 8-row vs 96-row forward (activation-quantised matmul paths, attention kernel).
        var ctl = Recompute(Enumerable.Range(N - subset.Length, subset.Length).ToArray());
        _out.WriteLine($"[1531 step3] layers={layers} subset (non-contiguous, qpos kernel x{sub.qposRuns}): K {sub.k:E2} V {sub.v:E2}, untouched {sub.other:E2}");
        _out.WriteLine($"[1531 step3] control (contiguous tail run, old kernels x{ctl.qposRuns}): K {ctl.k:E2} V {ctl.v:E2}, untouched {ctl.other:E2}");
        _out.WriteLine("[1531 step3] per-layer worst K  layer: subset / control: " + string.Join(" ", Enumerable.Range(0, layers).Where(l => l % 5 == 0 || l == layers - 1).Select(l => $"{l}:{sub.perLayerK[l]:E1}/{ctl.perLayerK[l]:E1}")));

        Assert.Equal(layers, (int)sub.qposRuns);   // the per-query-position kernel ran once per layer on the subset
        Assert.Equal(0L, ctl.qposRuns);            // and not on the tight control
        Assert.Equal(0.0, sub.other);              // rows outside the subset are byte-identical
        Assert.Equal(0.0, ctl.other);
        // The recompute is as close to the full prefill as the control, which has no non-contiguity at all.
        Assert.True(sub.k <= 2 * ctl.k + 1e-3 && sub.v <= 2 * ctl.v + 1e-3,
            $"subset recompute drifts more than the numerics floor: K {sub.k:E2} vs {ctl.k:E2}, V {sub.v:E2} vs {ctl.v:E2}");
    }
}
