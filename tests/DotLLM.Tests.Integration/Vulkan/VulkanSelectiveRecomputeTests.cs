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
    public void Qwen3_4B_NonContiguousSubsetRecompute_ReproducesFullPrefillRows() => RunSubsetRecompute(96, new[] { 3, 4, 5, 20, 33, 34, 70, 95 });

    /// <summary>
    /// #1540 at 2K: 64 scattered rows (runs and singletons, including the last position) of a 2048-token prompt through the tiled
    /// per-query-position kernel. The tiled kernel must not add error beyond the #1534 oracle (the ~9e-3 mean relative L2 vs the full prefill is the forward-shape noise floor of Q4_K_M, shared by the contiguous-tail control)
    /// .
    /// </summary>
    [SkippableFact]
    public void Qwen3_4B_2K_ScatteredRecompute_TiledKernel_WithinNumericsFloor()
    {
        var rng = new Random(1540);
        var set = new SortedSet<int> { 2047, 0, 1, 2 };
        while (set.Count < 64) { int s0 = rng.Next(2040); for (int j = 0; j < 1 + rng.Next(3); j++) set.Add(s0 + j); }
        RunSubsetRecompute(2048, set.ToArray(), tiledExpected: true);
    }

    private void RunSubsetRecompute(int N, int[] subset, bool tiledExpected = true, double kTol = 0, double vTol = 0)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fx.Path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, VulkanModelLoader.ResolveSpvDir());

        int[] ids = Enumerable.Range(0, N).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
        int[] pos = Enumerable.Range(0, N).ToArray();
        int layers = cfg.NumLayers;

        using var reference = (VulkanKvCache)vk.CreateKvCache(N + 32);
        using (vk.Forward(ids, pos, 0, reference)) { }
        int stride = reference.KvStride;

        // Recomputes `rows` (positions) on a fresh full-prefill cache whose selected rows were first poisoned, so a recompute that did
        // not run, or wrote elsewhere, cannot pass. Returns (worst rel L2 K, V over the rows, untouched-row worst, per-layer worst K, kernel count).
        long tiledRan = 0;
        (double k, double v, double other, double[] perLayerK, long qposRuns, double meanK, double meanV) Recompute(int[] rows, bool tiled = true)
        {
            using var c = (VulkanKvCache)vk.CreateKvCache(N + 32);
            using (vk.Forward(ids, pos, 0, c)) { }
            var junk = new float[stride];
            Array.Fill(junk, 7.5f);
            for (int l = 0; l < layers; l++)
                foreach (int p in rows) c.WriteRows(l, p, 1, junk, junk);
            vk.TiledQueryPositionAttentionEnabled = tiled;
            long before = vk.AttentionQueryPositionsCount;
            long tiledBefore = vk.AttentionQueryPositionsTiledCount;
            using (vk.Forward(rows.Select(p => ids[p]).ToArray(), rows, 0, c)) { }
            long ran = vk.AttentionQueryPositionsCount - before;
            tiledRan = vk.AttentionQueryPositionsTiledCount - tiledBefore;
            Assert.Equal(N, c.CurrentLength);
            double wk = 0, wv = 0, wo = 0, sk = 0, sv2 = 0; int ns = 0;
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
                    if (sel) { sk += relK; sv2 += relV; ns++; wk = Math.Max(wk, relK); wv = Math.Max(wv, relV); perLayer[l] = Math.Max(perLayer[l], relK); }
                    else wo = Math.Max(wo, Math.Max(relK, relV));
                }
            }
            vk.TiledQueryPositionAttentionEnabled = true;
            return (wk, wv, wo, perLayer, ran, sk / ns, sv2 / ns);
        }

        var sub = Recompute(subset);
        long subTiled = tiledRan;
        // Noise floor: an 8-row contiguous tail run on the same path the model always used (flash/coopmat attention, tight span). It differs
        // from the 96-row prefill only through the numerics of an 8-row vs 96-row forward (activation-quantised matmul paths, attention kernel).
        var refr = Recompute(subset, tiled: false);   // #1534 one-workgroup-per-row kernel: the oracle
        var ctl = Recompute(Enumerable.Range(N - subset.Length, subset.Length).ToArray());
        _out.WriteLine($"[1531 step3] layers={layers} subset (non-contiguous, qpos kernel x{sub.qposRuns}): K {sub.k:E2} V {sub.v:E2}, untouched {sub.other:E2}");
        _out.WriteLine($"[1531 step3] control (contiguous tail run, old kernels x{ctl.qposRuns}): K {ctl.k:E2} V {ctl.v:E2}, untouched {ctl.other:E2}");
        _out.WriteLine("[1531 step3] per-layer worst K  layer: subset / control: " + string.Join(" ", Enumerable.Range(0, layers).Where(l => l % 5 == 0 || l == layers - 1).Select(l => $"{l}:{sub.perLayerK[l]:E1}/{ctl.perLayerK[l]:E1}")));

        Assert.Equal(layers, (int)sub.qposRuns);
        if (tiledExpected && vk.SupportsTiledQueryPositionAttention) Assert.Equal(layers, (int)subTiled);
        _out.WriteLine($"[1540] N={N} rows={subset.Length} tiled layers={subTiled}");
        _out.WriteLine($"[1540] worst-row K/V vs prefill: tiled {sub.k:E2}/{sub.v:E2}  ref {refr.k:E2}/{refr.v:E2}  ctl {ctl.k:E2}/{ctl.v:E2}");
        _out.WriteLine($"[1540] mean-over-selected-rows K/V vs prefill: tiled {sub.meanK:E2}/{sub.meanV:E2}  ref {refr.meanK:E2}/{refr.meanV:E2}  ctl {ctl.meanK:E2}/{ctl.meanV:E2}");
        if (vk.SupportsTiledQueryPositionAttention)
        {
            // The tiled kernel adds no more error vs the full prefill than the #1534 oracle does (it IS the same math in a different tiling).
            Assert.True(sub.meanK <= refr.meanK * 1.15 + 1e-5 && sub.meanV <= refr.meanV * 1.15 + 1e-5,
                $"tiled mean K {sub.meanK:E2} V {sub.meanV:E2} vs ref {refr.meanK:E2} {refr.meanV:E2}");
        }
        if (kTol > 0) Assert.True(sub.meanK <= kTol && sub.meanV <= vTol, $"mean K {sub.meanK:E2} (<= {kTol:E1}) V {sub.meanV:E2} (<= {vTol:E1})");   // the per-query-position kernel ran once per layer on the subset
        Assert.Equal(0L, ctl.qposRuns);            // and not on the tight control
        Assert.Equal(0.0, sub.other);              // rows outside the subset are byte-identical
        Assert.Equal(0.0, ctl.other);
        // The recompute is as close to the full prefill as the control, which has no non-contiguity at all.
        Assert.True(sub.k <= 2 * ctl.k + 1e-3 && sub.v <= 2 * ctl.v + 1e-3,
            $"subset recompute drifts more than the numerics floor: K {sub.k:E2} vs {ctl.k:E2}, V {sub.v:E2} vs {ctl.v:E2}");
    }
}
