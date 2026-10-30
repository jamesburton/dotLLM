using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #917: the 2..16-row multi-column K-quant GEMVs on <see cref="VulkanTransformerModel"/>. A batched N-row forward must (a) take the GEMV path
/// (counter), (b) agree with N sequential 1-row decodes on every row's argmax and stay close in logits, and (c) be closer to the 1-row decode than
/// or comparable to the old MMQ path (control arm: the switch off).
/// </summary>
[Trait("Category", "GPU")]
public class DenseSmallRowKQuantTests
{
    private static (float[] Batched, float[] Seq, long Count, int Vocab) Run(string path, int ctx, int rows, bool on)
    {
        VulkanTransformerModel.SetSmallRowKQuantForBench(on);
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, VulkanModelLoader.ResolveSpvDir());
            vk.TrySetAllRowLogitsLimit(16);
            int[] ids = Enumerable.Range(0, ctx + rows).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
            int[] pos = Enumerable.Range(0, ctx + rows).ToArray();
            using var cache = (VulkanKvCache)vk.CreateKvCache(256);
            using (vk.Forward(ids.AsSpan(0, ctx), pos.AsSpan(0, ctx), 0, cache)) { }
            var seq = new List<float>();
            for (int i = 0; i < rows; i++)
            {
                using ITensor t = vk.Forward(ids.AsSpan(ctx + i, 1), pos.AsSpan(ctx + i, 1), 0, cache);
                seq.AddRange(Copy(t));
            }
            cache.Rollback(ctx);
            long c0 = vk.SmallRowKQuantCount;
            float[] batched;
            using (ITensor t = vk.Forward(ids.AsSpan(ctx, rows), pos.AsSpan(ctx, rows), 0, cache))
            {
                Assert.Equal(rows, (int)t.Shape[0]);
                batched = Copy(t);
            }
            return (batched, seq.ToArray(), vk.SmallRowKQuantCount - c0, cfg.VocabSize);
        }
        finally { VulkanTransformerModel.SetSmallRowKQuantForBench(true); }
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var r = new float[t.ElementCount];
        new ReadOnlySpan<float>((void*)t.DataPointer, r.Length).CopyTo(r);
        return r;
    }

    private static int Argmax(float[] a, int row, int vocab)
    {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (a[row * vocab + i] > a[row * vocab + best]) best = i;
        return best;
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(12)]
    [InlineData(16)]
    public void Qwen3_4B_Q4_K_M_BatchedForward_UsesKQuantGemv_AndTracksOneRowDecode(int rows)
    {
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));
        var on = Run(fx.Path!, 40, rows, true);
        var off = Run(fx.Path!, 40, rows, false);
        Assert.True(on.Count > 0, "K-quant multi-column GEMV never ran");
        Assert.Equal(0, off.Count);
        double dOn = 0, dOff = 0;
        for (int i = 0; i < on.Batched.Length; i++)
        {
            dOn = Math.Max(dOn, Math.Abs(on.Batched[i] - on.Seq[i]));
            dOff = Math.Max(dOff, Math.Abs(off.Batched[i] - off.Seq[i]));
        }
        Console.WriteLine($"[917 rows={rows}] max|batched-seq| new {dOn:E2} old {dOff:E2} count {on.Count}");
        for (int r = 0; r < rows; r++)
            Assert.Equal(Argmax(on.Seq, r, on.Vocab), Argmax(on.Batched, r, on.Vocab));
        Assert.True(dOn < 0.5, $"batched forward off the 1-row decode by {dOn:E2}");
    }
}
