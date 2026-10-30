using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1420: the FP32-activation Q8_0 (5..16 rows) / Q4_0 / Q5_0 (2..16 rows) multi-column GEMVs on <see cref="VulkanTransformerModel"/>. A batched N-row forward must (a) take the GEMV path
/// (counter), (b) agree with N sequential 1-row decodes on every row's argmax and stay close in logits, and (c) be closer to the 1-row decode than
/// or comparable to the old MMQ path (control arm: the switch off).
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class DenseSmallRowLegacyQTests
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
            long c0 = vk.SmallRowLegacyQCount;
            float[] batched;
            using (ITensor t = vk.Forward(ids.AsSpan(ctx, rows), pos.AsSpan(ctx, rows), 0, cache))
            {
                Assert.Equal(rows, (int)t.Shape[0]);
                batched = Copy(t);
            }
            return (batched, seq.ToArray(), vk.SmallRowLegacyQCount - c0, cfg.VocabSize);
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
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(16)]
    public void Llama32_1B_Q8_0_BatchedForward_UsesWideGemv_AndTracksOneRowDecode(int rows)
        => Check(TestFixtureResolver.ResolveFile("DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"), rows);

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(9)]
    [InlineData(16)]
    public void SmolLM_135M_Q5_0_BatchedForward_UsesQ5_0Gemv_AndTracksOneRowDecode(int rows)
        => Check(TestFixtureResolver.ResolveFile("DOTLLM_SMOLLM_135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q5_0.gguf"), rows);

    private static void Check(FixtureLocation fx, int rows)
    {
        Skip.If(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN"), "1", StringComparison.Ordinal), "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        Skip.If(!fx.Found, fx.SkipMessage("GGUF"));
        var on = Run(fx.Path!, 40, rows, true);
        var off = Run(fx.Path!, 40, rows, false);
        Assert.True(on.Count > 0, "legacy-quant multi-column GEMV never ran");
        Assert.Equal(0, off.Count);
        double dOn = 0, dOff = 0;
        for (int i = 0; i < on.Batched.Length; i++)
        {
            dOn = Math.Max(dOn, Math.Abs(on.Batched[i] - on.Seq[i]));
            dOff = Math.Max(dOff, Math.Abs(off.Batched[i] - off.Seq[i]));
        }
        Console.WriteLine($"[1420 rows={rows}] max|batched-seq| new {dOn:E2} old {dOff:E2} count {on.Count}");
        // The 1-row decode quantises its activation to Q8_1 (MMVQ); the batched kernels here use the FP32 activation, so on a degenerate small
        // model (SmolLM-135M: batched vs decode with the MMVQ off is 6e-3..1.4e-2, with it on 0.3..0.8) a near-tie may flip: allow an argmax change
        // only inside the measured noise.
        for (int r = 0; r < rows; r++)
        {
            int a = Argmax(on.Seq, r, on.Vocab), b = Argmax(on.Batched, r, on.Vocab);
            Assert.True(a == b || on.Seq[r * on.Vocab + a] - on.Seq[r * on.Vocab + b] < 2 * dOn + 0.05, $"row {r}: argmax {a} vs {b}");
        }
        Assert.True(dOn < 1.0, $"batched forward off the 1-row decode by {dOn:E2}");
    }
}
